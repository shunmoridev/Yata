from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from yata.api import create_app
from yata.archive import INTERRUPTED, UNKNOWN_DELIVERY, SessionArchive, default_data_dir
from yata.audio_socket import _base_capture_ms
from yata.models import SessionStore, parse_utc
from yata.slack import DeliveryResult
from yata.state import SessionError, SessionManager
from yata.topics import TopicPipeline

TOKEN = "test-token"


class NamingLLM:
    def __init__(self, digest_title="新製品の発表会"):
        self.digest_title = digest_title
        self.digests = 0

    def chat_json(self, model, *, task, system, user, schema, timeout=120.0):
        if task == "全体のまとめ":
            self.digests += 1
            return {"title": self.digest_title, "overview": "新製品が紹介されました。",
                    "highlights": [{"text": "新製品の紹介", "topic_ids": [user["topics"][0]["id"]]}]}
        if task == "話題の要約":
            first = user["lines"][0]
            return {"title": "話題 " + first["text"][:4], "overview": "紹介がありました。",
                    "points": [{"text": "紹介がありました。", "evidence_ids": [first["id"]]}], "key_facts": []}
        raise AssertionError(f"unexpected LLM task {task}")


class Recorder:
    enabled = True

    def __init__(self):
        self.sent = []

    def readiness(self):
        return {"enabled": True, "reason": "fixture"}

    def deliver(self, message):
        self.sent.append(message["text"])
        return DeliveryResult("sent", "ok")


def say(manager, session, text, start_ms):
    accepted, rejected = manager.ingest(session, [{"text": text, "capture_start_ms": start_ms, "capture_end_ms": start_ms + 4000,
                                                   "language": "ja"}], trusted_source_type="asr")
    assert accepted and not rejected, rejected
    return accepted[0]


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-sessions-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def manager(self, llm=None, sender=None, store=None):
        return SessionManager(TopicPipeline(llm or NamingLLM()), store or SessionStore(), background=False,
                              sender=sender, archive=SessionArchive(self.root))

    def create(self, manager, **settings):
        session = manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b", **settings)
        manager.set_status(session, "capturing_audio")
        return session

    def test_a_shared_tab_session_is_named_from_its_content(self):
        manager = self.manager()
        with self.assertRaises(SessionError):
            manager.create(source_url="", source_title="手入力の名前", source_kind="share", selected_model="qwen3:8b")
        session = self.create(manager)
        self.assertEqual(session.title_source, "auto")
        self.assertEqual(session.display_title(), "")
        say(manager, session, "新しいカメラを紹介します。", 0)
        manager.stop(session)
        self.assertEqual(session.source_title, "新製品の発表会")
        self.assertEqual(session.public_dict()["display_title"], "新製品の発表会")

    def test_a_generated_name_with_an_unsupported_number_falls_back_to_the_first_topic(self):
        manager = self.manager(NamingLLM(digest_title="2027年の新製品"))
        session = self.create(manager)
        say(manager, session, "新しいカメラを紹介します。", 0)
        manager.stop(session)
        self.assertEqual(session.source_title, "")
        self.assertEqual(session.display_title(), "話題 新しいカ")

    def test_a_finished_session_is_saved_and_loads_with_everything_the_page_shows(self):
        manager = self.manager()
        session = self.create(manager, caption_language="en")
        first = say(manager, session, "新しいカメラを紹介します。", 0)
        manager.record_gap(session, "音声フレームが欠落しました。", 4000, 4500)
        with session.lock:
            session.translations[first] = "Introducing a new camera."
            session.translation_cursor = len(session.evidence)
        manager.stop(session)
        cursor = session.next_event_id
        files = sorted(path.name for path in (self.root / "sessions").iterdir())
        self.assertEqual(files, [f"{session.id}.json", f"{session.id}.summary.json"])
        saved = (self.root / "sessions" / f"{session.id}.json").read_text(encoding="utf-8")
        self.assertNotIn("audio_diagnostics", saved)

        later = self.manager()
        rows = later.list()
        self.assertEqual([(row["id"], row["display_title"], row["status"], row["segment_count"], row["topic_count"]) for row in rows],
                         [(session.id, "新製品の発表会", "ended", 1, 1)])
        loaded = later.get(session.id)
        snapshot = loaded.snapshot()
        self.assertEqual(snapshot["session"]["status"], "ended")
        self.assertEqual([item["text"] for item in snapshot["evidence"]], ["新しいカメラを紹介します。"])
        self.assertEqual(snapshot["topics"][0]["summary_state"], "ready")
        self.assertEqual(snapshot["translations"], {first: "Introducing a new camera."})
        self.assertEqual(snapshot["digest"]["overview"], "新製品が紹介されました。")
        self.assertEqual([gap["reason"] for gap in snapshot["gaps"]], ["音声フレームが欠落しました。"])
        self.assertEqual(snapshot["cursor"], cursor - 1)
        self.assertGreater(loaded.emit("probe", {})["cursor"], snapshot["gaps"][0]["cursor"])

    def test_unchanged_sessions_are_not_rewritten_and_bad_ids_or_files_are_ignored(self):
        manager = self.manager()
        archive = manager.archive
        session = self.create(manager)
        archive.save(session)
        self.assertFalse(archive.save(session))
        say(manager, session, "こんにちは。", 0)
        self.assertTrue(archive.save(session))
        self.assertFalse(archive.save(session))
        self.assertIsNone(archive.load("../../etc/passwd"))
        with self.assertRaises(SessionError):
            manager.get("../outside")
        broken = "0" * 8 + "-0000-0000-0000-" + "0" * 12
        (self.root / "sessions" / f"{broken}.json").write_text("{not json", encoding="utf-8")
        (self.root / "sessions" / f"{broken}.summary.json").write_text(json.dumps({"id": broken}), encoding="utf-8")
        fresh = SessionArchive(self.root)
        self.assertIsNone(fresh.load(broken))
        self.assertIn(session.id, [row["id"] for row in fresh.summaries()])

    def test_a_session_cut_off_by_a_server_stop_loads_as_interrupted_and_resumable(self):
        manager = self.manager(sender=Recorder())
        session = self.create(manager, slack_delivery=True)
        say(manager, session, "新しいカメラを紹介します。", 0)
        say(manager, session, "価格は後日発表です。", 9000)
        manager.pipeline._assign(session)
        with session.lock:
            session.topics[0].delivery_state = "pending"
        manager.save_all()

        loaded = self.manager().get(session.id)
        self.assertEqual(loaded.status, "cancelled")
        self.assertEqual(loaded.status_reason, str(INTERRUPTED))
        self.assertIsNone(loaded.clock_started_utc)
        self.assertGreaterEqual(loaded.clock_offset_ms, 13000)
        self.assertEqual((loaded.topics[0].delivery_state, loaded.topics[0].delivery_reason), ("failed", str(UNKNOWN_DELIVERY)))

    def test_resume_appends_after_the_last_run_and_names_and_digests_again(self):
        sender = Recorder()
        llm = NamingLLM()
        manager = self.manager(llm, sender=sender)
        session = self.create(manager, slack_delivery=True)
        say(manager, session, "新しいカメラを紹介します。", 0)
        manager.stop(session)
        paused = session.clock_ms()
        self.assertEqual(session.clock_started_utc, None)
        self.assertGreaterEqual(paused, 4000)
        with self.assertRaises(SessionError):
            manager.resume(session, source_url="javascript:alert(1)")

        reopened_manager = self.manager(llm, sender=sender)
        reopened = reopened_manager.get(session.id)
        reopened_manager.resume(reopened, source_url="https://www.youtube.com")
        self.assertEqual(reopened.status, "created")
        self.assertEqual(reopened.source_url, "https://www.youtube.com")
        with self.assertRaises(SessionError):
            reopened_manager.resume(reopened)
        with self.assertRaises(SessionError):
            reopened_manager.delete(reopened)
        started = parse_utc(reopened.clock_started_utc)
        self.assertEqual(_base_capture_ms(reopened, (started + timedelta(seconds=2)).isoformat()), paused + 2000)
        reopened_manager.set_status(reopened, "capturing_audio")
        say(reopened_manager, reopened, "次はスピーカーです。", paused + 2000)
        reopened_manager.stop(reopened)

        self.assertEqual([topic.index for topic in reopened.topics], [1, 2])
        self.assertEqual(reopened.topics[1].start_ms, paused + 2000)
        self.assertEqual(llm.digests, 2)
        self.assertEqual(len([text for text in sender.sent if "全体のまとめ" in text]), 2)
        self.assertEqual(len([text for text in sender.sent if "話題 1:" in text]), 1)
        self.assertEqual(len([text for text in sender.sent if "話題 2:" in text]), 1)

    def test_delete_removes_a_finished_session_from_memory_and_disk(self):
        manager = self.manager()
        session = self.create(manager)
        with self.assertRaises(SessionError):
            manager.delete(session)
        manager.cancel(session)
        manager.delete(session)
        self.assertEqual(manager.list(), [])
        self.assertEqual(list((self.root / "sessions").iterdir()), [])
        session.emit("late", {})
        self.assertFalse(manager.archive.save(session))
        self.assertEqual(list((self.root / "sessions").iterdir()), [])
        with self.assertRaises(SessionError):
            manager.get(session.id)

    def test_a_full_store_saves_a_finished_session_before_dropping_it(self):
        manager = self.manager(store=SessionStore(max_sessions=1))
        first = self.create(manager)
        say(manager, first, "新しいカメラを紹介します。", 0)
        manager.stop(first)
        second = self.create(manager)
        self.assertIsNone(manager.store.get(first.id))
        self.assertEqual(len(manager.list()), 2)
        manager.cancel(second)
        again = manager.get(first.id)
        self.assertIsNot(again, first)
        self.assertEqual([item.text for item in again.evidence], ["新しいカメラを紹介します。"])

    def test_the_default_data_folder_is_local_app_data_unless_configured(self):
        self.assertEqual(default_data_dir({"LOCALAPPDATA": r"C:\Users\u\AppData\Local"}), Path(r"C:\Users\u\AppData\Local") / "Yata")
        self.assertEqual(default_data_dir({"YATA_DATA_DIR": r"D:\yata-data", "LOCALAPPDATA": "x"}), Path(r"D:\yata-data"))


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-api-"))
        self.addCleanup(shutil.rmtree, self.root, True)

        class Ollama:
            def models(self):
                return ["qwen3:8b"]

            def chat_json(self, *args, **kwargs):
                return NamingLLM().chat_json(*args, **kwargs)

        self.manager = SessionManager(TopicPipeline(NamingLLM()), SessionStore(), background=False, archive=SessionArchive(self.root))
        self.app = create_app(access_token=TOKEN, ollama=Ollama(), asr=None, manager=self.manager)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8765")
        self.client.get("/")

    def test_list_open_resume_and_delete(self):
        created = self.client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "qwen3:8b",
                                                             "source_url": "https://www.youtube.com"})
        self.assertEqual(created.status_code, 200, created.text)
        session_id = created.json()["session"]["id"]
        self.assertEqual(self.client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "qwen3:8b",
                                                                     "source_title": "typed"}).status_code, 422)
        session = self.manager.get(session_id)
        self.manager.set_status(session, "capturing_audio")
        say(self.manager, session, "新しいカメラを紹介します。", 0)

        listed = self.client.get("/api/v1/sessions").json()["sessions"]
        self.assertEqual([(row["id"], row["status"], row["segment_count"]) for row in listed], [(session_id, "capturing_audio", 1)])
        self.assertEqual(self.client.post(f"/api/v1/sessions/{session_id}/resume").status_code, 409)
        self.assertEqual(self.client.delete(f"/api/v1/sessions/{session_id}").status_code, 409)

        self.assertEqual(self.client.post(f"/api/v1/sessions/{session_id}/stop").status_code, 200)
        snapshot = self.client.get(f"/api/v1/sessions/{session_id}").json()
        self.assertEqual(snapshot["session"]["display_title"], "新製品の発表会")
        self.assertEqual(len(snapshot["evidence"]), 1)
        self.assertEqual(self.client.get(f"/api/v1/sessions/{session_id}/events?after={snapshot['cursor']}").json()["events"], [])

        resumed = self.client.post(f"/api/v1/sessions/{session_id}/resume", json={"source_url": ""})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.assertEqual(resumed.json()["session"]["status"], "created")
        self.assertEqual(resumed.json()["session"]["source_url"], "https://www.youtube.com")
        self.assertEqual(self.client.post(f"/api/v1/sessions/{session_id}/cancel").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/v1/sessions/{session_id}").json(), {"deleted": session_id})
        self.assertEqual(self.client.get(f"/api/v1/sessions/{session_id}").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/sessions").json()["sessions"], [])

    def test_routes_need_the_page_cookie(self):
        anonymous = TestClient(self.app, base_url="http://127.0.0.1:8765")
        self.assertEqual(anonymous.get("/api/v1/sessions").status_code, 401)
        self.assertEqual(anonymous.delete("/api/v1/sessions/" + "0" * 8 + "-0000-0000-0000-" + "0" * 12).status_code, 401)


if __name__ == "__main__":
    unittest.main()
