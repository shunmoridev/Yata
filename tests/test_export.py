from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

from fastapi.testclient import TestClient

from test_sessions import TOKEN, NamingLLM, say
from yata.api import create_app
from yata.archive import SessionArchive
from yata.export import filename, session_markdown
from yata.models import SessionStore
from yata.state import MAX_NOTES_CHARS, SessionError, SessionManager
from yata.topics import TopicPipeline


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-export-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.manager = SessionManager(TopicPipeline(NamingLLM()), SessionStore(), background=False, archive=SessionArchive(self.root))

    def finished(self):
        session = self.manager.create(source_url="https://example.com/live", source_title="", source_kind="share",
                                      selected_model="qwen3:8b", caption_language="en")
        self.manager.set_status(session, "capturing_audio")
        first = say(self.manager, session, "新しいカメラを紹介します。", 0)
        second = say(self.manager, session, "価格は *特価* で [予約] できます。", 9000)
        self.manager.record_gap(session, "音声フレームが欠落しました。", 5000, 8000)
        with session.lock:
            session.translations[first] = "Introducing a new camera."
            session.translation_cursor = len(session.evidence)
            for segment in session.evidence:
                segment.speaker = "S1"
        self.manager.stop(session)
        with session.lock:
            summary = session.topics[0].summary
            summary["key_facts"] = [{"label": "製品", "value": "新しいカメラ", "evidence_id": first}]
            summary["points"].append({"text": "", "evidence_ids": [second], "needs_review": True})
        self.manager.set_notes(session, "- 質問: 発売日は？\n- [00:00:09] 価格を確認")
        return session, first, second

    def test_the_document_reads_from_the_overview_down_to_the_transcript(self):
        session, _first, _second = self.finished()
        text = session_markdown(session)
        order = [text.index(heading) for heading in ("# 新製品の発表会", "## メモ", "## 全体のまとめ", "## 話題", "## 文字起こし")]
        self.assertEqual(order, sorted(order))
        self.assertIn("- 配信: <https://example.com/live>", text)
        self.assertIn("- 言語: 配信 自動判定・要約 日本語・字幕の翻訳 English", text)
        self.assertIn("- 要約のモデル: qwen3:8b（Ollama）", text)
        self.assertIn("- 質問: 発売日は？\n- [00:00:09] 価格を確認", text, "the memo goes in as the user wrote it")
        self.assertIn("- 新製品の紹介（T01）", text)
        self.assertIn("### T01 話題 新しいカ", text)
        self.assertIn("- 製品: **新しいカメラ**（00:00:00）", text)
        self.assertIn("- 紹介がありました。（00:00:00）", text)
        self.assertIn("確かめられなかったため、この文は省きました", text)
        self.assertIn("- 00:00:00 **S1** 新しいカメラを紹介します。\n  - 訳: Introducing a new camera.", text)
        self.assertIn("- *00:00:05 – 00:00:08 は音声が届いていません*", text)
        self.assertIn(r"価格は \*特価\* で \[予約\] できます。", text)
        self.assertLess(text.index("音声が届いていません"), text.index("価格は"), "a gap sits where it happened")
        self.assertNotIn("\n\n\n", text, "one blank line between blocks")

    def test_a_captions_only_session_without_lines_still_exports(self):
        session = self.manager.create(source_url="", source_title="", source_kind="share", selected_model="", summaries=False)
        text = session_markdown(session)
        self.assertIn("- 要約: なし（字幕だけを記録）", text)
        self.assertIn("（取り込み中に書き出しました）", text)
        self.assertIn("*（文字起こしはまだありません）*", text)
        self.assertNotIn("## 話題", text)
        self.assertNotIn("## メモ", text)

    def test_the_file_name_has_the_start_and_a_safe_title(self):
        session, *_ = self.finished()
        with session.lock:
            session.source_title = 'a/b:c*d?"e<f>|g'
        name = filename(session)
        self.assertRegex(name, r"^\d{4}-\d{2}-\d{2}_\d{4}_a b c d e f g\.md$")

    def test_the_memo_is_checked_kept_and_saved_with_the_session(self):
        session, *_ = self.finished()
        with self.assertRaises(SessionError):
            self.manager.set_notes(session, "x" * (MAX_NOTES_CHARS + 1))
        with self.assertRaises(SessionError):
            self.manager.set_notes(session, None)
        revision = session.next_event_id
        self.manager.set_notes(session, "同じ")
        self.manager.set_notes(session, "同じ")
        self.assertEqual(session.next_event_id, revision + 1, "an unchanged memo does not touch the session")
        self.manager.save(session)
        reloaded = SessionManager(TopicPipeline(NamingLLM()), SessionStore(), background=False, archive=SessionArchive(self.root))
        self.assertEqual(reloaded.get(session.id).snapshot()["session"]["notes"], "同じ")


class ExportApiTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-export-api-"))
        self.addCleanup(shutil.rmtree, self.root, True)

        class Ollama:
            def models(self):
                return ["qwen3:8b"]

        self.manager = SessionManager(TopicPipeline(NamingLLM()), SessionStore(), background=False, archive=SessionArchive(self.root))
        self.app = create_app(access_token=TOKEN, ollama=Ollama(), asr=None, manager=self.manager)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8765")
        self.client.get("/")

    def test_the_memo_is_saved_and_the_export_downloads_with_the_page_cookie(self):
        session_id = self.client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "qwen3:8b"}).json()["session"]["id"]
        session = self.manager.get(session_id)
        self.manager.set_status(session, "capturing_audio")
        say(self.manager, session, "新しいカメラを紹介します。", 0)

        saved = self.client.put(f"/api/v1/sessions/{session_id}/notes", json={"notes": "メモです"})
        self.assertEqual(saved.json(), {"notes": "メモです"})
        self.assertEqual(self.client.get(f"/api/v1/sessions/{session_id}").json()["session"]["notes"], "メモです")
        self.assertEqual(self.client.put(f"/api/v1/sessions/{session_id}/notes", json={"notes": "x" * (MAX_NOTES_CHARS + 1)}).status_code, 422)

        exported = self.client.get(f"/api/v1/sessions/{session_id}/export.md")
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.headers["content-type"], "text/markdown; charset=utf-8")
        disposition = exported.headers["content-disposition"]
        self.assertTrue(disposition.startswith('attachment; filename="yata-session.md"; filename*=UTF-8\'\''))
        self.assertRegex(unquote(disposition.split("''", 1)[1]), r"^\d{4}-\d{2}-\d{2}_\d{4}_タブ音声 \d{2} \d{2} \d{2} \d{2}\.md$")
        self.assertIn("## メモ\n\nメモです", exported.text)
        self.assertIn("新しいカメラを紹介します。", exported.text)

        anonymous = TestClient(self.app, base_url="http://127.0.0.1:8765")
        self.assertEqual(anonymous.get(f"/api/v1/sessions/{session_id}/export.md").status_code, 401)
        self.assertEqual(anonymous.put(f"/api/v1/sessions/{session_id}/notes", json={"notes": "x"}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
