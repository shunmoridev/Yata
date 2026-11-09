from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from yata.api import _asr_public, create_app
from yata.archive import SessionArchive
from yata.features import Features
from yata.inputs.asr import FasterWhisperEngine, Qwen3AsrEngine, create_asr_engines, resolve_asr_settings
from yata.models import SessionStore
from yata.state import SessionError, SessionManager
from yata.topics import TopicPipeline


class NoLLM:
    def __init__(self):
        self.calls = []

    def chat_json(self, model, *, task, **_kwargs):
        self.calls.append(task)
        raise AssertionError(f"captions-only session called the LLM for {task}")

    def models(self):
        return []


class AsrProfileTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-models-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        for name in ("faster-whisper-large-v3-turbo", "faster-whisper-small"):
            (self.root / name).mkdir()
        self.env = {"YATA_MODEL_ROOT": str(self.root)}

    def test_without_cuda_the_light_profile_runs_small_in_int8_on_cpu(self):
        settings = resolve_asr_settings(self.env, cuda=False)
        self.assertEqual((settings.profile, settings.device, settings.compute_type, settings.partials), ("light", "cpu", "int8", False))
        self.assertEqual(settings.model_path, self.root / "faster-whisper-small")
        final, partial = create_asr_engines(self.env, cuda=False)
        self.assertIsInstance(final, FasterWhisperEngine)
        self.assertEqual((final.device, final.compute_type, final.cpu_threads), ("cpu", "int8", 0))
        self.assertIsNone(partial)

    def test_with_cuda_the_full_profile_runs_turbo_with_partials(self):
        settings = resolve_asr_settings(self.env, cuda=True)
        self.assertEqual((settings.profile, settings.device, settings.compute_type, settings.partials), ("full", "cuda", "default", True))
        self.assertEqual(settings.model_path, self.root / "faster-whisper-large-v3-turbo")
        final, partial = create_asr_engines(self.env, cuda=True)
        self.assertIs(partial, final)

    def test_each_setting_overrides_what_the_profile_implies(self):
        light_on_gpu = resolve_asr_settings({**self.env, "YATA_ASR_PROFILE": "light", "YATA_ASR_CPU_THREADS": "2"}, cuda=True)
        self.assertEqual((light_on_gpu.device, light_on_gpu.compute_type, light_on_gpu.cpu_threads), ("cpu", "int8", 2))
        self.assertIn("2 threads", light_on_gpu.describe())
        explicit = resolve_asr_settings({**self.env, "YATA_ASR_COMPUTE_TYPE": "float32", "YATA_ASR_MODEL_PATH": str(self.root / "x"),
                                         "YATA_ASR_PARTIALS": "on"}, cuda=False)
        self.assertEqual((explicit.compute_type, explicit.model_path.name, explicit.partials), ("float32", "x", True))
        unknown = resolve_asr_settings({**self.env, "YATA_ASR_COMPUTE_TYPE": "int3", "YATA_ASR_PROFILE": "turbo",
                                        "YATA_ASR_CPU_THREADS": "many"}, cuda=False)
        self.assertEqual((unknown.profile, unknown.compute_type, unknown.cpu_threads), ("light", "int8", 0))
        qwen, partial = create_asr_engines({"YATA_ASR_ENGINE": "qwen3-asr", "YATA_ASR_MODEL_PATH": "q"}, cuda=True)
        self.assertIsInstance(qwen, Qwen3AsrEngine)
        self.assertIsNone(partial)

    def test_health_reports_the_setup_by_folder_name_only(self):
        final, _ = create_asr_engines(self.env, cuda=False)
        public = _asr_public(final)
        self.assertEqual((public["profile"], public["model"], public["device"], public["compute_type"]),
                         ("light", "faster-whisper-small", "cpu", "int8"))
        self.assertNotIn(str(self.root), str(public))


class CaptionsOnlyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-captions-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.llm = NoLLM()

    def manager(self, background=False):
        manager = SessionManager(TopicPipeline(self.llm), SessionStore(), background=background, archive=SessionArchive(self.root))
        self.addCleanup(manager.close)
        return manager

    def create(self, manager, **settings):
        return manager.create(source_url="", source_title="", source_kind="share", selected_model="", summaries=False, **settings)

    def test_a_captions_only_session_needs_and_uses_no_llm(self):
        manager = self.manager()
        session = self.create(manager)
        manager.set_status(session, "capturing_audio")
        accepted, _ = manager.ingest(session, [{"text": "こんにちは。", "capture_start_ms": 0, "capture_end_ms": 2000}], trusted_source_type="asr")
        self.assertEqual(len(accepted), 1)
        manager.analyze(session)
        with self.assertRaises(SessionError):
            manager.summarize(session)
        manager.stop(session)
        self.assertEqual(session.status, "ended")
        self.assertEqual((session.topics, session.digest, session.source_title), ([], None, ""))
        self.assertEqual(self.llm.calls, [])
        self.assertFalse(session.public_dict()["summaries"])
        reopened = self.manager().get(session.id)
        self.assertFalse(reopened.summaries)
        self.assertFalse(reopened.summary_dict()["summaries"])
        manager.resume(session)
        self.assertEqual(session.status, "created")

    def test_settings_that_need_the_llm_are_refused(self):
        manager = self.manager()
        for settings in ({"caption_language": "en"}, {"slack_delivery": True}):
            with self.assertRaises(SessionError):
                self.create(manager, **settings)
        with self.assertRaises(SessionError):
            manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b", summaries=False)
        with self.assertRaises(SessionError):
            manager.create(source_url="", source_title="", source_kind="share", selected_model="")

    def test_no_analysis_worker_runs_but_the_session_is_still_saved(self):
        manager = self.manager(background=True)
        session = self.create(manager)
        self.assertNotIn(session.id, manager._workers)
        self.assertIsNotNone(manager._autosave)
        self.assertTrue((self.root / "sessions" / f"{session.id}.json").is_file())

    def test_the_api_starts_captions_only_without_ollama(self):
        class Offline:
            def models(self):
                from yata.ollama import OllamaUnavailable
                raise OllamaUnavailable("Ollama に接続できません。")

        manager = self.manager()
        app = create_app(access_token="t", ollama=Offline(), asr=None, manager=manager, features=Features())
        client = TestClient(app, base_url="http://127.0.0.1:8765")
        client.get("/")
        self.assertFalse(client.get("/api/v1/health").json()["llm"]["connected"])
        created = client.post("/api/v1/sessions", json={"source_kind": "share", "summaries": False})
        self.assertEqual(created.status_code, 200, created.text)
        self.assertFalse(created.json()["session"]["summaries"])
        self.assertEqual(client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": ""}).status_code, 422)
        session_id = created.json()["session"]["id"]
        self.assertEqual(client.post(f"/api/v1/sessions/{session_id}/summarize").status_code, 409)


if __name__ == "__main__":
    unittest.main()
