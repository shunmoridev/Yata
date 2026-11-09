from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from yata.inputs.audio import resample_mono_float32
from yata.models import SessionStore
from yata.ollama import OllamaClient
from yata.slack import SlackSender
from yata.state import SessionManager
from yata.topics import TopicPipeline


class SilentLLM:
    def chat_json(self, *_args, **_kwargs):
        return {}


def make_manager(store=None):
    return SessionManager(TopicPipeline(SilentLLM()), store or SessionStore(), background=False)


def make_session(manager):
    return manager.create(
        source_url="https://example.invalid/live",
        source_title="",
        source_kind="share",
        selected_model="qwen3:8b",
    )


class SessionTests(unittest.TestCase):
    def test_duplicate_ingest_requires_nearby_capture_time(self):
        manager = make_manager()
        session = make_session(manager)
        first, _ = manager.ingest(session, [{"text": "同じ発言", "capture_start_ms": 1000, "capture_end_ms": 2000}])
        repeated, _ = manager.ingest(session, [{"text": "同じ発言", "capture_start_ms": 2500, "capture_end_ms": 3000}])
        later, _ = manager.ingest(session, [{"text": "同じ発言", "capture_start_ms": 4000, "capture_end_ms": 5000}])
        self.assertEqual(len(first), 1)
        self.assertEqual(repeated, [])
        self.assertEqual(len(later), 1)
        self.assertEqual({item.source_type for item in session.evidence}, {"asr"})

    def test_stop_and_cancel_have_separate_terminal_states(self):
        manager = make_manager()
        ended = make_session(manager)
        manager.stop(ended)
        self.assertEqual(ended.status, "ended")
        cancelled = make_session(manager)
        manager.ingest(cancelled, [{"text": "未処理の字幕です。"}])
        manager.cancel(cancelled)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancelled.topic_pending, [])

    def test_shared_audio_session_waits_for_the_audio_connection(self):
        manager = make_manager()
        session = manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b")
        self.assertEqual(session.status, "created")
        self.assertEqual(session.active_input, "shared_audio")
        manager.set_status(session, "capturing_audio")
        self.assertEqual(session.status, "capturing_audio")

    def test_session_languages_are_validated(self):
        manager = make_manager()
        session = manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b",
                                 asr_language="en", output_language="ja", caption_language="ja")
        self.assertEqual((session.asr_language, session.output_language, session.caption_language), ("en", "ja", "ja"))
        with self.assertRaises(ValueError):
            manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b", output_language="日本語")
        for retired in ("tab", "manual"):
            with self.assertRaises(ValueError):
                manager.create(source_url="", source_title="", source_kind=retired, selected_model="qwen3:8b")
        with self.assertRaises(ValueError):
            manager.create(source_url="", source_title="", source_kind="share", selected_model="llm-jp-4.1-33b-decision")

    def test_session_store_evicts_finished_sessions_and_rejects_extra_live_ones(self):
        manager = make_manager(SessionStore(max_sessions=2))
        first = make_session(manager)
        second = make_session(manager)
        manager.set_status(first, "ended")
        third = make_session(manager)
        self.assertIsNone(manager.store.get(first.id))
        self.assertIsNotNone(manager.store.get(second.id))
        self.assertIsNotNone(manager.store.get(third.id))
        with self.assertRaisesRegex(ValueError, "上限"):
            make_session(manager)


class AdapterTests(unittest.TestCase):
    def test_slack_boundary_is_disabled_and_keeps_update_local(self):
        sender = SlackSender()
        result = sender.deliver({"text_ja": "fixture"})
        self.assertFalse(sender.enabled)
        self.assertEqual(result.state, "local_only")

    def test_ollama_model_list_is_allowlisted_in_preference_order_without_inference(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self, _limit):
                return json.dumps({"models": [
                    {"name": "qwen3:8b"}, {"name": "mistral-nemo:12b"}, {"name": "qwen3.6:35b-a3b"},
                    {"name": "llm-jp-4.1-33b-decision:private"}, {"name": "unapproved:latest"},
                ]}).encode()
        with patch("urllib.request.urlopen", return_value=Response()) as fetch:
            names = OllamaClient().models()
        self.assertEqual(names, ["qwen3.6:35b-a3b", "qwen3:8b", "mistral-nemo:12b"])
        self.assertTrue(fetch.call_args.args[0].full_url.endswith("/api/tags"))

    def test_ollama_chat_is_schema_bound_non_streaming_and_non_thinking(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self, _limit):
                return json.dumps({"message": {"content": json.dumps({"items": []})}}).encode()
        schema = {"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"]}
        with patch("urllib.request.urlopen", return_value=Response()) as fetch:
            result = OllamaClient().chat_json("qwen3:8b", task="test", system="s", user={"lines": []}, schema=schema)
        self.assertEqual(result, {"items": []})
        request = fetch.call_args.args[0]
        sent = json.loads(request.data.decode())
        self.assertFalse(sent["stream"])
        self.assertFalse(sent["think"])
        self.assertEqual(sent["format"], schema)
        self.assertTrue(request.full_url.endswith("/api/chat"))
        with self.assertRaises(RuntimeError):
            OllamaClient().chat_json("unapproved:latest", task="test", system="s", user={}, schema=schema)

    def test_audio_is_resampled_in_memory(self):
        samples = np.zeros(48_000, dtype="<f4").tobytes()
        converted = resample_mono_float32(samples, 48_000)
        self.assertEqual(len(converted), 16_000)
        self.assertEqual(float(np.max(np.abs(converted))), 0.0)


if __name__ == "__main__":
    unittest.main()
