from __future__ import annotations

import struct
import threading
import unittest

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from yata.api import create_app
from yata.inputs.asr import LocalAsr
from yata.models import SessionStore
from yata.state import SessionManager
from yata.topics import TopicPipeline

ORIGIN = "http://127.0.0.1:8765"
TOKEN = "fixture-app-token"


class FakeOllama:
    def models(self):
        return ["qwen3:8b"]

    def chat_json(self, *_args, **_kwargs):
        return {"title": "", "overview": "", "points": [], "key_facts": []}


class FakeAsr(LocalAsr):
    name = "fake"
    fast_partials = True

    def __init__(self, text="字幕から復元した発言"):
        super().__init__("unused")
        self.text = text
        self.calls = 0

    def readiness(self):
        return {"state": "weights_present", "weights_present": True, "reason": "fixture model"}

    def _open(self, _root):
        return object()

    def transcribe(self, audio, language=None, prompt=None):
        self.calls += 1
        return [{"start": 0.0, "end": len(audio) / 16_000, "text": self.text, "language": "ja", "language_probability": 0.98}]


class EnergyScorer:
    def reset(self):
        pass

    def __call__(self, frames):
        blocks = frames.reshape(-1, 512)
        return np.where(np.sqrt((blocks ** 2).mean(axis=1)) > 0.01, 0.9, 0.05).astype(np.float32)


def frames(audio, start_seq=0, start_sample=0, size=4_096):
    packets, seq, sample = [], start_seq, start_sample
    for offset in range(0, len(audio), size):
        chunk = audio[offset:offset + size].astype("<f4")
        packets.append(struct.pack("<IQ", seq, sample) + chunk.tobytes())
        seq += 1
        sample += len(chunk)
    return packets, seq, sample


def speech(seconds, rate=16_000):
    t = np.arange(int(seconds * rate)) / rate
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds, rate=16_000):
    return np.zeros(int(seconds * rate), dtype=np.float32)


def receive_until(socket, wanted):
    seen = []
    while True:
        message = socket.receive_json()
        seen.append(message)
        if message["type"] == wanted:
            return message, seen


class AudioApiTests(unittest.TestCase):
    def setUp(self):
        self.asr = FakeAsr()
        self.manager = SessionManager(TopicPipeline(FakeOllama()), SessionStore())
        self.app = create_app(access_token=TOKEN, ollama=FakeOllama(), asr=self.asr, partial_asr=self.asr,
                              manager=self.manager, scorer_factory=EnergyScorer)
        self.client = TestClient(self.app, base_url=ORIGIN)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)

    def create(self, source_kind="share"):
        response = self.client.post("/api/v1/sessions", json={
            "source_url": "https://example.invalid/live", "source_kind": source_kind,
            "selected_model": "qwen3:8b", "asr_language": "auto", "output_language": "ja",
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["session"]["id"]

    def connect(self, session_id, **overrides):
        headers = {"origin": ORIGIN, "host": "127.0.0.1:8765", "cookie": f"yata_session={TOKEN}"}
        headers.update(overrides)
        return self.client.websocket_connect(f"/api/v1/sessions/{session_id}/audio", headers=headers)

    def hello(self, socket, rate=16_000):
        socket.send_json({"format": "pcm_f32le", "sample_rate": rate, "channels": 1, "captured_at_utc": "2026-11-04T00:00:00Z"})
        self.assertEqual(socket.receive_json()["type"], "asr_loading")
        self.assertEqual(socket.receive_json()["type"], "asr_ready")

    def events(self, session_id):
        return self.client.get(f"/api/v1/sessions/{session_id}/events").json()

    def test_page_cookie_is_httponly_and_foreign_callers_are_rejected(self):
        page = self.client.get("/")
        self.assertNotIn(TOKEN, page.text)
        self.assertIn("httponly", page.headers["set-cookie"].lower())
        self.assertIn("samesite=strict", page.headers["set-cookie"].lower())
        self.assertIn("ws://127.0.0.1:8765", page.headers["content-security-policy"])
        self.assertEqual(self.client.get("/", headers={"origin": "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}).status_code, 403)
        self.assertEqual(self.client.get("/", headers={"host": "127.0.0.1:9999"}).status_code, 403)
        self.assertEqual(self.client.get("/api/v1/health", headers={"origin": "https://evil.example"}).status_code, 403)
        anonymous = TestClient(self.app, base_url=ORIGIN)
        self.assertEqual(anonymous.get("/api/v1/health").status_code, 401)
        health = self.client.get("/api/v1/health").json()
        self.assertEqual(health["asr"]["engine"], "fake")
        self.assertFalse(health["asr"]["loaded"])
        self.assertTrue(health["partial_captions"])

        session_id = self.create()
        for headers in ({"cookie": "yata_session=wrong"}, {"origin": "https://evil.example"}, {"origin": "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}):
            with self.assertRaises(WebSocketDisconnect) as closed:
                with self.connect(session_id, **headers) as socket:
                    socket.receive_json()
            self.assertEqual(closed.exception.code, 4403)
        with self.assertRaises(WebSocketDisconnect) as closed:
            with self.connect("00000000-0000-4000-8000-000000000000") as socket:
                socket.receive_json()
        self.assertEqual(closed.exception.code, 4404)

    def test_caption_file_sessions_and_imports_are_gone(self):
        response = self.client.post("/api/v1/sessions", json={
            "source_url": "", "source_title": "fixture", "source_kind": "manual", "selected_model": "qwen3:8b",
        })
        self.assertEqual(response.status_code, 422)
        session_id = self.create()
        imported = self.client.post(f"/api/v1/sessions/{session_id}/import", json={"text": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n字幕"})
        self.assertEqual(imported.status_code, 404)
        self.assertEqual([e for e in self.events(session_id)["events"] if e["type"] == "evidence_segment"], [])

    def test_speech_becomes_partial_then_final_caption_and_evidence(self):
        session_id = self.create()
        with self.connect(session_id) as socket:
            self.hello(socket, rate=48_000)
            packets, _seq, _sample = frames(np.concatenate([speech(2.5, 48_000), quiet(1.0, 48_000)]))
            for packet in packets:
                socket.send_bytes(packet)
            final, seen = receive_until(socket, "caption_final")
            self.assertIn("caption_partial", [message["type"] for message in seen])
            self.assertEqual(final["text"], "字幕から復元した発言")
            self.assertEqual(final["language"], "ja")
            socket.send_json({"type": "finish_audio"})
            receive_until(socket, "finish_audio_ready")
        state = self.events(session_id)
        evidence = [event["data"] for event in state["events"] if event["type"] == "evidence_segment"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["source_type"], "asr")
        self.assertEqual(evidence[0]["language"], "ja")
        self.assertEqual(evidence[0]["id"], final["segment_id"])
        self.assertIsNone(state["session"]["live_caption"])
        diagnostics = state["session"]["audio_diagnostics"]
        self.assertEqual(diagnostics["asr_engine"], "fake")
        self.assertEqual(diagnostics["asr_accepted"], 1)
        self.assertEqual(diagnostics["server_frames"], len(packets))

        stopped = self.client.post(f"/api/v1/sessions/{session_id}/stop")
        self.assertEqual(stopped.json()["session"]["status"], "ended")

    def test_ui_stop_asks_the_client_to_flush_the_unfinished_utterance(self):
        session_id = self.create()
        result = {}
        with self.connect(session_id) as socket:
            self.hello(socket)
            packets, _seq, _sample = frames(speech(1.2))  # still speaking when the user stops
            for packet in packets:
                socket.send_bytes(packet)
            stopper = threading.Thread(target=lambda: result.update(self.client.post(f"/api/v1/sessions/{session_id}/stop").json()))
            stopper.start()
            receive_until(socket, "flush_requested")
            socket.send_json({"type": "finish_audio"})
            final, _seen = receive_until(socket, "caption_final")
            receive_until(socket, "finish_audio_ready")
            stopper.join(5)
            self.assertFalse(stopper.is_alive())
        self.assertEqual(result["session"]["status"], "ended")
        self.assertEqual(final["text"], "字幕から復元した発言")

    def test_queue_loss_and_frame_holes_are_recorded_once_without_rewinding(self):
        session_id = self.create()
        with self.connect(session_id) as socket:
            self.hello(socket)
            first, seq, sample = frames(quiet(0.5))
            for packet in first:
                socket.send_bytes(packet)
            marker = {"type": "audio_gap", "start_sample": sample, "end_sample": sample + 4_096,
                      "next_sample": sample + 4_096, "next_seq": seq + 1}
            socket.send_json(marker)
            resumed, seq, sample = frames(quiet(0.5), seq + 1, sample + 4_096)
            for packet in resumed:
                socket.send_bytes(packet)
            socket.send_json(marker)  # replayed marker
            socket.send_bytes(resumed[0])  # replayed frame
            hole, _seq, _sample = frames(quiet(0.3), seq + 5, sample + 8_000)
            for packet in hole:
                socket.send_bytes(packet)
            socket.send_json({"type": "finish_audio"})
            receive_until(socket, "finish_audio_ready")
        state = self.events(session_id)
        gaps = [event["data"] for event in state["events"] if event["type"] == "gap"]
        self.assertEqual(len(gaps), 2)
        self.assertEqual(sum("bounded send queue" in gap["reason"] for gap in gaps), 1)
        self.assertEqual(sum("欠落" in gap["reason"] for gap in gaps), 1)
        self.assertTrue(all(gap["capture_start_ms"] < gap["capture_end_ms"] for gap in gaps))
        self.assertEqual(state["session"]["audio_diagnostics"]["server_duplicate_frames"], 1)

    def test_disconnect_records_a_gap_and_allows_an_immediate_reconnect(self):
        session_id = self.create()
        with self.connect(session_id) as socket:
            self.hello(socket)
            packets, _seq, _sample = frames(quiet(0.5))
            socket.send_bytes(packets[0])
        state = self.events(session_id)
        self.assertEqual(state["session"]["status"], "reconnecting")
        self.assertTrue(any("切れました" in event["data"]["reason"] for event in state["events"] if event["type"] == "gap"))
        with self.connect(session_id) as socket:
            self.hello(socket)
            socket.send_json({"type": "finish_audio"})
            receive_until(socket, "finish_audio_ready")
        self.assertEqual(self.events(session_id)["session"]["status"], "capturing_audio")

    def test_cancel_closes_the_audio_socket(self):
        session_id = self.create()
        with self.connect(session_id) as socket:
            self.hello(socket)
            cancelled = self.client.post(f"/api/v1/sessions/{session_id}/cancel").json()
            self.assertEqual(cancelled["session"]["status"], "cancelled")
            with self.assertRaises(WebSocketDisconnect):
                while True:
                    socket.receive_json()


if __name__ == "__main__":
    unittest.main()
