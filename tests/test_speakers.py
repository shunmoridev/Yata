from __future__ import annotations

import asyncio
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from yata.inputs.speakers import DEFAULT_MODEL, SpeakerEmbedder, SpeakerTracker
from yata.models import SessionStore
from yata.state import SessionManager
from yata.topics import TopicPipeline

from test_live_asr import FakeEngine, TranscriberHarness, utterances


def voice(seed, noise=0.0, rng=None):
    base = np.random.default_rng(seed).standard_normal(192)
    if rng is not None:
        base = base + noise * rng.standard_normal(192)
    return (base / np.linalg.norm(base)).astype(np.float32)


class SpeakerTrackerTests(unittest.TestCase):
    def test_utterances_cluster_into_stable_labels(self):
        rng = np.random.default_rng(1)
        tracker = SpeakerTracker()
        sequence = [1, 1, 2, 1, 2, 2, 3, 1]
        labels = [tracker.assign(voice(speaker, 0.5, rng), 3.0) for speaker in sequence]
        self.assertEqual(labels, ["S1", "S1", "S2", "S1", "S2", "S2", "S3", "S1"])

    def test_short_utterances_get_no_label_and_the_speaker_count_is_capped(self):
        tracker = SpeakerTracker(max_speakers=2)
        self.assertIsNone(tracker.assign(voice(1), 0.5))
        self.assertEqual([tracker.assign(voice(seed), 2.0) for seed in (1, 2, 3)][:2], ["S1", "S2"])
        self.assertIn(tracker.assign(voice(4), 2.0), {"S1", "S2"})

    def test_readiness_reports_missing_model_without_loading(self):
        self.assertFalse(SpeakerEmbedder("").readiness()["enabled"])
        with tempfile.TemporaryDirectory() as temp:
            missing = SpeakerEmbedder(Path(temp) / DEFAULT_MODEL)
            self.assertFalse(missing.readiness()["enabled"])
            self.assertIsNone(missing._extractor)


class SpeakerFlowTests(unittest.TestCase):
    def test_final_captions_and_evidence_carry_the_label(self):
        harness = TranscriberHarness(FakeEngine())
        harness.transcriber.speaker = lambda _audio, _duration: "S2"

        async def scenario():
            worker = asyncio.create_task(harness.transcriber.run())
            harness.transcriber.submit(utterances())
            await harness.transcriber.drain(5)
            harness.transcriber.close()
            await worker

        asyncio.run(scenario())
        self.assertEqual(harness.ingested[0]["speaker"], "S2")
        self.assertEqual(harness.sent[0]["speaker"], "S2")

    def test_a_failing_labeler_keeps_the_caption(self):
        harness = TranscriberHarness(FakeEngine())

        def broken(_audio, _duration):
            raise RuntimeError("embedding failed")

        harness.transcriber.speaker = broken

        async def scenario():
            worker = asyncio.create_task(harness.transcriber.run())
            harness.transcriber.submit(utterances())
            await harness.transcriber.drain(5)
            harness.transcriber.close()
            await worker

        asyncio.run(scenario())
        self.assertIsNone(harness.ingested[0]["speaker"])
        self.assertEqual(harness.session.audio_diagnostics["speaker_errors"], 1)

    def test_only_well_formed_labels_are_stored_and_they_reach_summary_prompts(self):
        seen = []

        class Recorder:
            def chat_json(self, model, *, task, system, user, schema, timeout=120.0):
                seen.append(user)
                return {"title": "", "overview": "", "points": [], "key_facts": []}

        manager = SessionManager(TopicPipeline(Recorder()), SessionStore(), background=False)
        session = manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b")
        manager.set_status(session, "capturing_audio")
        manager.ingest(session, [
            {"text": f"発言{i}", "capture_start_ms": i * 6000, "capture_end_ms": i * 6000 + 5000,
             "speaker": "S1" if i % 2 else "<script>"} for i in range(4)
        ], trusted_source_type="asr")
        self.assertEqual([item.speaker for item in session.evidence], [None, "S1", None, "S1"])
        manager.analyze(session)
        lines = next(user["lines"] for user in seen if "lines" in user)
        self.assertEqual([line.get("speaker") for line in lines], [None, "S1", None, "S1"])


MODEL = Path(r"F:\model\Yata\speaker") / DEFAULT_MODEL
BENCH = Path(__file__).resolve().parent.parent / "bench" / "data"


@unittest.skipUnless(MODEL.is_file() and (BENCH / "ja_talk.wav").is_file(), "local speaker model and bench audio")
class RealModelTests(unittest.TestCase):
    def test_two_japanese_speakers_are_told_apart(self):
        def clip(name, start):
            with wave.open(str(BENCH / name), "rb") as handle:
                audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
            return audio[start * 16000:(start + 4) * 16000]

        embedder = SpeakerEmbedder(MODEL)
        tracker = SpeakerTracker()
        labels = [tracker.assign(embedder.embed(clip(name, start)), 4.0)
                  for name, start in [("ja_talk.wav", 45), ("ja_tech.wav", 40), ("ja_talk.wav", 70), ("ja_tech.wav", 100)]]
        self.assertEqual(labels, ["S1", "S2", "S1", "S2"])


if __name__ == "__main__":
    unittest.main()
