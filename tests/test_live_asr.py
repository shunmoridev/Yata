from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from yata.inputs.asr import FasterWhisperEngine, Qwen3AsrEngine, clean_results, create_asr_engines, is_stock_phrase
from yata.inputs.audio import StreamResampler
from yata.inputs.vad import FRAME, RATE, SegmenterConfig, SileroVad, UtteranceSegmenter
from yata.live import LanguagePolicy, LiveTranscriber
from yata.models import Session


class EnergyScorer:
    def reset(self):
        pass

    def __call__(self, frames):
        blocks = frames.reshape(-1, FRAME)
        return np.where(np.sqrt((blocks ** 2).mean(axis=1)) > 0.01, 0.9, 0.05).astype(np.float32)


class SingingScorer:
    """Silero on singing over music: loud, yet below the speech threshold."""

    def reset(self):
        pass

    def __call__(self, frames):
        return np.full(len(frames) // FRAME, 0.2, dtype=np.float32)


def tone(seconds, amplitude=0.2):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def feed(segmenter, audio, chunk=1_365):
    events = []
    for start in range(0, len(audio), chunk):
        events.extend(segmenter.push(audio[start:start + chunk]))
    return events


class SegmenterTests(unittest.TestCase):
    def test_pause_ends_an_utterance_with_padding_and_periodic_partials(self):
        segmenter = UtteranceSegmenter(EnergyScorer())
        events = feed(segmenter, np.concatenate([silence(1.0), tone(2.5), silence(1.0)]))
        finals = [item for item in events if item.kind == "final"]
        partials = [item for item in events if item.kind == "partial"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(len(partials), 2)  # at ~1.0 s and ~2.0 s of the growing utterance
        self.assertTrue(all(item.id == finals[0].id for item in partials))
        # 200 ms of pre-roll before speech onset and of trailing pad after it.
        self.assertAlmostEqual(finals[0].start_sample / RATE, 0.8, delta=0.05)
        self.assertAlmostEqual(finals[0].end_sample / RATE, 3.7, delta=0.05)
        self.assertEqual(len(finals[0].audio), finals[0].end_sample - finals[0].start_sample)
        self.assertGreater(finals[0].speech_score, 0.6)

    def test_long_speech_is_split_at_the_cap_into_contiguous_utterances(self):
        segmenter = UtteranceSegmenter(EnergyScorer())
        events = feed(segmenter, tone(33.0)) + segmenter.flush()
        finals = [item for item in events if item.kind == "final"]
        self.assertEqual(len(finals), 3)
        self.assertTrue(all(item.duration_s <= 15.0 + 1e-6 for item in finals))
        self.assertEqual(len({item.id for item in finals}), 3)
        for previous, following in zip(finals, finals[1:]):
            self.assertEqual(previous.end_sample, following.start_sample)

    def test_short_click_is_not_an_utterance(self):
        segmenter = UtteranceSegmenter(EnergyScorer())
        events = feed(segmenter, np.concatenate([silence(0.5), tone(0.1), silence(1.0)]))
        self.assertEqual([item for item in events if item.kind == "final"], [])

    def test_flush_closes_open_speech_and_reset_moves_the_timeline(self):
        segmenter = UtteranceSegmenter(EnergyScorer())
        segmenter.reset(10 * RATE)
        events = feed(segmenter, tone(1.2)) + segmenter.flush()
        finals = [item for item in events if item.kind == "final"]
        self.assertEqual(len(finals), 1)
        self.assertGreaterEqual(finals[0].start_sample, 10 * RATE)
        self.assertEqual(segmenter.position, 10 * RATE + len(tone(1.2)))

    def test_loud_audio_without_vad_speech_is_cut_into_music_windows(self):
        segmenter = UtteranceSegmenter(SingingScorer())
        events = feed(segmenter, np.concatenate([silence(1.0), tone(20.0), silence(1.0)]))
        finals = [item for item in events if item.kind == "final"]
        self.assertTrue(finals and all(item.music for item in finals))
        self.assertAlmostEqual(finals[0].start_sample / RATE, 1.0, delta=0.05)
        self.assertTrue(all(item.duration_s <= 8.0 + 1e-6 for item in finals))
        for previous, following in zip(finals, finals[1:]):
            self.assertEqual(previous.end_sample, following.start_sample)
        # The last window ends at the quiet with the usual trailing pad.
        self.assertAlmostEqual(finals[-1].end_sample / RATE, 21.2, delta=0.05)
        self.assertLess(max(item.speech_score for item in finals), 0.6)
        self.assertTrue(any(item.kind == "partial" and item.music for item in events))

    def test_short_or_quiet_non_speech_opens_no_music_window(self):
        segmenter = UtteranceSegmenter(SingingScorer())
        burst = np.concatenate([tone(2.5), silence(1.0), tone(2.5), silence(1.0)])  # applause-like bursts
        hum = tone(10.0, amplitude=0.005)  # room tone below the music level
        self.assertEqual(feed(segmenter, np.concatenate([burst, hum])) + segmenter.flush(), [])

    def test_vad_speech_still_ends_at_pauses_and_is_not_music(self):
        segmenter = UtteranceSegmenter(EnergyScorer())
        events = feed(segmenter, np.concatenate([tone(4.0), silence(1.0), tone(4.0), silence(1.0)]))
        finals = [item for item in events if item.kind == "final"]
        self.assertEqual(len(finals), 2)
        self.assertFalse(any(item.music for item in events))

    def test_silero_scores_are_identical_whether_streamed_or_batched(self):
        rng = np.random.default_rng(3)
        audio = (rng.standard_normal(FRAME * 6) * 0.05).astype(np.float32)
        batched = SileroVad()
        streamed = SileroVad()
        whole = batched(audio)
        pieces = np.concatenate([streamed(audio[:FRAME * 2]), streamed(audio[FRAME * 2:])])
        self.assertEqual(whole.shape, (6,))
        np.testing.assert_allclose(whole, pieces, atol=1e-5)
        self.assertTrue(((whole >= 0) & (whole <= 1)).all())


class LanguagePolicyTests(unittest.TestCase):
    def test_dominant_language_guides_short_utterances_and_low_confidence_retries(self):
        policy = LanguagePolicy("auto")
        self.assertIsNone(policy.for_final(1.0))
        for _ in range(3):
            policy.observe("en", 0.95, 4.0)
        policy.observe("ja", 0.3, 4.0)  # low confidence does not vote
        self.assertEqual(policy.dominant, "en")
        self.assertEqual(policy.for_final(1.0), "en")
        self.assertIsNone(policy.for_final(5.0))
        self.assertEqual(policy.retry_language("ja", 0.4), "en")
        self.assertIsNone(policy.retry_language("ja", 0.9))

    def test_fixed_language_is_always_used(self):
        policy = LanguagePolicy("ja")
        self.assertEqual(policy.for_final(10.0), "ja")
        self.assertEqual(policy.for_partial(), "ja")
        self.assertIsNone(policy.retry_language("en", 0.1))


class FakeEngine:
    context_prompt = False

    def __init__(self, text="発言です", delay=0.0, fail=False, results=None):
        self.text = text
        self.delay = delay
        self.fail = fail
        self.results = results
        self.calls = []

    def transcribe(self, audio, language=None, prompt=None):
        self.calls.append((len(audio), language, prompt))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("engine exploded")
        if self.results is not None:
            return [dict(item) for item in self.results]
        return [{"start": 0.0, "end": len(audio) / RATE, "text": self.text, "language": "ja", "language_probability": 0.99}]


def utterances(kind="final", seconds=2.0, score=0.9, music=False):
    segmenter = UtteranceSegmenter(EnergyScorer(), SegmenterConfig())
    events = feed(segmenter, np.concatenate([tone(seconds), silence(1.0)]))
    chosen = [item for item in events if item.kind == kind]
    for item in chosen:
        item.speech_score = score
        item.music = music
    return chosen


class TranscriberHarness:
    def __init__(self, engine, partial_engine=None):
        self.session = Session(
            id="s", source_url="", source_title="", source_kind="share",
            active_input="shared_audio", selected_model="qwen3:8b", started_at_utc="2026-11-04T00:00:00Z",
        )
        self.ingested, self.gaps, self.sent, self.evidence_calls = [], [], [], 0
        self.transcriber = LiveTranscriber(
            session=self.session, ingest=self._ingest, record_gap=lambda *gap: self.gaps.append(gap),
            engine=engine, partial_engine=partial_engine, base_capture_ms=5_000,
            send=self._send, on_evidence=self._evidence, language="auto",
        )

    def _ingest(self, segments):
        self.ingested.extend(segments)
        return [f"seg-{len(self.ingested)}"], []

    async def _send(self, message):
        self.sent.append(message)

    def _evidence(self):
        self.evidence_calls += 1


class LiveTranscriberTests(unittest.TestCase):
    def test_a_line_finished_after_a_cancel_is_dropped_without_an_error(self):
        harness = TranscriberHarness(FakeEngine())
        harness.session.status = "cancelled"

        async def scenario():
            worker = asyncio.create_task(harness.transcriber.run())
            harness.transcriber.submit(utterances())
            self.assertTrue(await harness.transcriber.drain(5))
            harness.transcriber.close()
            await worker

        asyncio.run(scenario())
        self.assertEqual(harness.ingested, [])
        self.assertEqual(harness.sent, [])
        self.assertEqual(harness.session.audio_diagnostics["asr_after_cancel"], 1)
        self.assertNotIn("ingest_error", [event["type"] for event in harness.session.events])

    def test_final_becomes_evidence_with_capture_times_and_caption(self):
        harness = TranscriberHarness(FakeEngine())

        async def scenario():
            worker = asyncio.create_task(harness.transcriber.run())
            harness.transcriber.submit(utterances())
            self.assertTrue(await harness.transcriber.drain(5))
            harness.transcriber.close()
            await worker

        asyncio.run(scenario())
        self.assertEqual(len(harness.ingested), 1)
        segment = harness.ingested[0]
        self.assertEqual(segment["text"], "発言です")
        self.assertEqual(segment["language"], "ja")
        self.assertGreaterEqual(segment["capture_start_ms"], 5_000)
        self.assertEqual([message["type"] for message in harness.sent], ["caption_final"])
        self.assertEqual(harness.sent[0]["segment_id"], "seg-1")
        self.assertEqual(harness.evidence_calls, 1)

    def test_slow_engine_does_not_block_submission(self):
        harness = TranscriberHarness(FakeEngine(delay=0.3))

        async def scenario():
            worker = asyncio.create_task(harness.transcriber.run())
            began = time.perf_counter()
            for _ in range(3):
                harness.transcriber.submit(utterances())
            submit_seconds = time.perf_counter() - began
            await harness.transcriber.drain(5)
            harness.transcriber.close()
            await worker
            return submit_seconds

        self.assertLess(asyncio.run(scenario()), 0.1)
        self.assertEqual(len(harness.ingested), 3)

    def test_only_the_newest_partial_is_drafted_and_never_stored(self):
        partial_engine = FakeEngine(text="途中")
        harness = TranscriberHarness(FakeEngine(), partial_engine)
        partials = utterances("partial", seconds=3.0)
        self.assertGreaterEqual(len(partials), 2)

        async def scenario():
            harness.transcriber.submit(partials)
            worker = asyncio.create_task(harness.transcriber.run())
            await asyncio.sleep(0.2)
            harness.transcriber.close()
            await worker

        asyncio.run(scenario())
        self.assertEqual(len(partial_engine.calls), 1)
        self.assertEqual(harness.ingested, [])
        self.assertEqual(harness.sent[0]["type"], "caption_partial")
        self.assertEqual(harness.session.live_caption["text"], "途中")

    def test_music_windows_keep_only_confident_segments(self):
        engine = FakeEngine(results=[
            {"text": "歌詞です", "language": "ja", "language_probability": 0.9, "avg_logprob": -0.4},
            {"text": "pmm'ama'ama", "language": "ja", "language_probability": 0.9, "avg_logprob": -2.2},
        ])
        harness = TranscriberHarness(engine)

        async def scenario():
            worker = asyncio.create_task(harness.transcriber.run())
            harness.transcriber.submit(utterances(score=0.1, music=True))
            harness.transcriber.submit(utterances(score=0.9))
            await harness.transcriber.drain(5)
            harness.transcriber.close()
            await worker

        asyncio.run(scenario())
        self.assertEqual([item["text"] for item in harness.ingested], ["歌詞です", "歌詞ですpmm'ama'ama"])
        self.assertEqual(harness.session.audio_diagnostics["asr_music_windows"], 1)
        self.assertEqual(harness.session.audio_diagnostics["asr_music_unsure"], 1)

    def test_stock_phrase_from_weak_speech_is_suppressed(self):
        harness = TranscriberHarness(FakeEngine(text="ご視聴ありがとうございました"))

        async def scenario():
            worker = asyncio.create_task(harness.transcriber.run())
            harness.transcriber.submit(utterances(score=0.4))
            await harness.transcriber.drain(5)
            harness.transcriber.close()
            await worker

        asyncio.run(scenario())
        self.assertEqual(harness.ingested, [])
        self.assertEqual(harness.session.audio_diagnostics["asr_suppressed"], 1)

    def test_engine_failure_is_recorded_as_a_gap_and_the_worker_continues(self):
        engine = FakeEngine(fail=True)
        harness = TranscriberHarness(engine)

        async def scenario():
            worker = asyncio.create_task(harness.transcriber.run())
            harness.transcriber.submit(utterances())
            await harness.transcriber.drain(5)
            engine.fail = False
            harness.transcriber.submit(utterances())
            await harness.transcriber.drain(5)
            harness.transcriber.close()
            await worker

        asyncio.run(scenario())
        self.assertEqual(len(harness.gaps), 1)
        self.assertEqual(harness.session.audio_diagnostics["asr_errors"], 1)
        self.assertEqual(len(harness.ingested), 1)


class EngineTests(unittest.TestCase):
    def test_readiness_never_loads_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "model.bin").write_bytes(b"")
            whisper = FasterWhisperEngine(temp)
            self.assertEqual(whisper.readiness()["state"], "invalid")
            self.assertFalse(whisper.loaded)
            qwen = Qwen3AsrEngine(temp)
            self.assertEqual(qwen.readiness()["state"], "missing")
            self.assertIn("model.safetensors", qwen.readiness()["reason"])

    def test_partials_follow_the_device_unless_overridden(self):
        _final, partial = create_asr_engines({"YATA_ASR_MODEL_PATH": "x", "YATA_ASR_DEVICE": "cpu"})
        self.assertIsNone(partial)
        _final, partial = create_asr_engines({"YATA_ASR_MODEL_PATH": "x", "YATA_ASR_DEVICE": "cpu", "YATA_ASR_PARTIALS": "on"})
        self.assertIsNotNone(partial)
        _final, partial = create_asr_engines({"YATA_ASR_MODEL_PATH": "x", "YATA_ASR_DEVICE": "cuda", "YATA_ASR_PARTIALS": "off"})
        self.assertIsNone(partial)

    def test_engine_selection_and_partial_fallback(self):
        final, partial = create_asr_engines({"YATA_ASR_MODEL_PATH": "x", "YATA_ASR_DEVICE": "cuda"})
        self.assertIsInstance(final, FasterWhisperEngine)
        self.assertIs(partial, final)
        final, partial = create_asr_engines({"YATA_ASR_ENGINE": "qwen3-asr", "YATA_ASR_MODEL_PATH": "q", "YATA_ASR_DEVICE": "cuda"})
        self.assertIsInstance(final, Qwen3AsrEngine)
        self.assertIsNone(partial)
        final, partial = create_asr_engines({
            "YATA_ASR_ENGINE": "qwen3-asr", "YATA_ASR_MODEL_PATH": "q", "YATA_ASR_PARTIAL_MODEL_PATH": "w", "YATA_ASR_DEVICE": "cuda",
        })
        self.assertIsInstance(partial, FasterWhisperEngine)

    def test_cleanup_collapses_repetition_and_stock_phrases_are_recognized(self):
        cleaned = clean_results([{"text": "はいはいはいはいはいはいはい"}, {"text": "  "}])
        self.assertEqual(len(cleaned), 1)
        self.assertLess(len(cleaned[0]["text"]), 10)
        self.assertTrue(is_stock_phrase(" Thank you."))
        self.assertTrue(is_stock_phrase("ご視聴ありがとうございました。"))
        self.assertTrue(is_stock_phrase("次回予告"))
        self.assertFalse(is_stock_phrase("Thank you for coming to the keynote today."))


class ResamplerTests(unittest.TestCase):
    def test_streamed_chunks_keep_the_overall_rate(self):
        resampler = StreamResampler(48_000)
        total = 0
        for _ in range(30):
            total += len(resampler.push(np.zeros(4_096, dtype="<f4").tobytes()))
        total += len(resampler.flush())
        self.assertAlmostEqual(total, 30 * 4_096 // 3, delta=64)

    def test_same_rate_passes_through(self):
        data = np.arange(10, dtype="<f4")
        self.assertTrue(np.array_equal(StreamResampler(16_000).push(data.tobytes()), data))


if __name__ == "__main__":
    unittest.main()
