from __future__ import annotations

import asyncio
from collections import Counter, deque
from typing import Any, Awaitable, Callable

from starlette.concurrency import run_in_threadpool

from .inputs.asr import LocalAsr, is_stock_phrase
from .i18n import t
from .inputs.vad import RATE, Utterance
from .models import TERMINAL, Session, utc_now

_NO_SPACE_LANGUAGES = {"ja", "zh", "yue", "th"}
MAX_QUEUED_AUDIO_S = 120.0
# Music windows carry no VAD evidence of a voice, so only segments the engine was reasonably sure
# of count as words there (Whisper's usual log-probability bar; engines without scores are kept).
MUSIC_MIN_LOGPROB = -1.0


def confident(results: list[dict[str, Any]], music: bool) -> list[dict[str, Any]]:
    if not music:
        return results
    return [item for item in results if item.get("avg_logprob") is None or item["avg_logprob"] >= MUSIC_MIN_LOGPROB]


class LanguagePolicy:
    def __init__(self, fixed: str | None) -> None:
        self.fixed = fixed if fixed and fixed != "auto" else None
        self._votes: Counter[str] = Counter()

    @property
    def dominant(self) -> str | None:
        if self.fixed:
            return self.fixed
        total = sum(self._votes.values())
        if total < 3:
            return None
        language, count = self._votes.most_common(1)[0]
        return language if count / total >= 0.6 else None

    def for_final(self, duration_s: float) -> str | None:
        # Language ID on very short clips is unreliable; lean on what the stream has shown so far.
        return self.fixed or (self.dominant if duration_s < 2.0 else None)

    def for_partial(self) -> str | None:
        return self.dominant

    def retry_language(self, detected: str | None, probability: float | None) -> str | None:
        dominant = self.dominant
        if self.fixed or not dominant or not detected or detected == dominant:
            return None
        return dominant if probability is not None and probability < 0.5 else None

    def observe(self, language: str | None, probability: float | None, duration_s: float) -> None:
        if language and duration_s >= 2.0 and (probability is None or probability >= 0.7):
            self._votes[language] += 1


def join_text(results: list[dict[str, Any]]) -> tuple[str, str | None, float | None]:
    if not results:
        return "", None, None
    language = results[0].get("language")
    separator = "" if language in _NO_SPACE_LANGUAGES else " "
    text = separator.join(str(item["text"]).strip() for item in results if str(item.get("text", "")).strip())
    return text.strip(), language, results[0].get("language_probability")


class LiveTranscriber:
    def __init__(
        self,
        *,
        session: Session,
        ingest: Callable[[list[dict[str, Any]]], tuple[list[str], list[dict[str, str]]]],
        record_gap: Callable[[str, int | None, int | None], None],
        engine: LocalAsr,
        partial_engine: LocalAsr | None,
        base_capture_ms: int,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        on_evidence: Callable[[], None],
        language: str | None = None,
        prompt: str | None = None,
        speaker: Callable[[Any, float], str | None] | None = None,
    ) -> None:
        self.session = session
        self.ingest = ingest
        self.record_gap = record_gap
        self.engine = engine
        self.partial_engine = partial_engine
        self.base_capture_ms = base_capture_ms
        self.send = send
        self.on_evidence = on_evidence
        self.policy = LanguagePolicy(language)
        # Whisper tends to echo an initial prompt on silence, so only engines built for context get one.
        self.prompt = prompt if engine.context_prompt else None
        self.speaker = speaker
        self._finals: deque[Utterance] = deque()
        self._partial: Utterance | None = None
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False

    def capture_ms(self, sample: int) -> int:
        return self.base_capture_ms + int(sample * 1000 / RATE)

    def submit(self, utterances: list[Utterance]) -> None:
        for utterance in utterances:
            if utterance.kind == "final":
                self._finals.append(utterance)
                if self._partial is not None and self._partial.id == utterance.id:
                    self._partial = None
            elif self.partial_engine is not None:
                # Only the newest snapshot matters; older partials of any utterance are stale.
                self._partial = utterance
        self._shed_backlog()
        if self._finals or self._partial is not None:
            self._idle.clear()
            self._wake.set()
        self._update_queue_diagnostics()

    def _shed_backlog(self) -> None:
        queued = sum(item.duration_s for item in self._finals)
        while self._finals and queued > MAX_QUEUED_AUDIO_S:
            dropped = self._finals.popleft()
            queued -= dropped.duration_s
            self.record_gap(
                t("ASR が追い付かず、認識待ちの発話を破棄しました。", "The ASR fell behind, so waiting utterances were dropped."),
                self.capture_ms(dropped.start_sample), self.capture_ms(dropped.end_sample),
            )
            self._count("asr_dropped_utterances")

    async def run(self) -> None:
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            while not self._closed:
                if self._finals:
                    await self._final(self._finals.popleft())
                elif self._partial is not None:
                    utterance, self._partial = self._partial, None
                    await self._draft(utterance)
                else:
                    break
                self._update_queue_diagnostics()
            if not self._finals and self._partial is None:
                self._idle.set()

    async def drain(self, timeout: float = 120.0) -> bool:
        self._partial = None
        if self._finals:
            self._idle.clear()
            self._wake.set()
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    async def _final(self, utterance: Utterance) -> None:
        language = self.policy.for_final(utterance.duration_s)
        self._set("asr_state", "busy")
        self._count("asr_utterances")
        if utterance.music:
            self._count("asr_music_windows")
        started = asyncio.get_running_loop().time()
        try:
            results = await run_in_threadpool(self.engine.transcribe, utterance.audio, language, self.prompt)
            text, detected, probability = join_text(results)
            retry = self.policy.retry_language(detected, probability)
            if retry:
                results = await run_in_threadpool(self.engine.transcribe, utterance.audio, retry, self.prompt)
                text, detected, probability = join_text(results)
            kept = confident(results, utterance.music)
            if len(kept) < len(results):
                self._count("asr_music_unsure", len(results) - len(kept))
                text = join_text(kept)[0]
        except Exception as exc:
            self._count("asr_errors")
            self._set("asr_last_error", type(exc).__name__)
            self._set("asr_last_result", "error")
            self.record_gap(t(f"ASR に失敗しました: {type(exc).__name__}", f"The ASR failed: {type(exc).__name__}"),
                            self.capture_ms(utterance.start_sample), self.capture_ms(utterance.end_sample))
            await self._safe_send({"type": "asr_error", "reason": str(exc)[:300]})
            self._clear_live(utterance.id)
            self._set("asr_state", "ready")
            return
        self._count("asr_successes")
        self._set("asr_last_duration_ms", round((asyncio.get_running_loop().time() - started) * 1000))
        self._set("asr_last_at_utc", utc_now())
        self._set("asr_last_error", None)
        self._set("asr_state", "ready")
        if text and utterance.speech_score < 0.6 and is_stock_phrase(text):
            text = ""
            self._count("asr_suppressed")
        self.policy.observe(detected, probability, utterance.duration_s)
        self._clear_live(utterance.id)
        if not text:
            self._set("asr_last_result", "empty")
            await self._safe_send({"type": "caption_clear", "utterance_id": utterance.id})
            return
        self._set("asr_last_result", "recognized")
        if self.session.status in TERMINAL:
            # Cancelled while this line was being transcribed: it is dropped quietly, as the cancel asked.
            self._count("asr_after_cancel")
            return
        label = None
        if self.speaker is not None:
            try:
                label = await run_in_threadpool(self.speaker, utterance.audio, utterance.duration_s)
            except Exception as exc:
                # Speaker labels are optional; the caption is kept without one.
                self._count("speaker_errors")
                self._set("speaker_last_error", type(exc).__name__)
        segment = {
            "text": text,
            "capture_start_ms": self.capture_ms(utterance.start_sample),
            "capture_end_ms": self.capture_ms(utterance.end_sample),
            "source_url": self.session.source_url,
            "language": detected,
            "speaker": label,
        }
        try:
            accepted, rejected = self.ingest([segment])
        except Exception as exc:
            self._count("ingest_errors")
            self._set("ingest_last_error", type(exc).__name__)
            self.session.emit("ingest_error", {"message": t("認識結果の採用に失敗しました。", "Could not keep a recognized line."), "error_type": type(exc).__name__})
            return
        self._count("asr_accepted", len(accepted))
        self._count("asr_rejected", len(rejected))
        await self._safe_send({
            "type": "caption_final", "utterance_id": utterance.id, "segment_id": accepted[0] if accepted else None,
            "text": text, "language": detected, "speaker": label,
            **{key: segment[key] for key in ("capture_start_ms", "capture_end_ms")},
        })
        if accepted:
            self.on_evidence()

    async def _draft(self, utterance: Utterance) -> None:
        assert self.partial_engine is not None
        try:
            results = await run_in_threadpool(self.partial_engine.transcribe, utterance.audio, self.policy.for_partial(), None)
        except Exception:
            self._count("asr_partial_errors")
            return
        text, detected, _probability = join_text(confident(results, utterance.music))
        if not text or (utterance.speech_score < 0.6 and is_stock_phrase(text)):
            return
        # A final for this utterance may have been queued while the draft was running.
        if any(item.id == utterance.id for item in self._finals):
            return
        self._count("asr_partials")
        caption = {
            "utterance_id": utterance.id, "text": text, "language": detected,
            "capture_start_ms": self.capture_ms(utterance.start_sample), "updated_at_utc": utc_now(),
        }
        with self.session.lock:
            self.session.live_caption = caption
        await self._safe_send({"type": "caption_partial", **caption})

    def _clear_live(self, utterance_id: str) -> None:
        with self.session.lock:
            if self.session.live_caption and self.session.live_caption.get("utterance_id") == utterance_id:
                self.session.live_caption = None

    async def _safe_send(self, message: dict[str, Any]) -> None:
        try:
            await self.send(message)
        except Exception:
            # The socket may be gone during a drain after disconnect; captions stay in session events.
            pass

    def _count(self, key: str, amount: int = 1) -> None:
        with self.session.lock:
            self.session.audio_diagnostics[key] = self.session.audio_diagnostics.get(key, 0) + amount

    def _set(self, key: str, value: Any) -> None:
        with self.session.lock:
            self.session.audio_diagnostics[key] = value

    def _update_queue_diagnostics(self) -> None:
        with self.session.lock:
            self.session.audio_diagnostics["asr_queue_utterances"] = len(self._finals)
            self.session.audio_diagnostics["asr_queue_ms"] = round(sum(item.duration_s for item in self._finals) * 1000)
