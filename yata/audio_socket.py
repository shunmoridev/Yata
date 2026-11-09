from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .inputs.asr import LocalAsr
from .i18n import t
from .inputs.audio import StreamResampler
from .inputs.vad import RATE, FrameScorer, UtteranceSegmenter
from .live import LiveTranscriber
from .models import TERMINAL, Session, parse_utc, utc_now
from .state import SessionError, SessionManager


def _base_capture_ms(session: Session, captured_at: str) -> int:
    return session.clock_ms(parse_utc(captured_at))


async def serve_audio(
    websocket: WebSocket,
    session: Session,
    *,
    manager: SessionManager,
    engine: LocalAsr,
    partial_engine: LocalAsr | None,
    scorer_factory: Callable[[], FrameScorer],
    stop_event: asyncio.Event,
    on_evidence: Callable[[Session], None],
    release: Callable[[], None],
    speaker: Callable[[Any, float], str | None] | None = None,
) -> None:
    hello = await websocket.receive_json()
    if not isinstance(hello, dict) or hello.get("format") != "pcm_f32le" or hello.get("channels") != 1:
        await websocket.send_json({"type": "error", "reason": t("PCM 形式が未対応です。", "Unsupported PCM format.")})
        await websocket.close(code=4400)
        return
    rate = hello.get("sample_rate")
    if not isinstance(rate, int) or isinstance(rate, bool) or not 8_000 <= rate <= 96_000:
        await websocket.close(code=4400, reason="invalid sample rate")
        return
    if session.status not in {"created", "starting_audio", "reconnecting", "capturing_audio"}:
        await websocket.send_json({"type": "error", "reason": t("セッションが音声取得を開始できる状態ではありません。", "The session cannot take audio now.")})
        await websocket.close(code=4409)
        return

    diagnostics = session.audio_diagnostics

    def update(**values: Any) -> None:
        with session.lock:
            diagnostics.update(values)

    def count(key: str, amount: int = 1) -> None:
        with session.lock:
            diagnostics[key] = diagnostics.get(key, 0) + amount

    update(sample_rate=rate, server_socket_state="open", asr_state="loading", asr_engine=engine.name)
    manager.set_status(session, "starting_audio")
    await websocket.send_json({"type": "asr_loading"})
    try:
        await run_in_threadpool(engine._load)
        if partial_engine is not None and partial_engine is not engine:
            await run_in_threadpool(partial_engine._load)
    except Exception as exc:
        count("asr_errors")
        update(asr_state="unavailable", asr_last_result="error", asr_last_error=type(exc).__name__, asr_last_at_utc=utc_now())
        manager.set_status(session, "created", t(f"ASR を読み込めませんでした: {str(exc)[:200]}", f"Could not load the ASR: {str(exc)[:200]}"))
        await websocket.send_json({"type": "asr_unavailable", "reason": str(exc)[:300]})
        await websocket.close(code=4409)
        return
    if session.status in TERMINAL:
        await websocket.send_json({"type": "session_stopped", "status": session.status})
        await websocket.close(code=1000)
        return
    if stop_event.is_set():
        await websocket.send_json({"type": "finish_audio_ready"})
        await websocket.close(code=1000)
        return

    base_capture_ms = _base_capture_ms(session, str(hello.get("captured_at_utc", ""))[:40])

    def capture_ms(source_sample: int) -> int:
        return base_capture_ms + int(source_sample * 1000 / rate)

    def record_gap(reason: str, start_ms: int | None = None, end_ms: int | None = None) -> None:
        manager.record_gap(session, reason, start_ms, end_ms)

    async def send(message: dict[str, Any]) -> None:
        await websocket.send_json(message)

    transcriber = LiveTranscriber(
        session=session,
        ingest=lambda segments: manager.ingest(session, segments, trusted_source_type="asr"),
        record_gap=record_gap,
        engine=engine,
        partial_engine=partial_engine,
        base_capture_ms=base_capture_ms,
        send=send,
        on_evidence=lambda: on_evidence(session),
        language=session.asr_language,
        # Only a title the user typed is context; one Yata generated from the content adds nothing.
        prompt=session.source_title if session.title_source == "user" and session.source_title else None,
        speaker=speaker,
    )
    worker = asyncio.create_task(transcriber.run())
    segmenter = UtteranceSegmenter(scorer_factory())
    resampler = StreamResampler(rate)
    expected_sample: int | None = None
    flush_requested = False

    def close_stream() -> None:
        nonlocal resampler
        transcriber.submit(segmenter.push(resampler.flush()) + segmenter.flush())
        resampler = StreamResampler(rate)

    def restart_at(source_sample: int) -> None:
        close_stream()
        segmenter.reset(round(source_sample * RATE / rate))

    update(asr_state="ready")
    await websocket.send_json({"type": "asr_ready"})
    manager.set_status(session, "capturing_audio")
    try:
        while True:
            if session.status in TERMINAL:
                await websocket.send_json({"type": "session_stopped", "status": session.status})
                await websocket.close(code=1000)
                return
            if stop_event.is_set() and not flush_requested:
                # A local UI stop asks the client to flush its AudioWorklet tail first.
                flush_requested = True
                await websocket.send_json({"type": "flush_requested"})
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            text_message = message.get("text")
            if text_message is not None:
                try:
                    control = json.loads(text_message)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "reason": "Invalid control message"})
                    continue
                kind = control.get("type") if isinstance(control, dict) else None
                if kind == "finish_audio":
                    close_stream()
                    if not await transcriber.drain():
                        record_gap(t("最後の音声の文字起こしが時間内に終わりませんでした。", "The last audio was not transcribed in time."))
                    await websocket.send_json({"type": "finish_audio_ready"})
                    await websocket.close(code=1000)
                    return
                if kind == "audio_gap":
                    start, end = control.get("start_sample"), control.get("end_sample")
                    next_sample, next_seq = control.get("next_sample"), control.get("next_seq")
                    if not (all(isinstance(value, int) and not isinstance(value, bool) for value in (start, end, next_sample, next_seq))
                            and 0 <= start < end == next_sample <= rate * 60 * 60 * 24
                            and 0 <= next_seq <= 0xFFFFFFFF):
                        await websocket.send_json({"type": "error", "reason": "Invalid audio gap marker"})
                        continue
                    if expected_sample is not None and end <= expected_sample:
                        continue  # a replayed marker must not create another gap or rewind the stream
                    restart_at(next_sample)
                    record_gap("Audio exceeded the bounded send queue; some frames were discarded.",
                               capture_ms(max(start, expected_sample or 0)), capture_ms(end))
                    expected_sample = next_sample
                    continue
                await websocket.send_json({"type": "error", "reason": "Unknown control message"})
                continue
            payload = message.get("bytes")
            if payload is None:
                continue
            if len(payload) < 16 or len(payload) > 65_536 or (len(payload) - 12) % 4:
                record_gap(t("音声フレーム形式が不正です。", "Invalid audio frame."))
                continue
            _seq, sample_start = struct.unpack_from("<IQ", payload, 0)
            pcm = payload[12:]
            sample_count = len(pcm) // 4
            count("server_frames")
            count("server_bytes", len(pcm))
            update(server_last_frame_at_utc=utc_now())
            if expected_sample is not None and sample_start < expected_sample:
                overlap = expected_sample - sample_start
                discarded = min(overlap, sample_count)
                count("server_duplicate_frames")
                count("server_duplicate_samples", discarded)
                if overlap >= sample_count:
                    continue  # an already received packet replayed after a reconnect
                pcm = pcm[overlap * 4:]
                sample_start = expected_sample
                sample_count -= overlap
            if expected_sample is None:
                segmenter.reset(round(sample_start * RATE / rate))
            elif sample_start > expected_sample:
                record_gap(t("音声フレームが欠落しました。", "Audio frames were lost."), capture_ms(expected_sample), capture_ms(sample_start))
                restart_at(sample_start)
            expected_sample = sample_start + sample_count
            try:
                samples = resampler.push(pcm)
            except ValueError:
                record_gap(t("音声フレームに不正な値が含まれています。", "An audio frame held invalid values."), capture_ms(sample_start), capture_ms(expected_sample))
                restart_at(expected_sample)
                continue
            transcriber.submit(segmenter.push(samples))
            count("server_accepted_samples", sample_count)
    except WebSocketDisconnect:
        close_stream()
        # Free the connection slot first so the client can reconnect while the tail is transcribed.
        release()
        update(server_socket_state="reconnecting")
        try:
            manager.set_status(session, "reconnecting", t("音声接続が切れました。再接続を待っています。",
                                                         "The audio connection dropped. Waiting for it to reconnect."))
            record_gap(t("音声接続が切れました。", "The audio connection dropped."))
        except SessionError:
            pass
        # Utterances already captured are still transcribed; the client may reconnect meanwhile.
        await transcriber.drain(timeout=60)
    finally:
        transcriber.close()
        try:
            await asyncio.wait_for(worker, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            worker.cancel()
        update(server_socket_state="closed", asr_queue_utterances=0, asr_queue_ms=0)
