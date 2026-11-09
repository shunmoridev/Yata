"""Silero scores singing over music far below the speech threshold, so a stream
that stays loud without VAD speech for a few seconds is cut into short "music"
windows by level instead, and the ASR decides whether they contain words.
"""
from __future__ import annotations

import os
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

RATE = 16_000
FRAME = 512
_CONTEXT = 64


class FrameScorer(Protocol):
    def reset(self) -> None: ...
    def __call__(self, frames: np.ndarray) -> np.ndarray: ...


class SileroVad:
    def __init__(self, model_path: str | None = None) -> None:
        if model_path is None:
            import faster_whisper
            model_path = os.path.join(os.path.dirname(faster_whisper.__file__), "assets", "silero_vad_v6.onnx")
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=options)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        blocks = frames.reshape(-1, FRAME)
        if not len(blocks):
            return np.empty(0, dtype=np.float32)
        contexts = np.concatenate([self._context[None, :], blocks[:-1, -_CONTEXT:]], axis=0)
        batch = np.concatenate([contexts, blocks], axis=1).astype(np.float32, copy=False)
        probs, self._h, self._c = self._session.run(None, {"input": batch, "h": self._h, "c": self._c})
        self._context = blocks[-1, -_CONTEXT:].copy()
        return np.asarray(probs, dtype=np.float32).reshape(-1)


@dataclass
class SegmenterConfig:
    threshold: float = 0.5
    neg_threshold: float = 0.35
    min_speech_ms: int = 250
    min_silence_ms: int = 600
    pad_ms: int = 200
    max_utterance_s: float = 15.0
    partial_interval_s: float = 1.0
    min_partial_s: float = 0.8
    music_rms: float = 0.01
    music_after_s: float = 3.0
    music_max_s: float = 8.0


@dataclass
class Utterance:
    id: str
    kind: str
    start_sample: int
    end_sample: int
    audio: np.ndarray = field(repr=False)
    speech_score: float = 0.0
    music: bool = False

    @property
    def duration_s(self) -> float:
        return (self.end_sample - self.start_sample) / RATE


class UtteranceSegmenter:
    def __init__(self, scorer: FrameScorer, config: SegmenterConfig | None = None) -> None:
        self.scorer = scorer
        self.config = config or SegmenterConfig()
        self._ms_per_frame = FRAME * 1000 / RATE
        self._pad_frames = max(0, round(self.config.pad_ms / self._ms_per_frame))
        self._music_after_frames = max(1, round(self.config.music_after_s * 1000 / self._ms_per_frame))
        self.reset(0)

    def reset(self, position: int) -> None:
        self.scorer.reset()
        self._position = position
        self._pending = np.empty(0, dtype=np.float32)
        self._preroll: deque[tuple[np.ndarray, float, float]] = deque(maxlen=self._pad_frames)
        self._loud: deque[tuple[np.ndarray, float, float]] = deque(maxlen=self._music_after_frames)
        self._frames: list[np.ndarray] = []
        self._probs: list[float] = []
        self._levels: list[float] = []
        self._start = position
        self._in_speech = False
        self._music = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._last_partial_frames = 0
        self._utterance_id = ""

    @property
    def position(self) -> int:
        return self._position + len(self._pending)

    def push(self, samples: np.ndarray) -> list[Utterance]:
        if samples.size:
            self._pending = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        whole = len(self._pending) // FRAME * FRAME
        if not whole:
            return []
        chunk, self._pending = self._pending[:whole], self._pending[whole:]
        probs = self.scorer(chunk)
        events: list[Utterance] = []
        for index, prob in enumerate(probs):
            frame = chunk[index * FRAME:(index + 1) * FRAME]
            events.extend(self._step(frame, float(prob)))
            self._position += FRAME
        return events

    def flush(self) -> list[Utterance]:
        events = []
        if self._pending.size:
            padded = np.concatenate([self._pending, np.zeros(FRAME - len(self._pending), dtype=np.float32)])
            # Scored as a silent continuation so a cut-off word still ends the utterance cleanly.
            prob = float(self.scorer(padded)[0])
            events.extend(self._step(padded, prob))
            self._position += len(self._pending)
            self._pending = np.empty(0, dtype=np.float32)
        if self._in_speech:
            final = self._emit_final(len(self._frames))
            if final:
                events.append(final)
            self._end_speech()
        return events

    def _voiced(self, prob: float, level: float) -> bool:
        return prob >= self.config.threshold or (self._music and level >= self.config.music_rms)

    def _open(self, items: list[tuple[np.ndarray, float, float]], *, music: bool) -> None:
        self._in_speech = True
        self._music = music
        self._utterance_id = str(uuid.uuid4())
        self._frames = [item[0] for item in items]
        self._probs = [item[1] for item in items]
        self._levels = [item[2] for item in items]
        self._start = self._position - len(self._frames) * FRAME
        self._preroll.clear()
        self._loud.clear()
        self._silence_frames = 0
        self._speech_frames = sum(self._voiced(prob, level) for prob, level in zip(self._probs, self._levels))
        self._last_partial_frames = 0

    def _step(self, frame: np.ndarray, prob: float) -> list[Utterance]:
        config = self.config
        level = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        loud = level >= config.music_rms
        if not self._in_speech:
            if prob >= config.threshold:
                self._open(list(self._preroll), music=False)
            else:
                self._preroll.append((frame, prob, level))
                if not loud:
                    self._loud.clear()
                    return []
                self._loud.append((frame, prob, level))
                if len(self._loud) < self._music_after_frames:
                    return []
                self._open(list(self._loud)[:-1], music=True)
        self._frames.append(frame)
        self._probs.append(prob)
        self._levels.append(level)
        if self._voiced(prob, level):
            self._speech_frames += 1
        quiet = prob < config.neg_threshold and not (self._music and loud)
        self._silence_frames = self._silence_frames + 1 if quiet else 0

        events: list[Utterance] = []
        if self._silence_frames * self._ms_per_frame >= config.min_silence_ms:
            keep = len(self._frames) - self._silence_frames + self._pad_frames
            tail = list(zip(self._frames[keep:], self._probs[keep:], self._levels[keep:]))
            final = self._emit_final(keep)
            if final:
                events.append(final)
            self._end_speech()
            self._preroll.extend(tail[-self._pad_frames:] if self._pad_frames else [])
            return events
        cap_s = config.music_max_s if self._music else config.max_utterance_s
        if len(self._frames) * FRAME >= cap_s * RATE:
            # Split where speech (or, in music, the level) is weakest within the last 3 s so words
            # are rarely cut in half.
            weights = self._levels if self._music else self._probs
            window = min(len(weights) - 1, round(3000 / self._ms_per_frame))
            split = len(weights) - window + int(np.argmin(weights[-window:])) + 1
            carry = list(zip(self._frames[split:], self._probs[split:], self._levels[split:]))
            # Once VAD hears speech for most of a music window, the rest is cut at pauses again.
            music = self._music and float(np.mean([p >= config.threshold for p in self._probs[:split]])) < 0.5
            final = self._emit_final(split)
            if final:
                events.append(final)
            next_start = self._start + split * FRAME
            self._end_speech()
            self._open(carry, music=music)
            self._start = next_start
            return events
        frames_per_partial = config.partial_interval_s * RATE / FRAME
        # A draft during the closing pause would only repeat the previous one.
        pausing = self._silence_frames * self._ms_per_frame >= 300
        if (not pausing and len(self._frames) * FRAME >= config.min_partial_s * RATE
                and len(self._frames) - self._last_partial_frames >= frames_per_partial):
            self._last_partial_frames = len(self._frames)
            events.append(self._utterance("partial", len(self._frames)))
        return events

    def _emit_final(self, frame_count: int) -> Utterance | None:
        if self._speech_frames * self._ms_per_frame < self.config.min_speech_ms or frame_count <= 0:
            return None
        return self._utterance("final", frame_count)

    def _utterance(self, kind: str, frame_count: int) -> Utterance:
        audio = np.concatenate(self._frames[:frame_count]) if frame_count else np.empty(0, dtype=np.float32)
        probs = self._probs[:frame_count]
        return Utterance(
            id=self._utterance_id, kind=kind, start_sample=self._start,
            end_sample=self._start + frame_count * FRAME, audio=audio,
            speech_score=float(np.mean(probs)) if probs else 0.0, music=self._music,
        )

    def _end_speech(self) -> None:
        self._in_speech = False
        self._music = False
        self._frames = []
        self._probs = []
        self._levels = []
        self._silence_frames = 0
        self._speech_frames = 0
        self._last_partial_frames = 0
