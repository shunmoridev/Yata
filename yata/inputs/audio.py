from __future__ import annotations

from typing import Any

import av
import numpy as np


class StreamResampler:
    def __init__(self, source_rate: int, target_rate: int = 16_000) -> None:
        if not 8_000 <= source_rate <= 192_000:
            raise ValueError("Audio sample rate is outside the supported range.")
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._resampler = None if source_rate == target_rate else av.AudioResampler(format="fltp", layout="mono", rate=target_rate)

    def push(self, samples: bytes) -> np.ndarray:
        if len(samples) % 4:
            raise ValueError("PCM float32 payload is not aligned.")
        source = np.frombuffer(samples, dtype="<f4")
        if source.size and not np.isfinite(source).all():
            raise ValueError("PCM contains a non-finite sample.")
        if self._resampler is None or source.size == 0:
            return source.astype(np.float32, copy=True)
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(source.reshape(1, -1)), format="flt", layout="mono")
        frame.sample_rate = self.source_rate
        return self._collect(self._resampler.resample(frame))

    def flush(self) -> np.ndarray:
        if self._resampler is None:
            return np.empty(0, dtype=np.float32)
        output = self._collect(self._resampler.resample(None))
        self._resampler = av.AudioResampler(format="fltp", layout="mono", rate=self.target_rate)
        return output

    @staticmethod
    def _collect(frames: list[Any]) -> np.ndarray:
        if not frames:
            return np.empty(0, dtype=np.float32)
        return np.ascontiguousarray(np.concatenate([item.to_ndarray().reshape(-1) for item in frames]).astype(np.float32, copy=False))


def resample_mono_float32(samples: bytes, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    if not 8_000 <= source_rate <= 192_000 or source_rate == 0:
        raise ValueError("Audio sample rate is outside the supported range.")
    if len(samples) % 4:
        raise ValueError("PCM float32 payload is not aligned.")
    source = np.frombuffer(samples, dtype="<f4")
    if source.size == 0:
        return np.empty(0, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("PCM contains a non-finite sample.")
    source = np.ascontiguousarray(source.reshape(1, -1))
    frame = av.AudioFrame.from_ndarray(source, format="flt", layout="mono")
    frame.sample_rate = source_rate
    converter = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
    output = [converted.to_ndarray().reshape(-1).astype(np.float32, copy=False) for converted in converter.resample(frame)]
    output.extend(converted.to_ndarray().reshape(-1).astype(np.float32, copy=False) for converted in converter.resample(None))
    if not output:
        return np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(output))
