"""CAM++ is the model: on this machine's benchmark clips it separated different speakers (cosine p95 0.23)
from the same speaker (p5 0.53); WeSpeaker ResNet34 trained on VoxCeleb could not tell two Japanese
speakers apart, so it is not the default.
"""
from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..i18n import t

DEFAULT_MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"


class SpeakerEmbedder:
    def __init__(self, model_path: str | Path | None = None) -> None:
        configured = model_path or os.environ.get("YATA_SPEAKER_MODEL_PATH", "").strip()
        self.model_path = Path(configured).expanduser() if configured else None
        self._extractor: Any = None
        self._lock = threading.Lock()

    def readiness(self) -> dict[str, Any]:
        if self.model_path is None:
            return {"enabled": False, "reason": t("話者ラベルのモデルが未設定です（YATA_SPEAKER_MODEL_PATH）。",
                                                   "No speaker label model is set (YATA_SPEAKER_MODEL_PATH).")}
        if not self.model_path.is_file() or self.model_path.stat().st_size <= 0:
            return {"enabled": False, "reason": t("話者ラベルのモデルファイルがありません。", "The speaker label model file is missing.")}
        if importlib.util.find_spec("sherpa_onnx") is None:
            return {"enabled": False, "reason": t("話者ラベルには sherpa-onnx が必要です（uv sync --inexact --extra speakers）。",
                                                   "Speaker labels need sherpa-onnx (uv sync --inexact --extra speakers).")}
        return {"enabled": True, "reason": t("話者ラベルを付けます（自動判定のため誤ることがあります）。", "Labels speakers (automatic, so it can be wrong).")}

    def embed(self, audio_16k_mono: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._extractor is None:
                import sherpa_onnx
                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.model_path), num_threads=2)
                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            stream = self._extractor.create_stream()
            stream.accept_waveform(16_000, np.ascontiguousarray(audio_16k_mono, dtype=np.float32))
            stream.input_finished()
            vector = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class SpeakerTracker:
    def __init__(self, threshold: float = 0.45, max_speakers: int = 8, min_seconds: float = 1.0) -> None:
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.min_seconds = min_seconds
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []

    def assign(self, embedding: np.ndarray, duration_s: float) -> str | None:
        if duration_s < self.min_seconds or not embedding.size:
            return None
        if self._centroids:
            similarities = [float(centroid @ embedding) for centroid in self._centroids]
            best = int(np.argmax(similarities))
            if similarities[best] >= self.threshold or len(self._centroids) >= self.max_speakers:
                # Running mean with a cap, so a speaker's voice can drift slowly over a long stream.
                weight = min(self._counts[best], 20)
                merged = self._centroids[best] * weight + embedding
                self._centroids[best] = merged / float(np.linalg.norm(merged))
                self._counts[best] += 1
                return f"S{best + 1}"
        self._centroids.append(embedding)
        self._counts.append(1)
        return f"S{len(self._centroids)}"
