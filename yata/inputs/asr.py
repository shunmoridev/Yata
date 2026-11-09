from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..i18n import t


# Phrases Whisper-family models emit on silence or music. turbo reports
# no_speech_prob 0.0 for them, so the live segmenter drops them only when the
# utterance's VAD speech score was weak (see is_stock_phrase callers).
_HALLUCINATIONS = {
    "thank you", "thank you for watching", "thanks for watching", "please subscribe",
    "subscribe to my channel", "you", "bye", "ご視聴ありがとうございました", "ご清聴ありがとうございました",
    "チャンネル登録よろしくお願いします", "おやすみなさい", "字幕視聴ありがとうございました",
    "最後までご視聴頂きありがとうございました", "最後までご視聴いただきありがとうございました", "次回予告",
}
_REPEAT = re.compile(r"(.{2,30}?)\1{4,}")

LANGUAGE_NAMES = {
    "ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean", "fr": "French", "es": "Spanish",
    "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "id": "Indonesian", "th": "Thai", "vi": "Vietnamese", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    "sv": "Swedish", "yue": "Cantonese",
}
_NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


def is_stock_phrase(text: str) -> bool:
    key = re.sub(r"[\s。、．.!！?？,，…♪「」]+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return not key or key in _HALLUCINATIONS


def clean_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for item in results:
        text = _REPEAT.sub(r"\1\1", str(item.get("text", "")).strip())
        if text:
            cleaned.append({**item, "text": text})
    return cleaned


def _hf_snapshot(repo_dir: str) -> Path | None:
    cache = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub") / repo_dir
    try:
        revision = (cache / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    candidate = cache / "snapshots" / revision
    return candidate if candidate.is_dir() else None


class LocalAsr:
    name = "asr"
    weights_file = "model.bin"
    fast_partials = False
    context_prompt = False
    settings: "AsrSettings | None" = None

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else self.default_model_path()
        self._model: Any = None
        self._lock = threading.Lock()

    @classmethod
    def default_model_path(cls) -> Path | None:
        return None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        self._load()

    def runtime_missing(self) -> str | None:
        return None

    def readiness(self) -> dict[str, Any]:
        if self.model_path is None:
            return {"state": "missing", "weights_present": False, "reason": t("ASR モデルのローカルパスが未設定です。", "No folder is set for the ASR model.")}
        weights = self.model_path / self.weights_file if self.model_path.is_dir() else self.model_path
        if not weights.is_file():
            return {"state": "missing", "weights_present": False, "reason": t(f"指定先に {self.weights_file} がありません。", f"The folder has no {self.weights_file}.")}
        try:
            size = weights.stat().st_size
        except OSError:
            size = 0
        if size <= 0:
            return {"state": "invalid", "weights_present": False, "reason": t(f"ASR {self.weights_file} が空です。", f"The ASR {self.weights_file} is empty.")}
        missing = self.runtime_missing()
        if missing:
            return {"state": "missing", "weights_present": False, "reason": missing}
        return {"state": "weights_present", "weights_present": True, "reason": t("重みを検出しました。実ロードは音声取得後に行います。",
                                                                                 "Weights found. The model loads once audio arrives.")}

    def _load(self) -> Any:
        ready = self.readiness()
        if not ready["weights_present"]:
            raise RuntimeError(ready["reason"])
        with self._lock:
            if self._model is None:
                assert self.model_path is not None
                root = self.model_path if self.model_path.is_dir() else self.model_path.parent
                self._model = self._open(root)
        return self._model

    def _open(self, root: Path) -> Any:
        raise NotImplementedError

    def transcribe(self, audio_16k_mono: Any, language: str | None = None, prompt: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError


class FasterWhisperEngine(LocalAsr):
    name = "faster-whisper"
    fast_partials = True

    def __init__(self, model_path: str | Path | None = None, device: str = "auto", compute_type: str = "default",
                 cpu_threads: int = 0) -> None:
        super().__init__(model_path)
        self.device = device
        self.compute_type = compute_type
        # 0 keeps CTranslate2's default (4 threads); only used on CPU.
        self.cpu_threads = cpu_threads

    @classmethod
    def default_model_path(cls) -> Path | None:
        return next(filter(None, map(_hf_snapshot, PROFILE_HF_REPOS["full"])), None)

    def _open(self, root: Path) -> Any:
        use_bundled_cuda()
        from faster_whisper import WhisperModel
        return WhisperModel(str(root), device=self.device, compute_type=self.compute_type, cpu_threads=self.cpu_threads,
                            local_files_only=True)

    def transcribe(self, audio_16k_mono: Any, language: str | None = None, prompt: str | None = None) -> list[dict[str, Any]]:
        model = self._load()
        segments, info = model.transcribe(
            audio_16k_mono,
            language=language,
            initial_prompt=prompt or None,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        results = []
        for item in segments:
            # Whisper's own silence guard: likely non-speech and low confidence together.
            if item.no_speech_prob > 0.6 and item.avg_logprob < -1.0:
                continue
            results.append({
                "start": float(item.start), "end": float(item.end), "text": item.text.strip(),
                "language": info.language, "language_probability": float(info.language_probability),
                "avg_logprob": float(item.avg_logprob),
            })
        return clean_results(results)


class Qwen3AsrEngine(LocalAsr):
    name = "qwen3-asr"
    weights_file = "model.safetensors"
    context_prompt = True

    def __init__(self, model_path: str | Path | None = None, device: str = "auto") -> None:
        super().__init__(model_path)
        self.device = device
        self._processor: Any = None

    def runtime_missing(self) -> str | None:
        for module in ("torch", "transformers"):
            if importlib.util.find_spec(module) is None:
                return t(f"Qwen3-ASR には {module} が必要です（uv sync --inexact --extra qwen）。", f"Qwen3-ASR needs {module} (uv sync --inexact --extra qwen).")
        return None

    def _open(self, root: Path) -> Any:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(str(root), local_files_only=True)
        return AutoModelForMultimodalLM.from_pretrained(str(root), dtype=dtype, local_files_only=True).to(device).eval()

    def transcribe(self, audio_16k_mono: Any, language: str | None = None, prompt: str | None = None) -> list[dict[str, Any]]:
        import torch
        model = self._load()
        inputs = self._processor.apply_transcription_request(audio=audio_16k_mono, language=language, prompt=prompt or None)
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=384, do_sample=False)
        parsed = self._processor.decode(output[:, inputs["input_ids"].shape[1]:], return_format="parsed")[0]
        detected = language or _NAME_TO_CODE.get(str(parsed.get("language", "")).strip().lower())
        duration = len(audio_16k_mono) / 16_000
        return clean_results([{
            "start": 0.0, "end": duration, "text": str(parsed.get("transcription", "")).strip(),
            "language": detected, "language_probability": None,
        }])


ENGINES = {FasterWhisperEngine.name: FasterWhisperEngine, Qwen3AsrEngine.name: Qwen3AsrEngine}

PROFILES = ("auto", "full", "light")
# faster-whisper model folders under YATA_MODEL_ROOT (the catalog's folders, yata/catalog.py), and the
# Hugging Face cache repos used without one (turbo moved from mobiuslabsgmbh to dropbox-dash).
PROFILE_MODELS = {"full": "faster-whisper-large-v3-turbo", "light": "faster-whisper-small"}
PROFILE_HF_REPOS = {
    "full": ("models--dropbox-dash--faster-whisper-large-v3-turbo", "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"),
    "light": ("models--Systran--faster-whisper-small",),
}
COMPUTE_TYPES = frozenset({"default", "auto", "int8", "int8_float16", "int8_float32", "int8_bfloat16", "int16",
                           "float16", "bfloat16", "float32"})
_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}


@dataclass(frozen=True)
class AsrSettings:
    profile: str
    engine: str
    device: str
    compute_type: str
    model_path: Path | None
    cpu_threads: int
    partials: bool
    partial_model_path: Path | None

    def describe(self) -> str:
        model = self.model_path.name if self.model_path else t("（モデル未設定）", "(no model set)")
        threads = f", {self.cpu_threads} threads" if self.device == "cpu" and self.cpu_threads else ""
        return (f"{self.engine} {model} · {self.device.upper()} {self.compute_type}{threads} · {self.profile}"
                + (t(" · 部分字幕あり", " · live captions") if self.partials else t(" · 部分字幕なし", " · no live captions")))


def resolve_asr_settings(environ: dict[str, str] | None = None, *, cuda: bool | Callable[[], bool] | None = None) -> AsrSettings:
    env = os.environ if environ is None else environ
    probed: list[bool] = [] if cuda is None or callable(cuda) else [cuda]

    def has_cuda() -> bool:
        # Probing initialises the CUDA runtime (hundreds of MiB), so it only happens when the answer matters.
        if not probed:
            probed.append(cuda() if callable(cuda) else _cuda_available())
        return probed[0]

    value = lambda name: env.get(name, "").strip()  # noqa: E731
    profile = value("YATA_ASR_PROFILE").lower() or "auto"
    if profile not in PROFILES:
        profile = "auto"
    if profile == "auto":
        profile = "full" if has_cuda() else "light"
    device = value("YATA_ASR_DEVICE").lower() or ("cpu" if profile == "light" else "auto")
    if device not in {"auto", "cuda", "cpu"}:
        device = "auto"
    if device == "auto":
        device = "cuda" if has_cuda() else "cpu"
    compute_type = value("YATA_ASR_COMPUTE_TYPE").lower()
    if compute_type not in COMPUTE_TYPES:
        # int8 is about a third faster than the stored float weights on CPU with little loss for small.
        compute_type = "int8" if device == "cpu" else "default"
    engine = value("YATA_ASR_ENGINE") or FasterWhisperEngine.name
    if engine not in ENGINES:
        engine = FasterWhisperEngine.name
    model_path: Path | None = Path(value("YATA_ASR_MODEL_PATH")).expanduser() if value("YATA_ASR_MODEL_PATH") else None
    if model_path is None and engine == FasterWhisperEngine.name:
        root = value("YATA_MODEL_ROOT")
        candidate = Path(root).expanduser() / PROFILE_MODELS[profile] if root else None
        model_path = candidate if candidate is not None and candidate.is_dir() else next(
            filter(None, map(_hf_snapshot, PROFILE_HF_REPOS[profile])), None)
    try:
        cpu_threads = max(0, int(value("YATA_ASR_CPU_THREADS") or 0))
    except ValueError:
        cpu_threads = 0
    partials_value = value("YATA_ASR_PARTIALS").lower()
    # On CPU even whisper-small needs most of a second per utterance, so drafting every second cannot keep up.
    partials = partials_value in _ON or (partials_value not in _OFF and device == "cuda")
    partial_model = Path(value("YATA_ASR_PARTIAL_MODEL_PATH")).expanduser() if value("YATA_ASR_PARTIAL_MODEL_PATH") else None
    return AsrSettings(profile=profile, engine=engine, device=device, compute_type=compute_type, model_path=model_path,
                       cpu_threads=cpu_threads, partials=partials, partial_model_path=partial_model)


def create_asr_engines(environ: dict[str, str] | None = None, *,
                       cuda: bool | Callable[[], bool] | None = None) -> tuple[LocalAsr, LocalAsr | None]:
    settings = resolve_asr_settings(environ, cuda=cuda)
    if settings.engine == FasterWhisperEngine.name:
        final: LocalAsr = FasterWhisperEngine(settings.model_path, device=settings.device, compute_type=settings.compute_type,
                                              cpu_threads=settings.cpu_threads)
    else:
        final = ENGINES[settings.engine](settings.model_path, device=settings.device)
    final.settings = settings
    if not settings.partials:
        return final, None
    partial: LocalAsr | None = None
    if final.fast_partials:
        partial = final
    elif settings.partial_model_path is not None:
        partial = FasterWhisperEngine(settings.partial_model_path, device=settings.device, compute_type=settings.compute_type,
                                      cpu_threads=settings.cpu_threads)
    return final, partial


# What CTranslate2 loads by name to run on CUDA: cuBLAS for CUDA 12, and on Linux also cuDNN 9. With the NVIDIA
# driver alone it still counts the GPU, but its first CUDA call then fails and, in 4.8.5, the next one in the
# process hangs (tools/portable_cuda_check.py); so a GPU counts only when these load. macOS has no CUDA.
CUDA_LIBRARIES = {"win32": ("cublas64_12.dll", "cublasLt64_12.dll"),
                  "linux": ("libcublasLt.so.12", "libcublas.so.12", "libcudnn.so.9")}


def bundled_cuda_folders() -> list[Path]:
    folders = []
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        for location in (spec.submodule_search_locations or []) if spec is not None else []:
            folders += [Path(location) / name for name in ("bin", "lib") if (Path(location) / name).is_dir()]
    return folders


def use_bundled_cuda(platform: str = sys.platform) -> None:
    """Windows: put the `cuda` extra's DLL folders first on PATH, which Windows searches when CTranslate2 loads
    cuBLAS by name; without the extra nothing changes. Linux reads LD_LIBRARY_PATH only when a process starts,
    so there the launcher sets it (start-mac-linux.sh)."""
    if platform != "win32":
        return
    current = os.environ.get("PATH", "").split(os.pathsep)
    missing = [str(folder) for folder in bundled_cuda_folders() if str(folder) not in current]
    if missing:
        os.environ["PATH"] = os.pathsep.join(missing + current)


def cuda_libraries_load(platform: str = sys.platform, load: Callable[[str], Any] | None = None) -> bool:
    names = CUDA_LIBRARIES.get(platform)
    if not names:
        return False
    if load is None:
        # winmode=0: Windows' standard search order (with PATH), as CTranslate2's LoadLibrary uses.
        load = (lambda name: ctypes.WinDLL(name, winmode=0)) if platform == "win32" else ctypes.CDLL
    for name in names:
        try:
            load(name)
        except OSError:
            return False
    return True


def _cuda_available() -> bool:
    use_bundled_cuda()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False
    return cuda_libraries_load()
