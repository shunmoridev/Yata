from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data"
RESULTS = ROOT / "bench" / "results"
MODEL_ROOT = Path(os.environ.get("YATA_MODEL_ROOT", r"F:\model\Yata"))
RATE = 16_000
CHAR_LANGS = {"ja", "zh", "ko"}
QWEN_NAMES = {
    "japanese": "ja", "english": "en", "chinese": "zh", "korean": "ko", "french": "fr",
    "spanish": "es", "german": "de", "cantonese": "yue",
}

ENGINES = {
    "fw-tiny": ("faster-whisper", "faster-whisper-tiny", "cuda", "float16"),
    "fw-turbo": ("faster-whisper", "faster-whisper-large-v3-turbo", "cuda", "float16"),
    "qwen-0.6b": ("qwen3-asr", "Qwen3-ASR-0.6B-hf", "cuda", "bfloat16"),
    "qwen-1.7b": ("qwen3-asr", "Qwen3-ASR-1.7B-hf", "cuda", "bfloat16"),
    "fw-turbo-cpu": ("faster-whisper", "faster-whisper-large-v3-turbo", "cpu", "int8"),
    "fw-small-cpu": ("faster-whisper", "faster-whisper-small", "cpu", "int8"),
    "fw-medium-cpu": ("faster-whisper", "faster-whisper-medium", "cpu", "int8"),
    "qwen-0.6b-cpu": ("qwen3-asr", "Qwen3-ASR-0.6B-hf", "cpu", "float32"),
}


class FasterWhisperEngine:
    def __init__(self, path: Path, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(path), device=device, compute_type=compute_type, local_files_only=True)

    def transcribe(self, audio: np.ndarray, language: str | None) -> tuple[str, str | None]:
        segments, info = self.model.transcribe(
            audio, language=language, beam_size=1, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True,
        )
        return "".join(segment.text for segment in segments).strip(), info.language


class Qwen3AsrEngine:
    def __init__(self, path: Path, device: str, dtype: str) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(path), dtype=getattr(torch, dtype), local_files_only=True,
        ).to(device).eval()

    def transcribe(self, audio: np.ndarray, language: str | None) -> tuple[str, str | None]:
        inputs = self.processor.apply_transcription_request(audio=audio, language=language)
        inputs = inputs.to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=384, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        parsed = self.processor.decode(generated, return_format="parsed")[0]
        detected = QWEN_NAMES.get(str(parsed.get("language", "")).strip().lower()) if not language else language
        return str(parsed.get("transcription", "")).strip(), detected


def load_engine(name: str):
    kind, folder, device, precision = ENGINES[name]
    cls = FasterWhisperEngine if kind == "faster-whisper" else Qwen3AsrEngine
    return cls(MODEL_ROOT / folder, device, precision)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == RATE and handle.getnchannels() == 1
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768


def vad_utterances(audio: np.ndarray) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    options = VadOptions(threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=200, max_speech_duration_s=15)
    return [(item["start"], item["end"]) for item in get_speech_timestamps(audio, options)]


def normalize(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    kept = "".join(" " if unicodedata.category(ch)[0] in "PSZ" else ch for ch in text)
    if language in CHAR_LANGS:
        return [ch for ch in kept if not ch.isspace()]
    return kept.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, other in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (token != other))
        previous = current
    return previous[-1]


class VramMonitor:
    """Polls nvidia-smi so CTranslate2 and torch are measured the same way."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def used_mib() -> int:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            return int(output.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used_mib())
            self._stop.wait(0.25)

    def __enter__(self) -> "VramMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()


def run_clip(engine, clip: dict, audio: np.ndarray, mode: str, truncated: bool) -> dict:
    expected = clip["language"] or None
    forced = expected if mode == "oracle" else None
    utterances = vad_utterances(audio)
    texts, latencies, detected = [], [], []
    for start, end in utterances:
        began = time.perf_counter()
        text, language = engine.transcribe(audio[start:end], forced)
        latencies.append(time.perf_counter() - began)
        texts.append(text)
        detected.append(language)
    joiner = "" if clip["language"] in CHAR_LANGS else " "
    hypothesis = joiner.join(text for text in texts if text)
    result = {
        "clip": clip["id"], "mode": mode, "language": clip["language"], "utterances": len(utterances),
        "audio_s": round(len(audio) / RATE, 2), "rtf": round(sum(latencies) / max(len(audio) / RATE, 1e-9), 4),
        "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_max_s": round(max(latencies), 3) if latencies else None,
        "max_utterance_s": round(max((e - s) / RATE for s, e in utterances), 2) if utterances else 0,
        "hypothesis": hypothesis,
    }
    if expected and truncated:
        # The reference covers the whole clip, so an error rate here would only count missing audio.
        result["metric"] = "speed-only"
    elif expected:
        ref = normalize(clip["reference"], clip["language"])
        hyp = normalize(hypothesis, clip["language"])
        result["metric"] = "CER" if clip["language"] in CHAR_LANGS else "WER"
        result["error_rate"] = round(edit_distance(ref, hyp) / max(len(ref), 1), 4)
        if mode == "auto":
            result["lid_accuracy"] = round(sum(d == expected for d in detected) / max(len(detected), 1), 3)
    else:
        # Hallucination check: the VAD path and a raw pass over the whole clip should both stay empty.
        raw_text, _lang = engine.transcribe(audio, None)
        result["hallucinated_chars_vad"] = len(hypothesis)
        result["hallucinated_chars_raw"] = len(raw_text)
        result["raw_text"] = raw_text[:200]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", default="fw-tiny,fw-turbo,qwen-0.6b,qwen-1.7b")
    parser.add_argument("--clips", default="")
    parser.add_argument("--modes", default="auto,oracle")
    parser.add_argument("--max-seconds", type=float, default=0)
    args = parser.parse_args()

    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if args.clips:
        wanted = set(args.clips.split(","))
        manifest = [clip for clip in manifest if clip["id"] in wanted]
    audio_by_clip, truncated = {}, set()
    for clip in manifest:
        audio = read_wav(DATA / clip["path"])
        if args.max_seconds and len(audio) > args.max_seconds * RATE:
            audio = audio[: int(args.max_seconds * RATE)]
            truncated.add(clip["id"])
        audio_by_clip[clip["id"]] = audio

    report = {"started_at": datetime.now().isoformat(timespec="seconds"), "engines": {}}
    for name in args.engines.split(","):
        baseline = VramMonitor.used_mib()
        with VramMonitor() as monitor:
            began = time.perf_counter()
            engine = load_engine(name)
            load_s = time.perf_counter() - began
            # One warm-up call so CUDA kernel setup is not billed to the first clip.
            engine.transcribe(np.zeros(RATE, dtype=np.float32), "en")
            rows = []
            for clip in manifest:
                modes = ["auto"] if not clip["language"] else args.modes.split(",")
                for mode in modes:
                    row = run_clip(engine, clip, audio_by_clip[clip["id"]], mode, clip["id"] in truncated)
                    rows.append(row)
                    if "error_rate" in row:
                        score = f"{row['metric']} {row['error_rate']:.3f}"
                    elif "hallucinated_chars_vad" in row:
                        score = f"halluc vad={row['hallucinated_chars_vad']} raw={row['hallucinated_chars_raw']}"
                    else:
                        score = row["metric"]
                    print(f"{name:14s} {clip['id']:13s} {mode:6s} {score:16s} rtf {row['rtf']:.3f} "
                          f"p50 {row['latency_p50_s']}s lid {row.get('lid_accuracy', '-')}", flush=True)
        report["engines"][name] = {
            "load_s": round(load_s, 1), "vram_mib": max(0, monitor.peak - baseline), "results": rows,
        }
        del engine
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"asr_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
