"""Soak test: stream long audio through Yata's live path in real time and measure delay, memory and loss."""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import platform
import statistics
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "bench" / "data"
RESULTS = ROOT / "bench" / "results"
BROWSER_RATE = 48_000
FRAME = 4096  # samples per frame at 48 kHz (about 85 ms), like the AudioWorklet batches
GAP_S = 1.5


def pin_cores(count: int, first: int = 0) -> list[int]:
    """Windows numbers the two hardware threads of a core next to each other, so 4 logical CPUs from 0 are
    2 physical cores with SMT, roughly a 2-core / 4-thread entry CPU (but with this host's per-core speed).
    """
    total = os.cpu_count() or 1
    if not count:
        return list(range(total))
    cores = [core for core in range(first, first + count) if core < total] or [0]
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        mask = sum(1 << core for core in cores)
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    else:
        os.sched_setaffinity(0, cores)
    return cores


class _MemoryCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) if name != "cb" and name != "PageFaultCount" else (name, ctypes.c_uint32) for name in (
        "cb", "PageFaultCount", "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
        "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


def memory_mb() -> dict[str, float]:
    if sys.platform == "win32":
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return {"working_set": counters.WorkingSetSize / 2**20, "private": counters.PrivateUsage / 2**20}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {"working_set": peak, "private": peak}


def load_clip(name: str) -> Any:
    import numpy as np

    from yata.inputs.audio import StreamResampler
    with wave.open(str(DATA / f"{name}.wav"), "rb") as handle:
        rate, width, channels = handle.getframerate(), handle.getsampwidth(), handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{name}: expected 16-bit PCM")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate == BROWSER_RATE:
        return audio
    resampler = StreamResampler(rate, BROWSER_RATE)
    return np.concatenate([resampler.push(audio.tobytes()), resampler.flush()]).astype(np.float32)


def frames(audio: dict[str, Any], clips: list[str], minutes: float, order: list[str]):
    """Frames are cut from the loaded clips as they are fed, so the tool holds each clip once, not the stream.
    """
    import numpy as np

    silence = np.zeros(FRAME, dtype=np.float32)
    remaining = int(minutes * 60 * BROWSER_RATE)
    while remaining > 0:
        for name in clips:
            order.append(name)
            clip = audio[name]
            for start in range(0, clip.size, FRAME):
                piece = clip[start:start + min(FRAME, remaining)]
                remaining -= piece.size
                yield piece
                if remaining <= 0:
                    return
            for _ in range(int(GAP_S * BROWSER_RATE / FRAME)):
                remaining -= FRAME
                yield silence
                if remaining <= 0:
                    return


def percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))]


def summary(values: list[float]) -> dict[str, float | None]:
    return {"p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": max(values) if values else None,
            "mean": statistics.fmean(values) if values else None}


class _NoLLM:
    def chat_json(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("the soak test runs captions only")


def prepare(args: argparse.Namespace) -> list[int]:
    """Set the process up like the smaller PC before any library starts its thread pools."""
    cores = pin_cores(args.cores, args.first_core)
    if args.cores:
        # OpenBLAS (numpy) commits a buffer per thread it sees: about 750 MiB on a 24-thread host, 110 MiB on 4.
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ.setdefault(name, str(len(cores)))
    if args.hide_gpu:
        # Like a PC without an NVIDIA GPU: no CUDA runtime is loaded at all.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return cores


async def run(args: argparse.Namespace, cores: list[int]) -> dict[str, Any]:
    env = dict(os.environ)
    if args.profile:
        env["YATA_ASR_PROFILE"] = args.profile
    if args.cpu_threads:
        env["YATA_ASR_CPU_THREADS"] = str(args.cpu_threads)
    if args.compute_type:
        env["YATA_ASR_COMPUTE_TYPE"] = args.compute_type
    if args.partials:
        env["YATA_ASR_PARTIALS"] = args.partials

    from yata.inputs.asr import create_asr_engines
    from yata.inputs.audio import StreamResampler
    from yata.inputs.vad import RATE, SileroVad, UtteranceSegmenter
    from yata.live import LiveTranscriber
    from yata.models import SessionStore
    from yata.state import SessionManager
    from yata.topics import TopicPipeline

    clips = [name.strip() for name in args.clips.split(",") if name.strip()]
    audio = {name: load_clip(name) for name in clips}
    audio_buffer_mb = sum(clip.nbytes for clip in audio.values()) / 2**20
    order: list[str] = []
    memory = {"start": memory_mb()}
    engine, partial_engine = create_asr_engines(env)
    settings = engine.settings
    print("ASR:", settings.describe() if settings else engine.name, "| cores:", len(cores), flush=True)
    loaded_at = time.perf_counter()
    await asyncio.to_thread(engine._load)
    if partial_engine is not None and partial_engine is not engine:
        await asyncio.to_thread(partial_engine._load)
    load_s = time.perf_counter() - loaded_at
    memory["after_load"] = memory_mb()

    manager = SessionManager(TopicPipeline(_NoLLM()), SessionStore(), background=False)
    session = manager.create(source_url="", source_title="", source_kind="share", selected_model="", summaries=False,
                             asr_language=args.language)
    manager.set_status(session, "capturing_audio")
    ends: dict[str, float] = {}
    delays: list[float] = []
    asr_ms: list[float] = []
    counts = {"finals": 0, "captions": 0, "empty": 0, "partials": 0, "gaps": 0}

    async def send(message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "caption_partial":
            counts["partials"] += 1
            return
        if kind not in {"caption_final", "caption_clear"}:
            return
        counts["finals"] += 1
        counts["captions" if kind == "caption_final" else "empty"] += 1
        duration = session.audio_diagnostics.get("asr_last_duration_ms")
        if isinstance(duration, (int, float)):
            asr_ms.append(float(duration))
        end = ends.pop(message.get("utterance_id"), None)
        if end is not None and kind == "caption_final":
            delays.append(time.perf_counter() - end)

    def record_gap(*_gap: Any) -> None:
        counts["gaps"] += 1

    transcriber = LiveTranscriber(
        session=session, ingest=lambda segments: manager.ingest(session, segments, trusted_source_type="asr"),
        record_gap=record_gap, engine=engine, partial_engine=partial_engine, base_capture_ms=0, send=send,
        on_evidence=lambda: None, language=args.language,
    )
    worker = asyncio.create_task(transcriber.run())
    segmenter = UtteranceSegmenter(SileroVad())
    segmenter.reset(0)
    resampler = StreamResampler(BROWSER_RATE)
    timeline: list[dict[str, Any]] = []
    peak = dict(memory["after_load"])
    queue_max = {"utterances": 0, "ms": 0}
    started = time.perf_counter()
    cpu_started = time.process_time()
    last = (started, cpu_started)

    def sample(now: float) -> None:
        nonlocal last
        current = memory_mb()
        for key, value in current.items():
            peak[key] = max(peak[key], value)
        cpu = time.process_time()
        busy = (cpu - last[1]) / max(1e-6, now - last[0])
        last = (now, cpu)
        queued = session.audio_diagnostics.get("asr_queue_utterances", 0)
        queued_ms = session.audio_diagnostics.get("asr_queue_ms", 0)
        queue_max["utterances"] = max(queue_max["utterances"], queued)
        queue_max["ms"] = max(queue_max["ms"], queued_ms)
        timeline.append({"t_s": round(now - started, 1), "working_set_mb": round(current["working_set"], 1),
                         "private_mb": round(current["private"], 1), "cpu_cores_busy": round(busy, 2),
                         "queue": queued, "queue_ms": queued_ms, "captions": counts["captions"]})

    next_sample = started
    fed_samples = 0
    for piece in frames(audio, clips, args.minutes, order):
        target = started + fed_samples / BROWSER_RATE / args.speed
        now = time.perf_counter()
        if target > now:
            await asyncio.sleep(target - now)
        fed_samples += piece.size
        utterances = segmenter.push(resampler.push(piece.tobytes()))
        fed_until = time.perf_counter()
        for utterance in utterances:
            if utterance.kind == "final":
                # When the utterance's last sample was fed: the moment the speaker stopped, as heard live.
                ends[utterance.id] = started + (utterance.end_sample / RATE) / args.speed
        transcriber.submit(utterances)
        if fed_until >= next_sample:
            sample(fed_until)
            next_sample = fed_until + args.sample_every
            if args.progress and len(timeline) % max(1, int(60 / args.sample_every)) == 0:
                row = timeline[-1]
                print(f"  {row['t_s']:7.1f}s  captions {row['captions']:4d}  queue {row['queue']}  "
                      f"cpu {row['cpu_cores_busy']:.2f} cores  ws {row['working_set_mb']:.0f} MiB", flush=True)
    feed_s = time.perf_counter() - started
    transcriber.submit(segmenter.push(resampler.flush()) + segmenter.flush())
    drained = await transcriber.drain(timeout=300)
    tail_s = time.perf_counter() - started - feed_s
    sample(time.perf_counter())
    transcriber.close()
    await worker
    wall_s = time.perf_counter() - started
    cpu_s = time.process_time() - cpu_started
    memory["peak"] = peak
    memory["end"] = memory_mb()
    audio_s = fed_samples / BROWSER_RATE
    diagnostics = session.audio_diagnostics
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                 "cores_used": len(cores), "first_core": cores[0] if cores else 0,
                 "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
        "asr": {"description": settings.describe() if settings else engine.name, "profile": getattr(settings, "profile", None),
                "device": getattr(settings, "device", None), "compute_type": getattr(settings, "compute_type", None),
                "cpu_threads": getattr(settings, "cpu_threads", None), "partials": partial_engine is not None, "load_s": round(load_s, 2)},
        "input": {"clips": clips, "order": order, "audio_s": round(audio_s, 1), "speed": args.speed, "language": args.language,
                  "audio_buffer_mb": round(audio_buffer_mb, 1), "gpu_hidden": args.hide_gpu},
        "run": {"wall_s": round(wall_s, 1), "feed_s": round(feed_s, 1), "tail_after_audio_s": round(tail_s, 1), "drained": drained},
        "captions": {**counts, "dropped_utterances": diagnostics.get("asr_dropped_utterances", 0),
                     "suppressed": diagnostics.get("asr_suppressed", 0), "music_windows": diagnostics.get("asr_music_windows", 0),
                     "errors": diagnostics.get("asr_errors", 0), "evidence_segments": len(session.evidence)},
        "delay_s": {key: round(value, 2) if value is not None else None for key, value in summary(delays).items()},
        "asr_ms": {key: round(value) if value is not None else None for key, value in summary(asr_ms).items()},
        "rtf": round(sum(asr_ms) / 1000 / audio_s, 3) if audio_s else None,
        "queue_max": queue_max,
        "memory_mb": {stage: {key: round(value, 1) for key, value in values.items()} for stage, values in memory.items()},
        "cpu": {"process_s": round(cpu_s, 1), "cores_busy_mean": round(cpu_s / wall_s, 2) if wall_s else None},
        "timeline": timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--minutes", type=float, default=10.0, help="length of the audio stream (clips repeat)")
    parser.add_argument("--clips", default="ja_talk,ja_tech,ja_lecture,en_earnings", help="bench/data clip names, in order")
    parser.add_argument("--profile", default="light", help="YATA_ASR_PROFILE for the run: light / full / auto")
    parser.add_argument("--cpu-threads", type=int, default=0, help="YATA_ASR_CPU_THREADS (0 = library default)")
    parser.add_argument("--compute-type", default="", help="YATA_ASR_COMPUTE_TYPE override")
    parser.add_argument("--partials", default="", help="YATA_ASR_PARTIALS override (on/off)")
    parser.add_argument("--cores", type=int, default=0, help="pin to this many logical CPUs (0 = all)")
    parser.add_argument("--first-core", type=int, default=0, help="first logical CPU to pin to (to run several soaks side by side)")
    parser.add_argument("--hide-gpu", action=argparse.BooleanOptionalAction, default=None,
                        help="hide CUDA like a PC without an NVIDIA GPU (default: on for the light profile)")
    parser.add_argument("--language", default="auto", help="stream language hint (auto or an ISO code)")
    parser.add_argument("--speed", type=float, default=1.0, help="feed speed; 1.0 is real time")
    parser.add_argument("--sample-every", type=float, default=5.0, help="seconds between memory/CPU/queue samples")
    parser.add_argument("--label", default="", help="added to the result file name")
    parser.add_argument("--progress", action="store_true", help="print a line every minute")
    args = parser.parse_args()
    if args.hide_gpu is None:
        args.hide_gpu = args.profile == "light"
    cores = prepare(args)
    result = asyncio.run(run(args, cores))
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"soak_{datetime.now():%Y%m%d_%H%M%S}_{args.profile}_{args.cores or 'all'}c{('_' + args.label) if args.label else ''}.json"
    (RESULTS / name).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {key: result[key] for key in ("asr", "input", "run", "captions", "delay_s", "asr_ms", "rtf", "queue_max", "memory_mb", "cpu")}
    brief["input"] = {key: value for key, value in brief["input"].items() if key != "order"}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print("saved", RESULTS / name)


if __name__ == "__main__":
    main()
