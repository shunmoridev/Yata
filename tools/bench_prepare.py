"""Parquet shards are read by row group over HTTP range requests and the YODAS2 tar
is streamed until the wanted videos appear, so only the needed audio is fetched.
"""
from __future__ import annotations

import io
import json
import tarfile
import wave
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "data"
RATE = 16_000
CLIP_SECONDS = 180

YODAS_SHARD = "00000000"
YODAS_CLIPS = {
    "Y6ojv3ZK2OI": ("ja_talk", "casual monologue with fillers", 40.0),
    "Yvxe6zMziwY": ("ja_tech", "tech walkthrough mixing English product names", 30.0),
    "YCofEN1fGPI": ("ja_lecture", "lecture with fractions and probabilities", 110.0),
}
EARNINGS_SHARD = "chunked/test-00034-of-00038-4dd1ba7370c3337f.parquet"
FLEURS = {"ja": "ja_jp", "zh": "cmn_hans_cn", "ko": "ko_kr", "fr": "fr_fr", "es": "es_419"}
FLEURS_UTTERANCES = 6


def decode(data: bytes) -> np.ndarray:
    resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):
        chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, np.float32)


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())


def entry(clip_id: str, language: str, style: str, audio: np.ndarray, reference: str, source: str, license_: str) -> dict:
    path = OUT / f"{clip_id}.wav"
    write_wav(path, audio)
    print(f"  wrote {path.name}: {len(audio) / RATE:.1f}s, {len(reference)} ref chars")
    return {
        "id": clip_id, "path": path.name, "language": language, "style": style,
        "duration_s": round(len(audio) / RATE, 2), "reference": reference,
        "source": source, "license": license_,
    }


def yodas() -> list[dict]:
    print("YODAS2 ja000: reading subtitles")
    text_path = hf_hub_download("espnet/yodas2", f"data/ja000/text/{YODAS_SHARD}.json", repo_type="dataset")
    subtitles = {item["audio_id"]: item["text"] for item in json.loads(Path(text_path).read_text(encoding="utf-8"))}
    wanted = dict(YODAS_CLIPS)
    results = []
    fs = HfFileSystem()
    print("YODAS2 ja000: streaming audio tar until the wanted videos appear")
    with fs.open(f"datasets/espnet/yodas2/data/ja000/audio/{YODAS_SHARD}.tar.gz", "rb", block_size=8 * 1024 * 1024) as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as archive:
            for member in archive:
                audio_id = Path(member.name).stem
                if audio_id not in wanted or not member.isfile():
                    continue
                clip_id, style, start = wanted.pop(audio_id)
                audio = decode(archive.extractfile(member).read())
                segments = sorted(
                    (int(key.rsplit("-", 2)[1]) / 100, int(key.rsplit("-", 2)[2]) / 100, text.strip())
                    for key, text in subtitles[audio_id].items()
                )
                # Snap the window to subtitle boundaries so the reference matches the audio exactly.
                first = next(i for i, seg in enumerate(segments) if seg[0] >= start)
                window_start = segments[first][0]
                chosen = [seg for seg in segments[first:] if seg[1] <= window_start + CLIP_SECONDS]
                window_end = chosen[-1][1]
                clip = audio[int(window_start * RATE): int(window_end * RATE)]
                reference = "".join(text for _s, _e, text in chosen)
                results.append(entry(
                    clip_id, "ja", style, clip, reference,
                    f"espnet/yodas2 ja000 {audio_id} {window_start:.2f}-{window_end:.2f}s", "CC-BY-3.0",
                ))
                if not wanted:
                    break
    for audio_id in wanted:
        print(f"  not found in shard: {audio_id}")
    return results


def earnings() -> dict:
    print("Earnings-22: reading segment index")
    fs = HfFileSystem()
    with fs.open(f"datasets/distil-whisper/earnings22/{EARNINGS_SHARD}", "rb", block_size=4 * 1024 * 1024) as raw:
        parquet = pq.ParquetFile(raw)
        index = parquet.read(columns=["file_id", "start_ts", "end_ts"]).to_pylist()
        rows_per_group = [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
        file_id = max({row["file_id"] for row in index}, key=lambda fid: sum(r["file_id"] == fid for r in index))
        mine = sorted((row["start_ts"], i) for i, row in enumerate(index) if row["file_id"] == file_id)
        # Skip the operator intro and take ~3 minutes of consecutive segments.
        start_at = mine[len(mine) // 3][0]
        picked = [(ts, i) for ts, i in mine if start_at <= ts and index[i]["end_ts"] <= start_at + CLIP_SECONDS]
        wanted_rows = {i for _ts, i in picked}
        groups, offsets, offset = set(), [], 0
        for group, count in enumerate(rows_per_group):
            offsets.append(offset)
            if any(offset <= row < offset + count for row in wanted_rows):
                groups.add(group)
            offset += count
        rows: dict[int, dict] = {}
        for group in sorted(groups):
            table = parquet.read_row_group(group, columns=["audio", "transcription", "start_ts", "end_ts"]).to_pylist()
            for local, row in enumerate(table):
                if offsets[group] + local in wanted_rows:
                    rows[offsets[group] + local] = row
    timeline = np.zeros(int((index[picked[-1][1]]["end_ts"] - start_at) * RATE) + RATE, dtype=np.float32)
    texts = []
    for ts, i in picked:
        samples = decode(rows[i]["audio"]["bytes"])
        begin = int((ts - start_at) * RATE)
        end = min(len(timeline), begin + len(samples))
        timeline[begin:end] = samples[: end - begin]
        texts.append(rows[i]["transcription"].strip())
    return entry(
        "en_earnings", "en", "earnings call presentation and Q&A with numbers", timeline, " ".join(texts),
        f"distil-whisper/earnings22 {file_id} {start_at:.1f}s+", "CC-BY-SA-4.0",
    )


def fleurs() -> list[dict]:
    results = []
    fs = HfFileSystem()
    gap = np.zeros(int(0.8 * RATE), dtype=np.float32)
    for language, config in FLEURS.items():
        print(f"FLEURS {config}: reading first row group")
        with fs.open(f"datasets/google/fleurs/parquet-data/{config}/validation-00000-of-00001.parquet", "rb",
                     block_size=4 * 1024 * 1024) as raw:
            parquet = pq.ParquetFile(raw)
            table = parquet.read_row_group(0, columns=["audio", "raw_transcription"]).to_pylist()
        pieces, texts = [], []
        for row in table[:FLEURS_UTTERANCES]:
            pieces += [decode(row["audio"]["bytes"]), gap]
            texts.append(row["raw_transcription"].strip())
        results.append(entry(
            f"{language}_fleurs", language, "read sentences", np.concatenate(pieces),
            " ".join(texts), f"google/fleurs {config} validation rows 0-{FLEURS_UTTERANCES - 1}", "CC-BY-4.0",
        ))
    return results


def noise() -> dict:
    rng = np.random.default_rng(7)
    silence = np.zeros(10 * RATE, dtype=np.float32)
    white = rng.standard_normal(10 * RATE).astype(np.float32)
    pink = np.cumsum(white) / 400
    pink = (pink - np.convolve(pink, np.ones(800) / 800, mode="same")) * 0.01
    t = np.arange(10 * RATE) / RATE
    tones = (0.05 * np.sin(2 * np.pi * (220 + 220 * (t % 1)) * t)).astype(np.float32)
    return entry("silence_noise", "", "silence, low noise and tones; expect no text",
                 np.concatenate([silence, pink.astype(np.float32), tones]), "", "synthetic", "n/a")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = yodas() + [earnings()] + fleurs() + [noise()]
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT.parent / "README.md").write_text(
        "Local ASR benchmark data. Not tracked by git. Licenses per clip are in data/manifest.json.\n"
        "Rebuild with: uv run --group bench python tools/bench_prepare.py\n",
        encoding="utf-8",
    )
    print(f"manifest: {len(manifest)} clips")


if __name__ == "__main__":
    main()
