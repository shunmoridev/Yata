"""Check with a portable build's own Python that CTranslate2 runs on CUDA from the libraries in the folder."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path


def _executable() -> Path:
    """This python.exe, from Windows (sys.executable can be relative to a directory the caller left)."""
    buffer = ctypes.create_unicode_buffer(32768)
    ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer))
    return Path(buffer.value)


PYTHON = _executable()
RUNTIME = PYTHON.parent
FOLDER = RUNTIME.parent
# Libraries of the CUDA toolkit and cuDNN: these must come from the folder, never from a toolkit on the PC.
# The driver's own DLLs (nvcuda.dll and what it loads from System32 / DriverStore) are the PC's by design.
TOOLKIT_PREFIXES = ("cublas", "cudnn", "cudart", "nvrtc", "nvjitlink", "cufft", "curand", "cusparse", "cusolver",
                    "nvblas", "nppc", "nvinfer")
DRIVER_NAMES = ("nvcuda.dll",)
# What ctranslate2 4.8.5 loads by name for CUDA (cublas64_12.dll imports cublasLt64_12.dll). It names no
# cuDNN library: its Conv1D on CUDA is its own im2col + cuBLAS GEMM, so cuDNN is not bundled.
CUDA_LIBRARIES = ("cublas64_12.dll", "cublasLt64_12.dll")
COMPUTE_TYPES = ("float32", "float16", "int8_float16")  # turbo runs in float16 ("default" for its weights)
MELS, FRAMES, WIDTH, HEADS = 128, 3000, 64, 2  # large-v3(-turbo) has 128 mel bins and 30 s windows
# LoadLibraryW without flags: the same search as ctranslate2.dll's LoadLibraryA, in a separate process so
# that this one still loads the libraries through CTranslate2 itself.
FIND_BY_NAME = """
import ctypes, json, sys
from ctypes import wintypes
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.LoadLibraryW.restype = wintypes.HMODULE
kernel32.LoadLibraryW.argtypes = (wintypes.LPCWSTR,)
kernel32.GetModuleFileNameW.argtypes = (wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)
found = {}
for name in sys.argv[1:]:
    handle, buffer = kernel32.LoadLibraryW(name), ctypes.create_unicode_buffer(32768)
    found[name] = buffer.value if handle and kernel32.GetModuleFileNameW(handle, buffer, 32768) else None
print(json.dumps(found))
"""


def loaded_modules() -> list[Path]:
    psapi, kernel32 = ctypes.WinDLL("psapi", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModulesEx.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD), wintypes.DWORD)
    kernel32.GetModuleFileNameW.argtypes = (wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)
    process, needed = kernel32.GetCurrentProcess(), wintypes.DWORD()
    modules = (wintypes.HMODULE * 4096)()
    if not psapi.EnumProcessModulesEx(process, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3):
        raise OSError(ctypes.get_last_error(), "EnumProcessModulesEx failed")
    paths, buffer = [], ctypes.create_unicode_buffer(32768)
    for module in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        if kernel32.GetModuleFileNameW(module, buffer, len(buffer)):
            paths.append(Path(buffer.value))
    return paths


def cuda_libraries() -> list[dict]:
    found = []
    for path in loaded_modules():
        name = path.name.lower()
        if name.startswith(TOOLKIT_PREFIXES) or name in DRIVER_NAMES:
            inside = path.resolve().is_relative_to(FOLDER)
            found.append({"name": path.name, "path": str(path), "inside": inside,
                          "kind": "driver" if name in DRIVER_NAMES else "toolkit"})
    return sorted(found, key=lambda item: item["name"].lower())


def tiny_whisper(target: Path, rng) -> list[int]:
    import numpy as np
    from ctranslate2.specs import whisper_spec

    def random(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def norm(spec):
        spec.gamma, spec.beta = np.ones(WIDTH, np.float32), np.zeros(WIDTH, np.float32)

    def linear(spec, rows, cols):
        spec.weight, spec.bias = random(rows, cols), random(rows)

    def attention(spec, fused):
        norm(spec.layer_norm)
        if fused:
            linear(spec.linear[0], 3 * WIDTH, WIDTH)
        else:
            linear(spec.linear[0], WIDTH, WIDTH)
            linear(spec.linear[1], 2 * WIDTH, WIDTH)
        linear(spec.linear[-1], WIDTH, WIDTH)

    def ffn(spec):
        norm(spec.layer_norm)
        linear(spec.linear_0, 4 * WIDTH, WIDTH)
        linear(spec.linear_1, WIDTH, 4 * WIDTH)

    special = ["<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|ja|>", "<|translate|>", "<|transcribe|>",
               "<|startoflm|>", "<|startofprev|>", "<|nocaptions|>", "<|notimestamps|>"]
    tokens = [f"t{index}" for index in range(100)] + special + [f"<|{index * 0.02:.2f}|>" for index in range(18)]
    spec = whisper_spec.WhisperSpec(1, HEADS, 1, HEADS)
    encoder, decoder = spec.encoder, spec.decoder
    encoder.conv1.weight, encoder.conv1.bias = random(WIDTH, MELS, 3), random(WIDTH)
    encoder.conv2.weight, encoder.conv2.bias = random(WIDTH, WIDTH, 3), random(WIDTH)
    encoder.position_encodings.encodings = random(FRAMES // 2, WIDTH)
    norm(encoder.layer_norm)
    for layer in encoder.layer:
        attention(layer.self_attention, fused=True)
        ffn(layer.ffn)
    decoder.embeddings.weight = random(len(tokens), WIDTH)
    decoder.position_encodings.encodings = random(448, WIDTH)
    norm(decoder.layer_norm)
    for layer in decoder.layer:
        attention(layer.self_attention, fused=True)
        attention(layer.attention, fused=False)
        ffn(layer.ffn)
    decoder.projection.weight = random(len(tokens), WIDTH)
    spec.register_vocabulary(tokens)
    index = {token: position for position, token in enumerate(tokens)}
    spec.config.suppress_ids, spec.config.suppress_ids_begin = [], []
    spec.config.lang_ids = [index["<|en|>"], index["<|ja|>"]]
    spec.config.alignment_heads = [(0, head) for head in range(HEADS)]
    spec.validate()
    target.mkdir(parents=True)
    spec.save(str(target))
    return [index["<|startoftranscript|>"], index["<|ja|>"], index["<|transcribe|>"], index["<|notimestamps|>"]]


def check(require_gpu: bool) -> dict:
    report: dict = {"python": sys.version.split()[0], "folder": str(FOLDER), "problems": []}
    problems = report["problems"]
    path = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    report["path_with_cuda_libraries"] = [entry for entry in path if (Path(entry) / CUDA_LIBRARIES[0]).is_file()]
    out = subprocess.run([str(PYTHON), "-I", "-B", "-c", FIND_BY_NAME, *CUDA_LIBRARIES], capture_output=True,
                         text=True, timeout=60).stdout
    found = report["found_by_name"] = json.loads(out)
    for name, where in found.items():
        if where is None:
            problems.append(f"{name} is not found by name (it belongs in {RUNTIME})")
        elif not Path(where).resolve().is_relative_to(FOLDER):
            problems.append(f"{name} is found outside the folder: {where}")

    import ctranslate2
    import numpy as np

    report["ctranslate2"] = ctranslate2.__version__
    devices = ctranslate2.get_cuda_device_count()
    report["cuda_devices"] = devices
    report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices == 0:
        report["runs"] = "skipped: no CUDA device is visible"
        if require_gpu:
            problems.append("no CUDA device is visible (NVIDIA driver missing, or CUDA_VISIBLE_DEVICES hides it)")
    elif any(where is None for where in found.values()):
        report["runs"] = "skipped: the CUDA libraries are not found (a second CUDA call would hang)"
    else:
        report["cuda_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cuda"))
        rng = np.random.default_rng(0)
        scratch = Path(tempfile.mkdtemp(prefix="yata-cuda-check-"))
        runs = report["runs"] = []
        try:
            prompt = tiny_whisper(scratch / "model", rng)
            features = rng.standard_normal((1, MELS, FRAMES)).astype(np.float32)
            cpu = ctranslate2.models.Whisper(str(scratch / "model"), device="cpu", compute_type="float32")
            expected = np.array(cpu.encode(ctranslate2.StorageView.from_array(features), to_cpu=True))
            del cpu
            scale = float(np.abs(expected).max())
            for compute_type in COMPUTE_TYPES:
                run: dict = {"compute_type": compute_type}
                runs.append(run)
                try:
                    started = time.perf_counter()
                    model = ctranslate2.models.Whisper(str(scratch / "model"), device="cuda", compute_type=compute_type)
                    loaded = time.perf_counter()
                    output = np.array(model.encode(ctranslate2.StorageView.from_array(features), to_cpu=True),
                                      dtype=np.float32)
                    encoded = time.perf_counter()
                    result = model.generate(ctranslate2.StorageView.from_array(features), [prompt], max_length=8,
                                            beam_size=1)[0]
                    finished = time.perf_counter()
                    difference = float(np.abs(output - expected).max()) / scale
                    run.update({"load_ms": round((loaded - started) * 1000), "encode_ms": round((encoded - loaded) * 1000),
                                "generate_ms": round((finished - encoded) * 1000), "tokens": len(result.sequences_ids[0]),
                                "max_difference_vs_cpu": round(difference, 5)})
                    # Relative to the largest value: float32 should agree closely, the reduced types loosely.
                    limit = 1e-3 if compute_type == "float32" else 5e-2
                    if not np.isfinite(output).all() or difference > limit:
                        problems.append(f"{compute_type}: CUDA encoder output differs from the CPU's ({difference:.2g} > {limit})")
                    del model
                except Exception as exc:  # noqa: BLE001 - reported, not raised
                    run["error"] = f"{type(exc).__name__}: {exc}"
                    problems.append(f"{compute_type} on CUDA: {run['error']}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    libraries = report["cuda_libraries"] = cuda_libraries()
    for item in libraries:
        if item["kind"] == "toolkit" and not item["inside"]:
            problems.append(f"{item['name']} was loaded from outside the folder: {item['path']}")
    if isinstance(report.get("runs"), list):
        names = {item["name"].lower() for item in libraries if item["inside"]}
        for name in CUDA_LIBRARIES:
            if name.lower() not in names:
                problems.append(f"{name} was not loaded from the folder")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--require-gpu", action="store_true", help="fail when no CUDA device is visible")
    args = parser.parse_args()
    report = check(args.require_gpu)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
