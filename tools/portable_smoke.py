"""Smoke-test a portable build (tools/build_portable.py) the way a fresh PC would start it."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
CUDA_CHECK = Path(__file__).resolve().parent / "portable_cuda_check.py"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
WINDOWS_ENV = ("SystemRoot", "SystemDrive", "windir", "ComSpec", "PATHEXT", "OS", "COMPUTERNAME", "USERNAME",
               "USERDOMAIN", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
               "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432",
               "CommonProgramFiles", "CommonProgramFiles(x86)", "CommonProgramW6432", "ProgramData",
               "ALLUSERSPROFILE", "PUBLIC", "DriverData")


def powershell_json(script: str) -> list[dict]:
    # Get-Net* cmdlets fail (exit 1) when nothing matches; an empty answer means none.
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script + "; exit 0"],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return []
    value = json.loads(out)
    return value if isinstance(value, list) else [value]


def listeners(port: int) -> list[dict]:
    return powershell_json(f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
                           "Select-Object LocalAddress, LocalPort, OwningProcess | ConvertTo-Json -Compress")


def process_sockets(pid: int) -> tuple[list[dict], list[dict]]:
    tcp = powershell_json(f"Get-NetTCPConnection -OwningProcess {pid} -ErrorAction SilentlyContinue | Select-Object "
                          "LocalAddress, LocalPort, RemoteAddress, RemotePort, @{n='State';e={[string]$_.State}} | "
                          "ConvertTo-Json -Compress")
    udp = powershell_json(f"Get-NetUDPEndpoint -OwningProcess {pid} -ErrorAction SilentlyContinue | "
                          "Select-Object LocalAddress, LocalPort | ConvertTo-Json -Compress")
    return tcp, udp


def process_path(pid: int) -> str:
    found = powershell_json(f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object Path | ConvertTo-Json -Compress")
    return str(found[0].get("Path") or "") if found else ""


def get(path: str, cookie: str = "") -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(BASE + path, headers={"Cookie": cookie} if cookie else {})
    try:
        with OPENER.open(request, timeout=30) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()


def fresh_pc_env(scratch: Path, hide_gpu: bool = False) -> dict[str, str]:
    # Windows' own variables stay (NVML, which nvidia-smi uses, fails without ProgramFiles); PATH is
    # Windows only, and nothing else (PYTHON*, VIRTUAL_ENV, UV_*, CUDA_PATH, ...) is passed on. A CUDA
    # toolkit on PATH would let ctranslate2 load its cuBLAS instead of the folder's.
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    env = {name: os.environ[name] for name in WINDOWS_ENV if name in os.environ}
    for name in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "TEMP", "TMP"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
        env[name] = str(scratch / name)
    env["YATA_DATA_DIR"] = str(scratch / "data")
    env["PATH"] = os.pathsep.join([str(Path(system_root) / "System32"), system_root,
                                   str(Path(system_root) / "System32" / "Wbem")])
    # A tripwire: HTTP(S) through urllib/httpx goes to a closed port, except to this PC.
    env.update({"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "127.0.0.1,localhost"})
    if hide_gpu:  # CTranslate2 then sees no CUDA device, as on a PC without an NVIDIA GPU (nvidia-smi still does)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def cuda_check(folder: Path, env: dict[str, str], require_gpu: bool, check) -> None:
    command = [str(folder / "runtime" / "python.exe"), "-I", "-B", str(CUDA_CHECK)] + (["--require-gpu"] if require_gpu else [])
    try:
        result = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, timeout=600)
        report = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        check(False, "CUDA check ran", f"{type(exc).__name__}: {str(getattr(exc, 'stderr', '') or exc)[-300:]}")
        return
    found = report.get("found_by_name", {})
    check(all(where and Path(where).resolve().is_relative_to(folder) for where in found.values()),
          "cuBLAS found by name in the folder", "; ".join(f"{name}: {where}" for name, where in found.items()))
    runs = report.get("runs")
    if isinstance(runs, list):
        for run in runs:
            check("error" not in run, f"tiny Whisper on CUDA in {run['compute_type']}", run.get("error") or
                  f"load {run['load_ms']} ms, encode {run['encode_ms']} ms, generate {run['generate_ms']} ms, "
                  f"vs CPU {run['max_difference_vs_cpu']}")
    else:
        print(f"        CUDA run: {runs}")
    loaded = report.get("cuda_libraries", [])
    print("        CUDA libraries loaded: " + ("; ".join(f"{item['name']} ({'folder' if item['inside'] else item['path']})"
                                                       for item in loaded) or "none"))
    check(not report.get("problems"), "CUDA check found no problem", "; ".join(report.get("problems", []))[:500])


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def smoke(folder: Path, keep: bool, hide_gpu: bool) -> list[str]:
    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(("  ok    " if ok else "  FAIL  ") + label + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    launcher = folder / "Yata.cmd"
    if not launcher.is_file():
        raise SystemExit(f"{launcher} not found; build it with tools/build_portable.py")
    try:
        edition = json.loads((folder / "BUILD-INFO.json").read_text(encoding="utf-8")).get("edition", "cpu")
    except (OSError, ValueError):
        edition = "cpu"
    busy = listeners(PORT)
    if busy:
        raise SystemExit(f"port {PORT} is in use (process {busy[0]['OwningProcess']}); stop that server first, "
                         "this test does not touch it")

    scratch = Path(tempfile.mkdtemp(prefix="yata-portable-smoke-"))
    log_path = scratch / "console.log"
    print(f"Folder:  {folder} ({edition} edition{', NVIDIA GPU hidden' if hide_gpu else ''})\nScratch: {scratch}")
    env = fresh_pc_env(scratch, hide_gpu)
    toolkits = [entry for entry in env["PATH"].split(os.pathsep) if (Path(entry) / "cublas64_12.dll").is_file()]
    check(not toolkits, "no CUDA toolkit on PATH", "; ".join(toolkits))
    expect_cuda = False
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(["cmd.exe", "/d", "/c", str(launcher)], cwd=scratch, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    try:
        server = None
        while time.monotonic() - started < 120 and process.poll() is None:
            found = listeners(PORT)
            if found:
                server = found
                break
            time.sleep(0.5)
        check(server is not None, "server started", f"{time.monotonic() - started:.1f} s" if server else
              f"exit code {process.poll()}")
        if server is None:
            return failures
        pid = int(server[0]["OwningProcess"])
        expected = (folder / "runtime" / "python.exe").resolve()
        actual = process_path(pid)
        check(Path(actual).resolve() == expected if actual else False, "server is the bundled python.exe", actual)

        status, headers, body = get("/")
        ready_at = time.monotonic() - started
        cookie = next((part.split(";", 1)[0] for key, part in headers.items()
                       if key.lower() == "set-cookie" and part.startswith("yata_session=")), "")
        check(status == 200 and b"<html" in body.lower() and bool(cookie), "GET / serves the page and the cookie",
              f"{status}, {len(body)} bytes, first answer after {ready_at:.1f} s")
        for asset in ("/assets/app.js", "/assets/prepare.js", "/assets/style.css", "/assets/logo-36.png",
                      "/assets/favicon-16.png"):
            status, _, body = get(asset)
            check(status == 200 and len(body) > 0, f"GET {asset}", f"{status}, {len(body)} bytes")
        status, _, _ = get("/api/v1/setup")
        check(status == 401, "GET /api/v1/setup without the cookie is refused", str(status))

        status, _, body = get("/api/v1/health", cookie)
        asr = json.loads(body).get("asr", {}) if status == 200 else {}
        check(status == 200, "GET /api/v1/health", f"{status}, ASR: {asr.get('description')} ({asr.get('state')})")
        status, _, body = get("/api/v1/setup", cookie)
        setup = json.loads(body).get("setup", {}) if status == 200 else {}
        check(status == 200 and bool(setup), "GET /api/v1/setup", str(status))
        if setup:
            storage = setup.get("storage", {})
            models_dir = Path(storage.get("path", ""))
            check(models_dir.resolve() == (scratch / "data" / "models").resolve(), "model folder is under the temp data folder",
                  f"{storage.get('path')} ({storage.get('source')})")
            hardware = setup.get("hardware", {})
            gpu = hardware.get("gpu") or {}
            print(f"        hardware: {json.dumps({key: hardware.get(key) for key in ('cpu', 'memory', 'gpu')}, ensure_ascii=False)[:300]}")
            print(f"        recommended: {json.dumps(setup.get('recommended'), ensure_ascii=False)[:300]}")
            # nvidia-smi names the GPU; CTranslate2 decides "cuda". Only the GPU edition can use it.
            expect_cuda = edition == "gpu" and bool(gpu.get("name")) and not hide_gpu
            recommended = (setup.get("recommended") or {}).get("asr")
            if expect_cuda:
                check(gpu.get("cuda") is True, "GPU edition reports cuda: true", json.dumps(gpu, ensure_ascii=False)[:160])
                check(recommended == "whisper-turbo", "whisper-turbo is recommended", str(recommended))
            else:
                why = ("CPU edition" if edition == "cpu" else "GPU edition, NVIDIA GPU hidden" if hide_gpu and gpu.get("name")
                       else "GPU edition on a PC without an NVIDIA GPU")
                check(not gpu.get("cuda"), f"no CUDA ({why})", json.dumps(gpu, ensure_ascii=False)[:160])
                check(recommended == "whisper-small", "whisper-small (CPU) is recommended", str(recommended))
            asr = setup.get("asr", {}).get("models", [])
            check(bool(asr) and not any(item.get("state") == "ready" for item in asr), "no ASR model counts as ready",
                  ", ".join(f"{item.get('id')}={item.get('state')}" for item in asr))
            check(setup.get("vad", {}).get("state") == "ready", "Silero VAD found in the bundle", setup.get("vad", {}).get("state", ""))
            speakers = setup.get("speakers", {})
            print(f"        speakers package: {speakers.get('package')}; llm: "
                  f"{json.dumps(setup.get('llm', {}).get('ollama'), ensure_ascii=False)[:200]}")

        tcp, udp = process_sockets(pid)
        listening = sorted({(item["LocalAddress"], int(item["LocalPort"])) for item in tcp if item["State"] == "Listen"})
        check(listening == [("127.0.0.1", PORT)], "listens only on 127.0.0.1:8765", str(listening))
        remote = [item for item in tcp if item["State"] != "Listen"
                  and item["RemoteAddress"] not in ("127.0.0.1", "::1", "0.0.0.0", "::")]
        check(not remote, "no connection off this PC", str(remote))
        check(not udp, "no UDP sockets", str(udp))
        if edition == "gpu":
            print("        CUDA check (tools/portable_cuda_check.py, same environment):")
            cuda_check(folder, env, expect_cuda, check)
    finally:
        stop(process)
        time.sleep(0.5)
        written = sorted(str(path.relative_to(scratch)) for path in (scratch / "data").rglob("*")) if (scratch / "data").exists() else []
        models = [name for name in written if name.startswith(os.path.join("data", "models") + os.sep)]
        check(not models, "nothing written under the model folder", ", ".join(models))
        print(f"        data folder after the run: {written or 'empty'}")
        console = log_path.read_bytes().decode("utf-8", "replace") if log_path.exists() else ""
        print("        console:\n" + "\n".join("          " + line for line in console.strip().splitlines()[-8:]))
        check(not listeners(PORT), "server stopped")
        if keep:
            print(f"Kept {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)
    return failures


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("the portable build runs on Windows")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="the built folder (dist\\portable\\Yata-<version>-cpu or -gpu)")
    parser.add_argument("--keep", action="store_true", help="keep the temp folder (console log, data) afterwards")
    parser.add_argument("--hide-gpu", action="store_true",
                        help="hide NVIDIA GPUs from CTranslate2, as on a PC without one (the GPU edition must fall back to the CPU)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = smoke(Path(args.folder).resolve(), args.keep, args.hide_gpu)
    print("PASS" if not failures else f"FAIL ({len(failures)}): " + "; ".join(failures))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
