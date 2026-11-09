"""Build a portable Windows folder of Yata that starts without Python or uv on the PC."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import third_party_notices as notices  # noqa: E402  (tools/third_party_notices.py: what the folder may contain)

ROOT = Path(__file__).resolve().parent.parent
CUDA_CHECK = ROOT / "tools" / "portable_cuda_check.py"

# python.org embeddable package. The SHA-256 was taken from the file whose sigstore bundle verified
# against the 3.14 release manager (hugo@python.org, https://github.com/login/oauth). 3.14 is chosen
# because python.org no longer publishes Windows binaries for 3.12 or 3.13 security releases.
PYTHON_VERSION = "3.14.10"
PYTHON_SHA256 = "90905e64a840557d8c1bc4f5b74db3d7ee85e3ffb217611cac1c7813acea4159"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
PYTHON_SIGNER = "Python Software Foundation"
PYTHON_TAG = "python" + "".join(PYTHON_VERSION.split(".")[:2])

EDITIONS = ("cpu", "gpu")
EXTRAS = ("speakers",)  # the only optional extra this prototype can bundle
IMPORT_CHECK = ("fastapi", "uvicorn", "websockets", "numpy", "av", "ctranslate2", "faster_whisper", "onnxruntime")

# CUDA 12 cuBLAS for the GPU edition, from NVIDIA's wheel on PyPI (12.9.2.10 is the newest with a win_amd64
# file). It is pinned here because uv.lock has no CUDA 12 parts (its NVIDIA packages are torch's CUDA 13,
# Linux only). ctranslate2 4.8.5 loads cublas64_12.dll by name, which imports cublasLt64_12.dll. It names
# no cuDNN library (its CUDA Conv1D is its own im2col + cuBLAS), so cuDNN is not bundled; the cuDNN shim
# in its wheel stays unused. tools/portable_cuda_check.py verifies this on the built folder.
CUBLAS_VERSION = "12.9.2.10"
CUBLAS_URL = ("https://files.pythonhosted.org/packages/20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
              f"nvidia_cublas_cu12-{CUBLAS_VERSION}-py3-none-win_amd64.whl")
CUBLAS_SHA256 = "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661"
CUBLAS_FILES = ("nvidia/cublas/bin/cublas64_12.dll", "nvidia/cublas/bin/cublasLt64_12.dll")
CUBLAS_LICENSE = f"nvidia_cublas_cu12-{CUBLAS_VERSION}.dist-info/licenses/License.txt"

# Visual C++ runtime DLLs: not part of Windows, so imports of these must resolve inside the folder.
VC_RUNTIME = frozenset({"msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
                        "msvcp140_codecvt_ids.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll",
                        "vccorlib140.dll", "vcomp140.dll"})
# python.exe ships these two; they are replaced only by a newer copy from the same redist folder.
VC_BASE = ("vcruntime140.dll", "vcruntime140_1.dll")

CRLF = "\r\n"
# ASCII only: cmd.exe reads a batch file in the console code page.
LAUNCHER = CRLF.join([
    "@echo off",
    "rem Yata portable ({edition} edition). Starts Yata on http://127.0.0.1:8765/ (this PC only).",
    "rem Data and models stay in %LOCALAPPDATA%\\Yata, or the model folder chosen on the setup page.",
    "rem Nothing is downloaded at startup; models come only from the setup page.",
    "setlocal",
    'cd /d "%~dp0"',
    # Both editions: ctranslate2 counts an NVIDIA GPU through the driver even without cuBLAS, so without it
    # the setup page would recommend the GPU model and its load test would fail on cublas64_12.dll. Hide
    # the GPU from this process then (as tools/soak.py --hide-gpu does), unless the user chose a device.
    "rem Without CUDA 12 cuBLAS in runtime\\ (CPU edition), NVIDIA GPUs are hidden and Yata runs on the CPU.",
    'if not exist "%~dp0runtime\\cublas64_12.dll" if not defined YATA_ASR_DEVICE if not defined CUDA_VISIBLE_DEVICES '
    'set "CUDA_VISIBLE_DEVICES=-1"',
    'echo Yata: open http://127.0.0.1:8765/ in Chrome. Press Ctrl+C in this window to stop.',
    '"%~dp0runtime\\python.exe" -I -B -m yata',
    'set "code=%errorlevel%"',
    'if not "%code%"=="0" (',
    "  echo.",
    "  echo Yata stopped with an error. If port 8765 is in use, close the other Yata window first.",
    "  pause",
    ")",
    "endlocal & exit /b %code%",
    "",
])

README = """Yata ポータブル版（{edition_ja}）{version}

起動
  Yata.cmd をダブルクリックし、Chrome で http://127.0.0.1:8765/ を開きます。
  初回は「この PC の準備」が開きます。字幕のモデルはそこでボタンを押したときだけ取得します。
  Python や uv の導入は不要です。止めるときは黒いウィンドウで Ctrl+C を押します。

保存場所
  セッションと設定: %LOCALAPPDATA%\\Yata（YATA_DATA_DIR で変更できます）
  モデル: 既定は %LOCALAPPDATA%\\Yata\\models（準備画面で変更できます）
  このフォルダには何も書き込みません。Yata は 127.0.0.1:8765 でだけ待ち受けます。

{edition_note}

更新と削除
  更新: Yata を止めてから、このフォルダを新しい版のフォルダに置き換えます。データとモデルはそのまま使えます。
  削除: このフォルダを削除します。データとモデルも消すときは %LOCALAPPDATA%\\Yata も削除します。

初回の警告
  インターネットから取得した zip を展開すると、Windows の SmartScreen が Yata.cmd の実行前に
  「Windows によって PC が保護されました」と表示することがあります（この版は署名していません）。
  配布元を確かめたうえで「詳細情報」→「実行」を選ぶか、zip のプロパティで「許可する」を選んでから展開します。

版の情報は BUILD-INFO.json にあります。
ライセンス: Yata は LICENSE.txt（MIT）です。同梱した Python・パッケージ・ライブラリとそのライセンスは
THIRD_PARTY_NOTICES.md に一覧があり、各ライセンスの本文は runtime\\Lib\\site-packages の *.dist-info と
licenses フォルダ（パッケージに入っていない分）にあります。
"""
NOTES = {
    "cpu": """CPU 版について
  NVIDIA の GPU があっても CPU で認識します（GPU 用の CUDA ライブラリを同梱していません）。
  GPU で動かすには GPU 版を使うか、CUDA 12 の cuBLAS（CUDA Toolkit 12 に含まれます）を入れたうえで
  環境変数 YATA_ASR_DEVICE=cuda を設定して起動します。""",
    "gpu": """GPU 版について
  NVIDIA の GPU があれば、字幕を GPU（CUDA）で認識します。準備画面は whisper-large-v3-turbo を勧めます。
  GPU 用の CUDA 12 cuBLAS を runtime フォルダに同梱しているので、CUDA Toolkit の導入は不要です
  （NVIDIA のドライバは要ります）。
  NVIDIA の GPU がない PC（AMD や Intel の GPU、GPU なし）では、自動で CPU で動きます。
  準備画面は CPU 用の whisper-small を勧めます。
  cuBLAS の使用条件は runtime\\cuBLAS-LICENSE.txt（NVIDIA）にあります。""",
}
JA_EDITION = {"cpu": "CPU", "gpu": "GPU"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(arg) for arg in args), flush=True)
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def fetch(url: str, expected: str, cache: Path) -> Path:
    target = cache / url.rsplit("/", 1)[1]
    if target.is_file() and sha256(target) == expected:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, 1 << 20)
    actual = sha256(partial)
    if actual != expected:
        partial.unlink()
        raise SystemExit(f"{target.name}: SHA-256 {actual} does not match the pinned {expected}")
    partial.replace(target)
    return target


def check_signatures(paths: list[Path], signer: str) -> None:
    script = ("$ErrorActionPreference='Stop'; Get-AuthenticodeSignature -LiteralPath ($env:YATA_SIGNED -split '\\|') | "
              "ForEach-Object { [pscustomobject]@{ path = $_.Path; status = [string]$_.Status; "
              "subject = [string]$_.SignerCertificate.Subject } } | ConvertTo-Json -Compress")
    env = {**os.environ, "YATA_SIGNED": "|".join(str(path) for path in paths)}
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], env=env,
                         check=True, capture_output=True, text=True).stdout
    results = json.loads(out)
    for item in results if isinstance(results, list) else [results]:
        if item["status"] != "Valid" or signer not in item["subject"]:
            raise SystemExit(f"{item['path']}: signature {item['status']} by {item['subject']!r}, expected {signer}")
        print(f"  signed: {Path(item['path']).name} ({signer})")


def pe_imports(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        return []
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return []
    sections = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    directories = optional + (112 if struct.unpack_from("<H", data, optional)[0] == 0x20B else 96)
    table = []
    for index in range(sections):
        header = optional + optional_size + index * 40
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from("<IIII", data, header + 8)
        table.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def offset(rva: int) -> int | None:
        for address, size, raw in table:
            if address <= rva < address + size:
                return rva - address + raw
        return None

    names = []
    for directory, step, name_field in ((1, 20, 12), (13, 32, 4)):  # import table, delay-load table
        rva = struct.unpack_from("<I", data, directories + directory * 8)[0]
        position = offset(rva) if rva else None
        while position is not None and position + step <= len(data):
            name_rva = struct.unpack_from("<I", data, position + name_field)[0]
            start = offset(name_rva) if name_rva else None
            if start is None:
                break
            names.append(data[start:data.index(b"\0", start)].decode("ascii", "replace").lower())
            position += step
    return names


def file_version(path: Path) -> tuple[int, ...]:
    version = ctypes.windll.version
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return (0, 0, 0, 0)
    buffer = ctypes.create_string_buffer(size)
    version.GetFileVersionInfoW(str(path), 0, size, buffer)
    pointer, length = ctypes.c_void_p(), ctypes.c_uint()
    if not version.VerQueryValueW(buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)):
        return (0, 0, 0, 0)
    high, low = struct.unpack_from("<II", ctypes.string_at(pointer, length.value), 8)
    return (high >> 16, high & 0xFFFF, low >> 16, low & 0xFFFF)


def vc_redist_folders() -> list[Path]:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    roots = []
    if vswhere.is_file():
        out = subprocess.run([str(vswhere), "-all", "-products", "*", "-property", "installationPath"],
                             capture_output=True, text=True).stdout
        roots = [Path(line.strip()) for line in out.splitlines() if line.strip()]
    return [folder for root in roots for folder in root.glob("VC/Redist/MSVC/*/x64/Microsoft.VC14*.CRT")
            if (folder / "msvcp140.dll").is_file()]


def bundle_vc_runtime(folder: Path, runtime: Path, source: Path | None) -> dict[str, str]:
    wanted = set()
    for path in folder.rglob("*"):
        if path.suffix.lower() in {".dll", ".pyd", ".exe"} and path.is_file():
            for name in pe_imports(path):
                if name in VC_RUNTIME and not (path.parent / name).is_file() and not (runtime / name).is_file():
                    wanted.add(name)
    if not wanted:
        return {}
    candidates = [source] if source else vc_redist_folders()
    candidates = [path for path in candidates if path and all((path / name).is_file() for name in wanted)]
    if not candidates:
        raise SystemExit(f"{', '.join(sorted(wanted))} needed but no Visual Studio redist folder has them; "
                         "pass --vc-runtime <...\\VC\\Redist\\MSVC\\<version>\\x64\\Microsoft.VC14x.CRT>")
    chosen = max(candidates, key=lambda path: file_version(path / "msvcp140.dll"))
    copied = {}
    for name in sorted(wanted) + [name for name in VC_BASE if (chosen / name).is_file()]:
        target = runtime / name
        if target.is_file() and file_version(target) >= file_version(chosen / name):
            continue
        shutil.copy2(chosen / name, target)
        copied[name] = ".".join(map(str, file_version(target)))
    print(f"  C++ runtime from {chosen}: {', '.join(f'{name} {ver}' for name, ver in copied.items())}")
    return copied


def missing_vc_imports(folder: Path, runtime: Path) -> list[str]:
    problems = []
    for path in folder.rglob("*"):
        if path.suffix.lower() in {".dll", ".pyd", ".exe"} and path.is_file():
            for name in pe_imports(path):
                if name in VC_RUNTIME and not (path.parent / name).is_file() and not (runtime / name).is_file():
                    problems.append(f"{path.relative_to(folder)} -> {name}")
    return problems


def bundle_cublas(runtime: Path, cache: Path) -> dict[str, object]:
    """Put CUDA 12 cuBLAS from the pinned wheel next to python.exe, where ctranslate2's LoadLibrary looks first."""
    wheel = fetch(CUBLAS_URL, CUBLAS_SHA256, cache)
    files = {}
    with zipfile.ZipFile(wheel) as bundle:
        for member in CUBLAS_FILES:
            target = runtime / PurePosixPath(member).name
            with bundle.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle, 1 << 20)
            files[target.name] = sha256(target)
        (runtime / "cuBLAS-LICENSE.txt").write_bytes(bundle.read(CUBLAS_LICENSE))
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    missing = [f"{name} -> {imported}" for name in files for imported in pe_imports(runtime / name)
               if not imported.startswith("api-ms-win-") and not (runtime / imported).is_file()
               and not (system32 / imported).is_file()]
    if missing:
        raise SystemExit("cuBLAS imports not found in runtime\\ or System32:\n  " + "\n  ".join(missing))
    print(f"  cuBLAS {CUBLAS_VERSION}: {', '.join(f'{name} {(runtime / name).stat().st_size / 2**20:.0f} MiB' for name in files)}")
    return {"package": "nvidia-cublas-cu12", "version": CUBLAS_VERSION, "wheel_sha256": CUBLAS_SHA256, "files": files}


def check_cuda(stage: Path, runtime: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="yata-build-cuda-") as scratch:
        try:
            result = subprocess.run([str(runtime / "python.exe"), "-I", "-B", str(CUDA_CHECK)], cwd=stage,
                                    env=isolated_env(Path(scratch), cpu_only=False), capture_output=True, text=True,
                                    timeout=600)
        except subprocess.TimeoutExpired:
            raise SystemExit("the CUDA check did not finish in 10 minutes") from None
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"the CUDA check failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}") from None

    def shown(path: str | None) -> str:
        if not path:
            return "not found"
        resolved = Path(path).resolve()
        return str(resolved.relative_to(stage.resolve())) if resolved.is_relative_to(stage.resolve()) else path

    for name, where in report["found_by_name"].items():
        print(f"  {name} found by name: {shown(where)}")
    runs = report.get("runs")
    for run in runs if isinstance(runs, list) else []:
        print(f"  {run['compute_type']} on CUDA: " + (run.get("error") or
              f"encode {run['encode_ms']} ms, generate {run['generate_ms']} ms, vs CPU {run['max_difference_vs_cpu']}"))
    for item in report["cuda_libraries"]:
        print(f"  loaded {item['name']} from {shown(item['path'])}")
    if report["problems"]:
        raise SystemExit("CUDA check failed:\n  " + "\n  ".join(report["problems"]))
    if not isinstance(runs, list):
        print(f"  WARNING: {runs}; the CUDA run was not checked on this PC. Run tools/portable_smoke.py on a PC "
              "with an NVIDIA GPU before shipping this folder.")
    return {"cuda_devices": report["cuda_devices"], "ctranslate2": report["ctranslate2"],
            "runs": runs if isinstance(runs, list) else str(runs),
            "loaded": {item["name"]: shown(item["path"]) for item in report["cuda_libraries"]}}


def isolated_env(scratch: Path, *, cpu_only: bool) -> dict[str, str]:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    env = {name: os.environ[name] for name in ("SystemRoot", "SystemDrive", "windir", "COMPUTERNAME",
                                               "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
                                               "ComSpec", "PATHEXT", "ProgramFiles", "ProgramW6432",
                                               "ProgramData", "CommonProgramFiles") if name in os.environ}
    for name in ("LOCALAPPDATA", "APPDATA", "USERPROFILE", "TEMP", "TMP"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
        env[name] = str(scratch / name)
    env["YATA_DATA_DIR"] = str(scratch / "data")
    env["PATH"] = os.pathsep.join([str(Path(system_root) / "System32"), system_root,
                                   str(Path(system_root) / "System32" / "Wbem")])
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def build(args: argparse.Namespace) -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is needed to build (not to run the built folder)")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = tuple(item for item in (part.strip() for part in args.extras.split(",")) if item)
    unknown = [item for item in extras if item not in EXTRAS]
    if unknown:
        raise SystemExit(f"--extras {','.join(unknown)}: this prototype bundles only {', '.join(EXTRAS)} "
                         "(Qwen3-ASR needs a CUDA torch add-on; see docs/development.md)")
    edition = args.edition
    out, cache = Path(args.out).resolve(), Path(args.cache).resolve()
    name = f"Yata-{project['version']}-{edition}"
    final, stage = out / name, out / f".{name}.partial"
    if stage.exists():
        shutil.rmtree(stage)
    runtime, app = stage / "runtime", stage / "app"
    site = runtime / "Lib" / "site-packages"

    print(f"[1/7] Python {PYTHON_VERSION} (embeddable)")
    archive = fetch(PYTHON_URL, PYTHON_SHA256, cache)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(runtime)
    check_signatures([runtime / "python.exe", runtime / f"{PYTHON_TAG}.dll"], PYTHON_SIGNER)
    # The ._pth file fixes sys.path (PYTHONPATH and PYTHONHOME are ignored) and lets site add
    # runtime\Lib\site-packages, as in a normal installation.
    (runtime / f"{PYTHON_TAG}._pth").write_text(
        CRLF.join([f"{PYTHON_TAG}.zip", ".", r"..\app", "import site", ""]), encoding="ascii", newline="")

    print(f"[2/7] Locked packages ({', '.join(('base',) + extras)})")
    requirements = stage / "requirements.txt"
    export = [uv, "export", "--frozen", "--no-dev", "--no-emit-project", "--no-header", "--format", "requirements.txt",
              "--output-file", requirements]
    for extra in extras:
        export += ["--extra", extra]
    run(export, cwd=ROOT, stdout=subprocess.DEVNULL)
    # --python is the bundled interpreter (markers and bytecode match it); copies, not links to uv's cache.
    run([uv, "pip", "install", "--target", site, "--python", runtime / "python.exe", "--require-hashes",
         "--no-deps", "--only-binary", ":all:", "--link-mode", "copy", "--compile-bytecode", "--no-config",
         "--requirements", requirements], cwd=ROOT)
    # Console-script launchers point at this build PC's paths; Yata starts through the .cmd instead.
    for scripts in (site / "bin", site / "Scripts"):
        if scripts.is_dir():
            shutil.rmtree(scripts)
    (runtime / "requirements.lock.txt").write_bytes(requirements.read_bytes())
    requirements.unlink()
    # Files a wheel carries that Yata never loads (cuDNN in ctranslate2); THIRD_PARTY_NOTICES.md says why.
    for package, file_name, _ in notices.REMOVED:
        (site / notices.FOLDERS[package][0] / file_name).unlink(missing_ok=True)
    installed = {dist.metadata["Name"]: dist.version for dist in metadata.distributions(path=[str(site)])}
    problems = (notices.check_bundle(site, installed, notices.NOTICES.read_text(encoding="utf-8"))
                + notices.check_licence_texts(installed))
    if problems:
        raise SystemExit("THIRD_PARTY_NOTICES.md does not describe this build (run tools/third_party_notices.py, "
                         "and describe a new DLL in its NATIVE table):\n  " + "\n  ".join(problems))

    print("[3/7] App (yata, web)")
    app.mkdir()
    shutil.copytree(ROOT / "yata", app / "yata", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "web", app / "web", ignore=shutil.ignore_patterns("__pycache__"))
    run([runtime / "python.exe", "-I", "-m", "compileall", "-q", "-j", "0", app / "yata"], stdout=subprocess.DEVNULL)

    print("[4/7] C++ runtime")
    vc = bundle_vc_runtime(stage, runtime, Path(args.vc_runtime) if args.vc_runtime else None)
    problems = missing_vc_imports(stage, runtime)
    if problems:
        raise SystemExit("C++ runtime imports still missing:\n  " + "\n  ".join(problems))

    print("[5/7] CUDA 12 cuBLAS" + ("" if edition == "gpu" else ": none (CPU edition)"))
    cuda = bundle_cublas(runtime, cache) if edition == "gpu" else None

    print("[6/7] Launcher and notes")
    (stage / "Yata.cmd").write_text(LAUNCHER.format(edition=edition), encoding="ascii", newline="")
    shutil.copy2(ROOT / "LICENSE", stage / "LICENSE.txt")
    shutil.copy2(notices.NOTICES, stage / "THIRD_PARTY_NOTICES.md")
    shutil.copytree(notices.LICENSES, stage / "licenses")
    (stage / "README.txt").write_text(
        README.format(edition_ja=JA_EDITION[edition], version=project["version"], edition_note=NOTES[edition])
        .replace("\n", CRLF), encoding="utf-8-sig", newline="")

    print("[7/7] Checks with the bundled runtime (no Python, uv or CUDA toolkit on PATH, data in a temp folder)")
    modules = IMPORT_CHECK + (("sherpa_onnx",) if "speakers" in extras else ())
    code = ("import sys, importlib; mods = " + repr(modules) + "; [importlib.import_module(m) for m in mods]; "
            "import ctranslate2, yata; print(sys.version.split()[0], 'ctranslate2', ctranslate2.__version__, "
            "'cpu', sorted(ctranslate2.get_supported_compute_types('cpu')), 'cuda devices', ctranslate2.get_cuda_device_count())")
    with tempfile.TemporaryDirectory(prefix="yata-build-check-") as scratch:
        # As the launcher does: the CPU edition hides NVIDIA GPUs, the GPU edition does not.
        run([runtime / "python.exe", "-I", "-B", "-c", code], cwd=stage,
            env=isolated_env(Path(scratch), cpu_only=edition == "cpu"))
    cuda_check = check_cuda(stage, runtime) if edition == "gpu" else None

    packages = {}
    for line in (runtime / "requirements.lock.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", line)
        if match:
            packages[match.group(1)] = match.group(2)
    info = {"name": "Yata", "version": project["version"], "edition": edition, "python": PYTHON_VERSION,
            "extras": list(extras), **git_state(), "uv_lock_sha256": sha256(ROOT / "uv.lock"),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "vc_runtime": vc, "cuda": cuda, "cuda_check": cuda_check, "packages": packages}
    (stage / "BUILD-INFO.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if final.exists():
        if not (final / "BUILD-INFO.json").is_file():
            raise SystemExit(f"{final} exists and is not a Yata build; move it away first")
        shutil.rmtree(final)
    stage.rename(final)
    size = sum(path.stat().st_size for path in final.rglob("*") if path.is_file())
    print(f"Built {final} ({size / 2**20:.0f} MiB)")
    if args.zip:
        target = out / f"{name}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(final.rglob("*")):
                if path.is_file():
                    bundle.write(path, Path(name) / path.relative_to(final))
        print(f"Zipped {target} ({target.stat().st_size / 2**20:.0f} MiB)")
    return final


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("the portable build targets Windows and runs on Windows")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--edition", choices=EDITIONS, default="cpu",
                        help="cpu (default), or gpu: the same plus CUDA 12 cuBLAS for NVIDIA GPUs (CPU elsewhere)")
    parser.add_argument("--out", default=str(ROOT / "dist" / "portable"), help="output folder (default: dist\\portable)")
    parser.add_argument("--cache", default=str(ROOT / "dist" / "cache"), help="download cache (default: dist\\cache)")
    parser.add_argument("--extras", default=",".join(EXTRAS), help="optional extras to bundle (default: speakers; '' for none)")
    parser.add_argument("--vc-runtime", help="a Microsoft.VC14x.CRT redist folder (default: the newest Visual Studio one)")
    parser.add_argument("--zip", action="store_true", help="also write <name>.zip next to the folder")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
