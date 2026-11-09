"""Write or check THIRD_PARTY_NOTICES.md: what the portable Windows build puts next to Yata, and under which licences."""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

from packaging.markers import Marker
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
LOCK = ROOT / "uv.lock"
LICENSES = ROOT / "licenses"
# Kept equal to tools/build_portable.py (PYTHON_VERSION, EXTRAS, CUBLAS_VERSION); a test compares them.
PYTHON_VERSION = "3.14.10"
EXTRAS = ("speakers",)
CUBLAS_VERSION = "12.9.2.10"
TARGET = {
    "os_name": "nt", "sys_platform": "win32", "platform_system": "Windows", "platform_machine": "AMD64",
    "platform_release": "", "platform_version": "", "implementation_name": "cpython",
    "platform_python_implementation": "CPython", "implementation_version": PYTHON_VERSION,
    "python_version": ".".join(PYTHON_VERSION.split(".")[:2]), "python_full_version": PYTHON_VERSION, "extra": "",
}
REVIEW = "**Check before release**"

NATIVE: tuple[tuple[str, str, str, str, bool], ...] = (
    ("av", r"^(avcodec|avdevice|avfilter|avformat|avutil|swresample|swscale)-", "FFmpeg (libavcodec, libavformat, libavutil, libavfilter, libavdevice, libswresample, libswscale)",
     "LGPL-3.0-or-later as configured (--enable-version3), but linked with x264 and x265: PyAV states the wheels as \"LGPL-2.1-or-later + commercial exceptions\" OR \"GPL-2.0-or-later\"", True),
    ("av", r"^libx264-", "x264", "GPL-2.0-or-later", True),
    ("av", r"^libx265-", "x265", "GPL-2.0-or-later", True),
    ("av", r"^libSvtAv1Enc-", "SVT-AV1", "BSD-3-Clause-Clear with the Alliance for Open Media Patent License 1.0", False),
    ("av", r"^libvmaf-", "VMAF", "BSD-2-Clause-Patent", False),
    ("av", r"^libdav1d-", "dav1d", "BSD-2-Clause", False),
    ("av", r"^libvpx-", "libvpx", "BSD-3-Clause", False),
    ("av", r"^libopus-", "Opus", "BSD-3-Clause", False),
    ("av", r"^lib(webp|webpmux|sharpyuv)-", "libwebp", "BSD-3-Clause", False),
    ("av", r"^libvpl-", "oneVPL dispatcher", "MIT", False),
    ("av", r"^libiconv-", "GNU libiconv (the library)", "LGPL-2.1 (the iconv program, not bundled, is GPL)", True),
    ("av", r"^zlib1-", "zlib", "Zlib", False),
    ("av", r"^lib(gcc_s_seh|stdc\+\+)-", "GCC runtime (libgcc, libstdc++)", "GPL-3.0-or-later WITH GCC-exception-3.1", False),
    ("av", r"^libwinpthread-", "mingw-w64 winpthreads", "MIT (mingw-w64 project), with parts under a BSD-3-Clause-style licence (Lockless Inc.)", False),
    ("ctranslate2", r"^ctranslate2\.dll$", "CTranslate2, with Intel oneMKL 2025.3, oneDNN 3.1.1, cpu_features, spdlog and (in its CUDA kernels) Thrust and CUB built in",
     "MIT; oneMKL: Intel Simplified Software License; oneDNN, cpu_features: Apache-2.0; spdlog: MIT; Thrust, CUB: Apache-2.0, BSD-3-Clause. Texts in `licenses\\ctranslate2\\`", False),
    ("ctranslate2", r"^libiomp5md\.dll$", "Intel OpenMP runtime 2025.3 (the same file as in Intel's intel-openmp 2025.3.0 package)",
     "Intel End User License Agreement for Developer Tools (August 2024), text in `licenses\\ctranslate2\\`", True),
    ("numpy", r"^libscipy_openblas64_-", "OpenBLAS with LAPACK and the GCC runtime (listed in numpy's licence file)",
     "BSD-3-Clause; GCC runtime GPL-3.0-or-later WITH GCC-exception-3.1", False),
    ("numpy", r"^msvcp140-", "Microsoft C++ runtime (numpy's own copy)", "Microsoft Visual C++ Redistributable terms", True),
    ("onnxruntime", r"^onnxruntime(_providers_shared)?\.dll$", "ONNX Runtime", "MIT (the package's ThirdPartyNotices.txt lists what it includes)", False),
    ("sherpa-onnx", r"^onnxruntime\.dll$", "ONNX Runtime", "MIT", False),
    ("sherpa-onnx", r"^sherpa-onnx-c(xx)?-api\.dll$", "sherpa-onnx", "Apache-2.0", False),
)
FOLDERS = {"av": ("av", "av.libs"), "ctranslate2": ("ctranslate2",), "numpy": ("numpy", "numpy.libs"),
           "onnxruntime": ("onnxruntime",), "sherpa-onnx": ("sherpa_onnx",)}
# Files a wheel carries that Yata never loads; the build deletes them (package, file in its first folder, why).
# They have no NATIVE entry, so one left behind stops the build.
REMOVED: tuple[tuple[str, str, str], ...] = (
    ("ctranslate2", "cudnn64_9.dll",
     "NVIDIA cuDNN 9.10.2. CTranslate2 builds its Windows wheel without cuDNN (`-DWITH_CUDNN=OFF`), so `ctranslate2.dll` "
     "never loads it (Whisper on CUDA ran without it and loaded only cuBLAS), and cuDNN's licence covers applications on "
     "systems with NVIDIA GPUs, which the CPU edition is not limited to"),
)
# Licence texts a wheel leaves out, kept in licenses/<package>/ and copied to the folder's licenses\<package>\:
# package -> (the version they were collected for, ((file, what it covers, where it came from), ...)). A test and
# the build fail when uv.lock moves the package to another version, so the texts are collected again.
LICENSE_TEXTS: dict[str, tuple[str, tuple[tuple[str, str, str], ...]]] = {
    "ctranslate2": ("4.8.5", (
        ("CTranslate2-LICENSE.txt", "CTranslate2 (MIT)", "https://github.com/OpenNMT/CTranslate2/blob/v4.8.5/LICENSE"),
        ("oneMKL-LICENSE.txt", "Intel oneMKL 2025.3, linked into `ctranslate2.dll` (Intel Simplified Software License)",
         "`onemkl_license-2025.3.0` wheel, <https://pypi.org/project/onemkl-license/2025.3.0/>"),
        ("oneMKL-third-party-programs.txt", "what oneMKL includes", "the same wheel"),
        ("oneDNN-LICENSE.txt", "oneDNN 3.1.1, linked into `ctranslate2.dll` (Apache-2.0)",
         "https://github.com/oneapi-src/oneDNN/blob/v3.1.1/LICENSE"),
        ("oneDNN-THIRD-PARTY-PROGRAMS.txt", "what oneDNN includes (Xbyak, ITT API and others)",
         "https://github.com/oneapi-src/oneDNN/blob/v3.1.1/THIRD-PARTY-PROGRAMS"),
        ("Intel-OpenMP-EULA.txt", "`libiomp5md.dll` (Intel End User License Agreement for Developer Tools, August 2024)",
         "`intel_openmp-2025.3.0` wheel, <https://pypi.org/project/intel-openmp/2025.3.0/>, whose `libiomp5md.dll` is the same file"),
        ("Intel-OpenMP-third-party-programs.txt", "what the OpenMP runtime includes", "the same wheel"),
        ("cpu_features-LICENSE.txt", "cpu_features (Apache-2.0), linked into `ctranslate2.dll` on x86-64",
         "https://github.com/google/cpu_features/blob/8a494eb1e158ec2050e5f699a504fbc9b896a43b/LICENSE"),
        ("spdlog-LICENSE.txt", "spdlog (MIT), built into `ctranslate2.dll`",
         "https://github.com/gabime/spdlog/blob/76fb40d95455f249bd70824ecfcae7a8f0930fa3/LICENSE"),
        ("CCCL-LICENSE.txt", "Thrust, CUB and libcu++ (NVIDIA CCCL), used by the CUDA kernels in `ctranslate2.dll`",
         "https://github.com/NVIDIA/cccl/blob/b5fe509fd11a925f90d6495176707cc1184eed9d/LICENSE"),
    )),
}

RUNTIMES = f"""## Runtimes the build adds

| Component | Where in the folder | Licence | |
|---|---|---|---|
| CPython {PYTHON_VERSION} (python.org embeddable package) | `runtime\\` | PSF-2.0; `runtime\\LICENSE.txt` also covers what Python bundles (OpenSSL, libffi, bzip2, expat, SQLite, zlib, mpdecimal and others) | |
| Microsoft Visual C++ runtime (`msvcp140.dll` and the DLLs it needs, copied app-local from a Visual Studio redist folder) | `runtime\\` | Distributable Code under the Visual Studio licence (the REDIST list) | {REVIEW} |
| NVIDIA cuBLAS {CUBLAS_VERSION} (`cublas64_12.dll`, `cublasLt64_12.dll`; GPU edition only) | `runtime\\`, with `runtime\\cuBLAS-LICENSE.txt` | CUDA EULA; cuBLAS is listed in its redistributable files (Attachment A) | {REVIEW} |
"""

CHECKS = f"""## {REVIEW}

These are the points still open before the portable ZIP is published, as checked against each project's own files on
2026-11-05 (sources below). They do not affect running from source, where uv installs the same packages from PyPI on
the user's PC.

1. **PyAV's FFmpeg links GPL code (x264, x265).** faster-whisper (MIT) bundles no FFmpeg itself; its README only says
   that PyAV bundles the FFmpeg libraries, and names no licence. PyAV's own licence (BSD-3-Clause) covers its source,
   not the wheels. Its FFmpeg build (pyav-ffmpeg) patches FFmpeg's `configure` to move libx264 and libx265 from the
   GPL list to the "version3" list and builds with `--enable-version3`, not `--enable-gpl`, so FFmpeg reports LGPLv3;
   x264 and x265 remain GPL-2.0-or-later. PyAV's maintainer states the wheels' licence as "LGPLv2.1 or later +
   Commercial exceptions" OR "GPLv2 or later", the commercial route being licences from the x264 and x265 owners,
   negotiated case by case. For the ZIP that leaves: meet GPL-2.0-or-later for these parts (the licence texts and the
   corresponding source of FFmpeg, x264, x265 and pyav-ffmpeg's build scripts and patches), obtain x264/x265
   commercial licences, or build the folder against an FFmpeg without GPL parts (Yata only decodes audio).
2. **The licence texts of the libraries inside PyAV.** The `av` wheel carries only PyAV's own licence; the libraries in
   `av.libs` (the native table below) come without theirs, and most of them ask for their notice or licence text to
   go with a binary. GNU libiconv's LGPL-2.1 also asks for a way to get its source. Which libraries stay depends on
   item 1, so collect their texts into `licenses\\av\\` once it is decided.
3. **Intel's OpenMP runtime (`libiomp5md.dll`, inside ctranslate2).** Intel ships this file under its End User License
   Agreement for Developer Tools (August 2024), not under oneMKL's simplified licence. The agreement lets the files
   Intel lists as redistributable be passed on only as part of a product, in binary form, and under a licence
   agreement that prohibits reverse engineering, decompiling or disassembling them; the product may not be marketed
   with Intel's names, and its distributor answers for it to its users and to Intel. Yata's MIT licence has no such
   term, so the ZIP's terms need a line covering this DLL. Also confirm that Intel's redistributable list for oneAPI
   2025.3 names `libiomp5md.dll` (Intel publishes it on its own as the `intel-openmp` package).
4. **Microsoft Visual C++ runtime.** Redistribution is allowed to licensed Visual Studio users as "Distributable Code"
   under the Visual Studio licence (the REDIST list). Microsoft recommends installing the Visual C++ Redistributable
   package over copying the DLLs next to the application (app-local), because app-local copies are not serviced by
   Windows Update. The build copies them app-local.

Settled in the build:

- NVIDIA cuDNN (`cudnn64_9.dll`) is deleted from the folder (see "Removed by the build").
- CTranslate2's licence, the licences of the code built into `ctranslate2.dll` (oneMKL, oneDNN, cpu_features, spdlog,
  Thrust, CUB) and the OpenMP runtime's agreement go to `licenses\\ctranslate2\\` (see "Licence texts the build adds").
- NVIDIA cuBLAS (GPU edition): the CUDA EULA lists the cuBLAS DLLs (`cublas.dll`, `cublasLt.dll` and their versioned
  names) as redistributable inside an application that adds material functionality, used only by that application,
  under terms that protect its users. Its licence travels in `runtime\\cuBLAS-LICENSE.txt`.

Sources: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[PyAV issue #2270](https://github.com/PyAV-Org/PyAV/issues/2270),
[pyav-ffmpeg build-ffmpeg.py](https://github.com/PyAV-Org/pyav-ffmpeg/blob/master/scripts/build-ffmpeg.py) and
[ffmpeg.patch](https://github.com/PyAV-Org/pyav-ffmpeg/blob/master/patches/ffmpeg.patch),
[CTranslate2 4.8.5 Windows wheel build](https://github.com/OpenNMT/CTranslate2/blob/v4.8.5/python/tools/prepare_build_environment_windows.sh)
and [CMakeLists.txt](https://github.com/OpenNMT/CTranslate2/blob/v4.8.5/CMakeLists.txt),
[cuDNN licence](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html),
[CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html),
[intel-openmp 2025.3.0](https://pypi.org/project/intel-openmp/2025.3.0/) and
[onemkl-license 2025.3.0](https://pypi.org/project/onemkl-license/2025.3.0/) on PyPI (Intel's own licence files),
[Redistributing Visual C++ files](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files),
[SVT-AV1 LICENSE.md](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/LICENSE.md),
[winpthreads COPYING](https://github.com/mingw-w64/mingw-w64/blob/master/mingw-w64-libraries/winpthreads/COPYING),
[libiconv README](https://git.savannah.gnu.org/cgit/libiconv.git/tree/README).
"""


def bundled_packages(lock_path: Path = LOCK, extras: tuple[str, ...] = EXTRAS, target: dict[str, str] = TARGET) -> dict[str, str]:
    packages = {canonicalize_name(item["name"]): item for item in tomllib.loads(lock_path.read_text(encoding="utf-8"))["package"]}
    root = next(item for item in packages.values() if item.get("source", {}).get("virtual") == ".")

    def holds(entry: dict, extra: str = "") -> bool:
        marker = entry.get("marker")
        return marker is None or Marker(marker).evaluate({**target, "extra": extra})

    wanted = [entry for entry in root.get("dependencies", []) if holds(entry)]
    for extra in extras:
        wanted += [entry for entry in root.get("optional-dependencies", {}).get(extra, []) if holds(entry, extra)]
    found: dict[str, str] = {}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    while wanted:
        entry = wanted.pop()
        name = canonicalize_name(entry["name"])
        extras_of = tuple(sorted(entry.get("extra", [])))
        if (name, extras_of) in seen:
            continue
        seen.add((name, extras_of))
        package = packages[name]
        found[name] = package["version"]
        wanted += [dep for dep in package.get("dependencies", []) if holds(dep)]
        for extra in extras_of:
            wanted += [dep for dep in package.get("optional-dependencies", {}).get(extra, []) if holds(dep, extra)]
    return dict(sorted(found.items()))


def listed_packages(text: str) -> dict[str, str]:
    section = text.split("## Python packages", 1)[1].split("\n## ", 1)[0] if "## Python packages" in text else ""
    rows = re.findall(r"^\| `([^`]+)` \| ([^ |]+) \|", section, re.MULTILINE)
    return {canonicalize_name(name): version for name, version in rows}


def native_entry(package: str, file_name: str) -> tuple[str, str, str, str, bool] | None:
    return next((item for item in NATIVE if item[0] == package and re.search(item[1], file_name)), None)


def check_bundle(site: Path, installed: dict[str, str], notices: str) -> list[str]:
    listed = listed_packages(notices)
    problems = [f"{name} {version} is bundled but THIRD_PARTY_NOTICES.md lists {listed.get(name, 'nothing')}"
                for name, version in sorted((canonicalize_name(key), value) for key, value in installed.items())
                if listed.get(name) != version]
    owners = {folder.lower(): package for package, folders in FOLDERS.items() for folder in folders}
    for dll in sorted(site.rglob("*.dll")):
        owner = owners.get(dll.relative_to(site).parts[0].lower())
        if owner is None or native_entry(owner, dll.name) is None:
            problems.append(f"{dll.relative_to(site)} is not described in THIRD_PARTY_NOTICES.md (tools/third_party_notices.py NATIVE)")
    return problems


def check_licence_texts(installed: dict[str, str], licenses: Path = LICENSES) -> list[str]:
    versions = {canonicalize_name(key): value for key, value in installed.items()}
    problems = []
    for package, (version, files) in LICENSE_TEXTS.items():
        if versions.get(package) != version:
            problems.append(f"licenses/{package}/ was collected for {package} {version}, but {package} "
                            f"{versions.get(package, '(none)')} is bundled: collect its licence texts again")
        present = {path.name for path in (licenses / package).glob("*")} if (licenses / package).is_dir() else set()
        listed = {name for name, _, _ in files}
        problems += [f"licenses/{package}/{name} is missing" for name in sorted(listed - present)]
        problems += [f"licenses/{package}/{name} is not in LICENSE_TEXTS" for name in sorted(present - listed)]
    present = {path.name for path in licenses.iterdir()} if licenses.is_dir() else set()
    problems += [f"licenses/{name} is not in LICENSE_TEXTS" for name in sorted(present - set(LICENSE_TEXTS))]
    return problems


def _pypi(name: str, version: str) -> dict:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
        return json.load(response)["info"]


def _licence(info: dict) -> str:
    expression = (info.get("license_expression") or "").strip()
    if expression:
        return expression
    text = (info.get("license") or "").strip()
    if text and "\n" not in text and len(text) <= 60:
        return text
    classifiers = [item.split(" :: ")[-1] for item in info.get("classifiers") or [] if item.startswith("License :: ")]
    return " / ".join(classifiers) if classifiers else "see the package's licence file"


def _source(name: str, info: dict) -> str:
    urls = {key.lower(): value for key, value in (info.get("project_urls") or {}).items()}
    for key in ("source", "source code", "repository", "homepage", "home", "github"):
        if urls.get(key):
            return urls[key]
    return info.get("home_page") or f"https://pypi.org/project/{name}/"


def render(packages: dict[str, str], metadata: dict[str, dict]) -> str:
    rows = "\n".join(f"| `{name}` | {version} | {_licence(metadata[name]).replace('|', '/')} | <{_source(name, metadata[name])}> |"
                     for name, version in packages.items())
    native = "\n".join(f"| `{package}` | `{pattern.strip('^$')}` | {component} | {licence} | {REVIEW if review else ''} |"
                       for package, pattern, component, licence, review in NATIVE)
    removed = "\n".join(f"| `{package}` | `{file_name}` | {why} |" for package, file_name, why in REMOVED)
    texts = "\n".join(f"| `licenses\\{package}\\{name}` | {covers} | {f'<{source}>' if source.startswith('https://') else source} |"
                      for package, (_, files) in LICENSE_TEXTS.items() for name, covers, source in files)
    collected = ", ".join(f"{package} {version}" for package, (version, _) in LICENSE_TEXTS.items())
    return f"""# Third-party notices

Yata itself is under the MIT License ([LICENSE](LICENSE)). This file lists what the **portable Windows build**
(`tools/build_portable.py`, see [docs/development.md](docs/development.md#ポータブル版のビルド)) puts next to it, so it can be read
before a build exists. Running from source bundles nothing: uv installs the same packages from PyPI on the user's
PC, each under its own licence. The models Yata fetches on the setup page are not bundled either; the page shows
each one's licence before the download.

The package list is generated from `uv.lock` by `tools/third_party_notices.py` for Windows x64, CPython
{PYTHON_VERSION}, with the `{", ".join(EXTRAS)}` extra. A test keeps it equal to `uv.lock`, and the build refuses to
run when it installs a package, a version or a DLL this file does not describe. In the folder, each package's licence
texts stay in `runtime\\Lib\\site-packages\\<package>.dist-info\\licenses\\` (and ONNX Runtime's
`ThirdPartyNotices.txt` in its package folder), the texts a wheel leaves out are added in `licenses\\`, and this file
is copied next to `Yata.cmd`.

{CHECKS}
## Python packages

| Package | Version | Licence (as declared) | Source |
|---|---|---|---|
{rows}

## Native libraries inside the packages

| Package | Files | Component | Licence | |
|---|---|---|---|---|
{native}

## Removed by the build

| Package | File | Why |
|---|---|---|
{removed}

## Licence texts the build adds

The wheels below carry no licence files for what they contain. The texts are kept in this repository's `licenses/`
folder (collected for {collected}; a test fails when `uv.lock` moves to another version) and copied to the folder's
`licenses\\`.

| In the folder | Covers | Source |
|---|---|---|
{texts}

{RUNTIMES}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when THIRD_PARTY_NOTICES.md does not match uv.lock")
    args = parser.parse_args()
    packages = bundled_packages()
    if args.check:
        listed = listed_packages(NOTICES.read_text(encoding="utf-8")) if NOTICES.is_file() else {}
        if listed != packages:
            changed = sorted(set(listed.items()) ^ set(packages.items()))
            raise SystemExit("THIRD_PARTY_NOTICES.md does not match uv.lock; run tools/third_party_notices.py\n  "
                             + "\n  ".join(f"{name} {version}" for name, version in changed))
        print(f"THIRD_PARTY_NOTICES.md matches uv.lock ({len(packages)} packages)")
        return
    metadata = {name: _pypi(name, version) for name, version in packages.items()}
    NOTICES.write_text(render(packages, metadata), encoding="utf-8", newline="\n")
    print(f"Wrote {NOTICES.relative_to(ROOT)} ({len(packages)} packages)")


if __name__ == "__main__":
    sys.exit(main())
