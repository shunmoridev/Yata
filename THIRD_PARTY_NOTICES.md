# Third-party notices

Yata itself is under the MIT License ([LICENSE](LICENSE)). This file lists what the **portable Windows build**
(`tools/build_portable.py`, see [docs/development.md](docs/development.md#ポータブル版のビルド)) puts next to it, so it can be read
before a build exists. Running from source bundles nothing: uv installs the same packages from PyPI on the user's
PC, each under its own licence. The models Yata fetches on the setup page are not bundled either; the page shows
each one's licence before the download.

The package list is generated from `uv.lock` by `tools/third_party_notices.py` for Windows x64, CPython
3.14.10, with the `speakers` extra. A test keeps it equal to `uv.lock`, and the build refuses to
run when it installs a package, a version or a DLL this file does not describe. In the folder, each package's licence
texts stay in `runtime\Lib\site-packages\<package>.dist-info\licenses\` (and ONNX Runtime's
`ThirdPartyNotices.txt` in its package folder), the texts a wheel leaves out are added in `licenses\`, and this file
is copied next to `Yata.cmd`.

## **Check before release**

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
   item 1, so collect their texts into `licenses\av\` once it is decided.
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
  Thrust, CUB) and the OpenMP runtime's agreement go to `licenses\ctranslate2\` (see "Licence texts the build adds").
- NVIDIA cuBLAS (GPU edition): the CUDA EULA lists the cuBLAS DLLs (`cublas.dll`, `cublasLt.dll` and their versioned
  names) as redistributable inside an application that adds material functionality, used only by that application,
  under terms that protect its users. Its licence travels in `runtime\cuBLAS-LICENSE.txt`.

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

## Python packages

| Package | Version | Licence (as declared) | Source |
|---|---|---|---|
| `annotated-doc` | 0.0.5 | MIT | <https://github.com/fastapi/annotated-doc> |
| `annotated-types` | 0.8.0 | MIT | <https://github.com/annotated-types/annotated-types> |
| `anyio` | 4.16.0 | MIT | <https://github.com/agronholm/anyio> |
| `av` | 19.0.2 | BSD-3-Clause | <https://github.com/PyAV-Org/PyAV> |
| `click` | 8.5.0 | BSD-3-Clause | <https://github.com/pallets/click/> |
| `colorama` | 0.4.6 | BSD License | <https://github.com/tartley/colorama> |
| `ctranslate2` | 4.8.5 | MIT | <https://github.com/OpenNMT/CTranslate2> |
| `fastapi` | 0.162.0 | MIT | <https://github.com/fastapi/fastapi> |
| `faster-whisper` | 1.2.1 | MIT | <https://github.com/SYSTRAN/faster-whisper> |
| `filelock` | 4.2.7 | MIT | <https://github.com/tox-dev/filelock> |
| `flatbuffers` | 25.12.19 | Apache 2.0 | <https://github.com/google/flatbuffers> |
| `fsspec` | 2026.9.0 | BSD-3-Clause | <https://github.com/fsspec/filesystem_spec> |
| `h11` | 0.16.0 | MIT | <https://github.com/python-hyper/h11> |
| `hf-xet` | 1.8.0 | Apache-2.0 | <https://github.com/huggingface/xet-core.git> |
| `httpcore2` | 2.16.0 | BSD-3-Clause | <https://github.com/pydantic/httpx2/blob/main/src/httpcore2> |
| `httpx2` | 2.16.0 | BSD-3-Clause | <https://github.com/pydantic/httpx2> |
| `huggingface-hub` | 2.4.0 | Apache-2.0 | <https://github.com/huggingface/huggingface_hub> |
| `idna` | 3.20 | BSD-3-Clause | <https://github.com/kjd/idna> |
| `numpy` | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | <https://github.com/numpy/numpy> |
| `onnxruntime` | 1.31.1 | MIT License | <https://onnxruntime.ai> |
| `opentelemetry-api` | 1.45.2 | Apache-2.0 | <https://github.com/open-telemetry/opentelemetry-python> |
| `packaging` | 26.3 | Apache-2.0 OR BSD-2-Clause | <https://github.com/pypa/packaging> |
| `protobuf` | 7.36.2 | 3-Clause BSD License | <https://developers.google.com/protocol-buffers/> |
| `pydantic` | 2.14.2 | MIT | <https://github.com/pydantic/pydantic> |
| `pydantic-core` | 2.50.2 | MIT | <https://github.com/pydantic/pydantic/tree/main/pydantic-core> |
| `pyyaml` | 6.0.3 | MIT | <https://github.com/yaml/pyyaml> |
| `sherpa-onnx` | 2.0.1 | Apache licensed, as found in the LICENSE file | <https://github.com/k2-fsa/sherpa-onnx> |
| `sherpa-onnx-core` | 2.0.1 | Apache-2.0 | <https://github.com/k2-fsa/sherpa-onnx> |
| `starlette` | 1.10.0 | BSD-3-Clause | <https://github.com/Kludex/starlette> |
| `tokenizers` | 0.23.3 | Apache Software License | <https://github.com/huggingface/tokenizers> |
| `tqdm` | 4.70.1 | MPL-2.0 AND MIT | <https://github.com/tqdm/tqdm> |
| `truststore` | 0.10.5 | MIT | <https://github.com/sethmlarson/truststore> |
| `typing-extensions` | 4.16.0 | PSF-2.0 | <https://github.com/python/typing_extensions> |
| `typing-inspection` | 0.4.4 | MIT | <https://github.com/pydantic/typing-inspection> |
| `uvicorn` | 0.54.0 | BSD-3-Clause | <https://github.com/Kludex/uvicorn> |
| `websockets` | 17.4 | BSD-3-Clause | <https://github.com/python-websockets/websockets> |

## Native libraries inside the packages

| Package | Files | Component | Licence | |
|---|---|---|---|---|
| `av` | `(avcodec|avdevice|avfilter|avformat|avutil|swresample|swscale)-` | FFmpeg (libavcodec, libavformat, libavutil, libavfilter, libavdevice, libswresample, libswscale) | LGPL-3.0-or-later as configured (--enable-version3), but linked with x264 and x265: PyAV states the wheels as "LGPL-2.1-or-later + commercial exceptions" OR "GPL-2.0-or-later" | **Check before release** |
| `av` | `libx264-` | x264 | GPL-2.0-or-later | **Check before release** |
| `av` | `libx265-` | x265 | GPL-2.0-or-later | **Check before release** |
| `av` | `libSvtAv1Enc-` | SVT-AV1 | BSD-3-Clause-Clear with the Alliance for Open Media Patent License 1.0 |  |
| `av` | `libvmaf-` | VMAF | BSD-2-Clause-Patent |  |
| `av` | `libdav1d-` | dav1d | BSD-2-Clause |  |
| `av` | `libvpx-` | libvpx | BSD-3-Clause |  |
| `av` | `libopus-` | Opus | BSD-3-Clause |  |
| `av` | `lib(webp|webpmux|sharpyuv)-` | libwebp | BSD-3-Clause |  |
| `av` | `libvpl-` | oneVPL dispatcher | MIT |  |
| `av` | `libiconv-` | GNU libiconv (the library) | LGPL-2.1 (the iconv program, not bundled, is GPL) | **Check before release** |
| `av` | `zlib1-` | zlib | Zlib |  |
| `av` | `lib(gcc_s_seh|stdc\+\+)-` | GCC runtime (libgcc, libstdc++) | GPL-3.0-or-later WITH GCC-exception-3.1 |  |
| `av` | `libwinpthread-` | mingw-w64 winpthreads | MIT (mingw-w64 project), with parts under a BSD-3-Clause-style licence (Lockless Inc.) |  |
| `ctranslate2` | `ctranslate2\.dll` | CTranslate2, with Intel oneMKL 2025.3, oneDNN 3.1.1, cpu_features, spdlog and (in its CUDA kernels) Thrust and CUB built in | MIT; oneMKL: Intel Simplified Software License; oneDNN, cpu_features: Apache-2.0; spdlog: MIT; Thrust, CUB: Apache-2.0, BSD-3-Clause. Texts in `licenses\ctranslate2\` |  |
| `ctranslate2` | `libiomp5md\.dll` | Intel OpenMP runtime 2025.3 (the same file as in Intel's intel-openmp 2025.3.0 package) | Intel End User License Agreement for Developer Tools (August 2024), text in `licenses\ctranslate2\` | **Check before release** |
| `numpy` | `libscipy_openblas64_-` | OpenBLAS with LAPACK and the GCC runtime (listed in numpy's licence file) | BSD-3-Clause; GCC runtime GPL-3.0-or-later WITH GCC-exception-3.1 |  |
| `numpy` | `msvcp140-` | Microsoft C++ runtime (numpy's own copy) | Microsoft Visual C++ Redistributable terms | **Check before release** |
| `onnxruntime` | `onnxruntime(_providers_shared)?\.dll` | ONNX Runtime | MIT (the package's ThirdPartyNotices.txt lists what it includes) |  |
| `sherpa-onnx` | `onnxruntime\.dll` | ONNX Runtime | MIT |  |
| `sherpa-onnx` | `sherpa-onnx-c(xx)?-api\.dll` | sherpa-onnx | Apache-2.0 |  |

## Removed by the build

| Package | File | Why |
|---|---|---|
| `ctranslate2` | `cudnn64_9.dll` | NVIDIA cuDNN 9.10.2. CTranslate2 builds its Windows wheel without cuDNN (`-DWITH_CUDNN=OFF`), so `ctranslate2.dll` never loads it (Whisper on CUDA ran without it and loaded only cuBLAS), and cuDNN's licence covers applications on systems with NVIDIA GPUs, which the CPU edition is not limited to |

## Licence texts the build adds

The wheels below carry no licence files for what they contain. The texts are kept in this repository's `licenses/`
folder (collected for ctranslate2 4.8.5; a test fails when `uv.lock` moves to another version) and copied to the folder's
`licenses\`.

| In the folder | Covers | Source |
|---|---|---|
| `licenses\ctranslate2\CTranslate2-LICENSE.txt` | CTranslate2 (MIT) | <https://github.com/OpenNMT/CTranslate2/blob/v4.8.5/LICENSE> |
| `licenses\ctranslate2\oneMKL-LICENSE.txt` | Intel oneMKL 2025.3, linked into `ctranslate2.dll` (Intel Simplified Software License) | `onemkl_license-2025.3.0` wheel, <https://pypi.org/project/onemkl-license/2025.3.0/> |
| `licenses\ctranslate2\oneMKL-third-party-programs.txt` | what oneMKL includes | the same wheel |
| `licenses\ctranslate2\oneDNN-LICENSE.txt` | oneDNN 3.1.1, linked into `ctranslate2.dll` (Apache-2.0) | <https://github.com/oneapi-src/oneDNN/blob/v3.1.1/LICENSE> |
| `licenses\ctranslate2\oneDNN-THIRD-PARTY-PROGRAMS.txt` | what oneDNN includes (Xbyak, ITT API and others) | <https://github.com/oneapi-src/oneDNN/blob/v3.1.1/THIRD-PARTY-PROGRAMS> |
| `licenses\ctranslate2\Intel-OpenMP-EULA.txt` | `libiomp5md.dll` (Intel End User License Agreement for Developer Tools, August 2024) | `intel_openmp-2025.3.0` wheel, <https://pypi.org/project/intel-openmp/2025.3.0/>, whose `libiomp5md.dll` is the same file |
| `licenses\ctranslate2\Intel-OpenMP-third-party-programs.txt` | what the OpenMP runtime includes | the same wheel |
| `licenses\ctranslate2\cpu_features-LICENSE.txt` | cpu_features (Apache-2.0), linked into `ctranslate2.dll` on x86-64 | <https://github.com/google/cpu_features/blob/8a494eb1e158ec2050e5f699a504fbc9b896a43b/LICENSE> |
| `licenses\ctranslate2\spdlog-LICENSE.txt` | spdlog (MIT), built into `ctranslate2.dll` | <https://github.com/gabime/spdlog/blob/76fb40d95455f249bd70824ecfcae7a8f0930fa3/LICENSE> |
| `licenses\ctranslate2\CCCL-LICENSE.txt` | Thrust, CUB and libcu++ (NVIDIA CCCL), used by the CUDA kernels in `ctranslate2.dll` | <https://github.com/NVIDIA/cccl/blob/b5fe509fd11a925f90d6495176707cc1184eed9d/LICENSE> |

## Runtimes the build adds

| Component | Where in the folder | Licence | |
|---|---|---|---|
| CPython 3.14.10 (python.org embeddable package) | `runtime\` | PSF-2.0; `runtime\LICENSE.txt` also covers what Python bundles (OpenSSL, libffi, bzip2, expat, SQLite, zlib, mpdecimal and others) | |
| Microsoft Visual C++ runtime (`msvcp140.dll` and the DLLs it needs, copied app-local from a Visual Studio redist folder) | `runtime\` | Distributable Code under the Visual Studio licence (the REDIST list) | **Check before release** |
| NVIDIA cuBLAS 12.9.2.10 (`cublas64_12.dll`, `cublasLt64_12.dll`; GPU edition only) | `runtime\`, with `runtime\cuBLAS-LICENSE.txt` | CUDA EULA; cuBLAS is listed in its redistributable files (Attachment A) | **Check before release** |
