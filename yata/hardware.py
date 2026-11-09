from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_static: dict[str, Any] = {}
_lock = threading.Lock()


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
            if name:
                return name
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def memory() -> dict[str, int | None]:
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong), ("page_free", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong), ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total": int(status.total), "available": int(status.available)}
        return {"total": None, "available": None}
    total = available = None
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError):
        pass
    return {"total": total, "available": available}


def _nvidia_gpu() -> dict[str, Any] | None:
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return None
    try:
        output = subprocess.run([tool, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = output.strip().splitlines()[0] if output.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 2 or not parts[0]:
        return None
    try:
        vram = int(float(parts[1])) * 1024 * 1024
    except ValueError:
        vram = None
    return {"name": parts[0], "vram": vram, "driver": parts[2] if len(parts) > 2 else None}


def cuda_available() -> bool:
    with _lock:
        if "cuda" not in _static:
            from .inputs.asr import _cuda_available
            _static["cuda"] = _cuda_available()
        return _static["cuda"]


def disk_free(path: Path) -> int | None:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def probe() -> dict[str, Any]:
    with _lock:
        if "cpu" not in _static:
            _static["cpu"] = {"name": _cpu_name(), "logical": os.cpu_count() or 1}
            _static["gpu"] = _nvidia_gpu()
    gpu = dict(_static["gpu"]) if _static["gpu"] else None
    if gpu is not None:
        gpu["cuda"] = cuda_available()
    return {"cpu": dict(_static["cpu"]), "memory": memory(), "gpu": gpu, "os": platform.system()}
