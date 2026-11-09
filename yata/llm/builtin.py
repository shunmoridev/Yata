"""Loading a model can take minutes, so the lock is held only to look and to launch, never during the load:
the setup page (state), a change of backend (release) and exit (close) answer at once, and exit also stops
a server that is still loading.

The Vulkan build runs on NVIDIA, AMD and Intel GPUs and falls back to the CPU. The device is chosen
explicitly (pick_device), because llama.cpp would otherwise also offload to an integrated GPU.
"""
from __future__ import annotations

import collections
import contextlib
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from ..i18n import Text, capitalize, t
from .base import LABELS, LlmUnavailable, local_opener
from .chat import ChatClient

CONTEXT = 16_384  # the same context Yata asks of Ollama
IDLE_S = 30 * 60.0  # like Ollama's keep_alive of 30 minutes
START_TIMEOUT_S = 600.0
_CREATE_NO_WINDOW = 0x08000000
_DEVICE = re.compile(r"^\s*([A-Za-z]+\d+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)\s*$")
_INTEGRATED = re.compile(r"Radeon\(TM\) Graphics|Radeon Graphics|UHD Graphics|Iris|Intel\(R\) Graphics|Basic Render", re.IGNORECASE)


@dataclass(frozen=True)
class LaunchSpec:
    executable: Path
    model_path: Path
    model_id: str
    device: str | None


def parse_devices(text: str) -> list[dict[str, Any]]:
    devices = []
    for line in text.splitlines():
        match = _DEVICE.match(line)
        if match:
            devices.append({"id": match.group(1), "name": match.group(2), "memory": int(match.group(3)) * 1024 * 1024,
                            "free": int(match.group(4)) * 1024 * 1024})
    return devices


def list_devices(executable: Path, run: Callable[..., Any] = subprocess.run, timeout: float = 60.0) -> list[dict[str, Any]]:
    try:
        done = run([str(executable), "--list-devices"], capture_output=True, timeout=timeout, cwd=str(executable.parent),
                   stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LlmUnavailable(t(f"llama.cpp を起動できませんでした（{type(exc).__name__}）。", f"Could not start llama.cpp ({type(exc).__name__}).")) from exc
    text = (done.stdout or b"").decode("utf-8", "replace") + "\n" + (done.stderr or b"").decode("utf-8", "replace")
    if done.returncode != 0 and "Available devices" not in text:
        last = [line for line in text.splitlines() if line.strip()][-1:] or [t(f"終了コード {done.returncode}", f"exit code {done.returncode}")]
        raise LlmUnavailable(t(f"llama.cpp を起動できませんでした: {last[0][:200]}", f"Could not start llama.cpp: {last[0][:200]}"))
    return parse_devices(text)


def pick_device(devices: list[dict[str, Any]], gpu_name: str | None) -> str:
    if not devices:
        return "none"
    if gpu_name:
        wanted = gpu_name.lower()
        for device in devices:
            name = device["name"].lower()
            if wanted in name or name in wanted:
                return device["id"]
    discrete = [device for device in devices if not _INTEGRATED.search(device["name"])]
    return (discrete or devices)[0]["id"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _tie_to_parent(process: Any) -> Any:
    """Windows: put the child in a job object that kills it when Yata's handle closes (also when Yata dies)."""
    if os.name != "nt" or not hasattr(process, "_handle"):
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("Read", "Write", "Other", "ReadBytes", "WriteBytes", "OtherBytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Counters), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) or \
                not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle))):
            kernel32.CloseHandle(job)
            return None
        return (kernel32, job)
    except Exception:  # noqa: BLE001 - stopping on exit still works without the job object
        return None


class LlamaServer:
    def __init__(
        self,
        *,
        label: Text | str,
        task: Text | str,
        context: int,
        extra: Sequence[str] = (),
        spawn: Callable[..., Any] = subprocess.Popen,
        opener: urllib.request.OpenerDirector | None = None,
        idle_s: float = IDLE_S,
        start_timeout: float = START_TIMEOUT_S,
        port: Callable[[], int] = _free_port,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.label = label
        self.task = task
        self.context = context
        self.extra = tuple(extra)
        self._spawn = spawn
        self._opener = opener or local_opener()
        self.idle_s = idle_s
        self.start_timeout = start_timeout
        self._port = port
        self._clock = clock
        self._lock = threading.RLock()
        self._ready = threading.Condition(self._lock)
        self._starting: LaunchSpec | None = None
        self._process: Any = None
        self._job: Any = None
        self._spec: LaunchSpec | None = None
        self._url = ""
        self._busy = 0
        self._last = 0.0
        self._tail: collections.deque[str] = collections.deque(maxlen=30)
        self._watcher: threading.Thread | None = None
        self._closed = threading.Event()

    @contextlib.contextmanager
    def use(self, spec: LaunchSpec) -> Iterator[str]:
        url = self._ensure(spec, hold=True)
        try:
            yield url
        finally:
            with self._lock:
                self._busy -= 1
                self._last = self._clock()

    def warm(self, spec: LaunchSpec) -> None:
        self._ensure(spec)

    def state(self) -> dict[str, Any]:
        with self._lock:
            alive = self._process is not None and self._process.poll() is None
            ready = alive and bool(self._url)
            spec = self._spec if alive else None
            return {"running": ready, "starting": alive and not ready, "model": spec.model_id if spec else None,
                    "device": spec.device if spec else None}

    def release(self) -> None:
        with self._lock:
            if not self._busy and self._starting is None:
                self._stop()

    def close(self) -> None:
        """Stop the server, also one still loading (the request waiting for it fails at once). Nothing starts
        after this, so a request still running while Yata exits cannot leave a server behind."""
        self._closed.set()
        with self._lock:
            self._stop()
            self._ready.notify_all()

    def _ensure(self, spec: LaunchSpec, hold: bool = False) -> str:
        """The base URL of the server for `spec`, started if needed. With hold the server counts as busy before the
        lock is let go, so it cannot be stopped under the request (the caller then decrements _busy)."""
        with self._lock:
            while True:
                if self._closed.is_set():
                    raise LlmUnavailable(t(f"Yata を終了しているため、{self.label} は起動しません。", f"Yata is closing, so {self.label} does not start."))
                if self._starting is not None:
                    self._ready.wait()  # another request is loading a model: look again once it is up or failed
                    continue
                if self._process is not None and self._process.poll() is None and self._spec == spec and self._url:
                    return self._claim(self._url, hold)
                if self._busy and self._spec is not None and self._spec != spec:
                    raise LlmUnavailable(t(f"{self.label} は別のモデルで{self.task}中です。終わってからもう一度試してください。",
                                            f"{capitalize(self.label)} is busy with {self.task} on another model. Try again when it is done."))
                self._stop()
                process, url, reader = self._launch(spec)
                self._starting = spec
                break
        # The model loads without the lock.
        try:
            self._wait_ready(process, url, reader)
        except BaseException:
            with self._lock:
                if self._process is process:
                    self._stop()
                self._starting = None
                self._ready.notify_all()
            raise
        with self._lock:
            self._starting = None
            self._ready.notify_all()
            if self._process is not process:  # stopped while it loaded: Yata is closing
                raise LlmUnavailable(t(f"{self.label} の起動を中止しました。", f"Stopped starting {self.label}."))
            self._url = url
            if self._watcher is None:
                self._watcher = threading.Thread(target=self._watch, name="yata-llama-idle", daemon=True)
                self._watcher.start()
            return self._claim(url, hold)

    def _claim(self, url: str, hold: bool) -> str:
        self._last = self._clock()
        if hold:
            self._busy += 1
        return url

    def _launch(self, spec: LaunchSpec) -> tuple[Any, str, threading.Thread | None]:
        if not spec.executable.is_file():
            raise LlmUnavailable(t("llama.cpp の実行ファイルが見つかりません。「この PC の準備」で準備してください。",
                                    "The llama.cpp program is missing. Get it under \"Set up this PC\"."))
        if not spec.model_path.is_file():
            raise LlmUnavailable(t(f"{self.label} のモデルファイルが見つかりません。「この PC の準備」で準備してください。",
                                    f"The model file for {self.label} is missing. Get it under \"Set up this PC\"."))
        port = self._port()
        args = [str(spec.executable), "-m", str(spec.model_path), "--host", "127.0.0.1", "--port", str(port),
                "-c", str(self.context), "-np", "1", "--fit", "on", *self.extra, "--no-webui", "--offline",
                "--alias", spec.model_id]
        if spec.device:
            args += ["--device", spec.device]
        self._tail.clear()
        try:
            process = self._spawn(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  cwd=str(spec.executable.parent), creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            raise LlmUnavailable(t(f"llama.cpp を起動できませんでした（{type(exc).__name__}）。", f"Could not start llama.cpp ({type(exc).__name__}).")) from exc
        url = f"http://127.0.0.1:{port}"
        self._process, self._spec = process, spec
        self._job = _tie_to_parent(process)
        reader = None
        if getattr(process, "stderr", None) is not None:
            reader = threading.Thread(target=self._read_log, args=(process,), name="yata-llama-log", daemon=True)
            reader.start()
        return process, url, reader

    def _wait_ready(self, process: Any, url: str, reader: threading.Thread | None) -> None:
        deadline = self._clock() + self.start_timeout
        while True:
            if self._closed.is_set() or self._process is not process:
                raise LlmUnavailable(t(f"{self.label} の起動を中止しました。", f"Stopped starting {self.label}."))
            if process.poll() is not None:
                if reader is not None:
                    reader.join(timeout=2)  # the last lines say why it stopped
                reason = self._last_line() or t(f"終了コード {process.poll()}", f"exit code {process.poll()}")
                raise LlmUnavailable(t(f"{self.label} が起動できませんでした: {reason}", f"{capitalize(self.label)} could not start: {reason}"))
            if self._healthy(url):
                return
            if self._clock() > deadline:
                raise LlmUnavailable(t(f"{self.label} のモデルの読み込みが時間内に終わりませんでした。", f"{capitalize(self.label)} did not finish loading its model in time."))
            self._closed.wait(0.25)

    def _healthy(self, url: str) -> bool:
        try:
            with self._opener.open(url + "/health", timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError, ValueError):
            return False  # 503 while the model loads, or not listening yet

    def _read_log(self, process: Any) -> None:
        try:
            for raw in process.stderr:
                line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                if line:
                    self._tail.append(line)
        except (OSError, ValueError):
            pass

    def _last_line(self) -> str:
        lines = [line for line in self._tail if "error" in line.lower() or "failed" in line.lower()] or list(self._tail)
        return lines[-1][:240] if lines else ""

    def _stop(self) -> None:
        process, self._process, self._spec, self._url = self._process, None, None, ""
        job, self._job = self._job, None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    process.kill()
                except Exception:  # noqa: BLE001
                    pass
        if job is not None:
            try:
                job[0].CloseHandle(job[1])
            except Exception:  # noqa: BLE001
                pass

    def _watch(self) -> None:
        while not self._closed.wait(min(15.0, max(0.05, self.idle_s / 4))):
            with self._lock:
                if self._process is not None and self._starting is None and not self._busy and self._clock() - self._last > self.idle_s:
                    self._stop()


class BuiltinBackend:
    kind = "builtin"

    def __init__(
        self,
        *,
        resolve: Callable[[str], LaunchSpec],
        available: Callable[[], list[str]],
        allowed: Iterable[str],
        spawn: Callable[..., Any] = subprocess.Popen,
        opener: urllib.request.OpenerDirector | None = None,
        context: int = CONTEXT,
        idle_s: float = IDLE_S,
        start_timeout: float = START_TIMEOUT_S,
        port: Callable[[], int] = _free_port,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._resolve = resolve
        self._available = available
        self._allowed = tuple(allowed)
        self._opener = opener or local_opener()
        # Thinking off: every summary request asks for JSON only.
        self.server = LlamaServer(label=Text("内蔵 LLM", "the built-in LLM"), task=Text("要約", "summaries"), context=context, extra=("--reasoning", "off"), spawn=spawn,
                                  opener=self._opener, idle_s=idle_s, start_timeout=start_timeout, port=port, clock=clock)
        self._client: ChatClient | None = None

    def models(self) -> list[str]:
        return self._available()

    def allows(self, model: str) -> bool:
        return model in self._allowed

    def warm(self, model: str) -> None:
        try:
            self.server.warm(self._resolve(model))
        except Exception:  # noqa: BLE001 - the first real request reports the failure
            pass

    def chat_json(self, model: str, *, task: str, system: str, user: dict[str, Any], schema: dict[str, Any],
                  timeout: float = 120.0) -> dict[str, Any]:
        if not self.allows(model):
            raise LlmUnavailable(t("内蔵 LLM に許可されたモデルではありません。", "This model is not allowed for the built-in LLM."))
        spec = self._resolve(model)
        with self.server.use(spec) as url:
            return self._chat(url).chat_json(spec.model_id, task=task, system=system, user=user, schema=schema, timeout=timeout)

    def state(self) -> dict[str, Any]:
        return self.server.state()

    def release(self) -> None:
        self.server.release()

    def close(self) -> None:
        self.server.close()

    def _chat(self, url: str) -> ChatClient:
        """The chat client for the server at `url` (a restarted server listens on a new port)."""
        client = self._client
        if client is None or client.base_url != url + "/v1":
            client = self._client = ChatClient(url + "/v1", label=LABELS[self.kind], opener=self._opener)
        return client
