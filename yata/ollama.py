from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .i18n import t
from .llm.base import LlmCancelled, LlmUnavailable


OLLAMA_URL = "http://127.0.0.1:11434"
# Deliberately narrow and ordered by preference: the first installed one is the UI default.
# SystemOne and decision models are never candidates; names alone are not trusted as safe.
SUMMARY_MODELS = ("qwen3.6:35b-a3b", "qwen3.8:27b", "qwen3:8b", "mistral-nemo:12b")
SUMMARY_MODEL_ALLOWLIST = set(SUMMARY_MODELS)
INSTALL_URL = "https://ollama.com/download"


class OllamaUnavailable(LlmUnavailable):
    pass


class PullCancelled(LlmCancelled):
    pass


def installed_locally(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    if shutil.which("ollama"):
        return True
    candidates = []
    if sys.platform == "win32" and env.get("LOCALAPPDATA"):
        candidates.append(Path(env["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe")
    if sys.platform == "darwin":
        candidates.append(Path("/Applications/Ollama.app"))
    return any(path.exists() for path in candidates)


def store_dir(environ: dict[str, str] | None = None, reported: Path | None = None) -> Path:
    """The desktop app's own "Model location" setting reaches only the server it starts, not Yata's environment, so
    without a model to ask about the server's log is the one place that names it.
    """
    env = os.environ if environ is None else environ
    if reported is not None:
        return reported
    logged = _logged_store(env)
    if logged is not None:
        return logged
    configured = env.get("OLLAMA_MODELS", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".ollama" / "models"


# The settings Ollama's server logs when it starts: `msg="server config" env="map[... OLLAMA_MODELS:<folder> OLLAMA_...]"`,
# the folder with its backslashes doubled. The value ends where the next setting (an upper-case name and ":") begins.
_LOGGED_STORE = re.compile(r'msg="server config".*?OLLAMA_MODELS:(.+?)(?= [A-Z][A-Z0-9_]*:|\]")')


def _logged_store(env: dict[str, str]) -> Path | None:
    if sys.platform == "win32" and env.get("LOCALAPPDATA"):
        log = Path(env["LOCALAPPDATA"]) / "Ollama" / "server.log"
    elif sys.platform == "darwin" and env.get("HOME"):
        log = Path(env["HOME"]) / ".ollama" / "logs" / "server.log"
    else:
        return None
    try:
        with log.open("rb") as handle:  # a new log per start: the settings are near its top
            head = handle.read(1_000_000).decode("utf-8", "replace")
    except OSError:
        return None
    found = _LOGGED_STORE.findall(head)
    if not found:
        return None
    folder = Path(found[-1].replace("\\\\", "\\"))
    return folder if folder.is_dir() else None


class OllamaClient:
    kind = "ollama"

    def __init__(self, base_url: str = OLLAMA_URL, num_ctx: int = 16_384) -> None:
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self._models_dir: tuple[float, Path | None] | None = None

    def allows(self, model: str) -> bool:
        return model in SUMMARY_MODEL_ALLOWLIST

    def tags(self) -> dict[str, str]:
        request = urllib.request.Request(f"{self.base_url}/api/tags", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                payload = json.loads(response.read(2_000_000).decode("utf-8"))
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise OllamaUnavailable(t("Ollama に接続できません。起動状態を確認してください。", "Cannot connect to Ollama. Check that it is running.")) from exc
        digests = {str(item.get("name", "")): str(item.get("digest", "")) for item in payload.get("models", []) if isinstance(item, dict)}
        return {name: digests[name] for name in SUMMARY_MODELS if name in digests}

    def models(self) -> list[str]:
        return list(self.tags())

    def models_dir(self) -> Path | None:
        now = time.monotonic()
        if self._models_dir is not None and now - self._models_dir[0] < 60:
            return self._models_dir[1]
        found = None
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as response:
                listed = json.loads(response.read(2_000_000).decode("utf-8")).get("models", [])
            names = [str(item.get("name", "")) for item in listed if isinstance(item, dict) and item.get("name")]
            if names:
                body = json.dumps({"model": next((name for name in SUMMARY_MODELS if name in names), names[0])}).encode("utf-8")
                request = urllib.request.Request(f"{self.base_url}/api/show", data=body, method="POST",
                                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
                with urllib.request.urlopen(request, timeout=3) as response:
                    modelfile = str(json.loads(response.read(8_000_000).decode("utf-8")).get("modelfile") or "")
                # "FROM <folder>/blobs/sha256-..." for a model pulled into Ollama's folder.
                blob = next((Path(line[5:].strip()) for line in modelfile.splitlines() if line.startswith("FROM ")), None)
                found = blob.parent.parent if blob is not None and blob.parent.name == "blobs" else None
        except (OSError, urllib.error.URLError, ValueError, AttributeError):
            found = None
        self._models_dir = (now, found)
        return found

    def version(self) -> str | None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/version", timeout=3) as response:
                return str(json.loads(response.read(10_000).decode("utf-8")).get("version") or "") or None
        except (OSError, urllib.error.URLError, ValueError):
            return None

    def pull(self, model: str, on_progress: Callable[[str, int, int], None], cancelled: Callable[[], bool]) -> None:
        if model not in SUMMARY_MODEL_ALLOWLIST:
            raise OllamaUnavailable(t("要約に許可されたモデルではありません。", "This model is not allowed for summaries."))
        body = json.dumps({"model": model, "stream": True}).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/api/pull", data=body, method="POST",
                                         headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"})
        layers: dict[str, tuple[int, int]] = {}
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                for line in response:
                    if cancelled():
                        raise PullCancelled(t("中止しました。取得済みの分は Ollama が次回に使います。",
                                                     "Cancelled. Ollama uses the part already fetched next time."))
                    if not line.strip():
                        continue
                    item = json.loads(line.decode("utf-8"))
                    if item.get("error"):
                        raise OllamaUnavailable(t(f"Ollama がモデルを取得できませんでした: {str(item['error'])[:200]}",
                                                   f"Ollama could not get the model: {str(item['error'])[:200]}"))
                    digest, total = item.get("digest"), item.get("total")
                    if digest and isinstance(total, int) and total > 0:
                        layers[digest] = (int(item.get("completed") or 0), total)
                    on_progress(str(item.get("status", "")), sum(done for done, _ in layers.values()),
                                sum(size for _, size in layers.values()))
                    if item.get("status") == "success":
                        return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise OllamaUnavailable(t(f"Ollama からの取得が途中で止まりました（{type(exc).__name__}）。再試行すると続きから取得します。",
                                     f"The download from Ollama stopped ({type(exc).__name__}). Try again to continue it.")) from exc
        raise OllamaUnavailable(t("Ollama の応答が途中で終わりました。再試行すると続きから取得します。",
                                   "Ollama's answer ended early. Try again to continue the download."))

    def warm(self, model: str) -> None:
        if model not in SUMMARY_MODEL_ALLOWLIST:
            return
        body = json.dumps({"model": model, "keep_alive": "30m"}).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                response.read(100_000)
        except (OSError, urllib.error.URLError, ValueError):
            pass  # the first real request reports the failure

    def chat_json(
        self, model: str, *, task: str, system: str, user: dict[str, Any], schema: dict[str, Any], timeout: float = 120.0,
    ) -> dict[str, Any]:
        if model not in SUMMARY_MODEL_ALLOWLIST:
            raise OllamaUnavailable(t("要約に許可されたモデルではありません。", "This model is not allowed for summaries."))
        body = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": "30m",
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx},
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read(4_000_000).decode("utf-8"))
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise OllamaUnavailable(t(f"Ollama の {task} に失敗しました（{type(exc).__name__}）。接続とモデル状態を確認してください。",
                                     f"Ollama failed at {task} ({type(exc).__name__}). Check the connection and the model.")) from exc
        try:
            result = json.loads(payload.get("message", {}).get("content", ""))
        except (TypeError, ValueError) as exc:
            raise OllamaUnavailable(t(f"Ollama が {task} で有効な JSON を返しませんでした。", f"Ollama returned no valid JSON for {task}.")) from exc
        if not isinstance(result, dict):
            raise OllamaUnavailable(t(f"Ollama の {task} の応答形式が不正です。", f"Ollama answered {task} in an unexpected form."))
        return result
