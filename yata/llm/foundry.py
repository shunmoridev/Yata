"""The service listens on a port it picks itself; `foundry service status` prints the URL, which is how
Microsoft's own SDK finds it. Foundry's chat API documents no structured output, so the JSON schema goes
into the prompt.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from ..i18n import t
from .base import LABELS, LlmCancelled, LlmUnavailable, local_opener, loopback_url
from .chat import ChatClient

STATUS_TTL_S = 30.0
LOAD_TTL_S = 30 * 60
_URL = re.compile(r"https?://[^\s/\"'<>]+")
_EPS = re.compile(r"Valid EPs:\s*(.+)")
_PROGRESS = re.compile(r"\(\s*\"([^\"]*)\"\s*,\s*([0-9.]+)\s*\)")
_CREATE_NO_WINDOW = 0x08000000
# Variant name = alias + one of these (Foundry's naming), optionally followed by ":<version>".
_SUFFIXES = ("-cuda-gpu", "-generic-gpu", "-generic-cpu", "-trtrtx-gpu")


def variant_alias(name: str, aliases: Iterable[str]) -> str | None:
    base = name.split(":", 1)[0].lower()
    for alias in aliases:
        if any(base == alias.lower() + suffix for suffix in _SUFFIXES):
            return alias
    # Foundry also names some variants after the instruct model (qwen2.5-7b-instruct-cuda-gpu for qwen2.5-7b).
    for alias in aliases:
        if any(base == alias.lower() + "-instruct" + suffix for suffix in _SUFFIXES):
            return alias
    return None


class FoundryLocal:
    kind = "foundry"

    def __init__(self, allowed: Iterable[str], *, which: Callable[[str], str | None] = shutil.which,
                 run: Callable[..., Any] = subprocess.run, opener: urllib.request.OpenerDirector | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._allowed = tuple(allowed)
        self._which = which
        self._run = run
        self._opener = opener or local_opener()
        self._clock = clock
        self._lock = threading.Lock()
        self._service: dict[str, Any] | None = None
        self._checked = 0.0
        self._loaded: dict[str, float] = {}

    def installed(self) -> bool:
        return self._which("foundry") is not None

    def _cli(self, *args: str, timeout: float) -> str:
        done = self._run(["foundry", *args], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL,
                         creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0)
        return (done.stdout or b"").decode("utf-8", "replace") + "\n" + (done.stderr or b"").decode("utf-8", "replace")

    def service(self, refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            if not refresh and self._service is not None and self._clock() - self._checked < STATUS_TTL_S:
                return dict(self._service)
            if not self.installed():
                result: dict[str, Any] = {"state": "absent", "url": None, "eps": [], "reason": t("Foundry Local は入っていません。", "Foundry Local is not installed.")}
            else:
                result = self._status()
            self._service, self._checked = result, self._clock()
            return dict(result)

    def _status(self) -> dict[str, Any]:
        try:
            text = self._cli("service", "status", timeout=20)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"state": "stopped", "url": None, "eps": [], "reason": t(f"foundry コマンドを実行できませんでした（{type(exc).__name__}）。",
                                                                     f"Could not run the foundry command ({type(exc).__name__}).")}
        eps = [item.strip() for item in (_EPS.search(text).group(1).split(",") if _EPS.search(text) else []) if item.strip()]
        match = _URL.search(text)
        if match is None or "not running" in text.lower():
            return {"state": "stopped", "url": None, "eps": eps, "reason": t("Foundry Local のサービスが起動していません。", "The Foundry Local service is not running.")}
        try:
            url = loopback_url(match.group(0))
        except ValueError:
            return {"state": "stopped", "url": None, "eps": eps, "reason": t("Foundry Local のサービスがこの PC 以外の URL を示しました。",
                                                                         "The Foundry Local service named a URL outside this PC.")}
        return {"state": "running", "url": url, "eps": eps, "reason": None}

    def start(self) -> None:
        if not self.installed():
            raise LlmUnavailable(t("Foundry Local は入っていません。", "Foundry Local is not installed."))
        try:
            self._cli("service", "start", timeout=180)
        except (OSError, subprocess.SubprocessError) as exc:
            raise LlmUnavailable(t(f"Foundry Local のサービスを起動できませんでした（{type(exc).__name__}）。",
                                    f"Could not start the Foundry Local service ({type(exc).__name__}).")) from exc
        if self.service(refresh=True)["state"] != "running":
            raise LlmUnavailable(t("Foundry Local のサービスが起動しませんでした。foundry service status で状態を確認してください。",
                                    "The Foundry Local service did not start. Check it with foundry service status."))

    def _url(self) -> str:
        service = self.service()
        if service["state"] != "running":
            raise LlmUnavailable(service["reason"] or t("Foundry Local のサービスが起動していません。", "The Foundry Local service is not running."))
        return service["url"]

    def _get(self, path: str, timeout: float = 5.0) -> Any:
        try:
            with self._opener.open(self._url() + path, timeout=timeout) as response:
                body = response.read(8_000_000).decode("utf-8")
        except LlmUnavailable:
            raise
        except (OSError, urllib.error.URLError, ValueError) as exc:
            # The port changes when the service restarts: ask the CLI again next time.
            with self._lock:
                self._service = None
            raise LlmUnavailable(t(f"Foundry Local に接続できません（{type(exc).__name__}）。", f"Cannot connect to Foundry Local ({type(exc).__name__}).")) from exc
        return json.loads(body) if body.strip() else None

    def model_dir(self) -> Path | None:
        try:
            status = self._get("/openai/status")
        except (LlmUnavailable, ValueError):
            return None
        value = status.get("modelDirPath") or status.get("ModelDirPath") if isinstance(status, dict) else None
        return Path(value) if value else None

    def cached(self) -> list[str]:
        names = self._get("/openai/models")
        return [str(name) for name in names] if isinstance(names, list) else []

    def variants(self, cached: list[str] | None = None) -> dict[str, str]:
        found: dict[str, str] = {}
        for name in cached if cached is not None else self.cached():
            alias = variant_alias(name, self._allowed)
            if alias is None:
                continue
            # More than one variant cached: prefer CUDA, then GPU, then CPU (the order of _SUFFIXES).
            rank = next((index for index, suffix in enumerate(_SUFFIXES) if suffix in name.lower()), len(_SUFFIXES))
            current = found.get(alias)
            if current is None or rank < next((index for index, suffix in enumerate(_SUFFIXES) if suffix in current.lower()), len(_SUFFIXES)):
                found[alias] = name
        return found

    def models(self) -> list[str]:
        found = self.variants()
        return [alias for alias in self._allowed if alias in found]

    def allows(self, model: str) -> bool:
        return model in self._allowed

    def _variant(self, alias: str) -> str:
        if not self.allows(alias):
            raise LlmUnavailable(t("Foundry Local で要約に許可されたモデルではありません。", "This model is not allowed for summaries with Foundry Local."))
        name = self.variants().get(alias)
        if name is None:
            raise LlmUnavailable(t(f"Foundry Local に {alias} がありません。「この PC の準備」で取得してください。",
                                    f"Foundry Local has no {alias}. Get it under \"Set up this PC\"."))
        return name

    def _load(self, name: str) -> None:
        if self._clock() - self._loaded.get(name, -LOAD_TTL_S) < LOAD_TTL_S / 2:
            return
        self._get(f"/openai/load/{urllib.parse.quote(name, safe='')}?ttl={LOAD_TTL_S}", timeout=600)
        self._loaded[name] = self._clock()

    def warm(self, model: str) -> None:
        try:
            self._load(self._variant(model))
        except Exception:  # noqa: BLE001 - the first real request reports the failure
            pass

    def chat_json(self, model: str, *, task: str, system: str, user: dict[str, Any], schema: dict[str, Any],
                  timeout: float = 120.0) -> dict[str, Any]:
        name = self._variant(model)
        self._load(name)
        client = ChatClient(self._url() + "/v1", label=LABELS[self.kind], structured="prompt", opener=self._opener)
        return client.chat_json(name, task=task, system=system, user=user, schema=schema, timeout=timeout)

    def catalog_entry(self, alias: str, *, cuda: bool, gpu: bool) -> dict[str, Any]:
        if not self.allows(alias):
            raise LlmUnavailable(t("Foundry Local で要約に許可されたモデルではありません。", "This model is not allowed for summaries with Foundry Local."))
        listed = self._get("/foundry/list", timeout=60)
        entries = listed.get("models", []) if isinstance(listed, dict) else listed if isinstance(listed, list) else []
        candidates = [item for item in entries if isinstance(item, dict) and item.get("alias") == alias]
        eps = set(self.service()["eps"])
        order = (["CUDAExecutionProvider"] if cuda and "CUDAExecutionProvider" in eps else []) \
            + (["WebGpuExecutionProvider"] if gpu else []) + ["CPUExecutionProvider"]
        for provider in order:
            for item in candidates:
                if (item.get("runtime") or {}).get("executionProvider") == provider:
                    return item
        raise LlmUnavailable(t(f"Foundry Local のカタログに、この PC で動く {alias} がありません。",
                                f"Foundry Local's catalog has no {alias} that runs on this PC."))

    def download(self, entry: dict[str, Any], on_progress: Callable[[str, float], None], cancelled: Callable[[], bool]) -> None:
        body = {"model": {"Uri": entry.get("uri"), "Name": entry.get("name"), "ProviderType": entry.get("providerType"),
                          "Publisher": entry.get("publisher") or "", "PromptTemplate": entry.get("promptTemplate")},
                "ignorePipeReport": True}
        request = urllib.request.Request(self._url() + "/openai/download", data=json.dumps(body).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/json"})
        text = ""
        try:
            with self._opener.open(request, timeout=600) as response:
                read = getattr(response, "read1", response.read)
                while True:
                    if cancelled():
                        raise LlmCancelled(t("中止しました。Foundry Local が取得済みの分をどう扱うかは Foundry 次第です。",
                                              "Cancelled. What Foundry Local does with the part already fetched is up to Foundry."))
                    chunk = read(8192)
                    if not chunk:
                        break
                    text += chunk.decode("utf-8", "replace")
                    progress = list(_PROGRESS.finditer(text[-4096:]))
                    if progress:
                        name, value = progress[-1].group(1), float(progress[-1].group(2))
                        on_progress(name, value / 100 if value > 1 else value)
        except (LlmCancelled, LlmUnavailable):
            raise
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise LlmUnavailable(t(f"Foundry Local からの取得が途中で止まりました（{type(exc).__name__}）。",
                                    f"The download from Foundry Local stopped ({type(exc).__name__}).")) from exc
        final = text[text.rfind("{"):] if "{" in text else ""
        try:
            result = json.loads(final) if final else {}
        except ValueError:
            result = {}
        if result.get("Success") is False or result.get("success") is False:
            message = str(result.get("ErrorMessage") or result.get("errorMessage") or "")[:200]
            raise LlmUnavailable(t(f"Foundry Local がモデルを取得できませんでした: {message}", f"Foundry Local could not get the model: {message}"))
