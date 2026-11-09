"""Every backend talks only to a loopback address, never through a proxy and never following a redirect,
so a local server cannot forward a transcript elsewhere through Yata.
"""
from __future__ import annotations

import ipaddress
import json
import re
import urllib.error
import urllib.request
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from ..i18n import Text, t

# The order the page shows them in.
BACKENDS = ("builtin", "ollama", "openai", "foundry")
LABELS = {"builtin": Text("内蔵（llama.cpp）", "Built-in (llama.cpp)"), "ollama": Text("Ollama", "Ollama"),
          "openai": Text("OpenAI 互換 API", "OpenAI-compatible API"), "foundry": Text("Foundry Local", "Foundry Local")}
# Yata's own port: an LLM "server" there would be Yata itself.
YATA_PORT = 8765


class LlmUnavailable(RuntimeError):
    pass


class LlmCancelled(RuntimeError):
    pass


class LlmBackend(Protocol):
    kind: str

    def models(self) -> list[str]:
        ...

    def allows(self, model: str) -> bool:
        ...

    def chat_json(self, model: str, *, task: str, system: str, user: dict[str, Any], schema: dict[str, Any],
                  timeout: float = 120.0) -> dict[str, Any]:
        ...

    def warm(self, model: str) -> None:
        ...


def loopback_url(raw: str) -> str:
    text = str(raw or "").strip()
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError as exc:
        raise ValueError(t("URL の形式が正しくありません。", "The URL is not valid.")) from exc
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(t("URL は http:// または https:// で始めてください（例: http://127.0.0.1:1234/v1）。",
                           "Start the URL with http:// or https:// (for example http://127.0.0.1:1234/v1)."))
    if parts.username or parts.password:
        raise ValueError(t("URL に利用者名やパスワードを含めないでください。", "Do not put a user name or password in the URL."))
    if not is_loopback(parts.hostname):
        raise ValueError(t("接続先はこの PC（127.0.0.1・localhost・::1）だけです。字幕の文字起こしを PC の外に送らないためです。",
                           "Only servers on this PC (127.0.0.1, localhost, ::1) are allowed, so transcripts never leave this PC."))
    if port == YATA_PORT:
        raise ValueError(t(f"{YATA_PORT} 番は Yata 自身のポートです。", f"Port {YATA_PORT} is Yata's own port."))
    if parts.query or parts.fragment:
        raise ValueError(t("URL に ? や # 以降を含めないでください。", "Do not put ? or # (or anything after them) in the URL."))
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001 - urllib's signature
        raise urllib.error.HTTPError(req.full_url, code, t("Yata は LLM サーバーの転送先には送りません。",
                                                            "Yata does not follow an LLM server's redirects."), headers, fp)


def local_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def reply_json(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = _FENCE.sub("", _THINK.sub("", text).strip()).strip()
    try:
        value = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(body[start:end + 1])
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def schema_instruction(schema: dict[str, Any]) -> str:
    return ("Reply with exactly one JSON object and nothing else (no code fences, no explanation). "
            "It must match this JSON schema: " + json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
