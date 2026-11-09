from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from ..i18n import t
from .base import LlmUnavailable, local_opener, loopback_url, reply_json, schema_instruction

Opener = Callable[..., Any]


class ChatClient:
    def __init__(self, base_url: str, *, label: str, api_key: str | None = None, structured: str = "json_schema",
                 extra: dict[str, Any] | None = None, opener: urllib.request.OpenerDirector | None = None) -> None:
        # Checked here too, so no caller can point a client off this PC.
        self.base_url = loopback_url(base_url)
        self.label = label
        self.api_key = api_key or None
        self.structured = structured
        self.extra = dict(extra or {})
        self._opener = opener or local_opener()

    def _request(self, path: str, body: dict[str, Any] | None, timeout: float) -> Any:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method="GET" if body is None else "POST")
        with self._opener.open(request, timeout=timeout) as response:
            return json.loads(response.read(8_000_000).decode("utf-8"))

    def get_json(self, path: str, timeout: float = 3.0) -> Any:
        try:
            return self._request(path, None, timeout)
        except urllib.error.HTTPError as exc:
            raise LlmUnavailable(t(f"{self.label} が {path} に {exc.code} を返しました。", f"{self.label} returned {exc.code} for {path}.")) from exc
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise LlmUnavailable(t(f"{self.label} に接続できません（{type(exc).__name__}）。起動状態と URL を確認してください。",
                                    f"Cannot connect to {self.label} ({type(exc).__name__}). Check that it runs and the URL.")) from exc

    def model_ids(self, timeout: float = 3.0) -> list[str]:
        payload = self.get_json("/models", timeout)
        items = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        ids = [str(item.get("id") if isinstance(item, dict) else item) for item in items]
        return [item for item in ids if item and item != "None"]

    def chat_json(self, model: str, *, task: str, system: str, user: dict[str, Any], schema: dict[str, Any],
                  timeout: float = 120.0) -> dict[str, Any]:
        try:
            return self._chat(model, task, system, user, schema, timeout, self.structured)
        except urllib.error.HTTPError as exc:
            detail = _error_text(exc)
            # A server without structured output says so about response_format; ask through the prompt instead.
            if self.structured == "json_schema" and exc.code in (400, 422, 500, 501) and (
                    "response_format" in detail or "json_schema" in detail):
                self.structured = "prompt"
                try:
                    return self._chat(model, task, system, user, schema, timeout, "prompt")
                except urllib.error.HTTPError as again:
                    raise LlmUnavailable(t(f"{self.label} の {task} に失敗しました（HTTP {again.code}）。{_error_text(again)[:160]}",
                                            f"{self.label} failed at {task} (HTTP {again.code}). {_error_text(again)[:160]}")) from again
            raise LlmUnavailable(t(f"{self.label} の {task} に失敗しました（HTTP {exc.code}）。{detail[:160]}",
                                    f"{self.label} failed at {task} (HTTP {exc.code}). {detail[:160]}")) from exc
        except LlmUnavailable:
            raise
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise LlmUnavailable(t(f"{self.label} の {task} に失敗しました（{type(exc).__name__}）。接続とモデル状態を確認してください。",
                                    f"{self.label} failed at {task} ({type(exc).__name__}). Check the connection and the model.")) from exc

    def _chat(self, model: str, task: str, system: str, user: dict[str, Any], schema: dict[str, Any], timeout: float,
              structured: str) -> dict[str, Any]:
        prompt = system if structured == "json_schema" else system + "\n\n" + schema_instruction(schema)
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            "stream": False,
            "temperature": 0.2,
            **self.extra,
        }
        if structured == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "yata_reply", "schema": schema, "strict": True}}
        payload = self._request("/chat/completions", body, timeout)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmUnavailable(t(f"{self.label} の {task} の応答形式が不正です。", f"{self.label} answered {task} in an unexpected form.")) from exc
        result = reply_json(content)
        if result is None:
            raise LlmUnavailable(t(f"{self.label} が {task} で有効な JSON を返しませんでした。", f"{self.label} returned no valid JSON for {task}."))
        return result


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read(4_000).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - the status code is enough
        return ""
