from __future__ import annotations

import os
from typing import Any, Callable

from ..i18n import t
from .base import LABELS, LlmUnavailable
from .chat import ChatClient

KEY_ENV = "YATA_OPENAI_API_KEY"


class OpenAICompatBackend:
    kind = "openai"

    def __init__(self, config: Callable[[], dict[str, Any]], environ: dict[str, str] | None = None, opener: Any = None) -> None:
        # config() -> {"base_url": str, "model": str}, read on every call so a saved change applies at once.
        self._config = config
        self._env = os.environ if environ is None else environ
        self._opener = opener
        self._client: ChatClient | None = None

    def key_present(self) -> bool:
        return bool(self._env.get(KEY_ENV, "").strip())

    def client(self) -> ChatClient:
        config = self._config()
        base_url = str(config.get("base_url") or "")
        if not base_url:
            raise LlmUnavailable(t("OpenAI 互換 API の URL が設定されていません。", "No URL is set for the OpenAI-compatible API."))
        key = self._env.get(KEY_ENV, "").strip() or None
        current = self._client
        if current is None or current.base_url != base_url.rstrip("/") or current.api_key != key:
            try:
                current = ChatClient(base_url, label=LABELS[self.kind], api_key=key, opener=self._opener)
            except ValueError as exc:
                raise LlmUnavailable(str(exc)) from exc
            self._client = current
        return current

    def configured_model(self) -> str:
        return str(self._config().get("model") or "")

    def server_models(self) -> list[str]:
        return self.client().model_ids()

    def models(self) -> list[str]:
        model = self.configured_model()
        if not model:
            raise LlmUnavailable(t("OpenAI 互換 API のモデルが設定されていません。", "No model is set for the OpenAI-compatible API."))
        listed = self.server_models()
        if listed and model not in listed:
            raise LlmUnavailable(t(f"OpenAI 互換 API のサーバーにモデル {model} がありません。", f"The OpenAI-compatible server has no model {model}."))
        return [model]

    def allows(self, model: str) -> bool:
        return bool(model) and model == self.configured_model()

    def chat_json(self, model: str, *, task: str, system: str, user: dict[str, Any], schema: dict[str, Any],
                  timeout: float = 120.0) -> dict[str, Any]:
        if not self.allows(model):
            raise LlmUnavailable(t("設定した OpenAI 互換 API のモデルではありません。", "This is not the model set for the OpenAI-compatible API."))
        return self.client().chat_json(model, task=task, system=system, user=user, schema=schema, timeout=timeout)

    def warm(self, model: str) -> None:
        return None  # the server loads models itself
