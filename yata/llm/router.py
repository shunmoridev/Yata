from __future__ import annotations

from typing import Any

from ..i18n import t
from .base import BACKENDS, LlmBackend, LlmUnavailable


class LlmRouter:
    def __init__(self, backends: dict[str, LlmBackend]) -> None:
        unknown = set(backends) - set(BACKENDS)
        if unknown:
            raise ValueError(f"unknown LLM backends: {sorted(unknown)}")
        self.backends = dict(backends)

    def route(self, kind: str) -> LlmBackend:
        backend = self.backends.get(kind)
        if backend is None:
            raise LlmUnavailable(t("この種類の LLM は使えません。", "This kind of LLM cannot be used."))
        return backend

    def allows(self, kind: str, model: str) -> bool:
        backend = self.backends.get(kind)
        return backend is not None and bool(model) and backend.allows(model)

    def close(self) -> None:
        for backend in self.backends.values():
            close: Any = getattr(backend, "close", None)
            if callable(close):
                close()
