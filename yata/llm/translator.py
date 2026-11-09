"""A small context (4,096 tokens) is enough: one caption line and the two lines before it with their
translations. Anything in caption text shaped like a special token is removed before it goes into the PFML3
prompt, so captions can never add prompt structure.
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Sequence

from ..i18n import Text, capitalize, t
from .base import LlmUnavailable, local_opener
from .builtin import IDLE_S, START_TIMEOUT_S, LaunchSpec, LlamaServer, _free_port

LABEL = Text("翻訳モデル（PLaMo）", "the translation model (PLaMo)")
CONTEXT = 4_096
MAX_TOKENS = 256
STOP = ("<|plamo:tag|>", "<|plamo:bos|>")
# Lines longer than this are left untranslated: with two lines of history they would crowd the context, and
# their translation would not fit in MAX_TOKENS anyway.
MAX_LINE_CHARS = 1_200
MAX_HISTORY_CHARS = 300
# On the CPU the speed is the same with 4 threads as with 12 (memory-bound, measured on a Ryzen 9 9900X), and
# fewer threads leave room for the speech recognition that shares the CPU.
THREADS = max(1, min(4, (os.cpu_count() or 2) // 2))

PLAMO_LANGUAGES = frozenset({
    "ja", "en", "zh-hans", "zh-hant", "ko", "ar", "az", "be", "bg", "bn", "ca", "ceb", "de", "es", "et", "eu", "fa", "fr",
    "gu", "he", "hi", "hr", "hu", "hy", "id", "it", "jv", "ka", "kk", "km", "lt", "lv", "ml", "mr", "ms", "my", "ne", "nl",
    "pa", "pl", "ru", "si", "sk", "sl", "sr", "sw", "ta", "te", "th", "tl", "ur", "uz", "vi",
})
_ALIASES = {"zh": "zh-hans"}
_SPECIAL = re.compile(r"<\|[^<>|]{0,64}\|>")


ARCHITECTURE = "plamo3"
# GGUF value types: fixed sizes of the scalar ones; 8 is a string and 9 an array.
_GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_MAX_KEYS = 64


def gguf_architecture(path: Any) -> str | None:
    """general.architecture of a GGUF file, or None when it is not a GGUF file or names none among its first keys."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(24)
            if len(head) < 24 or head[:4] != b"GGUF":
                return None
            count = struct.unpack_from("<Q", head, 16)[0]

            def read(size: int) -> bytes:
                data = handle.read(size)
                if len(data) != size:
                    raise ValueError("truncated")
                return data

            def string(limit: int) -> bytes:
                size = struct.unpack("<Q", read(8))[0]
                if size > limit:
                    raise ValueError("string too long")
                return read(size)

            def skip(kind: int) -> None:
                if kind in _GGUF_SIZES:
                    handle.seek(_GGUF_SIZES[kind], 1)
                elif kind == 8:
                    handle.seek(struct.unpack("<Q", read(8))[0], 1)
                elif kind == 9:
                    inner, items = struct.unpack("<IQ", read(12))
                    if inner in _GGUF_SIZES:
                        handle.seek(_GGUF_SIZES[inner] * items, 1)
                    else:
                        for _ in range(items):
                            skip(inner)
                else:
                    raise ValueError("unknown value type")

            for _ in range(min(count, _GGUF_MAX_KEYS)):
                key = string(256)
                kind = struct.unpack("<I", read(4))[0]
                if key == b"general.architecture" and kind == 8:
                    return string(256).decode("utf-8", "replace")
                skip(kind)
    except (OSError, ValueError, struct.error):
        return None
    return None


def plamo_language(code: str | None) -> str | None:
    value = str(code or "").strip().lower()
    value = _ALIASES.get(value, value)
    return value if value in PLAMO_LANGUAGES else None


def caption_languages() -> list[str]:
    return sorted(code for code in PLAMO_LANGUAGES | set(_ALIASES) if re.fullmatch(r"[a-z]{2,3}", code))


def clean(text: Any) -> str:
    return _SPECIAL.sub(" ", str(text or "")).strip()


def plamo_prompt(line: str, target: str, history: Sequence[tuple[str, str]] = ()) -> str:
    """The beginning-of-sequence token is left out: llama.cpp adds it."""
    prompt = "<|plamo:tag|>dataset<|plamo:msg|>translation"
    for source, output in history:
        prompt += (f"<|plamo:tag|>input<|plamo:msg|>{source}"
                   f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target}<|plamo:msg|>{output}")
    return prompt + f"<|plamo:tag|>input<|plamo:msg|>{line}<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target}<|plamo:msg|>"


class PlamoTranslator:
    def __init__(
        self,
        *,
        resolve: Callable[[str], LaunchSpec],
        available: Callable[[], list[str]],
        allowed: Iterable[str],
        spawn: Callable[..., Any] = subprocess.Popen,
        opener: urllib.request.OpenerDirector | None = None,
        idle_s: float = IDLE_S,
        start_timeout: float = START_TIMEOUT_S,
        port: Callable[[], int] = _free_port,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._resolve = resolve
        self._available = available
        self._allowed = tuple(allowed)
        self._opener = opener or local_opener()
        self.server = LlamaServer(label=LABEL, task=Text("翻訳", "translation"), context=CONTEXT, extra=("-t", str(THREADS)), spawn=spawn,
                                  opener=self._opener, idle_s=idle_s, start_timeout=start_timeout, port=port, clock=clock)

    def models(self) -> list[str]:
        return self._available()

    def allows(self, model: str) -> bool:
        return model in self._allowed

    def warm(self, model: str) -> None:
        try:
            if self.allows(model):
                self.server.warm(self._resolve(model))
        except Exception:  # noqa: BLE001 - the first real request reports the failure
            pass

    def translate(self, model: str, text: str, *, target: str, history: Sequence[tuple[str, str]] = (),
                  timeout: float = 60.0) -> str:
        """A translation cut off at MAX_TOKENS returns "", like an untranslated line: it is never shown half done.
        """
        if not self.allows(model):
            raise LlmUnavailable(t("翻訳に許可されたモデルではありません。", "This model is not allowed for translation."))
        language = plamo_language(target)
        if language is None:
            raise LlmUnavailable(t(f"{LABEL} はこの言語（{target}）に翻訳できません。", f"{capitalize(LABEL)} cannot translate into this language ({target})."))
        line = clean(text)
        if not line or len(line) > MAX_LINE_CHARS:
            return ""
        pairs = [(source, output) for source, output in ((clean(a), clean(b)) for a, b in history)
                 if source and output and len(source) <= MAX_HISTORY_CHARS and len(output) <= MAX_HISTORY_CHARS][-2:]
        body = {"prompt": plamo_prompt(line, language, pairs), "n_predict": MAX_TOKENS, "temperature": 0,
                "stop": list(STOP), "cache_prompt": True}
        spec = self._resolve(model)
        with self.server.use(spec) as url:
            payload = self._post(url + "/completion", body, timeout)
        content = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(content, str):
            raise LlmUnavailable(t(f"{LABEL} の応答形式が不正です。", f"{capitalize(LABEL)} answered in an unexpected form."))
        if payload.get("stopped_limit"):
            return ""
        return clean(content)

    def state(self) -> dict[str, Any]:
        return self.server.state()

    def release(self) -> None:
        self.server.release()

    def close(self) -> None:
        self.server.close()

    def _post(self, url: str, body: dict[str, Any], timeout: float) -> Any:
        request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self._opener.open(request, timeout=timeout) as response:
                return json.loads(response.read(2_000_000).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LlmUnavailable(t(f"{LABEL} の翻訳に失敗しました（HTTP {exc.code}）。", f"{capitalize(LABEL)} failed to translate (HTTP {exc.code}).")) from exc
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise LlmUnavailable(t(f"{LABEL} の翻訳に失敗しました（{type(exc).__name__}）。", f"{capitalize(LABEL)} failed to translate ({type(exc).__name__}).")) from exc
