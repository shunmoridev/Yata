from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

from .i18n import t

SLACK_HOSTS = frozenset({"hooks.slack.com"})


@dataclass(frozen=True)
class DeliveryResult:
    state: str
    reason: str


class SlackSender:
    def __init__(
        self,
        webhook_url: str | None = None,
        *,
        allowed_hosts: frozenset[str] = SLACK_HOSTS,
        require_https: bool = True,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout: float = 10.0,
        switched_off: bool = False,
    ) -> None:
        self._switched_off = switched_off
        url = "" if switched_off else webhook_url if webhook_url is not None else os.environ.get("YATA_SLACK_WEBHOOK_URL", "")
        self._url = url.strip()
        self._allowed_hosts = allowed_hosts
        self._require_https = require_https
        self._opener = opener
        self._timeout = timeout

    def __repr__(self) -> str:  # keep the secret URL out of tracebacks and logs
        return f"SlackSender(enabled={self.enabled})"

    @property
    def enabled(self) -> bool:
        return self.readiness()["enabled"]

    def readiness(self) -> dict[str, Any]:
        if self._switched_off:
            return {"enabled": False, "reason": t("Slack 連携は機能トグルでオフです（YATA_FEATURE_SLACK）。",
                                                   "Slack is switched off by its feature toggle (YATA_FEATURE_SLACK).")}
        if not self._url:
            return {"enabled": False, "reason": t("Slack の Incoming Webhook が未設定です（YATA_SLACK_WEBHOOK_URL）。",
                                                   "No Slack incoming webhook is set (YATA_SLACK_WEBHOOK_URL).")}
        parts = urlsplit(self._url)
        if (self._require_https and parts.scheme != "https") or parts.scheme not in {"https", "http"}:
            return {"enabled": False, "reason": t("Slack の Webhook は https の URL にしてください。", "The Slack webhook must be an https URL.")}
        if parts.hostname not in self._allowed_hosts:
            return {"enabled": False, "reason": t("Slack の Incoming Webhook（hooks.slack.com）以外には送信しません。",
                                                   "Yata sends only to Slack incoming webhooks (hooks.slack.com).")}
        return {"enabled": True, "reason": t("セッションで選んだ場合だけ、検証済みの話題要約を送ります。",
                                              "Sends verified topic summaries only for sessions that choose it.")}

    def deliver(self, message: dict[str, Any]) -> DeliveryResult:
        if not self.enabled:
            return DeliveryResult("local_only", t("Slack 送信は無効です。", "Sending to Slack is off."))
        body = json.dumps({"text": str(message.get("text", ""))[:39_000]}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self._url, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with self._opener(request, timeout=self._timeout) as response:
                status = getattr(response, "status", 200)
                response.read(1_000)
        except urllib.error.HTTPError as exc:
            return DeliveryResult("failed", t(f"Slack が HTTP {exc.code} を返しました。", f"Slack returned HTTP {exc.code}."))
        except (OSError, urllib.error.URLError, ValueError) as exc:
            return DeliveryResult("failed", t(f"Slack に送信できませんでした（{type(exc).__name__}）。", f"Could not send to Slack ({type(exc).__name__})."))
        if not 200 <= status < 300:
            return DeliveryResult("failed", t(f"Slack が HTTP {status} を返しました。", f"Slack returned HTTP {status}."))
        return DeliveryResult("sent", t("Slack に送信しました。", "Sent to Slack."))


def _clock(ms: int | None) -> str:
    seconds = max(0, (ms or 0) // 1000)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def format_topic(stream_title: str, topic: dict[str, Any]) -> str:
    summary = topic.get("summary") or {}
    name = t(f"話題 {topic.get('index')}", f"Topic {topic.get('index')}")
    title = summary.get("title") or topic.get("title") or name
    lines = [f"*{stream_title + ' · ' if stream_title else ''}{name}: {title}*"
             f"  ({_clock(topic.get('start_ms'))}–{_clock(topic.get('end_ms'))})"]
    if summary.get("overview"):
        lines.append(summary["overview"])
    lines += [f"• {point['text']}" for point in summary.get("points", []) if point.get("text") and not point.get("needs_review")]
    facts = [f"{fact['label']}: {fact['value']}" for fact in summary.get("key_facts", [])]
    if facts:
        lines.append(t("重要事実: ", "Key facts: ") + " / ".join(facts))
    lines.append(t("_自動の文字起こしと要約です。誤りを含む場合があります。_", "_Automatic transcript and summary; it may contain errors._"))
    return "\n".join(lines)


def format_digest(stream_title: str, digest: dict[str, Any]) -> str:
    lines = [f"*{stream_title + ' · ' if stream_title else ''}" + t("全体のまとめ", "Debrief") + "*"]
    if digest.get("overview"):
        lines.append(digest["overview"])
    lines += [f"• {item['text']}" for item in digest.get("highlights", []) if item.get("text") and not item.get("needs_review")]
    lines.append(t("_自動の文字起こしと要約です。誤りを含む場合があります。_", "_Automatic transcript and summary; it may contain errors._"))
    return "\n".join(lines)
