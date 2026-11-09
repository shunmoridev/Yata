from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .i18n import t


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


MAX_SAVED_GAPS = 500
TERMINAL = frozenset({"ended", "cancelled", "error"})


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class EvidenceSegment:
    id: str
    source_type: str
    text: str
    capture_start_ms: int
    capture_end_ms: int
    media_start_ms: int | None
    media_end_ms: int | None
    observed_at_utc: str
    source_url: str
    confidence: float | None = None
    is_final: bool = True
    language: str | None = None
    speaker: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Topic:
    id: str
    index: int
    status: str = "open"
    title: str = ""
    segment_ids: list[str] = field(default_factory=list)
    start_ms: int | None = None
    end_ms: int | None = None
    close_reason: str | None = None
    progress: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    summary_state: str = "none"
    delivery_state: str = "local_only"
    delivery_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Session:
    id: str
    source_url: str
    source_title: str
    source_kind: str
    active_input: str
    selected_model: str
    started_at_utc: str
    # Sessions saved before there was a choice of backend used Ollama.
    llm_backend: str = "ollama"
    asr_language: str = "auto"
    output_language: str = "ja"
    caption_language: str | None = None
    slack_delivery: bool = False
    summaries: bool = True
    translator: str | None = None
    title_source: str = "user"
    notes: str = ""
    # Session clock: capture times count listened time only, so a resumed session continues where it
    # stopped. clock_started_utc is when the current run began (None while stopped).
    clock_offset_ms: int = 0
    clock_started_utc: str | None = None
    updated_at_utc: str = ""
    status: str = "created"
    status_reason: str | None = None
    last_ingest_at_utc: str | None = None
    evidence: list[EvidenceSegment] = field(default_factory=list)
    topics: list[Topic] = field(default_factory=list)
    topic_cursor: int = 0
    topic_pending: list[str] = field(default_factory=list)
    topic_tentative: str | None = None
    topic_recheck_size: int = 0
    topic_split_next: bool = False
    topic_progress_size: int = 0
    analysis_retry_at: float = 0.0
    translation_cursor: int = 0
    translations: dict[str, str] = field(default_factory=dict)
    digest: dict[str, Any] | None = None
    digest_delivery_state: str = "local_only"
    gaps: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    next_event_id: int = 1
    text_bytes: int = 0
    audio_diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)
    live_caption: dict[str, Any] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def clock_ms(self, at: datetime | None = None) -> int:
        with self.lock:
            started = parse_utc(self.clock_started_utc)
            if started is None:
                return self.clock_offset_ms
            moment = at or datetime.now(timezone.utc)
            return self.clock_offset_ms + max(0, int((moment - started).total_seconds() * 1000))

    def pause_clock(self) -> None:
        """Stop the session clock; it never stops before the end of the last line, so a resume never overlaps it."""
        with self.lock:
            if self.clock_started_utc is not None:
                self.clock_offset_ms = self.clock_ms()
                self.clock_started_utc = None
            self.clock_offset_ms = max([self.clock_offset_ms] + [segment.capture_end_ms for segment in self.evidence])

    def display_title(self) -> str:
        with self.lock:
            return self.source_title or next((topic.title for topic in self.topics if topic.title), "")

    def summary_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "display_title": self.display_title(),
                "title_source": self.title_source,
                "source_url": self.source_url,
                "source_kind": self.source_kind,
                "status": self.status,
                "summaries": self.summaries,
                "translator": self.translator,
                "started_at_utc": self.started_at_utc,
                "updated_at_utc": self.updated_at_utc or self.started_at_utc,
                "duration_ms": self.clock_ms(),
                "topic_count": len(self.topics),
                "segment_count": len(self.evidence),
                "output_language": self.output_language,
            }

    def public_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "source_url": self.source_url,
                "source_title": self.source_title,
                "title_source": self.title_source,
                "display_title": self.display_title(),
                "source_kind": self.source_kind,
                "active_input": self.active_input,
                "selected_model": self.selected_model,
                "llm_backend": self.llm_backend,
                "started_at_utc": self.started_at_utc,
                "asr_language": self.asr_language,
                "output_language": self.output_language,
                "caption_language": self.caption_language,
                "slack_delivery": self.slack_delivery,
                "summaries": self.summaries,
                "translator": self.translator,
                "notes": self.notes,
                "clock_offset_ms": self.clock_offset_ms,
                "clock_started_utc": self.clock_started_utc,
                "updated_at_utc": self.updated_at_utc or self.started_at_utc,
                "status": self.status,
                "status_reason": self.status_reason,
                "last_ingest_at_utc": self.last_ingest_at_utc,
                "audio_diagnostics": dict(self.audio_diagnostics),
                "live_caption": dict(self.live_caption) if self.live_caption else None,
            }

    def emit(self, event_type: str, data: Any) -> dict[str, Any]:
        """Append an event; next_event_id doubles as the revision the archive saves against."""
        with self.lock:
            event = {"cursor": self.next_event_id, "type": event_type, "data": data}
            self.next_event_id += 1
            self.updated_at_utc = utc_now()
            self.events.append(event)
            del self.events[:-5000]
            return event

    def add_gap(self, data: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            event = self.emit("gap", data)
            self.gaps.append({**data, "cursor": event["cursor"]})
            del self.gaps[:-MAX_SAVED_GAPS]
            return event

    def events_after(self, cursor: int) -> list[dict[str, Any]]:
        with self.lock:
            return [event for event in self.events if event["cursor"] > cursor]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session": self.public_dict(),
                "evidence": [segment.as_dict() for segment in self.evidence],
                "topics": [topic.as_dict() for topic in self.topics],
                "translations": dict(self.translations),
                "digest": self.digest,
                "gaps": [dict(gap) for gap in self.gaps],
                "cursor": self.next_event_id - 1,
            }


class SessionStore:
    def __init__(self, max_sessions: int = 8) -> None:
        self.max_sessions = max_sessions
        self.on_evict: Callable[[Session], None] | None = None
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def add(self, session: Session) -> None:
        with self._lock:
            while len(self._sessions) >= self.max_sessions:
                expired = next((key for key, item in self._sessions.items() if item.status in TERMINAL), None)
                if expired is None:
                    raise SessionStoreFull(t("メモリ内セッションの上限に達しました。既存セッションを終了してください。",
                                            "Too many sessions are open. End one of them first."))
                if self.on_evict is not None:
                    self.on_evict(self._sessions[expired])
                del self._sessions[expired]
            self._sessions[session.id] = session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def sessions(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())


class SessionStoreFull(RuntimeError):
    pass

