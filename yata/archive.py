from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from .models import TERMINAL, EvidenceSegment, Session, Topic, parse_utc
from .i18n import Text

FORMAT = 1
SESSION_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SESSION_FIELDS = (
    "id", "source_url", "source_title", "title_source", "source_kind", "active_input", "selected_model", "llm_backend",
    "started_at_utc", "updated_at_utc", "asr_language", "output_language", "caption_language", "slack_delivery", "summaries",
    "translator", "notes", "clock_offset_ms", "clock_started_utc", "status", "status_reason", "last_ingest_at_utc",
    "topic_cursor", "topic_pending", "topic_tentative", "topic_recheck_size", "topic_split_next", "topic_progress_size",
    "translation_cursor", "digest", "digest_delivery_state", "text_bytes", "next_event_id",
)
_SEGMENT_FIELDS = {item.name for item in fields(EvidenceSegment)}
_TOPIC_FIELDS = {item.name for item in fields(Topic)}
INTERRUPTED = Text("サーバーの停止で中断しました。続きを取り込めます。", "Stopped when the server stopped. You can continue it.")
UNKNOWN_DELIVERY = Text("送信中にサーバーが停止したため、届いたか分かりません。",
                        "The server stopped while sending, so it is unknown whether it arrived.")


def default_data_dir(environ: dict[str, str] | None = None, platform: str = sys.platform) -> Path:
    env = os.environ if environ is None else environ
    configured = env.get("YATA_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    base = env.get("LOCALAPPDATA", "").strip()
    if base:
        return Path(base) / "Yata"
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Yata"
    xdg = env.get("XDG_DATA_HOME", "").strip()
    return (Path(xdg) if xdg else Path.home() / ".local" / "share") / "Yata"


def to_record(session: Session) -> dict[str, Any]:
    with session.lock:
        return {
            "format": FORMAT,
            "session": {name: getattr(session, name) for name in _SESSION_FIELDS},
            "evidence": [segment.as_dict() for segment in session.evidence],
            "topics": [topic.as_dict() for topic in session.topics],
            "translations": dict(session.translations),
            "gaps": [dict(gap) for gap in session.gaps],
        }


def from_record(record: dict[str, Any]) -> Session:
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise ValueError("unsupported session file")
    values = record["session"]
    if not isinstance(values, dict) or not SESSION_ID.fullmatch(str(values.get("id", ""))):
        raise ValueError("invalid session id")
    session = Session(**{name: values[name] for name in _SESSION_FIELDS if name in values})
    session.evidence = [EvidenceSegment(**{key: value for key, value in item.items() if key in _SEGMENT_FIELDS})
                        for item in record.get("evidence", [])]
    session.topics = [Topic(**{key: value for key, value in item.items() if key in _TOPIC_FIELDS})
                      for item in record.get("topics", [])]
    translations = record.get("translations", {})
    session.translations = {str(key): str(value) for key, value in translations.items()} if isinstance(translations, dict) else {}
    session.gaps = [dict(item) for item in record.get("gaps", []) if isinstance(item, dict)]
    for name in ("clock_offset_ms", "next_event_id", "topic_cursor", "translation_cursor", "text_bytes"):
        value = getattr(session, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid {name}")
    return session


def settle(session: Session) -> bool:
    before = (session.status, session.clock_offset_ms, session.clock_started_utc, session.digest_delivery_state,
              [topic.delivery_state for topic in session.topics])
    if session.status not in TERMINAL:
        started = parse_utc(session.clock_started_utc)
        if started is not None:
            last_seen = parse_utc(session.updated_at_utc) or started
            session.clock_offset_ms = session.clock_ms(last_seen)
            session.clock_started_utc = None
        session.status, session.status_reason = "cancelled", str(INTERRUPTED)
    session.pause_clock()
    # A send that was in flight has an unknown outcome; show it as failed so the user can decide to resend.
    for topic in session.topics:
        if topic.delivery_state == "pending":
            topic.delivery_state, topic.delivery_reason = "failed", str(UNKNOWN_DELIVERY)
    if session.digest_delivery_state == "pending":
        session.digest_delivery_state = "failed"
    return before != (session.status, session.clock_offset_ms, session.clock_started_utc, session.digest_delivery_state,
                      [topic.delivery_state for topic in session.topics])


class SessionArchive:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.directory = self.root / "sessions"
        self._lock = threading.RLock()
        self._saved: dict[str, int] = {}
        # Deleted ids are never written again, even by an autosave that picked the session up just before.
        self._deleted: set[str] = set()
        self._summaries: dict[str, dict[str, Any]] | None = None

    def _path(self, session_id: str, suffix: str) -> Path:
        if not SESSION_ID.fullmatch(session_id):
            raise ValueError("invalid session id")
        return self.directory / f"{session_id}{suffix}"

    def summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._summaries is None:
                self._summaries = {}
                for path in sorted(self.directory.glob("*.summary.json")) if self.directory.is_dir() else []:
                    try:
                        summary = json.loads(path.read_text(encoding="utf-8"))
                        session_id = str(summary.get("id", ""))
                        if path.name == f"{session_id}.summary.json" and self._path(session_id, ".json").is_file():
                            self._summaries[session_id] = summary
                    except (OSError, ValueError, AttributeError):
                        continue
            return [dict(item) for item in self._summaries.values()]

    def dirty(self, session: Session) -> bool:
        with self._lock:
            return self._saved.get(session.id) != session.next_event_id

    def save(self, session: Session) -> bool:
        with self._lock:
            with session.lock:
                revision = session.next_event_id
                if self._saved.get(session.id) == revision or session.id in self._deleted:
                    return False
                record = to_record(session)
                summary = session.summary_dict()
            self.directory.mkdir(parents=True, exist_ok=True)
            _replace(self._path(session.id, ".json"), json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            _replace(self._path(session.id, ".summary.json"), json.dumps(summary, ensure_ascii=False))
            self._saved[session.id] = revision
            if self._summaries is not None:
                self._summaries[session.id] = summary
            return True

    def load(self, session_id: str) -> Session | None:
        with self._lock:
            try:
                session = from_record(json.loads(self._path(session_id, ".json").read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                return None
            if session.id != session_id:
                return None
            # A settled (interrupted) session is written back on the next save; an unchanged one is not.
            self._saved[session.id] = -1 if settle(session) else session.next_event_id
            return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            for suffix in (".summary.json", ".json"):
                try:
                    self._path(session_id, suffix).unlink()
                except FileNotFoundError:
                    pass
            self._saved.pop(session_id, None)
            self._deleted.add(session_id)
            if self._summaries is not None:
                self._summaries.pop(session_id, None)


def _replace(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            # Windows refuses the swap while another process (an indexer, a virus scan) has the file open.
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))

