from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from .analysis import AnalysisWorker
from .i18n import t
from .archive import SESSION_ID, SessionArchive
from .models import TERMINAL, EvidenceSegment, Session, SessionStore, SessionStoreFull, utc_now
from .ollama import SUMMARY_MODEL_ALLOWLIST
from .slack import SlackSender, format_digest, format_topic
from .topics import TopicPipeline


MAX_SEGMENTS = 10_000
MAX_SESSION_TEXT_BYTES = 2 * 1024 * 1024
MAX_SEGMENT_CHARS = 12_000
MAX_NOTES_CHARS = 20_000


@dataclass(frozen=True)
class SourceKind:
    active_input: str
    evidence_type: str
    typed_title: bool


SOURCES = {"share": SourceKind(active_input="shared_audio", evidence_type="asr", typed_title=False)}
SOURCE_FOR_INPUT = {kind.active_input: kind.evidence_type for kind in SOURCES.values()}
LIVE = {"created", "starting_audio", "capturing_audio", "reconnecting", "draining"}
LANGUAGE_CODE = re.compile(r"[a-z]{2,3}")
AUTOSAVE_INTERVAL_S = 3.0


class SessionError(ValueError):
    pass


def _valid_url(value: str) -> bool:
    if not value:
        return True
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and len(value) <= 2048


class SessionManager:
    def __init__(self, pipeline: TopicPipeline, store: SessionStore | None = None, *, background: bool = True,
                 sender: SlackSender | None = None, archive: SessionArchive | None = None,
                 llm_allows: Callable[[str, str], bool] | None = None,
                 translator_allows: Callable[[str, str], bool] | None = None) -> None:
        self.pipeline = pipeline
        self.llm_allows = llm_allows or (lambda backend, model: backend == "ollama" and model in SUMMARY_MODEL_ALLOWLIST)
        self.translator_allows = translator_allows or (lambda model, language: False)
        self.store = store or SessionStore()
        self.sender = sender or SlackSender("")
        self.archive = archive
        if archive is not None:
            self.store.on_evict = self.save
        self.background = background
        self._workers: dict[str, AnalysisWorker] = {}
        self._lock = threading.RLock()
        self._save_failed: set[str] = set()
        self._closed = threading.Event()
        self._autosave: threading.Thread | None = None

    def create(
        self,
        *,
        source_url: str,
        source_title: str,
        source_kind: str,
        selected_model: str,
        asr_language: str = "auto",
        output_language: str = "ja",
        caption_language: str | None = None,
        slack_delivery: bool = False,
        summaries: bool = True,
        llm_backend: str = "ollama",
        translator: str | None = None,
    ) -> Session:
        kind = SOURCES.get(source_kind)
        if kind is None:
            raise SessionError(t("入力方法が不正です。", "Unknown input source."))
        if not _valid_url(source_url):
            raise SessionError(t("出典 URL は http または https にしてください。", "The source URL must start with http or https."))
        if len(source_title) > 300 or len(selected_model) > 200:
            raise SessionError(t("セッション設定が長すぎます。", "The session settings are too long."))
        if source_title.strip() and not kind.typed_title:
            raise SessionError(t("この入力方法では、セッション名は内容から自動で付けます。", "With this source, the session is named from its content."))
        if summaries and not self.llm_allows(llm_backend, selected_model):
            raise SessionError(t("要約モデルが許可リストにありません。", "The summary model is not on the allowlist."))
        if not summaries:
            if selected_model:
                raise SessionError(t("字幕のみのセッションには要約モデルを指定しません。", "A captions-only session takes no summary model."))
            if caption_language is not None and translator is None:
                raise SessionError(t("要約なしで字幕を翻訳するには、翻訳モデル（PLaMo）が必要です。「この PC の準備」で準備できます。", "Translating captions without summaries needs the translation model (PLaMo). Get it under \"Set up this PC\"."))
            if slack_delivery:
                raise SessionError(t("字幕のみのセッションには Slack に送る要約がありません。", "A captions-only session has no summaries to send to Slack."))
        if translator is not None:
            if caption_language is None:
                raise SessionError(t("翻訳先の言語がないセッションには翻訳モデルを使いません。", "A session without a translation language uses no translation model."))
            if not self.translator_allows(translator, caption_language):
                raise SessionError(t("この翻訳モデルは使えないか、選んだ言語に翻訳できません。", "This translation model is unavailable or cannot translate into the chosen language."))
        if asr_language != "auto" and not LANGUAGE_CODE.fullmatch(asr_language):
            raise SessionError(t("配信の言語が不正です。", "Unknown stream language."))
        if not LANGUAGE_CODE.fullmatch(output_language):
            raise SessionError(t("要約の言語が不正です。", "Unknown summary language."))
        if caption_language is not None and not LANGUAGE_CODE.fullmatch(caption_language):
            raise SessionError(t("字幕の翻訳先の言語が不正です。", "Unknown caption translation language."))
        if slack_delivery and not self.sender.enabled:
            raise SessionError(t("Slack 送信は設定されていません。", "Sending to Slack is not set up. ") + self.sender.readiness()["reason"])
        started = utc_now()
        title = source_title.strip()
        session = Session(
            id=str(uuid.uuid4()),
            source_url=source_url,
            source_title=title,
            source_kind=source_kind,
            active_input=kind.active_input,
            selected_model=selected_model,
            llm_backend=llm_backend if summaries else "",
            started_at_utc=started,
            asr_language=asr_language,
            output_language=output_language,
            caption_language=caption_language,
            slack_delivery=bool(slack_delivery),
            summaries=bool(summaries),
            translator=translator,
            title_source="user" if title else "auto",
            clock_started_utc=started,
            status="created",
        )
        try:
            self.store.add(session)
        except SessionStoreFull as exc:
            raise SessionError(str(exc)) from exc
        session.emit("session_state", session.public_dict())
        self._start_worker(session)
        self.save(session)
        return session

    def _start_worker(self, session: Session) -> None:
        if not self.background:
            return
        # A captions-only session has no analysis to run, but it is saved like any other.
        if session.summaries or session.translator:
            self._workers[session.id] = AnalysisWorker(self.pipeline, session, after_step=lambda: self.deliver(session))
        self._start_autosave()

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self.store.get(session_id)
            if session is None and self.archive is not None and SESSION_ID.fullmatch(session_id):
                session = self.archive.load(session_id)
                if session is not None:
                    try:
                        self.store.add(session)
                    except SessionStoreFull as exc:
                        raise SessionError(str(exc)) from exc
        if session is None:
            raise SessionError(t("セッションが見つかりません。", "Session not found."))
        return session

    def list(self) -> list[dict[str, Any]]:
        rows = {item["id"]: item for item in (self.archive.summaries() if self.archive is not None else [])}
        rows.update({session.id: session.summary_dict() for session in self.store.sessions()})
        return sorted(rows.values(), key=lambda item: item.get("updated_at_utc") or "", reverse=True)

    def delete(self, session: Session) -> None:
        with session.lock:
            if session.status not in TERMINAL:
                raise SessionError(t("取り込み中のセッションは削除できません。先に終了してください。", "A session that is still capturing cannot be deleted. End it first."))
        self._workers.pop(session.id, None)
        self.store.remove(session.id)
        self._save_failed.discard(session.id)
        if self.archive is not None:
            self.archive.delete(session.id)

    def resume(self, session: Session, *, source_url: str = "") -> None:
        if not _valid_url(source_url):
            raise SessionError(t("出典 URL は http または https にしてください。", "The source URL must start with http or https."))
        with session.lock:
            if session.status not in TERMINAL:
                raise SessionError(t("終了したセッションだけを再開できます。", "Only an ended session can be continued."))
            if source_url:
                session.source_url = source_url
            session.status = "created"
            session.status_reason = None
            session.analysis_retry_at = 0.0
            session.clock_started_utc = utc_now()
            session.emit("session_state", session.public_dict())
        self._start_worker(session)
        self.save(session)

    def set_status(self, session: Session, status: str, reason: str | None = None) -> None:
        if status not in LIVE | TERMINAL:
            raise SessionError(t("セッション状態が不正です。", "Unknown session state."))
        with session.lock:
            if session.status in TERMINAL:
                return
            if status in TERMINAL:
                session.pause_clock()
            session.status = status
            session.status_reason = reason[:500] if reason else None
            session.emit("session_state", session.public_dict())

    def save(self, session: Session) -> None:
        if self.archive is None:
            return
        try:
            self.archive.save(session)
        except OSError as exc:
            if session.id not in self._save_failed:
                self._save_failed.add(session.id)
                session.emit("analysis_error", {"task": t("保存", "Saving"), "message": t(f"セッションを保存できません: {type(exc).__name__}",
                                                                                       f"Cannot save the session: {type(exc).__name__}")})
            return
        self._save_failed.discard(session.id)

    def save_all(self) -> None:
        for session in self.store.sessions():
            self.save(session)

    def _start_autosave(self) -> None:
        with self._lock:
            if self.archive is None or self._autosave is not None:
                return
            self._autosave = threading.Thread(target=self._autosave_loop, name="yata-autosave", daemon=True)
            self._autosave.start()

    def _autosave_loop(self) -> None:
        while not self._closed.wait(AUTOSAVE_INTERVAL_S):
            self.save_all()

    def close(self) -> None:
        self._closed.set()
        for worker in list(self._workers.values()):
            worker.close()
        self.save_all()

    def ingest(
        self,
        session: Session,
        values: list[dict[str, Any]],
        *,
        trusted_source_type: str | None = None,
    ) -> tuple[list[str], list[dict[str, str]]]:
        accepted: list[str] = []
        rejected: list[dict[str, str]] = []
        if session.status not in LIVE:
            raise SessionError(t("入力中のセッションではありません。", "The session is not taking input."))
        expected_type = trusted_source_type or SOURCE_FOR_INPUT.get(session.active_input)
        if expected_type != "asr":
            raise SessionError(t("現在の入力方法からこの字幕を受け取れません。", "The current source cannot take these captions."))
        for index, value in enumerate(values[:500]):
            try:
                segment = self._make_segment(session, value, expected_type)
            except SessionError as exc:
                rejected.append({"index": str(index), "reason": str(exc)})
                continue
            with session.lock:
                duplicate = any(
                    item.source_type == segment.source_type
                    and item.text == segment.text
                    and (
                        item.media_start_ms == segment.media_start_ms
                        and item.media_end_ms == segment.media_end_ms
                        if segment.media_start_ms is not None
                        else item.media_start_ms is None
                        and abs(item.capture_start_ms - segment.capture_start_ms) <= 2_000
                    )
                    for item in session.evidence[-300:]
                )
                byte_count = len(segment.text.encode("utf-8"))
                if duplicate:
                    continue
                if len(session.evidence) >= MAX_SEGMENTS or session.text_bytes + byte_count > MAX_SESSION_TEXT_BYTES:
                    rejected.append({"index": str(index), "reason": t("セッションの字幕容量上限に達しました。", "The session holds no more captions.")})
                    continue
                session.evidence.append(segment)
                session.text_bytes += byte_count
                session.last_ingest_at_utc = segment.observed_at_utc
                session.emit("evidence_segment", segment.as_dict())
                accepted.append(segment.id)
        if accepted:
            self.notify(session)
        return accepted, rejected

    def _make_segment(self, session: Session, value: dict[str, Any], source_type: str) -> EvidenceSegment:
        if not isinstance(value, dict):
            raise SessionError(t("字幕データの形式が不正です。", "Invalid caption data."))
        text = str(value.get("text", "")).strip()
        if not text or len(text) > MAX_SEGMENT_CHARS:
            raise SessionError(t("字幕は 1〜12,000 文字で入力してください。", "Captions must be 1 to 12,000 characters."))
        media_start = value.get("media_start_ms")
        media_end = value.get("media_end_ms")
        for number in (media_start, media_end):
            if number is not None and (not isinstance(number, int) or number < 0 or number > 7 * 24 * 60 * 60 * 1000):
                raise SessionError(t("字幕の時間が範囲外です。", "The caption time is out of range."))
        if media_start is not None and media_end is not None and media_end < media_start:
            raise SessionError(t("字幕の終了時間が開始時間より前です。", "The caption ends before it starts."))
        confidence = value.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise SessionError(t("信頼度の値が不正です。", "Invalid confidence value."))
        language = value.get("language")
        if language is not None and not (isinstance(language, str) and re.fullmatch(r"[a-z]{2,3}(-[A-Za-z]{2,4})?", language)):
            language = None
        speaker = value.get("speaker")
        if speaker is not None and not (isinstance(speaker, str) and re.fullmatch(r"S\d{1,2}", speaker)):
            speaker = None
        source_url = str(value.get("source_url", session.source_url))[:2048]
        if not _valid_url(source_url):
            source_url = session.source_url
        elapsed_ms = session.clock_ms()
        capture_start = value.get("capture_start_ms", elapsed_ms)
        capture_end = value.get("capture_end_ms", elapsed_ms)
        if not isinstance(capture_start, int) or not isinstance(capture_end, int) or capture_start < 0 or capture_end < capture_start:
            raise SessionError(t("字幕の取り込み時間が不正です。", "Invalid caption capture time."))
        return EvidenceSegment(
            id=str(uuid.uuid4()),
            source_type=source_type,
            text=text,
            capture_start_ms=capture_start,
            capture_end_ms=capture_end,
            media_start_ms=media_start,
            media_end_ms=media_end,
            observed_at_utc=str(value.get("observed_at_utc") or utc_now())[:40],
            source_url=source_url,
            confidence=float(confidence) if confidence is not None else None,
            is_final=bool(value.get("is_final", True)),
            language=language,
            speaker=speaker,
        )

    def notify(self, session: Session) -> None:
        worker = self._workers.get(session.id)
        if worker is not None:
            worker.notify()

    def _exclusive(self, session: Session, work: Any) -> Any:
        worker = self._workers.get(session.id)
        return worker.exclusive(work) if worker is not None else work()

    def analyze(self, session: Session) -> None:
        if not session.summaries and not session.translator:
            return

        def run() -> None:
            while self.pipeline.step(session):
                pass
        self._exclusive(session, run)

    def summarize(self, session: Session) -> list[dict[str, Any]]:
        if session.status not in LIVE:
            raise SessionError(t("終了したセッションは更新できません。", "An ended session cannot be updated."))
        if not session.summaries:
            raise SessionError(t("字幕のみのセッションには話題の整理がありません。", "A captions-only session has no topics to organize."))
        self._exclusive(session, lambda: self.pipeline.refresh(session))
        with session.lock:
            return [topic.as_dict() for topic in session.topics]

    def stop(self, session: Session) -> None:
        if session.status in {"ended", "cancelled"}:
            return
        self.set_status(session, "draining")
        worker = self._workers.pop(session.id, None)
        if worker is not None:
            worker.close()
        previous_digest = session.digest
        try:
            if not session.summaries and not session.translator:
                pass  # captions only: nothing to summarize or translate at the end
            elif worker is not None:
                worker.exclusive(lambda: self.pipeline.finalize(session))
            else:
                self.pipeline.finalize(session)
            with session.lock:
                # A resumed session writes a new digest at its next stop; that one is sent as well.
                if session.digest is not previous_digest and session.digest_delivery_state != "pending":
                    session.digest_delivery_state = "local_only"
            self.deliver(session)
        finally:
            if session.status == "draining":
                self.set_status(session, "ended")
            self.save(session)

    def deliver(self, session: Session) -> None:
        if not session.slack_delivery or not self.sender.enabled:
            return
        with session.lock:
            if session.status == "cancelled":
                return
            topics = [topic for topic in session.topics if topic.summary_state == "ready" and topic.delivery_state == "local_only"]
            for topic in topics:
                topic.delivery_state = "pending"
            messages = [(topic, format_topic(session.source_title, topic.as_dict())) for topic in topics]
            digest = None
            if session.digest and session.digest_delivery_state == "local_only":
                session.digest_delivery_state = "pending"
                digest = format_digest(session.source_title, session.digest)
        for topic, text in messages:
            result = self.sender.deliver({"text": text})
            with session.lock:
                topic.delivery_state = result.state
                topic.delivery_reason = result.reason
                session.emit("topic", topic.as_dict())
        if digest is not None:
            result = self.sender.deliver({"text": digest})
            with session.lock:
                session.digest_delivery_state = result.state
                session.emit("digest_delivery", {"state": result.state, "reason": result.reason})

    def redeliver(self, session: Session, topic_id: str) -> None:
        with session.lock:
            topic = next((item for item in session.topics if item.id == topic_id), None)
            if topic is None:
                raise SessionError(t("話題が見つかりません。", "Topic not found."))
            if topic.delivery_state != "failed":
                raise SessionError(t("送信に失敗した話題だけを再送できます。", "Only a topic that failed to send can be sent again."))
            topic.delivery_state = "local_only"
            if session.digest_delivery_state == "failed":
                session.digest_delivery_state = "local_only"
        self.deliver(session)
        self.save(session)

    def cancel(self, session: Session) -> None:
        worker = self._workers.pop(session.id, None)
        if worker is not None:
            worker.close()
        with session.lock:
            if session.status in {"ended", "cancelled"}:
                return
            session.topic_pending.clear()
            session.pause_clock()
            session.status = "cancelled"
            session.status_reason = t("利用者がキャンセルしました。", "Cancelled by the user.")
            session.emit("session_state", session.public_dict())
        self.save(session)

    def set_notes(self, session: Session, notes: Any) -> None:
        if not isinstance(notes, str):
            raise SessionError(t("メモが不正です。", "Invalid memo."))
        if len(notes) > MAX_NOTES_CHARS:
            raise SessionError(t(f"メモは {MAX_NOTES_CHARS:,} 文字までです。", f"The memo can be up to {MAX_NOTES_CHARS:,} characters."))
        with session.lock:
            if session.notes == notes:
                return
            session.notes = notes
            # The event moves the session's revision, which is what the autosave writes against.
            session.emit("session_notes", {"chars": len(notes)})

    def record_gap(self, session: Session, reason: str, start_ms: int | None = None, end_ms: int | None = None) -> None:
        if session.status in TERMINAL:
            return
        session.add_gap({"reason": reason[:300], "capture_start_ms": start_ms, "capture_end_ms": end_ms, "observed_at_utc": utc_now()})

