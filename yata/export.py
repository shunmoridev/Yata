from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .i18n import Text, t
from .llm.base import LABELS
from .models import TERMINAL, EvidenceSegment, Session, Topic, parse_utc

AUTONYMS = {
    "ja": "日本語", "en": "English", "zh": "中文", "ko": "한국어", "fr": "Français", "es": "Español", "de": "Deutsch",
    "it": "Italiano", "pt": "Português", "ru": "Русский", "ar": "العربية", "hi": "हिन्दी", "id": "Bahasa Indonesia",
    "th": "ไทย", "vi": "Tiếng Việt", "tr": "Türkçe", "nl": "Nederlands", "pl": "Polski", "sv": "Svenska", "yue": "粵語",
}
SOURCE_NAMES = {"share": Text("タブ音声", "Tab audio")}
STATUS_NAMES = {"ended": Text("終了", "Ended"), "cancelled": Text("中断", "Stopped"), "error": Text("中断（エラー）", "Stopped (error)")}
LEFT_OUT = Text("（数値を根拠の発言で確かめられなかったため、この文は省きました）",
                "(Left out: its numbers could not be confirmed in the cited lines.)")
OVERVIEW_LEFT_OUT = Text("*（概要は数値を発言で確かめられなかったため省きました）*",
                         "*(Overview left out: its numbers could not be confirmed in the transcript.)*")
_MARKUP = re.compile(r"([\\`*_\[\]<>|#~])")
_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def _inline(text: Any) -> str:
    return _MARKUP.sub(r"\\\1", " ".join(str(text or "").split()))


def clock(ms: int | None) -> str:
    seconds = max(0, int(ms or 0) // 1000)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def _start(segment: EvidenceSegment) -> int:
    return segment.media_start_ms if segment.media_start_ms is not None else segment.capture_start_ms


def _local(value: str | None) -> datetime | None:
    moment = parse_utc(value)
    return moment.astimezone() if moment is not None else None


def _language(code: str | None) -> str:
    return AUTONYMS.get(code or "", code or "")


def _title(session: Session) -> str:
    title = session.display_title()
    if title:
        return title
    started = _local(session.started_at_utc)
    return str(SOURCE_NAMES.get(session.source_kind) or t("セッション", "Session")) + (f" {started:%m/%d %H:%M}" if started else "")


def filename(session: Session) -> str:
    with session.lock:
        started = _local(session.started_at_utc)
        name = _UNSAFE_NAME.sub(" ", _title(session)).strip().strip(".")[:60].strip() or "session"
    return (f"{started:%Y-%m-%d_%H%M}_" if started else "") + name + ".md"


def _tag(topic: Topic) -> str:
    return "T" + str(topic.index).zfill(2)


def _cited(ids: list[str], segments: dict[str, EvidenceSegment]) -> str:
    times = [clock(_start(segments[value])) for value in ids if value in segments]
    if not times:
        return ""
    return t("（" + "・".join(times) + "）", " (" + ", ".join(times) + ")")


def _summary(summary: dict[str, Any], segments: dict[str, EvidenceSegment]) -> list[str]:
    lines: list[str] = []
    if summary.get("overview"):
        lines += [_inline(summary["overview"]), ""]
    elif summary.get("overview_needs_review"):
        lines += [str(OVERVIEW_LEFT_OUT), ""]
    facts = summary.get("key_facts") or []
    if facts:
        lines += [f"- {_inline(fact.get('label'))}: **{_inline(fact.get('value'))}**{_cited([fact.get('evidence_id')], segments)}"
                  for fact in facts]
        lines.append("")
    points = summary.get("points") or []
    for point in points:
        text = f"*{LEFT_OUT}*" if point.get("needs_review") else _inline(point.get("text"))
        lines.append(f"- {text}{_cited(point.get('evidence_ids') or [], segments)}")
    if points:
        lines.append("")
    return lines


def _about(session: Session) -> list[str]:
    started = _local(session.started_at_utc)
    live = session.status not in TERMINAL
    status = STATUS_NAMES.get(session.status, STATUS_NAMES["ended"])
    lines = [t(f"- 開始: {started:%Y-%m-%d %H:%M}", f"- Started: {started:%Y-%m-%d %H:%M}") if started else None,
             t(f"- 長さ: {clock(session.clock_ms())}", f"- Length: {clock(session.clock_ms())}")
             + (t("（取り込み中に書き出しました）", " (exported while capturing)") if live else ""),
             t(f"- 状態: {status}", f"- Status: {status}") if not live else None,
             t(f"- 配信: <{session.source_url}>", f"- Stream: <{session.source_url}>") if session.source_url else None]
    spoken = t("自動判定", "detected automatically") if session.asr_language == "auto" else _language(session.asr_language)
    languages = t(f"- 言語: 配信 {spoken}", f"- Languages: stream {spoken}")
    if session.summaries:
        languages += t(f"・要約 {_language(session.output_language)}", f", summaries {_language(session.output_language)}")
    if session.caption_language:
        languages += t(f"・字幕の翻訳 {_language(session.caption_language)}", f", caption translation {_language(session.caption_language)}")
    lines.append(languages)
    if session.summaries:
        backend = LABELS.get(session.llm_backend, session.llm_backend)
        lines.append(t(f"- 要約のモデル: {session.selected_model}（{backend}）", f"- Summary model: {session.selected_model} ({backend})"))
    else:
        lines.append(t("- 要約: なし（" + ("字幕と翻訳だけを記録" if session.translator else "字幕だけを記録") + "）",
                       "- Summaries: none (" + ("captions and translations only" if session.translator else "captions only") + ")"))
    exported = datetime.now().astimezone()
    lines.append(t(f"- 書き出し: {exported:%Y-%m-%d %H:%M}（Yata）", f"- Exported: {exported:%Y-%m-%d %H:%M} (Yata)"))
    return [line for line in lines if line]


def _gap_line(gap: dict[str, Any]) -> str:
    span = f"{clock(gap['capture_start_ms'])} – {clock(gap['capture_end_ms'])}"
    return t(f"- *{span} は音声が届いていません*", f"- *{span}: no audio arrived*")


def session_markdown(session: Session) -> str:
    with session.lock:
        segments = {segment.id: segment for segment in session.evidence}
        topics = sorted(session.topics, key=lambda topic: topic.index)
        speakers = any(segment.speaker for segment in session.evidence)
        lines = [f"# {_inline(_title(session))}", ""] + _about(session) + [""]
        parts = [t("字幕", "Captions")] + ([t("話者ラベル", "speaker labels")] if speakers else []) \
            + ([t("訳", "translations")] if session.translations else []) + ([t("要約", "summaries")] if session.summaries else [])
        notice = t("・".join(parts) + "は自動で作ったもので、誤りを含むことがあります。",
                   (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0])
                   + " were made automatically and may contain errors.")
        lines += [f"> {notice}", ""]

        if session.notes.strip():
            lines += [t("## メモ", "## Memo"), "", session.notes.strip(), ""]

        digest = session.digest
        if digest:
            by_id = {topic.id: topic for topic in topics}
            lines += [t("## 全体のまとめ", "## Debrief"), ""]
            if digest.get("overview"):
                lines += [_inline(digest["overview"]), ""]
            elif digest.get("overview_needs_review"):
                lines += [str(OVERVIEW_LEFT_OUT), ""]
            for item in digest.get("highlights") or []:
                tags = t("・", ", ").join(_tag(by_id[value]) for value in item.get("topic_ids") or [] if value in by_id)
                text = f"*{LEFT_OUT}*" if item.get("needs_review") else _inline(item.get("text"))
                lines.append(f"- {text}" + (t(f"（{tags}）", f" ({tags})") if tags else ""))
            lines.append("")

        if topics:
            lines += [t("## 話題", "## Topics"), ""]
            for topic in topics:
                live = topic.status == "open"
                lines.append(f"### {_tag(topic)} {_inline(topic.title) or t('（題名はまだありません）', '(no title yet)')}")
                lines.append("")
                count = len(topic.segment_ids)
                notes = [f"{clock(topic.start_ms)} – {clock(topic.end_ms)}", t(f"{count} 発言", f"{count} line" + ("" if count == 1 else "s"))]
                if live:
                    notes.append(t("進行中（暫定の要約）", "in progress (provisional summary)"))
                if topic.close_reason == "max_length":
                    notes.append(t("長いため区切りました", "split because it ran long"))
                lines += ["*" + " · ".join(notes) + "*", ""]
                summary = topic.progress if live else topic.summary
                if summary:
                    lines += _summary(summary, segments)
                elif topic.summary_state == "error":
                    lines += [t("*（要約できませんでした）*", "*(The summary failed.)*"), ""]
                elif not live:
                    lines += [t("*（要約はまだありません）*", "*(No summary yet.)*"), ""]

        lines += [t("## 文字起こし", "## Transcript"), ""]
        if not session.evidence:
            lines += [t("*（文字起こしはまだありません）*", "*(No transcript yet.)*"), ""]
        topic_of = {segment_id: topic for topic in topics for segment_id in topic.segment_ids}
        gaps = sorted((gap for gap in session.gaps if gap.get("capture_start_ms") is not None and gap.get("capture_end_ms") is not None),
                      key=lambda gap: gap["capture_start_ms"])
        current: Topic | None = None
        for segment in session.evidence:
            while gaps and gaps[0]["capture_start_ms"] <= segment.capture_start_ms:
                gap = gaps.pop(0)
                lines.append(_gap_line(gap))
            topic = topic_of.get(segment.id)
            if topic is not None and topic is not current:
                current = topic
                lines += ([""] if lines[-1] else []) + [f"### {_tag(topic)} {_inline(topic.title)}".rstrip(), ""]
            speaker = f" **{segment.speaker}**" if segment.speaker else ""
            lines.append(f"- {clock(_start(segment))}{speaker} {_inline(segment.text)}")
            translation = session.translations.get(segment.id)
            if translation:
                lines.append(t(f"  - 訳: {_inline(translation)}", f"  - Translation: {_inline(translation)}"))
        for gap in gaps:
            lines.append(_gap_line(gap))
    return "\n".join(lines).rstrip() + "\n"
