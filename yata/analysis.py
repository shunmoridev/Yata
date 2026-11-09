from __future__ import annotations

import threading
from typing import Any, Callable

from .i18n import t

from .models import Session
from .topics import TopicPipeline


class AnalysisWorker:
    def __init__(self, pipeline: TopicPipeline, session: Session, interval_s: float = 2.0,
                 after_step: Callable[[], None] | None = None) -> None:
        self.pipeline = pipeline
        self.session = session
        self.after_step = after_step
        self.interval_s = interval_s
        self.lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=f"yata-analysis-{session.id[:8]}", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        self._wake.set()

    def exclusive(self, work: Callable[[], Any]) -> Any:
        with self.lock:
            return work()

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    def _run(self) -> None:
        # Loading a large model can take a minute; start it when the session is created, not at the first request.
        self.pipeline.warm(self.session)
        while not self._closed:
            # The timer is a backstop; new evidence wakes the worker immediately.
            self._wake.wait(self.interval_s)
            self._wake.clear()
            while not self._closed:
                with self.lock:
                    if self._closed:
                        break
                    try:
                        more = self.pipeline.step(self.session)
                    except Exception as exc:  # keep the worker alive; the failure is reported, not hidden
                        self.session.emit("analysis_error", {"task": t("解析", "Analysis"), "message": f"{type(exc).__name__}: {str(exc)[:200]}"})
                        more = False
                if not more:
                    break
            if self.after_step is not None and not self._closed:
                try:
                    self.after_step()
                except Exception as exc:
                    self.session.emit("analysis_error", {"task": t("送信", "Sending"), "message": type(exc).__name__})
