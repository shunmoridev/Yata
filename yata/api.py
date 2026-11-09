from __future__ import annotations

import asyncio
import hmac
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import PORT, i18n
from .archive import SessionArchive, default_data_dir
from .audio_socket import serve_audio
from .export import filename as export_filename, session_markdown
from .features import Features, load_features
from .inputs.asr import LocalAsr
from .inputs.speakers import SpeakerEmbedder, SpeakerTracker
from .slack import SlackSender
from .inputs.vad import FrameScorer, SileroVad
from .models import Session, SessionStore
from .ollama import OllamaClient
from .preparation import Preparation, SetupError
from .state import SessionError, SessionManager
from .topics import TopicPipeline


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
TOKEN = secrets.token_urlsafe(32)
APP_ORIGINS = {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
    f"connect-src 'self' ws://127.0.0.1:{PORT} ws://localhost:{PORT}; media-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


def _bearer(request: Request) -> str:
    value = request.headers.get("authorization", "")
    scheme, _, credential = value.partition(" ")
    if scheme.lower() == "bearer" and credential.strip():
        return credential.strip()
    return request.cookies.get("yata_session", "")


def _check_app(request: Request) -> None:
    if not hmac.compare_digest(_bearer(request), request.app.state.access_token):
        raise HTTPException(status_code=401, detail=i18n.t("ローカルアプリのセッションが無効です。", "The local app session is not valid. Reload the page."))


def _local_host(host_header: str) -> bool:
    try:
        parsed = urlsplit("//" + host_header)
        return parsed.hostname in {"127.0.0.1", "localhost"} and parsed.port == PORT
    except ValueError:
        return False


def _asr_public(asr: LocalAsr) -> dict[str, Any]:
    readiness = asr.readiness()
    settings = asr.settings
    return {
        "engine": asr.name,
        "state": readiness["state"],
        "weights_present": readiness["weights_present"],
        "loaded": asr.loaded,
        "reason": readiness["reason"],
        # The resolved setup (profile, device, precision); the folder name only, never a full path.
        "profile": settings.profile if settings else None,
        "model": settings.model_path.name if settings and settings.model_path else None,
        "device": settings.device if settings else None,
        "compute_type": settings.compute_type if settings else None,
        "description": settings.describe() if settings else asr.name,
    }


def create_app(
    *,
    access_token: str | None = None,
    ollama: OllamaClient | None = None,
    asr: LocalAsr | None = None,
    partial_asr: LocalAsr | None = None,
    manager: SessionManager | None = None,
    scorer_factory: Callable[[], FrameScorer] | None = None,
    speakers: SpeakerEmbedder | None = None,
    slack: SlackSender | None = None,
    features: Features | None = None,
    preparation: Preparation | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await run_in_threadpool(_app.state.manager.close)
        await run_in_threadpool(_app.state.preparation.close)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.access_token = access_token or TOKEN
    app.state.features = features or load_features()
    sender = (slack or SlackSender()) if app.state.features.slack else SlackSender(switched_off=True)
    app.state.ollama = ollama or OllamaClient()
    app.state.preparation = preparation or Preparation(default_data_dir(), ollama=app.state.ollama)
    if asr is None:
        asr, partial_asr = app.state.preparation.build_asr()
    app.state.asr = asr
    app.state.partial_asr = partial_asr
    app.state.scorer_factory = scorer_factory or SileroVad
    app.state.speakers = speakers or app.state.preparation.build_speakers()

    def install_asr(final: LocalAsr, partial: LocalAsr | None) -> None:
        # Open audio sockets keep the engine they started with; the next share uses this one.
        app.state.asr = final
        app.state.partial_asr = partial

    def install_speakers(embedder: SpeakerEmbedder) -> None:
        app.state.speakers = embedder

    app.state.preparation.on_asr = install_asr
    app.state.preparation.on_speakers = install_speakers
    # One tracker per session so labels stay stable across reconnects.
    app.state.speaker_trackers = {}
    app.state.manager = manager or SessionManager(
        TopicPipeline(app.state.preparation.llm, translator=app.state.preparation.translator), SessionStore(), sender=sender,
        archive=SessionArchive(default_data_dir()), llm_allows=app.state.preparation.llm_allows,
        translator_allows=app.state.preparation.translator_allows)
    app.state.audio_sockets = {}
    app.state.audio_stop_events = {}
    app.state.audio_done_events = {}

    def request_locale(headers: Any) -> str:
        header = headers.get("accept-language")
        locale = i18n.for_request(app.state.preparation.locale_choice(), header)
        if header:
            i18n.remember_page(i18n.negotiate(header))
        return locale

    @app.middleware("http")
    async def local_only(request: Request, call_next: Any) -> Any:
        token = i18n.use(request_locale(request.headers))
        try:
            if not _local_host(request.headers.get("host", "")):
                return JSONResponse({"detail": i18n.t("このアプリは localhost からのみ利用できます。",
                                                      "This app can be used only from localhost.")}, status_code=403)
            origin = request.headers.get("origin")
            if origin and origin not in APP_ORIGINS:
                return JSONResponse({"detail": i18n.t("許可されていない送信元です。", "Requests from this origin are not allowed.")},
                                    status_code=403)
            response = await call_next(request)
        finally:
            i18n.reset(token)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def session_for(request: Request, session_id: str) -> Session:
        _check_app(request)
        try:
            return app.state.manager.get(session_id)
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        # The page learns which features are on, and the language chosen in the settings (none: the browser's), before
        # its first paint, so hidden controls never flash and the text starts in the right language.
        choice = app.state.preparation.locale_choice() or ""
        page = (WEB / "index.html").read_text(encoding="utf-8").replace(
            'data-features=""', 'data-features="' + " ".join(app.state.features.enabled()) + '"', 1).replace(
            '<html lang="ja"', f'<html lang="{i18n.locale()}"', 1).replace('data-locale=""', f'data-locale="{choice}"', 1)
        response = HTMLResponse(page, headers={"Cache-Control": "no-store"})
        response.set_cookie("yata_session", app.state.access_token, httponly=True, samesite="strict", path="/")
        response.headers["Content-Security-Policy"] = CSP
        return response

    @app.get("/assets/{asset_path:path}")
    async def asset(asset_path: str) -> Any:
        target = (WEB / "assets" / asset_path).resolve()
        if WEB.resolve() not in target.parents or not target.is_file():
            raise HTTPException(404)
        return FileResponse(target, headers={"Cache-Control": "no-store"})

    @app.get("/api/v1/health")
    async def health(request: Request) -> dict[str, Any]:
        _check_app(request)
        llm_state = await run_in_threadpool(app.state.preparation.llm_health)
        partial = app.state.partial_asr
        asr_state = _asr_public(app.state.asr)
        speakers_state = app.state.speakers.readiness()
        return {
            "llm": llm_state,
            "translator": await run_in_threadpool(app.state.preparation.translator_health),
            "asr": asr_state,
            "partial_captions": partial is not None,
            "speakers": speakers_state,
            "slack": app.state.manager.sender.readiness(),
            "features": app.state.features.public(),
            "locale": {"choice": app.state.preparation.locale_choice(), "current": i18n.locale(), "supported": list(i18n.SUPPORTED)},
            "setup": app.state.preparation.brief(asr_state["weights_present"], llm_state["models"], speakers_state["enabled"]),
        }

    def setup_status() -> dict[str, Any]:
        active = {**_asr_public(app.state.asr), "partial_captions": app.state.partial_asr is not None}
        return app.state.preparation.status(active, app.state.speakers.readiness()["enabled"])

    async def setup_action(request: Request, action: Callable[[], Any]) -> dict[str, Any]:
        _check_app(request)
        try:
            await run_in_threadpool(action)
        except SetupError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        return {"setup": await run_in_threadpool(setup_status)}

    @app.get("/api/v1/setup")
    async def setup(request: Request) -> dict[str, Any]:
        _check_app(request)
        return {"setup": await run_in_threadpool(setup_status)}

    @app.put("/api/v1/setup/plan")
    async def setup_plan(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.set_plan(
            body.get("summaries") is True, body.get("speakers") is True, translation=body.get("translation") is True))

    @app.put("/api/v1/setup/locale")
    async def setup_locale(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        _check_app(request)
        try:
            await run_in_threadpool(app.state.preparation.set_locale, str(body.get("locale") or ""))
        except SetupError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        return {"locale": {"choice": app.state.preparation.locale_choice(), "supported": list(i18n.SUPPORTED)}}

    @app.put("/api/v1/setup/storage")
    async def setup_storage(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.set_model_root(str(body.get("path") or "")))

    @app.put("/api/v1/setup/asr")
    async def setup_asr(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.choose_asr(str(body.get("model") or "")))

    @app.post("/api/v1/setup/models/{model_id}/download")
    async def setup_download(request: Request, model_id: str) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.start_download(model_id))

    @app.post("/api/v1/setup/models/{model_id}/verify")
    async def setup_verify(request: Request, model_id: str) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.start_verify(model_id))

    @app.post("/api/v1/setup/llm/pull")
    async def setup_pull(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.start_pull(str(body.get("model") or "")))

    @app.post("/api/v1/setup/llm/test")
    async def setup_llm_test(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.start_llm_test(
            str(body.get("model") or ""), str(body.get("backend") or "") or None))

    @app.put("/api/v1/setup/llm")
    async def setup_llm_choice(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.choose_llm(str(body.get("backend") or "")))

    @app.put("/api/v1/setup/translator")
    async def setup_translator(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        engine, model = body.get("engine"), body.get("model")
        return await setup_action(request, lambda: app.state.preparation.choose_translator(
            str(engine) if engine else None, str(model) if model else None))

    @app.put("/api/v1/setup/translator/path")
    async def setup_translator_path(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.set_translator_path(str(body.get("path") or "")))

    @app.put("/api/v1/setup/llm/openai")
    async def setup_llm_openai(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.set_openai(
            str(body.get("base_url") or ""), str(body.get("model") or "")))

    @app.post("/api/v1/setup/llm/foundry/start")
    async def setup_foundry_start(request: Request) -> dict[str, Any]:
        return await setup_action(request, app.state.preparation.start_foundry_service)

    @app.post("/api/v1/setup/llm/foundry/download")
    async def setup_foundry_download(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.start_foundry_download(str(body.get("model") or "")))

    @app.post("/api/v1/setup/jobs/{component}/cancel")
    async def setup_cancel(request: Request, component: str) -> dict[str, Any]:
        return await setup_action(request, lambda: app.state.preparation.cancel(component))

    @app.get("/api/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        _check_app(request)
        return await run_in_threadpool(app.state.preparation.llm_health)

    @app.post("/api/v1/sessions")
    async def create_session(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        _check_app(request)
        caption_language = str(body.get("caption_language") or "").strip() or None
        backend = app.state.preparation.llm_backend() or "ollama"
        summaries = body.get("summaries", True) is not False
        selected_model = str(body.get("selected_model", ""))
        # The translator is decided here, not by the page: without summaries it is the only way to translate;
        # with summaries, as chosen on the setup page (Preparation.translator_for).
        if caption_language is None:
            translator = None
        elif summaries:
            translator = await run_in_threadpool(app.state.preparation.translator_for, backend, selected_model, caption_language)
        else:
            translator = await run_in_threadpool(app.state.preparation.translator_ready)
        try:
            session = app.state.manager.create(
                source_url=str(body.get("source_url", "")),
                source_title=str(body.get("source_title", "")),
                source_kind=str(body.get("source_kind", "")),
                selected_model=selected_model,
                asr_language=str(body.get("asr_language") or "auto"),
                output_language=str(body.get("output_language") or "ja"),
                caption_language=caption_language,
                slack_delivery=body.get("slack_delivery") is True,
                summaries=summaries,
                llm_backend=backend,
                translator=translator,
            )
        except SessionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"session": session.public_dict()}

    @app.get("/api/v1/sessions")
    async def list_sessions(request: Request) -> dict[str, Any]:
        _check_app(request)
        return {"sessions": await run_in_threadpool(app.state.manager.list)}

    @app.get("/api/v1/sessions/{session_id}")
    async def session_snapshot(request: Request, session_id: str) -> dict[str, Any]:
        session = await run_in_threadpool(session_for, request, session_id)
        return session.snapshot()

    @app.put("/api/v1/sessions/{session_id}/notes")
    async def session_notes(request: Request, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        session = await run_in_threadpool(session_for, request, session_id)
        try:
            await run_in_threadpool(app.state.manager.set_notes, session, body.get("notes"))
        except SessionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"notes": session.notes}

    @app.get("/api/v1/sessions/{session_id}/export.md")
    async def export_markdown(request: Request, session_id: str) -> Response:
        # The page downloads it through a link, so the session cookie authenticates it like any other read.
        session = await run_in_threadpool(session_for, request, session_id)
        text = await run_in_threadpool(session_markdown, session)
        name = export_filename(session)
        return Response(text, media_type="text/markdown; charset=utf-8", headers={
            "Content-Disposition": f"attachment; filename=\"yata-session.md\"; filename*=UTF-8''{quote(name)}",
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(request: Request, session_id: str) -> dict[str, Any]:
        session = await run_in_threadpool(session_for, request, session_id)
        if session.id in app.state.audio_sockets:
            raise HTTPException(status_code=409, detail=i18n.t("音声の取り込み中は削除できません。", "Cannot delete while audio is being captured."))
        try:
            await run_in_threadpool(app.state.manager.delete, session)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        app.state.speaker_trackers.pop(session.id, None)
        return {"deleted": session.id}

    @app.post("/api/v1/sessions/{session_id}/resume")
    async def resume_session(request: Request, session_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        session = await run_in_threadpool(session_for, request, session_id)
        try:
            await run_in_threadpool(lambda: app.state.manager.resume(session, source_url=str((body or {}).get("source_url") or "")))
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"session": session.public_dict()}

    @app.post("/api/v1/sessions/{session_id}/topics/{topic_id}/deliver")
    async def redeliver(request: Request, session_id: str, topic_id: str) -> dict[str, Any]:
        session = session_for(request, session_id)
        if not app.state.features.slack:
            raise HTTPException(status_code=409, detail=i18n.t("Slack 連携は機能トグルでオフです。", "Slack is switched off by its feature toggle."))
        try:
            await run_in_threadpool(app.state.manager.redeliver, session, topic_id)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        with session.lock:
            topic = next(item for item in session.topics if item.id == topic_id)
            return {"topic": topic.as_dict()}

    @app.post("/api/v1/sessions/{session_id}/summarize")
    async def summarize(request: Request, session_id: str) -> dict[str, Any]:
        session = session_for(request, session_id)
        try:
            topics = await run_in_threadpool(app.state.manager.summarize, session)
        except SessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"topics": topics}

    @app.post("/api/v1/sessions/{session_id}/gap")
    async def client_gap(request: Request, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        session = session_for(request, session_id)
        start, end = body.get("capture_start_ms"), body.get("capture_end_ms")
        app.state.manager.record_gap(
            session, str(body.get("reason") or i18n.t("入力が途切れました。", "The input was interrupted.")),
            start if isinstance(start, int) else None, end if isinstance(end, int) else None,
        )
        return {"accepted": True}

    @app.get("/api/v1/sessions/{session_id}/events")
    async def events(request: Request, session_id: str, after: int = 0) -> dict[str, Any]:
        session = session_for(request, session_id)
        return {"session": session.public_dict(), "events": session.events_after(max(0, after))}

    @app.post("/api/v1/sessions/{session_id}/stop")
    async def stop(request: Request, session_id: str) -> dict[str, Any]:
        session = session_for(request, session_id)
        try:
            stop_event = app.state.audio_stop_events.get(session.id)
            done_event = app.state.audio_done_events.get(session.id)
            if stop_event is not None and done_event is not None and not done_event.is_set():
                stop_event.set()
                await asyncio.wait_for(done_event.wait(), timeout=120)
            await run_in_threadpool(app.state.manager.stop, session)
        except Exception as exc:
            app.state.manager.set_status(session, "error", i18n.t(f"停止時の最終要約に失敗しました: {exc}", f"The final summary failed: {exc}"))
            return {"session": session.public_dict(), "summary_error": str(exc)[:500]}
        return {"session": session.public_dict()}

    @app.post("/api/v1/sessions/{session_id}/cancel")
    async def cancel(request: Request, session_id: str) -> dict[str, Any]:
        session = session_for(request, session_id)
        app.state.manager.cancel(session)
        socket = app.state.audio_sockets.get(session.id)
        if socket is not None:
            try:
                await socket.close(code=4000, reason="session cancelled")
            except Exception:
                pass
        return {"session": session.public_dict()}

    @app.websocket("/api/v1/sessions/{session_id}/audio")
    async def audio(websocket: WebSocket, session_id: str) -> None:
        i18n.use(request_locale(websocket.headers))
        # Same-origin only: the Yata page shares the tab, and its HttpOnly cookie authenticates the socket.
        if (not _local_host(websocket.headers.get("host", ""))
                or websocket.headers.get("origin") not in APP_ORIGINS
                or not hmac.compare_digest(websocket.cookies.get("yata_session", ""), app.state.access_token)):
            await websocket.close(code=4403)
            return
        session = app.state.manager.store.get(session_id)
        if session is None or session.active_input != "shared_audio":
            await websocket.close(code=4404)
            return
        await websocket.accept()
        asr: LocalAsr = app.state.asr
        readiness = asr.readiness()
        if not readiness["weights_present"]:
            await websocket.send_json({"type": "asr_unavailable", "reason": readiness["reason"]})
            await websocket.close(code=4409)
            return
        if session.id in app.state.audio_sockets:
            await websocket.send_json({"type": "error", "reason": i18n.t("このセッションには音声接続がすでにあります。", "This session already has an audio connection.")})
            await websocket.close(code=4409)
            return
        stop_event = asyncio.Event()
        done_event = asyncio.Event()
        app.state.audio_sockets[session.id] = websocket
        app.state.audio_stop_events[session.id] = stop_event
        app.state.audio_done_events[session.id] = done_event

        def release() -> None:
            if app.state.audio_sockets.get(session.id) is websocket:
                app.state.audio_sockets.pop(session.id, None)
                app.state.audio_stop_events.pop(session.id, None)
                done = app.state.audio_done_events.pop(session.id, None)
                if done is not None:
                    done.set()

        label_speaker = None
        if app.state.speakers.readiness()["enabled"]:
            for known in [key for key in app.state.speaker_trackers if app.state.manager.store.get(key) is None]:
                del app.state.speaker_trackers[known]
            tracker = app.state.speaker_trackers.setdefault(session.id, SpeakerTracker())
            embedder = app.state.speakers

            def label_speaker(audio: Any, duration_s: float) -> str | None:
                return tracker.assign(embedder.embed(audio), duration_s) if duration_s >= tracker.min_seconds else None

        try:
            await serve_audio(
                websocket, session, manager=app.state.manager, engine=asr, partial_engine=app.state.partial_asr,
                scorer_factory=app.state.scorer_factory, stop_event=stop_event,
                on_evidence=app.state.manager.notify, release=release, speaker=label_speaker,
            )
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "reason": str(exc)[:300]})
                await websocket.close(code=4400)
            except Exception:
                pass
        finally:
            release()

    return app


app = create_app()
