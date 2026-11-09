from __future__ import annotations

import contextvars
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import hardware, i18n
from .i18n import Text, t
from .archive import _replace
from .catalog import (FOUNDRY_MODELS, LLM_MODELS, MODELS, PLAMO_TRANSLATE, ModelEntry, RemoteFile, memory_need, recommend,
                      translator_fits, translator_need)
from .download import PARTIAL, DownloadError, fetch, fingerprint, format_bytes, inspect, verify_hashes
from .inputs.asr import PROFILE_HF_REPOS, LocalAsr, _hf_snapshot, create_asr_engines, resolve_asr_settings
from .inputs.speakers import SpeakerEmbedder
from .llm.base import BACKENDS, LABELS, LlmCancelled, LlmUnavailable, loopback_url
from .llm.builtin import BuiltinBackend, LaunchSpec, list_devices, pick_device
from .llm.foundry import FoundryLocal
from .llm.openai_compat import KEY_ENV, OpenAICompatBackend
from .llm.router import LlmRouter
from .llm.translator import ARCHITECTURE, PlamoTranslator, caption_languages, gguf_architecture, plamo_language
from .ollama import INSTALL_URL, SUMMARY_MODEL_ALLOWLIST, OllamaClient, OllamaUnavailable, installed_locally, store_dir

SETTINGS = "setup.json"
ASR_ENV = ("YATA_ASR_MODEL_PATH", "YATA_ASR_PROFILE", "YATA_ASR_ENGINE")
LLM_ENV = "YATA_LLM_BACKEND"
OPENAI_ENV = ("YATA_OPENAI_BASE_URL", "YATA_OPENAI_MODEL")
LLAMA_ENV = "YATA_LLAMA_SERVER"
LLAMA_DEVICE_ENV = "YATA_LLAMA_DEVICE"
LLAMA_EXE = "llama-server.exe"
MIB = 1024 * 1024
TRANSLATION_ENGINES = ("translator", "llm")
CUSTOM_TRANSLATOR = "plamo-3-translate-custom"
TEST_SECONDS = 2.0
_RATE = 16_000
_TEST_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
_TEST_LINE = "新しいモデルは 3 月 14 日に 1,280 円で発売されます。"


class SetupError(RuntimeError):
    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    component: str
    target: str
    kind: str
    state: str = "running"
    phase: str = ""
    done: int = 0
    total: int = 0
    file: str = ""
    error: dict[str, str] | None = None
    note: str = ""
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    rate: float = 0.0
    cancel: threading.Event = field(default_factory=threading.Event)
    _mark: tuple[float, int] = (0.0, 0)

    def advance(self, done: int, total: int, file: str = "", phase: str = "") -> None:
        now = time.monotonic()
        since, before = self._mark
        if phase and phase != self.phase:
            self._mark = (now, done)
            self.rate = 0.0
        elif now - since >= 1.0:
            if since:
                self.rate = max(0.0, (done - before) / (now - since))
            self._mark = (now, done)
        self.done, self.total = done, total
        if file:
            self.file = file
        if phase:
            self.phase = phase

    def public(self) -> dict[str, Any]:
        end = self.finished or time.monotonic()
        return {"component": self.component, "target": self.target, "kind": self.kind, "state": self.state, "phase": self.phase,
                "done": self.done, "total": self.total, "file": self.file, "rate": int(self.rate), "error": self.error,
                "note": self.note, "elapsed_ms": int((end - self.started) * 1000)}


def installed_package(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _test_audio() -> np.ndarray:
    # Silence runs the encoder and the decoder once without any recorded voice. (Quiet noise makes Whisper
    # hallucinate and retry at higher temperatures, which measured 4-5 s instead of 1 s for whisper-small on CPU.)
    return np.zeros(int(TEST_SECONDS * _RATE), dtype=np.float32)


def _differs() -> str:
    return t("ファイルが配布元と異なり、読み込めませんでした（破損の可能性）。",
             "The files differ from the distributed ones and could not be loaded (they may be damaged).")


def _runtime_message(exc: BaseException, device: str) -> str:
    text = str(exc).strip()[:240] or type(exc).__name__
    lowered = text.lower()
    if device == "cuda" and any(word in lowered for word in ("cuda", "cublas", "cudnn", "gpu")):
        return t(f"GPU（CUDA）の実行環境を読み込めませんでした: {text}", f"Could not load the GPU (CUDA) runtime: {text}")
    return t(f"モデルを読み込めませんでした: {text}", f"Could not load the model: {text}")


def _unpack(archive: Path, target: Path) -> None:
    """Refuses entries that would land outside `target`."""
    staging = target.with_name(target.name + ".unpack")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                parts = Path(item.filename.replace("\\", "/")).parts
                if item.filename.startswith(("/", "\\")) or ".." in parts or any(":" in part for part in parts):
                    raise DownloadError("corrupt", t("実行環境の zip に想定外のパスがあるため展開しません。",
                                                     "The runtime zip contains unexpected paths, so it is not unpacked."))
            bundle.extractall(staging)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except zipfile.BadZipFile as exc:
        raise DownloadError("corrupt", t("実行環境の zip を開けません（破損の可能性）。", "Cannot open the runtime zip (it may be damaged).")) from exc
    except OSError as exc:
        raise DownloadError("permission", t(f"実行環境を展開できませんでした（{type(exc).__name__}）。",
                                            f"Could not unpack the runtime ({type(exc).__name__}).")) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)


class Preparation:
    def __init__(
        self,
        data_dir: str | Path,
        *,
        ollama: OllamaClient,
        environ: dict[str, str] | None = None,
        probe: Callable[[], dict[str, Any]] = hardware.probe,
        cuda: Callable[[], bool] = hardware.cuda_available,
        opener: Callable[..., Any] = urllib.request.urlopen,
        engines: Callable[..., tuple[LocalAsr, LocalAsr | None]] = create_asr_engines,
        embedder: Callable[[Path | None], SpeakerEmbedder] = SpeakerEmbedder,
        catalog: tuple[ModelEntry, ...] = MODELS,
        background: bool = True,
        foundry: FoundryLocal | None = None,
        llama_spawn: Callable[..., Any] = subprocess.Popen,
        run: Callable[..., Any] = subprocess.run,
        llm_opener: Any = None,
        platform: str = sys.platform,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.env = os.environ if environ is None else environ
        # The page fetches llama.cpp only on Windows (a Windows build); elsewhere llama-server comes from PATH.
        self.platform = platform
        self.ollama = ollama
        self._probe = probe
        self._cuda = cuda
        self._opener = opener
        self._engines = engines
        self._embedder = embedder
        self._run_command = run
        self.catalog = {entry.id: entry for entry in catalog}
        self.background = background
        self.jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._data = self._read()
        # Work outside a request (jobs aside, which keep their request's language) uses the saved language.
        i18n.set_choice(self.locale_choice())
        self.foundry = foundry or FoundryLocal([item.alias for item in FOUNDRY_MODELS])
        self.builtin = BuiltinBackend(resolve=self._builtin_spec, available=self._builtin_available,
                                      allowed=[entry.id for entry in self._builtin_entries()], spawn=llama_spawn, opener=llm_opener)
        self.openai = OpenAICompatBackend(self.openai_config, self.env, opener=llm_opener)
        self.llm = LlmRouter({"builtin": self.builtin, "ollama": ollama, "openai": self.openai, "foundry": self.foundry})
        self.translator = PlamoTranslator(resolve=self._translator_spec, available=self._translator_available,
                                          allowed=[entry.id for entry in self._translator_entries()] + [CUSTOM_TRANSLATOR],
                                          spawn=llama_spawn,
                                          opener=llm_opener)
        self.on_asr: Callable[[LocalAsr, LocalAsr | None], None] | None = None
        self.on_speakers: Callable[[SpeakerEmbedder], None] | None = None

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads((self.data_dir / SETTINGS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"checks": {}}
        if not isinstance(data, dict):
            return {"checks": {}}
        data.setdefault("checks", {})
        return data

    def _write(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        _replace(self.data_dir / SETTINGS, json.dumps(self._data, ensure_ascii=False, indent=1))

    def model_root(self) -> tuple[Path, str]:
        configured = self.env.get("YATA_MODEL_ROOT", "").strip()
        if configured:
            return Path(configured).expanduser(), "env"
        saved = str(self._data.get("model_root") or "").strip()
        if saved:
            return Path(saved), "settings"
        return self.data_dir / "models", "default"

    def asr_locked(self) -> str | None:
        return next((name for name in ASR_ENV if self.env.get(name, "").strip()), None)

    def plan(self) -> dict[str, bool] | None:
        plan = self._data.get("plan")
        if not isinstance(plan, dict):
            return None
        summaries = bool(plan.get("summaries"))
        return {"summaries": summaries, "translation": not summaries and bool(plan.get("translation")),
                "speakers": bool(plan.get("speakers"))}

    def locale_choice(self) -> str | None:
        value = self._data.get("locale")
        return value if value in i18n.SUPPORTED else None

    def _llm_data(self) -> dict[str, Any]:
        data = self._data.get("llm")
        return data if isinstance(data, dict) else {}

    def llm_choice(self) -> dict[str, Any]:
        configured = self.env.get(LLM_ENV, "").strip()
        if configured in BACKENDS:
            return {"backend": configured, "source": "env"}
        saved = self._llm_data().get("backend")
        if saved in BACKENDS:
            return {"backend": saved, "source": "settings"}
        if self._data.get("plan") is not None or any(str(key).startswith("llm:") for key in self._data["checks"]):
            return {"backend": "ollama", "source": "earlier"}
        return {"backend": None, "source": None}

    def llm_backend(self) -> str | None:
        return self.llm_choice()["backend"]

    def openai_config(self) -> dict[str, str]:
        saved = self._llm_data().get("openai")
        saved = saved if isinstance(saved, dict) else {}
        return {"base_url": self.env.get(OPENAI_ENV[0], "").strip() or str(saved.get("base_url") or ""),
                "model": self.env.get(OPENAI_ENV[1], "").strip() or str(saved.get("model") or "")}

    def llm_allows(self, backend: str, model: str) -> bool:
        return self.llm.allows(backend, model)

    def _llm_check_key(self, backend: str, model: str) -> str:
        return f"llm:{backend}:{model}"

    def _llm_record(self, backend: str, model: str) -> Any:
        checks = self._data["checks"]
        record = checks.get(self._llm_check_key(backend, model))
        # Test records from before the backends existed are Ollama's.
        return record if record is not None or backend != "ollama" else checks.get(f"llm:{model}")

    def _builtin_entries(self) -> list[ModelEntry]:
        return [entry for entry in self.catalog.values() if entry.component == "llm"]

    def _runtime_entry(self) -> ModelEntry | None:
        if self.platform != "win32":
            return None
        return next((entry for entry in self.catalog.values() if entry.component == "llm-runtime"), None)

    def llama_server(self) -> Path | None:
        configured = self.env.get(LLAMA_ENV, "").strip()
        if configured:
            path = Path(configured).expanduser()
            return path if path.is_file() else None
        if self.platform != "win32":
            found = shutil.which("llama-server", path=self.env.get("PATH"))
            return Path(found) if found else None
        entry = self._runtime_entry()
        if entry is None:
            return None
        path = self.locate(entry)[0] / "bin" / LLAMA_EXE
        return path if path.is_file() else None

    def _llama_install_hint(self) -> str | None:
        if self.platform == "win32" or self.env.get(LLAMA_ENV, "").strip() or self.llama_server() is not None:
            return None
        if self.platform == "darwin":
            return t("macOS では Yata は llama.cpp を取得しません。ターミナルで brew install llama.cpp を実行してから、"
                     "Yata を起動し直してください。",
                     "On macOS, Yata does not fetch llama.cpp. Run brew install llama.cpp in Terminal, then restart Yata.")
        return t("この OS では Yata は llama.cpp を取得しません。llama.cpp の llama-server を入れて PATH の通った場所に置くか、"
                 f"環境変数 {LLAMA_ENV} にその場所を指定してから、Yata を起動し直してください。",
                 "On this OS, Yata does not fetch llama.cpp. Install llama.cpp's llama-server somewhere on PATH, or set "
                 f"{LLAMA_ENV} to its location, then restart Yata.")

    def _builtin_model_path(self, entry: ModelEntry) -> Path:
        return self.locate(entry)[0] / entry.files[0].path

    def _builtin_available(self) -> list[str]:
        if self.llama_server() is None:
            return []
        return [entry.id for entry in self._builtin_entries() if inspect(entry, self.locate(entry)[0])["complete"]]

    def _builtin_spec(self, model: str) -> LaunchSpec:
        entry = next((item for item in self._builtin_entries() if item.id == model), None)
        if entry is None:
            raise LlmUnavailable(t("内蔵 LLM に許可されたモデルではありません。", "This model is not allowed for the built-in LLM."))
        executable = self.llama_server()
        if executable is None:
            raise LlmUnavailable(t("llama.cpp の実行環境がありません。「この PC の準備」で準備してください。",
                                    "The llama.cpp runtime is missing. Get it under \"Set up this PC\"."))
        return LaunchSpec(executable=executable, model_path=self._builtin_model_path(entry), model_id=entry.id,
                          device=self._llama_device())

    def _llama_device(self) -> str | None:
        device = self.env.get(LLAMA_DEVICE_ENV, "").strip() or None
        runtime = self._runtime_entry()
        record = self._data["checks"].get(f"llm-runtime:{runtime.id}") if runtime is not None and not device else None
        if isinstance(record, dict) and record.get("device"):
            device = str(record["device"])
        return device

    def _runtime_identity(self) -> str:
        runtime = self._runtime_entry()
        return str(self.llama_server() or (runtime.id if runtime else ""))

    def _translator_entries(self) -> list[ModelEntry]:
        custom = self._custom_translator()
        return [entry for entry in self.catalog.values() if entry.component == "translator"] + ([custom] if custom else [])

    def _translator_by_id(self, model: str) -> ModelEntry | None:
        return next((entry for entry in self._translator_entries() if entry.id == model), None)

    def custom_translator_path(self) -> Path | None:
        data = self._data.get("translator")
        path = data.get("path") if isinstance(data, dict) else None
        return Path(path) if isinstance(path, str) and path else None

    def _custom_translator(self) -> ModelEntry | None:
        """The user's own file has no source to fetch from and no hash: its size is whatever the file has now, and
        only the load test says it is usable."""
        path = self.custom_translator_path()
        if path is None:
            return None
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            size = 0
        return ModelEntry(
            id=CUSTOM_TRANSLATOR, component="translator", label=f"PLaMo 3 Translate（{path.name}）", folder=str(path.parent),
            source=t("この PC のファイル", "A file on this PC"), page="", license=PLAMO_TRANSLATE.license, base_url="",
            files=(RemoteFile(path.name, size),),
            summary=Text("指定した PLaMo 3 Translate の GGUF ファイルです。", "The PLaMo 3 Translate GGUF file you chose."),
            essential=(path.name,), license_page=PLAMO_TRANSLATE.license_page, terms=PLAMO_TRANSLATE.terms)

    def _translator_spec(self, model: str) -> LaunchSpec:
        entry = self._translator_by_id(model)
        if entry is None:
            raise LlmUnavailable(t("翻訳に許可されたモデルではありません。", "This model is not allowed for translation."))
        executable = self.llama_server()
        if executable is None:
            raise LlmUnavailable(t("llama.cpp の実行環境がありません。「この PC の準備」で準備してください。",
                                    "The llama.cpp runtime is missing. Get it under \"Set up this PC\"."))
        return LaunchSpec(executable=executable, model_path=self.locate(entry)[0] / entry.files[0].path, model_id=entry.id,
                          device=self._llama_device())

    def _translator_here(self) -> list[str]:
        return [entry.id for entry in self._translator_entries() if inspect(entry, self.locate(entry)[0])["complete"]]

    def _translator_available(self) -> list[str]:
        return self._translator_here() if self.llama_server() is not None else []

    def translator_choice(self) -> dict[str, str | None]:
        data = self._data.get("translator")
        data = data if isinstance(data, dict) else {}
        known = {entry.id for entry in self._translator_entries()}
        return {"engine": data.get("engine") if data.get("engine") in TRANSLATION_ENGINES else None,
                "model": data.get("model") if data.get("model") in known else None}

    def translator_entry(self) -> ModelEntry | None:
        """Files being here is enough, so a PC set up before the sizes existed keeps its 2B even while the runtime
        is missing."""
        entries = self._translator_entries()
        chosen = self.translator_choice()["model"]
        here = set(self._translator_here())
        return (next((entry for entry in entries if entry.id == chosen), None)
                or next((entry for entry in entries if entry.id in here), None) or next(iter(entries), None))

    def translator_ready(self) -> str | None:
        entry = self.translator_entry()
        return entry.id if entry is not None and entry.id in self._translator_available() else None

    def translator_allows(self, model: str, language: str) -> bool:
        return self.translator.allows(model) and plamo_language(language) is not None

    def _llm_need(self, backend: str, model: str) -> int | None:
        if backend == "builtin":
            entry = next((item for item in self._builtin_entries() if item.id == model), None)
            return memory_need(entry.size) if entry is not None else None
        if backend == "foundry":
            model = next((item.ollama for item in FOUNDRY_MODELS if item.alias == model), "")
        if backend in {"ollama", "foundry"}:
            return next((item.need for item in LLM_MODELS if item.name == model), None)
        return None

    def translator_for(self, backend: str, model: str, language: str | None) -> str | None:
        engine = self.translator_choice()["engine"]
        translator = self.translator_ready()
        if engine == "llm" or translator is None or not language or plamo_language(language) is None:
            return None
        if engine == "translator":
            return translator
        return translator if translator_fits(self._probe(), self._llm_need(backend, model), self._translator_by_id(translator)) else None

    def locate(self, entry: ModelEntry) -> tuple[Path, str]:
        """The llama.cpp runtime is a program, not a model: it goes to <data dir>/runtime."""
        if entry.component == "llm-runtime":
            return self.data_dir / "runtime" / entry.folder, "folder"
        if entry.id == CUSTOM_TRANSLATOR:
            return Path(entry.folder), "custom"
        folder = self.model_root()[0] / entry.folder
        if folder.is_dir() or entry.profile is None:
            return folder, "folder"
        for repo in PROFILE_HF_REPOS.get(entry.profile, ()):
            snapshot = _hf_snapshot(repo)
            if snapshot is not None:
                return snapshot, "hf-cache"
        return folder, "folder"

    def asr_entry(self) -> ModelEntry | None:
        if self.asr_locked():
            return None
        chosen = self.catalog.get(str(self._data.get("asr_model") or ""))
        if chosen is not None and chosen.component == "asr":
            return chosen
        models = [entry for entry in self.catalog.values() if entry.component == "asr"]
        if not models:
            return None
        preferred = "full" if self._cuda() else "light"
        models.sort(key=lambda entry: entry.profile != preferred)
        for entry in models:
            if inspect(entry, self.locate(entry)[0])["usable"]:
                return entry
        return models[0]

    def effective_env(self) -> dict[str, str]:
        env = dict(self.env)
        env["YATA_MODEL_ROOT"] = str(self.model_root()[0])
        entry = self.asr_entry()
        if entry is not None and entry.profile:
            env["YATA_ASR_PROFILE"] = entry.profile
            folder, where = self.locate(entry)
            if where == "hf-cache":
                env["YATA_ASR_MODEL_PATH"] = str(folder)
        return env

    def speaker_model_path(self) -> Path | None:
        configured = self.env.get("YATA_SPEAKER_MODEL_PATH", "").strip()
        if configured:
            return Path(configured).expanduser()
        entry = next((item for item in self.catalog.values() if item.component == "speakers"), None)
        return self.locate(entry)[0] / entry.files[0].path if entry is not None else None

    def build_asr(self) -> tuple[LocalAsr, LocalAsr | None]:
        return self._engines(self.effective_env(), cuda=self._cuda)

    def build_speakers(self) -> SpeakerEmbedder:
        return self._embedder(self.speaker_model_path())

    def _apply(self) -> None:
        if self.on_asr is not None:
            self.on_asr(*self.build_asr())
        if self.on_speakers is not None:
            self.on_speakers(self.build_speakers())

    def set_plan(self, summaries: bool, speakers: bool, translation: bool = False) -> None:
        with self._lock:
            self._data["plan"] = {"summaries": bool(summaries), "translation": not summaries and bool(translation),
                                  "speakers": bool(speakers)}
            self._write()

    def set_locale(self, value: str) -> None:
        text = str(value or "").strip().lower()
        if text not in ("", "auto") and text not in i18n.SUPPORTED:
            raise SetupError(i18n.t("表示言語が不正です。", "Unknown display language."), 422)
        with self._lock:
            if text in i18n.SUPPORTED:
                self._data["locale"] = text
            else:
                self._data.pop("locale", None)
            self._write()
        i18n.set_choice(self.locale_choice())

    def set_model_root(self, raw: str) -> None:
        if self.env.get("YATA_MODEL_ROOT", "").strip():
            raise SetupError(t("保存先は環境変数 YATA_MODEL_ROOT で指定されているため、ここでは変えられません。",
                                "The model folder is set by the environment variable YATA_MODEL_ROOT, so it cannot be changed here."))
        text = os.path.expandvars(str(raw or "").strip().strip('"'))
        path = Path(text).expanduser() if text else None
        if path is not None and not path.is_absolute():
            raise SetupError(t("保存先はドライブから始まるフォルダのパスで指定してください（例: D:\\Yata\\models）。",
                                "Enter the model folder as a full path starting with a drive (for example D:\\Yata\\models)."), 422)
        if path is not None and path.exists() and not path.is_dir():
            raise SetupError(t("保存先にはファイルではなくフォルダを指定してください。", "The model folder must be a folder, not a file."), 422)
        with self._lock:
            if any(job.state == "running" for job in self.jobs.values()):
                raise SetupError(t("準備が進行中です。終わるか中止してから保存先を変えてください。",
                                    "Setup is in progress. Wait for it to finish or cancel it before changing the folder."))
            if path is None:
                self._data.pop("model_root", None)
            else:
                self._data["model_root"] = str(path)
            self._write()
        self._apply()

    def choose_asr(self, entry_id: str) -> None:
        locked = self.asr_locked()
        if locked:
            raise SetupError(t(f"音声認識のモデルは環境変数 {locked} で指定されているため、ここでは変えられません。",
                                f"The speech recognition model is set by the environment variable {locked}, so it cannot be changed here."))
        entry = self.catalog.get(entry_id)
        if entry is None or entry.component != "asr":
            raise SetupError(t("音声認識のモデルではありません。", "This is not a speech recognition model."), 422)
        with self._lock:
            self._data["asr_model"] = entry.id
            self._write()
        if self.on_asr is not None:
            self.on_asr(*self.build_asr())

    def choose_llm(self, backend: str) -> None:
        if self.env.get(LLM_ENV, "").strip() in BACKENDS:
            raise SetupError(t(f"要約の LLM は環境変数 {LLM_ENV} で指定されているため、ここでは変えられません。",
                                f"The summary LLM is set by the environment variable {LLM_ENV}, so it cannot be changed here."))
        if backend not in BACKENDS:
            raise SetupError(t("要約の LLM の種類が不正です。", "Unknown kind of summary LLM."), 422)
        if backend == "foundry" and not self.foundry.installed():
            raise SetupError(t("Foundry Local はこの PC に入っていません。", "Foundry Local is not installed on this PC."))
        with self._lock:
            before = self.llm_backend()
            self._data.setdefault("llm", {})["backend"] = backend
            self._write()
        if before == "builtin" and backend != "builtin":
            self.builtin.release()  # frees the GPU; a live built-in session starts it again when it asks

    def choose_translator(self, engine: str | None = None, model: str | None = None) -> None:
        if engine is not None and engine not in TRANSLATION_ENGINES:
            raise SetupError(t("字幕の翻訳の方法が不正です。", "Unknown way to translate captions."), 422)
        if model is not None and model not in {entry.id for entry in self._translator_entries()}:
            raise SetupError(t("翻訳に許可されたモデルではありません。", "This model is not allowed for translation."), 422)
        with self._lock:
            before = self.translator_choice()
            saved = {key: value for key, value in before.items() if value is not None}
            path = self.custom_translator_path()
            self._data["translator"] = {**saved, **({"engine": engine} if engine else {}), **({"model": model} if model else {}),
                                        **({"path": str(path)} if path else {})}
            self._write()
        if (engine == "llm" and before["engine"] != "llm") or (model is not None and model != before["model"]):
            self.translator.release()  # frees its memory; a live session using it starts it again when it asks

    def set_translator_path(self, raw: str) -> None:
        """Choose a PLaMo 3 Translate GGUF file of the user's own; an empty value forgets it."""
        text = str(raw or "").strip().strip('"')
        path = None
        if text:
            path = Path(text).expanduser()
            if not path.is_absolute():
                raise SetupError(t("ファイルの場所は、ドライブやルートから始まる完全なパスで指定してください。",
                                    "Give the file's full path, starting from the drive or the root."), 422)
            if path.suffix.lower() != ".gguf":
                raise SetupError(t("GGUF ファイル（.gguf）を指定してください。", "Choose a GGUF file (.gguf)."), 422)
            if not path.is_file():
                raise SetupError(t("そのファイルが見つかりません。", "That file was not found."), 422)
            architecture = gguf_architecture(path)
            if architecture != ARCHITECTURE:
                found = architecture or t("形式を読み取れません", "unreadable format")
                raise SetupError(t(f"PLaMo 3 の GGUF ファイルではありません（{found}）。",
                                    f"This is not a PLaMo 3 GGUF file ({found})."), 422)
        with self._lock:
            before = self.custom_translator_path()
            data = dict(self._data.get("translator") or {})
            data.pop("path", None)
            if path is not None:
                data.update(path=str(path), model=CUSTOM_TRANSLATOR)
            elif data.get("model") == CUSTOM_TRANSLATOR:
                data.pop("model")
            self._data["translator"] = data
            if path != before:
                # A test of another file says nothing about this one.
                self._data["checks"].pop(f"translator:{CUSTOM_TRANSLATOR}", None)
            self._write()
        if path != before:
            self.translator.release()

    def set_openai(self, base_url: str, model: str) -> None:
        locked = next((name for name in OPENAI_ENV if self.env.get(name, "").strip()), None)
        if locked:
            raise SetupError(t(f"OpenAI 互換 API の接続先は環境変数 {locked} で指定されているため、ここでは変えられません。",
                                f"The OpenAI-compatible server is set by the environment variable {locked}, so it cannot be changed here."))
        url = str(base_url or "").strip()
        if url:
            try:
                url = loopback_url(url)
            except ValueError as exc:
                raise SetupError(str(exc), 422) from exc
        name = str(model or "").strip()
        if len(name) > 200 or any(ord(char) < 32 for char in name):
            raise SetupError(t("モデル名が不正です。", "Invalid model name."), 422)
        with self._lock:
            self._data.setdefault("llm", {})["openai"] = {"base_url": url, "model": name}
            self._write()

    def _start(self, component: str, target: str, kind: str, work: Callable[[Job], None]) -> dict[str, Any]:
        with self._lock:
            current = self.jobs.get(component)
            if current is not None and current.state == "running":
                raise SetupError(t("この項目の準備が進行中です。終わるまで待つか、中止してください。",
                                    "This item is already being set up. Wait for it to finish, or cancel it."))
            job = Job(component=component, target=target, kind=kind)
            self.jobs[component] = job
        if self.background:
            # The job reports in the language of the page that started it.
            context = contextvars.copy_context()
            threading.Thread(target=context.run, args=(self._run, job, work), name=f"yata-setup-{component}", daemon=True).start()
        else:
            self._run(job, work)
        return job.public()

    def _run(self, job: Job, work: Callable[[Job], None]) -> None:
        try:
            work(job)
            job.state = "done"
        except DownloadError as exc:
            job.state = "cancelled" if exc.kind == "cancelled" else "failed"
            job.error = {"kind": exc.kind, "message": str(exc)}
        except LlmCancelled as exc:
            job.state, job.error = "cancelled", {"kind": "cancelled", "message": str(exc)}
        except OllamaUnavailable as exc:
            job.state, job.error = "failed", {"kind": "ollama", "message": str(exc)}
        except LlmUnavailable as exc:
            job.state, job.error = "failed", {"kind": "llm", "message": str(exc)}
        except SetupError as exc:
            job.state, job.error = "failed", {"kind": "runtime", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001 - the page shows why, the server keeps running
            job.state, job.error = "failed", {"kind": "runtime", "message": str(exc)[:300] or type(exc).__name__}
        finally:
            job.finished = time.monotonic()

    def cancel(self, component: str) -> None:
        job = self.jobs.get(component)
        if job is None or job.state != "running":
            raise SetupError(t("中止できる準備はありません。", "Nothing is running that can be cancelled."))
        job.cancel.set()

    def _model(self, entry_id: str) -> ModelEntry:
        entry = self.catalog.get(entry_id) or (self._custom_translator() if entry_id == CUSTOM_TRANSLATOR else None)
        if entry is None:
            raise SetupError(t("そのモデルは準備できません。", "That model cannot be set up."), 404)
        return entry

    def start_download(self, entry_id: str) -> dict[str, Any]:
        entry = self._model(entry_id)
        folder, where = self.locate(entry)
        if where == "custom":
            raise SetupError(t("指定したファイルは取得できません。ファイルを置き直すか、別のファイルを指定してください。",
                                "A file you chose cannot be fetched. Put the file back, or choose another one."))
        if where == "hf-cache":
            raise SetupError(t("このモデルは Hugging Face のキャッシュにあります。Yata はキャッシュには書き込みません。",
                                "This model is in the Hugging Face cache. Yata does not write to the cache."))
        if entry.component == "speakers" and self.env.get("YATA_SPEAKER_MODEL_PATH", "").strip():
            raise SetupError(t("話者ラベルのモデルは環境変数 YATA_SPEAKER_MODEL_PATH で指定されているため、ここでは取得しません。",
                                "The speaker label model is set by the environment variable YATA_SPEAKER_MODEL_PATH, so it is not fetched here."))
        if entry.component == "llm-runtime" and self.env.get(LLAMA_ENV, "").strip():
            raise SetupError(t(f"llama.cpp は環境変数 {LLAMA_ENV} で指定されているため、ここでは取得しません。",
                                f"llama.cpp is set by the environment variable {LLAMA_ENV}, so it is not fetched here."))
        if entry.component == "llm-runtime" and self._runtime_entry() is None:
            raise SetupError(t("この OS では Yata は llama.cpp を取得しません（取得できるのは Windows 用のものだけです）。",
                                "On this OS, Yata does not fetch llama.cpp (it can fetch only the Windows build)."))
        state = inspect(entry, folder)
        if state["complete"]:
            raise SetupError(t("ファイルはすべて揃っています。「確認する」で読み込みを試せます。",
                                "All files are here. Use \"Check\" to try loading the model."))

        def work(job: Job) -> None:
            fetch(entry, folder, progress=lambda done, total, name, phase: job.advance(done, total, name, phase),
                  cancelled=job.cancel.is_set, opener=self._opener)
            self._test(entry, job)

        return self._start(entry.component, entry.id, "download", work)

    def start_verify(self, entry_id: str) -> dict[str, Any]:
        entry = self._model(entry_id)
        folder, _ = self.locate(entry)
        if entry.component == "speakers" and self.env.get("YATA_SPEAKER_MODEL_PATH", "").strip():
            folder = None
        if folder is not None and not inspect(entry, folder)["complete"]:
            raise SetupError(t("ファイルが足りないため確認できません。先に取得してください。",
                                "Files are missing, so the model cannot be checked. Get it first."))

        def work(job: Job) -> None:
            if folder is not None:
                differing = verify_hashes(entry, folder, progress=lambda done, total, name, phase: job.advance(done, total, name, phase),
                                          cancelled=job.cancel.is_set)
                if differing:
                    job.note = t("、".join(differing) + " の内容が配布元の版と異なります（別の版、または破損）。",
                                 ", ".join(differing) + " differ from the distributed version (another version, or damaged).")
            self._test(entry, job, differing=bool(job.note))

        return self._start(entry.component, entry.id, "verify", work)

    def _test(self, entry: ModelEntry, job: Job, differing: bool = False) -> None:
        if entry.component == "asr":
            self._test_asr(entry, job, differing)
        elif entry.component == "llm-runtime":
            self._test_runtime(entry, job, differing)
        elif entry.component == "llm":
            self._test_builtin(entry, job, differing)
        elif entry.component == "translator":
            self._test_translator(entry, job, differing)
        else:
            self._test_speakers(entry, job, differing)

    def _test_runtime(self, entry: ModelEntry, job: Job, differing: bool) -> None:
        if differing:
            raise DownloadError("corrupt", t("llama.cpp の zip が配布元と異なるため、展開も実行もしません。"
                                             "フォルダを削除してから、もう一度取得してください。",
                                             "The llama.cpp zip differs from the distributed one, so it is neither unpacked nor run. "
                                             "Delete the folder and get it again."))
        folder, _ = self.locate(entry)
        executable = folder / "bin" / LLAMA_EXE
        if not executable.is_file():
            job.advance(0, 0, "", "unpack")
            self.builtin.release()
            self.translator.release()
            _unpack(folder / entry.files[0].path, folder / "bin")
        if not executable.is_file():
            raise DownloadError("corrupt", t(f"zip に {LLAMA_EXE} がありません。", f"The zip has no {LLAMA_EXE}."))
        job.advance(0, 0, "", "load")
        started = time.monotonic()
        try:
            devices = list_devices(executable, run=self._run_command)
        except LlmUnavailable as exc:
            raise SetupError(str(exc)) from exc
        gpu = (self._probe().get("gpu") or {}).get("name")
        record = {"fingerprint": fingerprint(entry, folder), "devices": devices, "device": pick_device(devices, gpu),
                  "at": _now(), "test_ms": int((time.monotonic() - started) * 1000), "differs": False}
        with self._lock:
            self._data["checks"][f"llm-runtime:{entry.id}"] = record
            self._write()

    def _test_builtin(self, entry: ModelEntry, job: Job, differing: bool) -> None:
        if self.llama_server() is None:
            job.note = (job.note + " " if job.note else "") + t("モデルは揃いました。llama.cpp の実行環境を準備すると試せます。",
                                                                 "The model is here. Set up the llama.cpp runtime to test it.")
            return
        try:
            self._test_backend("builtin", entry.id, job)
        except LlmUnavailable as exc:
            if differing:
                raise DownloadError("corrupt", _differs()) from exc
            raise

    def _test_translator(self, entry: ModelEntry, job: Job, differing: bool) -> None:
        if self.llama_server() is None:
            job.note = (job.note + " " if job.note else "") + t("モデルは揃いました。llama.cpp の実行環境を準備すると試せます。",
                                                                 "The model is here. Set up the llama.cpp runtime to test it.")
            return
        job.advance(0, 0, "", "test")
        started = time.monotonic()
        try:
            sample = self.translator.translate(entry.id, _TEST_LINE, target="en", timeout=600)
        except LlmUnavailable as exc:
            if differing:
                raise DownloadError("corrupt", _differs()) from exc
            raise
        if not sample:
            raise SetupError(t("翻訳モデルが訳文を返しませんでした。", "The translation model returned no translation."))
        folder, _ = self.locate(entry)
        record = {"fingerprint": fingerprint(entry, folder), "runtime": self._runtime_identity(), "at": _now(),
                  "test_ms": int((time.monotonic() - started) * 1000), "device": self.translator.state().get("device"),
                  "sample": sample[:200], "differs": differing}
        with self._lock:
            self._data["checks"][f"translator:{entry.id}"] = record
            self._write()

    def _test_asr(self, entry: ModelEntry, job: Job, differing: bool) -> None:
        folder, _ = self.locate(entry)
        env = {**self.effective_env(), "YATA_ASR_PROFILE": entry.profile or "auto", "YATA_ASR_MODEL_PATH": str(folder)}
        settings = resolve_asr_settings(env, cuda=self._cuda)
        job.advance(0, 0, "", "load")
        started = time.monotonic()
        try:
            final, partial = self._engines(env, cuda=self._cuda)
            final.load()
            loaded = time.monotonic()
            final.transcribe(_test_audio(), language="ja")
        except Exception as exc:  # noqa: BLE001 - classified for the page
            if differing:
                raise DownloadError("corrupt", _differs() + t("フォルダを移すか削除してから、もう一度取得してください。",
                                                              " Move or delete the folder, then get it again.")) from exc
            raise SetupError(_runtime_message(exc, settings.device)) from exc
        tested = time.monotonic()
        record = {"fingerprint": fingerprint(entry, folder), "device": settings.device, "compute_type": settings.compute_type,
                  "at": _now(), "load_ms": int((loaded - started) * 1000), "test_ms": int((tested - loaded) * 1000),
                  "differs": differing}
        with self._lock:
            self._data["checks"][f"asr:{entry.id}"] = record
            self._write()
        current = self.asr_entry()
        if self.on_asr is not None and current is not None and current.id == entry.id:
            # The engine that passed the test is the one that runs, already loaded.
            self.on_asr(final, partial)

    def _test_speakers(self, entry: ModelEntry, job: Job, differing: bool) -> None:
        path = self.speaker_model_path()
        if not installed_package("sherpa_onnx"):
            job.note = (job.note + " " if job.note else "") + t("モデルは揃いました。sherpa-onnx を入れて（uv sync --inexact --extra speakers）再起動すると使えます。",
                                                                 "The model is here. Install sherpa-onnx (uv sync --inexact --extra speakers) and restart to use it.")
            return
        job.advance(0, 0, "", "load")
        started = time.monotonic()
        embedder = self._embedder(path)
        try:
            vector = embedder.embed(_test_audio())
        except Exception as exc:  # noqa: BLE001
            if differing:
                raise DownloadError("corrupt", _differs()) from exc
            raise SetupError(_runtime_message(exc, "cpu")) from exc
        if not getattr(vector, "size", 0):
            raise SetupError(t("話者ラベルのモデルが特徴を返しませんでした。", "The speaker label model returned no features."))
        folder, _ = self.locate(entry)
        record = {"fingerprint": fingerprint(entry, folder) if path and path.parent == folder else {"path": str(path)},
                  "at": _now(), "test_ms": int((time.monotonic() - started) * 1000), "differs": differing}
        with self._lock:
            self._data["checks"][f"speakers:{entry.id}"] = record
            self._write()
        if self.on_speakers is not None:
            self.on_speakers(embedder)

    def start_pull(self, name: str) -> dict[str, Any]:
        if name not in SUMMARY_MODEL_ALLOWLIST:
            raise SetupError(t("要約に許可されたモデルではありません。", "This model is not allowed for summaries."), 422)
        entry = next(item for item in LLM_MODELS if item.name == name)
        try:
            installed = self.ollama.tags()
        except OllamaUnavailable as exc:
            raise SetupError(str(exc)) from exc
        if name in installed:
            raise SetupError(t("このモデルは Ollama に入っています。「確認する」で応答を試せます。",
                                "Ollama already has this model. Use \"Check\" to test its answer."))
        free = hardware.disk_free(store_dir(self.env, self.ollama.models_dir()))
        if free is not None and free < entry.size + 512 * 1024 * 1024:
            raise SetupError(t(f"Ollama の保存先の空き容量が足りません（必要 {format_bytes(entry.size)}、空き {format_bytes(free)}）。",
                                f"Not enough free space in Ollama's folder (needs {format_bytes(entry.size)}, free {format_bytes(free)})."))

        def work(job: Job) -> None:
            job.advance(0, entry.size, "", "pull")
            self.ollama.pull(name, lambda status, done, total: job.advance(done, total or entry.size, status, "pull"), job.cancel.is_set)
            self._test_backend("ollama", name, job)

        return self._start("llm", name, "pull", work)

    def start_llm_test(self, name: str, backend: str | None = None) -> dict[str, Any]:
        backend = backend or self.llm_backend() or "ollama"
        if backend not in BACKENDS or not self.llm.allows(backend, name):
            raise SetupError(t("要約に許可されたモデルではありません。", "This model is not allowed for summaries."), 422)
        return self._start("llm", name, "test", lambda job: self._test_backend(backend, name, job))

    def _llm_identity(self, backend: str, model: str, installed: dict[str, str] | None = None) -> dict[str, Any]:
        if backend == "ollama":
            return {"digest": (installed if installed is not None else self.ollama.tags()).get(model, "")}
        if backend == "builtin":
            entry = next(item for item in self._builtin_entries() if item.id == model)
            return {"fingerprint": fingerprint(entry, self.locate(entry)[0]), "runtime": self._runtime_identity()}
        if backend == "openai":
            return {"base_url": self.openai_config()["base_url"], "model": model}
        return {"variant": self.foundry.variants().get(model, "")}

    def _test_backend(self, backend: str, model: str, job: Job) -> None:
        job.advance(0, 0, "", "test")
        started = time.monotonic()
        self.llm.route(backend).chat_json(model, task=t("動作テスト", "test"), system="Reply only with the requested JSON.",
                                          user={"request": "動作テストです。ok に true を入れて返してください。"}, schema=_TEST_SCHEMA,
                                          timeout=600)
        record = {**self._llm_identity(backend, model), "at": _now(), "test_ms": int((time.monotonic() - started) * 1000)}
        if backend == "builtin":
            record["device"] = self.builtin.state().get("device")
        with self._lock:
            self._data["checks"][self._llm_check_key(backend, model)] = record
            if backend == "ollama":
                self._data["checks"].pop(f"llm:{model}", None)
            self._write()

    def _llm_valid(self, backend: str, model: str, installed: dict[str, str] | None = None) -> bool:
        record = self._llm_record(backend, model)
        if not isinstance(record, dict):
            return False
        try:
            identity = self._llm_identity(backend, model, installed)
        except (LlmUnavailable, StopIteration):
            return False
        return all(record.get(key) == value for key, value in identity.items())

    def start_foundry_service(self) -> dict[str, Any]:
        if not self.foundry.installed():
            raise SetupError(t("Foundry Local はこの PC に入っていません。", "Foundry Local is not installed on this PC."))
        if self.foundry.service(refresh=True)["state"] == "running":
            raise SetupError(t("Foundry Local のサービスはもう動いています。", "The Foundry Local service is already running."))

        def work(job: Job) -> None:
            job.advance(0, 0, "", "start")
            self.foundry.start()

        return self._start("llm", "foundry", "start", work)

    def start_foundry_download(self, alias: str) -> dict[str, Any]:
        if not self.foundry.allows(alias):
            raise SetupError(t("Foundry Local で要約に許可されたモデルではありません。", "This model is not allowed for summaries with Foundry Local."), 422)
        try:
            if alias in self.foundry.variants():
                raise SetupError(t("このモデルは Foundry Local に入っています。「確認する」で応答を試せます。",
                                    "Foundry Local already has this model. Use \"Check\" to test its answer."))
        except LlmUnavailable as exc:
            raise SetupError(str(exc)) from exc
        gpu = bool((self._probe().get("gpu") or {}).get("name"))

        def work(job: Job) -> None:
            job.advance(0, 0, "", "pull")
            entry = self.foundry.catalog_entry(alias, cuda=gpu and self._cuda(), gpu=gpu)
            size = int(float(entry.get("fileSizeMb") or 0) * MIB)
            store = self.foundry.model_dir()
            free = hardware.disk_free(store) if store is not None else None
            if free is not None and size and free < size + 512 * MIB:
                raise SetupError(t(f"Foundry Local の保存先の空き容量が足りません（必要 {format_bytes(size)}、空き {format_bytes(free)}）。",
                                    f"Not enough free space in Foundry Local's folder (needs {format_bytes(size)}, free {format_bytes(free)})."))
            job.advance(0, size, str(entry.get("name") or alias), "pull")
            self.foundry.download(entry, lambda name, fraction: job.advance(int(fraction * size), size, name, "pull"), job.cancel.is_set)
            self._test_backend("foundry", alias, job)

        return self._start("llm", alias, "pull", work)

    def _check(self, key: str, valid: bool) -> dict[str, Any] | None:
        record = self._data["checks"].get(key)
        if not isinstance(record, dict) or not valid:
            return None
        return {name: record.get(name) for name in ("at", "load_ms", "test_ms", "device", "compute_type", "differs", "sample")
                if name != "sample" or record.get(name)}

    def _check_key(self, entry: ModelEntry) -> str:
        return self._llm_check_key("builtin", entry.id) if entry.component == "llm" else f"{entry.component}:{entry.id}"

    def _model_status(self, entry: ModelEntry, recommended: str | None) -> dict[str, Any]:
        folder, where = self.locate(entry)
        files = inspect(entry, folder)
        record = self._data["checks"].get(self._check_key(entry))
        valid = isinstance(record, dict) and record.get("fingerprint") == fingerprint(entry, folder)
        if valid and entry.component == "llm-runtime":
            valid = (folder / "bin" / LLAMA_EXE).is_file()
        if valid and entry.component == "llm":
            valid = self._llm_valid("builtin", entry.id)
        if valid and entry.component == "translator":
            valid = record.get("runtime") == self._runtime_identity()
        if valid and entry.component == "asr":
            env = {**self.effective_env(), "YATA_ASR_PROFILE": entry.profile or "auto", "YATA_ASR_MODEL_PATH": str(folder)}
            settings = resolve_asr_settings(env, cuda=self._cuda)
            valid = record.get("device") == settings.device and record.get("compute_type") == settings.compute_type
        if files["mismatched"]:
            state = "conflict"
        elif not files["present"]:
            state = "missing"
        elif files["missing"]:
            state = "incomplete"
        else:
            state = "ready" if valid else "unverified"
        partial = folder / PARTIAL
        partial_bytes = sum(path.stat().st_size for path in partial.glob("*.part")) if partial.is_dir() else 0
        job = self.jobs.get(entry.component)
        return {
            "id": entry.id, "label": entry.label, "summary": str(entry.summary), "target": entry.target, "size": entry.size,
            "folder": str(folder), "location": where, "source": entry.source, "page": entry.page, "license": entry.license,
            "state": state, "usable": files["usable"], "missing": files["missing"], "mismatched": files["mismatched"],
            "remaining": files["remaining"], "partial_bytes": partial_bytes, "recommended": entry.id == recommended,
            "need": memory_need(entry.size) if entry.component == "llm" else translator_need(entry) if entry.component == "translator" else None,
            "license_page": entry.license_page or None,
            "terms": [{"text": str(term.text), "link": str(term.link), "url": term.url} for term in entry.terms],
            "check": self._check(self._check_key(entry), valid and state == "ready"),
            "job": job.public() if job is not None and job.target == entry.id else None,
        }

    def _ollama_status(self, recommended: str | None) -> dict[str, Any]:
        try:
            installed = self.ollama.tags()
            ollama = {"state": "running", "version": self.ollama.version(), "reason": None}
        except OllamaUnavailable as exc:
            installed = {}
            ollama = {"state": "stopped" if installed_locally(self.env) else "absent", "version": None, "reason": str(exc)}
        store = store_dir(self.env, self.ollama.models_dir())
        ollama.update({"install_url": INSTALL_URL, "store": str(store), "free": hardware.disk_free(store)})
        job = self.jobs.get("llm")
        models = []
        for entry in LLM_MODELS:
            here = entry.name in installed
            valid = here and self._llm_valid("ollama", entry.name, installed)
            models.append({
                "name": entry.name, "size": entry.size, "need": entry.need, "summary": str(entry.summary),
                "installed": here, "recommended": entry.name == recommended,
                "state": "ready" if valid else "unverified" if here else "missing" if ollama["state"] == "running" else "unknown",
                "check": self._check(self._llm_check_key("ollama", entry.name), valid) or self._check(f"llm:{entry.name}", valid),
                "job": job.public() if job is not None and job.target == entry.name else None,
            })
        return {"ollama": ollama, "models": models}

    def _builtin_status(self) -> dict[str, Any]:
        runtime = self._runtime_entry()
        configured = self.env.get(LLAMA_ENV, "").strip()
        record = self._data["checks"].get(f"llm-runtime:{runtime.id}") if runtime is not None else None
        record = record if isinstance(record, dict) else {}
        models = [self._model_status(entry, None) for entry in self._builtin_entries()]
        return {
            "env": LLAMA_ENV if configured else None,
            "executable": self.llama_server() is not None,
            "runtime": self._model_status(runtime, None) if runtime is not None and not configured else None,
            "install": self._llama_install_hint(),
            "device": self.env.get(LLAMA_DEVICE_ENV, "").strip() or record.get("device"),
            "devices": record.get("devices") or [],
            "models": models,
            "server": self.builtin.state(),
        }

    def _openai_status(self) -> dict[str, Any]:
        config = self.openai_config()
        locked = [name for name in OPENAI_ENV if self.env.get(name, "").strip()]
        job = self.jobs.get("llm")
        result: dict[str, Any] = {**config, "key": self.openai.key_present(), "key_env": KEY_ENV, "locked": locked,
                                  "server_models": [], "reason": None, "check": None,
                                  "job": job.public() if job is not None and config["model"] and job.target == config["model"] else None}
        if not config["base_url"]:
            return {**result, "state": "unset"}
        try:
            listed = self.openai.server_models()
        except LlmUnavailable as exc:
            return {**result, "state": "unreachable", "reason": str(exc)}
        result["server_models"] = listed[:100]
        if not config["model"]:
            return {**result, "state": "unset"}
        if listed and config["model"] not in listed:
            return {**result, "state": "missing", "reason": t(f"サーバーにモデル {config['model']} がありません。", f"The server has no model {config['model']}.")}
        valid = self._llm_valid("openai", config["model"])
        return {**result, "state": "ready" if valid else "unverified",
                "check": self._check(self._llm_check_key("openai", config["model"]), valid)}

    def _foundry_status(self, recommended: str | None) -> dict[str, Any]:
        if not self.foundry.installed():
            return {"installed": False, "service": {"state": "absent", "reason": None}, "models": [], "job": None}
        service = self.foundry.service()
        variants: dict[str, str] = {}
        reason = service.get("reason")
        if service["state"] == "running":
            try:
                variants = self.foundry.variants()
            except LlmUnavailable as exc:
                reason = str(exc)
        job = self.jobs.get("llm")
        preferred = next((item.alias for item in FOUNDRY_MODELS if item.ollama == recommended), FOUNDRY_MODELS[0].alias)
        models = []
        for entry in FOUNDRY_MODELS:
            here = entry.alias in variants
            valid = here and self._llm_valid("foundry", entry.alias)
            models.append({
                "name": entry.alias, "summary": str(entry.summary), "installed": here, "variant": variants.get(entry.alias),
                "recommended": entry.alias == preferred,
                "state": "ready" if valid else "unverified" if here else "missing" if service["state"] == "running" else "unknown",
                "check": self._check(self._llm_check_key("foundry", entry.alias), valid),
                "job": job.public() if job is not None and job.target == entry.alias else None,
            })
        return {"installed": True, "service": {"state": service["state"], "reason": reason}, "models": models,
                "job": job.public() if job is not None and job.target == "foundry" else None}

    def _llm_status(self, recommended: str | None) -> dict[str, Any]:
        """Ollama's state keeps the keys it had before the choice existed ("ollama", "models")."""
        choice = self.llm_choice()
        ollama = self._ollama_status(recommended)
        builtin = self._builtin_status()
        openai = self._openai_status()
        foundry = self._foundry_status(recommended)
        usable = {
            "builtin": builtin["executable"] and any(model["usable"] for model in builtin["models"]),
            "ollama": ollama["ollama"]["state"] == "running" and any(model["installed"] for model in ollama["models"]),
            "openai": openai["state"] in {"ready", "unverified"},
            "foundry": any(model["installed"] for model in foundry["models"]),
        }
        offered = [kind for kind in BACKENDS if kind != "foundry" or foundry["installed"]]
        return {
            **ollama,
            "backend": choice["backend"], "source": choice["source"], "chosen": choice["backend"] is not None,
            "locked": LLM_ENV if choice["source"] == "env" else None,
            # A running Ollama that already has a model needs nothing more; otherwise the built-in one is the simplest,
            # unless the page cannot fetch llama.cpp here (macOS, Linux) and none is installed: then Ollama.
            "recommended": "ollama" if usable["ollama"] or builtin["install"] else "builtin",
            "options": [{"id": kind, "label": str(LABELS[kind]), "usable": bool(usable[kind])} for kind in offered],
            "builtin": builtin, "openai": openai, "foundry": foundry,
        }

    def _translator_status(self, advice: dict[str, Any], machine: dict[str, Any]) -> dict[str, Any] | None:
        current = self.translator_entry()
        if current is None:
            return None
        runtime = self._runtime_entry()
        configured = self.env.get(LLAMA_ENV, "").strip()
        record = self._data["checks"].get(f"llm-runtime:{runtime.id}") if runtime is not None else None
        choice = self.translator_choice()
        llm_need = next((item.need for item in LLM_MODELS if item.name == advice.get("llm")), None)
        models = []
        for entry in self._translator_entries():
            item = self._model_status(entry, advice.get("translator"))
            item.update(active=entry.id == current.id, fits_with_summaries=translator_fits(machine, llm_need, entry))
            models.append(item)
        ready = self.translator_ready()
        unchosen = "translator" if ready is not None and translator_fits(machine, llm_need, self._translator_by_id(ready)) else "llm"
        return {
            "env": LLAMA_ENV if configured else None,
            "executable": self.llama_server() is not None,
            "runtime": self._model_status(runtime, None) if runtime is not None and not configured else None,
            "install": self._llama_install_hint(),
            "device": self._llama_device(),
            "devices": (record.get("devices") if isinstance(record, dict) else None) or [],
            "model": next(item for item in models if item["active"]),
            "models": models,
            "model_chosen": choice["model"] is not None,
            "custom_path": str(self.custom_translator_path() or ""),
            "engine": choice["engine"] or unchosen,
            "engine_chosen": choice["engine"] is not None,
            "engine_recommended": "translator" if advice.get("translator_with_summaries") else "llm",
            "available": ready is not None,
            "languages": caption_languages(),
            "reason": advice.get("translator_reason"),
            "with_summaries": bool(advice.get("translator_with_summaries")),
            "server": self.translator.state(),
        }

    def translator_health(self) -> dict[str, Any]:
        entry = self.translator_entry()
        available = self.translator_ready() is not None
        missing = []
        if entry is not None and not available:
            if self.llama_server() is None:
                missing.append(t("llama.cpp の実行環境", "the llama.cpp runtime"))
            if entry.id not in self._translator_here():
                missing.append(entry.label)
        return {"id": entry.id if entry else None, "label": entry.label if entry else None, "available": available,
                "missing": missing, "languages": caption_languages() if entry else []}

    def llm_health(self) -> dict[str, Any]:
        backend = self.llm_backend()
        if backend is None:
            return {"backend": None, "label": None, "chosen": False, "connected": False, "models": [],
                    "reason": t("要約の LLM がまだ選ばれていません。「この PC の準備」で選べます。",
                                "No summary LLM is chosen yet. Choose one under \"Set up this PC\".")}
        try:
            names = self.llm.route(backend).models()
            connected, reason = True, None
        except LlmUnavailable as exc:
            names, connected, reason = [], False, str(exc)
        labels = {entry.id: entry.label for entry in self._builtin_entries()}
        return {"backend": backend, "label": str(LABELS[backend]), "chosen": True, "connected": connected,
                "models": [{"id": name, "label": labels.get(name, name)} for name in names], "reason": reason}

    def status(self, active_asr: dict[str, Any] | None = None, speakers_enabled: bool = False) -> dict[str, Any]:
        machine = self._probe()
        root, source = self.model_root()
        advice = recommend(machine)
        current = self.asr_entry()
        asr_models = [self._model_status(entry, advice["asr"]) for entry in self.catalog.values() if entry.component == "asr"]
        for item in asr_models:
            item["active"] = current is not None and item["id"] == current.id
        speaker_entry = next((entry for entry in self.catalog.values() if entry.component == "speakers"), None)
        speaker_env = self.env.get("YATA_SPEAKER_MODEL_PATH", "").strip()
        return {
            "hardware": machine,
            "storage": {"path": str(root), "source": source, "free": hardware.disk_free(root)},
            "plan": self.plan(),
            "recommended": advice,
            "asr": {"locked": self.asr_locked(), "chosen": self._data.get("asr_model"), "active": active_asr, "models": asr_models},
            "vad": _vad_status(),
            "llm": self._llm_status(advice["llm"]),
            "translator": self._translator_status(advice, machine),
            "speakers": {
                "package": installed_package("sherpa_onnx"), "command": "uv sync --inexact --extra speakers",
                "enabled": speakers_enabled, "env": bool(speaker_env),
                "model": self._model_status(speaker_entry, speaker_entry.id) if speaker_entry and not speaker_env else None,
            },
            "jobs": {name: job.public() for name, job in self.jobs.items()},
        }

    def brief(self, asr_ready: bool, summary_models: list[Any], speakers_enabled: bool) -> dict[str, Any]:
        plan = self.plan()
        attention = [] if asr_ready else ["asr"]
        if plan and plan["summaries"] and not summary_models:
            attention.append("llm")
        translates = plan and (plan["translation"] or (plan["summaries"] and self.translator_choice()["engine"] == "translator"))
        if translates and self.translator_ready() is None:
            attention.append("translator")
        if plan and plan["speakers"] and not speakers_enabled:
            attention.append("speakers")
        backend = self.llm_backend()
        return {"plan": plan, "attention": attention, "busy": any(job.state == "running" for job in self.jobs.values()),
                "llm": {"backend": backend, "chosen": backend is not None}}

    def close(self) -> None:
        self.llm.close()
        self.translator.close()


def _vad_status() -> dict[str, Any]:
    spec = importlib.util.find_spec("faster_whisper")
    path = Path(spec.origin).parent / "assets" / "silero_vad_v6.onnx" if spec and spec.origin else None
    ready = path is not None and path.is_file()
    return {"state": "ready" if ready else "missing", "label": "Silero VAD v6",
            "detail": t("faster-whisper に同梱されています。個別の取得は不要です。",
                        "Included with faster-whisper; nothing to fetch.") if ready
            else t("faster-whisper に同梱のファイルが見つかりません。uv sync をやり直してください。",
                   "The file included with faster-whisper is missing. Run uv sync again.")}
