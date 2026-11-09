from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from yata import preparation as preparation_module
from yata.api import create_app
from yata.archive import SessionArchive
from yata.catalog import GIB, LLM_MODELS, ModelEntry, RemoteFile, recommend
from yata.download import PARTIAL, DownloadError, fetch, inspect, verify_hashes
from yata.features import Features
from yata.inputs.asr import LocalAsr
from yata.llm.foundry import FoundryLocal
from yata.models import SessionStore
from yata.ollama import SUMMARY_MODELS, OllamaUnavailable, PullCancelled
from yata.preparation import Job, Preparation, SetupError
from yata.state import SessionManager
from yata.topics import TopicPipeline

FILES = {
    "config.json": b'{"model": "tiny"}',
    "model.bin": bytes(range(256)) * 4000 + b"end",
    "tokenizer.json": b'{"t": 1}' * 500,
}
SPEAKER = {"speaker.onnx": b"onnx" * 2000}
GPU_PC = {"cpu": {"name": "cpu", "logical": 8}, "memory": {"total": 64 * GIB, "available": 32 * GIB},
          "gpu": {"name": "Test GPU", "vram": 32 * GIB, "cuda": True}}
CPU_PC = {"cpu": {"name": "cpu", "logical": 4}, "memory": {"total": 8 * GIB, "available": 4 * GIB}, "gpu": None}


def no_foundry() -> FoundryLocal:
    return FoundryLocal(["qwen3-8b"], which=lambda _name: None)


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def remote(files: dict[str, bytes]) -> tuple[RemoteFile, ...]:
    return tuple(RemoteFile(name, len(data), sha256=hashlib.sha256(data).hexdigest()) if name.endswith((".bin", ".onnx"))
                 else RemoteFile(name, len(data), git_sha1=git_sha1(data)) for name, data in files.items())


def entry(base: str, *, id: str = "tiny-light", profile: str | None = "light", folder: str = "faster-whisper-small",
          files: dict[str, bytes] = FILES, component: str = "asr") -> ModelEntry:
    return ModelEntry(id=id, component=component, label=id, folder=folder, source="test", page="http://example.invalid",
                      license="MIT", base_url=base, files=remote(files), summary="", profile=profile, target="cpu",
                      essential=tuple(name for name in files if name in ("config.json", "model.bin", "speaker.onnx")))


class FileServer:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = dict(files)
        self.requests: list[tuple[str, str | None]] = []
        self.fail: int | None = None
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                name = self.path.rsplit("/", 1)[-1]
                outer.requests.append((name, self.headers.get("Range")))
                data = outer.files.get(name)
                if outer.fail or data is None:
                    self.send_error(outer.fail or 404)
                    return
                start = 0
                wanted = self.headers.get("Range")
                if wanted and wanted.startswith("bytes="):
                    start = int(wanted[6:].split("-")[0])
                    if start >= len(data):
                        self.send_error(416)
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                self.wfile.write(data[start:])

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/repo"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class FakeEngine:
    settings = None
    name = "fake"

    def __init__(self, env: dict[str, str], fail: str | None) -> None:
        self.env = env
        self.fail = fail
        self.loaded = False
        self.heard: list[int] = []

    def readiness(self) -> dict:
        return {"state": "weights_present", "weights_present": True, "reason": ""}

    def load(self) -> None:
        if self.fail:
            raise RuntimeError(self.fail)
        self.loaded = True

    def transcribe(self, audio, language=None, prompt=None):
        self.heard.append(len(audio))
        return []


class FakeOllama:
    def __init__(self, installed: dict[str, str] | None = None) -> None:
        self.installed = dict(installed or {})
        self.up = True
        self.pulls: list[str] = []
        self.chats: list[str] = []
        self.fail_pull: str | None = None
        self.cancel_pull = False

    def tags(self) -> dict[str, str]:
        if not self.up:
            raise OllamaUnavailable("Ollama に接続できません。")
        return {name: self.installed[name] for name in SUMMARY_MODELS if name in self.installed}

    def models(self) -> list[str]:
        return list(self.tags())

    def allows(self, model: str) -> bool:
        return model in SUMMARY_MODELS

    def version(self) -> str:
        return "0.40.0"

    def models_dir(self) -> Path | None:
        return None  # as an Ollama with no model to ask about: the folder comes from OLLAMA_MODELS

    def pull(self, model, on_progress, cancelled):
        self.pulls.append(model)
        on_progress("pulling 8eeb", 50, 100)
        if self.fail_pull:
            raise OllamaUnavailable(self.fail_pull)
        if self.cancel_pull:
            raise PullCancelled("中止しました。")
        on_progress("success", 100, 100)
        self.installed[model] = "sha256:pulled"

    def chat_json(self, model, *, task, system, user, schema, timeout=120.0):
        self.chats.append(model)
        return {"ok": True}


class RecommendTests(unittest.TestCase):
    def test_a_large_gpu_gets_turbo_and_the_best_summary_model_that_fits(self):
        advice = recommend(GPU_PC)
        self.assertEqual((advice["plan"], advice["asr"], advice["llm"]), ("summaries", "whisper-turbo", "qwen3.6:35b-a3b"))
        self.assertEqual(advice["plan_reason"], "", "summaries need no advice of their own")
        self.assertIn(GPU_PC["gpu"]["name"], advice["llm_reason"], "the reason names this PC's GPU, not a fixed one")

    def test_a_smaller_gpu_gets_a_model_that_fits_beside_the_asr(self):
        advice = recommend({**GPU_PC, "gpu": {"name": "GPU", "vram": 12 * GIB, "cuda": True}})
        self.assertEqual((advice["plan"], advice["llm"]), ("summaries", "qwen3:8b"))
        tight = recommend({**GPU_PC, "memory": {"total": 16 * GIB}, "gpu": {"name": "GPU", "vram": 8 * GIB, "cuda": True}})
        self.assertEqual((tight["plan"], tight["asr"], tight["llm"]), ("captions", "whisper-turbo", "qwen3:8b"))
        self.assertIn("VRAM 8 GB", tight["llm_reason"])

    def test_a_pc_without_cuda_gets_small_and_captions_only(self):
        self.assertEqual({key: recommend(CPU_PC)[key] for key in ("plan", "asr", "llm")},
                         {"plan": "captions", "asr": "whisper-small", "llm": None})
        roomy = recommend({**CPU_PC, "memory": {"total": 32 * GIB}})
        self.assertEqual((roomy["plan"], roomy["llm"]), ("captions", "qwen3:8b"))
        no_cuda = recommend({**CPU_PC, "gpu": {"name": "Old GPU", "vram": 4 * GIB, "cuda": False}})
        self.assertEqual(no_cuda["asr"], "whisper-small")
        self.assertIn("Old GPU", no_cuda["asr_reason"])

    def test_the_summary_catalog_follows_the_allowlist(self):
        self.assertEqual(tuple(item.name for item in LLM_MODELS), SUMMARY_MODELS)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.server = FileServer(FILES)
        self.addCleanup(self.server.close)
        self.folder = Path(tempfile.mkdtemp(prefix="yata-dl-")) / "model"
        self.addCleanup(shutil.rmtree, self.folder.parent, True)
        self.entry = entry(self.server.base)

    def test_missing_files_download_into_place_and_match_the_source(self):
        steps = []
        fetch(self.entry, self.folder, progress=lambda done, total, name, phase: steps.append((done, total, phase)))
        self.assertTrue(inspect(self.entry, self.folder)["complete"])
        self.assertEqual(verify_hashes(self.entry, self.folder), [])
        self.assertFalse((self.folder / PARTIAL).exists())
        self.assertEqual(steps[-1][0], self.entry.size)
        self.assertIn("verify", {phase for *_, phase in steps})

    def test_present_files_are_kept_and_a_partial_file_resumes(self):
        self.folder.mkdir(parents=True)
        (self.folder / "config.json").write_bytes(FILES["config.json"])
        (self.folder / PARTIAL).mkdir()
        (self.folder / PARTIAL / "model.bin.part").write_bytes(FILES["model.bin"][:1000])
        fetch(self.entry, self.folder)
        self.assertEqual(dict(self.server.requests), {"model.bin": "bytes=1000-", "tokenizer.json": None})
        self.assertEqual(verify_hashes(self.entry, self.folder), [])

    def test_a_damaged_download_is_discarded(self):
        damaged = bytearray(FILES["model.bin"])
        damaged[10] ^= 0xFF
        self.server.files["model.bin"] = bytes(damaged)
        with self.assertRaises(DownloadError) as caught:
            fetch(self.entry, self.folder)
        self.assertEqual(caught.exception.kind, "corrupt")
        self.assertFalse((self.folder / "model.bin").exists())
        self.assertFalse((self.folder / PARTIAL / "model.bin.part").exists())

    def test_a_different_existing_file_is_never_overwritten(self):
        self.folder.mkdir(parents=True)
        (self.folder / "model.bin").write_bytes(b"mine")
        with self.assertRaises(DownloadError) as caught:
            fetch(self.entry, self.folder)
        self.assertEqual(caught.exception.kind, "conflict")
        self.assertEqual((self.folder / "model.bin").read_bytes(), b"mine")
        self.assertEqual(self.server.requests, [])

    def test_failures_say_what_went_wrong(self):
        cases = []
        with self.assertRaises(DownloadError) as caught:
            fetch(self.entry, self.folder, margin=10 ** 18)
        cases.append(caught.exception.kind)
        self.server.fail = 503
        with self.assertRaises(DownloadError) as caught:
            fetch(self.entry, self.folder)
        cases.append(caught.exception.kind)
        self.server.fail = None
        with self.assertRaises(DownloadError) as caught:
            fetch(entry("http://127.0.0.1:9/repo"), self.folder)
        cases.append(caught.exception.kind)
        with self.assertRaises(DownloadError) as caught:
            fetch(self.entry, self.folder, cancelled=lambda: True)
        cases.append(caught.exception.kind)
        self.assertEqual(cases, ["disk", "network", "network", "cancelled"])

    def test_a_changed_file_of_the_same_size_is_found(self):
        fetch(self.entry, self.folder)
        changed = bytearray(FILES["tokenizer.json"])
        changed[0] = ord("[")
        (self.folder / "tokenizer.json").write_bytes(bytes(changed))
        self.assertEqual(verify_hashes(self.entry, self.folder), ["tokenizer.json"])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.server = FileServer({**FILES, **SPEAKER})
        self.addCleanup(self.server.close)
        self.root = Path(tempfile.mkdtemp(prefix="yata-prep-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.light = entry(self.server.base)
        self.full = entry(self.server.base, id="tiny-full", profile="full", folder="faster-whisper-large-v3-turbo")
        self.speaker = entry(self.server.base, id="tiny-speaker", profile=None, folder="speaker", files=SPEAKER, component="speakers")
        self.ollama = FakeOllama({"qwen3:8b": "sha256:a"})
        self.load_error: str | None = None
        self.cuda = False
        self.built: list[FakeEngine] = []
        self.installed: list[tuple] = []

    def prep(self, **env) -> Preparation:
        environ = {"OLLAMA_MODELS": str(self.root / "ollama"), **env}

        def engines(environment, cuda=None):
            engine = FakeEngine(environment, self.load_error)
            self.built.append(engine)
            return engine, None

        class Embedder:
            def __init__(self, path):
                self.path = path

            def embed(self, audio):
                return np.ones(4, dtype=np.float32)

        prep = Preparation(self.root / "data", ollama=self.ollama, environ=environ, probe=lambda: CPU_PC, cuda=lambda: self.cuda,
                           engines=engines, embedder=Embedder, catalog=(self.full, self.light, self.speaker), background=False,
                           foundry=no_foundry())
        prep.on_asr = lambda final, partial: self.installed.append(("asr", final))
        prep.on_speakers = lambda embedder: self.installed.append(("speakers", embedder))
        return prep

    def model(self, status, entry_id):
        return next(item for item in status["asr"]["models"] + [status["speakers"]["model"]] if item["id"] == entry_id)

    def test_the_model_folder_comes_from_env_then_the_page_then_the_data_folder(self):
        prep = self.prep()
        self.assertEqual(prep.model_root(), (self.root / "data" / "models", "default"))
        chosen = self.root / "elsewhere"
        prep.set_model_root(str(chosen))
        self.assertEqual(self.prep().model_root(), (chosen, "settings"))
        for bad in ("relative\\models", str(self.root / "data" / "setup.json")):
            with self.assertRaises(SetupError) as caught:
                prep.set_model_root(bad)
            self.assertEqual(caught.exception.status, 422)
        prep.set_model_root("")
        self.assertEqual(prep.model_root()[1], "default")
        fixed = self.prep(YATA_MODEL_ROOT=str(chosen))
        self.assertEqual(fixed.model_root(), (chosen, "env"))
        with self.assertRaises(SetupError):
            fixed.set_model_root(str(self.root))

    def test_the_asr_model_suits_the_pc_unless_chosen_and_env_settings_fix_it(self):
        prep = self.prep()
        self.assertEqual(prep.asr_entry().id, "tiny-light")
        self.cuda = True
        self.assertEqual(prep.asr_entry().id, "tiny-full")
        self.cuda = False
        fetch(self.full, prep.locate(self.full)[0])
        self.assertEqual(prep.asr_entry().id, "tiny-full")
        self.assertEqual(prep.effective_env()["YATA_ASR_PROFILE"], "full")
        prep.choose_asr("tiny-light")
        self.assertEqual(self.prep().asr_entry().id, "tiny-light")
        self.assertEqual(self.installed[-1][0], "asr")
        locked = self.prep(YATA_ASR_PROFILE="light")
        self.assertIsNone(locked.asr_entry())
        with self.assertRaises(SetupError):
            locked.choose_asr("tiny-full")

    def test_a_forced_cpu_setup_never_probes_cuda(self):
        # Probing initialises the CUDA runtime (hundreds of MiB); a light PC forced to CPU must not pay it.
        asked = []
        prep = Preparation(self.root / "data", ollama=self.ollama, environ={"YATA_ASR_PROFILE": "light"}, probe=lambda: CPU_PC,
                           cuda=lambda: asked.append(1) or True, background=False)
        final, partial = prep.build_asr()
        self.assertEqual((final.settings.device, final.settings.compute_type, partial, asked), ("cpu", "int8", None, []))

    def test_a_download_loads_and_tests_the_model_before_it_counts_as_ready(self):
        prep = self.prep()
        self.assertEqual(self.model(prep.status(), "tiny-light")["state"], "missing")
        job = prep.start_download("tiny-light")
        self.assertEqual(job["state"], "done")  # jobs run inline here
        status = prep.status()
        light = self.model(status, "tiny-light")
        self.assertEqual((light["state"], light["job"]["state"], light["active"]), ("ready", "done", True))
        self.assertEqual((light["check"]["device"], light["check"]["compute_type"]), ("cpu", "int8"))
        engine = self.built[-1]
        self.assertTrue(engine.loaded)
        self.assertEqual(engine.heard, [32_000])
        self.assertEqual(engine.env["YATA_ASR_MODEL_PATH"], str(prep.locate(self.light)[0]))
        self.assertIs(self.installed[-1][1], engine)
        with self.assertRaises(SetupError):
            prep.start_download("tiny-light")
        model_bin = prep.locate(self.light)[0] / "model.bin"
        os.utime(model_bin, ns=(model_bin.stat().st_atime_ns, model_bin.stat().st_mtime_ns + 10_000_000_000))
        self.assertEqual(self.model(self.prep().status(), "tiny-light")["state"], "unverified")

    def test_load_failures_and_damaged_files_are_told_apart(self):
        prep = self.prep()
        fetch(self.full, prep.locate(self.full)[0])
        self.cuda = True
        self.load_error = "Library cublas64_12.dll is not found or cannot be loaded"
        prep.start_verify("tiny-full")
        job = prep.jobs["asr"].public()
        self.assertEqual((job["state"], job["error"]["kind"]), ("failed", "runtime"))
        self.assertIn("GPU（CUDA）", job["error"]["message"])
        changed = bytearray(FILES["model.bin"])
        changed[5] ^= 0x01
        (prep.locate(self.full)[0] / "model.bin").write_bytes(bytes(changed))
        prep.start_verify("tiny-full")
        job = prep.jobs["asr"].public()
        self.assertEqual(job["error"]["kind"], "corrupt")
        self.assertIn("model.bin", job["note"])
        with self.assertRaises(SetupError):
            prep.start_verify("tiny-light")

    def test_speaker_labels_need_the_model_and_the_package(self):
        prep = self.prep()
        with mock.patch.object(preparation_module, "installed_package", return_value=False):
            prep.start_download("tiny-speaker")
            status = prep.status()
        speaker = self.model(status, "tiny-speaker")
        self.assertEqual((speaker["state"], speaker["job"]["state"]), ("unverified", "done"))
        self.assertIn("sherpa-onnx", speaker["job"]["note"])
        self.assertFalse(status["speakers"]["package"])
        with mock.patch.object(preparation_module, "installed_package", return_value=True):
            prep.start_verify("tiny-speaker")
            self.assertEqual(self.model(prep.status(), "tiny-speaker")["state"], "ready")
        self.assertEqual(self.installed[-1][0], "speakers")
        self.assertEqual(prep.speaker_model_path(), prep.locate(self.speaker)[0] / "speaker.onnx")

    def test_summary_models_are_pulled_by_ollama_and_answer_a_test(self):
        prep = self.prep()
        llm = {item["name"]: item for item in prep.status()["llm"]["models"]}
        self.assertEqual((llm["qwen3:8b"]["state"], llm["qwen3.6:35b-a3b"]["state"]), ("unverified", "missing"))
        for name, status in (("evil:1b", 422), ("qwen3:8b", 409)):
            with self.assertRaises(SetupError) as caught:
                prep.start_pull(name)
            self.assertEqual(caught.exception.status, status)
        prep.start_pull("mistral-nemo:12b")
        self.assertEqual((self.ollama.pulls, self.ollama.chats), (["mistral-nemo:12b"], ["mistral-nemo:12b"]))
        llm = {item["name"]: item for item in prep.status()["llm"]["models"]}
        self.assertEqual(llm["mistral-nemo:12b"]["state"], "ready")
        prep.start_llm_test("qwen3:8b")
        self.assertEqual({item["name"]: item["state"] for item in prep.status()["llm"]["models"]}["qwen3:8b"], "ready")
        self.ollama.installed["qwen3:8b"] = "sha256:b"
        self.assertEqual({item["name"]: item["state"] for item in prep.status()["llm"]["models"]}["qwen3:8b"], "unverified")
        self.ollama.fail_pull = "pull model manifest: file does not exist"
        prep.start_pull("qwen3.8:27b")
        self.assertEqual(prep.jobs["llm"].error["kind"], "ollama")
        self.ollama.fail_pull, self.ollama.cancel_pull = None, True
        prep.start_pull("qwen3.8:27b")
        self.assertEqual(prep.jobs["llm"].state, "cancelled")
        self.ollama.up = False
        status = prep.status()["llm"]
        self.assertIn(status["ollama"]["state"], ("stopped", "absent"))
        self.assertEqual({item["state"] for item in status["models"]}, {"unknown"})

    def test_one_job_per_item_and_cancelling_it(self):
        prep = self.prep()
        prep.jobs["asr"] = Job(component="asr", target="tiny-light", kind="download")
        with self.assertRaises(SetupError):
            prep.start_download("tiny-full")
        with self.assertRaises(SetupError):
            prep.set_model_root(str(self.root / "other"))
        prep.cancel("asr")
        self.assertTrue(prep.jobs["asr"].cancel.is_set())
        with self.assertRaises(SetupError):
            prep.cancel("llm")

    def test_choices_are_saved_and_the_health_brief_names_what_is_missing(self):
        prep = self.prep()
        self.assertIsNone(prep.plan())
        prep.set_plan(summaries=True, speakers=True)
        reopened = self.prep()
        self.assertEqual(reopened.plan(), {"summaries": True, "translation": False, "speakers": True})
        self.assertEqual(reopened.brief(False, [], False)["attention"], ["asr", "llm", "speakers"])
        self.assertEqual(reopened.brief(True, ["qwen3:8b"], True)["attention"], [])
        saved = json.loads((self.root / "data" / "setup.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["plan"], {"summaries": True, "translation": False, "speakers": True})


class SetupApiTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-setup-api-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.ollama = FakeOllama()
        built = []

        def engines(environment, cuda=None):
            engine = FakeEngine(environment, None)
            built.append(engine)
            return engine, None

        self.built = built
        prep = Preparation(self.root / "data", ollama=self.ollama, environ={"OLLAMA_MODELS": str(self.root)}, probe=lambda: CPU_PC,
                           cuda=lambda: False, engines=engines, background=False, foundry=no_foundry())
        manager = SessionManager(TopicPipeline(self.ollama), SessionStore(), background=False, archive=SessionArchive(self.root / "sessions"))
        self.addCleanup(manager.close)
        self.app = create_app(access_token="t", ollama=self.ollama, asr=LocalAsr(None), manager=manager, features=Features(),
                              preparation=prep)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8765")

    def test_the_setup_api_needs_the_app_session(self):
        self.assertEqual(self.client.get("/api/v1/setup").status_code, 401)
        self.client.get("/")
        setup = self.client.get("/api/v1/setup").json()["setup"]
        self.assertEqual(set(setup) >= {"hardware", "storage", "plan", "recommended", "asr", "vad", "llm", "speakers", "jobs"}, True)
        self.assertEqual(setup["recommended"]["asr"], "whisper-small")
        self.assertEqual({item["id"] for item in setup["asr"]["models"]}, {"whisper-turbo", "whisper-small"})

    def test_health_names_what_the_chosen_use_lacks(self):
        self.client.get("/")
        self.assertEqual(self.client.get("/api/v1/health").json()["setup"]["attention"], ["asr"])
        saved = self.client.put("/api/v1/setup/plan", json={"summaries": True, "speakers": False})
        self.assertEqual(saved.json()["setup"]["plan"], {"summaries": True, "translation": False, "speakers": False})
        self.assertEqual(self.client.get("/api/v1/health").json()["setup"]["attention"], ["asr", "llm"])

    def test_choices_and_jobs_are_checked_on_the_server(self):
        self.client.get("/")
        self.assertEqual(self.client.put("/api/v1/setup/storage", json={"path": "models"}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/setup/models/nope/download").status_code, 404)
        self.assertEqual(self.client.post("/api/v1/setup/llm/pull", json={"model": "evil:1b"}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/setup/jobs/asr/cancel").status_code, 409)
        folder = self.root / "models"
        moved = self.client.put("/api/v1/setup/storage", json={"path": str(folder)})
        self.assertEqual(moved.json()["setup"]["storage"], {"path": str(folder), "source": "settings", "free": moved.json()["setup"]["storage"]["free"]})
        chosen = self.client.put("/api/v1/setup/asr", json={"model": "whisper-turbo"})
        self.assertEqual(chosen.status_code, 200)
        self.assertIs(self.app.state.asr, self.built[-1])
        self.assertEqual(self.built[-1].env["YATA_ASR_PROFILE"], "full")
        self.assertEqual(self.built[-1].env["YATA_MODEL_ROOT"], str(folder))


if __name__ == "__main__":
    unittest.main()
