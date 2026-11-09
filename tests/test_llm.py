from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from yata.api import create_app
from yata.archive import SessionArchive, from_record, to_record
from yata.catalog import ModelEntry, RemoteFile
from yata.features import Features
from yata.inputs.asr import LocalAsr
from yata.llm.base import LlmUnavailable, loopback_url, reply_json
from yata.llm.builtin import BuiltinBackend, LaunchSpec, parse_devices, pick_device
from yata.llm.chat import ChatClient
from yata.llm.foundry import FoundryLocal, variant_alias
from yata.llm.openai_compat import OpenAICompatBackend
from yata.llm.router import LlmRouter
from yata.models import SessionStore
from yata.ollama import SUMMARY_MODELS, OllamaClient, store_dir
from yata.preparation import Preparation, SetupError
from yata.state import SessionError, SessionManager
from yata.topics import TopicPipeline

OK_REPLY = {"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}]}
DEVICES = """ggml_vulkan: Found 2 Vulkan devices:
Available devices:
  Vulkan0: NVIDIA GeForce RTX 5090 (32187 MiB, 31419 MiB free)
  Vulkan1: AMD Radeon(TM) Graphics (157355 MiB, 149487 MiB free)
"""
GPU_PC = {"cpu": {"name": "cpu", "logical": 8}, "memory": {"total": 64 * 1024 ** 3}, "gpu": {"name": "NVIDIA GeForce RTX 5090", "vram": 32 * 1024 ** 3, "cuda": True}}


class FakeServer:
    def __init__(self, reply=None, port: int = 0) -> None:
        self.requests: list[tuple[str, str, dict, object]] = []
        self.reply = reply or (lambda method, path, body: (200, OK_REPLY))
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _serve(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                outer.requests.append((method, self.path, dict(self.headers), body))
                status, payload, *rest = outer.reply(method, self.path, body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                for key, value in (rest[0] if rest else {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._serve("GET")

            def do_POST(self):
                self._serve("POST")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def paths(self, method: str | None = None) -> list[str]:
        return [path for verb, path, _headers, _body in self.requests if method in (None, verb)]

    def bodies(self, path: str) -> list:
        return [body for _verb, at, _headers, body in self.requests if at == path]


class FakeLlama:
    def __init__(self, args, fail: bytes | None = None, reply=None, **_kwargs) -> None:
        self.args = list(args)
        self.returncode = 1 if fail else None
        self.stderr = io.BytesIO(fail or b"main: server is listening\n")
        self.server = None if fail else FakeServer(reply, port=int(args[args.index("--port") + 1]))

    def poll(self):
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            self.returncode = 0
            self.server.close()

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


class FakeOllama:
    def __init__(self, installed=None):
        self.installed = dict(installed or {})
        self.chats: list[str] = []

    def tags(self):
        return {name: self.installed[name] for name in SUMMARY_MODELS if name in self.installed}

    def models(self):
        return list(self.tags())

    def allows(self, model):
        return model in SUMMARY_MODELS

    def version(self):
        return "0.40.0"

    def models_dir(self):
        return None

    def chat_json(self, model, **_kwargs):
        self.chats.append(model)
        return {"ok": True}


def no_foundry() -> FoundryLocal:
    return FoundryLocal(["qwen3-8b"], which=lambda _name: None)


class RuleTests(unittest.TestCase):
    def test_only_servers_on_this_pc_are_accepted(self):
        self.assertEqual(loopback_url("http://127.0.0.1:1234/v1/"), "http://127.0.0.1:1234/v1")
        for good in ("http://localhost:1234/v1", "http://[::1]:8080/v1", "https://127.0.0.5:9000"):
            loopback_url(good)
        for bad in ("http://192.168.1.5:1234/v1", "https://api.openai.com/v1", "http://127.0.0.1.example.com/v1",
                    "http://user:pw@127.0.0.1:1/v1", "ftp://127.0.0.1/v1", "http://127.0.0.1:8765/v1",
                    "http://127.0.0.1:1/v1?x=1", "127.0.0.1:1234", "http://0.0.0.0:1234/v1", ""):
            with self.assertRaises(ValueError, msg=bad):
                loopback_url(bad)

    def test_replies_are_read_past_thinking_and_code_fences(self):
        self.assertEqual(reply_json('<think>まず…</think>\n```json\n{"ok": true}\n```'), {"ok": True})
        self.assertEqual(reply_json('Here it is: {"a": 1} done'), {"a": 1})
        self.assertIsNone(reply_json("[1, 2]"))
        self.assertIsNone(reply_json("no json"))
        self.assertIsNone(reply_json(None))


class ChatClientTests(unittest.TestCase):
    def serve(self, reply=None) -> FakeServer:
        server = FakeServer(reply)
        self.addCleanup(server.close)
        return server

    def test_the_schema_constrains_the_reply_and_a_server_without_it_gets_it_in_the_prompt(self):
        def reply(method, path, body):
            if body and "response_format" in body:
                return 400, {"error": "response_format is not supported"}
            return 200, OK_REPLY

        server = self.serve(reply)
        client = ChatClient(server.url + "/v1", label="test")
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        self.assertEqual(client.chat_json("m", task="t", system="sys", user={"a": "字幕"}, schema=schema), {"ok": True})
        first, second = server.bodies("/v1/chat/completions")
        self.assertEqual(first["response_format"]["json_schema"]["schema"], schema)
        self.assertNotIn("response_format", second)
        self.assertIn("JSON schema", second["messages"][0]["content"])
        self.assertEqual(json.loads(second["messages"][1]["content"]), {"a": "字幕"})
        self.assertEqual(client.structured, "prompt")

    def test_redirects_and_proxies_are_never_followed(self):
        elsewhere = self.serve()
        server = self.serve(lambda method, path, body: (302, b"", {"Location": elsewhere.url + "/v1/chat/completions"}))
        client = ChatClient(server.url + "/v1", label="test")
        with self.assertRaises(LlmUnavailable):
            client.chat_json("m", task="t", system="s", user={}, schema={})
        self.assertEqual(elsewhere.requests, [], "a redirect never carries the transcript elsewhere")
        direct = self.serve()
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9", "NO_PROXY": ""}):
            self.assertEqual(ChatClient(direct.url + "/v1", label="t").chat_json("m", task="t", system="s", user={}, schema={}), {"ok": True})

    def test_an_off_pc_url_is_refused_before_any_request(self):
        with self.assertRaises(ValueError):
            ChatClient("https://api.example.com/v1", label="test")

    def test_the_openai_backend_uses_the_configured_model_and_a_key_from_the_environment_only(self):
        server = self.serve(lambda method, path, body: (200, {"data": [{"id": "local-model"}]} if path == "/v1/models" else OK_REPLY))
        config = {"base_url": server.url + "/v1", "model": "local-model"}
        backend = OpenAICompatBackend(lambda: config, {"YATA_OPENAI_API_KEY": "sk-test"})
        self.assertEqual(backend.models(), ["local-model"])
        self.assertTrue(backend.allows("local-model"))
        self.assertFalse(backend.allows("qwen3:8b"))
        self.assertEqual(backend.chat_json("local-model", task="t", system="s", user={}, schema={}), {"ok": True})
        self.assertEqual(server.requests[-1][2].get("Authorization"), "Bearer sk-test")
        with self.assertRaises(LlmUnavailable):
            backend.chat_json("other", task="t", system="s", user={}, schema={})
        config["model"] = "missing-model"
        with self.assertRaises(LlmUnavailable):
            backend.models()
        config["base_url"] = "https://api.openai.com/v1"
        with self.assertRaises(LlmUnavailable):
            backend.models()


class BuiltinTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-llama-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.exe = self.root / "llama-server.exe"
        self.exe.write_bytes(b"MZ")
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"GGUF")
        self.spawned: list[FakeLlama] = []
        self.fail: bytes | None = None
        self.reply = None

    def backend(self, **options) -> BuiltinBackend:
        def spawn(args, **kwargs):
            process = FakeLlama(args, self.fail, reply=self.reply, **kwargs)
            self.spawned.append(process)
            return process

        def resolve(model):
            return LaunchSpec(self.exe, self.model, model, "Vulkan0")

        backend = BuiltinBackend(resolve=resolve, available=lambda: ["m1"], allowed=["m1", "m2"], spawn=spawn, **options)
        self.addCleanup(backend.close)
        return backend

    def test_the_server_starts_on_first_use_on_this_pc_offline_and_is_reused(self):
        backend = self.backend()
        self.assertEqual(self.spawned, [], "nothing runs until a model is asked for")
        self.assertEqual(backend.chat_json("m1", task="t", system="s", user={}, schema={}), {"ok": True})
        self.assertEqual(backend.chat_json("m1", task="t", system="s", user={}, schema={}), {"ok": True})
        self.assertEqual(len(self.spawned), 1)
        args = self.spawned[0].args
        for flag in ("--offline", "--no-webui"):
            self.assertIn(flag, args)
        for flag, value in (("--host", "127.0.0.1"), ("--reasoning", "off"), ("--device", "Vulkan0"), ("-m", str(self.model)), ("--alias", "m1")):
            self.assertEqual(args[args.index(flag) + 1], value)
        sent = self.spawned[0].server.bodies("/v1/chat/completions")[0]
        self.assertEqual((sent["model"], sent["response_format"]["type"]), ("m1", "json_schema"))
        self.assertTrue(backend.state()["running"])
        backend.chat_json("m2", task="t", system="s", user={}, schema={})
        self.assertEqual((len(self.spawned), self.spawned[0].returncode), (2, 0))
        with self.assertRaises(LlmUnavailable):
            backend.chat_json("evil", task="t", system="s", user={}, schema={})

    def test_it_stops_when_idle_released_or_closed(self):
        backend = self.backend(idle_s=0.2)
        backend.warm("m1")
        deadline = time.monotonic() + 5
        while backend.state()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(backend.state()["running"])
        backend.chat_json("m1", task="t", system="s", user={}, schema={})
        backend.release()
        self.assertFalse(backend.state()["running"])
        backend.warm("m1")
        backend.close()
        self.assertEqual([process.returncode for process in self.spawned], [0, 0, 0])

    # A model can take minutes to load; nothing else may wait behind it (start_timeout keeps a regression from hanging).
    def loading(self) -> threading.Event:
        loaded = threading.Event()

        def reply(method, path, body):
            if path == "/health":
                return (200, {"status": "ok"}) if loaded.is_set() else (503, {"error": {"message": "Loading model"}})
            return 200, OK_REPLY

        self.reply = reply
        return loaded

    def ask(self, backend: BuiltinBackend, results: list) -> threading.Thread:
        def run():
            try:
                results.append(backend.chat_json("m1", task="t", system="s", user={}, schema={}))
            except Exception as exc:  # noqa: BLE001 - the test looks at it
                results.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def wait_until(self, condition, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise AssertionError("timed out")
            time.sleep(0.02)

    def test_state_release_and_other_requests_do_not_wait_behind_a_loading_model(self):
        loaded = self.loading()
        backend = self.backend(start_timeout=5)
        results: list = []
        first = self.ask(backend, results)
        self.wait_until(lambda: backend.state()["starting"])
        started = time.monotonic()
        state = backend.state()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual((state["running"], state["starting"], state["model"]), (False, True, "m1"))
        started = time.monotonic()
        backend.release()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertIsNone(self.spawned[0].returncode, "a request waits for this server, so release keeps it")
        second = self.ask(backend, results)
        time.sleep(0.3)
        self.assertEqual((len(self.spawned), results), (1, []), "the second request waits for the same server")
        loaded.set()
        first.join(5)
        second.join(5)
        self.assertEqual(results, [{"ok": True}, {"ok": True}])
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual((backend.state()["running"], backend.state()["starting"]), (True, False))

    def test_closing_stops_a_loading_server_at_once_and_nothing_starts_after(self):
        self.loading()
        backend = self.backend(start_timeout=5)
        results: list = []
        waiting = self.ask(backend, results)
        self.wait_until(lambda: backend.state()["starting"])
        started = time.monotonic()
        backend.close()
        self.assertLess(time.monotonic() - started, 2.0)
        waiting.join(5)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], LlmUnavailable)
        self.assertEqual(self.spawned[0].returncode, 0, "the loading server was stopped")
        with self.assertRaises(LlmUnavailable):
            backend.chat_json("m1", task="t", system="s", user={}, schema={})
        self.assertEqual(len(self.spawned), 1, "nothing starts after close")

    def test_a_server_that_stops_says_why_and_missing_files_stop_it_starting(self):
        self.fail = b"llama_model_load: error loading model: failed to load model\n"
        backend = self.backend()
        with self.assertRaises(LlmUnavailable) as caught:
            backend.chat_json("m1", task="t", system="s", user={}, schema={})
        self.assertIn("failed to load model", str(caught.exception))
        backend.warm("m1")
        self.exe.unlink()
        self.fail = None
        with self.assertRaises(LlmUnavailable) as caught:
            backend.chat_json("m1", task="t", system="s", user={}, schema={})
        self.assertIn("実行ファイル", str(caught.exception))

    def test_the_device_is_the_pc_gpu_else_a_discrete_one_else_the_cpu(self):
        devices = parse_devices(DEVICES)
        self.assertEqual([(item["id"], item["name"]) for item in devices],
                         [("Vulkan0", "NVIDIA GeForce RTX 5090"), ("Vulkan1", "AMD Radeon(TM) Graphics")])
        self.assertEqual(devices[0]["memory"], 32187 * 1024 * 1024)
        self.assertEqual(pick_device(devices, "NVIDIA GeForce RTX 5090"), "Vulkan0")
        self.assertEqual(pick_device(list(reversed(devices)), None), "Vulkan0", "an integrated GPU is not preferred")
        self.assertEqual(pick_device(devices[1:], None), "Vulkan1", "an integrated GPU still beats nothing")
        self.assertEqual(pick_device([], "NVIDIA GeForce RTX 5090"), "none")


class FoundryTests(unittest.TestCase):
    def foundry(self, server: FakeServer | None, output: str | None = None, which="foundry"):
        self.calls: list[list[str]] = []

        def run(args, **_kwargs):
            self.calls.append(list(args))
            text = output if output is not None else (
                f"\U0001F7E2 Model management service is running on {server.url}/openai/status\n"
                "Valid EPs: CPUExecutionProvider, WebGpuExecutionProvider, CUDAExecutionProvider\n")
            return mock.Mock(stdout=text.encode("utf-8"), stderr=b"", returncode=0)

        return FoundryLocal(["qwen3-8b", "mistral-nemo-12b-instruct"], which=lambda _name: which, run=run)

    def test_variants_belong_to_allowlisted_aliases_only(self):
        aliases = ["qwen3-8b", "mistral-nemo-12b-instruct", "qwen2.5-7b"]
        self.assertEqual(variant_alias("qwen3-8b-cuda-gpu:2", aliases), "qwen3-8b")
        self.assertEqual(variant_alias("qwen3-8b-generic-cpu", aliases), "qwen3-8b")
        self.assertEqual(variant_alias("mistral-nemo-12b-instruct-generic-gpu:1", aliases), "mistral-nemo-12b-instruct")
        self.assertEqual(variant_alias("qwen2.5-7b-instruct-cuda-gpu:4", aliases), "qwen2.5-7b")
        self.assertIsNone(variant_alias("qwen3-14b-cuda-gpu:2", aliases))
        self.assertIsNone(variant_alias("qwen3-8b-evil", aliases))

    def test_the_service_is_found_through_the_cli_and_only_on_this_pc(self):
        self.assertEqual(self.foundry(None, which=None).service()["state"], "absent")
        self.assertEqual(self.calls, [], "no CLI call when Foundry Local is not installed")
        stopped = self.foundry(None, "\U0001F534 Model management service is not running!\n")
        self.assertEqual(stopped.service()["state"], "stopped")
        remote = self.foundry(None, "Model management service is running on http://10.0.0.5:5272/openai/status\n")
        self.assertEqual(remote.service()["state"], "stopped")
        server = FakeServer(lambda method, path, body: (200, []))
        self.addCleanup(server.close)
        found = self.foundry(server)
        service = found.service()
        self.assertEqual((service["state"], service["url"]), ("running", server.url))
        self.assertIn("CUDAExecutionProvider", service["eps"])
        found.service()
        self.assertEqual(len(self.calls), 1, "the CLI answer is kept for a while")

    def test_models_download_and_chat_with_the_schema_in_the_prompt(self):
        catalog = [
            {"name": "qwen3-8b-cuda-gpu:2", "alias": "qwen3-8b", "uri": "azureml://qwen3-8b-cuda-gpu/2", "providerType": "AzureFoundry",
             "fileSizeMb": 10, "runtime": {"executionProvider": "CUDAExecutionProvider"}},
            {"name": "qwen3-8b-generic-gpu:2", "alias": "qwen3-8b", "uri": "azureml://qwen3-8b-generic-gpu/2", "providerType": "AzureFoundry",
             "fileSizeMb": 10, "runtime": {"executionProvider": "WebGpuExecutionProvider"}},
            {"name": "qwen3-8b-generic-cpu:2", "alias": "qwen3-8b", "uri": "azureml://qwen3-8b-generic-cpu/2", "providerType": "AzureFoundry",
             "fileSizeMb": 10, "runtime": {"executionProvider": "CPUExecutionProvider"}},
        ]
        cached = ["phi-4-mini-generic-cpu:5"]

        def reply(method, path, body):
            if path == "/openai/models":
                return 200, cached
            if path == "/foundry/list":
                return 200, catalog
            if path == "/openai/download":
                cached.append(body["model"]["Name"])
                return 200, b'("model.onnx", 0.5)\n("", 1)\n{"Success": true, "ErrorMessage": null}'
            if path.startswith("/openai/load/"):
                return 200, b""
            return 200, OK_REPLY

        server = FakeServer(reply)
        self.addCleanup(server.close)
        foundry = self.foundry(server)
        self.assertEqual(foundry.models(), [])
        self.assertEqual(foundry.catalog_entry("qwen3-8b", cuda=True, gpu=True)["name"], "qwen3-8b-cuda-gpu:2")
        self.assertEqual(foundry.catalog_entry("qwen3-8b", cuda=False, gpu=True)["name"], "qwen3-8b-generic-gpu:2")
        self.assertEqual(foundry.catalog_entry("qwen3-8b", cuda=False, gpu=False)["name"], "qwen3-8b-generic-cpu:2")
        with self.assertRaises(LlmUnavailable):
            foundry.catalog_entry("phi-4-mini", cuda=False, gpu=False)
        progress = []
        foundry.download(catalog[0], lambda name, fraction: progress.append(fraction), lambda: False)
        self.assertEqual(progress[-1], 1.0)
        self.assertEqual(server.bodies("/openai/download")[0]["model"]["Uri"], "azureml://qwen3-8b-cuda-gpu/2")
        self.assertEqual(foundry.models(), ["qwen3-8b"])
        for _ in range(2):
            self.assertEqual(foundry.chat_json("qwen3-8b", task="t", system="sys", user={}, schema={"type": "object"}), {"ok": True})
        self.assertEqual(len([path for path in server.paths() if path.startswith("/openai/load/qwen3-8b-cuda-gpu%3A2")]), 1)
        sent = server.bodies("/v1/chat/completions")[0]
        self.assertEqual(sent["model"], "qwen3-8b-cuda-gpu:2")
        self.assertNotIn("response_format", sent)
        self.assertIn("JSON schema", sent["messages"][0]["content"])
        with self.assertRaises(LlmUnavailable):
            foundry.chat_json("phi-4-mini", task="t", system="s", user={}, schema={})

    def test_a_failed_download_says_why(self):
        server = FakeServer(lambda method, path, body: (200, b'{"Success": false, "ErrorMessage": "disk full"}'))
        self.addCleanup(server.close)
        with self.assertRaises(LlmUnavailable) as caught:
            self.foundry(server).download({"name": "x", "uri": "u"}, lambda *_: None, lambda: False)
        self.assertIn("disk full", str(caught.exception))


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Files:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.server = FakeServer(lambda method, path, body: (200, self.files[path.rsplit("/", 1)[-1]])
                                 if path.rsplit("/", 1)[-1] in self.files else (404, b""))

    def entry(self, id: str, component: str, name: str, folder: str) -> ModelEntry:
        data = self.files[name]
        return ModelEntry(id=id, component=component, label=id, folder=folder, source="test", page="http://example.invalid", license="MIT",
                          base_url=self.server.url + "/files", files=(RemoteFile(name, len(data), sha256=hashlib.sha256(data).hexdigest()),),
                          summary="", essential=(name,))


def runtime_zip(names: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in names.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


class OllamaStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-ollama-store-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_the_running_ollama_names_its_folder_through_a_model_file(self):
        store = self.root / "AI models" / "ollama"
        replies = {"/api/tags": {"models": [{"name": "other:latest"}, {"name": "qwen3:8b"}]},
                   "/api/show": {"modelfile": f"# Modelfile\nFROM {store / 'blobs' / 'sha256-abc'}\nTEMPLATE x"}}
        server = FakeServer(lambda method, path, body: (200, replies[path]) if path in replies else (404, b""))
        self.addCleanup(server.close)
        client = OllamaClient(server.url)
        self.assertEqual(client.models_dir(), store)
        self.assertEqual(client.models_dir(), store)
        shows = [body for method, path, _headers, body in server.requests if path == "/api/show"]
        self.assertEqual(shows, [{"model": "qwen3:8b"}], "an allowlisted model when there is one; asked once a minute")
        self.assertEqual(store_dir({"OLLAMA_MODELS": str(self.root / "env")}, client.models_dir()), store, "it wins over OLLAMA_MODELS")
        replies["/api/tags"] = {"models": []}
        self.assertIsNone(OllamaClient(server.url).models_dir(), "no model to ask about")
        server.close()
        self.assertIsNone(OllamaClient(server.url).models_dir(), "not running")

    def test_without_a_model_the_server_log_names_the_folder_the_app_set(self):
        store = self.root / "AI models" / "ollama"
        store.mkdir(parents=True)
        log = self.root / "local" / "Ollama" / "server.log"
        log.parent.mkdir(parents=True)
        written = str(store).replace("\\", "\\\\")
        log.write_text('time=2026-11-04T00:09:41.000+09:00 level=INFO source=routes.go:1600 msg="server config" '
                       f'env="map[CUDA_VISIBLE_DEVICES: OLLAMA_HOST:http://127.0.0.1:11434 OLLAMA_MODELS:{written} '
                       'OLLAMA_MULTIUSER_CACHE:false OLLAMA_NOPRUNE:false]"\n', encoding="utf-8")
        env = {"LOCALAPPDATA": str(self.root / "local"), "OLLAMA_MODELS": str(self.root / "env")}
        with mock.patch("yata.ollama.sys.platform", "win32"):
            self.assertEqual(store_dir(env), store)
            store.rmdir()
            self.assertEqual(store_dir(env), self.root / "env", "a folder that is gone: OLLAMA_MODELS")
            self.assertEqual(store_dir({}), Path.home() / ".ollama" / "models", "nothing known: Ollama's default")

    def test_the_setup_page_shows_the_folder_the_running_ollama_names(self):
        store = self.root / "AI" / "ollama"
        ollama = FakeOllama({"qwen3:8b": "sha256:a"})
        ollama.models_dir = lambda: store
        prep = Preparation(self.root / "data", ollama=ollama, environ={"OLLAMA_MODELS": str(self.root / "env")}, probe=lambda: GPU_PC,
                           cuda=lambda: True, engines=lambda env, cuda=None: (LocalAsr(None), None), background=False, foundry=no_foundry())
        self.addCleanup(prep.close)
        self.assertEqual(prep.status()["llm"]["ollama"]["store"], str(store))


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-llm-settings-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.ollama = FakeOllama({"qwen3:8b": "sha256:a"})
        self.spawned: list[FakeLlama] = []
        self.runs: list[list[str]] = []
        self.files = Files({"llama.zip": runtime_zip({"llama-server.exe": b"MZ", "ggml.dll": b"dll"}),
                            "evil.zip": runtime_zip({"../evil.exe": b"MZ"}), "model.gguf": b"GGUF" * 100})
        self.addCleanup(self.files.server.close)
        self.runtime = self.files.entry("llama-cpp", "llm-runtime", "llama.zip", "llama.cpp-test")
        self.model = self.files.entry("builtin-test", "llm", "model.gguf", "Test-GGUF")

    def prep(self, catalog=None, **env) -> Preparation:
        def spawn(args, **kwargs):
            process = FakeLlama(args, **kwargs)
            self.spawned.append(process)
            return process

        def run(args, **_kwargs):
            self.runs.append(list(args))
            return mock.Mock(stdout=DEVICES.encode(), stderr=b"", returncode=0)

        prep = Preparation(self.root / "data", ollama=self.ollama, environ={"OLLAMA_MODELS": str(self.root / "ollama"), **env},
                           probe=lambda: GPU_PC, cuda=lambda: True, engines=lambda env, cuda=None: (LocalAsr(None), None),
                           catalog=catalog if catalog is not None else (self.runtime, self.model), background=False,
                           # The page fetches llama.cpp on Windows only (elsewhere: llama-server on PATH, tests/test_start.py).
                           foundry=no_foundry(), llama_spawn=spawn, run=run, platform="win32")
        self.addCleanup(prep.llm.close)
        return prep

    def saved(self) -> dict:
        return json.loads((self.root / "data" / "setup.json").read_text(encoding="utf-8"))

    def test_a_first_run_has_no_backend_and_an_earlier_setup_keeps_ollama_and_its_tests(self):
        prep = self.prep()
        self.assertEqual(prep.llm_choice(), {"backend": None, "source": None})
        self.assertEqual(prep.brief(True, [], False)["llm"], {"backend": None, "chosen": False})
        self.assertFalse(prep.llm_health()["chosen"])
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "setup.json").write_text(json.dumps({
            "plan": {"summaries": True, "speakers": False},
            "checks": {"llm:qwen3:8b": {"digest": "sha256:a", "at": "2026-11-04T10:00:00Z", "test_ms": 1000}}}), encoding="utf-8")
        earlier = self.prep()
        self.assertEqual(earlier.llm_choice(), {"backend": "ollama", "source": "earlier"})
        llm = earlier.status()["llm"]
        self.assertEqual((llm["backend"], llm["chosen"]), ("ollama", True))
        self.assertEqual({item["name"]: item["state"] for item in llm["models"]}["qwen3:8b"], "ready", "the old test record still counts")
        self.assertEqual(earlier.llm_health()["models"], [{"id": "qwen3:8b", "label": "qwen3:8b"}])
        earlier.start_llm_test("qwen3:8b")
        checks = self.saved()["checks"]
        self.assertIn("llm:ollama:qwen3:8b", checks)
        self.assertNotIn("llm:qwen3:8b", checks, "rewritten under the backend's key")
        self.assertNotIn("llm", self.saved(), "the choice itself is saved only when the user makes one")

    def test_the_choice_is_saved_checked_and_the_environment_wins(self):
        prep = self.prep()
        prep.choose_llm("builtin")
        self.assertEqual(self.prep().llm_choice(), {"backend": "builtin", "source": "settings"})
        for backend, status in (("foundry", 409), ("cloud", 422)):
            with self.assertRaises(SetupError) as caught:
                prep.choose_llm(backend)
            self.assertEqual(caught.exception.status, status)
        offered = [item["id"] for item in prep.status()["llm"]["options"]]
        self.assertEqual(offered, ["builtin", "ollama", "openai"], "Foundry Local is offered only where it is installed")
        fixed = self.prep(YATA_LLM_BACKEND="openai")
        self.assertEqual(fixed.llm_choice(), {"backend": "openai", "source": "env"})
        with self.assertRaises(SetupError):
            fixed.choose_llm("ollama")

    def test_the_openai_compatible_server_stays_on_this_pc_and_its_key_is_never_stored(self):
        server = FakeServer(lambda method, path, body: (200, {"data": [{"id": "local-model"}]} if path == "/v1/models" else OK_REPLY))
        self.addCleanup(server.close)
        prep = self.prep(YATA_OPENAI_API_KEY="sk-secret")
        with self.assertRaises(SetupError) as caught:
            prep.set_openai("https://api.openai.com/v1", "gpt-x")
        self.assertEqual(caught.exception.status, 422)
        prep.set_openai(server.url + "/v1/", "local-model")
        self.assertEqual(self.saved()["llm"]["openai"], {"base_url": server.url + "/v1", "model": "local-model"})
        openai = prep.status()["llm"]["openai"]
        self.assertEqual((openai["state"], openai["server_models"], openai["key"]), ("unverified", ["local-model"], True))
        prep.start_llm_test("local-model", "openai")
        self.assertEqual(prep.status()["llm"]["openai"]["state"], "ready")
        self.assertTrue(prep.llm_allows("openai", "local-model"))
        self.assertFalse(prep.llm_allows("openai", "qwen3:8b"))
        prep.set_openai(server.url + "/v1", "other-model")
        self.assertEqual(prep.status()["llm"]["openai"]["state"], "missing")
        dumped = json.dumps(prep.status(), ensure_ascii=False) + (self.root / "data" / "setup.json").read_text(encoding="utf-8")
        self.assertNotIn("sk-secret", dumped)
        fixed = self.prep(YATA_OPENAI_BASE_URL=server.url + "/v1")
        with self.assertRaises(SetupError):
            fixed.set_openai(server.url + "/v1", "local-model")

    def test_the_builtin_runtime_and_model_are_fetched_unpacked_and_tested(self):
        prep = self.prep()
        builtin = prep.status()["llm"]["builtin"]
        self.assertEqual((builtin["executable"], builtin["runtime"]["state"], builtin["models"][0]["state"]), (False, "missing", "missing"))
        self.assertEqual(self.runs, [], "nothing runs before the user asks")
        prep.start_download("builtin-test")
        self.assertIn("llama.cpp の実行環境を準備すると試せます", prep.jobs["llm"].note)
        prep.start_download("llama-cpp")
        executable = self.root / "data" / "runtime" / "llama.cpp-test" / "bin" / "llama-server.exe"
        self.assertTrue(executable.is_file())
        self.assertEqual(self.runs, [[str(executable), "--list-devices"]])
        builtin = prep.status()["llm"]["builtin"]
        self.assertEqual((builtin["executable"], builtin["runtime"]["state"], builtin["device"]), (True, "ready", "Vulkan0"))
        self.assertEqual(builtin["models"][0]["state"], "unverified")
        prep.start_verify("builtin-test")
        self.assertEqual(prep.jobs["llm"].state, "done", prep.jobs["llm"].error)
        status = prep.status()["llm"]
        self.assertEqual(status["builtin"]["models"][0]["state"], "ready")
        self.assertTrue(next(item for item in status["options"] if item["id"] == "builtin")["usable"])
        args = self.spawned[0].args
        self.assertEqual(args[args.index("--device") + 1], "Vulkan0")
        self.assertEqual(Path(args[args.index("-m") + 1]), prep.model_root()[0] / "Test-GGUF" / "model.gguf")
        self.assertEqual(prep.builtin.models(), ["builtin-test"])

    def test_a_runtime_zip_with_paths_outside_its_folder_is_not_unpacked(self):
        evil = self.files.entry("llama-cpp", "llm-runtime", "evil.zip", "llama.cpp-evil")
        prep = self.prep(catalog=(evil, self.model))
        prep.start_download("llama-cpp")
        self.assertEqual((prep.jobs["llm-runtime"].state, prep.jobs["llm-runtime"].error["kind"]), ("failed", "corrupt"))
        self.assertFalse((self.root / "data" / "runtime" / "evil.exe").exists())
        self.assertEqual(self.runs, [])

    def test_a_llama_server_from_the_environment_is_used_and_not_fetched(self):
        exe = self.root / "shipped" / "llama-server.exe"
        exe.parent.mkdir()
        exe.write_bytes(b"MZ")
        prep = self.prep(YATA_LLAMA_SERVER=str(exe), YATA_LLAMA_DEVICE="none")
        self.assertEqual(prep.llama_server(), exe)
        with self.assertRaises(SetupError):
            prep.start_download("llama-cpp")
        builtin = prep.status()["llm"]["builtin"]
        self.assertEqual((builtin["env"], builtin["runtime"], builtin["device"]), ("YATA_LLAMA_SERVER", None, "none"))


class ScriptedBackend:
    def __init__(self, kind: str, models: list[str]) -> None:
        self.kind = kind
        self._models = models
        self.calls: list[tuple[str, str]] = []
        self.warmed: list[str] = []

    def models(self):
        return list(self._models)

    def allows(self, model):
        return model in self._models

    def warm(self, model):
        self.warmed.append(model)

    def chat_json(self, model, *, task, **_kwargs):
        self.calls.append((model, task))
        return {}


class SessionTests(unittest.TestCase):
    def test_a_session_uses_the_backend_it_started_with_and_only_its_models(self):
        ollama, builtin = ScriptedBackend("ollama", ["qwen3:8b"]), ScriptedBackend("builtin", ["builtin-qwen3-8b"])
        router = LlmRouter({"ollama": ollama, "builtin": builtin})
        manager = SessionManager(TopicPipeline(router), SessionStore(), background=False, llm_allows=router.allows)
        with self.assertRaises(SessionError):
            manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b", llm_backend="builtin")
        session = manager.create(source_url="", source_title="", source_kind="share", selected_model="builtin-qwen3-8b", llm_backend="builtin")
        self.assertEqual(session.public_dict()["llm_backend"], "builtin")
        manager.pipeline.warm(session)
        manager.set_status(session, "capturing_audio")
        manager.ingest(session, [{"text": "新製品を発表します。", "capture_start_ms": 0, "capture_end_ms": 3000}], trusted_source_type="asr")
        manager.stop(session)
        self.assertEqual(builtin.warmed, ["builtin-qwen3-8b"])
        self.assertTrue(builtin.calls)
        self.assertEqual(ollama.calls, [])
        self.assertEqual({model for model, _task in builtin.calls}, {"builtin-qwen3-8b"})

    def test_sessions_saved_before_the_choice_used_ollama(self):
        manager = SessionManager(TopicPipeline(ScriptedBackend("ollama", ["qwen3:8b"])), SessionStore(), background=False)
        session = manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b")
        record = to_record(session)
        self.assertEqual(record["session"]["llm_backend"], "ollama")
        del record["session"]["llm_backend"]
        self.assertEqual(from_record(record).llm_backend, "ollama")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-llm-api-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.server = FakeServer(lambda method, path, body: (200, {"data": [{"id": "local-model"}]} if path == "/v1/models" else OK_REPLY))
        self.addCleanup(self.server.close)
        self.ollama = FakeOllama({"qwen3:8b": "sha256:a"})
        prep = Preparation(self.root / "data", ollama=self.ollama, environ={"OLLAMA_MODELS": str(self.root)}, probe=lambda: GPU_PC,
                           cuda=lambda: False, engines=lambda env, cuda=None: (LocalAsr(None), None), background=False, foundry=no_foundry())
        manager = SessionManager(TopicPipeline(prep.llm), SessionStore(), background=False, archive=SessionArchive(self.root / "sessions"),
                                 llm_allows=prep.llm_allows)
        self.addCleanup(manager.close)
        self.app = create_app(access_token="t", ollama=self.ollama, asr=LocalAsr(None), manager=manager, features=Features(), preparation=prep)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8765")
        self.client.get("/")

    def test_the_backend_is_chosen_and_checked_on_the_server_and_new_sessions_use_it(self):
        client = self.client
        health = client.get("/api/v1/health").json()
        self.assertEqual((health["llm"]["chosen"], health["setup"]["llm"]["chosen"]), (False, False))
        self.assertEqual(client.put("/api/v1/setup/llm", json={"backend": "cloud"}).status_code, 422)
        self.assertEqual(client.post("/api/v1/setup/llm/foundry/start").status_code, 409)
        chosen = client.put("/api/v1/setup/llm", json={"backend": "openai"})
        self.assertEqual(chosen.json()["setup"]["llm"]["backend"], "openai")
        self.assertFalse(client.get("/api/v1/health").json()["llm"]["connected"])
        self.assertEqual(client.put("/api/v1/setup/llm/openai", json={"base_url": "https://api.openai.com/v1", "model": "gpt"}).status_code, 422)
        saved = client.put("/api/v1/setup/llm/openai", json={"base_url": self.server.url + "/v1", "model": "local-model"})
        self.assertEqual(saved.json()["setup"]["llm"]["openai"]["state"], "unverified")
        health = client.get("/api/v1/health").json()["llm"]
        self.assertEqual((health["label"], health["connected"], health["models"]), ("OpenAI 互換 API", True, [{"id": "local-model", "label": "local-model"}]))
        self.assertEqual(client.get("/api/v1/models").json()["models"], health["models"])
        tested = client.post("/api/v1/setup/llm/test", json={"model": "local-model", "backend": "openai"})
        self.assertEqual(tested.json()["setup"]["llm"]["openai"]["state"], "ready")
        self.assertEqual(client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "qwen3:8b"}).status_code, 422)
        created = client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "local-model"})
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["session"]["llm_backend"], "openai")
        client.put("/api/v1/setup/llm", json={"backend": "ollama"})
        again = client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "qwen3:8b"}).json()["session"]
        self.assertEqual(again["llm_backend"], "ollama")
        self.assertEqual(self.app.state.manager.get(created.json()["session"]["id"]).llm_backend, "openai")


if __name__ == "__main__":
    unittest.main()
