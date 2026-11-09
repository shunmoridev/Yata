from __future__ import annotations

import contextlib
import http.server
import io
import os
import shutil
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import yata.__main__ as launcher
from yata.archive import default_data_dir
from yata.inputs import asr
from yata.ollama import OllamaClient
from yata.preparation import Preparation, SetupError
from yata.llm.foundry import FoundryLocal

ROOT = Path(__file__).resolve().parent.parent
CPU_PC = {"cpu": {"name": "Test CPU", "logical": 8}, "memory": {"total": 16 * 1024 ** 3, "available": 8 * 1024 ** 3}, "gpu": None, "os": "Linux"}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def page_on(port: int, body: bytes):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()


class LauncherTests(unittest.TestCase):
    def on_port(self, port: int):
        return mock.patch.multiple(launcher, PORT=port, URL=f"http://127.0.0.1:{port}/")

    def test_the_port_is_free_holds_yata_or_holds_something_else(self):
        port = free_port()
        with self.on_port(port):
            self.assertEqual(launcher.port_state(), "free")
            with page_on(port, b'<title>Yata</title><meta name="application-name" content="Yata">'):
                self.assertEqual(launcher.port_state(), "yata")
        port = free_port()
        with self.on_port(port), page_on(port, b"<title>YATA // LIVE OPS</title>"):
            self.assertEqual(launcher.port_state(), "yata", "a Yata from before the page named itself")
        port = free_port()
        with self.on_port(port), page_on(port, b"<title>another app</title>"):
            self.assertEqual(launcher.port_state(), "other")

    def test_a_running_yata_is_opened_instead_of_started_twice(self):
        out = io.StringIO()
        with mock.patch.object(launcher, "port_state", return_value="yata"), \
                mock.patch.object(launcher.webbrowser, "open") as opened, contextlib.redirect_stdout(out):
            self.assertEqual(launcher.main(["--open"]), 0)
        opened.assert_called_once_with(launcher.URL)
        self.assertIn("もう起動しています", out.getvalue())

    def test_another_program_on_the_port_stops_the_start_with_a_reason(self):
        err = io.StringIO()
        with mock.patch.object(launcher, "port_state", return_value="other"), \
                mock.patch.object(launcher.webbrowser, "open") as opened, contextlib.redirect_stderr(err):
            self.assertEqual(launcher.main(["--open"]), 1)
        opened.assert_not_called()
        self.assertIn("別のアプリ", err.getvalue())


class DataDirTests(unittest.TestCase):
    def test_each_os_keeps_app_data_in_its_own_place(self):
        home = Path.home()
        self.assertEqual(default_data_dir({"LOCALAPPDATA": r"C:\Users\u\AppData\Local"}, "win32"), Path(r"C:\Users\u\AppData\Local") / "Yata")
        self.assertEqual(default_data_dir({}, "darwin"), home / "Library" / "Application Support" / "Yata")
        self.assertEqual(default_data_dir({}, "linux"), home / ".local" / "share" / "Yata")
        self.assertEqual(default_data_dir({"XDG_DATA_HOME": "/data"}, "linux"), Path("/data") / "Yata")
        self.assertEqual(default_data_dir({"YATA_DATA_DIR": "/elsewhere", "LOCALAPPDATA": "C:/x"}, "win32"), Path("/elsewhere"))


class CudaTests(unittest.TestCase):
    def test_a_gpu_counts_only_when_the_cuda_libraries_load(self):
        asked: list[str] = []
        self.assertTrue(asr.cuda_libraries_load("win32", asked.append))
        self.assertEqual(asked, ["cublas64_12.dll", "cublasLt64_12.dll"])
        asked.clear()
        self.assertTrue(asr.cuda_libraries_load("linux", asked.append))
        self.assertIn("libcudnn.so.9", asked)

        def missing(name):
            raise OSError(f"{name} not found")

        self.assertFalse(asr.cuda_libraries_load("win32", missing))
        self.assertFalse(asr.cuda_libraries_load("darwin", asked.append), "macOS has no CUDA")

    def test_the_cuda_extra_goes_first_on_path_on_windows_once(self):
        folder = Path(tempfile.mkdtemp(prefix="yata-cublas-"))
        self.addCleanup(shutil.rmtree, folder, True)
        other = str(Path(tempfile.gettempdir()) / "other")
        with mock.patch.object(asr, "bundled_cuda_folders", return_value=[folder]), mock.patch.dict(os.environ, {"PATH": other}):
            asr.use_bundled_cuda("win32")
            asr.use_bundled_cuda("win32")
            self.assertEqual(os.environ["PATH"].split(os.pathsep), [str(folder), other])
            asr.use_bundled_cuda("linux")  # Linux: LD_LIBRARY_PATH, set by the start script
            self.assertEqual(os.environ["PATH"].split(os.pathsep), [str(folder), other])


class OtherSystemsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-os-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.bin = self.root / "bin"
        self.bin.mkdir()

    def prep(self, platform: str) -> Preparation:
        return Preparation(self.root / "data", ollama=OllamaClient("http://127.0.0.1:9"),
                           environ={"PATH": str(self.bin), "OLLAMA_MODELS": str(self.root / "ollama")}, probe=lambda: CPU_PC,
                           cuda=lambda: False, background=False, foundry=FoundryLocal(["qwen3-8b"], which=lambda _name: None),
                           platform=platform)

    def test_without_llama_server_the_page_says_how_to_install_it_and_suggests_ollama(self):
        for platform, hint in (("darwin", "brew install llama.cpp"), ("linux", "llama-server")):
            prep = self.prep(platform)
            status = prep.status()
            builtin = status["llm"]["builtin"]
            self.assertIsNone(builtin["runtime"], "no Windows build to fetch")
            self.assertFalse(builtin["executable"])
            self.assertIn(hint, builtin["install"])
            self.assertEqual(status["llm"]["recommended"], "ollama")
            self.assertEqual(status["translator"]["install"], builtin["install"])
            with self.assertRaises(SetupError):
                prep.start_download("llama-cpp")

    def test_a_llama_server_on_path_is_used(self):
        for name in ("llama-server", "llama-server.exe"):
            (self.bin / name).write_bytes(b"")
            (self.bin / name).chmod(0o755)
        prep = self.prep("linux")
        self.assertIsNotNone(prep.llama_server())
        self.assertIsNone(prep.status()["llm"]["builtin"]["install"])

    def test_windows_still_fetches_its_own_llama_cpp(self):
        status = self.prep("win32").status()
        self.assertIsNotNone(status["llm"]["builtin"]["runtime"])
        self.assertIsNone(status["llm"]["builtin"]["install"])


class ScriptTests(unittest.TestCase):
    def test_line_endings_and_encodings(self):
        ps1 = (ROOT / "scripts" / "start.ps1").read_bytes()
        self.assertTrue(ps1.startswith(b"\xef\xbb\xbf"), "Windows PowerShell 5.1 reads a .ps1 without a BOM as ANSI")
        self.assertNotIn(b"\n", ps1.replace(b"\r\n", b""))
        cmd = (ROOT / "start-windows.cmd").read_bytes()
        self.assertTrue(cmd.isascii(), "cmd reads a batch file in the console code page")
        for name in ("start-mac-linux.sh", "start-mac.command"):
            script = (ROOT / name).read_bytes()
            self.assertNotIn(b"\r", script, name)
            self.assertTrue(script.startswith(b"#!/bin/sh\n"), name)

    def test_every_script_starts_yata_with_the_same_packages(self):
        ps1 = (ROOT / "scripts" / "start.ps1").read_text(encoding="utf-8-sig")
        sh = (ROOT / "start-mac-linux.sh").read_text(encoding="utf-8")
        for script in (ps1, sh):
            self.assertIn("sync --frozen --inexact --no-dev", script.replace('@("sync", "--frozen", "--inexact", "--no-dev")', "sync --frozen --inexact --no-dev"))
            self.assertIn("--extra speakers", script.replace('@("--extra", "speakers")', "--extra speakers"))
            self.assertIn("-B -m yata --open", script)

    @unittest.skipUnless(shutil.which("sh"), "no POSIX shell here")
    def test_the_shell_script_parses(self):
        import subprocess
        for name in ("start-mac-linux.sh", "start-mac.command"):
            done = subprocess.run([shutil.which("sh"), "-n", str(ROOT / name)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
