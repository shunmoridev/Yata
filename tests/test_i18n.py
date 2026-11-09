from __future__ import annotations

import ast
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from test_sessions import NamingLLM, say
from test_setup import CPU_PC, FakeEngine, FakeOllama, no_foundry
from yata import __main__ as launcher
from yata import i18n
from yata.api import create_app
from yata.archive import SessionArchive
from yata.export import session_markdown
from yata.features import Features
from yata.inputs.asr import LocalAsr
from yata.models import SessionStore
from yata.preparation import Preparation, SetupError
from yata.state import SessionError, SessionManager
from yata.topics import TopicPipeline

ROOT = Path(__file__).resolve().parent.parent
ENGLISH = {"Accept-Language": "en-US,en;q=0.9,ja;q=0.8"}
JAPANESE = {"Accept-Language": "ja,en-US;q=0.9"}
KANA_OR_KANJI = re.compile(r"[぀-ヿ一-鿿]")


class NegotiationTests(unittest.TestCase):
    def test_the_first_supported_language_in_the_browser_list_wins_and_japanese_is_the_fallback(self):
        self.assertEqual(i18n.negotiate("fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"), "en")
        self.assertEqual(i18n.negotiate("en;q=0.5, ja;q=0.8"), "ja", "by preference, not by order")
        self.assertEqual(i18n.negotiate("de-DE,fr"), "ja", "a list without a supported language gets Japanese")
        self.assertIsNone(i18n.negotiate(""), "no list: the caller decides")
        self.assertIsNone(i18n.negotiate(None))
        self.assertEqual(i18n.negotiate("en;q=0, ja"), "ja", "q=0 means not acceptable")

    def test_a_request_takes_the_saved_choice_then_the_browser_then_the_background_language(self):
        self.assertEqual(i18n.for_request("ja", ENGLISH["Accept-Language"]), "ja")
        self.assertEqual(i18n.for_request(None, ENGLISH["Accept-Language"]), "en")
        self.assertEqual(i18n.for_request(None, None), "ja", "the tests pin the OS language to none")
        i18n.remember_page("en")
        self.assertEqual(i18n.for_request(None, None), "en", "work outside a page request follows the last page")
        i18n.set_choice("ja")
        self.assertEqual(i18n.background(), "ja", "a saved choice wins over the last page")

    def test_text_picks_its_language_when_it_is_shown(self):
        label = i18n.Text("話題", "Topic")
        self.assertEqual(str(label), "話題")
        token = i18n.use("en")
        try:
            self.assertEqual(str(label), "Topic")
            self.assertEqual(f"{label} 1", "Topic 1")
            self.assertEqual(i18n.t("はい", "yes"), "yes")
        finally:
            i18n.reset(token)
        self.assertEqual(i18n.t("はい", "yes"), "はい")

    def test_the_os_language_is_read_from_the_locale_variables_off_windows(self):
        self.assertEqual(i18n.normalize("en_US.UTF-8"), "en")
        self.assertEqual(i18n.normalize("ja-JP"), "ja")
        self.assertIsNone(i18n.normalize("C"))


class MessageCoverageTests(unittest.TestCase):
    def test_every_message_has_japanese_and_english(self):
        problems = []
        for path in sorted((ROOT / "yata").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name not in {"t", "Text"} or path.name == "i18n.py":
                    continue
                where = f"{path.relative_to(ROOT)}:{node.lineno}"
                if len(node.args) != 2:
                    problems.append(f"{where}: needs two texts")
                    continue
                english = ast.get_source_segment(path.read_text(encoding="utf-8"), node.args[1]) or ""
                if KANA_OR_KANJI.search(english):
                    problems.append(f"{where}: the English text has Japanese in it")
        self.assertEqual(problems, [])

    def test_no_interface_text_is_left_only_in_japanese(self):
        not_interface = {
            "_HALLUCINATIONS",  # inputs/asr.py: stock phrases the recognizer drops
            "TRANSITION",  # topics.py: topic-boundary cues in transcripts
            "_SCALE_WORDS", "_KANJI_DIGITS", "_KANJI_SMALL", "_KANJI_LARGE", "_CJK_UNITS", "_ARABIC", "_KANJI",  # numbers.py
            "AUTONYMS",  # export.py: each language under its own name
            "_TEST_LINE", "chat_json",  # preparation.py: the translator's and the LLM's test inputs
            "Term",  # catalog.py: PLaMo license terms (deferred)
        }
        found = []
        for path in sorted((ROOT / "yata").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            inside = set()
            for node in ast.walk(tree):
                name = ""
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                if name in {"t", "Text"} | not_interface:
                    inside.update(id(child) for child in ast.walk(node))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in inside:
                    if KANA_OR_KANJI.search(node.value) and not _docstring(tree, node):
                        found.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}: {node.value[:40]}")
        self.assertEqual(found, [], "Japanese text outside t()/Text()")


def _docstring(tree: ast.AST, constant: ast.Constant) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and first.value is constant:
                return True
    return False


class LocaleApiTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-i18n-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.ollama = FakeOllama()
        self.prep = self.preparation()
        self.manager = SessionManager(TopicPipeline(NamingLLM()), SessionStore(), background=False,
                                      archive=SessionArchive(self.root / "sessions"))
        self.addCleanup(self.manager.close)
        self.app = create_app(access_token="t", ollama=self.ollama, asr=LocalAsr(None), manager=self.manager,
                              features=Features(), preparation=self.prep)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8765")
        self.client.get("/")

    def preparation(self) -> Preparation:
        return Preparation(self.root / "data", ollama=self.ollama, environ={"OLLAMA_MODELS": str(self.root)}, probe=lambda: CPU_PC,
                           cuda=lambda: False, engines=lambda environment, cuda=None: (FakeEngine(environment, None), None),
                           background=False, foundry=no_foundry())

    def test_without_a_choice_the_server_answers_in_the_browser_language(self):
        english = self.client.get("/api/v1/sessions/nope", headers=ENGLISH)
        self.assertEqual(english.status_code, 404)
        self.assertEqual(english.json()["detail"], "Session not found.")
        self.assertEqual(self.client.get("/api/v1/sessions/nope", headers=JAPANESE).json()["detail"], "セッションが見つかりません。")
        health = self.client.get("/api/v1/health", headers=ENGLISH).json()
        self.assertEqual(health["locale"], {"choice": None, "current": "en", "supported": ["ja", "en"]})
        self.assertEqual(health["llm"]["reason"], 'No summary LLM is chosen yet. Choose one under "Set up this PC".')
        self.assertIn("No folder is set for the ASR model.", health["asr"]["reason"])

    def test_the_choice_in_the_settings_is_saved_and_wins_over_the_browser(self):
        saved = self.client.put("/api/v1/setup/locale", json={"locale": "en"}, headers=JAPANESE)
        self.assertEqual(saved.json()["locale"], {"choice": "en", "supported": ["ja", "en"]})
        self.assertEqual(json.loads((self.root / "data" / "setup.json").read_text(encoding="utf-8"))["locale"], "en")
        self.assertEqual(self.client.get("/api/v1/sessions/nope", headers=JAPANESE).json()["detail"], "Session not found.")
        self.assertEqual(self.preparation().locale_choice(), "en")
        page = self.client.get("/", headers=JAPANESE).text
        self.assertIn('<html lang="en"', page)
        self.assertIn('data-locale="en"', page)
        self.client.put("/api/v1/setup/locale", json={"locale": "auto"})
        self.assertNotIn("locale", json.loads((self.root / "data" / "setup.json").read_text(encoding="utf-8")))
        page = self.client.get("/", headers=ENGLISH).text
        self.assertIn('<html lang="en"', page)
        self.assertIn('data-locale=""', page)
        self.assertEqual(self.client.put("/api/v1/setup/locale", json={"locale": "fr"}, headers=ENGLISH).json()["detail"],
                         "Unknown display language.")
        with self.assertRaises(SetupError):
            self.prep.set_locale("klingon")

    def test_setup_texts_and_job_messages_follow_the_page(self):
        setup = self.client.get("/api/v1/setup", headers=ENGLISH).json()["setup"]
        self.assertEqual(setup["recommended"]["asr_reason"],
                         "There is no GPU that can use CUDA, so a light model that runs on the CPU is recommended.")
        small = next(model for model in setup["asr"]["models"] if model["id"] == "whisper-small")
        self.assertTrue(small["summary"].startswith("Light. Runs on the CPU"))
        self.assertEqual(setup["llm"]["options"][0]["label"], "Built-in (llama.cpp)")
        self.assertEqual(self.client.get("/api/v1/setup", headers=JAPANESE).json()["setup"]["llm"]["options"][0]["label"], "内蔵（llama.cpp）")
        refused = self.client.post("/api/v1/setup/jobs/asr/cancel", headers=ENGLISH)
        self.assertEqual(refused.json()["detail"], "Nothing is running that can be cancelled.")

    def test_session_errors_and_the_export_follow_the_page(self):
        self.client.put("/api/v1/setup/locale", json={"locale": "en"})
        session = self.manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b")
        self.manager.set_status(session, "capturing_audio")
        say(self.manager, session, "新しいカメラを紹介します。", 0)
        self.manager.record_gap(session, "gap", 5000, 8000)
        self.manager.stop(session)
        exported = self.client.get(f"/api/v1/sessions/{session.id}/export.md").text
        for heading in ("> Captions and summaries were made", "## Topics", "*(The summary failed.)*", "## Transcript", "- Languages: stream detected automatically, summaries 日本語",
                        "- Summary model: qwen3:8b (Ollama)", "were made automatically and may contain errors.", ": no audio arrived*"):
            self.assertIn(heading, exported)
        self.assertIn("新しいカメラを紹介します。", exported, "the transcript stays in its own language")
        self.assertNotIn("## 文字起こし", exported)
        with self.assertRaises(SessionError) as raised:
            self.manager.delete(self.manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b"))
        self.assertEqual(str(raised.exception), "A session that is still capturing cannot be deleted. End it first.")
        token = i18n.use("ja")
        try:
            self.assertIn("## 文字起こし", session_markdown(session))
        finally:
            i18n.reset(token)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="yata-i18n-launcher-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_the_launcher_reads_the_saved_language(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"YATA_DATA_DIR": str(self.root)}):
            self.assertIsNone(launcher._saved_locale())
            (self.root / "setup.json").write_text(json.dumps({"locale": "en", "checks": {}}), encoding="utf-8")
            self.assertEqual(launcher._saved_locale(), "en")
            (self.root / "setup.json").write_text("{broken", encoding="utf-8")
            self.assertIsNone(launcher._saved_locale())


if __name__ == "__main__":
    unittest.main()
