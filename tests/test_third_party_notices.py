from __future__ import annotations

import importlib.util
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("third_party_notices", ROOT / "tools" / "third_party_notices.py")
notices = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(notices)


class NoticesTests(unittest.TestCase):
    def test_the_file_lists_exactly_what_uv_lock_bundles(self):
        listed = notices.listed_packages(notices.NOTICES.read_text(encoding="utf-8"))
        self.assertEqual(listed, notices.bundled_packages(),
                         "THIRD_PARTY_NOTICES.md is out of date: run `uv run python tools/third_party_notices.py`")
        self.assertIn("av", listed)
        self.assertIn("sherpa-onnx", listed, "the build bundles the speakers extra")
        self.assertNotIn("pytest", listed, "dev tools are not bundled")
        self.assertNotIn("httpx2-jsfetch", listed, "markers are evaluated for Windows")

    def test_it_follows_the_build_settings(self):
        build = (ROOT / "tools" / "build_portable.py").read_text(encoding="utf-8")
        self.assertEqual(re.search(r'^PYTHON_VERSION = "([^"]+)"', build, re.MULTILINE).group(1), notices.PYTHON_VERSION)
        self.assertEqual(re.search(r'^CUBLAS_VERSION = "([^"]+)"', build, re.MULTILINE).group(1), notices.CUBLAS_VERSION)
        self.assertEqual(re.search(r"^EXTRAS = \(([^)]*)\)", build, re.MULTILINE).group(1).strip().strip(","), '"speakers"')
        self.assertIn("notices.check_bundle(", build, "the build refuses what the file does not describe")
        self.assertIn("notices.check_licence_texts(", build, "the build refuses licence texts collected for another version")
        self.assertIn("notices.REMOVED", build, "the build deletes the files the notices say it removes")
        self.assertIn('stage / "THIRD_PARTY_NOTICES.md"', build)
        self.assertIn('stage / "licenses"', build)

    def test_the_points_to_check_name_the_licences_that_matter(self):
        text = notices.NOTICES.read_text(encoding="utf-8")
        for needle in ("libx264", "GPL-2.0-or-later", "cudnn64_9.dll", "libiomp5md.dll", "cuBLAS", "Visual C++",
                       "Intel End User License Agreement for Developer Tools", "licenses\\ctranslate2\\CTranslate2-LICENSE.txt"):
            self.assertIn(needle, text)

    def test_the_licence_texts_are_there_for_the_locked_versions(self):
        self.assertEqual(notices.check_licence_texts(notices.bundled_packages()), [])
        eula = (notices.LICENSES / "ctranslate2" / "Intel-OpenMP-EULA.txt").read_text(encoding="utf-8")
        self.assertTrue(eula.startswith("Intel End User License Agreement for Developer Tools"))
        self.assertIn("SYSTRAN", (notices.LICENSES / "ctranslate2" / "CTranslate2-LICENSE.txt").read_text(encoding="utf-8"))

    def test_the_licence_check_catches_another_version_and_missing_or_extra_files(self):
        licenses = Path(tempfile.mkdtemp(prefix="yata-licenses-"))
        self.addCleanup(shutil.rmtree, licenses, True)
        shutil.copytree(notices.LICENSES / "ctranslate2", licenses / "ctranslate2")
        version = notices.LICENSE_TEXTS["ctranslate2"][0]
        self.assertEqual(notices.check_licence_texts({"ctranslate2": version}, licenses), [])
        (licenses / "ctranslate2" / "spdlog-LICENSE.txt").unlink()
        (licenses / "ctranslate2" / "stray.txt").write_text("", encoding="utf-8")
        (licenses / "other").mkdir()
        problems = notices.check_licence_texts({"ctranslate2": "0.0.1"}, licenses)
        self.assertEqual(len(problems), 4, problems)
        self.assertTrue(any("collect its licence texts again" in item for item in problems))
        self.assertTrue(any("spdlog-LICENSE.txt is missing" in item for item in problems))
        self.assertTrue(any("stray.txt" in item for item in problems))
        self.assertTrue(any("licenses/other" in item for item in problems))

    def test_the_build_check_catches_unlisted_packages_versions_and_dlls(self):
        site = Path(tempfile.mkdtemp(prefix="yata-notices-"))
        self.addCleanup(shutil.rmtree, site, True)
        for relative in ("av.libs/libx264-165-0f71.dll", "ctranslate2/libiomp5md.dll", "numpy.libs/msvcp140-a4c2.dll"):
            (site / relative).parent.mkdir(parents=True, exist_ok=True)
            (site / relative).write_bytes(b"")
        text = notices.NOTICES.read_text(encoding="utf-8")
        listed = notices.listed_packages(text)
        self.assertEqual(notices.check_bundle(site, {"av": listed["av"], "numpy": listed["numpy"]}, text), [])
        (site / "ctranslate2" / "mystery.dll").write_bytes(b"")
        (site / "ctranslate2" / "cudnn64_9.dll").write_bytes(b"")  # a removed file left behind
        (site / "unknownpkg").mkdir()
        (site / "unknownpkg" / "thing.dll").write_bytes(b"")
        problems = notices.check_bundle(site, {"av": "0.0.1", "leftpad": "1.0"}, text)
        self.assertEqual(len(problems), 5, problems)
        self.assertTrue(any("av 0.0.1" in item for item in problems))
        self.assertTrue(any("leftpad" in item for item in problems))
        self.assertTrue(any("mystery.dll" in item for item in problems))
        self.assertTrue(any("cudnn64_9.dll" in item for item in problems))
        self.assertTrue(any("thing.dll" in item for item in problems))


if __name__ == "__main__":
    unittest.main()
