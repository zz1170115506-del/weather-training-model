"""A clean release must run without private paths, data, Git or source checkout."""
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_task1_release as release


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_area = Path(__file__).resolve().parents[3] / "tmp"
        cls.test_area.mkdir(exist_ok=True)
        cls.directory = cls.test_area / ("release_tests_" + uuid.uuid4().hex)
        cls.directory.mkdir()
        cls.addClassCleanup(cls.cleanup)
        cls.config = cls.directory / "release_fixture.json"
        cls.raw = {"schema_version": 1,
                   "description": "Synthetic release fixture, never course data.",
                   "models": [{"name": "fixture_cart", "model_id": "numpy_cart",
                               "model_params": {"max_depth": 2}}]}
        cls.config.write_text(json.dumps(cls.raw), encoding="utf-8")
        cls.package = cls.directory / "clean_package"
        # The one live verification uses generated CSVs and `check`, never fit.
        cls.result = release.export_release(cls.package, cls.config)

    @classmethod
    def cleanup(cls):
        target = cls.directory.resolve()
        if target.parent != cls.test_area.resolve() or not target.name.startswith("release_tests_"):
            raise RuntimeError("Unsafe release test cleanup target.")
        shutil.rmtree(target)

    def test_clean_export_runs_help_import_and_128_generated_record_check_without_training(self):
        self.assertEqual(self.result["synthetic_training_rows"], 128)
        self.assertEqual(self.result["synthetic_check_status"], "ready")
        self.assertEqual(self.result["production_model_factories"], 1)
        self.assertTrue(self.result["cli_help_verified"])
        self.assertFalse(self.result["training_started"])
        self.assertFalse(self.result["course_data_read"])

    def test_only_allowlisted_code_and_one_config_are_published_with_exact_hashes(self):
        manifest = json.loads((self.package / release.MANIFEST).read_text(encoding="utf-8"))
        actual = {p.relative_to(self.package).as_posix() for p in self.package.rglob("*") if p.is_file()}
        self.assertEqual(actual, {*release.RUNTIME_FILES, "README.md", release.MANIFEST,
                                  "configs/task1/release_fixture.json"})
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(release._digest(self.package / name), digest)
        self.assertNotRegex(json.dumps(manifest), r"(?<![A-Za-z])[A-Za-z]:[\\/]+")
        self.assertNotIn("dataset1_baseline.py", actual)
        self.assertNotIn("dataset1_custom.py", actual)
        self.assertEqual((self.package / manifest["config_file"]).read_bytes(), self.config.read_bytes())
        self.assertIn("--data-dir", (self.package / "README.md").read_text(encoding="utf-8"))

    def test_existing_release_is_never_overwritten(self):
        before = (self.package / release.MANIFEST).read_bytes()
        with self.assertRaises(FileExistsError):
            release.export_release(self.package, self.config)
        self.assertEqual(before, (self.package / release.MANIFEST).read_bytes())

    def test_changed_runtime_is_rejected_before_executing_anything(self):
        target = self.directory / "changed_release"
        shutil.copytree(self.package, target)
        with (target / "code/run_task1.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# changed fixture\n")
        with patch.object(release, "_run", side_effect=AssertionError("executed changed code")), \
                self.assertRaisesRegex(ValueError, "Missing or changed release file"):
            release.verify_release(target)

    def test_unsafe_manifest_config_path_is_rejected_before_execution(self):
        target = self.directory / "unsafe_release"
        shutil.copytree(self.package, target)
        path = target / release.MANIFEST
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["config_file"] = "../outside.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(release, "_run", side_effect=AssertionError("executed unsafe package")), \
                self.assertRaisesRegex(ValueError, "safe relative"):
            release.verify_release(target)

    def test_external_configs_and_private_paths_are_rejected_without_creating_output(self):
        for index, extra in enumerate((
            {"external_data": {}}, {"data_dir": "C:\\Users\\private\\dataset"},
            {"token": "fixture-not-a-secret"},
        )):
            config = self.directory / f"bad_fixture_{index}.json"
            config.write_text(json.dumps(self.raw | extra), encoding="utf-8")
            target = self.directory / f"should_not_exist_{index}"
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                release.export_release(target, config)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
