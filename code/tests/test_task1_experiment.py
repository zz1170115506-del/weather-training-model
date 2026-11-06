"""Configuration provenance checks using handwritten artifacts, never models."""

from copy import deepcopy
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1.artifacts import content_hash, file_hash, new_run
from task1.config import resolve_config, validate_experiment
from task1.results import compare_runs, print_results, summarize_run


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.output_area = Path(__file__).resolve().parents[2] / "outputs"
        self.output_area.mkdir(exist_ok=True)
        self.directory = self.output_area / f"experiment_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup_directory)

    def cleanup_directory(self):
        target = self.directory.resolve()
        if target.parent != self.output_area.resolve() or not target.name.startswith("experiment_test_"):
            raise RuntimeError("Refusing cleanup outside the experiment test area.")
        shutil.rmtree(target)

    def raw_config(self):
        return {"schema_version": 1, "models": [{"name": "hgb_round03_120", "model_id": "hgb",
                                                  "model_params": {"max_iter": 120}}]}

    def experiment(self):
        return {"round": 3, "label": "round3_hgb120",
                "parent_run_id": "20261106T125741Z_train_cd70c482_5b00b4",
                "change_note": "HGB max_iter=120; preserve the model architecture."}

    def fixture(self, name="fixture"):
        """A four-row export and handwritten CV summaries, with no fit call."""
        root = self.directory / name
        root.mkdir()
        model = self.raw_config()["models"][0]["name"]
        row = {"name": model, "model_id": "hgb", "fold_macro_f1_mean": 0.916,
               "fold_macro_f1_std": 0.005, "oof_macro_f1": 0.9159}
        with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        write_json(root / "models" / model / "summary.json",
                   {**row, "oof_metrics": {"macro_f1": 0.9159, "accuracy": 0.915}})
        content = b"id,label\n1,1\n2,2\n3,3\n4,4\n"
        (root / "submissions").mkdir()
        (root / "submissions/candidate_a.csv").write_bytes(content)
        canonical = b'task1-predictions-v1\n["integer","1",1]\n["integer","2",2]\n["integer","3",3]\n["integer","4",4]\n'
        metadata = {"rows": 4, "schema": ["id", "label"], "id_type": "integer", "label_format": "integer",
                    "class_counts": {str(k): 1 for k in (1, 2, 3, 4)},
                    "csv_sha256": hashlib.sha256(content).hexdigest(),
                    "canonical_prediction_sha256": hashlib.sha256(canonical).hexdigest(),
                    "canonical_prediction_version": "task1-predictions-v1"}
        protocol = {"method": "StratifiedKFold", "n_splits": 2, "seed": 5434, "fold_sha256": "f" * 64}
        write_json(root / "training_summary.json",
                   {"comparison": {"fold_protocol": protocol, "metrics": [row]},
                    "candidates": {"candidate_a": {"model_name": model, "csv": "submissions/candidate_a.csv",
                                                   "submission": metadata, "fit_and_export": {"full_test_rows": 4}}}})
        write_json(root / "run.json", {"run_id": name, "stage": "train", "status": "complete",
                   "source_sha256": {"train.csv": "a" * 64, "test.csv": "b" * 64, "sample_submission.csv": "c" * 64}})
        self.reseal(root)
        return root

    def reseal(self, root):
        path = root / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files_sha256"] = {file.relative_to(root).as_posix(): file_hash(file)
                                    for file in root.rglob("*") if file.is_file() and file.name != "run.json"}
        write_json(path, manifest)

    def attach_snapshot(self, root, experiment=True):
        raw = self.raw_config()
        raw["models"][0]["noise_weighting"] = {"floor": 0.2}
        if experiment:
            raw["experiment"] = self.experiment()
        raw["diagnostics"] = {"learning_curve_every": 10}
        config = resolve_config(raw)
        write_json(root / "resolved_config.json", config)
        path = root / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(config_sha256=content_hash(config), resolved_config_sha256=content_hash(config),
                        resolved_config_file_sha256=file_hash(root / "resolved_config.json"))
        if experiment:
            manifest["experiment"] = config["experiment"]
        write_json(path, manifest)
        self.reseal(root)
        return config

    def test_old_configs_do_not_gain_experiment_or_diagnostic_defaults(self):
        result = resolve_config(self.raw_config())
        self.assertNotIn("experiment", result)
        self.assertNotIn("diagnostics", result)
        self.assertEqual(set(result), {"schema_version", "models", "validation", "finalization",
                                      "prediction_strategy", "blend", "training"})

    def test_valid_experiment_and_diagnostics_are_preserved_without_mutating_raw(self):
        raw = {**self.raw_config(), "experiment": self.experiment(), "diagnostics": {"learning_curve_every": 10}}
        original = deepcopy(raw)
        result = resolve_config(raw)
        self.assertEqual(result["experiment"], original["experiment"])
        self.assertEqual(result["diagnostics"], {"learning_curve_every": 10})
        self.assertEqual(raw, original)

    def test_experiment_round_rejects_bool_float_nonpositive_and_strings(self):
        for value in (True, False, 0, -1, 3.0, "3", None):
            experiment = {**self.experiment(), "round": value}
            with self.subTest(round=value), self.assertRaises(ValueError):
                resolve_config({**self.raw_config(), "experiment": experiment})

    def test_experiment_requires_exact_fields_safe_names_and_short_note(self):
        base = self.experiment()
        bad = [None, {}, {**base, "extra": "ignored"}, {key: value for key, value in base.items() if key != "parent_run_id"},
               {**base, "label": "../round3"}, {**base, "label": "a" * 81},
               {**base, "parent_run_id": "C:/some/run"}, {**base, "parent_run_id": ""},
               {**base, "change_note": " "}, {**base, "change_note": "a" * 501},
               {**base, "change_note": "two\nlines"}, {**base, "change_note": 3}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_experiment(value)

    def test_diagnostics_rejects_unknown_or_invalid_interval(self):
        for diagnostics in (None, {}, {"learning_curve_every": 10, "auto_select": True},
                            {"learning_curve_every": 0}, {"learning_curve_every": -1},
                            {"learning_curve_every": True}, {"learning_curve_every": 10.0},
                            {"learning_curve_every": "10"}):
            with self.subTest(diagnostics=diagnostics), self.assertRaises(ValueError):
                resolve_config({**self.raw_config(), "diagnostics": diagnostics})

    def test_new_run_saves_round_semantic_and_file_hashes_without_git_operations(self):
        config = resolve_config({**self.raw_config(), "experiment": self.experiment()})
        with patch("task1.artifacts.git_state", return_value={"available": False}), patch("task1.artifacts.environment", return_value={}):
            root = new_run("train", config, self.directory / "artifacts")
            second = new_run("train", config, self.directory / "artifacts")
        manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["experiment"], self.experiment())
        self.assertEqual(manifest["config_sha256"], content_hash(config))
        self.assertEqual(manifest["resolved_config_sha256"], content_hash(config))
        self.assertEqual(manifest["resolved_config_file_sha256"], file_hash(root / "resolved_config.json"))
        self.assertNotEqual(root, second)
        config["models"][0]["model_params"]["max_iter"] = 500
        saved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["models"][0]["model_params"]["max_iter"], 120)

    def test_new_untagged_run_does_not_invent_round(self):
        with patch("task1.artifacts.git_state", return_value={"available": False}), patch("task1.artifacts.environment", return_value={}):
            root = new_run("train", resolve_config(self.raw_config()), self.directory / "artifacts")
        self.assertNotIn("experiment", json.loads((root / "run.json").read_text(encoding="utf-8")))

    def test_legacy_result_without_snapshot_does_not_infer_round_from_model_name(self):
        summary = summarize_run(self.fixture())
        self.assertIsNone(summary["experiment"])
        self.assertEqual(summary["config_snapshot"]["status"], "unavailable_legacy_run")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print_results(summary)
        self.assertIn("历史记录未标注", stream.getvalue())
        self.assertNotIn("第 3 轮", stream.getvalue())

    def test_result_verifies_snapshot_and_prints_separate_hgb_and_weight_iterations(self):
        root = self.fixture()
        config = self.attach_snapshot(root)
        summary = summarize_run(root)
        self.assertEqual(summary["experiment"], self.experiment())
        self.assertEqual(summary["config_snapshot"]["status"], "verified")
        self.assertEqual(summary["config_snapshot"]["resolved_config_sha256"], content_hash(config))
        self.assertEqual(summary["config_snapshot"]["diagnostics"], {"learning_curve_every": 10})
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print_results(summary)
        output = stream.getvalue()
        self.assertIn("第 3 轮", output)
        self.assertIn("max_iter=120", output)
        self.assertIn("max_iter=30", output)
        self.assertIn("显式设置", output)
        json.dumps(summary, allow_nan=False)

    def test_legacy_snapshot_with_no_round_is_verified_without_new_provenance_fields(self):
        root = self.fixture()
        self.attach_snapshot(root, experiment=False)
        path = root / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.pop("resolved_config_sha256")
        manifest.pop("resolved_config_file_sha256")
        write_json(path, manifest)
        summary = summarize_run(root)
        self.assertIsNone(summary["experiment"])
        self.assertEqual(summary["config_snapshot"]["status"], "verified")

    def test_snapshot_semantic_hash_mismatch_is_rejected_even_after_file_inventory_update(self):
        root = self.fixture()
        self.attach_snapshot(root)
        path = root / "resolved_config.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["models"][0]["model_params"]["max_iter"] = 999
        write_json(path, value)
        self.reseal(root)
        with self.assertRaisesRegex(ValueError, "semantic SHA256 mismatch"):
            summarize_run(root)

    def test_snapshot_file_hash_mismatch_is_rejected_even_with_identical_parameters(self):
        root = self.fixture()
        self.attach_snapshot(root)
        path = root / "resolved_config.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(value, indent=4), encoding="utf-8")
        self.reseal(root)
        with self.assertRaisesRegex(ValueError, "file SHA256 mismatch"):
            summarize_run(root)

    def test_tagged_run_must_have_snapshot_and_matching_experiment(self):
        root = self.fixture()
        path = root / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["experiment"] = self.experiment()
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "requires resolved_config"):
            summarize_run(root)
        self.attach_snapshot(root)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["experiment"]["round"] = 4
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "experiment metadata.*disagree"):
            summarize_run(root)

    def test_comparison_reports_actual_current_round_without_inventing_previous_round(self):
        previous = self.fixture("previous")
        current = self.fixture("current")
        self.attach_snapshot(current)
        comparison = compare_runs(current, previous)
        self.assertEqual(comparison["current_experiment"], self.experiment())
        self.assertIsNone(comparison["previous_experiment"])
        self.assertEqual(comparison["candidates"]["candidate_a"]["oof_macro_f1_delta"], 0)


if __name__ == "__main__":
    unittest.main()
