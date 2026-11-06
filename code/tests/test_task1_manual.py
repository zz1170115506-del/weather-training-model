"""Manual-run orchestration checks use synthetic records, never competition data."""
from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset, small_config
from task1.artifacts import new_run, finish_run
from task1.config import resolve_config
from task1.features import Features
from task1.models import fit_model
from task1.pipeline import execute, record_score, run_finalize, run_train


class ManualWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[2] / "outputs" / ("manual_test_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.data = synthetic_dataset()
        self.data.audit["rows"] = {"train": len(self.data.train), "test": len(self.data.test), "sample": len(self.data.sample)}
        self.data.audit["source_sha256"] = {key: "a" * 64 for key in self.data.audit["source_sha256"]}

    def tearDown(self):
        target = self.directory.resolve()
        root = Path(__file__).resolve().parents[2].resolve() / "outputs"
        if not target.is_relative_to(root) or not target.name.startswith("manual_test_"):
            raise RuntimeError("Unsafe temporary-test cleanup path.")
        shutil.rmtree(target)

    def test_two_fixed_exports_are_lossless_and_never_overwrite(self):
        config = small_config()
        second = deepcopy(config["models"][0])
        second["name"] = "second_cart"
        config["models"].append(second)
        config["training"] = {"export_models": [second["name"], "synthetic_cart"]}
        with redirect_stdout(io.StringIO()):
            summary = run_train(self.data, config, self.directory, "sample")
        self.assertEqual(summary["candidates"]["candidate_a"]["model_name"], "second_cart")
        a = self.directory / "submissions/candidate_a.csv"
        b = self.directory / "submissions/candidate_b.csv"
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertIn(b"1003,Rainy", a.read_bytes())
        self.assertTrue((self.directory / "submissions/candidate_a_integer.csv").exists())
        self.assertFalse(summary["candidates"]["candidate_a"]["platform_acceptance_verified"])
        before = a.read_bytes()
        with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            run_train(self.data, config, self.directory, "sample")
        self.assertEqual(a.read_bytes(), before)

    def test_record_score_is_local_append_only_and_rejects_changed_csv(self):
        with redirect_stdout(io.StringIO()):
            out = new_run("train", small_config(), self.directory)
            summary = run_train(self.data, small_config(), out, "integer")
            finish_run(out, {"source_sha256": self.data.audit["source_sha256"], "result": summary})
        args = Namespace(run_dir=out, candidate="candidate_a", submission_id="real-id-from-test-fixture",
                         public_score=0.7, note="Synthetic test entry, not a competition result.")
        with redirect_stdout(io.StringIO()):
            record_score(args)
        log = json.loads((out / "platform_scores.json").read_text(encoding="utf-8"))
        self.assertEqual(log[0]["public_macro_f1"], 0.7)
        self.assertIn("User-reported", log[0]["source"])
        with self.assertRaises(ValueError):
            record_score(args)

        args.submission_id = "another-test-id"
        (out / "submissions/candidate_a.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            record_score(args)

    def test_legacy_name_run_records_the_integer_file_actually_uploaded(self):
        with redirect_stdout(io.StringIO()):
            out = new_run("train", small_config(), self.directory)
            summary = run_train(self.data, small_config(), out, "name")
            finish_run(out, {"source_sha256": self.data.audit["source_sha256"], "result": summary})
        candidate = summary["candidates"]["candidate_a"]
        english_hash = candidate["submission"]["csv_sha256"]
        integer_hash = candidate["fit_and_export"]["alternative_submission"]["csv_sha256"]
        self.assertNotEqual(english_hash, integer_hash)
        args = Namespace(run_dir=out, candidate="candidate_a", submission_id="legacy-integer-fixture",
                         public_score=0.8, note="Synthetic fixture only.")
        with redirect_stdout(io.StringIO()):
            record_score(args)
        saved = json.loads((out / "platform_scores.json").read_text(encoding="utf-8"))[0]
        self.assertEqual(saved["csv_sha256"], integer_hash)
        self.assertEqual(saved["label_format"], "integer")
        args.submission_id = "another-test-id"
        (out / "submissions/candidate_a.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            record_score(args)

    def test_record_score_requires_explicit_candidate_before_reading_run(self):
        args = Namespace(run_dir=self.directory / "not_a_run", candidate=None,
                         submission_id="fixture", public_score=0.9)
        with self.assertRaisesRegex(ValueError, "--candidate"):
            record_score(args)

    def test_check_does_not_fit_or_start_training(self):
        config_file = self.directory / "config.json"
        config_file.write_text(json.dumps(small_config()), encoding="utf-8")
        args = Namespace(stage="check", data_dir=self.directory, config=config_file, artifact_root=None, label_format="integer")
        output = io.StringIO()
        with patch("task1.pipeline.load_data", return_value=self.data), \
                patch("task1.pipeline.fit_model", side_effect=AssertionError("check started training")), \
                patch("task1.pipeline.new_run", side_effect=AssertionError("check created a run")), \
                redirect_stdout(output):
            execute(args)
        self.assertFalse(json.loads(output.getvalue())["training_started"])

    def test_train_command_completes_and_points_to_only_completed_local_run(self):
        config_file = self.directory / "config.json"
        config_file.write_text(json.dumps(small_config()), encoding="utf-8")
        args = Namespace(stage="train", data_dir=self.directory, config=config_file,
                         artifact_root=self.directory, label_format="integer", output=None,
                         allow_dirty=False)
        self.data.audit["source_sha256"] = {key: "a" * 64 for key in self.data.audit["source_sha256"]}
        with patch("task1.pipeline.load_data", return_value=self.data), \
                patch("task1.pipeline.git_state", return_value={"dirty": False}), \
                patch("task1.plots.export_results", return_value={"status": "tested separately"}), \
                redirect_stdout(io.StringIO()):
            execute(args)
        pointer = json.loads((self.directory / "latest_training_run.json").read_text(encoding="utf-8"))
        out = Path(pointer["run_dir"])
        self.assertTrue(Path(pointer["candidates"]["candidate_a"]).is_file())
        self.assertEqual(json.loads((out / "run.json").read_text(encoding="utf-8"))["status"], "complete")
        manifest = json.loads((out / "run.json").read_text(encoding="utf-8"))
        self.assertIn((out / "resolved_config.json").as_posix(), manifest["frozen_reproduce_command_template"])
        self.assertEqual(json.loads((out / "training_summary.json").read_text(encoding="utf-8"))["figures"]["status"], "tested separately")
        self.assertFalse((out / "platform_scores.json").exists())

    def test_training_guard_does_not_start_when_shared_source_is_dirty(self):
        config_file = self.directory / "config.json"
        config_file.write_text(json.dumps(small_config()), encoding="utf-8")
        args = Namespace(stage="train", data_dir=self.directory, config=config_file,
                         artifact_root=self.directory, label_format="integer", output=None,
                         allow_dirty=False)
        with patch("task1.pipeline.load_data", return_value=self.data), \
                patch("task1.pipeline.git_state", return_value={"dirty": True}), \
                patch("task1.pipeline.new_run", side_effect=AssertionError("started unexpectedly")), \
                self.assertRaises(ValueError):
            execute(args)

    def test_multi_model_finalize_requires_explicit_research_blend_opt_in(self):
        config = small_config()
        second = deepcopy(config["models"][0])
        second["name"] = "control"
        config["models"].append(second)
        with patch("task1.pipeline.fit_model", side_effect=AssertionError("training started")), self.assertRaises(ValueError):
            run_finalize(self.data, config, self.directory, "name")

    @threadpool_limits.wrap(limits=1)
    def test_training_only_noise_weights_match_all_one_control(self):
        config = {"name": "hgb_test", "model_id": "hgb",
                  "model_params": {"max_iter": 4, "max_leaf_nodes": 3, "min_samples_leaf": 2}}
        weighted = deepcopy(config)
        weighted["noise_weighting"] = {"floor": 1.0, "n_bins": 4, "max_iter": 2}
        plain, _ = fit_model(self.data.X, self.data.y, config, 43)
        with_weights, metadata = fit_model(self.data.X, self.data.y, weighted, 43)
        X = plain["features"].transform(self.data.X)
        np.testing.assert_allclose(plain["model"].predict_proba(X), with_weights["model"].predict_proba(X))
        self.assertEqual(metadata["noise_weighting"]["mean_weight"], 1.0)
        self.assertEqual(metadata["preprocessing_fitted_rows"], len(self.data.train))

    def test_native_histogram_categories_are_fit_only_on_training(self):
        feature = Features(native_categories="hgb").fit(self.data.X.iloc[:4])
        valid = self.data.X.iloc[:1].copy()
        valid["Cloud Cover"] = "never-seen"
        transformed = feature.transform(valid)
        self.assertEqual(str(transformed["Cloud Cover"].dtype), "category")
        self.assertTrue(transformed["Cloud Cover"].isna().all())
        self.assertNotIn("never-seen", feature.category_levels_["Cloud Cover"])

    def test_exports_and_noise_options_are_validated_before_fitting(self):
        for training in ({"export_models": []}, {"export_models": ["missing"]},
                         {"export_models": ["synthetic_cart", "synthetic_cart"]}):
            config = small_config()
            config["training"] = training
            with self.subTest(training=training), self.assertRaises(ValueError):
                resolve_config(config)
        config = small_config()
        config["models"][0]["noise_weighting"] = {"floor": -0.1}
        with self.assertRaises(ValueError):
            resolve_config(config)


if __name__ == "__main__":
    unittest.main()
