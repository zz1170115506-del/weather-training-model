"""Retired source-label paths must fail closed; fixtures contain no real data."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset, small_config
import test_task1_results as result_fixtures
from task1.config import resolve_config
from task1.pipeline import (execute, require_course_only_bundle, run_cv, run_finalize,
                            run_predict, run_train, record_score)
from task1.plots import export_results
from task1.results import compare_runs, print_results, summarize_run
from task1.validation import cross_validate


class CourseOnlyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / "outputs"
        self.directory = self.root / ("course_only_test_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.data = synthetic_dataset()

    def cleanup(self):
        target = self.directory.resolve()
        if target.parent != self.root.resolve() or not target.name.startswith("course_only_test_"):
            raise RuntimeError("Unsafe test cleanup target.")
        shutil.rmtree(target)

    def test_external_config_and_true_model_flag_are_explicitly_rejected(self):
        for value in (None, {}, {"path": "not_read.csv"}):
            config = small_config()
            config["external_data"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "External data is disabled"):
                resolve_config(config)
        config = small_config()
        config["models"][0]["use_external_data"] = True
        with self.assertRaisesRegex(ValueError, "External data is disabled"):
            resolve_config(config)
        config["models"][0]["use_external_data"] = False
        self.assertFalse(resolve_config(config)["models"][0]["use_external_data"])

    def test_check_rejects_retired_recipe_before_loading_any_data(self):
        config = small_config()
        config["external_data"] = {"path": "no_source_read"}
        path = self.directory / "retired.json"
        result_fixtures.write_json(path, config)
        args = Namespace(stage="check", config=path, data_dir=self.directory, label_format="integer")
        with patch("task1.pipeline.load_data", side_effect=AssertionError("data read")), \
                self.assertRaisesRegex(ValueError, "External data is disabled"):
            execute(args)

    def test_optional_course_hash_pin_has_exact_schema_and_preserves_legacy_configs(self):
        base = {"train.csv": "a" * 64, "test.csv": "b" * 64, "sample_submission.csv": "c" * 64}
        valid = small_config()
        valid["expected_source_sha256"] = base
        self.assertEqual(resolve_config(valid)["expected_source_sha256"], base)
        self.assertNotIn("expected_source_sha256", small_config())
        for hashes in ({}, {"train.csv": "a" * 64}, base | {"extra.csv": "d" * 64},
                       base | {"test.csv": "b" * 63}, base | {"test.csv": "B" * 64},
                       base | {"test.csv": True}):
            config = small_config()
            config["expected_source_sha256"] = hashes
            with self.subTest(hashes=hashes), self.assertRaisesRegex(ValueError, "expected_source_sha256"):
                resolve_config(config)

    def test_crossfit_parameters_resolve_before_training_and_weighting_modes_cannot_mix(self):
        config = small_config()
        config["models"][0]["model_id"] = "hgb"
        config["models"][0]["model_params"] = {}
        config["models"][0]["crossfit_weighting"] = {"floor": 0.6}
        resolved = resolve_config(config)
        weighting = resolved["models"][0]["crossfit_weighting"]
        self.assertEqual(weighting["floor"], 0.6)
        self.assertEqual(weighting["n_splits"], 3)
        self.assertEqual(weighting["teacher_params"]["max_iter"], 600)
        config["models"][0]["noise_weighting"] = {}
        with self.assertRaisesRegex(ValueError, "not both"):
            resolve_config(config)
        config["models"][0].pop("noise_weighting")
        config["models"][0]["crossfit_weighting"] = {"threshold": 1.0}
        with self.assertRaisesRegex(ValueError, "threshold"):
            resolve_config(config)

    def test_hash_mismatch_blocks_check_and_direct_training_apis_before_any_fit(self):
        config = small_config()
        config["expected_source_sha256"] = {
            "train.csv": "a" * 64, "test.csv": "b" * 64, "sample_submission.csv": "c" * 64}
        self.data.audit["source_sha256"] = config["expected_source_sha256"] | {"train.csv": "d" * 64}
        config_file = self.directory / "pinned.json"
        result_fixtures.write_json(config_file, config)
        args = Namespace(stage="check", config=config_file, data_dir=self.directory, label_format="integer")
        with patch("task1.pipeline.load_data", return_value=self.data), \
                patch("task1.pipeline.make_model", side_effect=AssertionError("model creation")), \
                self.assertRaisesRegex(ValueError, "Course dataset SHA256 mismatch"):
            execute(args)
        with patch("task1.pipeline.fit_model", side_effect=AssertionError("fit started")), \
                patch("task1.pipeline.cross_validate", side_effect=AssertionError("CV started")):
            for operation in (
                lambda: run_cv(self.data, config, self.directory),
                lambda: run_train(self.data, config, self.directory, "integer"),
                lambda: run_finalize(self.data, config, self.directory, "integer"),
            ):
                with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "Course dataset SHA256 mismatch"):
                    operation()

    def test_direct_cv_and_finalize_refuse_augmented_context_before_fit(self):
        self.data.audit["external_data"] = {"policy": "retired"}
        config = small_config()
        with patch("task1.validation.fit_model", side_effect=AssertionError("fit started")), \
                self.assertRaisesRegex(ValueError, "External data is disabled"):
            cross_validate(self.data, config["models"][0], [0] * len(self.data.y), verbose=False)
        with patch("task1.pipeline.fit_model", side_effect=AssertionError("fit started")), \
                self.assertRaisesRegex(ValueError, "External data is disabled"):
            run_finalize(self.data, config, self.directory, "integer")

    def test_old_source_bundle_cannot_produce_a_new_submission(self):
        saved = {"config": {}, "external_data_provenance": {"source_audit": {"source_rows": 13200}}}
        destination = self.directory / "should_not_exist.csv"
        with patch("task1.pipeline.joblib.load", return_value=saved), \
                patch("task1.pipeline.predict_probabilities", side_effect=AssertionError("inference started")), \
                self.assertRaisesRegex(ValueError, "source-assisted models are disabled"):
            run_predict(self.data, self.directory / "retired.joblib", destination, "integer")
        self.assertFalse(destination.exists())

    def test_bundle_actual_source_usage_and_nested_flags_are_rejected(self):
        for saved in (
            {"config": {}, "public_source_test_overlap_used": True},
            {"config": {}, "external_data_provenance": {"actual_training": {"old": {"external_fit_rows": 1}}}},
            {"config": {}, "models": {"old": {"config": {"use_external_data": True}}}},
        ):
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                require_course_only_bundle(saved)

    def test_legacy_course_only_bundles_remain_eligible(self):
        for saved in (
            {"config": small_config()},
            {"config": small_config(), "external_data_provenance": {
                "source_audit": None, "actual_training": {"old": {
                    "external_fit_rows": 0, "test_overlap_source_rows_used": 0,
                    "source_sha256": None, "source_hashes": {}, "augmentation_applied": False}}}},
        ):
            require_course_only_bundle(saved)

    def test_historical_source_results_are_readable_but_not_current_candidates(self):
        fixture = result_fixtures.ResultsTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        retired = fixture.fixture("retired")
        fixture.update_summary(retired, lambda value: value["candidates"]["candidate_a"]["fit_and_export"].update(
            public_source_test_overlap_used=True))
        summary = summarize_run(retired)
        self.assertFalse(summary["current_policy_eligible"])
        output = io.StringIO()
        with redirect_stdout(output):
            print_results(summary)
        self.assertIn("当前规则已禁止", output.getvalue())
        self.assertNotIn("整数上传 CSV:", output.getvalue())
        self.assertNotIn("已导出候选中，本地 OOF 最高", output.getvalue())
        with self.assertRaisesRegex(ValueError, "course-only policy"):
            compare_runs(summary, summarize_run(fixture.fixture("eligible")))
        args = Namespace(run_dir=retired, candidate="candidate_a", submission_id="fixture-only",
                         public_score=0.98, note="Synthetic historical fixture, not an actual platform score.")
        with self.assertRaisesRegex(ValueError, "excluded by the course-only policy"):
            record_score(args)
        self.assertFalse((retired / "platform_scores.json").exists())

    def test_exporter_will_not_republish_retired_source_run(self):
        run = self.directory / "retired"
        result_fixtures.write_json(run / "run.json", {"status": "complete", "external_data_provenance": {"policy": "retired"}})
        result_fixtures.write_json(run / "models/old/summary.json", {})
        destination = self.directory / "must_not_be_created"
        with self.assertRaisesRegex(ValueError, "course-only policy"):
            export_results([run], destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
