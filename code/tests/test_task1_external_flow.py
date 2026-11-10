"""Source-assisted training boundaries on 32 synthetic course records only."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.artifacts import file_hash, finish_run, new_run, save_json
from task1.config import resolve_config
from task1.external import attach_external, feature_keys
from task1.models import fit_model
from task1.pipeline import run_predict, run_train
from task1.results import compare_runs, summarize_run
from task1.validation import cross_validate, make_folds


class ExternalFlowTests(unittest.TestCase):
    def setUp(self):
        self.area = Path(__file__).resolve().parents[2] / "outputs"
        self.directory = self.area / ("external_flow_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.original = synthetic_dataset()
        self.original.test["Temperature"] += .125  # distinct course test identities
        self.original.audit["source_sha256"] = {key: "a" * 64 for key in self.original.audit["source_sha256"]}
        names = {1: "Sunny", 2: "Cloudy", 3: "Rainy", 4: "Snowy"}
        course = self.original.X.copy()
        course["Weather Type"] = [names[label] for label in self.original.y]
        test = self.original.test.drop(columns="id").copy()
        test["Weather Type"] = [names[label] for label in range(1, 5)]
        self.source = self.directory / "source.csv"
        pd.concat([course, test], ignore_index=True).to_csv(self.source, index=False)
        self.config = resolve_config({"schema_version": 1,
            "validation": {"n_splits": 2, "split_seed": 83, "model_seed": 91},
            "external_data": {"path": str(self.source), "sha256": file_hash(self.source),
                              "source_url": "https://example.org/synthetic-test-fixture", "policy": "public_source_training"},
            "models": [
                {"name": "tiny_control", "model_id": "hgb", "use_external_data": False,
                 "model_params": {"max_iter": 3, "min_samples_leaf": 2}},
                {"name": "tiny_source", "model_id": "hgb", "use_external_data": True,
                 "model_params": {"max_iter": 3, "min_samples_leaf": 2}}],
            "training": {"export_models": ["tiny_source"]}})
        self.data = attach_external(self.original, self.config["external_data"], self.directory)
        self.folds, _ = make_folds(self.data.y, self.data.train.id, self.data.groups, 2, 83)

    def cleanup(self):
        target = self.directory.resolve()
        if target.parent != self.area.resolve() or not target.name.startswith("external_flow_"):
            raise RuntimeError("Unsafe external-flow test cleanup.")
        shutil.rmtree(target)

    def test_course_validation_copies_never_enter_source_augmented_fit(self):
        observed = []
        def spy(X, y, config, seed):
            fold = len(observed)
            keys = set(feature_keys(X))
            self.assertFalse(keys & set(feature_keys(self.data.X.iloc[np.flatnonzero(self.folds == fold)])))
            self.assertEqual(len(X), 20)  # course16 plus public-source4
            observed.append(len(X))
            return fit_model(X, y, config, seed)
        with patch("task1.validation.fit_model", side_effect=spy), threadpool_limits(limits=1):
            summary, oof = cross_validate(self.data, self.config["models"][1], self.folds, 91, verbose=False)
        self.assertEqual(oof.shape, (32, 4))
        self.assertEqual(observed, [20, 20])
        for fold in summary["folds"]:
            self.assertEqual((fold["course_fit_rows"], fold["external_fit_rows"], fold["combined_fit_rows"]), (16, 4, 20))
            self.assertEqual(fold["preprocessing_fitted_rows"], 20)
        self.assertEqual(len(self.data.train), 32)

    def test_control_fit_does_not_consume_available_external_pool(self):
        with threadpool_limits(limits=1):
            summary, _ = cross_validate(self.data, self.config["models"][0], self.folds, 91, verbose=False)
        for fold in summary["folds"]:
            self.assertEqual(fold["external_fit_rows"], 0)
            self.assertEqual(fold["preprocessing_fitted_rows"], 16)

    def test_full_source_workflow_discloses_overlap_and_replays_without_source_file(self):
        with redirect_stdout(io.StringIO()), threadpool_limits(limits=1):
            out = new_run("train", self.config, self.directory)
            save_json(out / "audit.json", self.data.audit)
            result = run_train(self.data, self.config, out, "integer")
            finish_run(out, {"source_sha256": self.data.audit["source_sha256"],
                             "external_data_provenance": self.data.audit["external_data"], "result": result})
            summary = summarize_run(out)
            candidate = result["candidates"]["candidate_a"]
            self.assertTrue(candidate["fit_and_export"]["public_source_test_overlap_used"])
            self.assertIn("not an unseen-test", candidate["fit_and_export"]["evaluation_scope"])
            self.source.unlink()  # owned fixture; inference must not reload training CSV
            replay = run_predict(self.original, out / candidate["bundle"], out / "again.csv", "integer")
        self.assertEqual(replay["csv_sha256"], candidate["submission"]["csv_sha256"])
        self.assertEqual(summary["external_data_provenance"]["test_overlap_source_rows_used"], 4)
        previous = deepcopy(summary)
        previous["external_data_provenance"] = None
        self.assertTrue(compare_runs(summary, previous)["external_training_changed"])
        self.assertIsNone(self.original.external)

    def test_schema_requires_explicit_policy_and_boolean_model_override(self):
        raw = deepcopy(self.config)
        del raw["external_data"]["policy"]
        with self.assertRaisesRegex(ValueError, "policy"):
            resolve_config(raw)
        raw = deepcopy(self.config)
        raw["models"][0]["use_external_data"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            resolve_config(raw)


if __name__ == "__main__":
    unittest.main()
