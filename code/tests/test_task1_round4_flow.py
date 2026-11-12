"""Round-four orchestration on 32 synthetic records; no competition fitting."""
from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.artifacts import finish_run, new_run
from task1.config import resolve_config
from task1.pipeline import record_score, run_finalize, run_predict, run_train
from task1.results import summarize_run


def search_config():
    return resolve_config({"schema_version": 1,
        "validation": {"n_splits": 2, "split_seed": 83, "model_seed": 91},
        "models": [
            {"name": "a_cart", "model_id": "numpy_cart", "model_params": {"max_depth": 4}},
            {"name": "b_extra", "model_id": "extra_trees", "model_params": {"n_estimators": 3, "min_samples_leaf": 1}}],
        "search": {"inner_splits": 2, "blend_pool_size": 2, "minimum_blend_gain": .0005, "min_blend_fold_wins": 2},
        "blend": {"n_trials": 0, "allow_class_multipliers": False}})


class SearchWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / "outputs"
        self.directory = self.root / ("r4_flow_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.data = synthetic_dataset()
        self.data.audit["source_sha256"] = {key: "a" * 64 for key in self.data.audit["source_sha256"]}

    def cleanup(self):
        target = self.directory.resolve()
        if target.parent != self.root.resolve() or not target.name.startswith("r4_flow_"):
            raise RuntimeError("Unsafe round-four test cleanup.")
        shutil.rmtree(target)

    def test_nested_search_full_fit_exports_and_read_only_summary(self):
        config = search_config()
        with redirect_stdout(io.StringIO()), threadpool_limits(limits=1):
            out = new_run("train", config, self.directory)
            snapshot = (out / "resolved_config.json").read_bytes()
            result = run_train(self.data, config, out, "integer")
            finish_run(out, {"source_sha256": self.data.audit["source_sha256"], "result": result})
            summary = summarize_run(out)
        self.assertEqual((out / "resolved_config.json").read_bytes(), snapshot)
        self.assertEqual(result["search"]["selected_models"], [value["model_name"] for value in result["candidates"].values()])
        self.assertFalse(result["search"]["blend_export_eligible"])
        self.assertIsNone(summary["research_blend"])
        self.assertEqual(len(summary["models"]), 2)
        self.assertTrue((out / "selected_config.json").is_file())
        self.assertTrue((out / "search_summary.json").is_file())
        for candidate in result["candidates"].values():
            self.assertTrue(candidate["fit_and_export"]["saved_model_reload_exact_match"])
            self.assertEqual(candidate["submission"]["label_format"], "integer")
        args = Namespace(run_dir=out, candidate="candidate_b", submission_id="synthetic-r4-only",
                         public_score=.5, note="Synthetic fixture, not a competition score.")
        with redirect_stdout(io.StringIO()):
            record_score(args)
        self.assertEqual(summarize_run(out)["platform_scores"][0]["candidate"], "candidate_b")

    def test_frozen_recipe_is_replayed_without_a_new_weight_search(self):
        config = search_config()
        config.pop("search")
        config["prediction_strategy"] = "learned_blend"
        recipe = {"schema_version": 1, "labels": [1, 2, 3, 4], "model_ids": ["a_cart", "b_extra"],
                  "weights": [.7, .3], "class_multipliers": [1., 1., 1., 1.],
                  "selection_scope": "Handwritten synthetic fixture; not a measured result."}
        with patch("task1.pipeline._training_recipe", side_effect=AssertionError("new weight search")), threadpool_limits(limits=1):
            result = run_finalize(self.data, config, self.directory, "integer", frozen_recipe=recipe)
            replay = run_predict(self.data, self.directory / "model_bundle.joblib", self.directory / "replayed.csv", "integer")
        self.assertEqual(result["recipe"], recipe)
        self.assertEqual(replay["csv_sha256"], result["submission"]["csv_sha256"])
        self.assertEqual(result["weight_selection_protocol"]["method"], "frozen_full_training_oof_recipe")

    def test_invalid_frozen_membership_is_rejected_before_fit(self):
        config = search_config()
        config["prediction_strategy"] = "learned_blend"
        recipe = {"schema_version": 1, "labels": [1, 2, 3, 4], "model_ids": ["unknown"],
                  "weights": [1.], "class_multipliers": [1.] * 4}
        with patch("task1.pipeline.fit_model", side_effect=AssertionError("must reject before fitting")), self.assertRaises(ValueError):
            run_finalize(self.data, config, self.directory, "integer", frozen_recipe=recipe)

    def test_search_schema_rejects_invalid_and_excessive_budgets(self):
        for key, value in (("inner_splits", True), ("inner_splits", 1), ("blend_pool_size", 3),
                           ("minimum_blend_gain", float("nan")), ("minimum_blend_gain", -1),
                           ("min_blend_fold_wins", 3)):
            raw = deepcopy(search_config())
            raw["search"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                resolve_config(raw)


if __name__ == "__main__":
    unittest.main()
