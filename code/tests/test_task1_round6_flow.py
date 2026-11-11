"""Exercise the real round-six recipe on synthetic rows, never course data."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
import uuid

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.artifacts import content_hash, finish_run, new_run
from task1.config import resolve_config
from task1.data import FEATURES
from task1.pipeline import run_predict, run_train
from task1.results import summarize_run
from task1.search import _pool


class RoundSixFlowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / "outputs"
        self.directory = self.root / ("r6_flow_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        target = self.directory.resolve()
        if target.parent != self.root.resolve() or not target.name.startswith("r6_flow_"):
            raise RuntimeError("Unsafe round-six synthetic test cleanup.")
        shutil.rmtree(target)

    def test_pool_treats_sampled_hgb_and_plain_hgb_as_the_same_family(self):
        rank = [{"name": "sampled", "model_id": "bagged_hgb"},
                {"name": "plain", "model_id": "hgb"},
                {"name": "kernel", "model_id": "svc"}]
        self.assertEqual(_pool(rank, 2), ["sampled", "kernel"])

    @unittest.skipUnless(importlib.util.find_spec("catboost"), "Optional CatBoost is not installed")
    def test_all_seven_round_six_recipes_nested_search_and_prediction_replay(self):
        repository = Path(__file__).resolve().parents[2]
        raw = json.loads((repository / "configs/task1/sixth_run_course_only.json").read_text(encoding="utf-8"))
        template = deepcopy(raw)
        # The production file pins course data. This fixture deliberately uses
        # different synthetic rows and never loads a course/source CSV.
        raw.pop("expected_source_sha256")
        raw["validation"]["n_splits"] = 2
        raw["search"]["inner_splits"] = 2
        raw["search"]["min_blend_fold_wins"] = 2
        raw["diagnostics"]["learning_curve_every"] = 1
        raw["blend"]["n_trials"] = 1
        for candidate in raw["models"]:
            params = candidate["model_params"]
            if "max_iter" in params:
                params["max_iter"] = 3
                params["min_samples_leaf"] = 2
            if "iterations" in params:
                params["iterations"] = 3
                candidate["early_stopping_rounds"] = 2
            if "n_estimators" in params:
                params["n_estimators"] = 2
            if "n_members" in params:
                params["n_members"] = 2
            if "crossfit_weighting" in candidate:
                settings = candidate["crossfit_weighting"]
                settings["n_splits"] = 2
                settings["teacher_params"]["max_iter"] = 3
                settings["teacher_params"]["min_samples_leaf"] = 2
        config = resolve_config(raw)

        data = synthetic_dataset()
        blocks = []
        for block in range(4):
            frame = data.train.copy(deep=True)
            frame["id"] = np.arange(32) + 32 * block + 1
            frame["Temperature"] += block / 10000
            blocks.append(frame)
        train = pd.concat(blocks, ignore_index=True)
        test = data.test.copy(deep=True)
        test["Temperature"] += .123456
        audit = {"source_sha256": {
            "train.csv": content_hash(train.to_dict(orient="records")),
            "test.csv": content_hash(test.to_dict(orient="records")),
            "sample_submission.csv": content_hash(data.sample.to_dict(orient="records")),
        }}
        groups = pd.util.hash_pandas_object(train[FEATURES], index=False).to_numpy()
        self.assertEqual(len(set(groups)), 128)
        data = replace(data, train=train, test=test, audit=audit, groups=groups)
        original = data.train.copy(deep=True)

        with redirect_stdout(io.StringIO()), threadpool_limits(limits=1):
            out = new_run("train", config, self.directory)
            result = run_train(data, config, out, "integer")
            finish_run(out, {"source_sha256": audit["source_sha256"], "result": result})
            summary = summarize_run(out)
            for key, candidate in result["candidates"].items():
                replay = run_predict(data, out / candidate["bundle"],
                                     out / f"fresh_{key}.csv", "integer")
                self.assertEqual(replay["csv_sha256"], candidate["submission"]["csv_sha256"])
                self.assertTrue(candidate["fit_and_export"]["saved_model_reload_exact_match"])
        self.assertEqual(len(summary["models"]), 7)
        self.assertTrue(summary["current_policy_eligible"])
        self.assertEqual(len(result["search"]["outer_folds"]), 2)
        self.assertTrue(all(not fold["outer_validation_used_for_selection"]
                            for fold in result["search"]["outer_folds"]))
        self.assertEqual(len(result["candidates"]), 2)
        for model in config["models"]:
            detail = json.loads((out / "models" / model["name"] / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(detail["folds"]), 2)
            for fold in detail["folds"]:
                self.assertEqual(fold["fit_rows"], 64)
                self.assertEqual(fold["validation_rows"], 64)
                if model["model_id"] == "bagged_hgb":
                    self.assertEqual(fold["bagging"]["member_actual_iterations"], [3, 3])
                if "crossfit_weighting" in model:
                    self.assertEqual(fold["crossfit_weighting"]["method"], "cross_fitted_confidence_weighting")
                    self.assertIsNone(fold["noise_weighting"])
                if model["model_id"] == "catboost":
                    self.assertEqual(fold["effective_model_parameters"]["boosting_type"], "Ordered")
                    self.assertEqual(fold["actual_iterations"], fold["selected_iterations"])
        pd.testing.assert_frame_equal(original, data.train)
        self.assertEqual(json.loads((repository / "configs/task1/sixth_run_course_only.json").read_text(encoding="utf-8")), template)


if __name__ == "__main__":
    unittest.main()
