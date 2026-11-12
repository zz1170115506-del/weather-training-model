"""Fixed-budget round-seven recipe selection on synthetic rows only."""
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
from unittest.mock import patch
import uuid

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.artifacts import content_hash, finish_run, new_run
from task1.config import resolve_config
from task1.data import FEATURES
from task1.models import fit_model
from task1.pipeline import run_predict, run_train


def selection_config():
    return {"schema_version": 1,
            "validation": {"n_splits": 2, "split_seed": 83, "model_seed": 91},
            "models": [{"name": name, "model_id": "hgb", "model_params": {"max_iter": 2}}
                       for name in ("recipe_a", "recipe_b", "recipe_c")],
            "cv_selection": {"method": "rank_fixed_recipes", "n_exports": 2},
            "blend": {"n_trials": 0, "allow_class_multipliers": False}}


class RoundSevenFlowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / "outputs"
        self.directory = self.root / ("r7_flow_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        target = self.directory.resolve()
        if target.parent != self.root.resolve() or not target.name.startswith("r7_flow_"):
            raise RuntimeError("Unsafe round-seven synthetic test cleanup.")
        shutil.rmtree(target)

    def test_selection_schema_rejects_ambiguous_modes_and_invalid_export_counts(self):
        raw = selection_config()
        for exports in (1, 2, 3):
            raw["cv_selection"]["n_exports"] = exports
            self.assertEqual(resolve_config(raw)["cv_selection"]["n_exports"], exports)
        for bad in (None, {}, {"method": "other", "n_exports": 2},
                    {"method": "rank_fixed_recipes", "n_exports": True},
                    {"method": "rank_fixed_recipes", "n_exports": 0},
                    {"method": "rank_fixed_recipes", "n_exports": -1},
                    {"method": "rank_fixed_recipes", "n_exports": 1.5},
                    {"method": "rank_fixed_recipes", "n_exports": 4},
                    {"method": "rank_fixed_recipes", "n_exports": 2, "unknown": True}):
            raw = selection_config()
            raw["cv_selection"] = bad
            with self.subTest(selection=bad), self.assertRaisesRegex(ValueError, "cv_selection"):
                resolve_config(raw)
        raw = selection_config()
        raw["search"] = {"inner_splits": 2, "blend_pool_size": 2,
                         "minimum_blend_gain": .0005, "min_blend_fold_wins": 2}
        with self.assertRaisesRegex(ValueError, "separate evaluation modes"):
            resolve_config(raw)

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "Optional LightGBM is not installed")
    def test_real_round_seven_template_has_bounded_cv_ranked_exports_and_exact_replay(self):
        repository = Path(__file__).resolve().parents[2]
        template_path = repository / "configs/task1/seventh_run_course_only.json"
        raw = json.loads(template_path.read_text(encoding="utf-8"))
        template = deepcopy(raw)
        self.assertEqual(len(raw["models"]), 4)
        self.assertEqual([m["model_id"] for m in raw["models"]].count("ovr_hgb"), 2)
        self.assertNotIn("search", raw)
        raw["validation"]["n_splits"] = 2
        if "diagnostics" in raw:
            raw["diagnostics"]["learning_curve_every"] = 2
        for candidate in raw["models"]:
            params = candidate["model_params"]
            if candidate["model_id"] == "lightgbm":
                params["n_estimators"] = 4
                params["min_child_samples"] = 2
            else:
                params["max_iter"] = 4
                params["min_samples_leaf"] = 2

        data = synthetic_dataset()
        blocks = []
        for block in range(4):
            frame = data.train.copy(deep=True)
            frame["id"] = np.arange(32) + block * 32 + 1
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
        raw["expected_source_sha256"] = deepcopy(audit["source_sha256"])
        config = resolve_config(raw)
        frozen_config = deepcopy(config)
        groups = pd.util.hash_pandas_object(train[FEATURES], index=False).to_numpy()
        data = replace(data, train=train, test=test, audit=audit, groups=groups)
        original_train = data.train.copy(deep=True)
        self.assertEqual(len(set(groups)), 128)

        with redirect_stdout(io.StringIO()), threadpool_limits(limits=1), \
                patch("task1.search.evaluate_search", side_effect=AssertionError("Nested search must not run")) as nested, \
                patch("task1.pipeline.fit_blend", side_effect=AssertionError("No learned blend in fixed-recipe ranking")) as blend, \
                patch("task1.validation.fit_model", wraps=fit_model) as cv_fit, \
                patch("task1.pipeline.fit_model", wraps=fit_model) as final_fit:
            out = new_run("train", config, self.directory)
            frozen_snapshot = (out / "resolved_config.json").read_bytes()
            result = run_train(data, config, out, "integer")
            finish_run(out, {"source_sha256": audit["source_sha256"], "result": result})
            self.assertEqual(cv_fit.call_count, 8)  # Four fixed recipes, two synthetic folds.
            self.assertEqual(final_fit.call_count, 2)
            nested.assert_not_called()
            blend.assert_not_called()
            for key, candidate in result["candidates"].items():
                replay = run_predict(data, out / candidate["bundle"],
                                     out / f"fresh_{key}.csv", "integer")
                self.assertEqual(replay["csv_sha256"], candidate["submission"]["csv_sha256"])
                self.assertTrue(candidate["fit_and_export"]["saved_model_reload_exact_match"])

        selection = json.loads((out / "cv_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(selection, result["cv_selection"])
        expected_exports = [item["name"] for item in selection["rank"][:2]]
        self.assertEqual(selection["selected_models"], expected_exports)
        self.assertEqual([item["model_name"] for item in result["candidates"].values()], expected_exports)
        self.assertEqual(len(selection["rank"]), 4)
        self.assertIn("Development CV", selection["evaluation_scope"])
        self.assertIn("no independent test", selection["evaluation_scope"])
        selected = json.loads((out / "selected_config.json").read_text(encoding="utf-8"))
        self.assertNotIn("cv_selection", selected)
        self.assertNotIn("search", selected)
        self.assertEqual([item["name"] for item in selected["models"]], expected_exports)
        for candidate in result["candidates"].values():
            destination = out / "candidates" / candidate["model_name"]
            single = json.loads((destination / "submission_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(single.get("public_source_test_overlap_used", False))
            saved = joblib.load(destination / "model_bundle.joblib")
            self.assertNotIn("cv_selection", saved["config"])
            self.assertNotIn("search", saved["config"])
        self.assertEqual((out / "resolved_config.json").read_bytes(), frozen_snapshot)
        self.assertEqual(config, frozen_config)
        self.assertEqual(json.loads(template_path.read_text(encoding="utf-8")), template)
        pd.testing.assert_frame_equal(data.train, original_train)
        self.assertNotIn("search", result)
        for excluded in ("search", "search_summary.json", "nested_summary.json", "research_blend"):
            self.assertFalse((out / excluded).exists())


if __name__ == "__main__":
    unittest.main()
