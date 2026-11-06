"""Learning curves must observe fitted stages without changing predictions."""
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
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.diagnostics import boosting_curve
from task1.models import fit_model, predict_probabilities
from task1.validation import cross_validate, make_folds, metrics


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_dataset()
        self.config = {"name": "tiny_hgb", "model_id": "hgb", "model_params": {
            "max_iter": 4, "learning_rate": 0.1, "min_samples_leaf": 2, "max_leaf_nodes": 3}}

    def test_curve_endpoint_matches_retained_model_and_never_refits(self):
        with threadpool_limits(limits=1):
            bundle, metadata = fit_model(self.data.X, self.data.y, self.config, 83)
            before = predict_probabilities(bundle, self.data.X)
            with patch.object(bundle["model"], "fit", side_effect=AssertionError("diagnostics refitted model")), \
                    patch.object(bundle["features"], "fit", side_effect=AssertionError("diagnostics refitted features")):
                curve = boosting_curve(bundle, self.data.X, self.data.y, self.data.X, self.data.y, every=3)
            after = predict_probabilities(bundle, self.data.X)
        self.assertEqual([row["iteration"] for row in curve], [1, 3, 4])
        self.assertEqual(metadata["actual_iterations"], 4)
        self.assertEqual(metadata["trees_per_iteration"], 4)
        self.assertAlmostEqual(curve[-1]["validation_macro_f1"], metrics(self.data.y, before)["macro_f1"])
        np.testing.assert_array_equal(before, after)
        self.assertTrue(all(np.isfinite(value) for row in curve for value in row.values()))

    def test_rejects_invalid_interval_and_staged_count_disagreement(self):
        with threadpool_limits(limits=1):
            bundle, _ = fit_model(self.data.X, self.data.y, self.config, 83)
            for every in (0, -1, True, 1.5):
                with self.subTest(every=every), self.assertRaises(ValueError):
                    boosting_curve(bundle, self.data.X, self.data.y, self.data.X, self.data.y, every)
            with patch.object(bundle["model"], "staged_predict_proba", return_value=iter(())), self.assertRaises(RuntimeError):
                boosting_curve(bundle, self.data.X, self.data.y, self.data.X, self.data.y)

    def test_common_fold_workflow_saves_aggregate_curves_and_exact_final_stage(self):
        root = Path(__file__).resolve().parents[2] / "outputs"
        directory = root / ("curve_test_" + uuid.uuid4().hex)
        def cleanup():
            target = directory.resolve()
            if target.parent != root.resolve() or not target.name.startswith("curve_test_"):
                raise RuntimeError("Unsafe learning-curve test cleanup.")
            if target.exists():
                shutil.rmtree(target)
        self.addCleanup(cleanup)
        folds, _ = make_folds(self.data.y, self.data.train.id, self.data.groups, n_splits=2, seed=83)
        with threadpool_limits(limits=1):
            summary, probabilities = cross_validate(self.data, self.config, folds, 83,
                                                    out=directory, verbose=False, learning_curve_every=3)
        frame = pd.read_csv(directory / "learning_curve.csv")
        self.assertEqual(set(frame.columns) & {"id", "label", "prediction"}, set())
        self.assertEqual(len(frame), 6)
        for fold in range(2):
            endpoint = frame[(frame.fold == fold) & (frame.iteration == 4)].iloc[0]
            self.assertAlmostEqual(endpoint.validation_macro_f1,
                                   metrics(self.data.y[folds == fold], probabilities[folds == fold])["macro_f1"])
        saved = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["learning_curve"], summary["learning_curve"])
        self.assertIn("do not select", saved["learning_curve"]["scope"])
        self.assertTrue((directory / "learning_curve.png").is_file())


if __name__ == "__main__":
    unittest.main()
