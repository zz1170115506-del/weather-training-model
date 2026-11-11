"""Synthetic checks of teacher isolation, strict weighting and unchanged inputs."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1.crossfit_weights import resolve_settings, training_weights
from task1.features import Features


def sample(repeats=1):
    i = np.arange(24)
    frame = pd.DataFrame({
        "Temperature": i.astype(float), "Humidity": 40.0 + i,
        "Wind Speed": i / 5, "Precipitation (%)": i * 2.0,
        "Atmospheric Pressure": 980.0 + i, "UV Index": i % 7,
        "Visibility (km)": i / 2, "Cloud Cover": np.where(i % 2, "a", "b"),
        "Season": np.where(i % 2, "winter", "summer"), "Location": "inland",
    })
    labels = i % 4 + 1
    if repeats > 1:
        frame = frame.iloc[np.repeat(np.arange(len(frame)), repeats)].reset_index(drop=True)
        labels = np.repeat(labels, repeats)
    return frame, labels


class CrossfitWeightTests(unittest.TestCase):
    def mock_teacher(self, fit_sets, holdout_sets):
        case = self
        class Teacher:
            classes_ = np.array([4, 3, 2, 1])
            def __init__(self, **params):
                self.params = params
                case.assertFalse(params["early_stopping"])
            def fit(self, values, y):
                self.fit_values = set(values[:, 0])
                fit_sets.append(self.fit_values)
                case.assertEqual(set(y), {1, 2, 3, 4})
                return self
            def predict_proba(self, values):
                holdout = set(values[:, 0])
                holdout_sets.append(holdout)
                case.assertFalse(self.fit_values & holdout)
                result = []
                for value in values[:, 0].astype(int):
                    label = value % 4 + 1
                    mode = value % 4
                    confidence = [.95, .85, .8, .95][mode]
                    predicted = label if mode == 3 else label % 4 + 1
                    row = np.full(4, (1.0 - confidence) / 3)
                    row[int(np.flatnonzero(self.classes_ == predicted)[0])] = confidence
                    result.append(row)
                return np.asarray(result)
        return Teacher

    def test_oof_teachers_and_preprocessing_never_see_their_holdout(self):
        X, y = sample()
        original_X, original_y = X.copy(deep=True), y.copy()
        fit_sets, holdout_sets, feature_fits = [], [], []
        original_fit = Features.fit
        def recording_fit(features, frame):
            feature_fits.append(set(frame["Temperature"]))
            return original_fit(features, frame)
        with patch("task1.crossfit_weights.HistGradientBoostingClassifier", self.mock_teacher(fit_sets, holdout_sets)), \
                patch.object(Features, "fit", recording_fit):
            weights, metadata = training_weights(X, y, {}, 83)
        self.assertEqual(feature_fits, fit_sets)
        self.assertEqual(len(fit_sets), 3)
        self.assertEqual(sum(len(s) for s in holdout_sets), len(X))
        self.assertEqual(set.union(*holdout_sets), set(X.Temperature))
        for a in range(3):
            for b in range(a):
                self.assertFalse(holdout_sets[a] & holdout_sets[b])
        # Only the high-confidence wrong mode is reduced; a threshold tie,
        # low-confidence error and high-confidence correct prediction stay 1.
        np.testing.assert_allclose(weights[np.arange(len(y)) % 4 == 0], 2 / 3)
        np.testing.assert_array_equal(weights[np.arange(len(y)) % 4 != 0], 1)
        self.assertEqual(metadata["high_confidence_conflict_rows"], 6)
        self.assertTrue(metadata["oof_coverage"]["exactly_once"])
        self.assertEqual(metadata["oof_coverage"]["minimum_predictions_per_row"], 1)
        self.assertEqual(metadata["oof_coverage"]["maximum_predictions_per_row"], 1)
        pd.testing.assert_frame_equal(X, original_X)
        np.testing.assert_array_equal(y, original_y)
        for item in metadata["folds"]:
            self.assertEqual(item["preprocessing_fitted_rows"], item["fit_rows"])
            self.assertEqual(len(item["holdout_index_sha256"]), 64)
        serialized = json.dumps(metadata, allow_nan=False)
        self.assertNotIn('"probabilities"', serialized)
        self.assertNotIn('"indices"', serialized)
        self.assertNotIn('"weights"', serialized)

    def test_duplicate_raw_records_do_not_cross_teacher_boundaries(self):
        X, y = sample(repeats=2)
        fits, holdouts = [], []
        with patch("task1.crossfit_weights.HistGradientBoostingClassifier", self.mock_teacher(fits, holdouts)):
            weights, metadata = training_weights(X, y, {}, 81)
        self.assertEqual(metadata["split_method"], "StratifiedGroupKFold")
        self.assertEqual(len(weights), len(X))
        self.assertEqual(sum(item["holdout_rows"] for item in metadata["folds"]), len(X))
        np.testing.assert_array_equal(weights[::2], weights[1::2])

    def test_deterministic_small_real_hgb_on_synthetic_records(self):
        X, y = sample(repeats=2)
        settings = {"teacher_params": {"max_iter": 3, "max_leaf_nodes": 3, "min_samples_leaf": 2}}
        before = copy.deepcopy(settings)
        with threadpool_limits(limits=1):
            first, a = training_weights(X, y, settings, 5)
            second, b = training_weights(X, y, settings, 5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(a["folds"], b["folds"])
        self.assertEqual(settings, before)
        self.assertTrue((first > 0).all() and (first <= 1).all())
        self.assertFalse(a["teacher_parameters"]["early_stopping"])

    def test_invalid_settings_and_inputs_fail_before_teacher_fit(self):
        X, y = sample()
        invalid = [None, [], {"unknown": 1}, {"n_splits": True}, {"n_splits": 1},
                   {"threshold": 1}, {"threshold": .4}, {"threshold": np.nan},
                   {"floor": 0}, {"floor": -1}, {"floor": True}, {"floor": np.inf},
                   {"teacher_params": {"early_stopping": True}},
                   {"teacher_params": {"random_state": 1}},
                   {"teacher_params": {"max_iter": 0}},
                   {"teacher_params": {"max_leaf_nodes": 1}},
                   {"teacher_params": {"learning_rate": 0}},
                   {"teacher_params": {"max_bins": 256}},
                   {"teacher_params": {"l2_regularization": -1}}]
        with patch("task1.crossfit_weights.HistGradientBoostingClassifier") as teacher:
            for settings in invalid:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    training_weights(X, y, settings, 1)
            for bad_y in (y[:-1], np.zeros_like(y), np.ones_like(y, dtype=bool), y.astype(str)):
                with self.assertRaises(ValueError):
                    training_weights(X, bad_y, {}, 1)
            for bad_seed in (-1, True, 2**32):
                with self.assertRaises(ValueError):
                    training_weights(X, y, {}, bad_seed)
            with self.assertRaises(ValueError):
                training_weights(X.drop(columns="Temperature"), y, {}, 1)
            # Four rows per class are duplicates of one feature group each.
            few_X, few_y = sample()
            few_X = few_X.iloc[:4].loc[np.repeat(np.arange(4), 4)].reset_index(drop=True)
            few_y = np.repeat(few_y[:4], 4)
            with self.assertRaises(ValueError):
                training_weights(few_X, few_y, {}, 1)
            teacher.assert_not_called()

    def test_invalid_teacher_probabilities_are_rejected(self):
        X, y = sample()
        class BadTeacher:
            classes_ = np.arange(1, 5)
            def __init__(self, **params): pass
            def fit(self, values, y): return self
            def predict_proba(self, values): return np.full((len(values), 4), np.nan)
        with patch("task1.crossfit_weights.HistGradientBoostingClassifier", BadTeacher):
            with self.assertRaisesRegex(RuntimeError, "invalid weather probabilities"):
                training_weights(X, y, {}, 1)

    def test_resolved_defaults_are_independent_and_floor_one_is_identity(self):
        one, two = resolve_settings({}), resolve_settings({})
        one["teacher_params"]["max_iter"] = 1
        self.assertEqual(two["teacher_params"]["max_iter"], 600)
        X, y = sample()
        with patch("task1.crossfit_weights.HistGradientBoostingClassifier", self.mock_teacher([], [])):
            weights, _ = training_weights(X, y, {"floor": 1}, 83)
        np.testing.assert_array_equal(weights, 1)


if __name__ == "__main__":
    unittest.main()
