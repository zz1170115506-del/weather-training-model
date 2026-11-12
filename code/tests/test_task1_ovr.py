"""Synthetic OvR reduction, fixed probability mixing and LightGBM adapters."""
import importlib.util
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.models import fit_model, make_model, predict_probabilities
from task1.ovr_hgb import OvRWeatherHGB


class OvRReductionTests(unittest.TestCase):
    def test_four_binary_targets_positive_column_and_fixed_softmax_mix_by_hand(self):
        y = np.tile([1, 2, 3, 4], 3)
        before = y.copy()
        X = np.arange(len(y) * 2).reshape(len(y), 2)
        weights = np.linspace(.5, 1, len(y))
        fitted = []
        positive = [.2, .4, .6, .8]
        class Teacher:
            def __init__(self, **params): self.params = params
            def fit(self, values, target, sample_weight=None):
                self.target = np.asarray(target).copy()
                self.weights = sample_weight.copy()
                self.n_iter_ = self.params["max_iter"]
                self.role = len(fitted)
                self.classes_ = np.array([1, 0]) if self.role < 4 else np.array([4, 1, 3, 2])
                self.n_trees_per_iteration_ = 1 if self.role < 4 else 4
                fitted.append(self)
                return self
            def get_params(self): return dict(self.params)
            def predict_proba(self, values):
                if self.role < 4:
                    p = positive[self.role]
                    return np.tile([p, 1-p], (len(values), 1))
                return np.tile([.4, .1, .3, .2], (len(values), 1))
        with patch("task1.ovr_hgb.HistGradientBoostingClassifier", Teacher):
            model = OvRWeatherHGB(softmax_weight=.5, random_state=7, max_iter=2).fit(X, y, weights)
        self.assertEqual(len(fitted), 5)
        for column, member in enumerate(fitted[:4], 1):
            np.testing.assert_array_equal(member.target, (y == column).astype(int))
            np.testing.assert_array_equal(member.weights, weights)
            self.assertFalse(member.params["early_stopping"])
        np.testing.assert_array_equal(fitted[4].target, y)
        np.testing.assert_array_equal(y, before)
        expected = .5 * np.array(positive) / sum(positive) + .5 * np.array([.1, .2, .3, .4])
        np.testing.assert_allclose(model.predict_proba(X), np.tile(expected, (len(X), 1)))
        np.testing.assert_array_equal(model.classes_, [1, 2, 3, 4])
        self.assertEqual(model.n_trees_per_iteration_, 8)
        self.assertEqual(model.training_diagnostics_["total_trees"], 16)
        self.assertEqual(model.training_diagnostics_["member_actual_iterations"], [2] * 5)

    def test_pure_ovr_clips_zero_positive_probabilities_without_softmax_fit(self):
        fitted = []
        class ZeroTeacher:
            classes_ = np.array([0, 1])
            n_iter_ = 1
            n_trees_per_iteration_ = 1
            def __init__(self, **params): self.params = params
            def fit(self, X, y, **kwargs): fitted.append(self); return self
            def get_params(self): return dict(self.params)
            def predict_proba(self, X): return np.tile([1., 0.], (len(X), 1))
        with patch("task1.ovr_hgb.HistGradientBoostingClassifier", ZeroTeacher):
            model = OvRWeatherHGB(max_iter=1).fit(np.ones((8, 2)), np.tile([1, 2, 3, 4], 2))
        self.assertEqual(len(fitted), 4)
        self.assertIsNone(model.softmax_model_)
        np.testing.assert_allclose(model.predict_proba(np.zeros((2, 2))), .25)
        self.assertEqual(model.predict_proba(np.zeros((0, 2))).shape, (0, 4))
        self.assertEqual(model.n_trees_per_iteration_, 4)

    def test_tiny_real_models_have_stable_serialization_and_full_parameter_metadata(self):
        data = synthetic_dataset()
        configuration = {"name": "tiny_ovr", "model_id": "ovr_hgb", "model_params": {
            "max_iter": 3, "learning_rate": .1, "max_leaf_nodes": 3,
            "min_samples_leaf": 2, "softmax_weight": .5}}
        original = data.train.copy(deep=True)
        with threadpool_limits(limits=1):
            bundle, metadata = fit_model(data.X, data.y, configuration, 13)
            probability = predict_probabilities(bundle, data.test)
            stream = io.BytesIO()
            joblib.dump(bundle, stream)
            stream.seek(0)
            loaded = joblib.load(stream)
            np.testing.assert_array_equal(probability, predict_probabilities(loaded, data.test))
        pd.testing.assert_frame_equal(data.train, original)
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertEqual(metadata["actual_iterations"], 3)
        self.assertEqual(metadata["trees_per_iteration"], 8)
        self.assertEqual(metadata["ovr"]["total_trees"], 24)
        self.assertEqual(metadata["preprocessing_fitted_rows"], len(data.y))
        self.assertFalse(metadata["early_stopping_uses_evaluation_fold"])
        self.assertEqual(len(metadata["ovr"]["members"]), 5)
        for member in metadata["ovr"]["members"]:
            self.assertFalse(member["effective_model_parameters"]["early_stopping"])
        self.assertEqual(bundle["model"].get_params()["softmax_weight"], .5)

    def test_invalid_reduction_parameters_targets_and_predictions_fail_clearly(self):
        bad = [{"softmax_weight": -.1}, {"softmax_weight": 1.1}, {"softmax_weight": True},
               {"softmax_weight": np.nan}, {"random_state": -1}, {"random_state": True},
               {"early_stopping": True}, {"warm_start": True}, {"loss": "wrong"}]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ValueError):
                OvRWeatherHGB(**params)
        model = OvRWeatherHGB(max_iter=1)
        X, y = np.ones((8, 2)), np.tile([1, 2, 3, 4], 2)
        with self.assertRaises(RuntimeError): model.predict_proba(X)
        for invalid_y in [y[:-1], np.zeros(8), y.astype(str), np.ones(8, dtype=bool)]:
            with self.assertRaises(ValueError): model.fit(X, invalid_y)
        for invalid_weights in [np.zeros(8), np.full(8, np.nan), np.full(8, -1), np.ones(7)]:
            with self.assertRaises(ValueError): model.fit(X, y, invalid_weights)
        with self.assertRaisesRegex(ValueError, "fixed member stages"):
            make_model({"model_id": "ovr_hgb"}, 1, iterations=2)


class LightGBMAdapterTests(unittest.TestCase):
    def fake_module(self):
        class Booster:
            def __init__(self, model):
                self.params = model.params.copy()
                self.pandas_categorical = [model.fit_frame[c].cat.categories.tolist()
                    for c in model.fit_frame.select_dtypes(include="category")]
                self.iterations = model.n_iter_
            def num_model_per_iteration(self): return 4
            def num_trees(self): return 4 * self.iterations
        class FakeLGBM:
            def __init__(self, **params): self.params = params
            def fit(self, X, y, **kwargs):
                self.fit_frame = X.copy(deep=True)
                self.classes_ = np.array([4, 2, 1, 3])
                self.n_iter_ = self.params["n_estimators"]
                self.booster_ = Booster(self)
                return self
            def get_params(self): return dict(self.params)
            def predict_proba(self, X):
                self.last_frame = X.copy(deep=True)
                return np.tile([.1, .2, .3, .4], (len(X), 1))
        return types.SimpleNamespace(LGBMClassifier=FakeLGBM)

    def test_lightgbm_uses_fold_fitted_native_categories_and_weather_class_alignment(self):
        data = synthetic_dataset()
        config = {"name": "fake_lightgbm", "model_id": "lightgbm", "model_params": {"n_estimators": 2}}
        with patch.dict(sys.modules, {"lightgbm": self.fake_module()}):
            bundle, metadata = fit_model(data.X, data.y, config, 19)
            test = data.test.copy(deep=True)
            test.loc[0, "Cloud Cover"] = "never-seen-category"
            probability = predict_probabilities(bundle, test)
        self.assertEqual(bundle["features"].native_categories, "hgb")
        model = bundle["model"]
        self.assertEqual(len(model.fit_frame.select_dtypes(include="category").columns), 3)
        self.assertTrue(pd.isna(model.last_frame.loc[0, "Cloud Cover"]))
        np.testing.assert_allclose(probability, np.tile([.3, .2, .4, .1], (len(test), 1)))
        self.assertEqual(metadata["actual_iterations"], 2)
        self.assertEqual(metadata["trees_per_iteration"], 4)
        self.assertEqual(metadata["lightgbm"]["actual_trees"], 8)
        self.assertEqual(metadata["effective_model_parameters"]["n_jobs"], 1)
        self.assertTrue(metadata["effective_model_parameters"]["deterministic"])

    def test_lightgbm_overrides_cannot_bypass_cpu_stages_or_multiclass_probability_contract(self):
        invalid = [{"num_threads": 8}, {"seed": 2}, {"early_stopping_rounds": 10},
                   {"callbacks": []}, {"init_model": "anything"}, {"n_jobs": 2},
                   {"device_type": "gpu"}, {"deterministic": False}, {"force_col_wise": False},
                   {"objective": "multiclassova"}, {"num_class": 3}, {"boosting_type": "dart"}]
        with patch.dict(sys.modules, {"lightgbm": self.fake_module()}):
            for params in invalid:
                with self.subTest(params=params), self.assertRaises(ValueError):
                    make_model({"model_id": "lightgbm", "model_params": params}, 1)
            with self.assertRaisesRegex(ValueError, "fixed boosting stages"):
                make_model({"model_id": "lightgbm"}, 1, iterations=2)

    def test_missing_optional_library_explains_which_requirements_to_install(self):
        with patch.dict(sys.modules, {"lightgbm": None}):
            with self.assertRaisesRegex(ImportError, "requirements-round7.txt"):
                make_model({"model_id": "lightgbm"}, 1)

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "optional LightGBM not installed")
    def test_real_tiny_lightgbm_preserves_actual_iterations_and_reloads_exactly(self):
        data = synthetic_dataset()
        config = {"name": "tiny_lightgbm", "model_id": "lightgbm", "model_params": {
            "n_estimators": 3, "min_child_samples": 2, "num_leaves": 3, "min_data_per_group": 1}}
        with threadpool_limits(limits=1):
            bundle, metadata = fit_model(data.X, data.y, config, 23)
            predicted = predict_probabilities(bundle, data.test)
            stream = io.BytesIO()
            joblib.dump(bundle, stream)
            stream.seek(0)
            np.testing.assert_array_equal(predicted, predict_probabilities(joblib.load(stream), data.test))
        self.assertTrue(1 <= metadata["actual_iterations"] <= 3)
        self.assertEqual(metadata["lightgbm"]["actual_trees"], 4 * metadata["actual_iterations"])
        np.testing.assert_allclose(predicted.sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
