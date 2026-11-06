"""Training-only iteration selection and weather-label XGBoost adaptation."""
import importlib.util
import io
import json
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
from task1.features import Features
from task1.models import (WeatherXGBoost, _inner_split, fit_model, make_model,
                          predict_probabilities)


class InnerStoppingTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_dataset()
        self.config = {"name": "tiny_hgb", "model_id": "hgb", "model_params": {
            "max_iter": 8, "min_samples_leaf": 2, "max_leaf_nodes": 3}, "early_stopping_rounds": 2}

    def test_hgb_uses_best_inner_stage_then_refits_all_rows_without_outer_labels(self):
        class Selector:
            n_iter_ = 4
            validation_score_ = [-1.0, -0.5, -0.4, -0.41, -0.42]
            def set_params(self, **params):
                self.params = params
            def fit(self, X, y, **kwargs):
                self.fit_rows = len(X)
                self.stop_rows = len(kwargs["X_val"])
                self.stop_labels = kwargs["y_val"].copy()
        class Final:
            n_iter_ = 2
            n_trees_per_iteration_ = 4
            def fit(self, X, y):
                self.fit_rows = len(X)
            def get_params(self):
                return {"max_iter": 2, "early_stopping": False}
        selector, final = Selector(), Final()
        with patch("task1.models.make_model", side_effect=[selector, final]) as factory:
            bundle, metadata = fit_model(self.data.X, self.data.y, self.config, 83)
        self.assertEqual(metadata["selected_iterations"], 2)
        self.assertEqual(metadata["actual_iterations"], 2)
        self.assertEqual(metadata["iteration_selection"]["inner_stopped_iterations"], 4)
        self.assertEqual(factory.call_args_list[-1].args[-1], 2)
        self.assertEqual(final.fit_rows, len(self.data.y))
        self.assertEqual(selector.fit_rows + selector.stop_rows, len(self.data.y))
        self.assertEqual(bundle["features"].fitted_rows_, len(self.data.y))
        self.assertFalse(metadata["early_stopping_uses_evaluation_fold"])

    def test_preprocessing_and_weights_fit_inner_training_before_full_refit(self):
        expected_fit, expected_stop, _ = _inner_split(self.data.X, self.data.y, 83)
        fits, weights = [], []
        original = Features.fit
        def recording_fit(features, frame):
            fits.append(set(frame.index))
            return original(features, frame)
        def recording_weights(frame, y, config):
            weights.append(set(frame.index))
            return np.ones(len(frame)), None
        config = dict(self.config, noise_weighting={"floor": 0.2})
        with threadpool_limits(limits=1), patch.object(Features, "fit", recording_fit), \
                patch("task1.models._training_weights", side_effect=recording_weights):
            bundle, metadata = fit_model(self.data.X, self.data.y, config, 83)
        fit_indices = set(self.data.X.index[expected_fit])
        stop_indices = set(self.data.X.index[expected_stop])
        self.assertEqual(fits[0], fit_indices)
        self.assertFalse(fits[0] & stop_indices)
        self.assertEqual(weights, [fit_indices, set(self.data.X.index)])
        self.assertEqual(fits[-1], set(self.data.X.index))
        self.assertEqual(bundle["model"].n_iter_, metadata["selected_iterations"])
        self.assertFalse(bundle["model"].early_stopping)

    def test_duplicate_features_do_not_cross_internal_stopping_boundary(self):
        X = pd.concat([self.data.X, self.data.X], ignore_index=True)
        y = np.tile(self.data.y, 2)
        fit, stop, protocol = _inner_split(X, y, 83)
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        self.assertFalse(set(groups[fit]) & set(groups[stop]))
        self.assertEqual(set(y[fit]), {1, 2, 3, 4})
        self.assertEqual(set(y[stop]), {1, 2, 3, 4})
        self.assertEqual(protocol["method"], "StratifiedGroupKFold")

    def test_group_holdout_fails_instead_of_leaking_single_group_per_class(self):
        X = pd.concat([self.data.X.iloc[[0, 8, 16, 24]]] * 2, ignore_index=True)
        y = np.tile([1, 2, 3, 4], 2)
        with self.assertRaisesRegex(ValueError, "independent feature groups"):
            _inner_split(X, y, 83)

    def test_stopping_options_cannot_bypass_inner_isolation(self):
        for patience in (True, 0, -1, 1.5):
            config = dict(self.config, early_stopping_rounds=patience)
            with self.subTest(patience=patience), self.assertRaises(ValueError):
                fit_model(self.data.X, self.data.y, config, 83)
        for key in ("early_stopping_rounds", "callbacks", "eval_set", "xgb_model"):
            config = {"model_id": "xgboost", "model_params": {key: 2}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_model(config, 83)
        with self.assertRaises(ValueError):
            make_model({"model_id": "hgb", "model_params": {"early_stopping": True}}, 83)

    def test_native_hgb_selection_retains_training_only_category_transform(self):
        config = dict(self.config, model_id="hgb_native")
        with threadpool_limits(limits=1):
            bundle, metadata = fit_model(self.data.X, self.data.y, config, 83)
            probabilities = predict_probabilities(bundle, self.data.test)
        self.assertEqual(bundle["features"].native_categories, "hgb")
        self.assertEqual(metadata["actual_iterations"], metadata["selected_iterations"])
        self.assertFalse(bundle["model"].early_stopping)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    @unittest.skipUnless(importlib.util.find_spec("catboost"), "Optional CatBoost dependency is not installed")
    def test_tiny_catboost_keeps_inner_best_refit_and_actual_iteration_metadata(self):
        config = {"name": "tiny_catboost", "model_id": "catboost", "model_params": {
            "iterations": 8, "depth": 2, "learning_rate": 0.1}, "early_stopping_rounds": 2}
        with threadpool_limits(limits=1):
            bundle, metadata = fit_model(self.data.X, self.data.y, config, 83)
            probabilities = predict_probabilities(bundle, self.data.test)
        self.assertEqual(metadata["actual_iterations"], metadata["selected_iterations"])
        self.assertGreaterEqual(metadata["iteration_selection"]["inner_stopped_iterations"],
                                metadata["selected_iterations"])
        self.assertEqual(bundle["features"].fitted_rows_, len(self.data.y))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)


class XGBoostAdapterTests(unittest.TestCase):
    def test_weather_labels_encode_only_at_library_boundary(self):
        class Booster:
            def num_boosted_rounds(self):
                return 3
            def save_config(self):
                return '{"learner":{"objective":"multi:softprob"}}'
        class FakeClassifier:
            best_iteration = 1
            def __init__(self, **params):
                self.params = params
            def fit(self, X, y, **kwargs):
                self.fit_y = y
                self.eval_y = kwargs["eval_set"][0][1]
                return self
            def get_booster(self):
                return Booster()
            def predict_proba(self, X):
                return np.tile([0.1, 0.2, 0.3, 0.4], (len(X), 1))
        with patch.dict(sys.modules, {"xgboost": types.SimpleNamespace(XGBClassifier=FakeClassifier)}):
            model = WeatherXGBoost(n_estimators=5, objective="multi:softprob", num_class=4)
            model.fit(np.zeros((4, 2)), [1, 2, 3, 4],
                      eval_set=[(np.zeros((2, 2)), [4, 2])], early_stopping_rounds=2)
        np.testing.assert_array_equal(model.model_.fit_y, [0, 1, 2, 3])
        np.testing.assert_array_equal(model.model_.eval_y, [3, 1])
        np.testing.assert_array_equal(model.classes_, [1, 2, 3, 4])
        self.assertEqual(model.best_iteration_, 1)
        self.assertEqual(model.n_iter_, 3)
        self.assertEqual(model.model_.params["early_stopping_rounds"], 2)
        self.assertNotIn("early_stopping_rounds", model.get_params())
        self.assertEqual(model.predict_proba(np.zeros((1, 2))).shape, (1, 4))

    def test_adapter_rejects_invalid_labels_and_unfitted_inference(self):
        for labels in ([0, 1, 2, 3], [1, 2, 3, 4.5], [True, False], ["1", "2"], [1, 2, np.nan, 4]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                WeatherXGBoost._encode(labels)
        with self.assertRaises(ValueError):
            WeatherXGBoost._encode([1, 2], require_all=True)
        with self.assertRaises(RuntimeError):
            WeatherXGBoost().predict_proba(np.zeros((1, 2)))

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "Optional XGBoost dependency is not installed")
    def test_real_tiny_xgboost_inner_stop_full_refit_and_joblib_replay(self):
        data = synthetic_dataset()
        config = {"name": "tiny_xgb", "model_id": "xgboost", "model_params": {
            "n_estimators": 12, "max_depth": 2, "learning_rate": 0.15}, "early_stopping_rounds": 2}
        with threadpool_limits(limits=1):
            bundle, metadata = fit_model(data.X, data.y, config, 83)
            before = predict_probabilities(bundle, data.test)
            storage = io.BytesIO()
            joblib.dump(bundle, storage)
            storage.seek(0)
            after = predict_probabilities(joblib.load(storage), data.test)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(before.shape, (4, 4))
        np.testing.assert_allclose(before.sum(axis=1), 1)
        self.assertEqual(metadata["actual_iterations"], metadata["selected_iterations"])
        self.assertEqual(bundle["features"].fitted_rows_, 32)
        self.assertEqual(bundle["model"].get_params()["n_estimators"], metadata["selected_iterations"])
        self.assertNotIn("early_stopping_rounds", bundle["model"].get_params())
        self.assertIn("learner", metadata["xgboost_booster_configuration"])
        json.dumps(metadata, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
