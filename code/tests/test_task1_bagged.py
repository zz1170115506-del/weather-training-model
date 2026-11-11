"""Synthetic-only tests for stratified bagging, isolation and replay."""
import io
import json
import os
from hashlib import sha256
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.bagged_hgb import BaggedWeatherHGB
from task1.models import fit_model, make_model, predict_probabilities, _training_weights


class BaggedHGBTests(unittest.TestCase):
    def setUp(self):
        self.X = np.column_stack([np.arange(32), np.sin(np.arange(32))])
        self.y = np.tile([1, 2, 3, 4], 8)
        self.options = {"n_members": 3, "max_samples": 0.5, "random_state": 73,
                        "max_iter": 5, "min_samples_leaf": 2, "max_leaf_nodes": 3}

    def test_members_sample_only_supplied_rows_and_weights_follow_duplicate_indices(self):
        captured = []
        original = HistGradientBoostingClassifier.fit
        weights = 0.2 + np.arange(32) / 32

        def recording_fit(model, X, y, **kwargs):
            captured.append((X.copy(), np.asarray(y).copy(), kwargs["sample_weight"].copy()))
            return original(model, X, y, **kwargs)

        with threadpool_limits(limits=1), patch.object(HistGradientBoostingClassifier, "fit", recording_fit):
            model = BaggedWeatherHGB(**self.options).fit(self.X, self.y, sample_weight=weights)
        self.assertEqual(len(captured), 3)
        for (X, y, sampled_weights), detail in zip(captured, model.member_diagnostics_):
            indices = X[:, 0].astype(int)
            self.assertTrue(set(indices).issubset(set(range(32))))
            self.assertEqual(len(indices), 16)
            self.assertEqual(set(y), {1, 2, 3, 4})
            np.testing.assert_array_equal(y, self.y[indices])
            np.testing.assert_array_equal(sampled_weights, weights[indices])
            self.assertEqual(detail["class_counts"], {"1": 4, "2": 4, "3": 4, "4": 4})
            self.assertEqual(detail["sampling_indices_sha256"],
                             sha256(indices.astype("<i8").tobytes()).hexdigest())
            self.assertEqual(detail["actual_iterations"], 5)
            self.assertFalse(detail["effective_model_parameters"]["early_stopping"])
        hashes = [entry["sampling_indices_sha256"] for entry in model.member_diagnostics_]
        self.assertEqual(len(set(hashes)), 3)
        self.assertEqual(model.training_diagnostics_["total_boosting_stages"], 15)
        json.dumps(model.training_diagnostics_, allow_nan=False)

    def test_fixed_seed_and_joblib_replay_are_exact(self):
        with threadpool_limits(limits=1):
            first = BaggedWeatherHGB(**self.options).fit(self.X, self.y)
            second = BaggedWeatherHGB(**self.options).fit(self.X, self.y)
            expected = first.predict_proba(self.X[:4])
            np.testing.assert_array_equal(expected, second.predict_proba(self.X[:4]))
            self.assertEqual(first.member_diagnostics_, second.member_diagnostics_)
            stream = io.BytesIO()
            joblib.dump(first, stream)
            stream.seek(0)
            np.testing.assert_array_equal(expected, joblib.load(stream).predict_proba(self.X[:4]))
        np.testing.assert_allclose(expected.sum(axis=1), 1)
        self.assertEqual(first.get_params()["max_iter"], 5)

    def test_sampling_without_replacement_and_tiny_classes_keep_all_labels(self):
        options = dict(self.options, bootstrap=False)
        with threadpool_limits(limits=1):
            model = BaggedWeatherHGB(**options).fit(self.X, self.y)
            tiny = BaggedWeatherHGB(**options).fit(self.X[:4], self.y[:4])
        for detail in model.member_diagnostics_:
            self.assertEqual(detail["sampled_rows"], detail["unique_sampled_rows"])
        self.assertTrue(all(detail["sampled_rows"] == 4 for detail in tiny.member_diagnostics_))
        self.assertTrue(all(detail["class_counts"] == {"1": 1, "2": 1, "3": 1, "4": 1}
                            for detail in tiny.member_diagnostics_))

    def test_probability_mean_aligns_member_class_order(self):
        class FakeMember:
            def __init__(self, classes, probabilities):
                self.classes_ = np.asarray(classes)
                self.probabilities = probabilities
            def predict_proba(self, X):
                return np.tile(self.probabilities, (len(X), 1))

        model = BaggedWeatherHGB(**self.options)
        model.n_features_in_ = 2
        model.classes_ = np.asarray([1, 2, 3, 4])
        model.members_ = [FakeMember([1, 2, 3, 4], [.1, .2, .3, .4]),
                          FakeMember([4, 2, 1, 3], [.1, .1, .4, .4])]
        expected = np.array([[.25, .15, .35, .25]])
        np.testing.assert_allclose(model.predict_proba(self.X[:1]), expected)
        np.testing.assert_array_equal(model.predict(self.X[:1]), [3])

    def test_invalid_parameters_training_inputs_and_unfitted_inference_are_rejected(self):
        for change in ({"n_members": True}, {"n_members": 0}, {"n_members": 1.5},
                       {"max_samples": 0}, {"max_samples": 1.01}, {"max_samples": True},
                       {"max_samples": np.nan}, {"bootstrap": "yes"}, {"random_state": -1},
                       {"early_stopping": True}, {"early_stopping": "auto"},
                       {"warm_start": True}, {"categorical_features": "from_dtype"},
                       {"class_weight": "balanced"}, {"max_iter": 0}):
            with self.subTest(change=change), self.assertRaises((TypeError, ValueError)):
                BaggedWeatherHGB(**(self.options | change))
        for X, y in [(self.X, self.y[:-1]), (self.X[:-1], self.y),
                     (self.X[:3], self.y[:3]), (self.X[:4], [True] * 4),
                     (np.array([[np.nan, 1]] * 4), self.y[:4])]:
            with self.subTest(shape=np.shape(X)), self.assertRaises(ValueError):
                BaggedWeatherHGB(**self.options).fit(X, y)
        for weights in ([1] * 31, [0] * 32, [-1] * 32, [np.nan] * 32):
            with self.subTest(weights=weights[:1]), self.assertRaises(ValueError):
                BaggedWeatherHGB(**self.options).fit(self.X, self.y, sample_weight=weights)
        with self.assertRaises(RuntimeError):
            BaggedWeatherHGB(**self.options).predict_proba(self.X)

    def test_pipeline_metadata_records_members_not_a_fictitious_epoch_count(self):
        data = synthetic_dataset()
        config = {"name": "tiny_bagged", "model_id": "bagged_hgb", "model_params": {
            "n_members": 2, "max_samples": .7, "max_iter": 4,
            "min_samples_leaf": 2, "max_leaf_nodes": 3}}
        with threadpool_limits(limits=1):
            bundle, metadata = fit_model(data.X, data.y, config, 83)
            probabilities = predict_probabilities(bundle, data.test)
        self.assertEqual(bundle["features"].fitted_rows_, 32)
        self.assertEqual(metadata["bagging"]["member_count"], 2)
        self.assertEqual(metadata["bagging"]["member_actual_iterations"], [4, 4])
        self.assertEqual(metadata["bagging"]["base_learner_fit_count"], 2)
        self.assertIsNone(metadata["actual_iterations"])
        self.assertIsNone(metadata["selected_iterations"])
        self.assertFalse(metadata["early_stopping_uses_evaluation_fold"])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        json.dumps(metadata, allow_nan=False)
        with self.assertRaises(ValueError):
            make_model(config, 83, iterations=3)

    def test_weighting_methods_cannot_be_combined(self):
        config = {"model_id": "hgb", "noise_weighting": {"floor": .2},
                  "crossfit_weighting": {"floor": .5}}
        with self.assertRaisesRegex(ValueError, "not both"):
            _training_weights(None, None, config, 83)

    def test_crossfit_pipeline_records_separate_diagnostics_and_preserves_inputs(self):
        data = synthetic_dataset()
        X, y = data.X.copy(deep=True), data.y.copy()
        config = {"name": "tiny_weighted", "model_id": "hgb", "model_params": {
            "max_iter": 4, "min_samples_leaf": 2, "max_leaf_nodes": 3},
            "crossfit_weighting": {"n_splits": 2, "floor": .5, "threshold": .85,
                "teacher_params": {"max_iter": 3, "min_samples_leaf": 2, "max_leaf_nodes": 3}}}
        with threadpool_limits(limits=1):
            bundle, metadata = fit_model(data.X, data.y, config, 83)
            probabilities = predict_probabilities(bundle, data.test)
        self.assertIsNone(metadata["noise_weighting"])
        diagnostic = metadata["crossfit_weighting"]
        self.assertEqual(diagnostic["method"], "cross_fitted_confidence_weighting")
        self.assertEqual(len(diagnostic["folds"]), 2)
        self.assertTrue(all(fold["fit_rows"] + fold["holdout_rows"] == 32
                            for fold in diagnostic["folds"]))
        self.assertTrue(data.X.equals(X))
        np.testing.assert_array_equal(data.y, y)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        json.dumps(metadata, allow_nan=False)

    def test_empty_crossfit_settings_use_defaults_and_receive_the_model_seed(self):
        config = {"model_id": "hgb", "crossfit_weighting": {}}
        expected = (np.ones(4), {"method": "cross_fitted_confidence_weighting"})
        with patch("task1.crossfit_weights.training_weights", return_value=expected) as factory:
            actual = _training_weights("fit-frame", "fit-labels", config, 713)
        self.assertIs(actual, expected)
        self.assertEqual(factory.call_args.args, ("fit-frame", "fit-labels", {}, 713))
        self.assertEqual(factory.call_args.kwargs, {"feature_options": None})


if __name__ == "__main__":
    unittest.main()
