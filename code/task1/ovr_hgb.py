"""Explicit weather OvR reduction and fixed probability combination.

Four independent binary log-loss models estimate P(label=k | x). Their
positive-class probabilities are aligned, clipped and row-normalized. An
optional fifth multiclass model contributes a fixed softmax probability share;
the share is configuration, never chosen using held-out/test labels here.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


_LABELS = (1, 2, 3, 4)


class OvRWeatherHGB:
    """Handwritten four-binary-model reduction using fixed boosting stages."""

    def __init__(self, *, softmax_weight=0.0, random_state=0, **params):
        if (isinstance(softmax_weight, bool) or not isinstance(softmax_weight, Real)
                or not np.isfinite(softmax_weight) or not 0 <= softmax_weight <= 1):
            raise ValueError("softmax_weight must be finite and in [0,1].")
        if (isinstance(random_state, bool) or not isinstance(random_state, Integral)
                or not 0 <= random_state < 2**32):
            raise ValueError("random_state must be an integer in [0,2**32).")
        self.softmax_weight = float(softmax_weight)
        self.random_state = int(random_state)
        self.params = {
            "loss": "log_loss", "max_iter": 1200, "learning_rate": 0.02,
            "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 15.0,
            "early_stopping": False, "categorical_features": None,
        } | dict(params)
        if self.params.get("early_stopping") is not False or self.params.get("warm_start", False) is not False:
            raise ValueError("OvR HGB uses fresh fixed-stage models without early stopping or warm starts.")
        if self.params.get("loss") != "log_loss" or self.params.get("categorical_features") is not None:
            raise ValueError("OvR HGB requires log_loss and the fold-local numeric/OHE representation.")
        # Validate unknown keyword names immediately; sklearn validates numeric
        # parameter ranges when each fresh estimator is fitted.
        HistGradientBoostingClassifier(random_state=self.random_state, **self.params)

    def get_params(self, deep=True):
        return {"softmax_weight": self.softmax_weight, "random_state": self.random_state, **self.params}

    @staticmethod
    def _values(X):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not values.shape[1] or np.isinf(values).any():
            raise ValueError("OvR HGB requires a two-dimensional numeric feature matrix without infinities.")
        return values

    def fit(self, X, y, sample_weight=None):
        values = self._values(X)
        labels = np.asarray(y)
        if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iuf"
                or not np.isfinite(labels).all() or not np.isin(labels, _LABELS).all()
                or set(labels) != set(_LABELS)):
            raise ValueError("OvR weather fit requires all four numeric labels 1..4, matching X rows.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if (sample_weight.shape != (len(labels),) or not np.isfinite(sample_weight).all()
                    or (sample_weight < 0).any() or not (sample_weight > 0).any()):
                raise ValueError("sample_weight must be a finite nonnegative vector with positive total weight.")
        self.binary_models_, diagnostics = [], []
        for index, label in enumerate(_LABELS):
            member_seed = (self.random_state + 104729 * index) % 2**32
            model = HistGradientBoostingClassifier(random_state=member_seed, **self.params)
            # The original labels are never modified; binary indicators exist
            # only inside this standard OvR decomposition.
            model.fit(values, (labels == label).astype(np.int64), sample_weight=sample_weight)
            self.binary_models_.append(model)
            diagnostics.append({"role": "binary_ovr", "positive_weather_label": label,
                "fit_rows": len(labels), "actual_iterations": int(model.n_iter_),
                "trees_per_iteration": int(model.n_trees_per_iteration_),
                "effective_model_parameters": model.get_params()})
        self.softmax_model_ = None
        if self.softmax_weight > 0:
            member_seed = (self.random_state + 524287) % 2**32
            self.softmax_model_ = HistGradientBoostingClassifier(random_state=member_seed, **self.params)
            self.softmax_model_.fit(values, labels, sample_weight=sample_weight)
            model = self.softmax_model_
            diagnostics.append({"role": "multiclass_softmax", "fit_rows": len(labels),
                "actual_iterations": int(model.n_iter_),
                "trees_per_iteration": int(model.n_trees_per_iteration_),
                "effective_model_parameters": model.get_params()})
        self.classes_ = np.asarray(_LABELS)
        self.n_features_in_ = values.shape[1]
        self.n_iter_ = max(item["actual_iterations"] for item in diagnostics)
        self.n_trees_per_iteration_ = sum(item["trees_per_iteration"] for item in diagnostics)
        self.training_diagnostics_ = {
            "strategy": "four_binary_log_loss_models_with_normalized_probabilities",
            "softmax_weight": self.softmax_weight, "binary_models": len(self.binary_models_),
            "softmax_model_fitted": self.softmax_model_ is not None,
            "member_actual_iterations": [item["actual_iterations"] for item in diagnostics],
            "total_trees": sum(item["actual_iterations"] * item["trees_per_iteration"] for item in diagnostics),
            "members": diagnostics,
            "scope": "All members fit only the supplied training partition at fixed stages. Softmax mixture weight is fixed by configuration, without evaluation/test-label selection.",
        }
        return self

    @staticmethod
    def _valid_probabilities(raw, rows, columns):
        raw = np.asarray(raw, dtype=float)
        if (raw.shape != (rows, columns) or not np.isfinite(raw).all()
                or (raw < 0).any() or (raw > 1).any()
                or not np.allclose(raw.sum(axis=1), 1.0, atol=1e-8, rtol=1e-8)):
            raise ValueError("OvR member returned invalid probabilities.")
        return raw

    def predict_proba(self, X):
        if not hasattr(self, "classes_"):
            raise RuntimeError("Fit OvR HGB before prediction.")
        values = self._values(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("OvR prediction feature count differs from fitted features.")
        if not len(values):
            return np.empty((0, len(_LABELS)))
        positive = np.empty((len(values), len(_LABELS)))
        for column, model in enumerate(self.binary_models_):
            classes = np.asarray(model.classes_)
            if len(classes) != 2 or set(classes) != {0, 1}:
                raise ValueError("OvR member must contain binary classes 0 and 1.")
            raw = self._valid_probabilities(model.predict_proba(values), len(values), 2)
            positive[:, column] = raw[:, int(np.flatnonzero(classes == 1)[0])]
        epsilon = np.finfo(np.float64).eps
        positive = np.clip(positive, epsilon, 1.0)
        result = positive / positive.sum(axis=1, keepdims=True)
        if self.softmax_model_ is not None:
            classes = np.asarray(self.softmax_model_.classes_)
            if len(classes) != 4 or set(classes) != set(_LABELS):
                raise ValueError("Softmax member must contain all four weather classes.")
            raw = self._valid_probabilities(self.softmax_model_.predict_proba(values), len(values), 4)
            softmax = raw[:, [int(np.flatnonzero(classes == label)[0]) for label in _LABELS]]
            result = (1 - self.softmax_weight) * result + self.softmax_weight * softmax
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
