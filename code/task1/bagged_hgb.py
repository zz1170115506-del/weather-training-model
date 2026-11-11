"""Stratified bagging of fixed-stage HGB models using supplied fit rows only.

The sampler and probability reduction are implemented here. The base learner
is scikit-learn HGB. Per-class sample sizes are rounded up independently, so
every member retains all four weather classes, including on small fit sets.
No validation data, external data, or saved predictions enter this estimator.
"""
from __future__ import annotations

from hashlib import sha256
from numbers import Integral, Real

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import LABELS


class BaggedWeatherHGB:
    """Average independently sampled, independently seeded HGB probabilities.

    ``max_samples`` is a fraction in (0, 1]. ``bootstrap=True`` samples with
    replacement; otherwise each class is sampled without replacement. Strictly
    positive sample weights are supported and indexed with the sampled rows.
    A member always trains for its configured number of boosting stages.
    """

    def __init__(self, n_members=5, max_samples=0.85, bootstrap=True,
                 random_state=None, **hgb_params):
        self.n_members = n_members
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.hgb_params = dict(hgb_params)
        self._validate_parameters()

    def _validate_parameters(self):
        if (isinstance(self.n_members, bool) or not isinstance(self.n_members, Integral)
                or self.n_members < 1):
            raise ValueError("n_members must be a positive integer.")
        if (isinstance(self.max_samples, bool) or not isinstance(self.max_samples, Real)
                or not np.isfinite(self.max_samples) or not 0 < self.max_samples <= 1):
            raise ValueError("max_samples must be a fraction in (0, 1].")
        if not isinstance(self.bootstrap, bool):
            raise ValueError("bootstrap must be a boolean.")
        if self.random_state is not None and (
                isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral)
                or not 0 <= self.random_state < 2**32):
            raise ValueError("random_state must be None or an integer in [0, 2**32).")
        if self.hgb_params.get("early_stopping", False) is not False:
            raise ValueError("Bagged HGB uses fixed stages; early_stopping must be False.")
        if self.hgb_params.get("categorical_features") is not None:
            raise ValueError("Bagged HGB expects the pipeline's numeric/one-hot features.")
        if self.hgb_params.get("warm_start", False) is not False:
            raise ValueError("Bagged HGB members must start fresh; warm_start must be False.")
        if self.hgb_params.get("class_weight") is not None:
            raise ValueError("Bagged HGB uses stratified sampling; class_weight is not supported.")
        # Validate base parameter names and numerical constraints without fitting.
        learner = HistGradientBoostingClassifier(**self._member_parameters(0))
        learner._validate_params()

    def _member_parameters(self, seed):
        return {"max_iter": 1200, "learning_rate": 0.02, "max_leaf_nodes": 15,
                "min_samples_leaf": 40, "l2_regularization": 15.0,
                "max_features": 0.9, **self.hgb_params,
                "early_stopping": False, "categorical_features": None,
                "warm_start": False, "random_state": int(seed)}

    def get_params(self, deep=True):
        # Include resolved member defaults so parameter records are explicit.
        params = self._member_parameters(0)
        params.pop("random_state")
        return {"n_members": self.n_members, "max_samples": self.max_samples,
                "bootstrap": self.bootstrap, "random_state": self.random_state, **params}

    def fit(self, X, y, sample_weight=None):
        self._validate_parameters()
        values = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if (values.ndim != 2 or len(values) == 0 or values.shape[1] == 0
                or not np.isfinite(values).all()):
            raise ValueError("Bagged HGB requires a nonempty finite 2D feature array.")
        if (labels.shape != (len(values),) or labels.dtype.kind not in "iuf"
                or not np.isfinite(labels).all() or set(labels) != set(LABELS)):
            raise ValueError("Bagged HGB requires all four numeric weather labels 1..4.")
        labels = labels.astype(np.int64)
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if (weights.shape != labels.shape or not np.isfinite(weights).all()
                    or (weights <= 0).any()):
                raise ValueError("Bagged HGB sample_weight must be finite, positive and match y.")
        self.classes_ = np.asarray(LABELS)
        self.n_features_in_ = values.shape[1]
        self.members_, self.member_diagnostics_ = [], []
        sequences = np.random.SeedSequence(self.random_state).spawn(int(self.n_members))
        for index, sequence in enumerate(sequences):
            seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
            generator = np.random.default_rng(seed)
            samples = []
            for label in LABELS:
                available = np.flatnonzero(labels == label)
                count = max(1, int(np.ceil(float(self.max_samples) * len(available))))
                samples.append(generator.choice(available, size=count, replace=self.bootstrap))
            sampled = generator.permutation(np.concatenate(samples))
            learner = HistGradientBoostingClassifier(**self._member_parameters(seed))
            fit_options = {} if weights is None else {"sample_weight": weights[sampled]}
            learner.fit(values[sampled], labels[sampled], **fit_options)
            if set(learner.classes_) != set(LABELS):
                raise RuntimeError("A stratified HGB member lost a weather class.")
            self.members_.append(learner)
            self.member_diagnostics_.append({
                "member": index, "seed": seed, "supplied_fit_rows": len(values),
                "sampled_rows": len(sampled), "unique_sampled_rows": len(np.unique(sampled)),
                "class_counts": {str(label): int((labels[sampled] == label).sum()) for label in LABELS},
                "sampling_indices_sha256": sha256(sampled.astype("<i8").tobytes()).hexdigest(),
                "actual_iterations": int(learner.n_iter_),
                "trees_per_iteration": int(learner.n_trees_per_iteration_),
                "effective_model_parameters": learner.get_params(),
            })
        self.training_diagnostics_ = {
            "method": "stratified_probability_bagging",
            "member_count": len(self.members_), "base_learner_fit_count": len(self.members_),
            "bootstrap": self.bootstrap, "max_samples": float(self.max_samples),
            "sampling_rule": "ceil(max_samples * class_count) per class, minimum one; independently permuted per member",
            "member_actual_iterations": [entry["actual_iterations"] for entry in self.member_diagnostics_],
            "total_boosting_stages": sum(entry["actual_iterations"] for entry in self.member_diagnostics_),
            "members": self.member_diagnostics_,
            "scope": "Only supplied training rows and labels; no evaluation labels or early stopping. Member counts are not training epochs.",
        }
        return self

    def predict_proba(self, X):
        if not hasattr(self, "members_") or not self.members_:
            raise RuntimeError("Fit bagged HGB before requesting probabilities.")
        values = np.asarray(X, dtype=float)
        if (values.ndim != 2 or values.shape[1] != self.n_features_in_
                or not np.isfinite(values).all()):
            raise ValueError("Prediction features must be finite and match the fitted feature count.")
        mean = np.zeros((len(values), len(LABELS)), dtype=float)
        for member in self.members_:
            probability = np.asarray(member.predict_proba(values), dtype=float)
            if (probability.shape != mean.shape or not np.isfinite(probability).all()
                    or (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1)):
                raise RuntimeError("An HGB member returned invalid class probabilities.")
            for column, label in enumerate(member.classes_):
                mean[:, LABELS.index(int(label))] += probability[:, column]
        return mean / len(self.members_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
