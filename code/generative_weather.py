"""Supervised histogram Naive Bayes with a shared contamination component.

The model learns every bin boundary/category from fit data. Given class c,
the full feature vector comes either from a class-specific independent
histogram product or from one shared independent noise product. A single
latent clean/noise indicator is shared by all features of a record.
Training uses known labels and deterministic EM with additive smoothing;
no library estimator or weather-specific ranges are used.
"""

from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _finite_real(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} real number.")


def _smooth_histogram(counts, alpha):
    """Add alpha to every bin, avoiding overflow for large finite alpha."""
    total = float(np.sum(counts))
    scale = max(total, float(alpha))
    pseudocount = alpha / scale
    return (counts / scale + pseudocount) / (total / scale + len(counts) * pseudocount)


def _category_key(value):
    if value is None or (isinstance(value, Real) and np.isnan(value)):
        return ("missing",)
    if not np.isscalar(value):
        raise ValueError("Categorical values must be hashable scalars or missing.")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("Categorical values must be hashable scalars.") from exc
    return ("value", value)


class ContaminationHistogramNB:
    """Class-specific histogram products mixed with one shared noise product.

    P(x|c) = (1-eps) product_j P_clean(bin_j|c)
             + eps product_j P_noise(bin_j).

    categorical_features contains DataFrame column names, or integer column
    indices for a NumPy input. Other columns use global training quantile
    bins, plus a missing-value bin. Categorical columns always reserve an
    unknown-category bin. class_labels optionally fixes the output class
    order, including classes absent from a particular training partition.
    contamination=0 fits ordinary smoothed histogram Naive Bayes without EM.
    Positive contamination is initialized within [min_contamination,
    max_contamination] and its shared mixture weight is updated by EM.
    """

    def __init__(self, n_bins=24, alpha=0.5, contamination=0.1, max_iter=30,
                 tol=1e-5, categorical_features=None, class_labels=None,
                 min_contamination=0.01, max_contamination=0.4):
        self.n_bins = n_bins
        self.alpha = alpha
        self.contamination = contamination
        self.max_iter = max_iter
        self.tol = tol
        self.categorical_features = categorical_features
        self.class_labels = class_labels
        self.min_contamination = min_contamination
        self.max_contamination = max_contamination

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in (
            "n_bins", "alpha", "contamination", "max_iter", "tol",
            "categorical_features", "class_labels", "min_contamination", "max_contamination",
        )}

    def _validate_parameters(self):
        _positive_integer(self.n_bins, "n_bins", 2)
        _positive_integer(self.max_iter, "max_iter")
        _finite_real(self.alpha, "alpha", positive=True)
        _finite_real(self.tol, "tol")
        _finite_real(self.contamination, "contamination")
        _finite_real(self.min_contamination, "min_contamination", positive=True)
        _finite_real(self.max_contamination, "max_contamination", positive=True)
        if not self.min_contamination <= self.max_contamination < 1:
            raise ValueError("Require 0 < min_contamination <= max_contamination < 1.")
        if self.contamination != 0 and not self.min_contamination <= self.contamination <= self.max_contamination:
            raise ValueError("contamination must be 0 or within [min_contamination, max_contamination].")

    def _input_array(self, X, fitting=False):
        has_columns = hasattr(X, "columns") and hasattr(X, "to_numpy")
        values = X.to_numpy(dtype=object) if has_columns else np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] == 0 or (fitting and len(values) == 0):
            raise ValueError("X must be a two-dimensional array with features and at least one training record.")
        names = list(X.columns) if has_columns else list(range(values.shape[1]))
        if fitting:
            if len(set(names)) != len(names):
                raise ValueError("Feature names must be unique.")
            self.feature_names_in_ = tuple(names)
            self.n_features_in_ = values.shape[1]
            self._fitted_dataframe_ = has_columns
        elif values.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features.")
        elif has_columns and tuple(names) != self.feature_names_in_:
            raise ValueError("Prediction DataFrame columns must match fitted names and order.")
        return values

    def _fit_bins(self, X):
        requested = [] if self.categorical_features is None else self.categorical_features
        if isinstance(requested, (str, bytes)):
            raise ValueError("categorical_features must be a sequence of feature names/indices.")
        try:
            requested = list(requested)
            if len(set(requested)) != len(requested):
                raise ValueError("categorical_features must not contain duplicates.")
            if any(name not in self.feature_names_in_ for name in requested):
                raise ValueError("Unknown categorical feature name/index.")
        except TypeError as exc:
            raise ValueError("categorical_features must be a sequence of feature names/indices.") from exc
        self.categorical_features_ = tuple(requested)
        self.bin_edges_, self.category_maps_, self.feature_bin_counts_ = {}, {}, []
        for column, name in enumerate(self.feature_names_in_):
            if name in self.categorical_features_:
                mapping = {}
                for value in X[:, column]:
                    key = _category_key(value)
                    if key not in mapping:
                        mapping[key] = len(mapping)
                self.category_maps_[name] = mapping
                self.feature_bin_counts_.append(len(mapping) + 1)
            else:
                try:
                    numeric = np.asarray(X[:, column], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature {name!r} must be numeric or explicitly categorical.") from exc
                if np.isinf(numeric).any():
                    raise ValueError("Infinite numeric features are not supported; NaN has a dedicated bin.")
                observed = numeric[~np.isnan(numeric)]
                edges = np.array([], dtype=float)
                if len(observed) and observed.min() < observed.max():
                    edges = np.unique(np.quantile(observed, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                    edges = edges[(edges > observed.min()) & (edges < observed.max())]
                self.bin_edges_[name] = edges
                self.feature_bin_counts_.append(len(edges) + 2)  # finite intervals plus missing
        self.feature_bin_counts_ = np.array(self.feature_bin_counts_, dtype=int)

    def _encode(self, X):
        encoded = np.empty(X.shape, dtype=int)
        for column, name in enumerate(self.feature_names_in_):
            if name in self.category_maps_:
                mapping = self.category_maps_[name]
                encoded[:, column] = [mapping.get(_category_key(value), len(mapping)) for value in X[:, column]]
            else:
                try:
                    numeric = np.asarray(X[:, column], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature {name!r} must be numeric.") from exc
                if np.isinf(numeric).any():
                    raise ValueError("Infinite numeric features are not supported.")
                edges = self.bin_edges_[name]
                encoded[:, column] = np.searchsorted(edges, numeric, side="right")
                encoded[np.isnan(numeric), column] = len(edges) + 1
        return encoded

    def _update_histograms(self, bins, encoded_y, responsibility):
        self.clean_feature_prob_, self.noise_feature_prob_ = [], []
        for column, count in enumerate(self.feature_bin_counts_):
            clean = np.empty((len(self.classes_), count))
            for label in range(len(self.classes_)):
                selected = encoded_y == label
                weighted_counts = np.bincount(bins[selected, column], weights=responsibility[selected], minlength=count)
                clean[label] = _smooth_histogram(weighted_counts, self.alpha)
            noise_counts = np.bincount(bins[:, column], weights=1 - responsibility, minlength=count)
            self.clean_feature_prob_.append(clean)
            self.noise_feature_prob_.append(_smooth_histogram(noise_counts, self.alpha))
        self._clean_log_prob_ = [np.log(prob) for prob in self.clean_feature_prob_]
        self._noise_log_prob_ = [np.log(prob) for prob in self.noise_feature_prob_]

    def _component_log_likelihoods(self, bins):
        clean = np.zeros((len(bins), len(self.classes_)))
        noise = np.zeros(len(bins))
        for column in range(self.n_features_in_):
            clean += self._clean_log_prob_[column][:, bins[:, column]].T
            noise += self._noise_log_prob_[column][bins[:, column]]
        return clean, noise

    def _known_label_terms(self, bins, encoded_y):
        clean, noise = self._component_log_likelihoods(bins)
        clean = clean[np.arange(len(bins)), encoded_y]
        if self.contamination_ == 0:
            return np.ones(len(bins)), clean
        clean_term = np.log1p(-self.contamination_) + clean
        mixture = np.logaddexp(clean_term, np.log(self.contamination_) + noise)
        return np.exp(clean_term - mixture), mixture

    def fit(self, X, y):
        self._validate_parameters()
        values = self._input_array(X, fitting=True)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("y must be one-dimensional and match X rows.")
        if any(value is None or (isinstance(value, Real) and not np.isfinite(value)) for value in labels):
            raise ValueError("Training labels must be present and finite.")
        self.classes_ = np.unique(labels) if self.class_labels is None else np.asarray(self.class_labels)
        if self.classes_.ndim != 1 or len(self.classes_) == 0 or len(np.unique(self.classes_)) != len(self.classes_):
            raise ValueError("class_labels must be a nonempty one-dimensional sequence of unique labels.")
        if any(value is None or (isinstance(value, Real) and not np.isfinite(value)) for value in self.classes_):
            raise ValueError("class_labels must be present and finite.")
        label_lookup = {value: index for index, value in enumerate(self.classes_)}
        try:
            encoded_y = np.array([label_lookup[value] for value in labels], dtype=int)
        except (KeyError, TypeError) as exc:
            raise ValueError("Every training label must belong to class_labels.") from exc
        self.n_samples_fit_ = len(values)
        self.class_count_ = np.bincount(encoded_y, minlength=len(self.classes_))
        self.class_prior_ = _smooth_histogram(self.class_count_, self.alpha)
        self._fit_bins(values)
        bins = self._encode(values)
        self.contamination_ = float(self.contamination)
        # Initialization: class histograms and a pooled global noise histogram.
        self._update_histograms(bins, encoded_y, np.ones(len(bins)))
        for column, count in enumerate(self.feature_bin_counts_):
            pooled = np.bincount(bins[:, column], minlength=count)
            self.noise_feature_prob_[column] = _smooth_histogram(pooled, self.alpha)
            self._noise_log_prob_[column] = np.log(self.noise_feature_prob_[column])
        _, log_likelihood = self._known_label_terms(bins, encoded_y)
        self.log_likelihood_history_ = [float(log_likelihood.sum())]
        self.n_iter_, self.converged_ = 0, self.contamination_ == 0
        if self.contamination_ > 0:
            for iteration in range(self.max_iter):
                responsibility, _ = self._known_label_terms(bins, encoded_y)
                self._update_histograms(bins, encoded_y, responsibility)
                self.contamination_ = float(np.clip(np.mean(1 - responsibility), self.min_contamination, self.max_contamination))
                _, log_likelihood = self._known_label_terms(bins, encoded_y)
                current = float(log_likelihood.sum())
                previous = self.log_likelihood_history_[-1]
                self.log_likelihood_history_.append(current)
                self.n_iter_ = iteration + 1
                if abs(current - previous) <= self.tol * (1 + abs(previous)):
                    self.converged_ = True
                    break
        self.clean_responsibility_, _ = self._known_label_terms(bins, encoded_y)
        return self

    def predict_log_proba(self, X):
        if not hasattr(self, "class_prior_"):
            raise RuntimeError("Fit the model before prediction.")
        values = self._input_array(X)
        clean, noise = self._component_log_likelihoods(self._encode(values))
        if self.contamination_ == 0:
            conditional = clean
        else:
            conditional = np.logaddexp(np.log1p(-self.contamination_) + clean,
                                       np.log(self.contamination_) + noise[:, None])
        joint = conditional + np.log(self.class_prior_)
        return joint - logsumexp(joint, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        log_probabilities = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probabilities, axis=1)]
