"""NumPy implementations of CART classification and random forests.

Training and prediction in this module do not call scikit-learn. Numeric
features must already be finite; the experiment runner handles encoding and
imputation. Leaf probabilities are empirical class proportions. Forests
average these probabilities rather than taking a hard majority vote.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


@dataclass
class TreeNode:
    """A binary split, or a leaf when feature == -1."""

    probabilities: np.ndarray
    n_samples: int
    impurity: float
    feature: int = -1
    threshold: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _training_arrays(X, y):
    X = _feature_array(X)
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be a one-dimensional array matching X rows.")
    if y.dtype.kind in "fc" and not np.isfinite(y).all():
        raise ValueError("y must not contain missing or infinite labels.")
    if y.dtype.kind == "O" and any(value is None for value in y):
        raise ValueError("y must not contain missing labels.")
    classes, encoded = np.unique(y, return_inverse=True)
    return X, classes, encoded


def _feature_array(X, n_features=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a two-dimensional array with at least one feature.")
    if n_features is None and len(X) == 0:
        raise ValueError("Training data must contain at least one row.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, received {X.shape[1]}.")
    if not np.isfinite(X).all():
        raise ValueError("X must be finite. Impute missing values before fitting/predicting.")
    return X


class CARTClassifier:
    """Greedy binary classification tree minimizing sample-weighted Gini.

    At each node, sorted feature values define all distinct candidate
    partitions. Cumulative class counts score those partitions together.
    min_impurity_decrease uses the reduction weighted by node/root size.
    max_features limits the randomly selected candidate features per node;
    if none permits a valid split, additional features are inspected until
    a valid partition is found or all features have been checked.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, min_impurity_decrease=0.0, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def get_params(self):
        return {name: getattr(self, name) for name in (
            "max_depth", "min_samples_split", "min_samples_leaf", "max_features",
            "min_impurity_decrease", "random_state",
        )}

    def _validate_parameters(self, n_features):
        if self.max_depth is not None:
            _positive_int(self.max_depth, "max_depth")
        _positive_int(self.min_samples_split, "min_samples_split", 2)
        _positive_int(self.min_samples_leaf, "min_samples_leaf")
        if (not isinstance(self.min_impurity_decrease, Real)
                or not np.isfinite(self.min_impurity_decrease)
                or self.min_impurity_decrease < 0):
            raise ValueError("min_impurity_decrease must be finite and nonnegative.")
        value = self.max_features
        if value is None:
            return n_features
        if isinstance(value, str):
            if value == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if value == "log2":
                return max(1, int(np.log2(n_features)))
        elif isinstance(value, Integral) and not isinstance(value, bool):
            if 1 <= value <= n_features:
                return int(value)
        elif isinstance(value, Real) and not isinstance(value, bool):
            if 0 < value <= 1:
                return max(1, int(value * n_features))
        raise ValueError("max_features must be None, 'sqrt', 'log2', an integer in [1, p], or a fraction in (0, 1].")

    def fit(self, X, y):
        X, classes, encoded = _training_arrays(X, y)
        return self._fit_encoded(X, encoded, classes)

    def _fit_encoded(self, X, encoded, classes):
        """Internal forest entry point: retain the global class order."""
        self.n_features_in_ = X.shape[1]
        self.max_features_ = self._validate_parameters(self.n_features_in_)
        self.classes_ = np.array(classes, copy=True)
        self.n_classes_ = len(classes)
        self.n_samples_fit_ = len(X)
        self._rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(self.n_features_in_)
        self.root_ = self._make_node(encoded)
        self.node_count_, self.n_leaves_, self.depth_ = 1, 0, 0
        # An explicit stack avoids Python recursion limits for deep trees.
        stack = [(self.root_, np.arange(len(X)), 0)]
        while stack:
            node, indices, depth = stack.pop()
            self.depth_ = max(self.depth_, depth)
            should_stop = (
                node.impurity <= 0
                or node.n_samples < max(self.min_samples_split, 2 * self.min_samples_leaf)
                or (self.max_depth is not None and depth >= self.max_depth)
            )
            split = None if should_stop else self._best_split(X, encoded, indices, node)
            if split is None:
                self.n_leaves_ += 1
                continue
            feature, threshold, gain = split
            weighted_gain = node.n_samples / self.n_samples_fit_ * gain
            if weighted_gain + 1e-15 < self.min_impurity_decrease:
                self.n_leaves_ += 1
                continue
            goes_left = X[indices, feature] <= threshold
            left_indices, right_indices = indices[goes_left], indices[~goes_left]
            node.feature, node.threshold = feature, threshold
            node.left = self._make_node(encoded[left_indices])
            node.right = self._make_node(encoded[right_indices])
            self.feature_importances_[feature] += weighted_gain
            self.node_count_ += 2
            stack.append((node.right, right_indices, depth + 1))
            stack.append((node.left, left_indices, depth + 1))
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _make_node(self, encoded):
        counts = np.bincount(encoded, minlength=self.n_classes_)
        probabilities = counts / len(encoded)
        return TreeNode(probabilities, len(encoded), max(0.0, 1.0 - float(probabilities @ probabilities)))

    def _best_split(self, X, encoded, indices, node):
        n = len(indices)
        local_y = encoded[indices]
        total_counts = np.bincount(local_y, minlength=self.n_classes_)
        best, best_gain = None, -np.inf
        features = (np.arange(self.n_features_in_) if self.max_features_ == self.n_features_in_
                    else self._rng.permutation(self.n_features_in_))
        for inspected, feature in enumerate(features, 1):
            values = X[indices, feature]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
            positions = positions[(positions + 1 >= self.min_samples_leaf)
                                  & (n - positions - 1 >= self.min_samples_leaf)]
            if len(positions):
                indicators = np.zeros((n, self.n_classes_), dtype=np.int64)
                indicators[np.arange(n), local_y[order]] = 1
                left_counts = np.cumsum(indicators, axis=0)[positions]
                right_counts = total_counts - left_counts
                left_n = positions + 1
                right_n = n - left_n
                # child_Gini = 1 - (sum(L_k^2)/|L| + sum(R_k^2)/|R|)/n
                children_impurity = 1.0 - (
                    np.einsum("ij,ij->i", left_counts, left_counts) / left_n
                    + np.einsum("ij,ij->i", right_counts, right_counts) / right_n
                ) / n
                choice = int(np.argmin(children_impurity))
                gain = node.impurity - float(children_impurity[choice])
                if gain > best_gain + 1e-15:
                    position = positions[choice]
                    low, high = sorted_values[position], sorted_values[position + 1]
                    # Halved operands avoid overflow; using low is a safe
                    # fallback if adjacent floats have no midpoint below high.
                    threshold = float(low / 2.0 + high / 2.0)
                    if not low <= threshold < high:
                        threshold = float(low)
                    best = (int(feature), threshold, gain)
                    best_gain = gain
            if inspected >= self.max_features_ and best is not None:
                break
        # Zero-gain partitions can expose useful deeper interactions (e.g. XOR).
        return best

    def _predict_proba_validated(self, X):
        result = np.empty((len(X), self.n_classes_), dtype=float)
        stack = [(self.root_, np.arange(len(X)))]
        while stack:
            node, indices = stack.pop()
            if len(indices) == 0:
                continue
            if node.feature == -1:
                result[indices] = node.probabilities
            else:
                mask = X[indices, node.feature] <= node.threshold
                stack.append((node.right, indices[~mask]))
                stack.append((node.left, indices[mask]))
        return result

    def predict_proba(self, X):
        if not hasattr(self, "root_"):
            raise RuntimeError("Fit the tree before prediction.")
        return self._predict_proba_validated(_feature_array(X, self.n_features_in_))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class RandomForestClassifier:
    """Bootstrap CART ensemble with per-node random features and optional OOB.

    Each tree has an independent reproducible seed. A global class vocabulary
    is supplied even if a bootstrap sample happens to omit one class. OOB
    probabilities use only trees that did not train on the given record.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def get_params(self):
        return {name: getattr(self, name) for name in (
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "max_features", "min_impurity_decrease", "bootstrap", "oob_score", "random_state",
        )}

    def fit(self, X, y):
        X, self.classes_, encoded = _training_arrays(X, y)
        _positive_int(self.n_estimators, "n_estimators")
        if not isinstance(self.bootstrap, bool) or not isinstance(self.oob_score, bool):
            raise ValueError("bootstrap and oob_score must be booleans.")
        if self.oob_score and not self.bootstrap:
            raise ValueError("OOB evaluation requires bootstrap=True.")
        self.n_features_in_, self.n_classes_ = X.shape[1], len(self.classes_)
        self.n_samples_fit_ = len(X)
        rng = np.random.default_rng(self.random_state)
        self.estimators_, self.estimators_samples_ = [], []
        # Reset optional results when re-fitting a previously OOB-enabled model.
        for name in ("oob_decision_function_", "oob_counts_", "oob_score_", "oob_coverage_"):
            self.__dict__.pop(name, None)
        if self.oob_score:
            sums = np.zeros((len(X), self.n_classes_))
            counts = np.zeros(len(X), dtype=int)
        for _ in range(self.n_estimators):
            sample = rng.integers(0, len(X), size=len(X)) if self.bootstrap else np.arange(len(X))
            tree = CARTClassifier(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )._fit_encoded(X[sample], encoded[sample], self.classes_)
            self.estimators_.append(tree)
            self.estimators_samples_.append(sample)
            if self.oob_score:
                out_of_bag = np.bincount(sample, minlength=len(X)) == 0
                sums[out_of_bag] += tree._predict_proba_validated(X[out_of_bag])
                counts[out_of_bag] += 1
        self.feature_importances_ = np.mean([tree.feature_importances_ for tree in self.estimators_], axis=0)
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        if self.oob_score:
            covered = counts > 0
            self.oob_counts_ = counts
            self.oob_coverage_ = float(covered.mean())
            self.oob_decision_function_ = np.full_like(sums, np.nan)
            self.oob_decision_function_[covered] = sums[covered] / counts[covered, None]
            self.oob_score_ = (float(np.mean(np.argmax(sums[covered], axis=1) == encoded[covered]))
                               if covered.any() else float("nan"))
        return self

    def predict_proba(self, X):
        if not getattr(self, "estimators_", None):
            raise RuntimeError("Fit the forest before prediction.")
        X = _feature_array(X, self.n_features_in_)
        result = np.zeros((len(X), self.n_classes_))
        for tree in self.estimators_:
            result += tree._predict_proba_validated(X)
        return result / len(self.estimators_)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
