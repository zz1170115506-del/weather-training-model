"""Independent numerical checks for the handwritten algorithms."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weather_models import CARTClassifier, RandomForestClassifier


def gini(labels):
    """Small brute-force oracle, independent of the vectorized split code."""
    _, counts = np.unique(labels, return_counts=True)
    return 1 - sum((count / len(labels)) ** 2 for count in counts)


class CARTTests(unittest.TestCase):
    def test_root_matches_exhaustive_multiclass_search(self):
        rng = np.random.default_rng(12)
        X = rng.integers(0, 8, size=(45, 4)).astype(float)
        y = rng.choice([1, 3, 9], size=45)
        tree = CARTClassifier(max_depth=1, min_samples_leaf=4).fit(X, y)
        objectives = []
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for low, high in zip(values[:-1], values[1:]):
                mask = X[:, feature] <= (low + high) / 2
                if min(mask.sum(), (~mask).sum()) >= 4:
                    objectives.append((mask.sum() * gini(y[mask]) + (~mask).sum() * gini(y[~mask])) / len(y))
        root = tree.root_
        mask = X[:, root.feature] <= root.threshold
        actual = (mask.sum() * gini(y[mask]) + (~mask).sum() * gini(y[~mask])) / len(y)
        self.assertAlmostEqual(actual, min(objectives), places=13)
        np.testing.assert_allclose(tree.predict_proba(X[mask])[:, :],
                                   np.broadcast_to([np.mean(y[mask] == k) for k in tree.classes_], (mask.sum(), 3)))

    def test_zero_gain_root_can_learn_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([1, 4, 4, 1])
        tree = CARTClassifier(max_depth=2).fit(X, y)
        np.testing.assert_array_equal(tree.predict(X), y)
        self.assertEqual(tree.depth_, 2)

    def test_constant_features_and_string_labels(self):
        X = np.ones((6, 3))
        y = np.array(["rain", "sun", "sun", "rain", "sun", "sun"])
        tree = CARTClassifier().fit(X, y)
        self.assertEqual(tree.node_count_, 1)
        np.testing.assert_array_equal(tree.classes_, ["rain", "sun"])
        np.testing.assert_allclose(tree.predict_proba([[1, 1, 1]]), [[1/3, 2/3]])
        np.testing.assert_array_equal(tree.feature_importances_, [0, 0, 0])

    def test_depth_leaf_size_and_impurity_stopping(self):
        rng = np.random.default_rng(20)
        X = rng.normal(size=(100, 3))
        y = rng.integers(0, 4, size=100)
        tree = CARTClassifier(max_depth=3, min_samples_leaf=7).fit(X, y)
        self.assertLessEqual(tree.depth_, 3)
        nodes = [tree.root_]
        while nodes:
            node = nodes.pop()
            if node.feature == -1:
                self.assertGreaterEqual(node.n_samples, 7)
            else:
                self.assertEqual(node.n_samples, node.left.n_samples + node.right.n_samples)
                nodes.extend([node.left, node.right])
        stopped = CARTClassifier(min_impurity_decrease=1.0).fit(X, y)
        self.assertEqual(stopped.node_count_, 1)

    def test_extreme_and_adjacent_float_thresholds(self):
        for low, high in [(1.0, np.nextafter(1.0, 2.0)), (-1e308, 1e308), (1e308, 1.1e308)]:
            X = np.array([[low], [low], [high], [high]])
            tree = CARTClassifier().fit(X, [1, 1, 4, 4])
            self.assertTrue(np.isfinite(tree.root_.threshold))
            np.testing.assert_array_equal(tree.predict(X), [1, 1, 4, 4])

    def test_impurity_threshold_is_weighted_by_root_sample_count(self):
        X = np.arange(10).reshape(-1, 1)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])
        # Root gain is .32. The right child's local gain is .12, but its
        # root-weighted gain is .06, so a .07 minimum must reject that split.
        tree = CARTClassifier(min_impurity_decrease=0.07).fit(X, y)
        self.assertEqual(tree.node_count_, 3)
        self.assertEqual(tree.root_.right.n_samples, 5)
        self.assertEqual(tree.root_.right.feature, -1)

    def test_random_features_search_past_constant_candidates(self):
        X = np.zeros((20, 15))
        X[10:, -1] = 1
        y = np.array([2] * 10 + [5] * 10)
        tree = CARTClassifier(max_features=1, random_state=7).fit(X, y)
        np.testing.assert_array_equal(tree.predict(X), y)
        self.assertEqual(tree.root_.feature, 14)

    def test_input_errors_and_unfitted_prediction(self):
        with self.assertRaises(RuntimeError):
            CARTClassifier().predict([[1]])
        for X, y in [([], []), ([[np.nan]], [1]), ([[1], [2]], [1]), ([[np.inf]], [1])]:
            with self.assertRaises(ValueError):
                CARTClassifier().fit(X, y)
        for kwargs in [{"min_samples_leaf": 0}, {"max_features": 4}, {"max_depth": -1}, {"min_impurity_decrease": float("nan")}]:
            with self.assertRaises(ValueError):
                CARTClassifier(**kwargs).fit([[0], [1]], [0, 1])
        tree = CARTClassifier().fit([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            tree.predict([[0, 1]])
        self.assertEqual(tree.predict_proba(np.empty((0, 1))).shape, (0, 2))


class ForestTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(44)
        self.X = rng.normal(size=(75, 5))
        self.y = np.where(self.X[:, 0] > 0, 8, 2)
        self.y[0] = 19  # A rare class is omitted by some bootstrap samples.

    def test_probability_average_class_alignment_and_reproducibility(self):
        kwargs = dict(n_estimators=17, min_samples_leaf=2, random_state=32)
        forest = RandomForestClassifier(**kwargs).fit(self.X, self.y)
        repeated = RandomForestClassifier(**kwargs).fit(self.X, self.y)
        predictions = forest.predict_proba(self.X)
        manual = np.mean([tree.predict_proba(self.X) for tree in forest.estimators_], axis=0)
        np.testing.assert_allclose(predictions, manual, atol=1e-15)
        np.testing.assert_array_equal(predictions, repeated.predict_proba(self.X))
        np.testing.assert_allclose(predictions.sum(axis=1), 1)
        np.testing.assert_array_equal(forest.classes_, [2, 8, 19])
        self.assertTrue(any(19 not in self.y[sample] for sample in forest.estimators_samples_))
        self.assertTrue(all(np.array_equal(tree.classes_, forest.classes_) for tree in forest.estimators_))
        self.assertTrue(any(len(np.unique(sample)) < len(self.X) for sample in forest.estimators_samples_))
        self.assertAlmostEqual(forest.feature_importances_.sum(), 1)

    def test_oob_predictions_use_only_unseen_records(self):
        forest = RandomForestClassifier(n_estimators=15, oob_score=True, random_state=6).fit(self.X, self.y)
        expected = np.full((len(self.X), len(forest.classes_)), np.nan)
        for row in range(len(self.X)):
            eligible = [tree for tree, sample in zip(forest.estimators_, forest.estimators_samples_) if row not in sample]
            self.assertEqual(forest.oob_counts_[row], len(eligible))
            if eligible:
                expected[row] = np.mean([tree.predict_proba(self.X[row:row+1])[0] for tree in eligible], axis=0)
        np.testing.assert_allclose(forest.oob_decision_function_, expected, equal_nan=True)
        covered = forest.oob_counts_ > 0
        expected_accuracy = np.mean(forest.classes_[np.argmax(expected[covered], axis=1)] == self.y[covered])
        self.assertEqual(forest.oob_score_, expected_accuracy)

    def test_oob_uncovered_and_refit_reset(self):
        forest = RandomForestClassifier(n_estimators=1, oob_score=True, random_state=6).fit([[0]], [4])
        self.assertEqual(forest.oob_coverage_, 0)
        self.assertTrue(np.isnan(forest.oob_decision_function_).all())
        forest.oob_score = False
        forest.fit([[0], [1]], [1, 2])
        self.assertFalse(hasattr(forest, "oob_score_"))
        with self.assertRaises(ValueError):
            RandomForestClassifier(bootstrap=False, oob_score=True).fit(self.X, self.y)

    def test_no_bootstrap_and_model_serialization(self):
        import pickle
        forest = RandomForestClassifier(n_estimators=4, bootstrap=False, random_state=9).fit(self.X, self.y)
        self.assertTrue(all(np.array_equal(sample, np.arange(len(self.X))) for sample in forest.estimators_samples_))
        loaded = pickle.loads(pickle.dumps(forest))
        np.testing.assert_array_equal(forest.predict_proba(self.X), loaded.predict_proba(self.X))


if __name__ == "__main__":
    unittest.main()
