"""Numerical and validation checks for the independent blending component."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1.ensemble import apply_blend, fit_blend, predict_blend


def recipe(names, weights, multipliers=None):
    return {
        "schema_version": 1,
        "labels": [1, 2, 3, 4],
        "model_ids": names,
        "weights": weights,
        "class_multipliers": [1, 1, 1, 1] if multipliers is None else multipliers,
    }


class BlendTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([
            [0.6, 0.2, 0.1, 0.1],
            [0.1, 0.6, 0.2, 0.1],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.1, 0.2, 0.6],
        ])
        self.b = np.array([
            [0.3, 0.4, 0.2, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.1, 0.3, 0.5, 0.1],
            [0.2, 0.1, 0.2, 0.5],
        ])

    def test_weighted_probability_average_matches_hand_calculation(self):
        frozen = recipe(["a", "b"], [0.25, 0.75])
        actual = apply_blend({"a": self.a, "b": self.b}, frozen)
        np.testing.assert_allclose(actual[0], [0.375, 0.35, 0.175, 0.1])
        np.testing.assert_allclose(actual, 0.25 * self.a + 0.75 * self.b)
        np.testing.assert_array_equal(predict_blend({"a": self.a, "b": self.b}, frozen), [1, 2, 3, 4])

    def test_multipliers_are_applied_then_normalized(self):
        frozen = recipe(["a"], [1], [0.9, 1.1, 1, 1])
        actual = apply_blend({"a": self.a}, frozen)
        expected = np.array([0.54, 0.22, 0.1, 0.1]) / 0.96
        np.testing.assert_allclose(actual[0], expected)
        np.testing.assert_allclose(actual.sum(axis=1), 1)

    def test_search_deterministic_independent_of_mapping_insertion_order(self):
        y = np.array([1, 2, 3, 4])
        first = fit_blend({"b": self.b, "a": self.a}, y, seed=117, n_trials=24,
                          allow_class_multipliers=True)
        second = fit_blend({"a": self.a, "b": self.b}, y, seed=117, n_trials=24,
                           allow_class_multipliers=True)
        self.assertEqual(first, second)
        self.assertEqual(first["model_ids"], ["a", "b"])
        # Equal-weight predictions are already perfect; an equal score must
        # not add arbitrary class corrections or discard a component.
        self.assertEqual(first["weights"], [0.5, 0.5])
        self.assertEqual(first["class_multipliers"], [1, 1, 1, 1])
        self.assertEqual(first["selected_candidate"], "equal_weights")
        self.assertEqual(json.loads(json.dumps(first)), first)

    def test_single_model_candidate_can_improve_over_equal_weights(self):
        # Wrong model deliberately dominates equal averaging on each record.
        sharp_wrong = np.eye(4)[[1, 2, 3, 0]]
        y = np.array([1, 2, 3, 4])
        selected = fit_blend({"correct": self.a, "wrong": sharp_wrong}, y, n_trials=0)
        self.assertEqual(selected["selection_macro_f1"], 1)
        self.assertEqual(selected["selection_baseline_macro_f1"], 0)
        np.testing.assert_array_equal(predict_blend({"correct": self.a, "wrong": sharp_wrong}, selected), y)

    def test_small_class_multiplier_can_recover_an_ambiguous_class(self):
        probabilities = self.a.copy()
        probabilities[1] = [0.49, 0.48, 0.02, 0.01]
        selected = fit_blend({"one": probabilities}, [1, 2, 3, 4], n_trials=0,
                             allow_class_multipliers=True)
        self.assertGreater(selected["selection_delta"], 0)
        self.assertEqual(selected["selection_macro_f1"], 1)
        self.assertTrue(all(0.9 <= value <= 1.1 for value in selected["class_multipliers"]))
        np.testing.assert_array_equal(predict_blend({"one": probabilities}, selected), [1, 2, 3, 4])

    def test_selection_score_matches_fixed_four_class_metric(self):
        rng = np.random.default_rng(303)
        probabilities = {"one": rng.dirichlet(np.ones(4), size=32),
                         "two": rng.dirichlet(np.ones(4), size=32)}
        # Class 4 is intentionally absent; it still contributes zero F1.
        y = rng.integers(1, 4, size=32)
        selected = fit_blend(probabilities, y, n_trials=16, allow_class_multipliers=True)
        expected = f1_score(y, predict_blend(probabilities, selected),
                            labels=[1, 2, 3, 4], average="macro", zero_division=0)
        self.assertAlmostEqual(selected["selection_macro_f1"], expected, places=14)
        self.assertIn("not independent evaluation", selected["evaluation_scope"])
        self.assertIn("outer-fold", selected["evaluation_scope"])
        self.assertGreaterEqual(selected["selection_delta"], 0)

    def test_no_mutation_of_arrays_recipe_or_labels(self):
        probabilities = {"a": self.a.copy(), "b": self.b.copy()}
        original = copy.deepcopy(probabilities)
        y = np.array([1, 2, 3, 4])
        y_before = y.copy()
        selected = fit_blend(probabilities, y, n_trials=12, allow_class_multipliers=True)
        saved_recipe = copy.deepcopy(selected)
        apply_blend(probabilities, selected)
        for name in probabilities:
            np.testing.assert_array_equal(probabilities[name], original[name])
        np.testing.assert_array_equal(y, y_before)
        self.assertEqual(selected, saved_recipe)

    def test_probability_input_errors(self):
        invalid = [
            {}, {1: self.a}, {"": self.a}, {"a": np.empty((0, 4))},
            {"a": np.ones((4, 3)) / 3}, {"a": self.a[0]},
            {"a": self.a * 0.5}, {"a": self.a, "b": self.b[:2]},
            {"a": np.full((2, 4), np.nan)},
            {"a": np.array([[1.1, -0.1, 0, 0]])},
            {"a": [["bad", "bad", "bad", "bad"]]},
        ]
        for probabilities in invalid:
            with self.subTest(probabilities=str(probabilities)[:60]):
                with self.assertRaises(ValueError):
                    fit_blend(probabilities, [1, 2, 3, 4])

    def test_label_and_search_parameter_errors(self):
        for y in ([1, 2], [[1, 2, 3, 4]], [0, 1, 2, 3], [1, 2, 3, 5],
                  [1, 2, 3, np.nan], ["1", "2", "3", "4"], [True] * 4):
            with self.subTest(y=y):
                with self.assertRaises(ValueError):
                    fit_blend({"a": self.a}, y)
        for kwargs in ({"seed": -1}, {"seed": True}, {"seed": None},
                       {"n_trials": -1}, {"n_trials": 1.5}, {"n_trials": True},
                       {"allow_class_multipliers": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    fit_blend({"a": self.a}, [1, 2, 3, 4], **kwargs)

    def test_recipe_errors_and_exact_model_membership(self):
        base = recipe(["a"], [1])
        invalid = [None, {}, {**base, "schema_version": 2},
                   {**base, "labels": [4, 3, 2, 1]},
                   {**base, "model_ids": ["a", "a"]},
                   {**base, "weights": [0.5]}, {**base, "weights": [-1]},
                   {**base, "weights": [np.nan]},
                   {**base, "class_multipliers": [1, 1, 1]},
                   {**base, "class_multipliers": [1, 1, 1, 1.2]}]
        for frozen in invalid:
            with self.subTest(recipe=frozen):
                with self.assertRaises(ValueError):
                    apply_blend({"a": self.a}, frozen)
        for probabilities in ({"b": self.b}, {"a": self.a, "b": self.b}):
            with self.assertRaises(ValueError):
                apply_blend(probabilities, base)

    def test_ties_use_fixed_class_order_and_single_model_is_supported(self):
        probabilities = {"a": np.full((3, 4), 0.25)}
        selected = fit_blend(probabilities, [1, 2, 3], n_trials=20)
        self.assertEqual(selected["weights"], [1])
        self.assertEqual(selected["search"]["random_simplex_trials"], 0)
        np.testing.assert_array_equal(predict_blend(probabilities, selected), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
