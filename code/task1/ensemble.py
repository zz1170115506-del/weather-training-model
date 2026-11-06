"""Deterministic probability blending with explicit selection-score semantics.

Probability columns must already be aligned to LABELS. The handwritten search
chooses nonnegative simplex weights; optional class multipliers stay within
[0.9, 1.1]. Its score uses the labels supplied to ``fit_blend`` and is therefore
a tuning score, even when the supplied predictions are out of fold. An outer
evaluation must learn the recipe without seeing the outer validation labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np


LABELS = (1, 2, 3, 4)
SELECTION_SCOPE = (
    "Selection score on the supplied labels; OOF used to tune weights is not "
    "independent evaluation. Evaluate the frozen recipe on separate outer-fold "
    "predictions before claiming an improvement."
)


def _probability_arrays(probabilities, model_ids=None):
    """Validate without modifying caller arrays, and fix model ordering."""
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("probabilities must be a nonempty model-name mapping.")
    if any(not isinstance(name, str) or not name for name in probabilities):
        raise ValueError("Every model name must be a nonempty string.")
    if model_ids is None:
        model_ids = sorted(probabilities)
    elif set(probabilities) != set(model_ids):
        raise ValueError("Probability model names must exactly match recipe model_ids.")
    arrays, expected_rows = [], None
    for name in model_ids:
        try:
            values = np.asarray(probabilities[name], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}: probabilities must be numeric.") from error
        if values.ndim != 2 or values.shape[1] != len(LABELS) or len(values) == 0:
            raise ValueError(f"{name}: expected a nonempty (n, 4) probability matrix.")
        if expected_rows is not None and len(values) != expected_rows:
            raise ValueError("All probability matrices must have the same row count.")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{name}: probabilities must be finite and in [0, 1].")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{name}: each probability row must sum to one.")
        expected_rows = len(values)
        arrays.append(values)
    return list(model_ids), np.stack(arrays, axis=0)


def _encoded_labels(y, n_rows):
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("y must be a one-dimensional array matching probability rows.")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("y must contain finite numeric labels 1, 2, 3, 4.")
    if not np.isin(values, LABELS).all():
        raise ValueError("y must contain only labels 1, 2, 3, 4.")
    return values.astype(np.int64) - 1


def _macro_f1(encoded_y, scores):
    """Four fixed classes, including zero F1 when a class has no denominator."""
    predicted = np.argmax(scores, axis=1)
    confusion = np.bincount(4 * encoded_y + predicted, minlength=16).reshape(4, 4)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2.0 * np.diag(confusion), denominator,
        out=np.zeros(4, dtype=np.float64), where=denominator > 0,
    )
    return float(f1.mean())


def _recipe_arrays(recipe):
    if not isinstance(recipe, Mapping) or recipe.get("schema_version") != 1:
        raise ValueError("Expected a blend recipe with schema_version=1.")
    if recipe.get("labels") != list(LABELS):
        raise ValueError("Recipe labels must be [1, 2, 3, 4] in that order.")
    names = recipe.get("model_ids")
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Recipe model_ids must be a nonempty list of unique names.")
    try:
        weights = np.asarray(recipe.get("weights"), dtype=np.float64)
        multipliers = np.asarray(recipe.get("class_multipliers"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Recipe weights and multipliers must be numeric.") from error
    if (weights.shape != (len(names),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("Recipe weights must be nonnegative and sum to one.")
    if (multipliers.shape != (4,) or not np.isfinite(multipliers).all()
            or np.any(multipliers < 0.9) or np.any(multipliers > 1.1)):
        raise ValueError("Recipe class_multipliers must have four values in [0.9, 1.1].")
    return names, weights, multipliers


def apply_blend(probabilities, recipe):
    """Apply a frozen recipe, returning normalized probabilities in LABELS order.

    Mapping insertion order is ignored. The mapping must contain exactly the
    recipe model names, including any names whose selected weight is zero.
    """
    names, weights, multipliers = _recipe_arrays(recipe)
    _, stacked = _probability_arrays(probabilities, names)
    result = np.einsum("m,mnc->nc", weights, stacked, optimize=False) * multipliers
    return result / result.sum(axis=1, keepdims=True)


def predict_blend(probabilities, recipe):
    """Return integer weather labels; ties follow the fixed LABELS order."""
    return np.asarray(LABELS)[np.argmax(apply_blend(probabilities, recipe), axis=1)]


def fit_blend(probabilities, y, seed=5434, n_trials=128, allow_class_multipliers=False):
    """Search a small simplex budget and return a JSON-serializable recipe.

    Equal weights are the baseline. Single models and pairwise 25/50/75 percent
    mixtures are always considered, followed by ``n_trials`` seeded Dirichlet
    proposals. Optional class-multiplier coordinate search evaluates 0.9, 0.95,
    1.0, 1.05 and 1.1, for at most two passes. Strictly better selection F1 is
    required to replace a candidate, so ties preserve the simpler baseline.

    ``n_trials`` counts random simplex proposals, not the mandatory baselines
    or the optional multiplier proposals. The actual number is recorded.
    No part of this function is an independent validation evaluation.
    """
    for value, name in ((seed, "seed"), (n_trials, "n_trials")):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if not isinstance(allow_class_multipliers, bool):
        raise ValueError("allow_class_multipliers must be a boolean.")
    names, stacked = _probability_arrays(probabilities)
    encoded_y = _encoded_labels(y, stacked.shape[1])
    n_models = len(names)
    best_weights = np.full(n_models, 1.0 / n_models)
    best_multipliers = np.ones(4)
    initial_scores = np.einsum("m,mnc->nc", best_weights, stacked, optimize=False)
    best_f1 = baseline_f1 = _macro_f1(encoded_y, initial_scores)
    selected_candidate = "equal_weights"
    evaluated = 1

    def consider(weights, multipliers, candidate):
        nonlocal best_weights, best_multipliers
        nonlocal best_f1, selected_candidate, evaluated
        scores = np.einsum("m,mnc->nc", weights, stacked, optimize=False)
        score = _macro_f1(encoded_y, scores * multipliers)
        evaluated += 1
        if score > best_f1 + 1e-12:
            best_weights = weights.copy()
            best_multipliers = multipliers.copy()
            best_f1 = score
            selected_candidate = candidate

    if n_models > 1:
        for i in range(n_models):
            weights = np.zeros(n_models)
            weights[i] = 1
            consider(weights, np.ones(4), f"single_model:{names[i]}")
        for i in range(n_models):
            for j in range(i + 1, n_models):
                for left_weight in (0.25, 0.5, 0.75):
                    weights = np.zeros(n_models)
                    weights[i], weights[j] = left_weight, 1 - left_weight
                    consider(weights, np.ones(4), "pairwise_simplex")
        rng = np.random.default_rng(int(seed))
        for trial in range(int(n_trials)):
            concentration = (0.3, 1.0, 3.0)[trial % 3]
            weights = rng.dirichlet(np.full(n_models, concentration))
            consider(weights, np.ones(4), "seeded_simplex")

    # Freeze the selected model weights, then explore small decision changes.
    # These remain selection results and require an isolated outer evaluation.
    if allow_class_multipliers:
        for _ in range(2):
            previous_f1 = best_f1
            for column in range(4):
                for scale in (0.9, 0.95, 1.0, 1.05, 1.1):
                    multipliers = best_multipliers.copy()
                    multipliers[column] = scale
                    consider(best_weights, multipliers, "class_multiplier_coordinate_search")
            if best_f1 <= previous_f1 + 1e-12:
                break

    return {
        "schema_version": 1,
        "labels": list(LABELS),
        "model_ids": names,
        "weights": best_weights.tolist(),
        "class_multipliers": best_multipliers.tolist(),
        "selection_macro_f1": best_f1,
        "selection_baseline_macro_f1": baseline_f1,
        "selection_delta": best_f1 - baseline_f1,
        "selected_candidate": selected_candidate,
        "evaluation_scope": SELECTION_SCOPE,
        "search": {
            "seed": int(seed),
            "random_simplex_trials": int(n_trials) if n_models > 1 else 0,
            "allow_class_multipliers": allow_class_multipliers,
            "evaluated_candidates": evaluated,
        },
    }
