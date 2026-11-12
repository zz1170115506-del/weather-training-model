"""Training-only, out-of-fold confidence weights without deleting/relabeling rows.

This is a deliberately conservative, handwritten weighting rule inspired by
out-of-sample label-confidence methods. It is not a full implementation of
Confident Learning and does not claim to identify the true label-error rate.
Every teacher fits preprocessing and a fresh HGB only on its internal fit fold.
The API accepts one allowed training partition, never evaluation/test records.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from numbers import Integral, Real
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import LABELS
from .data import FEATURES
from .features import Features


_TEACHER_DEFAULTS = {
    "max_iter": 600, "learning_rate": 0.03, "max_leaf_nodes": 15,
    "min_samples_leaf": 40, "l2_regularization": 15.0,
}
_TEACHER_ALLOWED = set(_TEACHER_DEFAULTS) | {"max_bins", "max_depth"}
_QUANTILE_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be an integer in the accepted range.")
    return int(value)


def _number(value, name, minimum=0.0, maximum=None, strict_minimum=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            or (value <= minimum if strict_minimum else value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be finite and in the accepted range.")
    return float(value)


def resolve_settings(settings):
    """Resolve a small strict schema; no callbacks, warm starts or ES overrides."""
    if not isinstance(settings, dict) or set(settings) - {"n_splits", "threshold", "floor", "teacher_params"}:
        raise ValueError("crossfit_weighting supports n_splits, threshold, floor and teacher_params only.")
    result = {"n_splits": 3, "threshold": 0.85, "floor": 0.5} | deepcopy(settings)
    result["n_splits"] = _integer(result["n_splits"], "n_splits", 2, 10)
    result["threshold"] = _number(result["threshold"], "threshold", 0.5, 1.0)
    if result["threshold"] == 1.0:
        raise ValueError("threshold must be less than 1.")
    # Positive weights preserve every original observation in the final fit.
    result["floor"] = _number(result["floor"], "floor", 0.0, 1.0, strict_minimum=True)
    params = result.get("teacher_params", {})
    if not isinstance(params, dict) or set(params) - _TEACHER_ALLOWED:
        raise ValueError("Unknown teacher_params; teacher state, seeds and stopping cannot be supplied.")
    params = _TEACHER_DEFAULTS | params
    for key in ("max_iter", "min_samples_leaf"):
        params[key] = _integer(params[key], f"teacher_params.{key}")
    params["max_leaf_nodes"] = _integer(params["max_leaf_nodes"], "teacher_params.max_leaf_nodes", 2)
    params["learning_rate"] = _number(params["learning_rate"], "teacher_params.learning_rate", 0.0, 1.0, strict_minimum=True)
    params["l2_regularization"] = _number(params["l2_regularization"], "teacher_params.l2_regularization")
    if "max_bins" in params:
        params["max_bins"] = _integer(params["max_bins"], "teacher_params.max_bins", 2, 255)
    if "max_depth" in params and params["max_depth"] is not None:
        params["max_depth"] = _integer(params["max_depth"], "teacher_params.max_depth")
    result["teacher_params"] = params
    return result


def _digest(values, dtype):
    return sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def _quantiles(values):
    return np.quantile(values, _QUANTILE_LEVELS).tolist()


def training_weights(X, y, settings, seed, feature_options=None):
    """Return positive weights plus aggregate diagnostics for one fit partition.

    Only an OOF wrong-class prediction with maximum probability strictly above
    the threshold is downweighted. With confidence ``q``, the weight is
    ``1 - (1-floor)*(q-threshold)/(1-threshold)``; every other weight is 1.
    All rows and supplied labels stay unchanged. Identical raw feature groups
    cannot cross teacher fit/holdout boundaries, even if caller indices differ.
    """
    started = perf_counter()
    resolved = resolve_settings(settings)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    if not isinstance(X, pd.DataFrame) or not set(FEATURES).issubset(X.columns):
        raise ValueError("X must be a DataFrame containing the ten original course features.")
    if feature_options is not None and not isinstance(feature_options, dict):
        raise ValueError("feature_options must be an object or None.")
    labels = np.asarray(y)
    if (labels.ndim != 1 or len(labels) != len(X) or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or not np.isin(labels, LABELS).all()
            or set(labels) != set(LABELS)):
        raise ValueError("y must match X and contain all four finite weather labels 1..4.")
    labels = labels.astype(np.int64)
    frame = X[FEATURES]
    groups = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    n_splits = resolved["n_splits"]
    if min(len(np.unique(groups[labels == label])) for label in LABELS) < n_splits:
        raise ValueError("Each weather class needs at least n_splits independent raw feature groups.")
    split_seed = (seed + 700001) % 2**32
    if len(np.unique(groups)) == len(groups):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        splits = list(splitter.split(np.zeros((len(labels), 1)), labels))
        split_method = "StratifiedKFold"
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        splits = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
        split_method = "StratifiedGroupKFold"
    for fit, holdout in splits:
        if (set(groups[fit]) & set(groups[holdout]) or set(labels[fit]) != set(LABELS)
                or set(labels[holdout]) != set(LABELS)):
            raise ValueError("Cross-fit teacher folds must be group-disjoint and contain all four classes.")

    oof = np.full((len(labels), len(LABELS)), np.nan)
    coverage = np.zeros(len(labels), dtype=np.int64)
    records = []
    teacher_parameters = resolved["teacher_params"] | {"early_stopping": False, "categorical_features": None}
    for fold, (fit, holdout) in enumerate(splits):
        teacher_seed = (seed + 800011 + fold) % 2**32
        features = Features(feature_options)
        fit_X = features.fit_transform(frame.iloc[fit])
        holdout_X = features.transform(frame.iloc[holdout])
        teacher = HistGradientBoostingClassifier(random_state=teacher_seed, **teacher_parameters)
        teacher.fit(fit_X, labels[fit])
        raw = np.asarray(teacher.predict_proba(holdout_X), dtype=float)
        classes = np.asarray(teacher.classes_)
        if (raw.shape != (len(holdout), len(LABELS)) or set(classes) != set(LABELS)
                or len(classes) != len(LABELS) or not np.isfinite(raw).all()
                or (raw < 0).any() or (raw > 1).any()
                or not np.allclose(raw.sum(axis=1), 1.0, rtol=1e-8, atol=1e-8)):
            raise RuntimeError("Cross-fit teacher returned invalid weather probabilities.")
        oof[holdout] = raw[:, [int(np.flatnonzero(classes == label)[0]) for label in LABELS]]
        coverage[holdout] += 1
        records.append({
            "fold": fold, "fit_rows": len(fit), "holdout_rows": len(holdout),
            "fit_index_sha256": _digest(fit, "<i8"),
            "holdout_index_sha256": _digest(holdout, "<i8"),
            "fit_group_sha256": _digest(np.sort(groups[fit]), "<u8"),
            "holdout_group_sha256": _digest(np.sort(groups[holdout]), "<u8"),
            "preprocessing_fitted_rows": features.fitted_rows_, "teacher_seed": teacher_seed,
        })
    if not np.all(coverage == 1) or not np.isfinite(oof).all():
        raise RuntimeError("Each allowed training row must receive exactly one teacher OOF prediction.")
    maximum = oof.max(axis=1)
    predicted = np.asarray(LABELS)[np.argmax(oof, axis=1)]
    wrong = predicted != labels
    conflict = wrong & (maximum > resolved["threshold"])
    weights = np.ones(len(labels))
    weights[conflict] = 1.0 - (1.0 - resolved["floor"]) * (
        (maximum[conflict] - resolved["threshold"]) / (1.0 - resolved["threshold"]))
    weights = np.clip(weights, resolved["floor"], 1.0)
    observed = oof[np.arange(len(labels)), labels - 1]
    metadata = {
        "method": "cross_fitted_confidence_weighting",
        "settings": resolved,
        "teacher_parameters": teacher_parameters,
        "split_method": split_method, "split_seed": split_seed, "folds": records,
        "oof_coverage": {"rows": len(labels), "minimum_predictions_per_row": int(coverage.min()),
                         "maximum_predictions_per_row": int(coverage.max()), "exactly_once": True},
        "wrong_argmax_rows": int(wrong.sum()),
        "high_confidence_conflict_rows": int(conflict.sum()),
        "quantile_levels": _QUANTILE_LEVELS.copy(),
        "observed_label_probability_quantiles": _quantiles(observed),
        "maximum_probability_quantiles": _quantiles(maximum),
        "weight_quantiles": _quantiles(weights), "mean_weight": float(weights.mean()),
        "weighting_seconds": perf_counter() - started,
        "original_rows_preserved": True, "original_labels_preserved": True,
        "scope": "Internal out-of-fold predictions use only the supplied training partition. High-confidence conflicts are not known label errors; no rows are removed, labels changed or outer-validation/test labels used.",
    }
    return weights, metadata
