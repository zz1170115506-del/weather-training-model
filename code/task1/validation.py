"""Shared, group-aware folds and honest out-of-fold experiment metrics."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from . import LABELS, LABEL_NAMES
from .artifacts import content_hash, save_json
from .data import CATEGORICAL
from .models import fit_model, predict_probabilities


def metrics(y, probabilities):
    prediction = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
    return {"macro_f1": float(f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, prediction)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=LABELS).tolist(),
            "classification_report": classification_report(
                y, prediction, labels=LABELS, target_names=list(LABEL_NAMES.values()),
                output_dict=True, zero_division=0)}


def make_folds(y, ids, groups, n_splits=5, seed=5434):
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    y, ids, groups = np.asarray(y), np.asarray(ids), np.asarray(groups)
    if len(y) != len(ids) or len(groups) != len(y) or len(np.unique(ids)) != len(ids):
        raise ValueError("Fold inputs must have equal length and unique IDs.")
    if min(int((y == k).sum()) for k in LABELS) < n_splits:
        raise ValueError("Each class needs at least n_splits records.")
    grouped = len(np.unique(groups)) != len(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Need at least n_splits distinct groups.")
    splitter = (StratifiedGroupKFold if grouped else StratifiedKFold)(
        n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    for fold, (fit, valid) in enumerate(splitter.split(np.zeros((len(y), 1)), y,
                                                       groups if grouped else None)):
        if set(ids[fit]) & set(ids[valid]):
            raise RuntimeError("Training/validation IDs overlap.")
        if grouped and set(groups[fit]) & set(groups[valid]):
            raise RuntimeError("Duplicate-feature groups cross a fold boundary.")
        if np.any(folds[valid] >= 0):
            raise RuntimeError("Repeated OOF assignment.")
        folds[valid] = fold
    if np.any(folds < 0):
        raise RuntimeError("Missing OOF assignment.")
    metadata = {"method": type(splitter).__name__, "n_splits": n_splits,
                "seed": seed, "canonical_input_order": "ascending record id",
                "fold_sha256": content_hash(list(zip(ids.tolist(), folds.tolist()))),
                "probability_order": list(LABELS)}
    return folds, metadata


def group_diagnostics(data, probabilities):
    y = data.y
    rows = []
    for col in CATEGORICAL:
        values = data.train[col].fillna("__MISSING__").astype(str)
        for value in sorted(values.unique()):
            mask = (values == value).to_numpy()
            rows.append({"attribute": col, "value": value, "rows": int(mask.sum()),
                         "macro_f1": metrics(y[mask], probabilities[mask])["macro_f1"]})
    for col in ("Humidity", "Precipitation (%)"):
        anomaly = ((data.train[col] < 0) | (data.train[col] > 100)).to_numpy()
        for name, mask in (("outside", anomaly), ("within", ~anomaly)):
            if mask.any():
                rows.append({"attribute": col, "value": name, "rows": int(mask.sum()),
                             "macro_f1": metrics(y[mask], probabilities[mask])["macro_f1"]})
    return pd.DataFrame(rows)


def cross_validate(data, config, folds, model_seed=5434, out=None, verbose=True):
    X, y = data.X, data.y
    folds = np.asarray(folds)
    if folds.shape != y.shape or len(np.unique(folds)) < 2 or set(np.unique(folds)) != set(range(len(np.unique(folds)))):
        raise ValueError("Expected contiguous nonnegative folds for every training row.")
    oof = np.full((len(y), 4), np.nan)
    seen = np.zeros(len(y), dtype=int)
    records, details, importance_rows = [], [], []
    started = perf_counter()
    for fold in sorted(np.unique(folds)):
        fit, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        bundle, fit_info = fit_model(X.iloc[fit], y[fit], config, model_seed + int(fold))
        infer_started = perf_counter()
        probability = predict_probabilities(bundle, X.iloc[valid])
        inference_seconds = perf_counter() - infer_started
        oof[valid] = probability
        seen[valid] += 1
        result = metrics(y[valid], probability)
        train_result = metrics(y[fit], predict_probabilities(bundle, X.iloc[fit]))
        row = {"fold": int(fold), "fit_rows": len(fit), "validation_rows": len(valid),
               "train_macro_f1": train_result["macro_f1"], "macro_f1": result["macro_f1"],
               "accuracy": result["accuracy"], "inference_seconds_including_preprocessing": inference_seconds,
               **fit_info}
        records.append(row)
        details.append({**row, "metrics": result,
                        "fit_ids_sha256": content_hash(data.train.id.iloc[fit].tolist()),
                        "validation_ids_sha256": content_hash(data.train.id.iloc[valid].tolist())})
        model = bundle["model"]
        if hasattr(model, "feature_importances_"):
            for group, value in zip(bundle["features"].groups_, model.feature_importances_):
                importance_rows.append({"fold": int(fold), "feature": str(group), "importance": float(value)})
        if verbose:
            print(f"{config['name']}: fold {fold+1}/{len(np.unique(folds))} Macro-F1={row['macro_f1']:.6f}", flush=True)
    if not np.all(seen == 1) or not np.isfinite(oof).all():
        raise RuntimeError("Each training record must receive exactly one finite OOF prediction.")
    scores = [r["macro_f1"] for r in records]
    summary = {"name": config["name"], "model_id": config["model_id"],
               "config": config, "config_sha256": content_hash(config),
               "fold_macro_f1_mean": float(np.mean(scores)),
               "fold_macro_f1_std": float(np.std(scores, ddof=1)),
               "oof_metrics": metrics(y, oof), "folds": details,
               "total_cv_seconds": perf_counter() - started,
               "evaluation_scope": "Development cross-validation used for model selection; not an independent test or Kaggle score."}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        save_json(out / "summary.json", summary)
        pd.DataFrame(records).to_csv(out / "fold_metrics.csv", index=False, lineterminator="\n")
        np.savez_compressed(out / "oof.npz", ids=data.train.id.to_numpy(), y=y, folds=folds, probabilities=oof)
        group_diagnostics(data, oof).to_csv(out / "group_metrics.csv", index=False, lineterminator="\n")
        if importance_rows:
            importance = pd.DataFrame(importance_rows).groupby(["fold", "feature"], as_index=False).importance.sum()
            importance.to_csv(out / "feature_importance.csv", index=False, lineterminator="\n")
    return summary, oof
