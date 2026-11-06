"""Inspect already-fitted boosting stages; never fit, stop, or select a model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from . import LABELS, LABEL_NAMES
from .models import align_probabilities

CURVE_SCOPE = (
    "Post-fit development-fold diagnostics only. All configured iterations are retained; "
    "these curves do not select iterations or provide an independent evaluation."
)


def boosting_curve(bundle, X_fit, y_fit, X_valid, y_valid, every=10):
    """Measure stages of a single fitted HGB using its existing fold transformer."""
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError("learning_curve_every must be a positive integer.")
    model = bundle["model"]
    total = int(model.n_iter_)
    if total < 1:
        raise ValueError("Learning curves require a fitted boosting model.")
    fit_values = bundle["features"].transform(X_fit)
    valid_values = bundle["features"].transform(X_valid)
    checkpoints = {1, total, *range(every, total + 1, every)}
    fit_stages = model.staged_predict_proba(fit_values)
    valid_stages = model.staged_predict_proba(valid_values)
    rows = []
    for iteration in range(1, total + 1):
        try:
            train = next(fit_stages)
            valid = next(valid_stages)
        except StopIteration as error:
            raise RuntimeError("Staged predictions ended before the fitted iteration count.") from error
        if iteration not in checkpoints:
            continue
        train = align_probabilities(train, model.classes_)
        valid = align_probabilities(valid, model.classes_)
        train_prediction = np.asarray(LABELS)[train.argmax(axis=1)]
        valid_prediction = np.asarray(LABELS)[valid.argmax(axis=1)]
        per_class = f1_score(y_valid, valid_prediction, labels=LABELS, average=None, zero_division=0)
        rows.append({
            "iteration": iteration,
            "train_macro_f1": float(f1_score(y_fit, train_prediction, labels=LABELS, average="macro", zero_division=0)),
            "validation_macro_f1": float(per_class.mean()),
            "train_accuracy": float(accuracy_score(y_fit, train_prediction)),
            "validation_accuracy": float(accuracy_score(y_valid, valid_prediction)),
            "train_log_loss": float(log_loss(y_fit, train, labels=LABELS)),
            "validation_log_loss": float(log_loss(y_valid, valid, labels=LABELS)),
            **{f"validation_f1_{LABEL_NAMES[label].lower()}": float(value)
               for label, value in zip(LABELS, per_class)},
        })
    for stages in (fit_stages, valid_stages):
        if next(stages, None) is not None:
            raise RuntimeError("Staged predictions exceeded the fitted iteration count.")
    return rows


def save_boosting_curves(rows, output_dir):
    """Save fold-level aggregate metrics and an explicitly descriptive figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "learning_curve.csv", index=False, lineterminator="\n")
    curve = frame.groupby("iteration")[["train_macro_f1", "validation_macro_f1"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for column, label in (("train_macro_f1", "Training"), ("validation_macro_f1", "Development validation")):
        mean = curve[(column, "mean")]
        std = curve[(column, "std")].fillna(0)
        line, = ax.plot(curve.index, mean, label=label)
        ax.fill_between(curve.index, (mean - std).clip(0, 1), (mean + std).clip(0, 1),
                        alpha=0.15, color=line.get_color())
    ax.set(xlabel="Boosting iteration (no automatic selection)", ylabel="Macro-F1",
           title="Fold mean and standard deviation; development diagnostics")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "learning_curve.png", dpi=150)
    plt.close(fig)
    return {"csv": "learning_curve.csv", "figure": "learning_curve.png", "scope": CURVE_SCOPE}
