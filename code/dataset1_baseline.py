"""Audit Dataset 1 and run reproducible exploratory classification baselines.

This is an initial benchmark, not the project's final autonomous algorithm.
Run with --help for options. Requires the packages in requirements.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

# Bound numerical-library parallelism before importing numpy or sklearn.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_info


ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = {1: "Sunny", 2: "Cloudy", 3: "Rainy", 4: "Snowy"}
LABELS = list(LABEL_NAMES)


def save_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cpu_name():
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def audit_data(train, test, sample, data_dir):
    """Check schema/IDs and describe anomalies without silently altering data."""
    if not {"id", "label"}.issubset(train.columns):
        raise ValueError("Training data must contain id and label columns.")
    features = [c for c in train.columns if c not in ("id", "label")]
    if set(test.columns) != {"id", *features}:
        raise ValueError("Train/test feature columns differ.")
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("Expected sample submission columns: id,label.")
    for name, frame in (("train", train), ("test", test), ("sample", sample)):
        if frame["id"].isna().any() or not frame["id"].is_unique:
            raise ValueError(f"{name}: record IDs must be present and unique.")
    if set(sample["id"]) != set(test["id"]):
        raise ValueError("Sample submission IDs do not match test IDs.")
    if train["label"].isna().any() or set(train["label"]) != set(LABELS):
        raise ValueError("Expected all four training labels: 1,2,3,4.")

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    for frame in (train, test):
        for col in numeric:
            # Invalid numeric strings must surface rather than becoming missing.
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            if np.isinf(frame[col].to_numpy(dtype=float)).any():
                raise ValueError(f"{col}: infinite feature value.")

    # A shared feature signature treats numeric train/test representations alike.
    keys = pd.util.hash_pandas_object(pd.concat([train[features], test[features]], ignore_index=True), index=False)
    train_keys, test_keys = keys.iloc[:len(train)], keys.iloc[len(train):]
    conflicting = train.assign(_signature=train_keys.to_numpy()).groupby("_signature")["label"].nunique()
    if train_keys.duplicated().any():
        raise ValueError("Duplicate feature rows found: use a group-aware split before benchmarking.")

    percentage_checks = {}
    for name, frame in (("train", train), ("test", test)):
        percentage_checks[name] = {}
        for col in ("Humidity", "Precipitation (%)"):
            if col in numeric:
                invalid = (frame[col] < 0) | (frame[col] > 100)
                record = {"count": int(invalid.sum()), "fraction": float(invalid.mean())}
                if name == "train":
                    record["count_by_label"] = {str(k): int(v) for k, v in frame.loc[invalid, "label"].value_counts().sort_index().items()}
                percentage_checks[name][col] = record

    stats = {}
    for name, frame in (("train", train), ("test", test)):
        stats[name] = {}
        for col in numeric:
            series = frame[col]
            stats[name][col] = {
                "min": float(series.min()), "max": float(series.max()),
                "mean": float(series.mean()), "std": float(series.std()),
                "median": float(series.median()),
            }

    return {
        "rows": {"train": len(train), "test": len(test), "sample": len(sample)},
        "features": features, "numeric_features": numeric, "categorical_features": categorical,
        "label_mapping": LABEL_NAMES,
        "class_counts": {str(k): int(v) for k, v in train["label"].value_counts().sort_index().items()},
        "missing_counts": {name: {c: int(v) for c, v in frame.isna().sum().items()} for name, frame in (("train", train), ("test", test), ("sample", sample))},
        "duplicate_feature_rows_train": int(train_keys.duplicated().sum()),
        "conflicting_feature_groups_train": int((conflicting > 1).sum()),
        "train_test_feature_overlap_rows": int(test_keys.isin(set(train_keys)).sum()),
        "train_test_id_overlap": len(set(train["id"]) & set(test["id"])),
        "sample_ids_same_order_as_test": sample["id"].equals(test["id"]),
        "category_counts": {name: {c: {str(k): int(v) for k, v in frame[c].value_counts().items()} for c in categorical} for name, frame in (("train", train), ("test", test))},
        "unseen_test_categories": {c: sorted(set(test[c].dropna()) - set(train[c].dropna())) for c in categorical},
        "numeric_statistics": stats,
        "values_outside_percentage_range": percentage_checks,
        "sample_label_values": [str(v) for v in sample["label"].unique()],
        "submission_encoding_status": "UNCONFIRMED: assignment/train use integers 1..4; sample uses weather names.",
        "data_policy": "Retain source rows and out-of-range values; exclude id; fit preprocessing on training partition only.",
        "source_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in ("train.csv", "test.csv", "sample_submission.csv")},
    }


def make_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def make_models(seed):
    # Deliberately small benchmark set; all configurations are recorded in JSON.
    return {
        "majority_reference": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", n_jobs=1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=seed),
    }


def draw_figures(audit, metrics, details, best_name, output_dir, validation_fraction):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    counts = [audit["class_counts"][str(k)] for k in LABELS]
    axes[0].bar(list(LABEL_NAMES.values()), counts, color="#43868b")
    axes[0].set(title="Training class distribution", ylabel="Records")
    for i, count in enumerate(counts):
        axes[0].text(i, count + 25, str(count), ha="center")
    columns = ["Humidity", "Precipitation (%)"]
    for offset, name in ((-0.18, "train"), (0.18, "test")):
        values = [100 * audit["values_outside_percentage_range"][name][c]["fraction"] for c in columns]
        axes[1].bar(np.arange(2) + offset, values, width=0.36, label=name)
    axes[1].set(xticks=np.arange(2), xticklabels=columns, ylabel="Records (%)", title="Percentage values outside [0, 100]")
    axes[1].legend(frameon=False)
    fig.savefig(output_dir / "data_overview.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ranked = metrics.sort_values("validation_macro_f1")
    axes[0].barh(ranked["model"], ranked["validation_macro_f1"], color="#43868b")
    split_label = f"{100 * (1 - validation_fraction):g}/{100 * validation_fraction:g}"
    axes[0].set(xlim=(0, 1.06), xlabel="Macro-F1", title=f"One stratified {split_label} split (exploratory)")
    for i, value in enumerate(ranked["validation_macro_f1"]):
        axes[0].text(value + 0.01, i, f"{value:.4f}", va="center")
    matrix = np.array(details[best_name]["confusion_matrix"])
    axes[1].imshow(matrix, cmap="Blues")
    names = list(LABEL_NAMES.values())
    axes[1].set(xticks=range(4), yticks=range(4), xticklabels=names, yticklabels=names, xlabel="Predicted", ylabel="Actual", title=f"Validation: {best_name}")
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black")
    fig.savefig(output_dir / "baseline_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "comp-5434-20261-project-task-1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dataset1_baseline")
    parser.add_argument("--seed", type=int, default=5434)
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size must be between zero and one.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    train, test, sample = [pd.read_csv(args.data_dir / name) for name in ("train.csv", "test.csv", "sample_submission.csv")]
    audit = audit_data(train, test, sample, args.data_dir)
    save_json(output_dir / "audit.json", audit)
    save_json(output_dir / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": cpu_name(), "logical_cpu_count": os.cpu_count(), "servers": 1,
        "gpu": "not used", "model_n_jobs": 1,
        "threadpools": threadpool_info(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")},
    })
    features = audit["features"]
    X, y = train[features], train["label"].astype(int)
    fit_idx, val_idx = train_test_split(np.arange(len(train)), test_size=args.validation_size, random_state=args.seed, stratify=y)
    split = pd.DataFrame({"id": train["id"], "partition": "train"})
    split.loc[val_idx, "partition"] = "validation"
    split.to_csv(output_dir / "validation_split.csv", index=False)

    preprocessor = make_preprocessor(audit["numeric_features"], audit["categorical_features"])
    models = make_models(args.seed)
    rows, details, validation_predictions = [], {}, []
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
        fit_started = perf_counter()
        pipeline.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        fit_seconds = perf_counter() - fit_started
        predict_started = perf_counter()
        probabilities = pipeline.predict_proba(X.iloc[val_idx])
        pred = pipeline.classes_[np.argmax(probabilities, axis=1)]
        predict_seconds = perf_counter() - predict_started
        row = {
            "model": name,
            "train_macro_f1": f1_score(y.iloc[fit_idx], pipeline.predict(X.iloc[fit_idx]), labels=LABELS, average="macro", zero_division=0),
            "validation_macro_f1": f1_score(y.iloc[val_idx], pred, labels=LABELS, average="macro", zero_division=0),
            "validation_accuracy": accuracy_score(y.iloc[val_idx], pred),
            "fit_seconds_including_preprocessing": fit_seconds,
            "validation_inference_seconds": predict_seconds,
        }
        rows.append(row)
        details[name] = {
            "parameters": model.get_params(),
            "classification_report": classification_report(y.iloc[val_idx], pred, labels=LABELS, target_names=list(LABEL_NAMES.values()), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y.iloc[val_idx], pred, labels=LABELS).tolist(),
        }
        prediction_frame = pd.DataFrame({"model": name, "id": train.iloc[val_idx]["id"].to_numpy(), "actual": y.iloc[val_idx].to_numpy(), "predicted": pred})
        for j, label in enumerate(pipeline.classes_):
            prediction_frame[f"p_{label}"] = probabilities[:, j]
        validation_predictions.append(prediction_frame)
        print(f"{name}: validation Macro-F1={row['validation_macro_f1']:.6f}, fit={fit_seconds:.3f}s", flush=True)

    metrics = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(validation_predictions, ignore_index=True).to_csv(output_dir / "validation_predictions.csv", index=False)
    best_name = metrics.iloc[0]["model"]
    best = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(models[best_name]))])
    refit_started = perf_counter()
    best.fit(X, y)
    refit_seconds = perf_counter() - refit_started
    infer_started = perf_counter()
    test_probs = best.predict_proba(test[features])
    test_pred = best.classes_[np.argmax(test_probs, axis=1)]
    inference_seconds = perf_counter() - infer_started
    prediction = pd.DataFrame({"id": test["id"], "label": test_pred}).set_index("id").loc[sample["id"]].reset_index()
    if len(prediction) != len(sample) or prediction["label"].isna().any() or not set(prediction["label"]).issubset(LABELS):
        raise ValueError("Prediction output failed row-count/label checks.")
    prediction.to_csv(output_dir / "baseline_predictions_integer.csv", index=False)
    probability_frame = pd.DataFrame(test_probs, columns=[f"p_{k}" for k in best.classes_])
    probability_frame.insert(0, "id", test["id"].to_numpy())
    probability_frame.to_csv(output_dir / "baseline_probabilities.csv", index=False)
    draw_figures(audit, metrics, details, best_name, output_dir, args.validation_size)
    save_json(output_dir / "details.json", {
        "seed": args.seed, "validation_fraction": args.validation_size,
        "training_rows": len(fit_idx), "validation_rows": len(val_idx),
        "validation_class_counts": {str(k): int(v) for k,v in y.iloc[val_idx].value_counts().sort_index().items()},
        "id_used_as_feature": False, "preprocessing_fitted_on_validation": False,
        "evaluation_scope": "Exploratory model selection on one split; not an independent final test or Kaggle score.",
        "models": details, "best_model": best_name,
        "full_training_fit_seconds_including_preprocessing": refit_seconds,
        "full_test_inference_seconds": inference_seconds,
        "full_training_epochs": "not applicable to this tree-based baseline",
        "total_exploratory_workflow_seconds": perf_counter() - started,
        "prediction_label_counts": {str(k): int(v) for k,v in prediction["label"].value_counts().sort_index().items()},
        "submission_label_encoding": "Integer labels per assignment; confirm discrepancy with sample's English label before submitting.",
    })
    print(f"Best exploratory model: {best_name}. Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
