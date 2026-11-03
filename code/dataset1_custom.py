"""Compare handwritten CART/forest with matched library references.

Model training/prediction for custom_* comes from weather_models (NumPy).
scikit-learn supplies preprocessing, splitting, metrics, and reference models.
This is an exploratory holdout comparison, not cross-validation or Kaggle.
"""

from __future__ import annotations

import argparse
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

# Import first to apply the shared thread limits before numerical libraries.
from dataset1_baseline import (
    ROOT, LABELS, LABEL_NAMES, audit_data, make_preprocessor, save_json,
)
from weather_models import CARTClassifier, RandomForestClassifier

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as ReferenceForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as ReferenceTree


def make_models(seed, trees):
    tree_parameters = dict(max_depth=8, min_samples_leaf=5, random_state=seed)
    forest_parameters = dict(n_estimators=trees, min_samples_leaf=2,
                             max_features="sqrt", oob_score=True, random_state=seed)
    return {
        "custom_cart": CARTClassifier(**tree_parameters),
        "sklearn_cart": ReferenceTree(**tree_parameters),
        "custom_forest": RandomForestClassifier(**forest_parameters),
        "sklearn_forest": ReferenceForest(**forest_parameters, n_jobs=1),
    }


def structural_summary(model):
    if isinstance(model, CARTClassifier):
        return {"nodes": model.node_count_, "leaves": model.n_leaves_, "depth": model.depth_}
    if isinstance(model, RandomForestClassifier):
        return {
            "trees": len(model.estimators_),
            "mean_nodes": float(np.mean([tree.node_count_ for tree in model.estimators_])),
            "mean_depth": float(np.mean([tree.depth_ for tree in model.estimators_])),
            "mean_unique_bootstrap_records": float(np.mean([len(np.unique(sample)) for sample in model.estimators_samples_])),
        }
    return None


def feature_sources(preprocessor, numeric, categorical):
    """Map each one-hot column to its original feature without parsing names."""
    sources = list(numeric)
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encode"]
    for column, categories in zip(categorical, encoder.categories_):
        sources.extend([column] * len(categories))
    return sources


def draw_comparison(metrics, details, importances, out):
    labels = {
        "custom_cart": "NumPy CART", "sklearn_cart": "sklearn CART",
        "custom_forest": "NumPy forest", "sklearn_forest": "sklearn forest",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    ranked = metrics.sort_values("validation_macro_f1")
    colors = ["#43868b" if name.startswith("custom") else "#9aa8b4" for name in ranked.model]
    axes[0].barh([labels[name] for name in ranked.model], ranked.validation_macro_f1, color=colors)
    axes[0].set(xlim=(0, 1.05), xlabel="Validation Macro-F1", title="Same split and matched parameters")
    for i, value in enumerate(ranked.validation_macro_f1):
        axes[0].text(value + 0.01, i, f"{value:.4f}", va="center")
    matrix = np.array(details["custom_forest"]["confusion_matrix"])
    axes[1].imshow(matrix, cmap="Blues")
    names = list(LABEL_NAMES.values())
    axes[1].set(xticks=range(4), yticks=range(4), xticklabels=names, yticklabels=names,
                xlabel="Predicted", ylabel="Actual", title="NumPy forest: validation confusion matrix")
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", va="center",
                         color="white" if matrix[i, j] > matrix.max()/2 else "black")
    fig.savefig(out / "custom_comparison.png", dpi=170)
    plt.close(fig)

    ranked = importances.loc[importances.model == "custom_forest"].sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.barh(ranked.original_feature, ranked.importance, color="#43868b")
    ax.set(xlabel="Normalized impurity reduction", title="NumPy forest: feature importance (training split)")
    fig.savefig(out / "feature_importance.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "comp-5434-20261-project-task-1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dataset1_custom")
    parser.add_argument("--seed", type=int, default=5434)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.validation_size < 1 or args.n_estimators < 1:
        parser.error("Require 0 < validation-size < 1 and n-estimators >= 1.")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    train, test, sample = [pd.read_csv(args.data_dir / name) for name in ("train.csv", "test.csv", "sample_submission.csv")]
    audit = audit_data(train, test, sample, args.data_dir)
    save_json(out / "audit.json", audit)
    X, y = train[audit["features"]], train.label.astype(int)
    fit, val = train_test_split(np.arange(len(train)), test_size=args.validation_size,
                              random_state=args.seed, stratify=y)
    split = pd.DataFrame({"id": train.id, "partition": "train"})
    split.loc[val, "partition"] = "validation"
    split.to_csv(out / "validation_split.csv", index=False)
    baseline_split_path = ROOT / "outputs/dataset1_baseline/validation_split.csv"
    baseline_split_matches = None
    if baseline_split_path.exists():
        baseline_split_matches = split.equals(pd.read_csv(baseline_split_path))

    prep_started = perf_counter()
    preprocessor = make_preprocessor(audit["numeric_features"], audit["categorical_features"])
    X_fit = preprocessor.fit_transform(X.iloc[fit])
    X_val = preprocessor.transform(X.iloc[val])
    preprocessing_seconds = perf_counter() - prep_started
    y_fit, y_val = y.iloc[fit].to_numpy(), y.iloc[val].to_numpy()
    models = make_models(args.seed, args.n_estimators)
    records, model_details, predictions, importance_frames = [], {}, [], []
    for name, model in models.items():
        fit_started = perf_counter()
        model.fit(X_fit, y_fit)
        fit_seconds = perf_counter() - fit_started
        inference_started = perf_counter()
        probabilities = model.predict_proba(X_val)
        predicted = model.classes_[np.argmax(probabilities, axis=1)]
        inference_seconds = perf_counter() - inference_started
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
        row = {
            "model": name,
            "train_macro_f1": f1_score(y_fit, model.predict(X_fit), labels=LABELS, average="macro", zero_division=0),
            "validation_macro_f1": f1_score(y_val, predicted, labels=LABELS, average="macro", zero_division=0),
            "validation_accuracy": accuracy_score(y_val, predicted),
            "fit_seconds_model_only": fit_seconds,
            "preprocessing_seconds_shared": preprocessing_seconds,
            "validation_inference_seconds_model_only": inference_seconds,
        }
        detail = {
            "implementation": "NumPy" if name.startswith("custom") else "scikit-learn reference",
            "parameters": model.get_params(), "structure": structural_summary(model),
            "classification_report": classification_report(y_val, predicted, labels=LABELS,
                target_names=list(LABEL_NAMES.values()), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_val, predicted, labels=LABELS).tolist(),
        }
        if hasattr(model, "oob_decision_function_"):
            oob = model.oob_decision_function_
            covered = np.isfinite(oob).all(axis=1) & (oob.sum(axis=1) > 0)
            detail["oob"] = {
                "coverage": float(covered.mean()), "accuracy": float(model.oob_score_),
                "macro_f1": f1_score(y_fit[covered], model.classes_[np.argmax(oob[covered], axis=1)],
                                      labels=LABELS, average="macro", zero_division=0),
            }
        model_details[name] = detail
        records.append(row)
        frame = pd.DataFrame({"model": name, "id": train.iloc[val].id.to_numpy(),
                              "actual": y_val, "predicted": predicted})
        for j, label in enumerate(model.classes_):
            frame[f"p_{label}"] = probabilities[:, j]
        predictions.append(frame)
        importance_frames.append(pd.DataFrame({
            "model": name, "encoded_feature": preprocessor.get_feature_names_out(),
            "original_feature": feature_sources(preprocessor, audit["numeric_features"], audit["categorical_features"]),
            "importance": model.feature_importances_,
        }))
        print(f"{name}: validation Macro-F1={row['validation_macro_f1']:.6f}; fit={fit_seconds:.3f}s", flush=True)

    metrics = pd.DataFrame(records).sort_values("validation_macro_f1", ascending=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False)
    importances = pd.concat(importance_frames, ignore_index=True)
    importances.to_csv(out / "feature_importances.csv", index=False)
    grouped = importances.groupby(["model", "original_feature"], as_index=False).importance.sum()
    grouped.to_csv(out / "original_feature_importances.csv", index=False)

    best_name = metrics.loc[metrics.model.str.startswith("custom")].iloc[0].model
    parameters = models[best_name].get_params()
    if "oob_score" in parameters:
        parameters["oob_score"] = False  # Diagnostic only; does not change fitted predictions.
    full_started = perf_counter()
    full_preprocessor = make_preprocessor(audit["numeric_features"], audit["categorical_features"])
    X_full = full_preprocessor.fit_transform(X)
    best_model = type(models[best_name])(**parameters).fit(X_full, y.to_numpy())
    full_fit_seconds = perf_counter() - full_started
    inference_started = perf_counter()
    X_test = full_preprocessor.transform(test[audit["features"]])
    probabilities = best_model.predict_proba(X_test)
    test_predicted = best_model.classes_[np.argmax(probabilities, axis=1)]
    full_inference_seconds = perf_counter() - inference_started
    prediction = pd.DataFrame({"id": test.id, "label": test_predicted}).set_index("id").loc[sample.id].reset_index()
    assert len(prediction) == len(sample) and prediction.id.equals(sample.id)
    assert set(prediction.label).issubset(LABELS) and not prediction.label.isna().any()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    prediction.to_csv(out / "custom_predictions_integer.csv", index=False)
    probability_frame = pd.DataFrame(probabilities, columns=[f"p_{k}" for k in best_model.classes_])
    probability_frame.insert(0, "id", test.id.to_numpy())
    probability_frame.to_csv(out / "custom_probabilities.csv", index=False)
    bundle = {"preprocessor": full_preprocessor, "model": best_model, "features": audit["features"], "label_names": LABEL_NAMES}
    joblib.dump(bundle, out / "custom_model.joblib", compress=3)
    loaded = joblib.load(out / "custom_model.joblib")
    np.testing.assert_array_equal(probabilities, loaded["model"].predict_proba(loaded["preprocessor"].transform(test[loaded["features"]])))

    draw_comparison(metrics, model_details, grouped, out)
    save_json(out / "details.json", {
        "seed": args.seed, "validation_fraction": args.validation_size,
        "training_rows": len(fit), "validation_rows": len(val),
        "validation_class_counts": {str(k): int(v) for k, v in y.iloc[val].value_counts().sort_index().items()},
        "same_partition_as_saved_baseline": baseline_split_matches,
        "preprocessing_fitted_on_validation": False, "id_used_as_feature": False,
        "encoded_feature_count": X_fit.shape[1], "preprocessing_seconds_shared": preprocessing_seconds,
        "evaluation_scope": "Exploratory single holdout; library references use matched parameters and shared preprocessing. Different RNG/tie-breaking policies can produce different trees.",
        "models": model_details, "best_custom_model": best_name,
        "full_refit_parameters": parameters,
        "full_training_fit_seconds_including_preprocessing": full_fit_seconds,
        "full_test_inference_seconds_including_preprocessing": full_inference_seconds,
        "training_epochs": "not applicable; greedy trees and bootstrap ensemble",
        "saved_model_reload_exact_match": True,
        "total_workflow_seconds": perf_counter() - started,
        "source_sha256": audit["source_sha256"],
        "submission_encoding_status": audit["submission_encoding_status"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "training_threads": 1, "gpu": "not used",
                        "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "joblib")}},
    })
    print(f"Best handwritten model: {best_name}; outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
