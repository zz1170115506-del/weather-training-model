"""Export reviewed aggregate experiment results and figures, never row data.

Only complete candidate summaries from complete CV runs or explicitly stopped
CV runs are used. Development CV and nested outer-fold results stay separate.
The exporter refuses differing data/fold protocols and conflicting model names.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd


_LABEL_NAMES = ("Sunny", "Cloudy", "Rainy", "Snowy")
_SOURCE_FILES = ("train.csv", "test.csv", "sample_submission.csv")
_FOLD_COLUMNS = (
    "fold", "fit_rows", "validation_rows", "train_macro_f1", "macro_f1", "accuracy",
    "inference_seconds_including_preprocessing", "preprocessing_seconds", "fit_seconds",
    "training_seconds_including_inner_selection", "selected_iterations",
    "preprocessing_fitted_rows", "encoded_feature_count", "early_stopping_uses_evaluation_fold",
)
_COMPARISON_COLUMNS = (
    "name", "model_id", "fold_macro_f1_mean", "fold_macro_f1_std", "oof_macro_f1",
    "oof_accuracy", "total_cv_seconds", "mean_fit_seconds", "config_sha256", "run_id", "run_complete",
)


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _hash(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _number(value, name, lower=0, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name}: expected a finite number.")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name}: number is outside the accepted range.")
    return float(value)


def _public(value):
    """Strip machine path strings from otherwise useful configuration records."""
    if isinstance(value, dict):
        private = {"data_dir", "output_dir", "artifact_root", "cwd", "username",
                   "user", "password", "token", "api_key", "environment"}
        return {str(key): _public(item) for key, item in value.items()
                if str(key).lower() not in private}
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, str) and (re.match(r"^[A-Za-z]:[\\/]", value)
                                  or value.startswith(("/", "\\\\"))):
        return "<local path omitted>"
    return value


def _metrics(value):
    """Copy only class-level and aggregate metrics, with consistency checks."""
    score = _number(value["macro_f1"], "macro_f1", upper=1)
    accuracy = _number(value["accuracy"], "accuracy", upper=1)
    matrix = np.asarray(value["confusion_matrix"], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or (matrix < 0).any()
            or not np.equal(matrix, np.floor(matrix)).all() or matrix.sum() == 0):
        raise ValueError("Expected a nonempty 4x4 confusion matrix of counts.")
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    expected_f1 = np.divide(2 * matrix.diagonal(), denominator,
                            out=np.zeros(4), where=denominator > 0).mean()
    if not np.isclose(score, expected_f1, atol=1e-10, rtol=0):
        raise ValueError("Macro-F1 disagrees with the confusion matrix.")
    if not np.isclose(accuracy, matrix.trace() / matrix.sum(), atol=1e-10, rtol=0):
        raise ValueError("Accuracy disagrees with the confusion matrix.")
    report = {}
    for name in (*_LABEL_NAMES, "macro avg", "weighted avg"):
        if name in value.get("classification_report", {}):
            entry = value["classification_report"][name]
            report[name] = {
                field: _number(entry[field], f"{name}.{field}", upper=None if field == "support" else 1)
                for field in ("precision", "recall", "f1-score", "support")
            }
    return {"macro_f1": score, "accuracy": accuracy,
            "confusion_matrix": matrix.astype(np.int64).tolist(),
            "classification_report": report}


def _context(run_dir, run, nested=None):
    audit_path = run_dir / "audit.json"
    audit = _read(audit_path) if audit_path.is_file() else {}
    source = run.get("source_sha256") or audit.get("source_sha256")
    if not isinstance(source, dict) or not all(name in source for name in _SOURCE_FILES):
        raise ValueError(f"{run_dir.name}: source data hashes are missing.")
    source = {name: source[name] for name in _SOURCE_FILES}
    if any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
           for item in source.values()):
        raise ValueError(f"{run_dir.name}: invalid source data hash.")
    if run.get("source_sha256") and audit.get("source_sha256"):
        if any(run["source_sha256"].get(name) != audit["source_sha256"].get(name)
               for name in _SOURCE_FILES):
            raise ValueError(f"{run_dir.name}: run/audit source hashes disagree.")
    protocol_path = run_dir / "fold_protocol.json"
    protocol = (_read(protocol_path) if protocol_path.is_file()
                else (nested or {}).get("outer_protocol"))
    if protocol is None:
        protocol = run.get("result", {}).get("fold_protocol")
    required = {"method", "n_splits", "seed", "canonical_input_order",
                "fold_sha256", "probability_order"}
    if (not isinstance(protocol, dict) or not required.issubset(protocol)
            or protocol["probability_order"] != [1, 2, 3, 4]
            or not isinstance(protocol["n_splits"], int) or protocol["n_splits"] < 2
            or not re.fullmatch(r"[0-9a-f]{64}", str(protocol["fold_sha256"]))):
        raise ValueError(f"{run_dir.name}: incomplete or invalid fold protocol.")
    if nested and nested["outer_protocol"] != protocol:
        raise ValueError(f"{run_dir.name}: nested and recorded fold protocols disagree.")
    recorded_protocol = run.get("result", {}).get("fold_protocol") or run.get("result", {}).get("outer_protocol")
    if recorded_protocol is not None and recorded_protocol != protocol:
        raise ValueError(f"{run_dir.name}: manifest and file fold protocols disagree.")
    version_path = run_dir / "source_version.json"
    version = _read(version_path) if version_path.is_file() else {}
    code_hash = run.get("source_tree_sha256") or version.get("source_tree_sha256")
    return source, protocol, {
        "run_id": _public(run.get("run_id", run_dir.name)),
        "status": run.get("status"), "run_complete": run.get("status") == "complete",
        "timestamp_utc": run.get("timestamp_utc"),
        "git": {key: _public(run.get("git", {}).get(key)) for key in
                ("available", "commit", "dirty", "branch")},
        "source_tree_sha256": code_hash,
    }


def _candidate(summary_path, protocol, run_context):
    summary = _read(summary_path)
    name = summary.get("name")
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name)
            or name != summary_path.parent.name):
        raise ValueError("Candidate name must match its safe model directory name.")
    config = summary["config"]
    if config.get("name") != name or config.get("model_id") != summary.get("model_id"):
        raise ValueError(f"{name}: model identity and config disagree.")
    if _hash(config) != summary.get("config_sha256"):
        raise ValueError(f"{name}: config hash disagrees with the recorded config.")
    csv_path = summary_path.parent / "fold_metrics.csv"
    folds = pd.read_csv(csv_path)
    if not {"fold", "macro_f1", "fit_seconds"}.issubset(folds):
        raise ValueError(f"{name}: required fold metrics are missing.")
    folds = folds[[col for col in _FOLD_COLUMNS if col in folds]].copy()
    for col in folds:
        folds[col] = pd.to_numeric(folds[col], errors="raise")
        values = folds[col].to_numpy(dtype=float)
        if (np.isinf(values).any() or (col != "selected_iterations" and np.isnan(values).any())
                or np.any(values[np.isfinite(values)] < 0)):
            raise ValueError(f"{name}: invalid numeric fold metric {col}.")
    count = protocol["n_splits"]
    if len(folds) != count or sorted(folds.fold.tolist()) != list(range(count)):
        raise ValueError(f"{name}: candidate does not have exactly {count} complete folds.")
    folds = folds.sort_values("fold").reset_index(drop=True)
    if ((folds.macro_f1 < 0) | (folds.macro_f1 > 1)).any():
        raise ValueError(f"{name}: invalid fold F1.")
    mean = _number(summary["fold_macro_f1_mean"], f"{name}.mean", upper=1)
    std = _number(summary["fold_macro_f1_std"], f"{name}.std", upper=1)
    if (not np.isclose(mean, folds.macro_f1.mean(), rtol=0, atol=1e-10)
            or not np.isclose(std, folds.macro_f1.std(ddof=1), rtol=0, atol=1e-10)):
        raise ValueError(f"{name}: aggregate and CSV fold scores disagree.")
    oof = _metrics(summary["oof_metrics"])
    row = {"name": name, "model_id": summary["model_id"],
           "fold_macro_f1_mean": mean, "fold_macro_f1_std": std,
           "oof_macro_f1": oof["macro_f1"], "oof_accuracy": oof["accuracy"],
           "total_cv_seconds": _number(summary["total_cv_seconds"], f"{name}.cv_seconds"),
           "mean_fit_seconds": float(folds.fit_seconds.mean()),
           "config_sha256": summary["config_sha256"],
           "run_id": run_context["run_id"], "run_complete": run_context["run_complete"]}
    detail = {"config": _public(config), "config_sha256": summary["config_sha256"],
              "oof_metrics": oof, "source": run_context,
              "source_summary_sha256": _file_hash(summary_path),
              "source_fold_metrics_sha256": _file_hash(csv_path),
              "evaluation_scope": summary.get("evaluation_scope", "Development cross-validation")}
    return row, folds, detail


def _confusion_figure(plt, metric, title, path):
    counts = np.asarray(metric["confusion_matrix"])
    proportions = np.divide(counts, counts.sum(axis=1, keepdims=True),
                            out=np.zeros((4, 4), dtype=float), where=counts.sum(axis=1, keepdims=True) > 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, values, label in zip(axes, (counts, proportions), ("Record counts", "Within-class recall")):
        ax.imshow(values, cmap="Blues", vmin=0, vmax=None if label == "Record counts" else 1)
        ax.set(xticks=range(4), yticks=range(4), xticklabels=_LABEL_NAMES,
               yticklabels=_LABEL_NAMES, xlabel="Predicted", ylabel="Actual", title=label)
        for i in range(4):
            for j in range(4):
                text = str(int(values[i, j])) if label == "Record counts" else f"{values[i, j]:.1%}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if values[i, j] > values.max() / 2 else "black")
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _figures(output, comparison, details, nested):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "task1-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if not comparison.empty:
        ordered = comparison.sort_values("oof_macro_f1").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.32 * len(ordered) + 1.8)), constrained_layout=True)
        ax.errorbar(ordered.fold_macro_f1_mean, np.arange(len(ordered)),
                    xerr=ordered.fold_macro_f1_std, fmt="o", capsize=3, color="#337d85")
        ax.set(yticks=np.arange(len(ordered)), yticklabels=ordered.name,
               xlabel="Macro-F1: fold mean +/- sample standard deviation",
               title="Development cross-validation (used for model selection)")
        ax.grid(axis="x", alpha=0.2)
        fig.savefig(output / "comparison.png", dpi=170)
        plt.close(fig)
        best_name = comparison.iloc[0]["name"]
        _confusion_figure(plt, details[best_name]["oof_metrics"],
                          f"Best development OOF: {best_name} (not an independent test)",
                          output / "best_confusion.png")

        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        custom = comparison.model_id.str.startswith("numpy_")
        for mask, label, color in ((custom, "NumPy implementation", "#337d85"),
                                   (~custom, "Library candidate", "#b07847")):
            if mask.any():
                ax.scatter(comparison.loc[mask, "mean_fit_seconds"], comparison.loc[mask, "oof_macro_f1"],
                           label=label, color=color, alpha=0.85)
        for rank, (_, row) in enumerate(comparison.head(5).iterrows()):
            ax.annotate(row["name"], (row["mean_fit_seconds"], row["oof_macro_f1"]),
                        xytext=(5, 8 + 9 * (rank % 2)), textcoords="offset points", fontsize=8)
        if (comparison.mean_fit_seconds > 0).all():
            ax.set_xscale("log")
        ax.set(xlabel="Mean final model fit seconds per fold (log scale when positive)", ylabel="OOF Macro-F1",
               title="Development accuracy/time trade-off; preprocessing/inner selection excluded")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.savefig(output / "accuracy_time.png", dpi=170)
        plt.close(fig)

        # Tree-count ablations require every other recorded setting to match.
        tree_groups = {}
        for name, detail in details.items():
            config = json.loads(json.dumps(detail["config"]))
            trees = config.get("model_params", {}).pop("n_estimators", None)
            config.pop("name", None)
            if isinstance(trees, int) and not isinstance(trees, bool) and trees > 0:
                tree_groups.setdefault(_hash(config), []).append((trees, name))
        groups = [sorted(group) for group in tree_groups.values()
                  if len({trees for trees, _ in group}) > 1]
        if groups:
            fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
            for group in groups:
                rows = comparison.set_index("name").loc[[name for _, name in group]]
                ax.errorbar([trees for trees, _ in group], rows.fold_macro_f1_mean,
                            yerr=rows.fold_macro_f1_std, marker="o", capsize=3,
                            label=group[0][1])
            ax.set(xlabel="Tree count", ylabel="Fold mean Macro-F1 +/- sample std",
                   title="Tree-count ablation: other recorded configuration fields match")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            fig.savefig(output / "tree_count_ablation.png", dpi=170)
            plt.close(fig)

    if nested:
        labels, scores = [], []
        for i, entry in enumerate(nested, 1):
            for name, value in entry["ablations"].items():
                labels.append(f"Run {i}: {name}")
                scores.append(value["macro_f1"])
        if labels:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(labels) + 1)), constrained_layout=True)
            ax.plot(scores, np.arange(len(labels)), "o", color="#6b6196")
            ax.set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel="Outer OOF Macro-F1",
                   title="Fixed-candidate outer-fold results; prior search used these data")
            ax.grid(axis="x", alpha=0.2)
            fig.savefig(output / "nested_comparison.png", dpi=170)
            plt.close(fig)
        _confusion_figure(plt, nested[0]["oof_metrics"], "Inner-selected blend: outer-fold confusion",
                          output / "nested_blend_confusion.png")


def export_results(run_dirs, output_dir):
    """Export aggregate CSV/JSON/PNGs from comparable local experiment runs.

    The destination must not exist. Running/failed runs and incomplete candidate
    files are skipped with explicit notes. Stopped runs may contribute fully
    completed candidates and are marked ``run_complete=False``. Same-name
    records with different configs, protocols or aggregate scores are rejected.
    Identical aggregate reruns retain the first result and list all provenance.
    No raw records, OOF arrays, models, or full environment files are copied.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {output}")
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    rows, fold_frames, details, runs, nested, notes = [], [], {}, [], [], []
    common_source, common_protocol = None, None
    seen_paths = set()
    for raw_path in run_dirs:
        path = Path(raw_path).resolve()
        if path in seen_paths:
            raise ValueError("The same run directory was supplied more than once.")
        seen_paths.add(path)
        run = _read(path / "run.json")
        status = run.get("status")
        if status not in ("complete", "stopped_after_first_candidate"):
            notes.append({"run_id": _public(run.get("run_id", path.name)),
                          "reason": f"Run status {status!r} is not exportable."})
            continue
        nested_path = path / "nested_summary.json"
        nested_summary = _read(nested_path) if nested_path.is_file() else None
        candidate_paths = list(sorted((path / "models").glob("*/summary.json")))
        if not candidate_paths and nested_summary is None:
            notes.append({"run_id": _public(run.get("run_id", path.name)), "reason": "No candidate results."})
            continue
        source, protocol, context = _context(path, run, nested_summary)
        if common_source is not None and source != common_source:
            raise ValueError("Cannot compare runs with different source_sha256.")
        if common_protocol is not None and protocol != common_protocol:
            raise ValueError("Cannot compare runs with different fold_protocol or fold hash.")
        common_source, common_protocol = source, protocol
        runs.append(context)
        for summary_path in candidate_paths:
            if not (summary_path.parent / "fold_metrics.csv").is_file():
                notes.append({"run_id": context["run_id"], "candidate": summary_path.parent.name,
                              "reason": "Incomplete candidate: fold_metrics.csv missing."})
                continue
            row, folds, detail = _candidate(summary_path, protocol, context)
            name = row["name"]
            if name in details:
                previous = details[name]
                if (previous["config_sha256"] != detail["config_sha256"]
                        or previous["oof_metrics"] != detail["oof_metrics"]):
                    raise ValueError(f"Conflicting same-name candidate {name}; use distinct names or select one run.")
                previous.setdefault("additional_identical_aggregate_sources", []).append(detail["source"])
                notes.append({"candidate": name, "run_id": context["run_id"],
                              "reason": "Repeated identical config/OOF aggregate; first run supplies table timing."})
                continue
            rows.append(row)
            details[name] = detail
            folds.insert(0, "run_id", context["run_id"])
            folds.insert(0, "model_id", row["model_id"])
            folds.insert(0, "name", name)
            fold_frames.append(folds)
        if nested_summary is not None:
            resolved_path = path / "resolved_config.json"
            nested.append({"source": context, "inner_splits": nested_summary["inner_splits"],
                           "config": _public(_read(resolved_path)) if resolved_path.is_file() else None,
                           "outer_fold_macro_f1_mean": _number(nested_summary["outer_fold_macro_f1_mean"], "outer mean", upper=1),
                           "outer_fold_macro_f1_std": _number(nested_summary["outer_fold_macro_f1_std"], "outer std", upper=1),
                           "oof_metrics": _metrics(nested_summary["oof_metrics"]),
                           "ablations": {name: _metrics(value) for name, value in nested_summary["ablations"].items()},
                           "outer_folds": [{"fold": item["fold"], "metrics": _metrics(item["metrics"]),
                                            "recipe": _public(item.get("recipe"))}
                                           for item in nested_summary.get("folds", [])],
                           "evaluation_scope": nested_summary.get("evaluation_scope")})
    if not rows and not nested:
        raise ValueError("No complete comparable candidate or nested results were found.")
    comparison = pd.DataFrame(rows, columns=_COMPARISON_COLUMNS)
    if not comparison.empty:
        comparison = comparison.sort_values("oof_macro_f1", ascending=False, kind="stable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=False)
    comparison.to_csv(output / "comparison.csv", index=False, lineterminator="\n")
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame(columns=["name", "model_id", "run_id", "fold"])
    folds.to_csv(output / "fold_metrics.csv", index=False, lineterminator="\n")
    public = {"schema_version": 1, "labels": [1, 2, 3, 4], "label_names": list(_LABEL_NAMES),
              "source_sha256": common_source, "fold_protocol": common_protocol,
              "models": details, "runs": runs, "nested_evaluations": nested, "export_notes": notes,
              "development_scope": "CV used for model selection, not an independent test or Kaggle result.",
              "timing_scope": "Per-fold final model fit excludes preprocessing and inner iteration selection; total CV includes workflow costs. Cross-run speed comparisons require matching hardware."}
    (output / "model_details.json").write_text(json.dumps(public, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if nested:
        nested_rows = [{"run_id": entry["source"]["run_id"], "name": name,
                        "outer_oof_macro_f1": value["macro_f1"], "outer_oof_accuracy": value["accuracy"]}
                       for entry in nested for name, value in entry["ablations"].items()]
        pd.DataFrame(nested_rows).to_csv(output / "nested_comparison.csv", index=False, lineterminator="\n")
    _figures(output, comparison, details, nested)
    return {"output_dir": str(output), "candidate_count": len(rows), "nested_run_count": len(nested),
            "files": sorted(p.name for p in output.iterdir()), "notes": notes}
