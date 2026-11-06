"""Read completed user-started training runs without fitting or changing files."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath
import re

import pandas as pd

from .submission import validate_submission_file


LOCAL_NOTICE = "本地交叉验证（CV）结果，不是 Kaggle 公榜或私榜成绩。"
_HEX = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
_METRICS = ("fold_macro_f1_mean", "fold_macro_f1_std", "oof_macro_f1")


def _within(root, relative):
    if (not isinstance(relative, str) or not relative or "\x00" in relative
            or PureWindowsPath(relative).drive or relative.startswith(("/", "\\"))):
        raise ValueError("Artifact paths must be relative paths inside the training run.")
    normalized = relative.replace("\\", "/")
    if ".." in normalized.split("/"):
        raise ValueError("Artifact path traversal outside the training run is forbidden.")
    path = (root / normalized).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Artifact path escapes the training run (possibly through a link).") from error
    return path


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _sha(value, context):
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError(f"{context}: expected a lowercase SHA256 digest.")
    return value


def _bytes(root, relative, manifest):
    path = _within(root, relative)
    content = path.read_bytes()
    expected = manifest.get("files_sha256", {}).get(relative.replace("\\", "/"))
    if expected is not None and _hash(content) != _sha(expected, relative):
        raise ValueError(f"Artifact SHA256 mismatch: {relative}")
    return content


def _json(root, relative, manifest):
    try:
        return json.loads(_bytes(root, relative, manifest).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON artifact: {relative}") from error


def _number(value, context):
    if isinstance(value, bool):
        raise ValueError(f"{context}: expected a finite metric in [0, 1].")
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{context}: expected a finite metric in [0, 1].") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{context}: expected a finite metric in [0, 1].")
    return result


def _count(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context}: expected a positive integer row count.")
    return value


def _protocol(value):
    if not isinstance(value, dict):
        raise ValueError("Missing fold protocol.")
    _sha(value.get("fold_sha256"), "fold protocol")
    if _count(value.get("n_splits"), "n_splits") < 2:
        raise ValueError("n_splits must be at least two.")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Fold seed must be a nonnegative integer.")
    return value


def _data_hashes(value):
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("Run is missing source data SHA256 hashes.")
    return {name: _sha(digest, f"data {name}") for name, digest in value.items()}


def _metrics(root, summary, manifest):
    comparison = summary.get("comparison")
    if not isinstance(comparison, dict) or not isinstance(comparison.get("metrics"), list):
        raise ValueError("Training summary is missing its comparison metrics.")
    declared = {}
    for item in comparison["metrics"]:
        if not isinstance(item, dict) or item.get("name") in declared:
            raise ValueError("Training summary has invalid or repeated model names.")
        declared[item.get("name")] = item
    try:
        reader = csv.DictReader(io.StringIO(_bytes(root, "comparison.csv", manifest).decode("utf-8")))
        required = {"name", "model_id", *_METRICS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("comparison.csv is missing required metric columns.")
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("Invalid comparison.csv.") from error
    models = []
    seen = set()
    for row in rows:
        name = row.get("name", "")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None or name in seen:
            raise ValueError("comparison.csv has invalid or repeated model names.")
        seen.add(name)
        if name not in declared or row.get("model_id") != declared[name].get("model_id"):
            raise ValueError("comparison.csv and training_summary.json model identities differ.")
        metrics = {key: _number(row.get(key), f"{name}.{key}") for key in _METRICS}
        for key, value in metrics.items():
            if not math.isclose(value, _number(declared[name].get(key), f"summary {name}.{key}"), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"comparison.csv and summary disagree on {name}.{key}.")
        # Accuracy is not present in the historical comparison.csv. Read only
        # its small model summary, never the estimator, OOF archive, or data.
        details = _json(root, f"models/{name}/summary.json", manifest)
        if details.get("name") != name or details.get("model_id") != row["model_id"]:
            raise ValueError(f"Model summary identity mismatch: {name}")
        if not math.isclose(_number(details.get("oof_metrics", {}).get("macro_f1"), name), metrics["oof_macro_f1"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Model OOF summary disagrees with comparison: {name}")
        accuracy = _number(details.get("oof_metrics", {}).get("accuracy"), f"{name}.accuracy")
        models.append({"name": name, "model_id": row["model_id"], **metrics, "accuracy": accuracy})
    if not models or seen != set(declared):
        raise ValueError("comparison.csv and training summary must contain the same nonempty model set.")
    return models


def _checked_csv(root, relative, metadata, manifest):
    if not isinstance(metadata, dict) or metadata.get("label_format") not in ("integer", "name"):
        raise ValueError("Candidate CSV manifest must declare integer or name label format.")
    expected_rows = _count(metadata.get("rows"), "Candidate CSV")
    content = _bytes(root, relative, manifest)
    if _hash(content) != _sha(metadata.get("csv_sha256"), "Candidate CSV"):
        raise ValueError(f"Candidate CSV SHA256 mismatch: {relative}")
    try:
        records = list(csv.reader(io.StringIO(content.decode("utf-8"), newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"Invalid candidate CSV: {relative}") from error
    if not records or records[0] != ["id", "label"] or any(len(row) != 2 for row in records[1:]):
        raise ValueError(f"Candidate CSV schema must be exactly id,label: {relative}")
    if len(records) - 1 != expected_rows:
        raise ValueError(f"Candidate CSV row count differs from its manifest: {relative}")
    if metadata.get("id_type") == "integer":
        try:
            ids = [int(row[0]) for row in records[1:]]
        except ValueError as error:
            raise ValueError("Candidate CSV contains noninteger record IDs.") from error
    elif metadata.get("id_type") == "string":
        ids = [row[0] for row in records[1:]]
    else:
        raise ValueError("Candidate CSV manifest is missing a supported ID type.")
    sample = pd.DataFrame({"id": pd.Series(ids, dtype=object)})
    actual = validate_submission_file(_within(root, relative), sample, metadata["label_format"])
    for key in ("rows", "schema", "id_type", "label_format", "class_counts", "csv_sha256",
                "canonical_prediction_sha256", "canonical_prediction_version"):
        if actual[key] != metadata.get(key):
            raise ValueError(f"Candidate CSV disagrees with its manifest {key}: {relative}")
    return actual


def _platform_scores(root, candidates, notes):
    path = _within(root, "platform_scores.json")
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid platform_scores.json.") from error
    if not isinstance(entries, list):
        raise ValueError("platform_scores.json must contain a list of recorded results.")
    scores = []
    for index, entry in enumerate(entries):
        try:
            if not isinstance(entry, dict) or entry.get("candidate") not in candidates:
                raise ValueError("unknown candidate")
            if entry.get("status") in ("failed", "error", "rejected", "cancelled", "screenshot_failed", "upload_failed") or entry.get("accepted") is False:
                raise ValueError("failed upload is not a scored submission")
            if not isinstance(entry.get("submission_id"), str) or not entry["submission_id"].strip():
                raise ValueError("missing real submission ID")
            score = _number(entry.get("public_macro_f1"), "Kaggle Public Macro-F1")
            candidate = candidates[entry["candidate"]]
            if entry.get("model_name") != candidate["model_name"]:
                raise ValueError("model identity differs")
            if entry.get("csv_sha256") not in candidate["verified_csv_sha256"]:
                raise ValueError("score does not identify a verified candidate CSV")
        except ValueError as error:
            notes.append(f"Ignored platform score record {index}: {error}.")
            continue
        scores.append({**entry, "public_macro_f1": score,
                       "matches_integer_upload_csv": entry["csv_sha256"] == candidate["csv_sha256"],
                       "verification_scope": "Read from the user-reported platform_scores.json; no network verification."})
    return scores or None


def summarize_run(run_dir):
    """Return checked local metrics and the actual integer CSV for each candidate.

    Only complete ``train`` runs are supported. Small JSON/CSV artifacts are
    read; no training code, model binaries, or probability archives are loaded.
    Accuracy comes from models/<name>/summary.json because historical top-level
    comparison tables did not contain it. All returned paths are absolute and
    confined to the selected run. No history or score file is modified.
    """
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError("Training run directory does not exist.")
    manifest = _json(root, "run.json", {})
    if not isinstance(manifest, dict) or manifest.get("stage") != "train" or manifest.get("status") != "complete":
        raise ValueError("Results require a complete user-started train run.")
    summary = _json(root, "training_summary.json", manifest)
    if not isinstance(summary, dict):
        raise ValueError("training_summary.json must contain an object.")
    models = _metrics(root, summary, manifest)
    by_name = {model["name"]: model for model in models}
    protocol = _protocol(summary["comparison"].get("fold_protocol"))
    declared_protocol = manifest.get("result", {}).get("comparison", {}).get("fold_protocol")
    if declared_protocol is not None and declared_protocol != protocol:
        raise ValueError("Run and training-summary fold protocols differ.")
    declared_candidates = summary.get("candidates")
    if not isinstance(declared_candidates, dict) or not declared_candidates:
        raise ValueError("Training summary must identify its exported candidates.")
    candidates, notes = {}, []
    for key, candidate in declared_candidates.items():
        if not isinstance(candidate, dict) or candidate.get("model_name") not in by_name:
            raise ValueError("Exported candidate must identify a model in comparison.csv.")
        # Even unused references must not point outside this run. Bundle paths
        # are checked but estimator binaries are deliberately never opened.
        for field in ("csv", "alternative_csv", "bundle"):
            if field in candidate:
                _within(root, candidate[field])
        primary = _checked_csv(root, candidate.get("csv"), candidate.get("submission"), manifest)
        fit = candidate.get("fit_and_export", {})
        if "full_test_rows" in fit and _count(fit["full_test_rows"], "Full test") != primary["rows"]:
            raise ValueError("Candidate CSV row count differs from the full test row count.")
        alternative = None
        if "alternative_csv" in candidate:
            alternative = _checked_csv(root, candidate["alternative_csv"], fit.get("alternative_submission"), manifest)
            if candidate.get("alternative_format") != alternative["label_format"]:
                raise ValueError("Candidate alternative_format disagrees with its CSV manifest.")
            if alternative["rows"] != primary["rows"] or alternative["canonical_prediction_sha256"] != primary["canonical_prediction_sha256"]:
                raise ValueError("Primary and alternative predictions differ.")
        if primary["label_format"] == "integer":
            relative, integer = candidate["csv"], primary
        else:
            if candidate.get("alternative_format") != "integer" or alternative is None:
                raise ValueError("Candidate has no explicitly identified integer upload CSV.")
            relative = candidate.get("alternative_csv")
            integer = alternative
            if integer["label_format"] != "integer" or integer["rows"] != primary["rows"]:
                raise ValueError("Alternative candidate must be an integer CSV with matching row count.")
            if integer["canonical_prediction_sha256"] != primary["canonical_prediction_sha256"]:
                raise ValueError("Primary and integer alternative predictions differ.")
        candidates[key] = {
            "model_name": candidate["model_name"], "metrics": dict(by_name[candidate["model_name"]]),
            "integer_upload_csv": str(_within(root, relative)), "label_format": "integer",
            "rows": integer["rows"], "csv_sha256": integer["csv_sha256"],
            "canonical_prediction_sha256": integer["canonical_prediction_sha256"],
            "verified_csv_sha256": sorted({primary["csv_sha256"], (alternative or integer)["csv_sha256"]}),
            "platform_scores": None,
        }
    scores = _platform_scores(root, candidates, notes)
    for key, candidate in candidates.items():
        candidate["platform_scores"] = [entry for entry in scores or [] if entry["candidate"] == key] or None
    return {"run_id": manifest.get("run_id", root.name), "run_dir": str(root),
            "status": "complete", "stage": "train", "evaluation_scope": LOCAL_NOTICE,
            "data_sha256": _data_hashes(manifest.get("source_sha256")), "fold_protocol": protocol,
            "models": models, "candidates": candidates, "platform_scores": scores, "notes": notes}


def compare_runs(current, previous):
    """Describe current-minus-previous local changes only for identical data/folds.

    Arguments may be summarize_run dictionaries or paths to completed runs.
    Candidate letters identify export slots; model names are reported explicitly
    because a slot can point to different models across runs. No significance or
    Kaggle improvement is inferred from these descriptive differences.
    """
    current = current if isinstance(current, dict) else summarize_run(current)
    previous = previous if isinstance(previous, dict) else summarize_run(previous)
    if _data_hashes(current.get("data_sha256")) != _data_hashes(previous.get("data_sha256")):
        raise ValueError("Automatic comparison refused: source data hashes differ.")
    first, second = _protocol(current.get("fold_protocol")), _protocol(previous.get("fold_protocol"))
    if any(first.get(key) != second.get(key) for key in ("method", "fold_sha256", "n_splits", "seed")):
        raise ValueError("Automatic comparison refused: fold protocol or seed differs.")
    comparisons = {}
    for key in ("candidate_a", "candidate_b"):
        if key not in current.get("candidates", {}) or key not in previous.get("candidates", {}):
            continue
        now, before = current["candidates"][key], previous["candidates"][key]
        comparisons[key] = {
            "current_model_name": now["model_name"], "previous_model_name": before["model_name"],
            "same_model_name": now["model_name"] == before["model_name"],
            "oof_macro_f1_delta": _number(now["metrics"]["oof_macro_f1"], key) - _number(before["metrics"]["oof_macro_f1"], key),
            "fold_macro_f1_mean_delta": _number(now["metrics"]["fold_macro_f1_mean"], key) - _number(before["metrics"]["fold_macro_f1_mean"], key),
        }
    if not comparisons:
        raise ValueError("No matching candidate_a/candidate_b export slots to compare.")
    return {"current_run_id": current["run_id"], "previous_run_id": previous["run_id"],
            "same_data_and_folds": True, "candidates": comparisons,
            "evaluation_scope": "Descriptive same-fold local differences; not independent significance or Kaggle score gains."}


def print_results(summary):
    """Print a compact local table and the verified integer upload filenames."""
    print(f"Run: {summary['run_id']}")
    print(LOCAL_NOTICE)
    print(f"{'model':28} {'mean':>9} {'std':>9} {'OOF F1':>9} {'accuracy':>9}")
    for model in summary["models"]:
        name = model["name"]
        short = name if len(name) <= 28 else name[:25] + "..."
        print(f"{short:28} {model['fold_macro_f1_mean']:9.6f} {model['fold_macro_f1_std']:9.6f} "
              f"{model['oof_macro_f1']:9.6f} {model['accuracy']:9.6f}")
    for key, candidate in summary["candidates"].items():
        print(f"\n{key} = {candidate['model_name']} | integer | {candidate['rows']} rows")
        print(f"整数上传 CSV: {candidate['integer_upload_csv']}")
    if summary.get("platform_scores"):
        print("\nKaggle 平台记录（用户填报，未通过网络核验）：")
        for entry in summary["platform_scores"]:
            print(f"  {entry['candidate']} / {entry['submission_id']}: Public Macro-F1={entry['public_macro_f1']:.6f}")
    else:
        print("\nKaggle: 尚无真实平台成绩记录；上传或截图失败不能当作分数。")
    for note in summary.get("notes", []):
        print(f"Note: {note}")
