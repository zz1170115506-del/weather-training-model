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


def _config_snapshot(root, manifest, model_names):
    """Check persisted parameters; a missing legacy snapshot stays unknown."""
    from .artifacts import content_hash
    from .config import validate_experiment

    hint = ("参数摘要只列运行快照中的显式设置；未列出的分类器默认值和实际折内种子，"
            "见 models/<模型名>/summary.json 的 folds[*].effective_model_parameters；"
            "最终拟合参数见候选的 submission_manifest.json。HGB max_iter、权重生成器 max_iter、森林树数是不同参数。")
    path = _within(root, "resolved_config.json")
    if not path.exists():
        if any(key in manifest for key in ("experiment", "resolved_config_sha256", "resolved_config_file_sha256")):
            raise ValueError("A run with experiment/snapshot provenance requires resolved_config.json.")
        return None, {"status": "unavailable_legacy_run", "path": None,
                      "resolved_config_sha256": None, "resolved_config_file_sha256": None,
                      "models": [], "validation": None, "diagnostics": None,
                      "source_hint": "旧运行没有配置快照，不推断轮次或补造参数。" + hint}
    snapshot = _json(root, "resolved_config.json", manifest)
    if not isinstance(snapshot, dict):
        raise ValueError("resolved_config.json must contain a configuration object.")
    actual_hash = content_hash(snapshot)
    checked_hash = False
    for key in ("config_sha256", "resolved_config_sha256"):
        if key in manifest:
            if _sha(manifest[key], key) != actual_hash:
                raise ValueError(f"Configuration snapshot semantic SHA256 mismatch: {key}")
            checked_hash = True
    actual_file_hash = _hash(path.read_bytes())
    if "resolved_config_file_sha256" in manifest:
        if _sha(manifest["resolved_config_file_sha256"], "resolved_config_file_sha256") != actual_file_hash:
            raise ValueError("Configuration snapshot file SHA256 mismatch.")
        checked_hash = True
    if "resolved_config.json" in manifest.get("files_sha256", {}):
        checked_hash = True  # _json/_bytes already compared the actual bytes.
    declared = validate_experiment(manifest["experiment"]) if "experiment" in manifest else None
    configured = validate_experiment(snapshot["experiment"]) if "experiment" in snapshot else None
    if declared != configured:
        raise ValueError("Run experiment metadata and configuration snapshot disagree.")
    if configured is not None and not checked_hash:
        raise ValueError("An experiment run requires a recorded configuration snapshot SHA256.")
    raw_models = snapshot.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Configuration snapshot is missing model parameters.")
    params, seen = [], set()
    for model in raw_models:
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise ValueError("Configuration snapshot has invalid model entries.")
        name = model["name"]
        if name in seen or name not in model_names or not isinstance(model.get("model_params", {}), dict):
            raise ValueError("Configuration snapshot models do not match the reported comparison.")
        seen.add(name)
        params.append({"name": name, "model_id": model.get("model_id"),
                       "model_params": model.get("model_params", {}),
                       "early_stopping_rounds": model.get("early_stopping_rounds"),
                       "use_external_data": model.get("use_external_data"),
                       "noise_weighting": model.get("noise_weighting"),
                       "features": model.get("features", {}),
                       "group_features": model.get("group_features", False)})
    if seen != set(model_names):
        raise ValueError("Configuration snapshot models do not match the reported comparison.")
    return configured, {"status": "verified" if checked_hash else "unverified_legacy_snapshot",
                        "path": str(path), "resolved_config_sha256": actual_hash,
                        "resolved_config_file_sha256": actual_file_hash,
                        "models": params, "validation": snapshot.get("validation"),
                        "diagnostics": snapshot.get("diagnostics"), "source_hint": hint}


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


def _blend_recipe(recipe, base_model_names, context):
    """Validate the frozen JSON recipe without loading estimators or arrays."""
    if not isinstance(recipe, dict) or recipe.get("schema_version") != 1 or recipe.get("labels") != [1, 2, 3, 4]:
        raise ValueError(f"{context}: expected schema_version=1 and labels=[1,2,3,4].")
    names = recipe.get("model_ids")
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or _NAME.fullmatch(name) is None for name in names)
            or set(names) != set(base_model_names)):
        raise ValueError(f"{context}: recipe model members must match base_model_names.")
    weights = recipe.get("weights")
    if not isinstance(weights, list) or len(weights) != len(names):
        raise ValueError(f"{context}: expected one weight per model.")
    weights = [_number(value, f"{context}.weight") for value in weights]
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{context}: recipe weights must sum to one.")
    multipliers = recipe.get("class_multipliers")
    if not isinstance(multipliers, list) or len(multipliers) != 4:
        raise ValueError(f"{context}: expected four class multipliers.")
    for value in multipliers:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.9 <= value <= 1.1:
            raise ValueError(f"{context}: class multipliers must be finite and in [0.9,1.1].")
    return recipe


def _research_csv(root, candidate, manifest, base_model_names, recipe):
    """Check only declared CSVs and JSON; binary bundles are never opened."""
    if not isinstance(candidate, dict) or candidate.get("model_name") != "research_blend":
        raise ValueError("Research blend export must identify model_name=research_blend.")
    for field in ("csv", "bundle", "alternative_csv"):
        if field in candidate:
            _within(root, candidate[field])
    exported_recipe = _blend_recipe(candidate.get("recipe"), base_model_names, "Research blend export")
    if exported_recipe != recipe:
        raise ValueError("Research blend export recipe differs from the search's frozen recipe.")
    fit = candidate.get("fit_and_export")
    if not isinstance(fit, dict) or fit.get("recipe") != recipe:
        raise ValueError("Research blend fitted recipe differs from the frozen recipe.")
    primary = _checked_csv(root, candidate.get("csv"), candidate.get("submission"), manifest)
    if primary["label_format"] != "integer":
        raise ValueError("Research blend upload CSV must contain integer labels.")
    if fit.get("submission") != candidate["submission"]:
        raise ValueError("Research blend submission metadata differs from its full-fit export.")
    if _count(fit.get("full_test_rows"), "Research blend full test") != primary["rows"]:
        raise ValueError("Research blend full-test row count differs from its CSV.")
    verified = {primary["csv_sha256"]}
    if "alternative_csv" in candidate:
        alternative = _checked_csv(root, candidate["alternative_csv"], fit.get("alternative_submission"), manifest)
        if candidate.get("alternative_format") != alternative["label_format"]:
            raise ValueError("Research blend alternative_format differs from its CSV.")
        if (alternative["rows"] != primary["rows"]
                or alternative["canonical_prediction_sha256"] != primary["canonical_prediction_sha256"]):
            raise ValueError("Research blend primary and alternative predictions differ.")
        verified.add(alternative["csv_sha256"])
    return {"model_name": "research_blend", "integer_upload_csv": str(_within(root, candidate["csv"])),
            "rows": primary["rows"], "label_format": "integer", "csv_sha256": primary["csv_sha256"],
            "canonical_prediction_sha256": primary["canonical_prediction_sha256"],
            "verified_csv_sha256": sorted(verified), "platform_scores": None,
            "base_model_names": list(base_model_names), "recipe": recipe}


def _search_results(root, summary, manifest, models, candidates, protocol):
    report = summary.get("search")
    exported = summary.get("research_blend_export")
    if report is None:
        if exported is not None:
            raise ValueError("Research blend export requires an explicit checked search report.")
        return None, None
    saved = _json(root, "search_summary.json", manifest)
    if saved != report or not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Search report and training summary disagree.")
    names = report.get("candidate_model_names")
    if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names)) or set(names) != set(models)):
        raise ValueError("Search candidates must match all base models in the comparison.")
    rank = report.get("rank")
    if not isinstance(rank, list) or len(rank) != len(names):
        raise ValueError("Search ranking must cover every candidate.")
    seen = set()
    for index, item in enumerate(rank, 1):
        if not isinstance(item, dict) or item.get("name") not in models or item["name"] in seen or item.get("rank") != index:
            raise ValueError("Invalid search ranking identity/order.")
        model = models[item["name"]]
        seen.add(item["name"])
        for key in _METRICS:
            if not math.isclose(_number(item.get(key), key), model[key], abs_tol=1e-12, rel_tol=0):
                raise ValueError("Search ranking metrics disagree with base results.")
    ordered = sorted(models.values(), key=lambda model: (-model["oof_macro_f1"], model["fold_macro_f1_std"], model["name"]))
    if [item["name"] for item in rank] != [item["name"] for item in ordered]:
        raise ValueError("Search ranking violates the frozen deterministic order.")
    selected = report.get("selected_models")
    if not isinstance(selected, list) or selected != [value["model_name"] for value in candidates.values()] or selected[0] != rank[0]["name"]:
        raise ValueError("Search-selected models and actual exports disagree.")
    bases = report.get("base_model_names")
    if (not isinstance(bases, list) or not bases or any(not isinstance(name, str) for name in bases)
            or len(bases) != len(set(bases)) or not set(bases).issubset(models)):
        raise ValueError("Research blend base names must identify unique checked search candidates.")
    recipe = _blend_recipe(report.get("full_oof_recipe"), bases, "Search")
    for key in ("selection_scope", "evaluation_scope"):
        if not isinstance(report.get(key), str) or not report[key].strip():
            raise ValueError("Search must explicitly distinguish selection and evaluation scope.")
    single = _number(report.get("nested_single_metrics", {}).get("macro_f1"), "nested single")
    blend = _number(report.get("nested_blend_metrics", {}).get("macro_f1"), "nested blend")
    _number(report.get("full_oof_blend_metrics", {}).get("macro_f1"), "blend selection score")
    gate = report.get("blend_gate")
    if not isinstance(gate, dict) or not isinstance(report.get("blend_export_eligible"), bool):
        raise ValueError("Search must report its explicit blend export gate.")
    gain = _number(gate.get("minimum_blend_gain"), "minimum blend gain")
    count = _count(gate.get("outer_fold_count"), "outer fold count")
    min_wins = _count(gate.get("min_blend_fold_wins"), "minimum fold wins")
    folds = report.get("outer_folds")
    if count != protocol["n_splits"] or not isinstance(folds, list) or len(folds) != count or min_wins > count:
        raise ValueError("Search gate does not match the run's outer folds.")
    if sorted(item.get("fold") for item in folds) != list(range(count)):
        raise ValueError("Search outer fold identities are incomplete.")
    wins = sum(_number(item.get("blend_metrics", {}).get("macro_f1"), "fold blend") >
               _number(item.get("single_metrics", {}).get("macro_f1"), "fold single") + 1e-12 for item in folds)
    best = max(model["oof_macro_f1"] for model in models.values())
    eligible = blend - single + 1e-12 >= gain and blend - best + 1e-12 >= gain and wins >= min_wins
    if (gate.get("blend_fold_wins") != wins or gate.get("eligible") is not eligible
            or report["blend_export_eligible"] is not eligible):
        raise ValueError("Research blend export eligibility disagrees with the checked metrics.")
    if exported is not None and not eligible:
        raise ValueError("Research blend export is forbidden when the nested gate fails.")
    if eligible and exported is None:
        raise ValueError("A successful blend export gate is missing its declared frozen export.")
    research = _research_csv(root, exported, manifest, bases, recipe) if exported is not None else None
    if research is not None:
        research.update({"selection_metrics": report["full_oof_blend_metrics"],
                         "nested_procedure_metrics": report["nested_blend_metrics"],
                         "evaluation_scope": report["evaluation_scope"]})
    return report, research


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
    experiment, config_snapshot = _config_snapshot(root, manifest, by_name)
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
    search, research = _search_results(root, summary, manifest, by_name, candidates, protocol)
    all_candidates = dict(candidates)
    if research is not None:
        all_candidates["research_blend"] = research
    scores = _platform_scores(root, all_candidates, notes)
    for key, candidate in all_candidates.items():
        candidate["platform_scores"] = [entry for entry in scores or [] if entry["candidate"] == key] or None
    external = manifest.get("external_data_provenance")
    if external is not None:
        audit = _json(root, "audit.json", manifest)
        if audit.get("external_data") != external:
            raise ValueError("External-source provenance disagrees with the frozen data audit.")
        if external.get("policy") not in ("novel_only", "public_source_training"):
            raise ValueError("External-source provenance requires an explicit overlap policy.")
        _sha(external.get("source_sha256"), "external source")
        _sha(external.get("pool_sha256"), "external pool")
        for key in ("external_pool_rows", "test_overlap_source_rows_used", "independent_new_rows"):
            value = external.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("External-source row counts must be nonnegative integers.")
        if external["test_overlap_source_rows_used"] + external["independent_new_rows"] != external["external_pool_rows"]:
            raise ValueError("External-source independent/overlap row counts are inconsistent.")
    return {"run_id": manifest.get("run_id", root.name), "run_dir": str(root),
            "status": "complete", "stage": "train", "evaluation_scope": LOCAL_NOTICE,
            "experiment": experiment, "config_snapshot": config_snapshot,
            "external_data_provenance": external,
            "data_sha256": _data_hashes(manifest.get("source_sha256")), "fold_protocol": protocol,
            "models": models, "candidates": candidates, "platform_scores": scores, "notes": notes,
            "search": search, "research_blend": research}


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
            "current_experiment": current.get("experiment"), "previous_experiment": previous.get("experiment"),
            "same_data_and_folds": True, "candidates": comparisons,
            "data_comparison_scope": "Course train/test/sample and folds only; external training sources are listed separately.",
            "external_training_changed": current.get("external_data_provenance") != previous.get("external_data_provenance"),
            "current_external_data_provenance": current.get("external_data_provenance"),
            "previous_external_data_provenance": previous.get("external_data_provenance"),
            "evaluation_scope": "Descriptive same-fold local differences; not independent significance or Kaggle score gains."}


def print_results(summary):
    """Print a compact local table and the verified integer upload filenames."""
    print(f"Run: {summary['run_id']}")
    experiment = summary.get("experiment")
    if experiment is not None:
        print(f"实验轮次: 第 {experiment['round']} 轮 | {experiment['label']}")
        print(f"父运行: {experiment['parent_run_id']}")
        print(f"本轮改动: {experiment['change_note']}")
    else:
        print("实验轮次: 历史记录未标注，不自动推断。")
    snapshot = summary.get("config_snapshot", {})
    if snapshot.get("path"):
        print(f"配置快照: {snapshot['status']} | {snapshot['path']}")
        print(f"配置 SHA256: {snapshot['resolved_config_sha256']}")
        for model in snapshot["models"]:
            parameters = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                                   for key, value in sorted(model["model_params"].items()))
            print(f"  {model['name']}: {parameters or '(快照未显式列出分类器参数)'}")
            if model.get("early_stopping_rounds") is not None:
                print(f"    训练内部选轮 patience={model['early_stopping_rounds']}；上面的轮数是上限，实际轮数见拟合记录。")
            if model.get("use_external_data") is not None:
                print(f"    使用审计后的公开源训练池: {model['use_external_data']}")
            if model.get("noise_weighting"):
                weighting = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                                      for key, value in sorted(model["noise_weighting"].items()))
                print(f"    自主权重生成器（独立参数）: {weighting}")
    if snapshot.get("source_hint"):
        print(snapshot["source_hint"])
    print(LOCAL_NOTICE)
    if summary.get("external_data_provenance") is not None:
        source = summary["external_data_provenance"]
        print(f"公开源策略={source['policy']}，训练池={source['external_pool_rows']}条，"
              f"独立新增={source['independent_new_rows']}条，课程测试特征重叠={source['test_overlap_source_rows_used']}条。")
        if source["test_overlap_source_rows_used"]:
            print("该模式显式使用获准的同源公开标签；平台成绩不可称为独立未知测试的泛化效果。")
    print(f"{'model':28} {'mean':>9} {'std':>9} {'OOF F1':>9} {'accuracy':>9}")
    for model in summary["models"]:
        name = model["name"]
        short = name if len(name) <= 28 else name[:25] + "..."
        print(f"{short:28} {model['fold_macro_f1_mean']:9.6f} {model['fold_macro_f1_std']:9.6f} "
              f"{model['oof_macro_f1']:9.6f} {model['accuracy']:9.6f}")
    for key, candidate in summary["candidates"].items():
        print(f"\n{key} = {candidate['model_name']} | integer | {candidate['rows']} rows")
        print(f"整数上传 CSV: {candidate['integer_upload_csv']}")
    best_key, best = max(summary["candidates"].items(), key=lambda item: item[1]["metrics"]["oof_macro_f1"])
    print(f"\n已导出候选中，本地 OOF 最高: {best_key} ({best['model_name']}) "
          f"Macro-F1={best['metrics']['oof_macro_f1']:.6f}；仅供手动选型，不保证 Kaggle 更高。")
    if summary.get("search") is not None:
        search = summary["search"]
        print("\n搜索选择分数不等于独立成绩；嵌套值是选模程序的开发诊断，门控本身也有选择偏差。")
        print(f"  嵌套 single={search['nested_single_metrics']['macro_f1']:.6f}; "
              f"blend={search['nested_blend_metrics']['macro_f1']:.6f}; "
              f"融合导出门控={search['blend_export_eligible']}")
        if summary.get("research_blend") is not None:
            research = summary["research_blend"]
            print(f"  research_blend 整数上传CSV: {research['integer_upload_csv']}")
            print(f"  融合调参分数={research['selection_metrics']['macro_f1']:.6f}，不能当作独立/Kaggle成绩。")
    if summary.get("platform_scores"):
        print("\nKaggle 平台记录（用户填报，未通过网络核验）：")
        for entry in summary["platform_scores"]:
            print(f"  {entry['candidate']} / {entry['submission_id']}: Public Macro-F1={entry['public_macro_f1']:.6f}")
    else:
        print("\nKaggle: 尚无真实平台成绩记录；上传或截图失败不能当作分数。")
    for note in summary.get("notes", []):
        print(f"Note: {note}")
