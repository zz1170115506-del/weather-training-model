"""One data/validation/prediction workflow for all Dataset1 model families."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import json
import joblib
import numpy as np
import pandas as pd

from . import LABELS
from .artifacts import (ROOT, content_hash, file_hash, finish_run, git_state,
                        new_run, save_json)
from .config import load_config
from .data import FEATURES, infer_label_format, load_data
from .ensemble import apply_blend, fit_blend
from .models import fit_model, make_model, predict_probabilities
from .submission import validate_submission_file, write_submission
from .validation import cross_validate, group_diagnostics, make_folds, metrics


def source_version():
    files = {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p)
             for p in sorted((ROOT / "code").rglob("*.py"))}
    return {"code_files_sha256": files, "source_tree_sha256": content_hash(files)}


def _folds(data, validation):
    return make_folds(data.y, data.train.id.to_numpy(), data.groups,
                      validation["n_splits"], validation["split_seed"])


def _write_folds(out, data, folds):
    pd.DataFrame({"id": data.train.id, "fold": folds}).to_csv(
        out / "folds.csv", index=False, lineterminator="\n")


def run_cv(data, config, out):
    folds, fold_info = _folds(data, config["validation"])
    _write_folds(out, data, folds)
    save_json(out / "fold_protocol.json", fold_info)
    rows = []
    for candidate in config["models"]:
        result, _ = cross_validate(data, candidate, folds, config["validation"]["model_seed"],
                                   out / "models" / candidate["name"])
        rows.append({"name": candidate["name"], "model_id": candidate["model_id"],
                     "config_sha256": result["config_sha256"],
                     "fold_macro_f1_mean": result["fold_macro_f1_mean"],
                     "fold_macro_f1_std": result["fold_macro_f1_std"],
                     "oof_macro_f1": result["oof_metrics"]["macro_f1"],
                     "total_cv_seconds": result["total_cv_seconds"]})
        pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False, lineterminator="\n")
    print(pd.DataFrame(rows).sort_values("oof_macro_f1", ascending=False).to_string(index=False), flush=True)
    return {"fold_protocol": fold_info, "metrics": rows,
            "evaluation_scope": "Exploratory cross-validation for selection; not Kaggle/private results."}


def run_nested(data, config, out):
    """Fit blend weights inside each outer-train, never on outer validation."""
    validation = config["validation"]
    outer, protocol = _folds(data, validation)
    _write_folds(out, data, outer)
    ensemble_config = config.get("blend", {})
    inner_count = int(config.get("inner_splits", 3))
    oof = np.zeros((len(data.y), 4))
    equal_oof = np.zeros_like(oof)
    base_oof = {m["name"]: np.zeros_like(oof) for m in config["models"]}
    outer_records = []
    for fold in sorted(np.unique(outer)):
        fit, valid = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
        inner_data = replace(data, train=data.train.iloc[fit].reset_index(drop=True),
                             groups=data.groups[fit])
        inner_folds, inner_protocol = make_folds(
            inner_data.y, inner_data.train.id.to_numpy(), inner_data.groups,
            inner_count, validation["split_seed"] + 1000 + int(fold))
        inner_probabilities, valid_probabilities, model_times = {}, {}, {}
        for candidate in config["models"]:
            name = candidate["name"]
            _, inner_probabilities[name] = cross_validate(
                inner_data, candidate, inner_folds,
                validation["model_seed"] + 10000 * (int(fold) + 1), verbose=False)
            bundle, fit_info = fit_model(inner_data.X, inner_data.y, candidate,
                                         validation["model_seed"] + int(fold))
            valid_probabilities[name] = predict_probabilities(bundle, data.X.iloc[valid])
            base_oof[name][valid] = valid_probabilities[name]
            model_times[name] = fit_info
        recipe = fit_blend(inner_probabilities, inner_data.y,
                           seed=validation["model_seed"] + int(fold),
                           n_trials=int(ensemble_config.get("n_trials", 128)),
                           allow_class_multipliers=ensemble_config.get("allow_class_multipliers", False))
        oof[valid] = apply_blend(valid_probabilities, recipe)
        equal_oof[valid] = np.mean(list(valid_probabilities.values()), axis=0)
        result = metrics(data.y[valid], oof[valid])
        outer_records.append({"fold": int(fold), "metrics": result,
                              "inner_fold_protocol": inner_protocol, "recipe": recipe,
                              "full_outer_train_fit_times": model_times,
                              "outer_validation_used_for_selection": False})
        save_json(out / f"outer_fold_{int(fold)}.json", outer_records[-1])
        print(f"Nested fold {int(fold)+1}: blend Macro-F1={result['macro_f1']:.6f}", flush=True)
    ablations = {name: metrics(data.y, probabilities) for name, probabilities in base_oof.items()}
    ablations["equal_probability_average"] = metrics(data.y, equal_oof)
    ablations["inner_selected_blend"] = metrics(data.y, oof)
    np.savez_compressed(out / "outer_oof.npz", ids=data.train.id.to_numpy(), y=data.y,
                        folds=outer, blend=oof, equal=equal_oof, **base_oof)
    summary = {"outer_protocol": protocol, "inner_splits": inner_count,
               "outer_fold_macro_f1_mean": float(np.mean([r["metrics"]["macro_f1"] for r in outer_records])),
               "outer_fold_macro_f1_std": float(np.std([r["metrics"]["macro_f1"] for r in outer_records], ddof=1)),
               "oof_metrics": metrics(data.y, oof), "ablations": ablations,
               "folds": outer_records,
               "evaluation_scope": "Outer-fold evaluation of fixed candidate configs with inner-only blend selection. Candidate configs were previously explored on these data; not an independent test of the entire search."}
    save_json(out / "nested_summary.json", summary)
    return summary


def _training_recipe(data, config):
    if len(config["models"]) == 1 and not config.get("blend", {}).get("allow_class_multipliers", False):
        name = config["models"][0]["name"]
        return {"schema_version": 1, "labels": list(LABELS), "model_ids": [name],
                "weights": [1.0], "class_multipliers": [1.0] * 4,
                "evaluation_scope": "Frozen single-model prediction recipe."}, None
    # Weights are learned on full-training OOF only after candidate recipes are frozen.
    folds, fold_info = _folds(data, config["validation"])
    probabilities = {}
    for candidate in config["models"]:
        _, probabilities[candidate["name"]] = cross_validate(
            data, candidate, folds, config["validation"]["model_seed"], verbose=False)
    settings = config.get("blend", {})
    recipe = fit_blend(probabilities, data.y, seed=config["validation"]["model_seed"],
                       n_trials=int(settings.get("n_trials", 128)),
                       allow_class_multipliers=settings.get("allow_class_multipliers", False))
    return recipe, fold_info


def run_finalize(data, config, out, label_format):
    if len(config["models"]) > 1 and config.get("prediction_strategy") != "learned_blend":
        raise ValueError("Use train for separate fixed candidate exports; multi-model finalize requires explicit prediction_strategy=learned_blend.")
    started = perf_counter()
    recipe, weight_protocol = _training_recipe(data, config)
    save_json(out / "blend_recipe.json", recipe)
    bundles, times = {}, {}
    for candidate in config["models"]:
        bundle, timing = fit_model(data.X, data.y, candidate, config["validation"]["model_seed"])
        bundles[candidate["name"]] = bundle
        times[candidate["name"]] = timing
    infer_started = perf_counter()
    probabilities = {name: predict_probabilities(bundle, data.test[FEATURES])
                     for name, bundle in bundles.items()}
    blended = apply_blend(probabilities, recipe)
    inference_seconds = perf_counter() - infer_started
    predictions = np.asarray(LABELS)[blended.argmax(axis=1)]
    resolved_format = infer_label_format(data.sample) if label_format == "sample" else label_format
    submission = write_submission(data.test.id.to_numpy(), predictions, data.sample,
                                  out / "submission.csv", resolved_format)
    alternative = "integer" if resolved_format == "name" else "name"
    alternative_metadata = write_submission(data.test.id.to_numpy(), predictions, data.sample,
                                            out / f"submission_{alternative}.csv", alternative)
    np.savez_compressed(out / "test_probabilities.npz", ids=data.test.id.to_numpy(),
                        probabilities=blended, **probabilities)
    saved = {"schema_version": 1, "models": bundles, "recipe": recipe, "config": config,
             "source_sha256": data.audit["source_sha256"], "feature_order": FEATURES,
             "label_format": resolved_format, "expected_predictions": predictions,
             "expected_prediction_sha256": submission["canonical_prediction_sha256"]}
    joblib.dump(saved, out / "model_bundle.joblib", compress=3)
    reloaded = joblib.load(out / "model_bundle.joblib")
    replay = apply_blend({name: predict_probabilities(bundle, data.test[FEATURES])
                          for name, bundle in reloaded["models"].items()}, reloaded["recipe"])
    if not np.array_equal(blended, replay):
        raise RuntimeError("Saved model reload did not reproduce exact test probabilities.")
    result = {"submission": submission, "alternative_submission": alternative_metadata,
              "label_format_requested": label_format, "label_format_used": resolved_format,
              "format_basis": "sample_submission.csv" if label_format == "sample" else "explicit CLI selection",
              "platform_acceptance_verified": False,
              "weight_selection_protocol": weight_protocol, "recipe": recipe,
              "fit_times": times, "full_test_rows": len(data.test),
              "full_test_inference_seconds_including_preprocessing": inference_seconds,
              "offline_seconds_including_weight_selection": perf_counter() - started,
              "saved_model_reload_exact_match": True,
              "prediction_class_counts": {str(k): int((predictions == k).sum()) for k in LABELS},
              "training_epochs": "not applicable to trees; boosting iterations stored per model",
              "evaluation_scope": "Full-training predictions only. No test labels or Kaggle scores available."}
    save_json(out / "submission_manifest.json", result)
    return result


def run_predict(data, bundle_path, output, label_format):
    saved = joblib.load(bundle_path)
    if saved["source_sha256"] != data.audit["source_sha256"]:
        raise ValueError("Reproduction data hashes differ from the trained bundle.")
    probabilities = {name: predict_probabilities(bundle, data.test[FEATURES])
                     for name, bundle in saved["models"].items()}
    blended = apply_blend(probabilities, saved["recipe"])
    predictions = np.asarray(LABELS)[blended.argmax(axis=1)]
    if not np.array_equal(predictions, saved["expected_predictions"]):
        raise RuntimeError("Loaded code/environment changed the saved model's expected labels; output refused.")
    fmt = infer_label_format(data.sample) if label_format == "sample" else label_format
    result = write_submission(data.test.id.to_numpy(), predictions, data.sample, output, fmt)
    if result["canonical_prediction_sha256"] != saved["expected_prediction_sha256"]:
        raise RuntimeError("Reproduced semantic prediction hash differs from the frozen submission.")
    return result


def run_train(data, config, out, label_format):
    """User-started full workflow: common CV, fixed candidates, export and replay.

    Candidate model names are fixed in config.training.export_models. We do not
    silently replace them with an OOF winner or adopt a learned blend that has
    failed outer evaluation. All files are local and this function has no
    Kaggle/GitHub network calls.
    """
    print("[1/3] Common-fold development validation", flush=True)
    comparison = run_cv(data, config, out)
    candidates = {}
    submissions = out / "submissions"
    submissions.mkdir()
    for index, name in enumerate(config["training"]["export_models"]):
        candidate_key = f"candidate_{chr(97 + index)}" if index < 26 else f"candidate_{index+1}"
        single = deepcopy(config)
        single["models"] = [next(model for model in config["models"] if model["name"] == name)]
        single["blend"] = {"n_trials": 0, "allow_class_multipliers": False}
        single["training"] = {"export_models": [name]}
        print(f"[2/3] Full-data fit and export: {candidate_key} = {name}", flush=True)
        destination = out / "candidates" / name
        destination.mkdir(parents=True)
        result = run_finalize(data, single, destination, label_format)
        csv_name = candidate_key + ".csv"
        with (submissions / csv_name).open("xb") as stream:
            stream.write((destination / "submission.csv").read_bytes())
        alternative = "integer" if result["label_format_used"] == "name" else "name"
        with (submissions / f"{candidate_key}_{alternative}.csv").open("xb") as stream:
            stream.write((destination / f"submission_{alternative}.csv").read_bytes())
        prediction_path = destination / "replay_check.csv"
        replay = run_predict(data, destination / "model_bundle.joblib", prediction_path,
                             result["label_format_used"])
        if replay["csv_sha256"] != result["submission"]["csv_sha256"]:
            raise RuntimeError("Fresh bundle replay changed the submission bytes.")
        candidates[candidate_key] = {"model_name": name, "csv": f"submissions/{csv_name}",
                                     "bundle": f"candidates/{name}/model_bundle.joblib",
                                     "submission": result["submission"],
                                     "alternative_format": alternative,
                                     "alternative_csv": f"submissions/{candidate_key}_{alternative}.csv",
                                     "fit_and_export": result,
                                     "platform_acceptance_verified": False}
        print(f"READY_TO_UPLOAD={submissions / csv_name}", flush=True)
    result = {"comparison": comparison, "candidates": candidates,
              "candidate_policy": "Fixed named single-model candidates; no automatic Kaggle submission or OOF-based candidate replacement.",
              "manual_next_step": "Upload candidate_a.csv using the competition File Upload window; verify the platform response, then record the real score.",
              "score_log_policy": "Append-only user-reported platform_scores.json entries after training."}
    save_json(out / "training_summary.json", result)
    print("[3/3] CSV/schema/hash and model reload checks passed. Kaggle upload is manual.", flush=True)
    return result


def record_score(args):
    if args.run_dir is None or args.submission_id is None or args.public_score is None:
        raise ValueError("record-score requires --run-dir, --submission-id and --public-score.")
    if not args.submission_id.strip() or not np.isfinite(args.public_score) or not 0 <= args.public_score <= 1:
        raise ValueError("A real submission ID and a finite Public Macro-F1 in [0,1] are required.")
    out = args.run_dir.resolve()
    summary = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))
    if args.candidate not in summary["candidates"]:
        raise ValueError("Unknown candidate; use a key in training_summary.json.")
    candidate = summary["candidates"][args.candidate]
    path = out / candidate["csv"]
    if file_hash(path) != candidate["submission"]["csv_sha256"]:
        raise ValueError("The submitted CSV has changed since training.")
    log_path = out / "platform_scores.json"
    entries = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    if any(row["submission_id"] == args.submission_id for row in entries):
        raise ValueError("This submission ID is already recorded; previous records are not overwritten.")
    entry = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
             "candidate": args.candidate, "model_name": candidate["model_name"],
             "submission_id": args.submission_id, "public_macro_f1": args.public_score,
             "csv_sha256": candidate["submission"]["csv_sha256"],
             "note": args.note, "source": "User-reported Kaggle result; not verified by a network API."}
    entries.append(entry)
    save_json(log_path, entries)
    print(json.dumps(entry, indent=2, ensure_ascii=False))


def execute(args):
    if args.stage == "record-score":
        record_score(args)
        return
    overall_started = perf_counter()
    data = load_data(args.data_dir)
    data_load_seconds = perf_counter() - overall_started
    if args.stage == "validate-submission":
        if args.submission is None:
            raise ValueError("--submission is required.")
        fmt = infer_label_format(data.sample) if args.label_format == "sample" else args.label_format
        print(json.dumps(validate_submission_file(args.submission, data.sample, fmt), indent=2))
        return
    if args.stage == "predict":
        if args.bundle is None or args.output is None:
            raise ValueError("--bundle and --output are required.")
        print(json.dumps(run_predict(data, args.bundle, args.output, args.label_format), indent=2))
        return
    config = load_config(args.config) if args.config else {"schema_version": 1, "stage": "audit"}
    if args.stage != "audit" and args.config is None:
        raise ValueError("--config is required for this stage.")
    if args.stage == "check":
        if args.config is None:
            raise ValueError("check requires --config.")
        for model in config["models"]:
            make_model(model, config["validation"]["model_seed"])
        print(json.dumps({"status": "ready", "training_started": False,
                          "data_rows": data.audit["rows"], "source_sha256": data.audit["source_sha256"],
                          "models": [model["name"] for model in config["models"]],
                          "export_models": config["training"]["export_models"],
                          "label_format_from_sample": infer_label_format(data.sample),
                          "platform_acceptance_verified": False, "git": git_state()}, indent=2))
        return
    if args.stage in ("finalize", "train") and args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}.")
    state = git_state()
    if args.stage in ("finalize", "train") and state["dirty"] and not args.allow_dirty:
        raise ValueError("Training/finalize requires clean source. Commit shared changes first, or keep personal JSON configs outside the repository; --allow-dirty is development only.")
    out = new_run(args.stage, config, args.artifact_root)
    print(f"RUN_DIR={out}", flush=True)
    save_json(out / "audit.json", data.audit)
    version = source_version()
    save_json(out / "source_version.json", version)
    started = perf_counter()
    try:
        if args.stage == "audit":
            result = {"audit": data.audit}
        elif args.stage == "cv":
            result = run_cv(data, config, out)
        elif args.stage == "nested":
            result = run_nested(data, config, out)
        elif args.stage == "train":
            result = run_train(data, config, out, args.label_format)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                first_candidate = next(iter(result["candidates"].values()))
                with args.output.open("xb") as stream:
                    stream.write((out / first_candidate["csv"]).read_bytes())
                result["additional_csv_copy_sha256"] = file_hash(args.output)
        else:
            result = run_finalize(data, config, out, args.label_format)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("xb") as stream:
                    stream.write((out / "submission.csv").read_bytes())
                result["additional_csv_copy_sha256"] = file_hash(args.output)
        finish_run(out, {"source_sha256": data.audit["source_sha256"],
                         "source_tree_sha256": version["source_tree_sha256"],
                         "data_load_and_audit_seconds": data_load_seconds,
                         "workflow_seconds": perf_counter() - overall_started,
                         "timing_scope": "Whole command through evaluation/finalization, including initial data loading and metadata; excludes final manifest hashing.",
                         "reproduce_command_template": (
                             f"python code/run_task1.py {args.stage} "
                             + (f"--config {args.config.as_posix()} " if args.config else "")
                             + "--data-dir <DATA_DIR> --artifact-root <ARTIFACT_ROOT> "
                             + (f"--label-format {args.label_format}" if args.stage in ("finalize", "train") else "")),
                         "result": result})
        if args.stage == "train":
            try:
                from .plots import export_results
                result["figures"] = export_results([out], out / "summary")
            except Exception as error:
                result["figures"] = {"status": "failed", "error": str(error)}
                print(f"Figure export failed; validated CSVs remain available: {error}", flush=True)
            save_json(out / "training_summary.json", result)
            finish_run(out, {"result": result})
            save_json(out.parent / "latest_training_run.json", {"run_dir": str(out),
                       "candidates": {key: str(out / value["csv"]) for key, value in result["candidates"].items()}})
    except BaseException as error:
        save_json(out / "error.json", {"type": type(error).__name__, "message": str(error)})
        state = json.loads((out / "run.json").read_text(encoding="utf-8"))
        state.update({"status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__,
                      "workflow_seconds": perf_counter() - started})
        save_json(out / "run.json", state)
        raise
    print(f"ARTIFACT_DIR={out}", flush=True)
