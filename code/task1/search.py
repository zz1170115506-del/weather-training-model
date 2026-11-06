"""Nested, training-only recipe selection with an explicit development export gate.

The outer predictions already written by run_cv are reused; every decision for
an outer fold is learned afresh inside its training partition. The final model
rank, full-OOF recipe, and export gate reuse development labels and therefore
are not an independent assessment or a claim about Kaggle's hidden labels.
"""
from __future__ import annotations

from dataclasses import replace
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from . import LABELS
from .artifacts import content_hash, save_json
from .config import resolve_config
from .ensemble import apply_blend, fit_blend
from .validation import cross_validate, make_folds, metrics

SELECTION_SCOPE = (
    "Full-OOF ranking, pool selection and recipe tuning reuse development labels; "
    "their scores are selection scores, not independent evaluation or Kaggle results."
)
EVALUATION_SCOPE = (
    "Configured candidate selection and blend weights are learned only inside each "
    "outer training partition. Candidate designs were previously explored on these "
    "development data, and the export gate also selects using these outer results; "
    "the chosen export is not an independent test of the entire search."
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _probabilities(value, rows):
    array = np.asarray(value)
    if (array.shape != (rows, len(LABELS)) or array.dtype.kind not in "iuf"
            or not np.isfinite(array).all() or (array < 0).any() or (array > 1).any()
            or not np.allclose(array.sum(axis=1), 1, atol=1e-6, rtol=1e-6)):
        raise ValueError("Search requires finite, normalized (n,4) probabilities in label order 1..4.")
    return array.astype(np.float64, copy=True)


def _same_ids(actual, expected):
    """Use only safe integer or fixed-width Unicode NPZ identities, never pickle."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 1:
        return False
    values = expected.tolist()
    if actual.dtype.kind in "iu" and all(isinstance(value, Integral) and not isinstance(value, bool) for value in values):
        return actual.tolist() == values
    if actual.dtype.kind == "U" and all(isinstance(value, str) for value in values):
        return actual.tolist() == values
    return False


def _archive_ids(values):
    values = np.asarray(values).tolist()
    return np.asarray(values, dtype=str) if all(isinstance(value, str) for value in values) else np.asarray(values)


def _metric_matches(actual, expected):
    """Reject edited class or aggregate metrics without depending on JSON order."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_metric_matches(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_metric_matches(first, second) for first, second in zip(actual, expected)))
    if isinstance(expected, (int, float)):
        return (not isinstance(actual, bool) and isinstance(actual, (int, float))
                and np.isfinite(actual) and np.isclose(actual, expected, atol=1e-12, rtol=0))
    return actual == expected


def _rank(summaries):
    ordered = sorted(summaries, key=lambda item: (
        -item["oof_metrics"]["macro_f1"], item["fold_macro_f1_std"], item["name"]))
    return [{"rank": index, "name": item["name"], "model_id": item["model_id"],
             "config_sha256": item["config_sha256"],
             "oof_macro_f1": item["oof_metrics"]["macro_f1"],
             "fold_macro_f1_mean": item["fold_macro_f1_mean"],
             "fold_macro_f1_std": item["fold_macro_f1_std"],
             "oof_accuracy": item["oof_metrics"]["accuracy"]}
            for index, item in enumerate(ordered, 1)]


def _pool(rank, size):
    chosen, families = [], set()
    for item in rank:
        if item["model_id"] not in families:
            chosen.append(item["name"])
            families.add(item["model_id"])
            if len(chosen) == size:
                return chosen
    chosen.extend(item["name"] for item in rank if item["name"] not in chosen)
    return chosen[:size]


def _load_base(data, config, out, folds, protocol):
    if _read(out / "fold_protocol.json") != protocol:
        raise ValueError("Search fold protocol/hash differs from the configured common folds.")
    recorded = pd.read_csv(out / "folds.csv", converters={"id": str})
    if (list(recorded) != ["id", "fold"] or len(recorded) != len(folds)
            or recorded.id.tolist() != [str(value) for value in data.train.id]
            or recorded.fold.to_numpy().dtype.kind not in "iu"
            or not np.array_equal(recorded.fold.to_numpy(), folds)):
        raise ValueError("Saved fold identities/order differ from the configured data.")
    summaries, probabilities = [], {}
    for candidate in config["models"]:
        name = candidate["name"]
        directory = out / "models" / name
        summary = _read(directory / "summary.json")
        if (summary.get("name") != name or summary.get("model_id") != candidate["model_id"]
                or summary.get("config_sha256") != content_hash(candidate)
                or content_hash(summary.get("config")) != content_hash(candidate)):
            raise ValueError(f"{name}: base summary configuration/hash mismatch.")
        try:
            with np.load(directory / "oof.npz", allow_pickle=False) as archive:
                if set(archive.files) != {"ids", "y", "folds", "probabilities"}:
                    raise ValueError("OOF archive must contain exactly ids/y/folds/probabilities.")
                ids, labels, saved_folds = archive["ids"], archive["y"], archive["folds"]
                probability = _probabilities(archive["probabilities"], len(data.y))
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"{name}: invalid safe OOF archive; integer/fixed Unicode identities are required (no pickle).") from error
        if (not _same_ids(ids, data.train.id.to_numpy()) or labels.dtype.kind not in "iu"
                or not np.array_equal(labels, data.y) or not np.isin(labels, LABELS).all()
                or saved_folds.dtype.kind not in "iu" or not np.array_equal(saved_folds, folds)):
            raise ValueError(f"{name}: OOF ids, labels or folds do not match this training run.")
        if not _metric_matches(summary.get("oof_metrics"), metrics(data.y, probability)):
            raise ValueError(f"{name}: OOF metrics disagree with its archived probabilities.")
        details = summary.get("folds")
        if not isinstance(details, list) or len(details) != protocol["n_splits"]:
            raise ValueError(f"{name}: incomplete base fold summaries.")
        if sorted(item.get("fold", -1) for item in details) != list(range(protocol["n_splits"])):
            raise ValueError(f"{name}: invalid base fold identities.")
        scores = []
        for detail in details:
            fold = detail["fold"]
            valid = folds == fold
            measured = metrics(data.y[valid], probability[valid])
            if (not _metric_matches(detail.get("metrics"), measured)
                    or detail.get("fit_ids_sha256") != content_hash(data.train.id[~valid].tolist())
                    or detail.get("validation_ids_sha256") != content_hash(data.train.id[valid].tolist())):
                raise ValueError(f"{name}: base fold metrics/training identity mismatch.")
            scores.append(measured["macro_f1"])
        for key, expected in (("fold_macro_f1_mean", float(np.mean(scores))),
                              ("fold_macro_f1_std", float(np.std(scores, ddof=1)))):
            if not _metric_matches(summary.get(key), expected):
                raise ValueError(f"{name}: base aggregate fold metrics mismatch.")
        summaries.append(summary)
        probabilities[name] = probability
    return summaries, probabilities


def evaluate_search(data, config, out):
    """Evaluate nested selection, freeze final rank/recipe, and save local evidence.

    Called only by an explicit user-started train command after run_cv. No
    repository, network, submission, or original configuration mutation occurs.
    """
    config = resolve_config(config)
    if "search" not in config:
        raise ValueError("evaluate_search requires an explicit search configuration.")
    out = Path(out).resolve()
    directory = out / "search"
    for path in (directory, out / "selection.json", out / "search_summary.json"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite search artifacts: {path}")
    snapshot = out / "resolved_config.json"
    if snapshot.exists() and content_hash(_read(snapshot)) != content_hash(config):
        raise ValueError("Search configuration differs from the original frozen run snapshot.")
    validation, settings = config["validation"], config["search"]
    folds, protocol = make_folds(data.y, data.train.id.to_numpy(), data.groups,
                                validation["n_splits"], validation["split_seed"])
    summaries, base_probabilities = _load_base(data, config, out, folds, protocol)
    directory.mkdir()
    selected_oof = np.full((len(data.y), 4), np.nan)
    blend_oof = np.full_like(selected_oof, np.nan)
    seen = np.zeros(len(data.y), dtype=int)
    outer_details = []
    blend_settings = config["blend"]
    for fold in range(validation["n_splits"]):
        fit, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inner_data = replace(data, train=data.train.iloc[fit].reset_index(drop=True), groups=data.groups[fit])
        inner_folds, inner_protocol = make_folds(
            inner_data.y, inner_data.train.id.to_numpy(), inner_data.groups,
            settings["inner_splits"], validation["split_seed"] + 1000 + fold)
        fold_dir = directory / f"outer_{fold}"
        fold_dir.mkdir()
        inner_probabilities, inner_summaries, trial_paths = {}, [], []
        inner_model_seed = validation["model_seed"] + 10000 * (fold + 1)
        for index, candidate in enumerate(config["models"]):
            print(f"[search] outer {fold+1}/{validation['n_splits']}, trial {index+1}/{len(config['models'])}: {candidate['name']}", flush=True)
            summary, probability = cross_validate(inner_data, candidate, inner_folds, inner_model_seed, verbose=False)
            if (summary.get("name") != candidate["name"] or summary.get("model_id") != candidate["model_id"]
                    or summary.get("config_sha256") != content_hash(candidate)
                    or content_hash(summary.get("config")) != content_hash(candidate)
                    or not _metric_matches(summary.get("oof_metrics"), metrics(inner_data.y, probability))):
                raise ValueError("Inner trial summary/probabilities do not match the candidate.")
            inner_probabilities[candidate["name"]] = _probabilities(probability, len(fit))
            inner_summaries.append(summary)
            path = fold_dir / f"trial_{index}.json"
            save_json(path, {**summary, "search_inner_fold_protocol": inner_protocol,
                             "search_outer_fold": fold, "search_inner_model_seed": inner_model_seed})
            trial_paths.append(path.relative_to(out).as_posix())
        inner_rank = _rank(inner_summaries)
        winner = inner_rank[0]["name"]
        pool = _pool(inner_rank, settings["blend_pool_size"])
        recipe = fit_blend({name: inner_probabilities[name] for name in pool}, inner_data.y,
                           seed=validation["model_seed"] + fold,
                           n_trials=blend_settings["n_trials"],
                           allow_class_multipliers=blend_settings["allow_class_multipliers"])
        recipe["selection_scope"] = "Inner-training OOF labels only; no outer validation labels used."
        selected_oof[valid] = base_probabilities[winner][valid]
        blend_oof[valid] = apply_blend({name: base_probabilities[name][valid] for name in pool}, recipe)
        seen[valid] += 1
        np.savez_compressed(fold_dir / "inner_oof.npz", ids=_archive_ids(inner_data.train.id), y=inner_data.y,
                            folds=inner_folds, **inner_probabilities)
        detail = {"fold": fold, "inner_winner": winner, "inner_pool": pool, "inner_rank": inner_rank,
                  "recipe": recipe, "single_metrics": metrics(data.y[valid], selected_oof[valid]),
                  "blend_metrics": metrics(data.y[valid], blend_oof[valid]),
                  "trial_summary_paths": trial_paths, "inner_fold_protocol": inner_protocol,
                  "inner_model_seed": inner_model_seed,
                  "outer_fit_ids_sha256": content_hash(data.train.id.iloc[fit].tolist()),
                  "outer_validation_ids_sha256": content_hash(data.train.id.iloc[valid].tolist()),
                  "outer_validation_used_for_selection": False}
        outer_details.append(detail)
        save_json(fold_dir / "selection.json", detail)
    if not np.all(seen == 1) or not np.isfinite(selected_oof).all() or not np.isfinite(blend_oof).all():
        raise RuntimeError("Each record requires exactly one finite cross-fitted search prediction.")
    np.savez_compressed(directory / "cross_fitted.npz", ids=_archive_ids(data.train.id), y=data.y,
                        folds=folds, single_probabilities=selected_oof, blend_probabilities=blend_oof)
    rank = _rank(summaries)
    best = rank[0]
    runner_up = next((item for item in rank[1:] if item["model_id"] != best["model_id"]), rank[1])
    selected_models = [best["name"], runner_up["name"]]
    pool = _pool(rank, settings["blend_pool_size"])
    final_recipe = fit_blend({name: base_probabilities[name] for name in pool}, data.y,
                            seed=validation["model_seed"], n_trials=blend_settings["n_trials"],
                            allow_class_multipliers=blend_settings["allow_class_multipliers"])
    final_recipe["selection_scope"] = SELECTION_SCOPE
    single_metrics, blend_metrics = metrics(data.y, selected_oof), metrics(data.y, blend_oof)
    gain_single = blend_metrics["macro_f1"] - single_metrics["macro_f1"]
    gain_base = blend_metrics["macro_f1"] - best["oof_macro_f1"]
    wins = sum(item["blend_metrics"]["macro_f1"] > item["single_metrics"]["macro_f1"] + 1e-12 for item in outer_details)
    gain_pass = (gain_single + 1e-12 >= settings["minimum_blend_gain"]
                 and gain_base + 1e-12 >= settings["minimum_blend_gain"])
    wins_pass = wins >= settings["min_blend_fold_wins"]
    eligible = bool(gain_pass and wins_pass)
    gate = {"minimum_blend_gain": settings["minimum_blend_gain"],
            "min_blend_fold_wins": settings["min_blend_fold_wins"], "blend_fold_wins": wins,
            "outer_fold_count": validation["n_splits"],
            "nested_single_macro_f1": single_metrics["macro_f1"],
            "nested_blend_macro_f1": blend_metrics["macro_f1"],
            "best_fixed_base_model_name": best["name"], "best_fixed_base_macro_f1": best["oof_macro_f1"],
            "gain_over_nested_single": gain_single, "gain_over_best_fixed_base": gain_base,
            "gain_conditions_met": bool(gain_pass), "fold_win_condition_met": bool(wins_pass),
            "eligible": eligible, "evaluation_scope": EVALUATION_SCOPE}
    report = {"schema_version": 1, "candidate_model_names": [model["name"] for model in config["models"]],
              "rank": rank, "selected_models": selected_models,
              "base_model_names": final_recipe["model_ids"],
              "nested_single_metrics": single_metrics, "nested_blend_metrics": blend_metrics,
              "outer_folds": outer_details, "full_oof_recipe": final_recipe,
              "full_oof_blend_metrics": metrics(data.y, apply_blend({name: base_probabilities[name] for name in pool}, final_recipe)),
              "blend_export_eligible": eligible, "blend_gate": gate,
              "selection_scope": SELECTION_SCOPE, "evaluation_scope": EVALUATION_SCOPE,
              "outer_fold_protocol": protocol, "full_oof_pool": pool,
              "cross_fitted_predictions": "search/cross_fitted.npz"}
    save_json(out / "selection.json", {"schema_version": 1, "rank": rank, "selected_models": selected_models,
                                       "full_oof_pool": pool, "full_oof_recipe": final_recipe,
                                       "blend_export_eligible": eligible, "selection_scope": SELECTION_SCOPE})
    save_json(out / "search_summary.json", report)
    return report
