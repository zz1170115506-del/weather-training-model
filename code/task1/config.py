"""Versioned, deliberately small JSON configuration schema."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import re

from .models import MODEL_IDS


def validate_expected_source_hashes(value):
    """A pinned course dataset has exactly three immutable input digests."""
    names = {"train.csv", "test.csv", "sample_submission.csv"}
    if (not isinstance(value, dict) or set(value) != names
            or any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                   for digest in value.values())):
        raise ValueError("expected_source_sha256 requires exactly train.csv, test.csv and sample_submission.csv with lowercase 64-character SHA256 digests.")
    return dict(value)


def require_course_only_config(config):
    """Reject retired source-assisted recipes, including direct API callers."""
    if "external_data" in config:
        raise ValueError("External data is disabled. Use only the original course train.csv; source-assisted configurations cannot run.")
    if "expected_source_sha256" in config:
        validate_expected_source_hashes(config["expected_source_sha256"])
    models = config.get("models", [config])
    if not isinstance(models, list):
        return  # The full schema validator reports invalid model containers.
    for model in models:
        if not isinstance(model, dict):
            continue
        if "use_external_data" in model:
            value = model["use_external_data"]
            if not isinstance(value, bool):
                raise ValueError("use_external_data must be boolean; external data is disabled.")
            if value:
                raise ValueError("External data is disabled. use_external_data=true is forbidden.")


def validate_experiment(value):
    """Validate optional human-authored provenance without inventing rounds."""
    required = {"round", "label", "parent_run_id", "change_note"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("experiment requires exactly round, label, parent_run_id and change_note.")
    round_number = value["round"]
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError("experiment.round must be a positive integer.")
    for field, limit in (("label", 80), ("parent_run_id", 160)):
        text = value[field]
        if not isinstance(text, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0," + str(limit - 1) + "}", text) is None:
            raise ValueError(f"experiment.{field} must be a safe name of at most {limit} characters.")
    note = value["change_note"]
    if (not isinstance(note, str) or not note.strip() or len(note) > 500
            or any(ord(character) < 32 or ord(character) == 127 for character in note)):
        raise ValueError("experiment.change_note must be a nonempty single-line string of at most 500 characters.")
    return {"round": round_number, "label": value["label"],
            "parent_run_id": value["parent_run_id"], "change_note": note.strip()}


def resolve_config(raw):
    result = deepcopy(raw)
    require_course_only_config(result)
    if result.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1.")
    if not isinstance(result.get("models"), list) or not result["models"]:
        raise ValueError("models must be a nonempty list.")
    if "experiment" in result:
        result["experiment"] = validate_experiment(result["experiment"])
    if "diagnostics" in result:
        diagnostics = result["diagnostics"]
        if not isinstance(diagnostics, dict) or set(diagnostics) != {"learning_curve_every"}:
            raise ValueError("diagnostics supports only learning_curve_every.")
        every = diagnostics["learning_curve_every"]
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError("diagnostics.learning_curve_every must be a positive integer.")
    validation = {"n_splits": 5, "split_seed": 5434, "model_seed": 5434} | result.get("validation", {})
    if set(validation) != {"n_splits", "split_seed", "model_seed"}:
        raise ValueError("Unknown validation option.")
    if isinstance(validation["n_splits"], bool) or not isinstance(validation["n_splits"], int) or validation["n_splits"] < 2:
        raise ValueError("n_splits must be an integer >=2.")
    for key in ("split_seed", "model_seed"):
        if isinstance(validation[key], bool) or not isinstance(validation[key], int) or validation[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    result["validation"] = validation
    finalization = {"method": "full_refit"} | result.get("finalization", {})
    if finalization != {"method": "full_refit"}:
        raise ValueError("Only finalization.method=full_refit is implemented.")
    result["finalization"] = finalization
    result.setdefault("prediction_strategy", "independent_candidates")
    if result["prediction_strategy"] not in ("independent_candidates", "learned_blend"):
        raise ValueError("prediction_strategy must be independent_candidates or learned_blend.")
    blend = {"n_trials": 128, "allow_class_multipliers": False} | result.get("blend", {})
    if set(blend) != {"n_trials", "allow_class_multipliers"}:
        raise ValueError("Unknown blend option.")
    if isinstance(blend["n_trials"], bool) or not isinstance(blend["n_trials"], int) or blend["n_trials"] < 0:
        raise ValueError("blend.n_trials must be a nonnegative integer.")
    if not isinstance(blend["allow_class_multipliers"], bool):
        raise ValueError("blend.allow_class_multipliers must be boolean.")
    result["blend"] = blend
    if "search" in result:
        search = result["search"]
        required = {"inner_splits", "blend_pool_size", "minimum_blend_gain", "min_blend_fold_wins"}
        if not isinstance(search, dict) or set(search) != required:
            raise ValueError("search requires inner_splits, blend_pool_size, minimum_blend_gain and min_blend_fold_wins.")
        for key, lower in (("inner_splits", 2), ("blend_pool_size", 2), ("min_blend_fold_wins", 1)):
            if isinstance(search[key], bool) or not isinstance(search[key], int) or search[key] < lower:
                raise ValueError(f"search.{key} must be an integer >= {lower}.")
        if search["min_blend_fold_wins"] > validation["n_splits"]:
            raise ValueError("search.min_blend_fold_wins exceeds the outer fold count.")
        gain = search["minimum_blend_gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not 0 <= gain <= 1:
            raise ValueError("search.minimum_blend_gain must be finite and in [0,1].")
        if search["blend_pool_size"] > len(result["models"]):
            raise ValueError("search.blend_pool_size exceeds the number of candidates.")
        if not 2 <= len(result["models"]) <= 16:
            raise ValueError("Nested search requires between 2 and 16 complete candidate recipes.")
    if "cv_selection" in result:
        selection = result["cv_selection"]
        if "search" in result:
            raise ValueError("cv_selection and nested search are separate evaluation modes.")
        if (not isinstance(selection, dict) or set(selection) != {"method", "n_exports"}
                or selection["method"] != "rank_fixed_recipes"
                or isinstance(selection["n_exports"], bool)
                or not isinstance(selection["n_exports"], int)
                or not 1 <= selection["n_exports"] <= len(result["models"])):
            raise ValueError("cv_selection requires method=rank_fixed_recipes and valid n_exports.")
    if "inner_splits" in result and (isinstance(result["inner_splits"], bool)
                                      or not isinstance(result["inner_splits"], int) or result["inner_splits"] < 2):
        raise ValueError("inner_splits must be an integer >=2.")
    names = set()
    for model in result["models"]:
        name = model.get("name", "")
        reserved = {"ids", "y", "folds", "blend", "equal", "probabilities", "file", "allow_pickle"}
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) or name in names or name in reserved:
            raise ValueError("Model names must be unique safe directory names.")
        if model.get("model_id") not in MODEL_IDS:
            raise ValueError(f"Unknown model_id: {model.get('model_id')}")
        names.add(name)
        model.setdefault("model_params", {})
        if not isinstance(model["model_params"], dict):
            raise ValueError("model_params must be an object.")
        if "random_state" in model["model_params"] or "random_seed" in model["model_params"]:
            raise ValueError("Use validation.model_seed, not model_params random_state/random_seed.")
        if "early_stopping_rounds" in model:
            patience = model["early_stopping_rounds"]
            if (model["model_id"] not in ("hgb", "hgb_native", "catboost", "xgboost")
                    or isinstance(patience, bool) or not isinstance(patience, int) or patience < 1):
                raise ValueError("early_stopping_rounds requires a supported boosting model and positive integer patience.")
        features = {"percentage_policy": "retain", "out_of_range_flags": False,
                    "interactions": False} | model.get("features", {})
        if set(features) != {"percentage_policy", "out_of_range_flags", "interactions"}:
            raise ValueError("Unknown feature option.")
        if features["percentage_policy"] not in ("retain", "clip"):
            raise ValueError("Invalid percentage_policy.")
        for key in ("out_of_range_flags", "interactions"):
            if not isinstance(features[key], bool):
                raise ValueError(f"{key} must be boolean.")
        model["features"] = features
        model.setdefault("group_features", False)
        if not isinstance(model["group_features"], bool):
            raise ValueError("group_features must be boolean.")
        if "crossfit_weighting" in model:
            if "noise_weighting" in model:
                raise ValueError("Choose noise_weighting or crossfit_weighting, not both.")
            if model["model_id"] not in ("hgb", "hgb_native", "bagged_hgb", "sklearn_forest", "catboost", "xgboost"):
                raise ValueError("crossfit_weighting requires a model supporting sample_weight.")
            from .crossfit_weights import resolve_settings
            model["crossfit_weighting"] = resolve_settings(model["crossfit_weighting"])
        if "noise_weighting" in model and not isinstance(model["noise_weighting"], dict):
            raise ValueError("noise_weighting must be a parameter object.")
        if model.get("noise_weighting"):
            defaults = {"floor": 0.2, "n_bins": 24, "alpha": 0.5,
                        "contamination": 0.1, "max_iter": 30, "tol": 1e-5,
                        "min_contamination": 0.01, "max_contamination": 0.4}
            if set(model["noise_weighting"]) - set(defaults):
                raise ValueError("Unknown noise_weighting option.")
            model["noise_weighting"] = defaults | model["noise_weighting"]
            floor = model["noise_weighting"]["floor"]
            if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 <= floor <= 1:
                raise ValueError("noise_weighting.floor must be in [0,1].")
    training = result.get("training", {})
    if not isinstance(training, dict) or set(training) - {"export_models"}:
        raise ValueError("training supports only export_models.")
    exports = training.get("export_models", [result["models"][0]["name"]])
    if not isinstance(exports, list) or not exports or len(exports) != len(set(exports)):
        raise ValueError("training.export_models must be a nonempty unique list.")
    if any(name not in names for name in exports):
        raise ValueError("Every export model must occur in models.")
    result["training"] = {"export_models": exports}
    return result


def load_config(path):
    return resolve_config(json.loads(Path(path).read_text(encoding="utf-8")))
