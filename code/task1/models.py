"""Fresh model factories and fold-isolated training, including early stopping."""
from __future__ import annotations

from time import perf_counter
import numpy as np
from sklearn.model_selection import train_test_split

from . import LABELS
from .features import Features

MODEL_IDS = ("majority", "logistic", "numpy_cart", "numpy_forest", "sklearn_cart",
             "sklearn_forest", "extra_trees", "hgb", "hgb_native", "catboost", "svc", "histogram_nb")


def make_model(config, seed, groups=None, iterations=None):
    name = config["model_id"]
    params = dict(config.get("model_params", {}))
    if name not in MODEL_IDS:
        raise ValueError(f"Unknown model_id {name}; choose {MODEL_IDS}.")
    if name == "majority":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy="most_frequent", **params)
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=seed, **({"max_iter": 2000} | params))
    if name in ("numpy_cart", "numpy_forest"):
        from weather_models import CARTClassifier, RandomForestClassifier
        if config.get("group_features", False):
            params["feature_groups"] = groups
        model = CARTClassifier if name == "numpy_cart" else RandomForestClassifier
        return model(random_state=seed, **params)
    if name == "sklearn_cart":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed, **params)
    if name in ("sklearn_forest", "extra_trees"):
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        model = RandomForestClassifier if name == "sklearn_forest" else ExtraTreesClassifier
        return model(random_state=seed, **({"n_jobs": 1} | params))
    if name == "histogram_nb":
        from generative_weather import ContaminationHistogramNB
        return ContaminationHistogramNB(categorical_features=list(CATEGORICAL_COLUMNS),
                                         class_labels=list(LABELS), **params)
    if name in ("hgb", "hgb_native"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        # No outer-validation-driven early stopping. Optional selection below
        # uses a separate inner split, then refits the entire allowed training set.
        params = {"early_stopping": False, "categorical_features": "from_dtype" if name == "hgb_native" else None} | params
        if params["early_stopping"] is not False:
            raise ValueError("Use early_stopping_rounds in config, not model early_stopping.")
        if iterations is not None:
            params["max_iter"] = iterations
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise ImportError("Install code/requirements-boosting.txt for CatBoost.") from error
        params = {"iterations": 600, "loss_function": "MultiClass", "verbose": False,
                  "thread_count": 1, "allow_writing_files": False} | params
        if iterations is not None:
            params["iterations"] = iterations
        return CatBoostClassifier(random_seed=seed, **params)
    from sklearn.svm import SVC
    return SVC(probability=True, random_state=seed, **params)


def align_probabilities(probabilities, classes):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability shape/classes mismatch.")
    if len(set(classes)) != len(classes) or not set(classes).issubset(LABELS):
        raise ValueError("Unknown or repeated model class.")
    result = np.zeros((len(probabilities), 4))
    for j, label in enumerate(classes):
        result[:, LABELS.index(int(label))] = probabilities[:, j]
    if not np.isfinite(result).all() or (result < -1e-12).any():
        raise ValueError("Invalid probabilities.")
    if not np.allclose(result.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Probability rows must sum to 1.")
    return result


def predict_probabilities(bundle, X):
    values = bundle["features"].transform(X)
    model = bundle["model"]
    return align_probabilities(model.predict_proba(values), model.classes_)


def fit_model(X, y, config, seed):
    """Only X/y passed here are ever used to choose iteration counts."""
    started = perf_counter()
    name = config["model_id"]
    rounds = config.get("early_stopping_rounds")
    iterations = None
    if rounds is not None:
        if name != "catboost" or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("early_stopping_rounds currently supports positive CatBoost patience only.")
        inner_fit, inner_stop = train_test_split(np.arange(len(y)), test_size=0.15,
                                                 stratify=y, random_state=seed + 100003)
        inner_features = Features(config.get("features"), native_categories=True)
        inner_X = inner_features.fit_transform(X.iloc[inner_fit])
        stop_X = inner_features.transform(X.iloc[inner_stop])
        selector = make_model(config, seed)
        selector.fit(inner_X, np.asarray(y)[inner_fit], cat_features=list(CATEGORICAL_COLUMNS),
                     eval_set=(stop_X, np.asarray(y)[inner_stop]),
                     early_stopping_rounds=rounds, use_best_model=True)
        iterations = int(selector.get_best_iteration()) + 1
        if iterations < 1:
            iterations = int(selector.tree_count_)
    category_mode = "hgb" if name == "hgb_native" else name in ("catboost", "histogram_nb")
    features = Features(config.get("features"), native_categories=category_mode,
                        scale=name in ("logistic", "svc"))
    preprocess_started = perf_counter()
    values = features.fit_transform(X)
    preprocessing_seconds = perf_counter() - preprocess_started
    model = make_model(config, seed, features.groups_, iterations)
    sample_weight = None
    noise_diagnostics = None
    if config.get("noise_weighting"):
        weighting_started = perf_counter()
        if name not in ("hgb", "hgb_native", "sklearn_forest", "catboost"):
            raise ValueError("noise_weighting requires a model supporting sample_weight.")
        from generative_weather import ContaminationHistogramNB
        settings = dict(config["noise_weighting"])
        floor = settings.pop("floor", 0.2)
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 <= floor <= 1:
            raise ValueError("noise_weighting.floor must be in [0,1].")
        native = Features(config.get("features"), native_categories=True)
        weight_X = native.fit_transform(X)
        weighter = ContaminationHistogramNB(
            categorical_features=list(CATEGORICAL_COLUMNS), class_labels=list(LABELS),
            **({"n_bins": 24, "alpha": 0.5, "contamination": 0.1, "max_iter": 30} | settings))
        weighter.fit(weight_X, y)
        sample_weight = floor + (1 - floor) * weighter.clean_responsibility_
        noise_diagnostics = {"weight_floor": floor, "mean_weight": float(sample_weight.mean()),
                             "weighter_parameters": weighter.get_params(),
                             "weighter_iterations": int(weighter.n_iter_),
                             "weighter_converged": bool(weighter.converged_),
                             "weighting_seconds": perf_counter() - weighting_started,
                             "weight_quantiles": np.quantile(sample_weight, [0, .1, .5, .9, 1]).tolist(),
                             "estimated_contamination": float(weighter.contamination_),
                             "scope": "Training-only generative responsibilities, not known label-error indicators or an estimate of the true noise rate."}
    fit_started = perf_counter()
    if name == "catboost":
        model.fit(values, y, cat_features=list(CATEGORICAL_COLUMNS), sample_weight=sample_weight)
    elif sample_weight is not None:
        model.fit(values, y, sample_weight=sample_weight)
    else:
        model.fit(values, y)
    fit_seconds = perf_counter() - fit_started
    bundle = {"features": features, "model": model, "config": config,
              "labels": list(LABELS), "seed": seed}
    metadata = {"preprocessing_seconds": preprocessing_seconds, "fit_seconds": fit_seconds,
                "training_seconds_including_inner_selection": perf_counter() - started,
                "selected_iterations": iterations,
                "preprocessing_fitted_rows": features.fitted_rows_,
                "encoded_feature_count": len(features.groups_),
                "effective_model_parameters": (model.get_all_params() if name == "catboost" else model.get_params()),
                "noise_weighting": noise_diagnostics,
                "early_stopping_uses_evaluation_fold": False}
    return bundle, metadata


# CatBoost uses column names so derived numeric columns do not shift indices.
CATEGORICAL_COLUMNS = ("Cloud Cover", "Season", "Location")
