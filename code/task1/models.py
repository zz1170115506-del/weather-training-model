"""Fresh model factories and fold-isolated training, including early stopping."""
from __future__ import annotations

from time import perf_counter
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import LABELS
from .features import Features

MODEL_IDS = ("majority", "logistic", "numpy_cart", "numpy_forest", "sklearn_cart",
             "sklearn_forest", "extra_trees", "hgb", "hgb_native", "catboost", "xgboost", "svc", "histogram_nb")


class WeatherXGBoost:
    """Keep weather labels 1..4 outside XGBoost's zero-based label interface."""

    def __init__(self, **params):
        self.params = dict(params)

    def get_params(self, deep=True):
        return dict(self.params)

    @staticmethod
    def _encode(y, require_all=False):
        values = np.asarray(y)
        if (values.ndim != 1 or values.dtype.kind not in "iuf" or not np.isfinite(values).all()
                or not np.isin(values, LABELS).all()
                or (require_all and set(values) != set(LABELS))):
            raise ValueError("XGBoost weather labels must be numeric 1..4; fit requires all four classes.")
        return values.astype(np.int64) - 1

    def fit(self, X, y, sample_weight=None, *, eval_set=None, early_stopping_rounds=None):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("Install code/requirements-boosting.txt for XGBoost.") from error
        params = self.get_params()
        if early_stopping_rounds is not None:
            if not eval_set:
                raise ValueError("XGBoost early stopping requires an explicit inner validation set.")
            params["early_stopping_rounds"] = early_stopping_rounds
        self.model_ = XGBClassifier(**params)
        encoded_eval = ([(values, self._encode(labels)) for values, labels in eval_set]
                        if eval_set is not None else None)
        self.model_.fit(X, self._encode(y, require_all=True), sample_weight=sample_weight,
                        eval_set=encoded_eval, verbose=False)
        self.classes_ = np.asarray(LABELS)
        self.n_iter_ = int(self.model_.get_booster().num_boosted_rounds())
        self.n_trees_per_iteration_ = len(LABELS) * int(params.get("num_parallel_tree", 1))
        self.best_iteration_ = (int(self.model_.best_iteration) if early_stopping_rounds is not None
                                else self.n_iter_ - 1)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit XGBoost before requesting probabilities.")
        return self.model_.predict_proba(X)

    def booster_configuration(self):
        return json.loads(self.model_.get_booster().save_config())

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


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
    if name == "xgboost":
        forbidden = {"early_stopping_rounds", "callbacks", "eval_set", "xgb_model"}
        if forbidden & set(params):
            raise ValueError("Configure early_stopping_rounds at model level; training state/callbacks cannot bypass the inner split.")
        params = {"n_estimators": 1600, "learning_rate": 0.025, "tree_method": "hist",
                  "device": "cpu", "n_jobs": 1, "objective": "multi:softprob",
                  "num_class": 4, "eval_metric": "mlogloss", "verbosity": 0} | params
        if params["objective"] != "multi:softprob" or params["num_class"] != 4:
            raise ValueError("XGBoost weather models require multi:softprob and num_class=4.")
        if iterations is not None:
            params["n_estimators"] = iterations
        return WeatherXGBoost(random_state=seed, **params)
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


def _inner_split(X, y, seed):
    """An internal stopping set never splits identical raw feature records."""
    y = np.asarray(y)
    if set(y) != set(LABELS) or min(int((y == label).sum()) for label in LABELS) < 2:
        raise ValueError("Inner iteration selection requires at least two records per weather class.")
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    split_seed = seed + 100003
    if len(np.unique(groups)) == len(groups):
        stop_rows = max(len(LABELS), int(np.ceil(0.15 * len(y))))
        fit, stop = train_test_split(np.arange(len(y)), test_size=stop_rows,
                                    stratify=y, random_state=split_seed)
        method = "StratifiedShuffleSplit"
    else:
        count = min(7, min(len(np.unique(groups[y == label])) for label in LABELS))
        if count < 2:
            raise ValueError("Inner stopping requires two independent feature groups per class.")
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=split_seed)
        choices = [(fit, stop) for fit, stop in splitter.split(np.zeros((len(y), 1)), y, groups)
                   if set(y[fit]) == set(LABELS) and set(y[stop]) == set(LABELS)]
        if not choices:
            raise ValueError("No class-complete group-disjoint inner stopping split exists.")
        fit, stop = choices[0]
        method = "StratifiedGroupKFold"
    if set(groups[fit]) & set(groups[stop]):
        raise RuntimeError("Identical raw features cross the inner stopping boundary.")
    return fit, stop, {"method": method, "seed": split_seed, "fit_rows": len(fit),
                       "stopping_rows": len(stop), "duplicate_feature_groups_disjoint": True}


def _training_weights(X, y, config):
    if not config.get("noise_weighting"):
        return None, None
    started = perf_counter()
    if config["model_id"] not in ("hgb", "hgb_native", "sklearn_forest", "catboost", "xgboost"):
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
    weights = floor + (1 - floor) * weighter.clean_responsibility_
    return weights, {"weight_floor": floor, "mean_weight": float(weights.mean()),
                     "weighter_parameters": weighter.get_params(),
                     "weighter_iterations": int(weighter.n_iter_),
                     "weighter_converged": bool(weighter.converged_),
                     "weighting_seconds": perf_counter() - started,
                     "weight_quantiles": np.quantile(weights, [0, .1, .5, .9, 1]).tolist(),
                     "estimated_contamination": float(weighter.contamination_),
                     "scope": "Training-only generative responsibilities, not known label-error indicators or an estimate of the true noise rate."}


def fit_model(X, y, config, seed):
    """Only X/y passed here are ever used to choose iteration counts."""
    started = perf_counter()
    name = config["model_id"]
    rounds = config.get("early_stopping_rounds")
    iterations = None
    iteration_selection = None
    iteration_selection_seconds = 0.0
    if rounds is not None:
        if name not in ("catboost", "hgb", "hgb_native", "xgboost") or isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("early_stopping_rounds requires positive integer patience and an HGB, CatBoost or XGBoost model.")
        selection_started = perf_counter()
        inner_fit, inner_stop, inner_protocol = _inner_split(X, y, seed)
        inner_features = Features(config.get("features"), native_categories=(
            "hgb" if name == "hgb_native" else name == "catboost"))
        inner_X = inner_features.fit_transform(X.iloc[inner_fit])
        stop_X = inner_features.transform(X.iloc[inner_stop])
        selector = make_model(config, seed)
        inner_weights, _ = _training_weights(X.iloc[inner_fit], np.asarray(y)[inner_fit], config)
        if name in ("hgb", "hgb_native"):
            selector.set_params(early_stopping=True, validation_fraction=None,
                                n_iter_no_change=rounds, scoring="loss")
            selector.fit(inner_X, np.asarray(y)[inner_fit], sample_weight=inner_weights,
                         X_val=stop_X, y_val=np.asarray(y)[inner_stop])
            # validation_score_[0] is the intercept-only baseline. n_iter_
            # includes patience iterations and must not be called the best count.
            scores = np.asarray(selector.validation_score_[1:], dtype=float)
            if not len(scores) or not np.isfinite(scores).all():
                raise RuntimeError("Inner HGB selection did not yield finite validation scores.")
            iterations = int(np.argmax(scores)) + 1
            stopped_iterations = int(selector.n_iter_)
            criterion = "inner validation negative loss; earliest best post-baseline stage"
        elif name == "catboost":
            selector.fit(inner_X, np.asarray(y)[inner_fit], cat_features=list(CATEGORICAL_COLUMNS),
                         sample_weight=inner_weights, eval_set=(stop_X, np.asarray(y)[inner_stop]),
                         early_stopping_rounds=rounds, use_best_model=True)
            iterations = int(selector.get_best_iteration()) + 1
            if iterations < 1:
                iterations = int(selector.tree_count_)
            stopped_iterations = max((len(history) for history in selector.get_evals_result().get("validation", {}).values()),
                                     default=int(selector.tree_count_))
            criterion = "inner validation metric, CatBoost best iteration"
        else:
            selector.fit(inner_X, np.asarray(y)[inner_fit], sample_weight=inner_weights,
                         eval_set=[(stop_X, np.asarray(y)[inner_stop])], early_stopping_rounds=rounds)
            iterations = selector.best_iteration_ + 1
            stopped_iterations = selector.n_iter_
            criterion = "inner validation mlogloss, XGBoost best iteration"
        iteration_selection_seconds = perf_counter() - selection_started
        iteration_selection = {**inner_protocol, "patience": rounds,
                               "inner_stopped_iterations": stopped_iterations,
                               "inner_best_iteration": iterations, "criterion": criterion,
                               "scope": "Only the supplied training partition; external evaluation labels are never passed. Refit all supplied rows at the selected count.",
                               "inner_preprocessing_fitted_rows": inner_features.fitted_rows_}
    category_mode = "hgb" if name == "hgb_native" else name in ("catboost", "histogram_nb")
    features = Features(config.get("features"), native_categories=category_mode,
                        scale=name in ("logistic", "svc"))
    preprocess_started = perf_counter()
    values = features.fit_transform(X)
    preprocessing_seconds = perf_counter() - preprocess_started
    model = make_model(config, seed, features.groups_, iterations)
    sample_weight, noise_diagnostics = _training_weights(X, y, config)
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
                "iteration_selection_seconds": iteration_selection_seconds,
                "iteration_selection": iteration_selection,
                "selected_iterations": iterations,
                "actual_iterations": (int(model.n_iter_) if name in ("hgb", "hgb_native", "xgboost")
                                      else int(model.tree_count_) if name == "catboost" else None),
                "trees_per_iteration": int(model.n_trees_per_iteration_) if name in ("hgb", "hgb_native", "xgboost") else None,
                "preprocessing_fitted_rows": features.fitted_rows_,
                "encoded_feature_count": len(features.groups_),
                "effective_model_parameters": (model.get_all_params() if name == "catboost" else model.get_params()),
                "noise_weighting": noise_diagnostics,
                "early_stopping_uses_evaluation_fold": False}
    if name == "xgboost":
        metadata["xgboost_booster_configuration"] = model.booster_configuration()
    return bundle, metadata


# CatBoost uses column names so derived numeric columns do not shift indices.
CATEGORICAL_COLUMNS = ("Cloud Cover", "Season", "Location")
