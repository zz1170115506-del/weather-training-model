"""Fold isolation and reproducibility checks on small synthetic weather data."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1 import LABELS
from task1.artifacts import finish_run, new_run
from task1.config import resolve_config
from task1.data import CATEGORICAL, FEATURES, Dataset, load_data
from task1.ensemble import predict_blend
from task1.features import Features
from task1.models import align_probabilities, predict_probabilities
from task1.pipeline import run_cv, run_finalize, run_predict
from task1.submission import validate_submission_file, write_submission
from task1.validation import cross_validate, make_folds


def synthetic_dataset():
    """Four well-separated classes, independent test IDs and reordered sample."""
    rows = []
    clouds = ["clear", "cloudy", "overcast", "partly cloudy"]
    seasons = ["Summer", "Autumn", "Spring", "Winter"]
    locations = ["coastal", "inland", "mountain"]
    for label in LABELS:
        for offset in range(8):
            rows.append({
                "id": 10 * label + offset,
                "Temperature": 20.0 * label + offset / 100,
                "Humidity": 40.0 + offset,
                "Wind Speed": 1.0 + offset / 10,
                "Precipitation (%)": 10.0 * label + offset / 10,
                "Atmospheric Pressure": 1000.0 + label,
                "UV Index": float(label),
                "Visibility (km)": 10.0 - label,
                "Cloud Cover": clouds[label - 1],
                "Season": seasons[offset % 4],
                "Location": locations[offset % 3],
                "label": label,
            })
    train = pd.DataFrame(rows)
    test = train.iloc[[1, 10, 19, 28]].drop(columns="label").reset_index(drop=True)
    test["id"] = [1001, 1002, 1003, 1004]
    sample = pd.DataFrame({"id": [1003, 1001, 1004, 1002], "label": ["Rainy"] * 4})
    audit = {"source_sha256": {
        "train.csv": "synthetic-train", "test.csv": "synthetic-test",
        "sample_submission.csv": "synthetic-sample",
    }}
    return Dataset(train, test, sample, audit, np.arange(len(train)))


def small_config():
    return resolve_config({
        "schema_version": 1,
        "validation": {"n_splits": 3, "split_seed": 83, "model_seed": 91},
        "models": [{"name": "synthetic_cart", "model_id": "numpy_cart",
                    "model_params": {"max_depth": 4, "min_samples_leaf": 1}}],
    })


class FeatureIsolationTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_dataset()

    def test_unknown_categories_and_numeric_imputation_use_training_only(self):
        fit = self.data.X.iloc[:3].copy()
        fit["Temperature"] = [1.0, 3.0, np.nan]
        fit["Cloud Cover"] = ["clear", "clear", np.nan]
        valid = self.data.X.iloc[:2].copy()
        valid["Temperature"] = [1000.0, np.nan]
        valid["Cloud Cover"] = ["validation-only-cloud", np.nan]
        fit_before, valid_before = fit.copy(deep=True), valid.copy(deep=True)

        features = Features().fit(fit)
        transformed = features.transform(valid)
        numeric_fill = features.transformer_.named_transformers_["numeric"].named_steps["fill"]
        temperature_column = features.numeric_.index("Temperature")
        self.assertEqual(numeric_fill.statistics_[temperature_column], 2.0)
        self.assertEqual(transformed[0, temperature_column], 1000.0)
        self.assertEqual(transformed[1, temperature_column], 2.0)

        encoder = features.transformer_.named_transformers_["categorical"].named_steps["encode"]
        self.assertEqual(encoder.categories_[CATEGORICAL.index("Cloud Cover")].tolist(), ["clear"])
        cloud_columns = features.groups_ == "Cloud Cover"
        np.testing.assert_array_equal(transformed[0, cloud_columns], [0.0])
        np.testing.assert_array_equal(transformed[1, cloud_columns], [1.0])
        self.assertEqual(features.fitted_rows_, 3)
        self.assertEqual(len(features.groups_), transformed.shape[1])
        self.assertTrue(np.isfinite(transformed).all())
        pd.testing.assert_frame_equal(fit, fit_before)
        pd.testing.assert_frame_equal(valid, valid_before)

    def test_native_category_transform_retains_unseen_values_without_refitting(self):
        fit = self.data.X.iloc[:3].copy()
        fit["Temperature"] = [1.0, 3.0, np.nan]
        valid = self.data.X.iloc[:2].copy()
        valid["Temperature"] = [9999.0, np.nan]
        valid["Cloud Cover"] = ["unseen-cloud", np.nan]
        features = Features(native_categories=True).fit(fit)
        transformed = features.transform(valid)
        self.assertEqual(features.medians_["Temperature"], 2.0)
        self.assertEqual(transformed["Temperature"].tolist(), [9999.0, 2.0])
        self.assertEqual(transformed["Cloud Cover"].tolist(), ["unseen-cloud", "__MISSING__"])
        self.assertEqual(features.fitted_rows_, 3)
        self.assertEqual(list(transformed), FEATURES)

    def test_cross_validation_fits_features_only_on_each_training_partition(self):
        data, config = self.data, small_config()
        folds, _ = make_folds(data.y, data.train.id.to_numpy(), data.groups, n_splits=3, seed=83)
        observed_fit_indices = []
        original_fit = Features.fit

        def record_fit(instance, X):
            observed_fit_indices.append(tuple(X.index.tolist()))
            return original_fit(instance, X)

        with patch.object(Features, "fit", new=record_fit):
            summary, oof = cross_validate(data, config["models"][0], folds,
                                          model_seed=91, verbose=False)
        self.assertEqual(len(observed_fit_indices), 3)
        validation_count = np.zeros(len(data.train), dtype=int)
        for fold, actual in enumerate(observed_fit_indices):
            expected_fit = np.flatnonzero(folds != fold)
            expected_valid = np.flatnonzero(folds == fold)
            np.testing.assert_array_equal(actual, expected_fit)
            self.assertFalse(set(actual) & set(expected_valid))
            validation_count[expected_valid] += 1
            self.assertEqual(summary["folds"][fold]["preprocessing_fitted_rows"], len(expected_fit))
            self.assertEqual(summary["folds"][fold]["fit_rows"], len(expected_fit))
            self.assertFalse(summary["folds"][fold]["early_stopping_uses_evaluation_fold"])
        np.testing.assert_array_equal(validation_count, np.ones(len(data.train), dtype=int))
        self.assertEqual(oof.shape, (len(data.train), 4))
        self.assertTrue(np.isfinite(oof).all())
        np.testing.assert_allclose(oof.sum(axis=1), 1)
        np.testing.assert_array_equal(np.asarray(LABELS)[oof.argmax(axis=1)], data.y)


class ProtocolTests(unittest.TestCase):
    def test_duplicate_feature_groups_never_cross_folds_and_protocol_is_stable(self):
        y = np.repeat(np.asarray(LABELS), 8)
        ids = np.arange(100, 132)
        groups = np.repeat(np.arange(16), 2)
        folds, metadata = make_folds(y, ids, groups, n_splits=3, seed=83)
        replay, replay_metadata = make_folds(y.copy(), ids.copy(), groups.copy(), n_splits=3, seed=83)
        np.testing.assert_array_equal(folds, replay)
        self.assertEqual(metadata, replay_metadata)
        self.assertEqual(metadata["method"], "StratifiedGroupKFold")
        self.assertEqual(len(metadata["fold_sha256"]), 64)
        for fold in range(3):
            fit_groups = set(groups[folds != fold])
            valid_groups = set(groups[folds == fold])
            self.assertFalse(fit_groups & valid_groups)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(folds[groups == group])), 1)
        other_folds, other_metadata = make_folds(y, ids, groups, n_splits=3, seed=84)
        self.assertFalse(np.array_equal(folds, other_folds))
        self.assertNotEqual(metadata["fold_sha256"], other_metadata["fold_sha256"])

    def test_class_probabilities_align_to_fixed_global_order(self):
        actual = align_probabilities(np.array([[0.2, 0.5, 0.3]]), np.array([4, 1, 3]))
        np.testing.assert_allclose(actual, [[0.5, 0, 0.3, 0.2]])
        for values, classes in (
            (np.array([[0.5, 0.5]]), [1, 1]),
            (np.array([[0.5, 0.5]]), [1, 9]),
            (np.array([[np.nan, 1]]), [1, 2]),
            (np.array([[0.2, 0.2]]), [1, 2]),
            (np.array([0.5, 0.5]), [1, 2]),
        ):
            with self.subTest(classes=classes, values=values):
                with self.assertRaises(ValueError):
                    align_probabilities(values, classes)

    def test_config_rejects_unsafe_names_duplicate_names_and_invalid_seeds(self):
        raw = {"schema_version": 1,
               "models": [{"name": "valid_model", "model_id": "numpy_cart"}]}
        before = copy.deepcopy(raw)
        resolved = resolve_config(raw)
        self.assertEqual(raw, before)
        self.assertEqual(resolved["validation"]["n_splits"], 5)
        for name in ("../outside", "..", "/outside", r"C:\outside", "two words", "a/b", r"a\b"):
            bad = copy.deepcopy(raw)
            bad["models"][0]["name"] = name
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    resolve_config(bad)
        bad = copy.deepcopy(raw)
        bad["models"].append(copy.deepcopy(bad["models"][0]))
        with self.assertRaises(ValueError):
            resolve_config(bad)
        for key in ("split_seed", "model_seed"):
            for value in (-1, True, 1.5, "5434", None):
                bad = copy.deepcopy(raw)
                bad["validation"] = {key: value}
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        resolve_config(bad)
        for key in ("random_state", "random_seed"):
            bad = copy.deepcopy(raw)
            bad["models"][0]["model_params"] = {key: 123}
            with self.assertRaises(ValueError):
                resolve_config(bad)

    def test_model_names_cannot_collide_with_npz_arguments_or_record_fields(self):
        for name in ("file", "allow_pickle", "ids", "y", "folds", "blend", "equal", "probabilities"):
            raw = {"schema_version": 1,
                   "models": [{"name": name, "model_id": "numpy_cart"}]}
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    resolve_config(raw)
        resolved = resolve_config({"schema_version": 1,
                                   "models": [{"name": "file_model", "model_id": "numpy_cart"}]})
        self.assertEqual(resolved["models"][0]["name"], "file_model")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.output_area = Path(__file__).resolve().parents[2] / "outputs"
        self.output_area.mkdir(exist_ok=True)
        self.directory = self.output_area / f"flow_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup_directory)

    def cleanup_directory(self):
        target = self.directory.resolve()
        if target.parent != self.output_area.resolve() or not target.name.startswith("flow_test_"):
            raise RuntimeError("Refusing to remove a test directory outside its output area.")
        shutil.rmtree(target)

    def write_dataset(self, data, path, random_state):
        path.mkdir()
        data.train.sample(frac=1, random_state=random_state).to_csv(path / "train.csv", index=False)
        data.test.to_csv(path / "test.csv", index=False)
        data.sample.to_csv(path / "sample_submission.csv", index=False)

    def test_loader_canonicalizes_training_order_before_shared_folds(self):
        data = synthetic_dataset()
        first_path, second_path = self.directory / "data_a", self.directory / "data_b"
        self.write_dataset(data, first_path, 12)
        self.write_dataset(data, second_path, 65)
        first, second = load_data(first_path), load_data(second_path)
        pd.testing.assert_frame_equal(first.train, second.train)
        np.testing.assert_array_equal(first.groups, second.groups)
        folds_a, meta_a = make_folds(first.y, first.train.id.to_numpy(), first.groups, 3, 83)
        folds_b, meta_b = make_folds(second.y, second.train.id.to_numpy(), second.groups, 3, 83)
        np.testing.assert_array_equal(folds_a, folds_b)
        self.assertEqual(meta_a["fold_sha256"], meta_b["fold_sha256"])
        self.assertNotEqual(first.audit["source_sha256"]["train.csv"],
                            second.audit["source_sha256"]["train.csv"])

    def test_loader_and_submission_preserve_leading_zero_record_ids(self):
        data = synthetic_dataset()
        for frame in (data.train, data.test, data.sample):
            frame["id"] = frame["id"].map(lambda value: f"{value:05d}")
        data_path = self.directory / "leading_zero_data"
        self.write_dataset(data, data_path, 13)
        loaded = load_data(data_path)
        self.assertEqual(loaded.train.id.tolist(), sorted(data.train.id.tolist()))
        self.assertEqual(loaded.test.id.tolist(), ["01001", "01002", "01003", "01004"])
        self.assertEqual(loaded.sample.id.tolist(), ["01003", "01001", "01004", "01002"])
        output = self.directory / "leading_zero_submission.csv"
        metadata = write_submission(loaded.test.id.to_numpy(), [1, 2, 3, 4], loaded.sample,
                                    output, "integer")
        self.assertEqual(output.read_bytes(), b"id,label\n01003,3\n01001,1\n01004,4\n01002,2\n")
        self.assertEqual(metadata, validate_submission_file(output, loaded.sample, "integer"))

    def test_experiment_artifacts_can_complete_without_a_git_executable(self):
        with patch("task1.artifacts.subprocess.run", side_effect=OSError("git unavailable")):
            output = new_run("cv", small_config(), self.directory)
            finish_run(output, {"result": {"synthetic": True}})
        manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(manifest["git"]["available"])
        self.assertIsNone(manifest["git"]["commit"])
        self.assertIsNone(manifest["git"]["branch"])
        self.assertTrue(manifest["result"]["synthetic"])
        self.assertIn("resolved_config.json", manifest["files_sha256"])

    def test_single_model_class_corrections_are_learned_from_oof_predictions(self):
        data, config = synthetic_dataset(), small_config()
        config["blend"] = {"n_trials": 0, "allow_class_multipliers": True}
        calibrated_oof = np.full((len(data.y), 4), 0.01)
        calibrated_oof[np.arange(len(data.y)), data.y - 1] = 0.97
        # Class 2 is consistently near the decision boundary. A small class
        # correction can recover it; blindly using the single-model shortcut
        # would neither learn this correction nor produce a fold protocol.
        calibrated_oof[data.y == 2] = [0.49, 0.48, 0.02, 0.01]
        output = self.directory / "calibrated_single"
        output.mkdir()
        with patch("task1.pipeline.cross_validate", return_value=({}, calibrated_oof)):
            result = run_finalize(data, config, output, "integer")
        self.assertIsNotNone(result["weight_selection_protocol"])
        selected = result["recipe"]
        self.assertGreater(selected["selection_delta"], 0)
        self.assertEqual(selected["selection_macro_f1"], 1)
        self.assertNotEqual(selected["class_multipliers"], [1, 1, 1, 1])
        np.testing.assert_array_equal(
            predict_blend({"synthetic_cart": calibrated_oof}, selected), data.y)

    def test_single_model_cv_finalize_reload_and_predict_reproduce_same_csv(self):
        data, config = synthetic_dataset(), small_config()
        cv_out = self.directory / "cv"
        cv_out.mkdir()
        cv_summary = run_cv(data, config, cv_out)
        self.assertEqual(len(cv_summary["metrics"]), 1)
        saved_oof = np.load(cv_out / "models" / "synthetic_cart" / "oof.npz")
        np.testing.assert_array_equal(saved_oof["ids"], data.train.id.to_numpy())
        self.assertEqual(saved_oof["probabilities"].shape, (len(data.y), 4))
        np.testing.assert_allclose(saved_oof["probabilities"].sum(axis=1), 1)
        self.assertEqual(cv_summary["metrics"][0]["oof_macro_f1"], 1)

        final_out = self.directory / "final"
        final_out.mkdir()
        result = run_finalize(data, config, final_out, "name")
        bundle_path = final_out / "model_bundle.joblib"
        self.assertTrue(result["saved_model_reload_exact_match"])
        self.assertFalse(result["platform_acceptance_verified"])
        loaded = joblib.load(bundle_path)
        predictions = predict_probabilities(loaded["models"]["synthetic_cart"], data.test[FEATURES])
        np.testing.assert_array_equal(np.asarray(LABELS)[predictions.argmax(axis=1)], [1, 2, 3, 4])

        name_path = self.directory / "replayed_name.csv"
        integer_path = self.directory / "replayed_integer.csv"
        name_metadata = run_predict(data, bundle_path, name_path, "name")
        integer_metadata = run_predict(data, bundle_path, integer_path, "integer")
        self.assertEqual(name_path.read_bytes(), (final_out / "submission.csv").read_bytes())
        self.assertEqual(integer_path.read_bytes(), (final_out / "submission_integer.csv").read_bytes())
        self.assertEqual(name_metadata["canonical_prediction_sha256"],
                         integer_metadata["canonical_prediction_sha256"])
        self.assertEqual(name_metadata["csv_sha256"], hashlib.sha256(name_path.read_bytes()).hexdigest())
        self.assertEqual(name_metadata, validate_submission_file(name_path, data.sample, "name"))
        self.assertEqual(name_path.read_bytes(), b"id,label\n1003,Rainy\n1001,Sunny\n1004,Snowy\n1002,Cloudy\n")
        manifest = json.loads((final_out / "submission_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["full_test_rows"], 4)
        self.assertEqual(manifest["submission"]["csv_sha256"], name_metadata["csv_sha256"])

        # A reproduction mismatch must not overwrite an existing output.
        # Deliberately change only the frozen expectation, keeping the fitted
        # predictor intact, to simulate a bundle/code disagreement.
        loaded["expected_predictions"] = np.roll(loaded["expected_predictions"], 1)
        conflicting_bundle = self.directory / "conflicting_bundle.joblib"
        joblib.dump(loaded, conflicting_bundle)
        protected_output = self.directory / "existing_output.csv"
        protected_output.write_bytes(b"existing output must survive\n")
        with self.assertRaises(RuntimeError):
            run_predict(data, conflicting_bundle, protected_output, "name")
        self.assertEqual(protected_output.read_bytes(), b"existing output must survive\n")


if __name__ == "__main__":
    unittest.main()
