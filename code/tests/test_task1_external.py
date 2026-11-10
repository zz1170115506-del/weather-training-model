"""External overlap policy checks using local handwritten CSVs, never fitting."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1.data import Dataset, FEATURES, NUMERIC
from task1.external import attach_external, augment_training, feature_keys


def weather_row(value):
    return {"Temperature": float(value), "Humidity": 40.0 + value,
            "Wind Speed": value / 10.0, "Precipitation (%)": value / 2.0,
            "Atmospheric Pressure": 990.0 + value, "UV Index": float(value % 12),
            "Visibility (km)": 5.0 + value / 10.0,
            "Cloud Cover": ("clear", "cloudy", "overcast", "partly cloudy")[value % 4],
            "Season": ("Summer", "Autumn", "Spring", "Winter")[value % 4],
            "Location": ("coastal", "inland", "mountain")[value % 3]}


def course_data():
    train = pd.DataFrame([{"id": 100 + value, **weather_row(value), "label": value % 4 + 1}
                          for value in range(8)])
    test = pd.DataFrame([{"id": 1000 + value, **weather_row(value)} for value in range(20, 24)])
    sample = pd.DataFrame({"id": test.id.iloc[::-1].tolist(), "label": ["Rainy"] * 4})
    audit = {"source_sha256": {"train.csv": "a" * 64, "test.csv": "b" * 64,
                              "sample_submission.csv": "c" * 64}, "rows": {"train": 8, "test": 4}}
    return Dataset(train, test, sample, audit, np.arange(8))


class ExternalDataTests(unittest.TestCase):
    def setUp(self):
        self.output_area = Path(__file__).resolve().parents[2] / "outputs"
        self.output_area.mkdir(exist_ok=True)
        self.directory = self.output_area / f"external_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.repository = self.directory / "repo"
        self.repository.mkdir()
        self.source_path = self.directory / "source_data" / "weather.csv"
        self.source_path.parent.mkdir()
        self.addCleanup(self.cleanup_directory)
        self.data = course_data()

    def cleanup_directory(self):
        target = self.directory.resolve()
        if target.parent != self.output_area.resolve() or not target.name.startswith("external_test_"):
            raise RuntimeError("Refusing cleanup outside the external-data test area.")
        shutil.rmtree(target)

    def source(self, extra=None):
        names = {1: "Sunny", 2: "Cloudy", 3: "Rainy", 4: "Snowy"}
        train = self.data.train[FEATURES].copy()
        train["Weather Type"] = self.data.train.label.map(names)
        test = self.data.test[FEATURES].copy()
        test["Weather Type"] = ["Sunny", "Cloudy", "Rainy", "Snowy"]
        frames = [train, test]
        if extra:
            frames.append(pd.DataFrame(extra))
        return pd.concat(frames, ignore_index=True)

    def settings(self, source, policy="novel_only"):
        source.to_csv(self.source_path, index=False, encoding="utf-8", lineterminator="\n")
        return {"path": "../source_data/weather.csv",
                "sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
                "source_url": "https://example.test/public-weather-source", "policy": policy}

    def attach(self, source, policy="novel_only"):
        return attach_external(self.data, self.settings(source, policy), self.repository)

    def test_complete_course_source_has_zero_novel_rows(self):
        attached = self.attach(self.source())
        audit = attached.audit["external_data"]
        self.assertEqual(audit["source_rows"], 12)
        self.assertEqual(audit["course_train_overlap_source_rows"], 8)
        self.assertEqual(audit["course_test_overlap_source_rows"], 4)
        self.assertEqual(audit["external_pool_rows"], 0)
        self.assertEqual(audit["independent_new_rows"], 0)
        self.assertEqual(audit["test_overlap_source_rows_used"], 0)
        self.assertEqual(audit["status"], "no_novel_rows")
        X, y = self.data.X, self.data.y
        combined_X, combined_y, metadata = augment_training(attached, X, y)
        self.assertIs(combined_X, X)
        self.assertIs(combined_y, y)
        self.assertEqual(metadata["external_fit_rows"], 0)
        self.assertFalse(metadata["augmentation_applied"])
        json.dumps(audit, allow_nan=False)
        json.dumps(metadata, allow_nan=False)

    def test_public_source_training_uses_test_overlap_but_not_train_copies(self):
        attached = self.attach(self.source(), "public_source_training")
        context = attached.external
        self.assertEqual(len(context.frame), 4)
        self.assertEqual(context.audit["test_overlap_source_rows_used"], 4)
        self.assertEqual(context.audit["independent_new_rows"], 0)
        self.assertEqual(context.audit["status"], "source_assisted_only")
        self.assertEqual(context.labels.tolist(), [1, 2, 3, 4])
        self.assertFalse(set(context.feature_keys) & set(feature_keys(self.data.train)))
        X, y, metadata = augment_training(attached, self.data.X, self.data.y)
        self.assertEqual(len(X), 12)
        self.assertEqual(y.tolist()[-4:], [1, 2, 3, 4])
        self.assertEqual(metadata["course_fit_rows"], 8)
        self.assertEqual(metadata["external_fit_rows"], 4)
        self.assertEqual(metadata["combined_fit_rows"], 12)
        self.assertEqual(metadata["test_overlap_source_rows_used"], 4)
        self.assertEqual(metadata["independent_new_rows"], 0)
        self.assertTrue(metadata["public_source_test_overlap_declared"])
        self.assertEqual(X.index.tolist(), list(range(12)))

    def test_excluded_course_test_labels_do_not_change_novel_only_pool(self):
        source = self.source([{**weather_row(50), "Weather Type": "Sunny"}])
        first = self.attach(source)
        modified = source.copy(deep=True)
        modified.loc[:7, "Weather Type"] = "DO_NOT_CONSUME_COURSE_TRAIN_COPY"
        modified.loc[8:11, "Weather Type"] = "DO_NOT_CONSUME_COURSE_TEST_LABEL"
        second = self.attach(modified)
        pd.testing.assert_frame_equal(first.external.frame, second.external.frame)
        self.assertEqual(first.external.audit["pool_sha256"], second.external.audit["pool_sha256"])
        self.assertEqual(first.external.audit["training_pool_sha256"], second.external.audit["training_pool_sha256"])
        self.assertNotEqual(first.external.audit["source_sha256"], second.external.audit["source_sha256"])
        self.assertEqual(second.external.audit["test_overlap_source_rows_used"], 0)

    def test_nested_subset_inherits_global_course_training_exclusion(self):
        for policy in ("novel_only", "public_source_training"):
            attached = self.attach(self.source([{**weather_row(50), "Weather Type": "Sunny"}]), policy)
            outer_train = replace(attached, train=attached.train.iloc[:4].reset_index(drop=True), groups=attached.groups[:4])
            combined_X, _, metadata = augment_training(outer_train, outer_train.X, outer_train.y)
            held_out_keys = set(feature_keys(self.data.train.iloc[4:]))
            self.assertFalse(set(feature_keys(combined_X)) & held_out_keys)
            self.assertEqual(len(outer_train.external.course_train_feature_keys), 8)
            self.assertEqual(metadata["course_fit_rows"], 4)
            self.assertEqual(metadata["independent_new_rows"], 1)
            self.assertEqual(metadata["external_fit_rows"], 1 if policy == "novel_only" else 5)

    def test_conflicting_external_groups_quarantined_and_identical_labels_deduplicated(self):
        extra = [{**weather_row(50), "Weather Type": "Sunny"},
                 {**weather_row(50), "Weather Type": "Snowy"},
                 {**weather_row(51), "Weather Type": "Rainy"},
                 {**weather_row(51), "Weather Type": " Rainy "}]
        attached = self.attach(self.source(extra))
        audit = attached.external.audit
        self.assertEqual(audit["conflicting_label_groups"], 1)
        self.assertEqual(audit["excluded_conflicting_source_rows"], 2)
        self.assertEqual(audit["same_label_duplicate_rows_removed"], 1)
        self.assertEqual(audit["external_pool_rows"], 1)
        self.assertEqual(attached.external.labels.tolist(), [3])
        self.assertEqual(attached.external.source_row_indices, (14,))

    def test_source_hash_change_is_refused_before_using_changed_data(self):
        settings = self.settings(self.source())
        self.source_path.write_bytes(self.source_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            attach_external(self.data, settings, self.repository)

    def test_attach_and_augmentation_do_not_modify_original_course_data(self):
        before = deepcopy(self.data)
        attached = self.attach(self.source([{**weather_row(50), "Weather Type": "Sunny"}]))
        course_X = self.data.X.iloc[[1, 3, 5]]
        course_y = self.data.y[[1, 3, 5]]
        original_X, original_y = course_X.copy(deep=True), course_y.copy()
        combined_X, _, _ = augment_training(attached, course_X, course_y)
        pd.testing.assert_frame_equal(self.data.train, before.train)
        pd.testing.assert_frame_equal(self.data.test, before.test)
        pd.testing.assert_frame_equal(self.data.sample, before.sample)
        self.assertEqual(self.data.audit, before.audit)
        self.assertIsNone(self.data.external)
        self.assertNotIn("external_data", self.data.audit)
        pd.testing.assert_frame_equal(course_X, original_X)
        np.testing.assert_array_equal(course_y, original_y)
        self.assertEqual(combined_X.index.tolist(), [0, 1, 2, 3])
        self.assertTrue(all(attached.external.frame[column].dtype == np.float64 for column in NUMERIC))
        # Category casing in model inputs remains compatible with course data.
        self.assertEqual(attached.external.frame.Season.iloc[0], "Spring")

    def test_external_feature_and_label_properties_are_copies(self):
        attached = self.attach(self.source([{**weather_row(50), "Weather Type": "Sunny"}]))
        X, y = attached.external.features, attached.external.labels
        X.iloc[0, 0] = -999
        y[0] = 4
        self.assertEqual(attached.external.labels.tolist(), [1])
        self.assertNotEqual(attached.external.features.iloc[0, 0], -999)

    def test_canonical_keys_standardize_missing_case_index_numeric_dtype_and_negative_zero(self):
        first = pd.DataFrame([weather_row(0), weather_row(1)])
        second = first.copy(deep=True)
        second.index = [19, 27]
        first.loc[0, "Temperature"] = 0.0
        second.loc[19, "Temperature"] = -0.0
        first.loc[1, "Humidity"] = np.nan
        second.loc[27, "Humidity"] = pd.NA
        first.loc[1, "Location"] = None
        second.loc[27, "Location"] = " "
        second["Season"] = second.Season.str.upper().map(lambda value: f" {value} ")
        second["id"] = [999, 998]
        second["label"] = [4, 4]
        self.assertEqual(feature_keys(first), feature_keys(second))
        changed = first.copy(deep=True)
        changed.loc[0, "Temperature"] = np.nextafter(0.0, 1.0)
        self.assertNotEqual(feature_keys(first)[0], feature_keys(changed)[0])

    def test_no_external_context_preserves_original_objects(self):
        X, y = self.data.X, pd.Series(self.data.y, index=self.data.train.index)
        result_X, result_y, metadata = augment_training(self.data, X, y)
        self.assertIs(result_X, X)
        self.assertIs(result_y, y)
        self.assertEqual(metadata["external_fit_rows"], 0)
        self.assertEqual(metadata["combined_fit_rows"], len(X))

    def test_defensive_fit_overlap_is_not_added_twice(self):
        attached = self.attach(self.source([{**weather_row(50), "Weather Type": "Sunny"},
                                            {**weather_row(51), "Weather Type": "Snowy"}]))
        incoming_X = pd.concat([self.data.X.iloc[:3], attached.external.features.iloc[:1]], ignore_index=True)
        incoming_y = np.concatenate([self.data.y[:3], [1]])
        result_X, _, metadata = augment_training(attached, incoming_X, incoming_y)
        self.assertEqual(metadata["excluded_fit_overlap_rows"], 1)
        self.assertEqual(metadata["external_fit_rows"], 1)
        self.assertEqual(metadata["independent_new_rows"], 1)
        self.assertEqual(len(result_X), 5)
        self.assertEqual(len(set(feature_keys(result_X))), 5)

    def test_unknown_label_is_rejected_only_if_source_row_is_retained(self):
        with self.assertRaisesRegex(ValueError, "Retained source labels"):
            self.attach(self.source([{**weather_row(50), "Weather Type": "Foggy"}]))
        source = self.source()
        source.loc[8, "Weather Type"] = "Foggy"
        self.assertEqual(len(self.attach(source).external.frame), 0)
        with self.assertRaisesRegex(ValueError, "Retained source labels"):
            self.attach(source, "public_source_training")

    def test_changed_pool_after_audit_is_rejected(self):
        attached = self.attach(self.source([{**weather_row(50), "Weather Type": "Sunny"}]))
        attached.external.frame.loc[0, "label"] = 4
        with self.assertRaisesRegex(ValueError, "pool changed"):
            augment_training(attached, self.data.X, self.data.y)

    def test_settings_and_source_schema_are_strict(self):
        settings = self.settings(self.source())
        for bad in ({**settings, "policy": "automatic"}, {**settings, "sha256": "wrong"},
                    {**settings, "label_column": "label"}, {**settings, "source_url": ""}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                attach_external(self.data, bad, self.repository)
        source = self.source().rename(columns={"Weather Type": "label"})
        with self.assertRaisesRegex(ValueError, "source schema"):
            self.attach(source)

    def test_external_path_is_resolved_from_repository_not_snapshot_location(self):
        settings = self.settings(self.source())
        attached = attach_external(self.data, settings, self.repository)
        self.assertEqual(attached.external.audit["source_path"], str(self.source_path.resolve()))
        with self.assertRaisesRegex(ValueError, "Attach external data once"):
            attach_external(attached, settings, self.repository)


if __name__ == "__main__":
    unittest.main()
