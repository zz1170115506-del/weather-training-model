"""Nested selection tests use handwritten OOF arrays and mocked inner trials."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.artifacts import content_hash, save_json
from task1.config import resolve_config
from task1.search import _pool, _rank, evaluate_search
from task1.validation import make_folds, metrics


def probabilities(y, wrong=()):
    array = np.full((len(y), 4), 0.05)
    array[np.arange(len(y)), np.asarray(y) - 1] = 0.85
    for index in wrong:
        label = int(y[index]) - 1
        array[index] = 0.1
        array[index, label] = 0.2
        array[index, (label + 1) % 4] = 0.6
    return array


def summary_for(data, candidate, folds, probability):
    details = []
    for fold in sorted(np.unique(folds)):
        valid = folds == fold
        value = metrics(data.y[valid], probability[valid])
        details.append({"fold": int(fold), "metrics": value,
                        "fit_ids_sha256": content_hash(data.train.id[~valid].tolist()),
                        "validation_ids_sha256": content_hash(data.train.id[valid].tolist())})
    scores = [item["metrics"]["macro_f1"] for item in details]
    return {"name": candidate["name"], "model_id": candidate["model_id"],
            "config": candidate, "config_sha256": content_hash(candidate),
            "fold_macro_f1_mean": float(np.mean(scores)), "fold_macro_f1_std": float(np.std(scores, ddof=1)),
            "oof_metrics": metrics(data.y, probability), "folds": details,
            "total_cv_seconds": 0.01, "evaluation_scope": "Handwritten synthetic OOF fixture only."}


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.output_area = Path(__file__).resolve().parents[2] / "outputs"
        self.directory = self.output_area / ("search_test_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.data = synthetic_dataset()
        self.config = resolve_config({"schema_version": 1,
            "validation": {"n_splits": 2, "split_seed": 83, "model_seed": 91},
            "models": [
                {"name": "aa_hgb", "model_id": "hgb", "model_params": {"max_iter": 4}},
                {"name": "ab_hgb", "model_id": "hgb", "model_params": {"max_iter": 5}},
                {"name": "cb_xgb", "model_id": "xgboost", "model_params": {"n_estimators": 4}}],
            "search": {"inner_splits": 2, "blend_pool_size": 2, "minimum_blend_gain": 0.0005,
                       "min_blend_fold_wins": 2},
            "blend": {"n_trials": 0, "allow_class_multipliers": False}})
        self.folds, self.protocol = make_folds(self.data.y, self.data.train.id.to_numpy(), self.data.groups, 2, 83)
        self.calls = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        target = self.directory.resolve()
        if target.parent != self.output_area.resolve() or not target.name.startswith("search_test_"):
            raise RuntimeError("Unsafe search fixture cleanup path.")
        shutil.rmtree(target)

    def fixture(self, name="run", config=None):
        config = config or self.config
        out = self.directory / name
        out.mkdir()
        save_json(out / "resolved_config.json", config)
        save_json(out / "fold_protocol.json", self.protocol)
        pd.DataFrame({"id": self.data.train.id, "fold": self.folds}).to_csv(out / "folds.csv", index=False)
        first = [np.flatnonzero(self.folds == fold)[0] for fold in range(2)]
        second = [np.flatnonzero(self.folds == fold)[1] for fold in range(2)]
        third = [np.flatnonzero(self.folds == fold)[2] for fold in range(2)]
        wrong = {"aa_hgb": first, "ab_hgb": first + third, "cb_xgb": second}
        for candidate in config["models"]:
            probability = probabilities(self.data.y, wrong[candidate["name"]])
            directory = out / "models" / candidate["name"]
            directory.mkdir(parents=True)
            save_json(directory / "summary.json", summary_for(self.data, candidate, self.folds, probability))
            np.savez_compressed(directory / "oof.npz", ids=self.data.train.id.to_numpy(),
                                y=self.data.y, folds=self.folds, probabilities=probability)
        return out

    def inner_trial(self, data, candidate, folds, seed, **kwargs):
        outer_fold = (seed - self.config["validation"]["model_seed"]) // 10000 - 1
        ids = set(data.train.id)
        expected = set(self.data.train.id[self.folds != outer_fold])
        self.assertEqual(ids, expected)
        self.assertFalse(ids & set(self.data.train.id[self.folds == outer_fold]))
        expected_folds, _ = make_folds(data.y, data.train.id.to_numpy(), data.groups,
                                       2, 83 + 1000 + outer_fold)
        np.testing.assert_array_equal(folds, expected_folds)
        self.calls.append((int(outer_fold), candidate["name"], tuple(data.train.id)))
        offset = 1 if candidate["name"] == "cb_xgb" else 0
        wrong = [np.flatnonzero(data.y == label)[offset] for label in range(1, 5)]
        if candidate["name"] == "ab_hgb":
            wrong += [np.flatnonzero(data.y == label)[1] for label in range(1, 5)]
        probability = probabilities(data.y, wrong)
        return summary_for(data, candidate, folds, probability), probability

    def test_nested_trials_exclude_outer_records_and_save_reproducible_selection(self):
        out = self.fixture()
        snapshot = (out / "resolved_config.json").read_bytes()
        with patch("task1.search.cross_validate", side_effect=self.inner_trial), contextlib.redirect_stdout(io.StringIO()):
            report = evaluate_search(self.data, self.config, out)
        self.assertEqual(len(self.calls), 6)
        self.assertEqual((out / "resolved_config.json").read_bytes(), snapshot)
        self.assertEqual(report, json.loads((out / "search_summary.json").read_text(encoding="utf-8")))
        self.assertEqual(len(report["rank"]), 3)
        self.assertEqual(report["selected_models"][0], report["rank"][0]["name"])
        names = {candidate["name"]: candidate["model_id"] for candidate in self.config["models"]}
        self.assertNotEqual(names[report["selected_models"][0]], names[report["selected_models"][1]])
        self.assertEqual(set(report["base_model_names"]), {"aa_hgb", "cb_xgb"})
        self.assertIn("not independent", report["selection_scope"])
        self.assertIn("gate", report["evaluation_scope"])
        for detail in report["outer_folds"]:
            self.assertFalse(detail["outer_validation_used_for_selection"])
            self.assertEqual(len(detail["trial_summary_paths"]), 3)
            for relative in detail["trial_summary_paths"]:
                trial = json.loads((out / relative).read_text(encoding="utf-8"))
                self.assertIn("config", trial)
                self.assertIn("folds", trial)
                self.assertEqual(trial["search_outer_fold"], detail["fold"])
        with np.load(out / "search/cross_fitted.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["ids"], self.data.train.id)
            self.assertEqual(metrics(self.data.y, archive["single_probabilities"]), report["nested_single_metrics"])
            self.assertEqual(metrics(self.data.y, archive["blend_probabilities"]), report["nested_blend_metrics"])
        gate = report["blend_gate"]
        self.assertTrue(report["blend_export_eligible"])
        self.assertEqual(gate["blend_fold_wins"], 2)
        self.assertAlmostEqual(gate["gain_over_best_fixed_base"],
                               report["nested_blend_metrics"]["macro_f1"] - report["rank"][0]["oof_macro_f1"])
        with patch("task1.search.cross_validate", side_effect=AssertionError("must not rerun")), self.assertRaises(FileExistsError):
            evaluate_search(self.data, self.config, out)

    def test_gate_rejects_blend_even_when_its_full_oof_selection_score_is_high(self):
        config = resolve_config(self.config)
        config["search"]["minimum_blend_gain"] = 0.5
        out = self.fixture(config=config)
        with patch("task1.search.cross_validate", side_effect=self.inner_trial), contextlib.redirect_stdout(io.StringIO()):
            report = evaluate_search(self.data, config, out)
        self.assertEqual(report["full_oof_blend_metrics"]["macro_f1"], 1.0)
        self.assertFalse(report["blend_export_eligible"])
        self.assertFalse(report["blend_gate"]["gain_conditions_met"])

    def test_ranking_ties_and_family_pool_are_deterministic(self):
        data = [
            {"name": name, "model_id": family, "config_sha256": name,
             "oof_metrics": {"macro_f1": score, "accuracy": score},
             "fold_macro_f1_mean": score, "fold_macro_f1_std": std}
            for name, family, score, std in [("z_hgb", "hgb", .9, .01), ("a_hgb", "hgb", .9, .01),
                                              ("cb", "catboost", .89, .01), ("xgb", "xgboost", .88, .01)]]
        rank = _rank(data)
        self.assertEqual([item["name"] for item in rank], ["a_hgb", "z_hgb", "cb", "xgb"])
        self.assertEqual(_pool(rank, 3), ["a_hgb", "cb", "xgb"])
        self.assertEqual(_pool(rank, 4), ["a_hgb", "cb", "xgb", "z_hgb"])

    def test_archive_corruption_is_rejected_before_any_inner_fit(self):
        changes = {
            "ids": lambda arrays: arrays.update(ids=arrays["ids"][::-1]),
            "y": lambda arrays: arrays["y"].__setitem__(0, 4),
            "folds": lambda arrays: arrays["folds"].__setitem__(0, 1 - arrays["folds"][0]),
            "nan": lambda arrays: arrays["probabilities"].__setitem__((0, 0), np.nan),
            "normalization": lambda arrays: arrays["probabilities"].__setitem__((0, 0), 0.0),
            "object_ids": lambda arrays: arrays.update(ids=arrays["ids"].astype(object)),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                out = self.fixture(name)
                path = out / "models/aa_hgb/oof.npz"
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                change(arrays)
                np.savez_compressed(path, **arrays)
                with patch("task1.search.cross_validate", side_effect=AssertionError("corrupt archive reached fit")), self.assertRaises(ValueError):
                    evaluate_search(self.data, self.config, out)
                self.assertFalse((out / "search").exists())

    def test_summary_configuration_and_fold_hash_cannot_be_substituted(self):
        for kind in ("config", "fold_hash", "metric", "snapshot"):
            with self.subTest(kind=kind):
                out = self.fixture(kind)
                if kind in ("config", "metric"):
                    path = out / "models/aa_hgb/summary.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if kind == "config":
                        value["config_sha256"] = "0" * 64
                    else:
                        value["oof_metrics"]["macro_f1"] = 1.0
                elif kind == "fold_hash":
                    path = out / "fold_protocol.json"
                    value = dict(self.protocol, fold_sha256="0" * 64)
                else:
                    path = out / "resolved_config.json"
                    value = dict(self.config, description="Edited frozen configuration")
                save_json(path, value)
                with patch("task1.search.cross_validate", side_effect=AssertionError("corrupt artifacts reached fit")), self.assertRaises(ValueError):
                    evaluate_search(self.data, self.config, out)
                self.assertFalse((out / "search").exists())


if __name__ == "__main__":
    unittest.main()
