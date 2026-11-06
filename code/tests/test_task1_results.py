"""Read-only result tests built from handwritten artifacts, without fitting."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1.results import compare_runs, print_results, summarize_run


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.output_area = Path(__file__).resolve().parents[2] / "outputs"
        self.output_area.mkdir(exist_ok=True)
        self.directory = self.output_area / f"results_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup_directory)

    def cleanup_directory(self):
        target = self.directory.resolve()
        if target.parent != self.output_area.resolve() or not target.name.startswith("results_test_"):
            raise RuntimeError("Refusing cleanup outside the results test area.")
        shutil.rmtree(target)

    def reseal(self, root):
        path = root / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["files_sha256"] = {
            item.relative_to(root).as_posix(): digest(item.read_bytes())
            for item in root.rglob("*") if item.is_file() and item.name not in ("run.json", "platform_scores.json")
        }
        write_json(path, manifest)

    def fixture(self, name="current", delta=0.0, seed=5434):
        root = self.directory / name
        root.mkdir()
        names = ["hgb_weighted", "hgb_regularized"]
        metrics = []
        for index, model in enumerate(names):
            mean = 0.916 - index * 0.001 + delta
            oof = 0.9159 - index * 0.001 + delta
            item = {"name": model, "model_id": "hgb", "fold_macro_f1_mean": mean,
                    "fold_macro_f1_std": 0.005, "oof_macro_f1": oof}
            metrics.append(item)
            write_json(root / "models" / model / "summary.json",
                       {**item, "oof_metrics": {"macro_f1": oof, "accuracy": 0.915 + delta}})
        with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(metrics)
        protocol = {"method": "StratifiedKFold", "n_splits": 2, "seed": seed, "fold_sha256": "f" * 64}
        candidates = {}
        (root / "submissions").mkdir()
        for index, model in enumerate(names):
            key = f"candidate_{chr(97 + index)}"
            primary_format = "name" if index == 0 else "integer"
            primary, primary_metadata = self.csv_fixture(root, f"submissions/{key}.csv", primary_format)
            alternative_format = "integer" if primary_format == "name" else "name"
            alternative, alternative_metadata = self.csv_fixture(root, f"submissions/{key}_{alternative_format}.csv", alternative_format)
            candidates[key] = {
                "model_name": model, "csv": primary, "submission": primary_metadata,
                "alternative_format": alternative_format, "alternative_csv": alternative,
                "fit_and_export": {"submission": primary_metadata,
                                   "alternative_submission": alternative_metadata, "full_test_rows": 4},
            }
        summary = {"comparison": {"fold_protocol": protocol, "metrics": metrics}, "candidates": candidates}
        write_json(root / "training_summary.json", summary)
        write_json(root / "run.json", {"run_id": name, "stage": "train", "status": "complete",
                   "source_sha256": {"train.csv": "a" * 64, "test.csv": "b" * 64, "sample_submission.csv": "c" * 64},
                   "result": {"comparison": {"fold_protocol": protocol}}})
        self.reseal(root)
        return root

    def csv_fixture(self, root, relative, label_format):
        labels = [1, 2, 3, 4]
        ids = [101, 102, 103, 104]
        names = ["Sunny", "Cloudy", "Rainy", "Snowy"]
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, labels if label_format == "integer" else names))
        content = stream.getvalue().encode("utf-8")
        (root / relative).write_bytes(content)
        semantic = b'task1-predictions-v1\n["integer","101",1]\n["integer","102",2]\n["integer","103",3]\n["integer","104",4]\n'
        metadata = {"rows": 4, "schema": ["id", "label"], "id_type": "integer",
                    "label_format": label_format, "class_counts": {str(k): 1 for k in labels},
                    "csv_sha256": digest(content), "canonical_prediction_sha256": digest(semantic),
                    "canonical_prediction_version": "task1-predictions-v1"}
        return relative, metadata

    def update_summary(self, root, change):
        path = root / "training_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        change(summary)
        write_json(path, summary)
        self.reseal(root)

    def test_summarizes_metrics_and_selects_alternative_or_main_integer_csv(self):
        root = self.fixture()
        result = summarize_run(root)
        self.assertEqual(result["models"][0]["name"], "hgb_weighted")
        self.assertAlmostEqual(result["models"][0]["fold_macro_f1_mean"], 0.916)
        self.assertAlmostEqual(result["models"][0]["fold_macro_f1_std"], 0.005)
        self.assertAlmostEqual(result["models"][0]["oof_macro_f1"], 0.9159)
        self.assertAlmostEqual(result["models"][0]["accuracy"], 0.915)
        first, second = result["candidates"]["candidate_a"], result["candidates"]["candidate_b"]
        self.assertEqual(first["integer_upload_csv"], str((root / "submissions/candidate_a_integer.csv").resolve()))
        self.assertEqual(second["integer_upload_csv"], str((root / "submissions/candidate_b.csv").resolve()))
        self.assertEqual(first["label_format"], "integer")
        self.assertEqual(first["rows"], 4)
        self.assertEqual(first["csv_sha256"], digest(Path(first["integer_upload_csv"]).read_bytes()))
        self.assertIsNone(result["platform_scores"])
        self.assertIsNone(first["platform_scores"])
        self.assertIn("不是 Kaggle", result["evaluation_scope"])
        json.dumps(result, allow_nan=False)

    def test_reading_results_does_not_change_or_create_any_artifact(self):
        root = self.fixture()
        before = {item.relative_to(root).as_posix(): item.read_bytes() for item in root.rglob("*") if item.is_file()}
        result = summarize_run(root)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print_results(result)
        after = {item.relative_to(root).as_posix(): item.read_bytes() for item in root.rglob("*") if item.is_file()}
        self.assertEqual(before, after)
        self.assertIn("candidate_a_integer.csv", stream.getvalue())
        self.assertIn("尚无真实平台成绩", stream.getvalue())
        self.assertIn("0.916000", stream.getvalue())

    def test_only_complete_user_train_runs_are_accepted(self):
        root = self.fixture()
        path = root / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for stage, status in (("train", "running"), ("train", "failed"), ("train", "cancelled"), ("cv", "complete")):
            write_json(path, {**manifest, "stage": stage, "status": status})
            with self.subTest(stage=stage, status=status), self.assertRaisesRegex(ValueError, "complete.*train"):
                summarize_run(root)

    def test_upload_hint_follows_exported_metrics_instead_of_fixed_letter_a(self):
        result = summarize_run(self.fixture())
        candidates = result["candidates"]
        candidates["candidate_a"], candidates["candidate_b"] = candidates["candidate_b"], candidates["candidate_a"]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print_results(result)
        self.assertIn("本地 OOF 最高: candidate_b", stream.getvalue())
        self.assertIn("不保证 Kaggle 更高", stream.getvalue())

    def test_refuses_candidate_csv_changed_after_training(self):
        root = self.fixture()
        (root / "submissions/candidate_a_integer.csv").write_bytes(b"id,label\n101,4\n102,2\n103,3\n104,4\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            summarize_run(root)

    def test_refuses_manifest_checksum_changed_despite_matching_recorded_file_hash(self):
        root = self.fixture()
        path = root / "submissions/candidate_a_integer.csv"
        path.write_bytes(path.read_bytes().replace(b"101,1", b"101,4"))
        self.reseal(root)
        with self.assertRaisesRegex(ValueError, "CSV SHA256 mismatch"):
            summarize_run(root)

    def test_refuses_wrong_csv_row_count_and_inconsistent_full_test_count(self):
        root = self.fixture()
        self.update_summary(root, lambda value: value["candidates"]["candidate_a"]["fit_and_export"].update(full_test_rows=3300))
        with self.assertRaisesRegex(ValueError, "full test row count"):
            summarize_run(root)
        self.update_summary(root, lambda value: value["candidates"]["candidate_a"]["fit_and_export"].update(full_test_rows=4))
        self.update_summary(root, lambda value: value["candidates"]["candidate_a"]["fit_and_export"]["alternative_submission"].update(rows=3300))
        with self.assertRaisesRegex(ValueError, "row count"):
            summarize_run(root)

    def test_candidate_relative_paths_cannot_escape_run(self):
        for escaped in ("../outside.csv", "submissions/../../outside.csv", str(self.directory / "outside.csv"), "C:\\outside.csv"):
            root = self.fixture(f"path_{uuid.uuid4().hex}")
            self.update_summary(root, lambda value: value["candidates"]["candidate_a"].update(alternative_csv=escaped))
            with self.subTest(escaped=escaped), self.assertRaisesRegex(ValueError, "path|relative"):
                summarize_run(root)

    def test_refuses_missing_integer_alternative_and_wrong_format_claim(self):
        root = self.fixture()
        self.update_summary(root, lambda value: value["candidates"]["candidate_a"].update(alternative_format="name"))
        with self.assertRaisesRegex(ValueError, "alternative_format"):
            summarize_run(root)

    def test_unused_alternative_or_bundle_path_cannot_escape_run(self):
        for field in ("alternative_csv", "bundle"):
            root = self.fixture(f"unused_{field}")
            self.update_summary(root, lambda value: value["candidates"]["candidate_b"].update({field: "../outside"}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "path"):
                summarize_run(root)

    def test_refuses_comparison_table_and_summary_disagreement(self):
        root = self.fixture()
        self.update_summary(root, lambda value: value["comparison"]["metrics"][0].update(oof_macro_f1=0.99))
        with self.assertRaisesRegex(ValueError, "disagree"):
            summarize_run(root)

    def test_refuses_nonfinite_metric(self):
        root = self.fixture()
        path = root / "comparison.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("0.916", "nan", 1), encoding="utf-8")
        self.reseal(root)
        with self.assertRaisesRegex(ValueError, "finite metric"):
            summarize_run(root)

    def test_platform_scores_are_only_taken_from_log_and_keep_model_identity(self):
        root = self.fixture()
        local = summarize_run(root)
        candidate = local["candidates"]["candidate_a"]
        entry = {"candidate": "candidate_a", "model_name": "hgb_weighted", "submission_id": "submission-123",
                 "public_macro_f1": 0.904, "csv_sha256": candidate["csv_sha256"], "source": "user reported"}
        write_json(root / "platform_scores.json", [entry])
        result = summarize_run(root)
        self.assertEqual(result["platform_scores"][0]["public_macro_f1"], 0.904)
        self.assertTrue(result["platform_scores"][0]["matches_integer_upload_csv"])
        self.assertEqual(result["candidates"]["candidate_a"]["platform_scores"][0]["submission_id"], "submission-123")
        self.assertIsNone(result["candidates"]["candidate_b"]["platform_scores"])
        self.assertAlmostEqual(result["models"][0]["oof_macro_f1"], 0.9159)

    def test_failed_upload_or_screenshot_is_not_a_zero_score(self):
        root = self.fixture()
        candidate = summarize_run(root)["candidates"]["candidate_a"]
        entries = [
            {"candidate": "candidate_a", "status": "failed", "submission_id": "failed-1", "public_macro_f1": 0.0},
            {"candidate": "candidate_a", "status": "screenshot_failed", "public_macro_f1": None},
            {"candidate": "candidate_a", "model_name": "hgb_weighted", "submission_id": "bad-score",
             "public_macro_f1": float("nan"), "csv_sha256": candidate["csv_sha256"]},
        ]
        write_json(root / "platform_scores.json", entries)
        result = summarize_run(root)
        self.assertIsNone(result["platform_scores"])
        self.assertIsNone(result["candidates"]["candidate_a"]["platform_scores"])
        self.assertEqual(len(result["notes"]), 3)

    def test_real_recorded_zero_score_is_preserved(self):
        root = self.fixture()
        candidate = summarize_run(root)["candidates"]["candidate_a"]
        write_json(root / "platform_scores.json", [{"candidate": "candidate_a", "model_name": "hgb_weighted",
                   "submission_id": "real-scored-zero", "public_macro_f1": 0.0, "csv_sha256": candidate["csv_sha256"]}])
        self.assertEqual(summarize_run(root)["platform_scores"][0]["public_macro_f1"], 0.0)

    def test_same_data_and_protocol_compare_current_minus_previous(self):
        previous = self.fixture("previous")
        current = self.fixture("current", delta=0.002)
        result = compare_runs(summarize_run(current), summarize_run(previous))
        self.assertTrue(result["same_data_and_folds"])
        for candidate in ("candidate_a", "candidate_b"):
            self.assertAlmostEqual(result["candidates"][candidate]["oof_macro_f1_delta"], 0.002)
            self.assertAlmostEqual(result["candidates"][candidate]["fold_macro_f1_mean_delta"], 0.002)
        self.assertEqual(compare_runs(current, previous), result)
        json.dumps(result, allow_nan=False)

    def test_automatic_comparison_refuses_other_data(self):
        previous, current = self.fixture("previous"), self.fixture("current")
        path = previous / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["source_sha256"]["train.csv"] = "d" * 64
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "data hashes differ"):
            compare_runs(current, previous)

    def test_automatic_comparison_refuses_other_seed_even_if_hash_is_same(self):
        previous, current = self.fixture("previous", seed=99), self.fixture("current")
        with self.assertRaisesRegex(ValueError, "protocol or seed differs"):
            compare_runs(current, previous)

    def test_automatic_comparison_refuses_other_fold_hash_or_count(self):
        previous, current = summarize_run(self.fixture("previous")), summarize_run(self.fixture("current"))
        for key, value in (("fold_sha256", "e" * 64), ("n_splits", 3)):
            changed = {**previous, "fold_protocol": {**previous["fold_protocol"], key: value}}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "protocol or seed differs"):
                compare_runs(current, changed)


if __name__ == "__main__":
    unittest.main()
