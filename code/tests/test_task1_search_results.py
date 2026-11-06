"""Read-only nested-search result guards from handwritten four-row artifacts."""
from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_task1_results as fixtures
from task1.pipeline import record_score
from task1.results import print_results, summarize_run


class SearchResultsTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.ResultsTests()
        self.helper.setUp()
        self.addCleanup(self.helper.cleanup_directory)
        self.root = self.helper.fixture()
        self.summary = json.loads((self.root / "training_summary.json").read_text(encoding="utf-8"))
        raw = self.summary["comparison"]["metrics"]
        names = [item["name"] for item in raw]
        recipe = {"schema_version": 1, "labels": [1, 2, 3, 4], "model_ids": names,
                  "weights": [.5, .5], "class_multipliers": [1.] * 4,
                  "selection_scope": "Handwritten fixture selection, not an independent result."}
        self.report = {"schema_version": 1, "candidate_model_names": names,
            "rank": [{"rank": index, **item} for index, item in enumerate(raw, 1)],
            "selected_models": names, "base_model_names": names, "full_oof_recipe": recipe,
            "nested_single_metrics": {"macro_f1": .91}, "nested_blend_metrics": {"macro_f1": .918},
            "full_oof_blend_metrics": {"macro_f1": .925},
            "outer_folds": [{"fold": fold, "single_metrics": {"macro_f1": .91}, "blend_metrics": {"macro_f1": .918}} for fold in range(2)],
            "blend_export_eligible": True,
            "blend_gate": {"minimum_blend_gain": .0005, "outer_fold_count": 2,
                           "min_blend_fold_wins": 2, "blend_fold_wins": 2, "eligible": True},
            "selection_scope": "Fixture-only tuning score; not independent.",
            "evaluation_scope": "Fixture procedure evaluation; gate itself selects on development outcomes."}
        a, b = (self.summary["candidates"][key] for key in ("candidate_a", "candidate_b"))
        folder = self.root / "research_blend"
        folder.mkdir()
        (folder / "submission.csv").write_bytes((self.root / b["csv"]).read_bytes())
        (folder / "submission_name.csv").write_bytes((self.root / a["csv"]).read_bytes())
        self.summary["research_blend_export"] = {
            "model_name": "research_blend", "csv": "research_blend/submission.csv",
            "alternative_format": "name", "alternative_csv": "research_blend/submission_name.csv",
            "bundle": "research_blend/model_bundle.joblib", "recipe": recipe,
            "submission": b["submission"],
            "fit_and_export": {"recipe": recipe, "submission": b["submission"],
                               "alternative_submission": a["submission"], "full_test_rows": 4}}
        self.save()

    def save(self):
        self.summary["search"] = self.report
        fixtures.write_json(self.root / "search_summary.json", self.report)
        fixtures.write_json(self.root / "training_summary.json", self.summary)
        self.helper.reseal(self.root)

    def test_research_is_separate_and_read_only_and_scores_bind_exact_csv(self):
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        summary = summarize_run(self.root)
        stream = io.StringIO()
        with redirect_stdout(stream):
            print_results(summary)
        after = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(set(summary["candidates"]), {"candidate_a", "candidate_b"})
        self.assertEqual(len(summary["models"]), 2)
        self.assertEqual(summary["research_blend"]["rows"], 4)
        self.assertNotIn("metrics", summary["research_blend"])
        self.assertIn("搜索选择分数不等于独立成绩", stream.getvalue())
        args = Namespace(run_dir=self.root, candidate="research_blend", submission_id="synthetic-blend-fixture",
                         public_score=.5, note="Synthetic fixture, not a competition score.")
        with redirect_stdout(io.StringIO()):
            record_score(args)
        saved = summarize_run(self.root)["platform_scores"][0]
        self.assertEqual(saved["candidate"], "research_blend")
        self.assertEqual(saved["csv_sha256"], summary["research_blend"]["csv_sha256"])

    def test_rejects_export_when_nested_gate_fails(self):
        self.report["nested_blend_metrics"]["macro_f1"] = .90
        for fold in self.report["outer_folds"]:
            fold["blend_metrics"]["macro_f1"] = .90
        self.report["blend_export_eligible"] = False
        self.report["blend_gate"].update(eligible=False, blend_fold_wins=0)
        self.save()
        with self.assertRaisesRegex(ValueError, "gate fails"):
            summarize_run(self.root)

    def test_rejects_forged_gate_and_mismatched_recipe(self):
        self.report["blend_gate"]["eligible"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "eligibility"):
            summarize_run(self.root)
        self.report["blend_gate"]["eligible"] = True
        self.summary["research_blend_export"]["recipe"] = deepcopy(self.report["full_oof_recipe"])
        self.summary["research_blend_export"]["recipe"]["weights"] = [.6, .4]
        self.save()
        with self.assertRaisesRegex(ValueError, "differs"):
            summarize_run(self.root)

    def test_rejects_research_path_escape_and_changed_csv(self):
        self.summary["research_blend_export"]["bundle"] = "../outside.joblib"
        self.save()
        with self.assertRaisesRegex(ValueError, "traversal"):
            summarize_run(self.root)
        self.summary["research_blend_export"]["bundle"] = "research_blend/model_bundle.joblib"
        self.save()
        (self.root / "research_blend/submission.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            summarize_run(self.root)


if __name__ == "__main__":
    unittest.main()
