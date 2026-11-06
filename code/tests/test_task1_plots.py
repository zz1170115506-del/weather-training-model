"""Aggregate exports must stay comparable and must not publish row data."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset, small_config
from task1.artifacts import finish_run, new_run, save_json
from task1.pipeline import run_cv
from task1.plots import export_results

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))


class AggregateExportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / "outputs" / ("plot_test_" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.data = synthetic_dataset()
        self.data.audit["source_sha256"] = {key: "a" * 64 for key in self.data.audit["source_sha256"]}
        config = small_config()
        config["models"][0]["data_dir"] = "C:\\private-user\\do-not-publish"
        self.run = new_run("cv", config, self.root)
        save_json(self.run / "audit.json", self.data.audit)
        with redirect_stdout(io.StringIO()):
            result = run_cv(self.data, config, self.run)
        finish_run(self.run, {"source_sha256": self.data.audit["source_sha256"], "result": result})

    def tearDown(self):
        target = self.root.resolve()
        intended = Path(__file__).resolve().parents[2].resolve() / "outputs"
        if not target.is_relative_to(intended) or not target.name.startswith("plot_test_"):
            raise RuntimeError("Unsafe test cleanup path.")
        shutil.rmtree(target)

    def test_only_aggregate_allowlist_is_exported_and_existing_results_survive(self):
        folds = self.run / "models/synthetic_cart/fold_metrics.csv"
        frame = pd.read_csv(folds)
        frame["private_raw_id"] = 123456
        frame.to_csv(folds, index=False)
        output = self.root / "shared"
        exported = export_results([self.run], output)
        self.assertEqual(exported["candidate_count"], 1)
        self.assertTrue((output / "comparison.png").stat().st_size > 1000)
        self.assertTrue((output / "best_confusion.png").stat().st_size > 1000)
        self.assertNotIn("private_raw_id", pd.read_csv(output / "fold_metrics.csv").columns)
        detail = (output / "model_details.json").read_text(encoding="utf-8")
        self.assertNotIn("private-user", detail)
        self.assertFalse(list(output.glob("*.npz")))
        self.assertFalse(list(output.glob("*.joblib")))
        before = (output / "comparison.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            export_results([self.run], output)
        self.assertEqual((output / "comparison.csv").read_bytes(), before)

    def test_different_fold_protocol_is_rejected_before_creating_destination(self):
        second = self.root / "other_run"
        shutil.copytree(self.run, second)
        protocol = json.loads((second / "fold_protocol.json").read_text(encoding="utf-8"))
        protocol["seed"] += 1
        save_json(second / "fold_protocol.json", protocol)
        destination = self.root / "incomparable"
        with self.assertRaises(ValueError):
            export_results([self.run, second], destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
