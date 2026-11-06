"""Competition format regression: names in the sample must not drive exports."""
from pathlib import Path
from argparse import Namespace
import runpy
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1.config import load_config
from task1.pipeline import execute


class CompetitionFormatTests(unittest.TestCase):
    def test_cli_defaults_to_integer_even_when_user_omits_the_format(self):
        script = Path(__file__).resolve().parents[1] / "run_task1.py"
        with patch.object(sys, "argv", [str(script), "train", "--config", "first_run.json"]), \
                patch("task1.pipeline.execute") as dispatched:
            runpy.run_path(str(script), run_name="__main__")
        self.assertEqual(dispatched.call_args.args[0].label_format, "integer")

    def test_named_and_sample_formats_are_rejected_before_any_data_or_fit(self):
        for stage in ("check", "train", "finalize"):
            for label_format in ("name", "sample"):
                args = Namespace(stage=stage, label_format=label_format)
                with self.subTest(stage=stage, label_format=label_format), \
                        patch("task1.pipeline.load_data", side_effect=AssertionError("data/fit should not start")), \
                        self.assertRaisesRegex(ValueError, "integer labels"):
                    execute(args)

    def test_second_round_changes_only_hyperparameters_and_keeps_shared_protocol(self):
        root = Path(__file__).resolve().parents[2] / "configs/task1"
        first, second = load_config(root / "first_run.json"), load_config(root / "second_run.json")
        self.assertEqual(first["validation"], second["validation"])
        self.assertEqual(first["finalization"], second["finalization"])
        for old, new in zip(first["models"], second["models"]):
            self.assertEqual(old["model_id"], new["model_id"])
            self.assertEqual(old["features"], new["features"])
            self.assertEqual(old["group_features"], new["group_features"])
            self.assertEqual(old.get("noise_weighting"), new.get("noise_weighting"))
        for model in second["models"][:2]:
            self.assertEqual(model["model_params"]["max_iter"], 1600)
            self.assertEqual(model["model_params"]["learning_rate"], 0.015)
        self.assertEqual(second["models"][0]["model_params"], second["models"][1]["model_params"])
        self.assertEqual(first["models"][2], second["models"][2])


if __name__ == "__main__":
    unittest.main()
