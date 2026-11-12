"""Short synthetic checks of visible progress and unchanged CV calculations."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_task1_flow import synthetic_dataset
from task1.models import fit_model
from task1.progress import fold_progress
from task1.validation import cross_validate, make_folds


class NotifyingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.heartbeat = Event()
        self.flush_count = 0

    def write(self, text):
        result = super().write(text)
        if "仍在计算" in text:
            self.heartbeat.set()
        return result

    def flush(self):
        self.flush_count += 1
        return super().flush()


class FoldProgressTests(unittest.TestCase):
    def test_start_is_immediate_and_success_stops_the_daemon_without_waiting_an_interval(self):
        stream = NotifyingStream()
        clock = [100.0]
        with fold_progress("synthetic", 2, 5, 80, 20, interval_seconds=30,
                           stream=stream, clock=lambda: clock[0]) as progress:
            start = stream.getvalue()
            self.assertIn("model=synthetic fold=2/5", start)
            self.assertIn("fit_rows=80 valid_rows=20", start)
            self.assertGreater(stream.flush_count, 0)
            self.assertTrue(progress.thread.daemon)
            self.assertTrue(progress.thread.is_alive())
            clock[0] = 104.25
            self.assertEqual(progress.elapsed_seconds(), 4.25)
        self.assertTrue(progress.stopped.is_set())
        self.assertFalse(progress.thread.is_alive())
        self.assertNotIn("%", stream.getvalue())
        self.assertNotIn("ETA", stream.getvalue())

    def test_a_short_injected_interval_reports_elapsed_time_then_leaves_no_worker(self):
        stream = NotifyingStream()
        with fold_progress("slow_fixture", 1, 2, 16, 16, interval_seconds=.001,
                           stream=stream, clock=lambda: 50.0) as progress:
            self.assertTrue(stream.heartbeat.wait(1), "The heartbeat thread did not emit its short test message.")
        self.assertIn("仍在计算 elapsed=0.0s", stream.getvalue())
        self.assertFalse(progress.thread.is_alive())

    def test_body_exception_is_preserved_and_stops_the_worker(self):
        stream = NotifyingStream()
        error = RuntimeError("synthetic fit failed")
        with self.assertRaises(RuntimeError) as captured:
            with fold_progress("failed_fixture", 1, 2, 16, 16,
                               stream=stream, interval_seconds=30) as progress:
                raise error
        self.assertIs(captured.exception, error)
        self.assertFalse(progress.thread.is_alive())
        self.assertTrue(progress.stopped.is_set())

    def test_closed_console_does_not_change_the_work_or_mask_an_error(self):
        stream = io.StringIO()
        stream.close()
        with fold_progress("closed_console", 1, 2, 4, 4, stream=stream) as progress:
            result = 6 * 7
        self.assertEqual(result, 42)
        self.assertFalse(progress.thread.is_alive())

    def test_invalid_intervals_fail_before_any_worker_starts(self):
        for interval in (0, -1, True, float("nan"), float("inf"), "30"):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                with fold_progress("invalid", 1, 2, 4, 4, interval_seconds=interval):
                    self.fail("Invalid progress configuration reached work.")

    def test_cv_with_progress_keeps_fit_arguments_metrics_and_predictions_identical(self):
        data = synthetic_dataset()
        config = {"name": "tiny_cart", "model_id": "numpy_cart",
                  "model_params": {"max_depth": 3, "min_samples_leaf": 2}}
        snapshot = deepcopy(config)
        folds, _ = make_folds(data.y, data.train.id.to_numpy(), data.groups, n_splits=2, seed=83)
        observed = []

        def recording_fit(X, y, candidate, seed):
            observed.append((X.index.tolist(), y.tolist(), deepcopy(candidate), seed))
            return fit_model(X, y, candidate, seed)

        class SilentProgress:
            @staticmethod
            def elapsed_seconds():
                return 0.0

        with redirect_stdout(io.StringIO()), patch("task1.validation.fit_model", side_effect=recording_fit), \
                patch("task1.validation.fold_progress", return_value=nullcontext(SilentProgress())):
            reference, expected = cross_validate(data, config, folds, model_seed=91, verbose=False)
        reference_arguments = deepcopy(observed)
        observed.clear()
        stream = io.StringIO()
        with redirect_stdout(stream), patch("task1.validation.fit_model", side_effect=recording_fit):
            actual, probabilities = cross_validate(data, config, folds, model_seed=91, verbose=False)
        self.assertEqual(observed, reference_arguments)
        self.assertEqual(config, snapshot)
        np.testing.assert_array_equal(probabilities, expected)
        self.assertEqual(actual["oof_metrics"], reference["oof_metrics"])
        self.assertEqual(actual["fold_macro_f1_mean"], reference["fold_macro_f1_mean"])
        self.assertEqual(actual["fold_macro_f1_std"], reference["fold_macro_f1_std"])
        self.assertEqual(stream.getvalue().count("开始计算"), 2)
        self.assertIn("model=tiny_cart fold=1/2", stream.getvalue())
        self.assertIn("fit_rows=16 valid_rows=16", stream.getvalue())
        self.assertNotIn("Macro-F1=", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
