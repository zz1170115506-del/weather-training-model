"""Reproducible Dataset1 CV, nested blending, final predictions and CSV checks."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set limits before importing numerical libraries. A caller can override them.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "train", "audit", "cv", "nested", "finalize", "validate-submission", "predict", "record-score", "results"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Dataset")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--label-format", choices=("sample", "integer", "name"), default="integer",
                        help="Competition requires integers 1..4. name/sample are for inspecting historical files only.")
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path, help="Completed manual-training run for results or record-score; results defaults to the latest run.")
    parser.add_argument("--compare-run", type=Path, help="Prior same-data/same-fold run to compare in results.")
    parser.add_argument("--candidate", default="candidate_a", help="Candidate key printed by train.")
    parser.add_argument("--submission-id", help="Submission ID shown by Kaggle, recorded locally only.")
    parser.add_argument("--public-score", type=float)
    parser.add_argument("--note", default="")
    parser.add_argument("--allow-dirty", action="store_true", help="Development only: mark non-frozen source explicitly.")
    args = parser.parse_args()
    from task1.pipeline import execute
    execute(args)


if __name__ == "__main__":
    main()
