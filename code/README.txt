Task1: one course-only workflow and a reproducible code release
=============================================================

Use Python 3.12. The only maintained training entry point is code/run_task1.py.
The former dataset1_baseline.py / dataset1_custom.py standalone holdout scripts
were retired to local history; their aggregate historical results are retained.
Do not create a second training workflow by copying those scripts back.

Run these commands from the repository/package root in PowerShell. Calling
the virtual-environment interpreter directly avoids activation-policy issues:

  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r code/requirements-round7.txt

Use the original course Dataset1 only: train.csv, test.csv and
sample_submission.csv. Never add public-source labels or external records.
The formal configuration pins all three source SHA256 hashes. The program
refuses changed inputs and retired source-assisted configurations/bundles.

  $taskData = Read-Host 'Directory containing the three original course CSVs'
  .\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py check --config configs/task1/seventh_run_course_only.json --data-dir $taskData

check does not fit a model. The user starts real training manually:

  .\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py train --config configs/task1/seventh_run_course_only.json --data-dir $taskData --artifact-root ../local_artifacts/task1
  .\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py results --artifact-root ../local_artifacts/task1

Historical configs remain in configs/task1/ to explain and reproduce previous
course-only experiments; they are not the current recommended command. The
release exporter includes only the single configuration explicitly selected.

Module responsibilities
-----------------------

run_task1.py                  CLI and conservative CPU thread defaults.
task1/data.py                 Immutable course inputs, schema and source hashes.
task1/features.py             Fold-local preprocessing and feature transforms.
task1/models.py               Fresh estimators, training-only weighting/stopping.
weather_models.py             NumPy CART/Gini/forest/bootstrap/OOB algorithms.
generative_weather.py         Autonomous histogram mixture and EM weighting.
task1/bagged_hgb.py            Stratified member sampling and probability mean.
task1/crossfit_weights.py      Internal OOF confidence weights; labels unchanged.
task1/ovr_hgb.py               Custom one-vs-rest probability combination.
task1/validation.py            Shared folds, OOF coverage and per-class metrics.
task1/search.py, ensemble.py   Explicit selection and probability fusion.
task1/progress.py              Training progress/heartbeat records.
task1/artifacts.py             Frozen config, source/environment/data provenance.
task1/submission.py            Lossless IDs, integer labels, CSV/hash validation.
task1/results.py, plots.py     Verified local metrics and aggregate figures.
verify_task1_release.py        Data-free package export and non-training checks.

Library learners are controls/components. Self-developed sampling, weighting,
tree/forest, mixture-model and fusion details must be explained and evaluated;
merely calling a ready-made classifier is insufficient for the course rubric.

Outputs and scores
------------------

Every run has a new directory; historical configs, metrics and models are not
overwritten. Retain resolved_config.json, source/CSV/fold hashes, environment,
per-fold effective parameters, timing, confusion matrices, selection records,
candidate manifests and exact model-reload replay checks for later reporting.
Model binaries and record-level probabilities/predictions remain local.

Candidate letters describe export slots, not a fixed model. Read results before
uploading. Kaggle requires id,label with 3300 predictions and a header; weather
labels must be integers 1..4 (Sunny/Cloudy/Rainy/Snowy respectively). English
_name.csv files are inspection aids and must not be submitted. Upload manually,
then use record-score with the actual run, candidate, submission ID and score.
Development OOF is not Kaggle public/private performance. No hidden-test score
is guaranteed, capped, or manufactured by deliberately changing predictions.

Code-only release for a collaborator or teacher
----------------------------------------------

Choose a new output directory outside the source checkout:

  .\.venv\Scripts\python.exe -X utf8 -B code/verify_task1_release.py --config configs/task1/seventh_run_course_only.json --output ../local_releases/task1_round7

The exporter copies a fixed allowlist: runtime source, handwritten algorithms,
dependencies, this README, a package-root README and exactly one config. No
data, models, predictions, old configs/results, Git metadata, environment or
private notes are copied. It refuses an existing destination and writes a
SHA256 manifest. It verifies imports/model construction, --help and a separate
128-row generated-data check, never real-data training. That fixture does not
replace or relax the production source hashes and produces no accuracy claim.

Recheck an extracted package after installing dependencies:

  python -X utf8 -B code/verify_task1_release.py --verify-only --output .

The release root README supplies explicit --data-dir commands so a teacher can
provide course inputs separately. Keep an extracted package outside a dirty
parent Git checkout. On macOS/Linux use .venv/bin/python instead of the Windows
interpreter path. SHA256 proves consistency, not trust in an unknown archive.

Repository tests use synthetic data and handwritten artifact fixtures only:

  .\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s code/tests -v
