Dataset 1: data audit and exploratory baselines
=============================================

Run from the project root with Python 3.12:
  python3 -m venv .venv
  .venv/bin/python -m pip install -r code/requirements.txt
  .venv/bin/python code/dataset1_baseline.py

Inputs default to comp-5434-20261-project-task-1/.
Outputs default to outputs/dataset1_baseline/.
Use --help to change directories, seed, or validation fraction.

The split is stratified, with seed 5434 and 20% validation records.
All models use the same split. Preprocessing is fitted on training records
only. id and label are excluded from the model inputs. No outlier clipping
or record removal is applied. The source CSV files are never overwritten.

Models: majority-class reference, logistic regression, decision tree,
random forest, and histogram gradient boosting. These library models are
exploratory references, not the final self-developed project algorithm.
The best validation model is refitted on all training records to produce
baseline test predictions. Validation performance is a model-selection
estimate, not a Kaggle or independent final-test result.

IMPORTANT: train.csv uses integer labels 1..4, while sample_submission.csv
contains the string Rainy. Predictions use the integer mapping specified
by the assignment: 1 Sunny, 2 Cloudy, 3 Rainy, 4 Snowy. The actual Kaggle
label encoding still needs confirmation before a submission is made.
No Kaggle submission is performed by this script.

Main outputs:
  audit.json                 Schema, quality checks and source hashes.
  environment.json           Dependency versions and hardware information.
  metrics.csv                Validation scores and measured fit/predict times.
  details.json               Configuration, per-class scores, confusion matrices.
  validation_split.csv       Record IDs assigned to training or validation.
  validation_predictions.csv True/predicted labels and probabilities by model.
  baseline_predictions_integer.csv  Full-training-model test predictions.
  baseline_probabilities.csv  Probabilities for each test record.
  data_overview.png           Class balance and suspicious percentage counts.
  baseline_comparison.png     Validation comparison and best confusion matrix.

Runtime measurements are machine dependent. The exploratory script reports
fit/inference durations; it does not invent epoch timings for tree models.
