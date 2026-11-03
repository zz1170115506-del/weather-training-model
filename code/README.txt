Dataset 1: data audit, baselines and handwritten models
=====================================================

Run from the project root with Python 3.12:
  python3 -m venv .venv
  .venv/bin/python -m pip install -r code/requirements.txt
  .venv/bin/python code/dataset1_baseline.py

Run the handwritten CART and random forest comparison:
  .venv/bin/python code/dataset1_custom.py
  .venv/bin/python -m unittest discover -s code/tests -v

The custom runner writes outputs/dataset1_custom/ and defaults to 100 trees.
Its NumPy models and the sklearn reference forest use the same tree count,
leaf-size settings, split and fitted preprocessing. The saved baseline's
forest used 300 trees, so compare the matched models in the custom results.
Change the count with --n-estimators 300 if needed.

weather_models.py implements the following without sklearn:
  - Exact numeric CART split search using sorting and cumulative class counts.
  - Sample-weighted Gini, depth/leaf-size/impurity stopping conditions.
  - Leaf probabilities, batched prediction and impurity feature importance.
  - Bootstrap sampling, per-node random features and probability averaging.
  - Global class alignment, reproducible seeds and out-of-bag evaluation.
The runner still uses sklearn for preprocessing, metrics and reference models.

Bootstrap repeats count as repeated observations, including for leaf-size
constraints. RNG, tie-breaking and bootstrap bookkeeping differ from sklearn;
matched hyperparameters do not imply identical trees or predictions.

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

Custom outputs (outputs/dataset1_custom/):
  metrics.csv                Matched NumPy/sklearn scores and measured times.
  details.json               Configurations, structure summaries, OOB scores.
  feature_importances.csv    Importance of encoded columns on training split.
  original_feature_importances.csv  Importance summed by original attribute.
  custom_comparison.png       Matched scores and custom forest confusion matrix.
  feature_importance.png      Training-split custom forest attribute importance.
  validation_split.csv        IDs; uses the same seed/split as the baseline.
  validation_predictions.csv Validation labels/probabilities for all four models.
  custom_predictions_integer.csv  Best custom model's full-training predictions.
  custom_probabilities.csv    Test probabilities in integer-label order.
  custom_model.joblib         Fitted preprocessor, custom model and input schema.
The runner reloads the saved model and checks exact test-probability equality.
Record-level files and model binaries remain local and are Git-ignored.

Custom fit times exclude shared preprocessing and include OOB calculation for
both forest models. Total test inference includes preprocessing. Full-training
refit disables OOB evaluation, which does not change the trained predictors.
Feature importance is a training impurity statistic, not causal evidence.

Runtime measurements are machine dependent. The exploratory script reports
fit/inference durations; it does not invent epoch timings for tree models.
