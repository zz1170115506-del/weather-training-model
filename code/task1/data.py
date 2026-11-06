"""Read-only Dataset1 schema checks and common data versions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd

from . import LABELS
from .artifacts import content_hash, file_hash

NUMERIC = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
           "Atmospheric Pressure", "UV Index", "Visibility (km)"]
CATEGORICAL = ["Cloud Cover", "Season", "Location"]
FEATURES = NUMERIC + CATEGORICAL


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame
    audit: dict
    groups: np.ndarray

    @property
    def X(self):
        return self.train[FEATURES]

    @property
    def y(self):
        return self.train["label"].to_numpy(dtype=int)


def load_data(data_dir):
    data_dir = Path(data_dir)
    frames = {}
    for name in ("train", "test", "sample_submission"):
        frame = pd.read_csv(data_dir / f"{name}.csv", converters={"id": str})
        if "id" in frame:
            raw_ids = frame.id.tolist()
            if all(re.fullmatch(r"0|-?[1-9][0-9]*", value) for value in raw_ids):
                values = [int(value) for value in raw_ids]
                if all(-(2**63) <= value < 2**63 for value in values):
                    frame["id"] = np.asarray(values, dtype=np.int64)
                else:
                    frame["id"] = pd.Series(values, dtype=object)
        frames[name] = frame
    train, test, sample = (frames[k] for k in ("train", "test", "sample_submission"))
    if set(train.columns) != {"id", "label", *FEATURES}:
        raise ValueError("Dataset1 train schema must contain id, label and the ten weather attributes.")
    if set(test.columns) != {"id", *FEATURES} or list(sample.columns) != ["id", "label"]:
        raise ValueError("Dataset1 test/sample schema mismatch.")
    for name, frame in frames.items():
        if frame.empty or frame.id.isna().any() or not frame.id.is_unique or any(x == "" for x in frame.id):
            raise ValueError(f"{name}: nonempty data and unique nonmissing IDs required.")
    if set(test.id) != set(sample.id):
        raise ValueError("Sample/test ID sets differ.")
    if set(train.id) & set(test.id):
        raise ValueError("Training/test record IDs overlap.")
    if train.label.isna().any() or set(train.label) != set(LABELS):
        raise ValueError("Training labels must be exactly integers 1,2,3,4.")
    if not np.equal(train.label.to_numpy(), train.label.to_numpy(dtype=int)).all():
        raise ValueError("Fractional labels are not allowed.")
    for frame in (train, test):
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            if np.isinf(frame[col].to_numpy(dtype=float)).any():
                raise ValueError(f"{col}: infinite values are not allowed.")
    train = train.sort_values("id", kind="stable").reset_index(drop=True)
    keys = pd.util.hash_pandas_object(pd.concat([train[FEATURES], test[FEATURES]],
                                               ignore_index=True), index=False).to_numpy()
    groups, test_groups = keys[:len(train)], keys[len(train):]
    audit = {
        "rows": {"train": len(train), "test": len(test), "sample": len(sample)},
        "class_counts": {str(k): int((train.label == k).sum()) for k in LABELS},
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "feature_order": FEATURES, "id_used_as_feature": False,
        "schema_sha256": content_hash(FEATURES),
        "source_sha256": {f"{name}.csv": file_hash(data_dir / f"{name}.csv") for name in frames},
        "missing_counts": {name: {c: int(v) for c, v in frame.isna().sum().items()}
                           for name, frame in (("train", train), ("test", test))},
        "duplicate_feature_rows_train": int(pd.Series(groups).duplicated().sum()),
        "train_test_feature_overlap_rows": int(np.isin(test_groups, groups).sum()),
        "percentage_out_of_range": {
            name: {col: int(((frame[col] < 0) | (frame[col] > 100)).sum())
                   for col in ("Humidity", "Precipitation (%)")}
            for name, frame in (("train", train), ("test", test))},
        "sample_label_values": sorted(str(x) for x in sample.label.unique()),
        "sample_ids_same_order_as_test": sample.id.equals(test.id),
        "submission_format_basis": "Competition requires integers 1..4: the evaluator rejected English Rainy. The provided sample's name labels are not authoritative.",
    }
    return Dataset(train, test, sample, audit, groups)


def infer_label_format(sample):
    """Follow the provided sample, without claiming platform acceptance."""
    from . import LABEL_NAMES
    values = set(sample.label.dropna())
    if values and values.issubset(set(LABEL_NAMES.values())):
        return "name"
    if values and values.issubset(set(LABELS)):
        return "integer"
    raise ValueError("Cannot infer label encoding from sample; pass integer or name explicitly.")
