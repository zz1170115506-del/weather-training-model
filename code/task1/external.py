"""Audited local public-source training data, with explicit overlap policies.

``novel_only`` never consumes labels of source rows matching course train/test.
``public_source_training`` can retain public-source test-overlap labels and must
only be configured when that use has been explicitly authorized. Neither mode
adds another copy of any original course-training feature group.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd

from .data import CATEGORICAL, FEATURES, NUMERIC


POLICIES = ("novel_only", "public_source_training")
FEATURE_KEY_VERSION = "weather-features-float64-v1"
_LABEL_COLUMN = "Weather Type"
_LABELS = {"sunny": 1, "cloudy": 2, "rainy": 3, "snowy": 4}


@dataclass(frozen=True)
class ExternalContext:
    """A globally cleaned pool that survives nested Dataset ``replace`` calls.

    Frame contains only the ten features and internal integer label. Feature
    identity uses casefolded categories; model inputs retain original category
    capitalization to remain compatible with the existing feature pipeline.
    """

    frame: pd.DataFrame
    feature_keys: tuple[str, ...]
    source_row_indices: tuple[int, ...]
    course_train_feature_keys: frozenset[str]
    course_test_feature_keys: frozenset[str]
    audit: dict

    @property
    def features(self):
        return self.frame[FEATURES].copy(deep=True)

    @property
    def labels(self):
        return self.frame["label"].to_numpy(dtype=np.int64, copy=True)

    @property
    def X(self):
        return self.features

    @property
    def y(self):
        return self.labels


def _missing(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    result = pd.isna(value)
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _clean_features(frame):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not set(FEATURES).issubset(frame.columns):
        raise ValueError("External feature matching requires the ten unique weather feature columns.")
    cleaned = frame[FEATURES].copy(deep=True)
    for column in NUMERIC:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise").astype(np.float64)
        if np.isinf(cleaned[column].to_numpy()).any():
            raise ValueError(f"{column}: infinite values are not valid external features.")
    for column in CATEGORICAL:
        values = []
        for value in cleaned[column]:
            if _missing(value):
                values.append(np.nan)
            elif not isinstance(value, str):
                raise ValueError(f"{column}: categorical values must be strings or missing.")
            else:
                values.append(value.strip() or np.nan)
        cleaned[column] = pd.Series(values, index=cleaned.index, dtype=object)
    return cleaned


def _keys_from_cleaned(cleaned):
    keys = []
    numeric = cleaned[NUMERIC].to_numpy(dtype=np.float64)
    categorical = cleaned[CATEGORICAL].to_numpy(dtype=object)
    for numbers, categories in zip(numeric, categorical):
        digest = hashlib.sha256((FEATURE_KEY_VERSION + "\0").encode("ascii"))
        for number in numbers:
            if np.isnan(number):
                digest.update(b"N")
            else:
                # Fixed little-endian binary64, with -0 and +0 identified.
                digest.update(b"F" + struct.pack("<d", 0.0 if number == 0 else float(number)))
        for value in categories:
            if _missing(value):
                digest.update(b"N")
            else:
                encoded = value.strip().casefold().encode("utf-8")
                digest.update(b"S" + struct.pack("<Q", len(encoded)) + encoded)
        keys.append(digest.hexdigest())
    return tuple(keys)


def feature_keys(frame):
    """Canonical SHA256 keys, independent of IDs, labels, index and CSV dtypes."""
    return _keys_from_cleaned(_clean_features(frame))


def _pool_hash(keys, labels):
    digest = hashlib.sha256(b"task1-external-pool-v1\n")
    for key, label in sorted(zip(keys, labels)):
        digest.update(f"{key},{int(label)}\n".encode("ascii"))
    return digest.hexdigest()


def _frame_hash(frame):
    """Also fingerprint case-sensitive model inputs, not only identity keys."""
    digest = hashlib.sha256(b"task1-external-training-values-v1\n")
    for row in frame[FEATURES + ["label"]].itertuples(index=False, name=None):
        values = []
        for column, value in zip(FEATURES + ["label"], row):
            if _missing(value):
                values.append(None)
            elif column in NUMERIC:
                values.append(float(value).hex())
            elif column == "label":
                values.append(int(value))
            else:
                values.append(value)
        digest.update((json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _settings(settings):
    if not isinstance(settings, dict) or set(settings) != {"path", "sha256", "source_url", "policy"}:
        raise ValueError("external_data requires exactly path, sha256, source_url and policy.")
    if settings["policy"] not in POLICIES:
        raise ValueError("external_data.policy must be novel_only or public_source_training.")
    if not isinstance(settings["path"], str) or not settings["path"].strip() or "\x00" in settings["path"]:
        raise ValueError("external_data.path must be a nonempty local path.")
    if not isinstance(settings["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", settings["sha256"]) is None:
        raise ValueError("external_data.sha256 must be a lowercase SHA256 digest.")
    if not isinstance(settings["source_url"], str) or not settings["source_url"].strip():
        raise ValueError("external_data.source_url must identify the source.")


def attach_external(data, settings, root: Path):
    """Read a pinned CSV once, clean its pool, and return a new Dataset.

    Relative source paths are always relative to ``root`` (the repository),
    never relative to a mutable or copied configuration-file location. All
    identity exclusions happen before retained source labels are interpreted.
    """
    _settings(settings)
    if getattr(data, "external", None) is not None:
        raise ValueError("Attach external data once at the full-course data-loading boundary.")
    path = (Path(root) / settings["path"]).resolve()
    content = path.read_bytes()
    source_sha256 = hashlib.sha256(content).hexdigest()
    if source_sha256 != settings["sha256"]:
        raise ValueError("External source SHA256 mismatch; refusing changed source data.")
    source = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    if not source.columns.is_unique or set(source.columns) != {*FEATURES, _LABEL_COLUMN}:
        raise ValueError("External source schema must be the ten weather features plus Weather Type.")
    cleaned = _clean_features(source)
    keys = _keys_from_cleaned(cleaned)
    train_keys = frozenset(feature_keys(data.train))
    test_keys = frozenset(feature_keys(data.test))
    train_overlap = np.asarray([key in train_keys for key in keys], dtype=bool)
    test_overlap = np.asarray([key in test_keys for key in keys], dtype=bool)
    keep = ~train_overlap
    excluded_test = np.zeros(len(source), dtype=bool)
    if settings["policy"] == "novel_only":
        excluded_test = keep & test_overlap
        keep &= ~test_overlap

    indices = np.flatnonzero(keep)
    retained = cleaned.iloc[indices].copy().reset_index(drop=True)
    labels = []
    # Deliberately do not inspect labels belonging to excluded source rows.
    for value in source[_LABEL_COLUMN].iloc[indices]:
        if not isinstance(value, str) or value.strip().casefold() not in _LABELS:
            raise ValueError("Retained source labels must be Sunny, Cloudy, Rainy or Snowy.")
        labels.append(_LABELS[value.strip().casefold()])
    retained["label"] = pd.Series(labels, dtype=np.int64)
    retained_keys = [keys[index] for index in indices]
    key_labels = {}
    for key, label in zip(retained_keys, labels):
        key_labels.setdefault(key, set()).add(label)
    conflicts = {key for key, values in key_labels.items() if len(values) > 1}
    conflict_rows = sum(key in conflicts for key in retained_keys)
    selected, seen = [], set()
    for index, key in enumerate(retained_keys):
        if key not in conflicts and key not in seen:
            selected.append(index)
            seen.add(key)
    duplicate_rows = len(retained) - conflict_rows - len(selected)
    pool = retained.iloc[selected].copy().reset_index(drop=True)
    pool_keys = tuple(retained_keys[index] for index in selected)
    source_indices = tuple(int(indices[index]) for index in selected)
    test_used = sum(key in test_keys for key in pool_keys)
    novel_rows = sum(key not in test_keys and key not in train_keys for key in pool_keys)
    policy = settings["policy"]
    audit = {
        "policy": policy, "source_path": str(path), "configured_path": settings["path"],
        "source_url": settings["source_url"], "source_sha256": source_sha256,
        "source_rows": len(source), "feature_order": list(FEATURES),
        "feature_key_version": FEATURE_KEY_VERSION, "label_column": _LABEL_COLUMN,
        "course_train_feature_groups": len(train_keys), "course_test_feature_groups": len(test_keys),
        "course_train_overlap_source_rows": int(train_overlap.sum()),
        "course_test_overlap_source_rows": int(test_overlap.sum()),
        "excluded_course_train_source_rows": int(train_overlap.sum()),
        "excluded_course_test_source_rows": int(excluded_test.sum()),
        "remaining_before_source_deduplication": len(retained),
        "conflicting_label_groups": len(conflicts), "excluded_conflicting_source_rows": int(conflict_rows),
        "same_label_duplicate_rows_removed": int(duplicate_rows),
        "external_pool_rows": len(pool), "test_overlap_source_rows_used": int(test_used),
        "independent_new_rows": int(novel_rows), "augmentation_available": bool(len(pool)),
        "status": "no_novel_rows" if not len(pool) else "source_assisted_only" if not novel_rows else "novel_rows_available",
        "pool_sha256": _pool_hash(pool_keys, pool["label"].tolist()),
        "training_pool_sha256": _frame_hash(pool),
        "exclude_all_original_course_train_groups": True,
        "exclude_course_test_groups": policy == "novel_only",
        "test_label_use": "excluded before label mapping" if policy == "novel_only" else "public-source test-overlap labels explicitly retained by configured policy",
        "independence_scope": "Source-assisted test-overlap rows are not independent new observations; course-training copies are excluded globally, including nested held-out records.",
    }
    context = ExternalContext(pool, pool_keys, source_indices, train_keys, test_keys, audit)
    combined_audit = deepcopy(data.audit)
    combined_audit["external_data"] = deepcopy(audit)
    return replace(data, external=context, audit=combined_audit)


def augment_training(data, X, y):
    """Append the allowed source pool to any supplied course-training subset.

    The full original course-training exclusion context survives nested data
    slicing. X-overlap is excluded again defensively. Empty/absent augmentation
    returns the original X and y objects without changing their indices/types.
    """
    course_rows = len(X)
    if len(y) != course_rows:
        raise ValueError("Course training features and labels must have matching lengths.")
    metadata = {"course_fit_rows": course_rows, "external_fit_rows": 0,
                "combined_fit_rows": course_rows, "augmentation_applied": False,
                "policy": None, "pool_sha256": None, "used_pool_sha256": None,
                "source_sha256": None, "source_hashes": {},
                "test_overlap_source_rows_used": 0, "independent_new_rows": 0,
                "excluded_fit_overlap_rows": 0, "excluded_global_course_train_rows": 0}
    context = getattr(data, "external", None)
    if context is None:
        return X, y, metadata
    if not isinstance(context, ExternalContext):
        raise ValueError("Dataset external context must be an audited ExternalContext.")
    metadata.update(policy=context.audit["policy"], pool_sha256=context.audit["pool_sha256"],
                    source_sha256=context.audit["source_sha256"],
                    source_hashes={"external_source": context.audit["source_sha256"]},
                    training_pool_sha256=context.audit["training_pool_sha256"],
                    exclude_all_original_course_train_groups=True,
                    public_source_test_overlap_declared=context.audit["policy"] == "public_source_training")
    if context.frame.empty:
        metadata["used_pool_sha256"] = _pool_hash((), ())
        return X, y, metadata
    if feature_keys(context.frame) != context.feature_keys or _frame_hash(context.frame) != context.audit["training_pool_sha256"]:
        raise ValueError("External training pool changed after its audit.")
    if not isinstance(X, pd.DataFrame) or list(X.columns) != list(dict.fromkeys(X.columns)) or set(X.columns) != set(FEATURES):
        raise ValueError("Training augmentation requires exactly the ten weather feature columns.")
    fit_keys = frozenset(feature_keys(X))
    keep = []
    for index, key in enumerate(context.feature_keys):
        if key in context.course_train_feature_keys:
            metadata["excluded_global_course_train_rows"] += 1
        elif key in fit_keys:
            metadata["excluded_fit_overlap_rows"] += 1
        elif context.audit["policy"] == "novel_only" and key in context.course_test_feature_keys:
            raise ValueError("Novel-only context unexpectedly contains a course-test overlap.")
        else:
            keep.append(index)
    used_keys = tuple(context.feature_keys[index] for index in keep)
    used_labels = context.frame["label"].iloc[keep].to_numpy(dtype=np.int64)
    metadata.update(external_fit_rows=len(keep), combined_fit_rows=course_rows + len(keep),
                    augmentation_applied=bool(keep), used_pool_sha256=_pool_hash(used_keys, used_labels),
                    test_overlap_source_rows_used=sum(key in context.course_test_feature_keys for key in used_keys),
                    independent_new_rows=sum(key not in context.course_test_feature_keys and key not in context.course_train_feature_keys for key in used_keys))
    if not keep:
        return X, y, metadata
    values = np.asarray(y)
    if values.ndim != 1 or values.dtype.kind not in "iu" or not np.isin(values, [1, 2, 3, 4]).all():
        raise ValueError("Course training labels must be a one-dimensional integer array in 1..4.")
    combined_X = pd.concat([X.copy(deep=True), context.frame.iloc[keep][list(X.columns)].copy(deep=True)], ignore_index=True)
    combined_y = np.concatenate([values.astype(np.int64), used_labels])
    return combined_X, combined_y, metadata
