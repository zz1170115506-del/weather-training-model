"""Validate and export Task1 predictions without network access.

IDs are either strings or integers, never floats. CSV has no intrinsic ID type,
so file validation restores that type from ``sample`` rather than inferring it.
This preserves leading zeroes and integers beyond the float/64-bit range.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_NAMES = {1: "Sunny", 2: "Cloudy", 3: "Rainy", 4: "Snowy"}
_NAME_LABELS = {name: label for label, name in LABEL_NAMES.items()}
_SCHEMA = ["id", "label"]
_CANONICAL_VERSION = "task1-predictions-v1"


def _check_format(label_format):
    if not isinstance(label_format, str) or label_format not in ("integer", "name"):
        raise ValueError("label_format must be explicitly confirmed as 'integer' or 'name'.")


def _values(values, name):
    if isinstance(values, (str, bytes, pd.DataFrame)) or getattr(values, "ndim", 1) != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence.")
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence.") from exc


def _ids(values, context):
    """Return lossless typed keys, rejecting ambiguous/coerced ID columns."""
    keys, kind = [], None
    for value in values:
        if value is None or value is pd.NA or value is pd.NaT:
            raise ValueError(f"{context}: IDs must not be missing.")
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{context}: boolean IDs are not supported.")
        if isinstance(value, Integral):
            key = ("integer", str(int(value)))
        elif isinstance(value, str):
            if not value:
                raise ValueError(f"{context}: IDs must not be empty.")
            key = ("string", value)
        else:
            raise ValueError(f"{context}: IDs must be integers or strings; floating-point IDs are refused to avoid precision loss.")
        if kind is not None and key[0] != kind:
            raise ValueError(f"{context}: mixed integer/string ID types are not supported.")
        kind = key[0]
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{context}: IDs must be unique.")
    return keys, kind


def _sample_keys(sample):
    if not isinstance(sample, pd.DataFrame) or not sample.columns.is_unique or "id" not in sample.columns:
        raise ValueError("sample must be a DataFrame with a unique 'id' column.")
    if sample.empty:
        raise ValueError("sample must contain at least one record.")
    return _ids(sample["id"].tolist(), "sample")


def _integer_labels(values):
    labels = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) not in LABEL_NAMES:
            raise ValueError("Integer labels must be non-missing integers in {1, 2, 3, 4}.")
        labels.append(int(value))
    return labels


def _labels(values, label_format):
    if label_format == "integer":
        return _integer_labels(values)
    labels = []
    for value in values:
        if not isinstance(value, str) or value not in _NAME_LABELS:
            raise ValueError("Name labels must be exactly Sunny, Cloudy, Rainy, or Snowy.")
        labels.append(_NAME_LABELS[value])
    return labels


def _csv_bytes(keys, labels, label_format):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_SCHEMA)
    for (_, identifier), label in zip(keys, labels):
        writer.writerow((identifier, label if label_format == "integer" else LABEL_NAMES[label]))
    return stream.getvalue().encode("utf-8")


def _metadata(keys, labels, label_format, csv_bytes):
    # Versioned JSON-lines: ID type and exact decimal/string value are explicit.
    # Labels are always internal integers, so export formats have the same hash.
    canonical = hashlib.sha256((_CANONICAL_VERSION + "\n").encode("ascii"))
    for (kind, identifier), label in zip(keys, labels):
        row = json.dumps([kind, identifier, label], ensure_ascii=False, separators=(",", ":"))
        canonical.update((row + "\n").encode("utf-8"))
    return {
        "rows": len(labels),
        "schema": list(_SCHEMA),
        "id_type": keys[0][0],
        "label_format": label_format,
        "class_counts": {str(label): labels.count(label) for label in LABEL_NAMES},
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "canonical_prediction_sha256": canonical.hexdigest(),
        "canonical_prediction_version": _CANONICAL_VERSION,
    }


def validate_submission(frame, sample, label_format):
    """Validate exact schema, ID membership/order and labels; return metadata.

    ``sample`` supplies the authoritative ID order and type. Its label values
    are ignored, and it need only contain an ``id`` column. No input is mutated.
    ``csv_sha256`` describes the canonical UTF-8/LF serialization of ``frame``;
    use ``validate_submission_file`` for the hash of existing file bytes.
    """
    _check_format(label_format)
    sample_keys, sample_kind = _sample_keys(sample)
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != _SCHEMA:
        raise ValueError("Submission schema must be exactly ['id', 'label'] in that order.")
    if len(frame) != len(sample_keys):
        raise ValueError("Submission row count must match sample.")
    keys, kind = _ids(frame["id"].tolist(), "submission")
    if kind != sample_kind or set(keys) != set(sample_keys):
        raise ValueError("Submission IDs and their type must match sample exactly.")
    if keys != sample_keys:
        raise ValueError("Submission IDs must follow sample order.")
    labels = _labels(frame["label"].tolist(), label_format)
    return _metadata(keys, labels, label_format, _csv_bytes(keys, labels, label_format))


def _read_submission_bytes(content, sample, label_format):
    sample_keys, kind = _sample_keys(sample)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Submission must use UTF-8 encoding.") from exc
    if text.startswith("\ufeff"):
        raise ValueError("Submission must use UTF-8 without a BOM.")
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise ValueError("Submission contains malformed CSV.") from exc
    if not rows or rows[0] != _SCHEMA:
        raise ValueError("Submission CSV header must be exactly id,label.")
    ids, labels = [], []
    for row in rows[1:]:
        if len(row) != 2:
            raise ValueError("Each submission row must have exactly two fields.")
        raw_id, raw_label = row
        if kind == "integer":
            try:
                identifier = int(raw_id)
            except ValueError as exc:
                raise ValueError("Submission integer IDs must be exact decimal integers.") from exc
            if raw_id != str(identifier):
                raise ValueError("Submission integer IDs must use canonical decimal notation.")
        else:
            identifier = raw_id
        if label_format == "integer":
            if raw_label not in ("1", "2", "3", "4"):
                raise ValueError("Submission CSV integer labels must be 1, 2, 3, or 4.")
            label = int(raw_label)
        else:
            label = raw_label
        ids.append(identifier)
        labels.append(label)
    # Object dtype prevents inference/coercion of arbitrary-precision integers.
    frame = pd.DataFrame({"id": pd.Series(ids, dtype=object), "label": pd.Series(labels, dtype=object)})
    metadata = validate_submission(frame, sample, label_format)
    metadata["csv_sha256"] = hashlib.sha256(content).hexdigest()
    return metadata


def validate_submission_file(path, sample, label_format):
    """Read and validate CSV without lossy ID inference; hash actual bytes.

    Existing CRLF files can be checked: their semantic hash equals an LF file,
    while their byte hash differs. New files always use UTF-8 without BOM/LF.
    """
    _check_format(label_format)
    return _read_submission_bytes(Path(path).read_bytes(), sample, label_format)


def write_submission(ids, predictions, sample, path, label_format):
    """Export internal integer predictions in sample order and validate back.

    Returns metadata, including byte and semantic SHA256 hashes. Refuses to
    overwrite existing paths. Validation completes before creating directories
    or opening the output; floating-point IDs/labels are never silently cast.
    """
    _check_format(label_format)
    sample_keys, sample_kind = _sample_keys(sample)
    id_values = _values(ids, "ids")
    prediction_values = _values(predictions, "predictions")
    if len(id_values) != len(prediction_values) or len(id_values) != len(sample_keys):
        raise ValueError("IDs, predictions and sample must have the same row count.")
    keys, kind = _ids(id_values, "predictions")
    if kind != sample_kind or set(keys) != set(sample_keys):
        raise ValueError("Prediction IDs and their type must match sample exactly.")
    labels = _integer_labels(prediction_values)
    by_id = dict(zip(keys, labels))
    ordered_labels = [by_id[key] for key in sample_keys]
    content = _csv_bytes(sample_keys, ordered_labels, label_format)
    expected = _read_submission_bytes(content, sample, label_format)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with target.open("xb") as stream:
            created = True
            stream.write(content)
        actual = validate_submission_file(target, sample, label_format)
        if actual != expected:
            raise OSError("Written submission does not match validated output.")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return actual
