"""Fetch the pinned public weather source without login or model training.

The course/source overlap is recorded elsewhere; downloading this file is not
an independent-data or model-performance claim. Existing validated inputs and
historical provenance are retained. Only the exact CSV ZIP member is read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import stat
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REF = "nikhil7280/weather-type-classification"
SOURCE_VERSION = 1
CSV_FILENAME = "weather_classification_data.csv"
EXPECTED_CSV_SHA256 = "700fb0a5bd4b0b4ca837a098e9ba7be094a668795d2d621fd9cc0999a2917bf1"
DATASET_URL = f"https://www.kaggle.com/datasets/{SOURCE_REF}"
METADATA_URL = f"https://www.kaggle.com/api/v1/datasets/metadata/{SOURCE_REF}"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{SOURCE_REF}?datasetVersionNumber={SOURCE_VERSION}"
DEFAULT_OUTPUT_DIR = ROOT.parent / "external_data" / "weather_type_classification" / "v1"
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_CSV_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 512 * 1024
LICENSE_STATEMENT = "Anyone is free to share and use the data"


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _read_limited(stream, limit, context):
    chunks, size = [], 0
    while True:
        block = stream.read(min(65536, limit - size + 1))
        if not block:
            return b"".join(chunks)
        size += len(block)
        if size > limit:
            raise ValueError(f"{context} exceeds the {limit}-byte size limit.")
        chunks.append(block)


def _file_bytes(path, limit, context):
    if path.stat().st_size > limit:
        raise ValueError(f"{context} exceeds the {limit}-byte size limit.")
    with path.open("rb") as stream:
        return _read_limited(stream, limit, context)


def _download(url, limit, timeout):
    # No Kaggle SDK, login, credential files, Authorization, or cookies.
    request = Request(url, headers={"User-Agent": "weather-training-model-source-fetch/1"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                declared = int(length)
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid HTTP Content-Length.") from error
            if declared < 0 or declared > limit:
                raise ValueError(f"Download exceeds the {limit}-byte size limit.")
        return _read_limited(response, limit, "Download")


def _source_csv(archive_bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            matches = [entry for entry in archive.infolist() if entry.filename == CSV_FILENAME]
            if len(matches) != 1:
                raise ValueError(f"ZIP must contain exactly one member named {CSV_FILENAME!r}.")
            member = matches[0]
            unix_mode = member.external_attr >> 16
            if member.is_dir() or stat.S_ISLNK(unix_mode) or member.flag_bits & 1:
                raise ValueError("CSV ZIP member must be a regular unencrypted file.")
            if member.file_size > MAX_CSV_BYTES:
                raise ValueError("CSV ZIP member exceeds its uncompressed size limit.")
            with archive.open(member, "r") as stream:
                content = _read_limited(stream, MAX_CSV_BYTES, "Uncompressed CSV")
    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError) as error:
        raise ValueError("Invalid source ZIP archive.") from error
    if _sha256(content) != EXPECTED_CSV_SHA256:
        raise ValueError("Source CSV SHA256 mismatch; refusing an unrecognized source/version.")
    return content


def _metadata(raw):
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Metadata must be valid UTF-8 JSON.") from error
    info = value.get("info") if isinstance(value, dict) else None
    if not isinstance(info, dict):
        raise ValueError("Metadata is missing the official dataset info object.")
    if info.get("ownerUser") != "nikhil7280" or info.get("datasetSlug") != "weather-type-classification":
        raise ValueError("Metadata identifies a different dataset.")
    licenses = info.get("licenses")
    if not isinstance(licenses, list) or not any(isinstance(item, dict) and item.get("name") == "other" for item in licenses):
        raise ValueError("Expected the source's declared 'other' license.")
    description = info.get("description")
    if not isinstance(description, str) or LICENSE_STATEMENT.casefold() not in description.casefold():
        raise ValueError("Metadata does not contain the source's sharing/use statement.")
    return value


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path, content):
    # Exclusive creation also prevents accidental replacement during a race.
    with path.open("xb") as stream:
        stream.write(content)


def fetch_source(output_dir=None, timeout=30.0):
    """Download missing pinned inputs, or verify an existing CSV without refetch.

    A missing metadata file may be fetched separately. Existing provenance is
    never rewritten; a supplemental metadata retrieval gets its own timestamped
    record. Network access is limited to the two public constant URLs above.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number of seconds.")
    output = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
    csv_path, archive_path = output / CSV_FILENAME, output / "source.zip"
    metadata_path, provenance_path = output / "metadata.json", output / "download_provenance.json"
    had_csv, had_metadata, had_provenance = csv_path.exists(), metadata_path.exists(), provenance_path.exists()
    downloaded = []
    archive_bytes = None
    if had_csv:
        csv_bytes = _file_bytes(csv_path, MAX_CSV_BYTES, "Existing CSV")
        if _sha256(csv_bytes) != EXPECTED_CSV_SHA256:
            raise ValueError("Existing CSV SHA256 mismatch; refusing to overwrite it.")
    else:
        if archive_path.exists():
            archive_bytes = _file_bytes(archive_path, MAX_ARCHIVE_BYTES, "Existing ZIP")
        else:
            archive_bytes = _download(DOWNLOAD_URL, MAX_ARCHIVE_BYTES, timeout)
            downloaded.append(DOWNLOAD_URL)
        csv_bytes = _source_csv(archive_bytes)
    if had_metadata:
        metadata_bytes = _file_bytes(metadata_path, MAX_METADATA_BYTES, "Existing metadata")
    else:
        metadata_bytes = _download(METADATA_URL, MAX_METADATA_BYTES, timeout)
        downloaded.append(METADATA_URL)
    _metadata(metadata_bytes)
    if had_provenance:
        try:
            previous = json.loads(_file_bytes(provenance_path, MAX_METADATA_BYTES, "Existing provenance").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Existing provenance is invalid; refusing to overwrite history.") from error
        if (not isinstance(previous, dict)
                or previous.get("dataset_ref", SOURCE_REF) != SOURCE_REF
                or previous.get("version", SOURCE_VERSION) != SOURCE_VERSION
                or previous.get("source_csv_sha256", EXPECTED_CSV_SHA256) != EXPECTED_CSV_SHA256):
            raise ValueError("Existing provenance conflicts with the pinned source.")
    output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    if not archive_path.exists() and archive_bytes is not None:
        _write_new(archive_path, archive_bytes)
    if not had_csv:
        _write_new(csv_path, csv_bytes)
    if not had_metadata:
        _write_new(metadata_path, metadata_bytes)  # Retain exact original API bytes.
    if not had_provenance:
        if archive_bytes is None and archive_path.exists():
            archive_bytes = _file_bytes(archive_path, MAX_ARCHIVE_BYTES, "Existing ZIP")
        provenance = {
            "dataset_ref": SOURCE_REF, "version": SOURCE_VERSION, "dataset_url": DATASET_URL,
            "metadata_url": METADATA_URL, "download_url": DOWNLOAD_URL,
            "created_at_utc": timestamp, "verified_at_utc": timestamp,
            "retrieved_at_utc": timestamp if downloaded else None,
            "source_csv_sha256": EXPECTED_CSV_SHA256,
            "archive_sha256": _sha256(archive_bytes) if archive_bytes is not None else None,
            "metadata_sha256": _sha256(metadata_bytes), "archive_csv_member": CSV_FILENAME,
            "license": "other", "license_name": "Other (specified in description)",
            "license_statement": LICENSE_STATEMENT,
            "downloaded_urls_this_run": downloaded,
            "credentials_used": False, "training_started": False,
        }
        _write_new(provenance_path, _json_bytes(provenance))
    elif not had_metadata:
        # Preserve the original retrieval time/hash; describe this later repair separately.
        repair = {"dataset_ref": SOURCE_REF, "version": SOURCE_VERSION,
                  "retrieved_at_utc": timestamp, "metadata_url": METADATA_URL,
                  "metadata_sha256": _sha256(metadata_bytes), "license": "other",
                  "license_statement": LICENSE_STATEMENT,
                  "original_download_provenance_preserved": True,
                  "credentials_used": False, "training_started": False}
        _write_new(output / f"metadata_retrieval_{now:%Y%m%dT%H%M%S%fZ}.json", _json_bytes(repair))
    return {"status": "already_present" if had_csv and had_metadata and had_provenance else "prepared",
            "output_dir": str(output), "csv": str(csv_path), "source_csv_sha256": EXPECTED_CSV_SHA256,
            "metadata": str(metadata_path), "provenance": str(provenance_path),
            "downloaded_urls_this_run": downloaded,
            "credentials_used": False, "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(fetch_source(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
