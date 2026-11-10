"""Pinned source download checks; all HTTP responses are mocked."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_weather_source as fetch


class Response(io.BytesIO):
    def __init__(self, data, declared_length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if declared_length is None else declared_length)}


class SourceDownloadTests(unittest.TestCase):
    def setUp(self):
        self.output_area = Path(__file__).resolve().parents[2] / "outputs"
        self.output_area.mkdir(exist_ok=True)
        self.directory = self.output_area / f"weather_source_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup_directory)
        self.output = self.directory / "v1"
        self.csv = b"Temperature,Weather Type\n10,Sunny\n"
        self.sha = hashlib.sha256(self.csv).hexdigest()
        self.patch_hash = patch.object(fetch, "EXPECTED_CSV_SHA256", self.sha)
        self.patch_hash.start()
        self.addCleanup(self.patch_hash.stop)
        # Whitespace is intentional: the original metadata bytes must survive.
        self.metadata = json.dumps({"info": {"ownerUser": "nikhil7280", "datasetSlug": "weather-type-classification",
                                   "licenses": [{"name": "other"}],
                                   "description": "Synthetic practice data. License: " + fetch.LICENSE_STATEMENT}}, indent=3).encode("utf-8")

    def cleanup_directory(self):
        target = self.directory.resolve()
        if target.parent != self.output_area.resolve() or not target.name.startswith("weather_source_test_"):
            raise RuntimeError("Refusing cleanup outside the test output area.")
        shutil.rmtree(target)

    def archive(self, entries=None):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries or [(fetch.CSV_FILENAME, self.csv)]:
                archive.writestr(name, content)
        return buffer.getvalue()

    def network(self, archive=None, metadata=None):
        archive = self.archive() if archive is None else archive
        metadata = self.metadata if metadata is None else metadata
        def reply(request, timeout):
            self.assertEqual(request.get_method(), "GET")
            self.assertGreater(timeout, 0)
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn("Cookie", request.headers)
            self.assertIn(request.full_url, (fetch.DOWNLOAD_URL, fetch.METADATA_URL))
            return Response(archive if request.full_url == fetch.DOWNLOAD_URL else metadata)
        return patch.object(fetch, "urlopen", side_effect=reply)

    def test_download_pins_version_hash_and_retains_raw_metadata(self):
        with self.network() as network:
            result = fetch.fetch_source(self.output)
        self.assertEqual(network.call_count, 2)
        self.assertEqual((self.output / fetch.CSV_FILENAME).read_bytes(), self.csv)
        self.assertEqual((self.output / "metadata.json").read_bytes(), self.metadata)
        provenance = json.loads((self.output / "download_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["version"], 1)
        self.assertEqual(provenance["dataset_ref"], fetch.SOURCE_REF)
        self.assertEqual(provenance["source_csv_sha256"], self.sha)
        self.assertEqual(provenance["license"], "other")
        self.assertFalse(provenance["credentials_used"])
        self.assertFalse(provenance["training_started"])
        self.assertIn("datasetVersionNumber=1", provenance["download_url"])
        self.assertEqual(result["status"], "prepared")

    def test_existing_correct_inputs_are_unchanged_and_never_refetched(self):
        with self.network():
            fetch.fetch_source(self.output)
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with patch.object(fetch, "urlopen", side_effect=AssertionError("Unexpected network")):
            result = fetch.fetch_source(self.output)
        after = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(result["status"], "already_present")
        self.assertEqual(result["downloaded_urls_this_run"], [])

    def test_existing_wrong_csv_is_rejected_without_network_or_overwrite(self):
        self.output.mkdir()
        target = self.output / fetch.CSV_FILENAME
        target.write_bytes(b"unrecognized CSV")
        with patch.object(fetch, "urlopen", side_effect=AssertionError("Unexpected network")), \
                self.assertRaisesRegex(ValueError, "Existing CSV SHA256 mismatch"):
            fetch.fetch_source(self.output)
        self.assertEqual(target.read_bytes(), b"unrecognized CSV")

    def test_wrong_download_hash_creates_no_source_files(self):
        wrong = self.archive([(fetch.CSV_FILENAME, b"wrong data")])
        with self.network(archive=wrong), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            fetch.fetch_source(self.output)
        self.assertFalse(self.output.exists())

    def test_exact_member_only_and_no_path_extraction(self):
        archive = self.archive([(fetch.CSV_FILENAME, self.csv), ("../must_not_exist.txt", b"unsafe"),
                                ("unrelated/data.csv", b"ignored")])
        with self.network(archive=archive):
            fetch.fetch_source(self.output)
        self.assertFalse((self.output.parent / "must_not_exist.txt").exists())
        self.assertFalse((self.output / "unrelated").exists())
        missing = self.archive([("folder/" + fetch.CSV_FILENAME, self.csv)])
        with self.network(archive=missing), self.assertRaisesRegex(ValueError, "exactly one member"):
            fetch.fetch_source(self.output.parent / "other")

    def test_duplicate_or_invalid_zip_members_are_rejected(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            duplicate = self.archive([(fetch.CSV_FILENAME, self.csv), (fetch.CSV_FILENAME, self.csv)])
        for archive in [duplicate, b"not a zip"]:
            with self.subTest(archive=archive[:20]), self.network(archive=archive), self.assertRaises(ValueError):
                fetch.fetch_source(self.output)

    def test_declared_stream_and_uncompressed_size_limits(self):
        with patch.object(fetch, "urlopen", return_value=Response(b"", fetch.MAX_ARCHIVE_BYTES + 1)), \
                self.assertRaisesRegex(ValueError, "size limit"):
            fetch.fetch_source(self.output)
        stream = Response(b"x" * 9, declared_length=1)
        with patch.object(fetch, "urlopen", return_value=stream), patch.object(fetch, "MAX_ARCHIVE_BYTES", 8), \
                self.assertRaisesRegex(ValueError, "size limit"):
            fetch.fetch_source(self.output)
        with self.network(), patch.object(fetch, "MAX_CSV_BYTES", 5), self.assertRaisesRegex(ValueError, "size limit"):
            fetch.fetch_source(self.output)

    def test_missing_metadata_is_repaired_without_rewriting_history(self):
        with self.network():
            fetch.fetch_source(self.output)
        original = (self.output / "download_provenance.json").read_bytes()
        (self.output / "metadata.json").unlink()
        with self.network() as network:
            fetch.fetch_source(self.output)
        self.assertEqual(network.call_count, 1)
        self.assertEqual(network.call_args.args[0].full_url, fetch.METADATA_URL)
        self.assertEqual((self.output / "download_provenance.json").read_bytes(), original)
        repairs = list(self.output.glob("metadata_retrieval_*.json"))
        self.assertEqual(len(repairs), 1)
        self.assertTrue(json.loads(repairs[0].read_text(encoding="utf-8"))["original_download_provenance_preserved"])

    def test_existing_archive_can_be_used_without_downloading_it_again(self):
        self.output.mkdir()
        archive = self.archive()
        (self.output / "source.zip").write_bytes(archive)
        (self.output / "metadata.json").write_bytes(self.metadata)
        with patch.object(fetch, "urlopen", side_effect=AssertionError("Unexpected network")):
            fetch.fetch_source(self.output)
        self.assertEqual((self.output / "source.zip").read_bytes(), archive)
        self.assertEqual((self.output / fetch.CSV_FILENAME).read_bytes(), self.csv)

    def test_wrong_metadata_license_identity_and_invalid_timeout_are_rejected(self):
        for info in [{"ownerUser": "someone_else"}, {"ownerUser": "nikhil7280", "datasetSlug": "weather-type-classification",
                     "licenses": [{"name": "CC0"}], "description": fetch.LICENSE_STATEMENT}]:
            with self.network(metadata=json.dumps({"info": info}).encode()), self.assertRaisesRegex(ValueError, "Metadata|license"):
                fetch.fetch_source(self.output)
        with patch.object(fetch, "urlopen", side_effect=AssertionError("Unexpected network")):
            for timeout in (0, -1, True, float("nan"), "30"):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    fetch.fetch_source(self.output, timeout=timeout)


if __name__ == "__main__":
    unittest.main()
