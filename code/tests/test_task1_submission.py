"""CSV contract checks using independent expected records and byte strings."""

import csv
import hashlib
import io
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task1.submission import validate_submission, validate_submission_file, write_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        # Ordinary mkdir avoids tempfile's restrictive Windows directory mode,
        # which can make directories inaccessible in a managed workspace.
        self.output_area = Path(__file__).resolve().parents[2] / "outputs"
        self.output_area.mkdir(exist_ok=True)
        self.directory = self.output_area / f"submission_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(self.cleanup_directory)
        self.sample = pd.DataFrame({"id": [31, 12, 7], "label": ["Rainy"] * 3})

    def cleanup_directory(self):
        target = self.directory.resolve()
        if target.parent != self.output_area.resolve() or not target.name.startswith("submission_test_"):
            raise RuntimeError("Refusing to remove a test directory outside its output area.")
        shutil.rmtree(target)

    def output(self, filename="submission.csv"):
        return self.directory / filename

    def test_reorders_predictions_by_sample_and_writes_exact_bytes(self):
        path = self.output()
        metadata = write_submission([7, 31, 12], [4, 1, 2], self.sample, path, "integer")
        expected = b"id,label\n31,1\n12,2\n7,4\n"
        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(metadata["csv_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(metadata["rows"], 3)
        self.assertEqual(metadata["class_counts"], {"1": 1, "2": 1, "3": 0, "4": 1})
        self.assertEqual(metadata, validate_submission_file(path, self.sample, "integer"))

    def test_name_and_integer_exports_share_independent_canonical_hash(self):
        integer = write_submission([31, 12, 7], [1, 3, 4], self.sample, self.output("integer.csv"), "integer")
        name = write_submission([7, 12, 31], [4, 3, 1], self.sample, self.output("name.csv"), "name")
        canonical = b'task1-predictions-v1\n["integer","31",1]\n["integer","12",3]\n["integer","7",4]\n'
        self.assertEqual(integer["canonical_prediction_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(integer["canonical_prediction_sha256"], name["canonical_prediction_sha256"])
        self.assertNotEqual(integer["csv_sha256"], name["csv_sha256"])
        self.assertEqual(self.output("name.csv").read_bytes(), b"id,label\n31,Sunny\n12,Rainy\n7,Snowy\n")

    def test_uint64_and_arbitrary_precision_integer_ids_survive_roundtrip(self):
        for ids in ([2**53 + 1, 2**53 + 3, 2**64 - 1], [10**60 + 1, 10**60 + 2]):
            with self.subTest(ids=ids):
                sample = pd.DataFrame({"id": pd.Series(ids, dtype=object)})
                path = self.output(f"large_{len(ids)}.csv")
                metadata = write_submission(list(reversed(ids)), list(range(len(ids), 0, -1)), sample, path, "integer")
                records = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
                self.assertEqual([row[0] for row in records[1:]], [str(identifier) for identifier in ids])
                self.assertEqual(metadata, validate_submission_file(path, sample, "integer"))
        uint_ids = pd.Series([2**63 + 1, 2**64 - 1], dtype="uint64")
        sample = pd.DataFrame({"id": uint_ids})
        write_submission(uint_ids, [1, 2], sample, self.output("uint.csv"), "integer")

    def test_string_ids_preserve_leading_zeroes_na_literals_and_csv_quoting(self):
        ids = ["001", "NA", "comma,id", "line\nbreak", "weather-雪"]
        sample = pd.DataFrame({"id": ids})
        path = self.output()
        metadata = write_submission(ids[::-1], [1, 4, 3, 2, 1], sample, path, "name")
        records = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
        self.assertEqual([row[0] for row in records[1:]], ids)
        self.assertEqual(metadata["id_type"], "string")
        self.assertFalse(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual(metadata, validate_submission_file(path, sample, "name"))

    def test_does_not_mutate_sample_or_frame(self):
        frame = pd.DataFrame({"id": pd.Series([31, 12, 7], dtype="uint64"), "label": [1, 2, 3]})
        original_frame = frame.copy(deep=True)
        original_sample = self.sample.copy(deep=True)
        validate_submission(frame, self.sample, "integer")
        write_submission(frame["id"], frame["label"], self.sample, self.output(), "integer")
        pd.testing.assert_frame_equal(frame, original_frame)
        pd.testing.assert_frame_equal(self.sample, original_sample)

    def test_validate_rejects_wrong_order_instead_of_silently_reordering(self):
        frame = pd.DataFrame({"id": [7, 12, 31], "label": [1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "order"):
            validate_submission(frame, self.sample, "integer")

    def test_invalid_id_membership_duplicates_missing_and_extra_rows(self):
        for ids, labels in [([31, 31, 7], [1, 2, 3]), ([31, 12], [1, 2]),
                            ([31, 12, 7, 90], [1, 2, 3, 4]), ([31, 12, 99], [1, 2, 3]),
                            ([31, None, 7], [1, 2, 3]), ([31, 12, 7], [1, 2])]:
            with self.subTest(ids=ids, labels=labels), self.assertRaises(ValueError):
                write_submission(ids, labels, self.sample, self.output(), "integer")
        self.assertFalse(self.output().exists())

    def test_rejects_float_boolean_mixed_and_different_id_types(self):
        for ids in ([31.0, 12.0, 7.0], [True, 12, 7], [31, "12", 7], ["31", "12", "7"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                write_submission(ids, [1, 2, 3], self.sample, self.output(), "integer")

    def test_invalid_sample_ids(self):
        for sample in (pd.DataFrame({"id": [31, 31, 7]}), pd.DataFrame({"id": []}),
                       pd.DataFrame({"id": [31, np.nan, 7]}), pd.DataFrame({"not_id": [1, 2, 3]}),
                       pd.DataFrame({"id": ["", "a", "b"]})):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                write_submission([31, 12, 7], [1, 2, 3], sample, self.output(), "integer")

    def test_integer_labels_are_strict_and_non_missing(self):
        for invalid in (0, 5, -1, 1.0, "1", "Sunny", True, np.bool_(True), None, np.nan, pd.NA):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                write_submission([31, 12, 7], [1, invalid, 3], self.sample, self.output(), "integer")
        self.assertFalse(self.output().exists())

    def test_validate_name_labels_and_rejects_unknown_or_missing_names(self):
        frame = pd.DataFrame({"id": [31, 12, 7], "label": ["Sunny", "Cloudy", "Rainy"]})
        self.assertEqual(validate_submission(frame, self.sample, "name")["class_counts"]["2"], 1)
        for invalid in ("sunny", "Sunny ", "Foggy", "", None, 1, pd.NA):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_submission(frame.assign(label=["Sunny", invalid, "Rainy"]), self.sample, "name")

    def test_refuses_unconfirmed_unknown_and_non_string_formats(self):
        frame = pd.DataFrame({"id": [31, 12, 7], "label": [1, 2, 3]})
        for label_format in ("UNCONFIRMED", "auto", "Integer", "", None, 1):
            with self.subTest(label_format=label_format):
                with self.assertRaises(ValueError):
                    validate_submission(frame, self.sample, label_format)
                with self.assertRaises(ValueError):
                    write_submission([31, 12, 7], [1, 2, 3], self.sample, self.output(), label_format)
                with self.assertRaises(ValueError):
                    validate_submission_file(self.output("missing.csv"), self.sample, label_format)

    def test_schema_must_be_exact_and_in_order(self):
        frame = pd.DataFrame({"id": [31, 12, 7], "label": [1, 2, 3]})
        for bad in (frame[["label", "id"]], frame.assign(index=[0, 1, 2]), frame.drop(columns="label"),
                    pd.DataFrame([[31, 1, 1]], columns=["id", "label", "label"])):
            with self.subTest(columns=list(bad.columns)), self.assertRaises(ValueError):
                validate_submission(bad, self.sample, "integer")

    def test_two_dimensional_inputs_are_rejected(self):
        for ids, labels in ((np.array([[31], [12], [7]]), [1, 2, 3]),
                            ([31, 12, 7], np.array([[1], [2], [3]]))):
            with self.subTest(), self.assertRaises(ValueError):
                write_submission(ids, labels, self.sample, self.output(), "integer")

    def test_invalid_input_does_not_create_parent_directories(self):
        path = self.directory / "not_created" / "submission.csv"
        with self.assertRaises(ValueError):
            write_submission([31, 12, 7], [1, 9, 3], self.sample, path, "integer")
        self.assertFalse(path.parent.exists())

    def test_refuses_to_overwrite_existing_file(self):
        path = self.output()
        path.write_bytes(b"original content")
        with self.assertRaises(FileExistsError):
            write_submission([31, 12, 7], [1, 2, 3], self.sample, path, "integer")
        self.assertEqual(path.read_bytes(), b"original content")

    def test_file_validator_rejects_malformed_schema_ids_and_labels(self):
        cases = [b"id,label,index\n31,1,0\n12,2,1\n7,3,2\n",
                 b"label,id\n1,31\n2,12\n3,7\n", b"id,label\n31,1\n12,2\n",
                 b"id,label\n31,1\n31,2\n7,3\n", b"id,label\n31,1\n12,2\n99,3\n",
                 b"id,label\n7,3\n12,2\n31,1\n", b"id,label\n31,1.0\n12,2\n7,3\n",
                 b"id,label\n31,1\n12,\n7,3\n", b"id,label\n31,1\n12,2\n7,3\n\n",
                 b"id,label\n31.0,1\n12,2\n7,3\n", b"id,label\n031,1\n12,2\n7,3\n",
                 b'id,label\n"unterminated,1\n12,2\n7,3\n', b"id,label\n31,1\n12,2\n7,\xff\n"]
        path = self.output()
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                path.write_bytes(content)
                validate_submission_file(path, self.sample, "integer")

    def test_rejects_bom(self):
        path = self.output()
        path.write_bytes(b"\xef\xbb\xbfid,label\n31,1\n12,2\n7,3\n")
        with self.assertRaisesRegex(ValueError, "BOM"):
            validate_submission_file(path, self.sample, "integer")

    def test_crlf_has_different_byte_hash_but_same_semantic_hash(self):
        path = self.output()
        lf = write_submission([31, 12, 7], [1, 2, 4], self.sample, path, "integer")
        crlf_path = self.output("crlf.csv")
        crlf_path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        crlf = validate_submission_file(crlf_path, self.sample, "integer")
        self.assertNotEqual(lf["csv_sha256"], crlf["csv_sha256"])
        self.assertEqual(lf["canonical_prediction_sha256"], crlf["canonical_prediction_sha256"])

    def test_canonical_hash_changes_with_id_type_or_prediction(self):
        first = write_submission([31, 12, 7], [1, 2, 3], self.sample, self.output("first.csv"), "integer")
        changed = write_submission([31, 12, 7], [1, 2, 4], self.sample, self.output("changed.csv"), "integer")
        string_sample = pd.DataFrame({"id": ["31", "12", "7"]})
        strings = write_submission(["31", "12", "7"], [1, 2, 3], string_sample, self.output("strings.csv"), "integer")
        self.assertNotEqual(first["canonical_prediction_sha256"], changed["canonical_prediction_sha256"])
        self.assertNotEqual(first["canonical_prediction_sha256"], strings["canonical_prediction_sha256"])


if __name__ == "__main__":
    unittest.main()
