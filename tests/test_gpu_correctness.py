"""Focused tests for the GPU gate; no GPU or PyTorch required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.gpu_correctness import compare_results, run_check, save_record


class TestGpuCorrectness(unittest.TestCase):
    def test_exact_match(self):
        result = compare_results([1.0, -2.0, 0.0], [1.0, -2.0, 0.0])
        self.assertEqual(result["result"], "PASS")
        self.assertEqual(result["max_absolute_error"], 0)

    def test_metrics(self):
        result = compare_results([2.0, 4.0], [2.5, 3.0])
        self.assertEqual(result["max_absolute_error"], 1.0)
        self.assertEqual(result["mean_absolute_error"], 0.75)
        self.assertEqual(result["max_relative_error"], 0.25)
        self.assertEqual(result["mean_relative_error"], 0.25)
        self.assertEqual(result["failed_elements"], 2)
        self.assertEqual(result["result"], "FAIL")

    def test_combined_tolerance(self):
        # Relative error can exceed rtol near zero while the gate still passes.
        result = compare_results([0.0001, 1000.0], [0.0002, 1000.5])
        self.assertEqual(result["result"], "PASS")
        self.assertGreater(result["max_relative_error"], 0.001)

    def test_zero_reference_and_boundary(self):
        result = compare_results([0.0, 0.0], [0.001, 0.0])
        self.assertEqual(result["result"], "PASS")
        self.assertAlmostEqual(result["max_relative_error"], 1e9)
        self.assertEqual(compare_results([0.0], [0.00101])["result"], "FAIL")

    def test_nonfinite_fails(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for ref, actual in (([value], [value]), ([0.0], [value]), ([value], [0.0])):
                result = compare_results(ref, actual)
                self.assertEqual(result["result"], "FAIL")
                json.dumps(result, allow_nan=False)

    def test_invalid_lengths(self):
        for ref, actual in (([], []), ([1.0], [1.0, 2.0])):
            with self.assertRaises(ValueError):
                compare_results(ref, actual)

    def test_invalid_tolerances(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                compare_results([1.0], [1.0], atol=value)
            with self.assertRaises(ValueError):
                compare_results([1.0], [1.0], rtol=value)

    def test_exclusive_json_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            save_record(path, {"result": "PASS"})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_record(path, {"result": "FAIL"})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(original)["result"], "PASS")

    def test_missing_torch_is_not_pass(self):
        with patch.dict("sys.modules", {"torch": None}):
            record = run_check()
        self.assertEqual(record["status"], "UNAVAILABLE")
        self.assertEqual(record["result"], "NOT_RUN")
        self.assertIsNone(record["max_absolute_error"])
        json.dumps(record, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
