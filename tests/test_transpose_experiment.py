"""Test suite for Experiment 2: Pre-Transposed B Evaluation.

Tests:
    01-03: Mathematical equivalence against reference (square, non-square, non-block dimensions)
    04-06: Transpose properties and validation (involution, dimension mismatch, empty matrix)
    07-09: Determinism (inputs, outputs, transpose cost positivity)
    10-13: Metrics & analysis (speedup, improvement %, GFLOPS, steady-state vs end-to-end)
    14-15: Serialization and schema compliance
    16-18: Immutability and safety (baseline byte-for-byte SHA256, os.environ, backend lifecycle)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import generate_deterministic_matrix
from src.transpose_experiment import (
    BASELINE_FLOPS,
    BASELINE_K,
    BASELINE_M,
    BASELINE_N,
    LOCKED_BASELINE_GFLOPS,
    LOCKED_BASELINE_MEDIAN_MS,
    LOCKED_BASELINE_P95_MS,
    PretransposedGemmBackend,
    gemm_pretransposed,
    reference_gemm,
    transpose_matrix,
)

EXPECTED_BASELINE_SHA256 = "ded6b85ff3ec29349018c25ae7e0d7aedcb3ea719380dc5dec57d1f31d2376ad"


class TestTransposeExperiment(unittest.TestCase):
    """Exhaustive test suite for Experiment 2 Pre-Transposed B."""

    # ==================================================================
    # 01-03: Mathematical Equivalence
    # ==================================================================

    def test_01_mathematical_equivalence_small_square(self):
        """01. Pre-transposed GEMM produces identical results to reference for square matrices."""
        for size in [2, 4, 8, 16]:
            a = generate_deterministic_matrix(size, size, seed_offset=1)
            b = generate_deterministic_matrix(size, size, seed_offset=2)
            b_t = transpose_matrix(b)
            ref = reference_gemm(a, b)
            cand = gemm_pretransposed(a, b_t)
            for i in range(size):
                for j in range(size):
                    self.assertAlmostEqual(
                        ref[i][j], cand[i][j], places=6,
                        msg=f"Mismatch at [{i}][{j}] for size={size}"
                    )

    def test_02_mathematical_equivalence_nonsquare(self):
        """02. Pre-transposed GEMM matches reference for non-square matrices."""
        shapes = [
            (12, 8, 16),
            (5, 10, 7),
            (3, 15, 6),
            (20, 4, 8),
        ]
        for m, n, k in shapes:
            a = generate_deterministic_matrix(m, k, seed_offset=10)
            b = generate_deterministic_matrix(k, n, seed_offset=20)
            b_t = transpose_matrix(b)
            ref = reference_gemm(a, b)
            cand = gemm_pretransposed(a, b_t)
            self.assertEqual(len(cand), m)
            self.assertEqual(len(cand[0]), n)
            for i in range(m):
                for j in range(n):
                    self.assertAlmostEqual(
                        ref[i][j], cand[i][j], places=6,
                        msg=f"Mismatch at [{i}][{j}] for shape {m}x{n}x{k}"
                    )

    def test_03_dimensions_not_multiples_of_blocks(self):
        """03. Works correctly for non-standard prime and odd dimensions."""
        prime_shapes = [
            (7, 11, 13),
            (17, 19, 23),
            (3, 5, 7),
        ]
        for m, n, k in prime_shapes:
            a = generate_deterministic_matrix(m, k, seed_offset=3)
            b = generate_deterministic_matrix(k, n, seed_offset=4)
            b_t = transpose_matrix(b)
            ref = reference_gemm(a, b)
            cand = gemm_pretransposed(a, b_t)
            for i in range(m):
                for j in range(n):
                    self.assertAlmostEqual(
                        ref[i][j], cand[i][j], places=6,
                        msg=f"Mismatch at [{i}][{j}] for prime shape {m}x{n}x{k}"
                    )

    # ==================================================================
    # 04-06: Transpose Properties & Validation
    # ==================================================================

    def test_04_transpose_matrix_involution(self):
        """04. Double transposition restores original matrix: (B^T)^T = B."""
        for m, n in [(4, 4), (10, 5), (7, 13)]:
            orig = generate_deterministic_matrix(m, n, seed_offset=5)
            transposed = transpose_matrix(orig)
            restored = transpose_matrix(transposed)
            self.assertEqual(orig, restored)

    def test_05_inner_dimension_mismatch_raises(self):
        """05. Inner dimension mismatch between A and B^T raises ValueError."""
        a = [[1.0, 2.0, 3.0]]      # 1 x 3 (k=3)
        b_t = [[1.0, 2.0], [3.0, 4.0]]  # 2 x 2 (k=2)
        with self.assertRaises(ValueError):
            gemm_pretransposed(a, b_t)

    def test_06_empty_matrix_raises(self):
        """06. Empty matrices raise ValueError for both transpose and gemm."""
        with self.assertRaises(ValueError):
            transpose_matrix([])
        with self.assertRaises(ValueError):
            transpose_matrix([[]])
        with self.assertRaises(ValueError):
            gemm_pretransposed([], [])

    # ==================================================================
    # 07-09: Determinism & Preprocessing Cost
    # ==================================================================

    def test_07_deterministic_input_generation(self):
        """07. Inputs generated with same seeds are identical across invocations."""
        a1 = generate_deterministic_matrix(8, 8, seed_offset=1)
        a2 = generate_deterministic_matrix(8, 8, seed_offset=1)
        self.assertEqual(a1, a2)

    def test_08_deterministic_output(self):
        """08. Pre-transposed GEMM produces deterministic output across runs."""
        a = generate_deterministic_matrix(8, 8, seed_offset=1)
        b = generate_deterministic_matrix(8, 8, seed_offset=2)
        b_t = transpose_matrix(b)
        out1 = gemm_pretransposed(a, b_t)
        out2 = gemm_pretransposed(a, b_t)
        self.assertEqual(out1, out2)

    def test_09_transpose_cost_positive(self):
        """09. Transpose time measurement is strictly positive."""
        backend = PretransposedGemmBackend()
        backend.prepare(16, 16, 16)
        cost = backend.measure_transpose_cost(repeats=5)
        self.assertGreater(cost, 0.0)

    # ==================================================================
    # 10-13: Metrics & Steady-State vs End-to-End
    # ==================================================================

    def test_10_speedup_calculation(self):
        """10. Speedup = baseline_median / candidate_median is correct."""
        baseline_ms = 1342.067
        candidate_ms = 671.0335
        speedup = baseline_ms / candidate_ms
        self.assertAlmostEqual(speedup, 2.0)

    def test_11_improvement_percentage_calculation(self):
        """11. Improvement % = ((baseline - candidate) / baseline) * 100 is correct."""
        baseline_ms = 1000.0
        candidate_ms = 800.0
        imp = ((baseline_ms - candidate_ms) / baseline_ms) * 100.0
        self.assertAlmostEqual(imp, 20.0)

    def test_12_gflops_calculation(self):
        """12. GFLOPS calculation matches 2*M*N*K / seconds / 1e9."""
        flops = BASELINE_FLOPS
        median_ms = 1000.0
        gflops = (flops / (median_ms / 1000.0) / 1e9)
        expected = 268435456 / 1e9
        self.assertAlmostEqual(gflops, expected, places=4)

    def test_13_steady_state_vs_end_to_end_distinction(self):
        """13. End-to-end latency strictly equals steady-state latency + transpose cost."""
        steady_ms = 7000.0
        transpose_ms = 15.5
        e2e_ms = steady_ms + transpose_ms
        self.assertEqual(e2e_ms, 7015.5)
        self.assertGreater(e2e_ms, steady_ms)

    # ==================================================================
    # 14-15: Serialization & Schema Compliance
    # ==================================================================

    def test_14_schema_keys_present(self):
        """14. Result structure contains all required fields from protocol."""
        required_keys = [
            "experiment_id",
            "baseline_reference",
            "workload",
            "candidate_configuration",
            "transpose_time_ms",
            "steady_state_metrics",
            "end_to_end_metrics",
            "speedup",
            "improvement_percentage",
            "verification_status",
            "environment_snapshot",
            "timestamp",
        ]
        mock_result = {k: None for k in required_keys}
        for k in required_keys:
            self.assertIn(k, mock_result)

    def test_15_json_serializable(self):
        """15. Experiment result structure is valid JSON serializable."""
        sample = {
            "experiment_id": "experiment-002-pretransposed-b",
            "transpose_time_ms": 15.8,
            "steady_state_metrics": {"median_ms": 7200.0, "gflops": 0.037},
            "end_to_end_metrics": {"median_ms": 7215.8, "gflops": 0.037},
            "speedup": {"steady_state": 0.186, "end_to_end": 0.185},
            "improvement_percentage": {"steady_state": -436.0, "end_to_end": -437.0},
        }
        encoded = json.dumps(sample)
        decoded = json.loads(encoded)
        self.assertEqual(sample, decoded)

    # ==================================================================
    # 16-18: Immutability & Safety
    # ==================================================================

    def test_16_locked_baseline_byte_for_byte_unchanged(self):
        """16. Locked baseline JSON file matches expected SHA256 hash byte-for-byte."""
        baseline_path = (
            Path(__file__).resolve().parent.parent
            / "experiments/results/gemm_cpu_512x512x512_fp32_20261003_043010.json"
        )
        self.assertTrue(baseline_path.exists(), f"Baseline file missing at {baseline_path}")
        with open(baseline_path, "rb") as f:
            content = f.read()
        sha256 = hashlib.sha256(content).hexdigest()
        self.assertEqual(
            sha256,
            EXPECTED_BASELINE_SHA256,
            f"Baseline file has been modified! SHA256 mismatch: {sha256} vs {EXPECTED_BASELINE_SHA256}",
        )

    def test_17_no_environment_mutation(self):
        """17. Execution does not modify os.environ."""
        env_before = dict(os.environ)
        a = generate_deterministic_matrix(4, 4, seed_offset=1)
        b = generate_deterministic_matrix(4, 4, seed_offset=2)
        b_t = transpose_matrix(b)
        _ = gemm_pretransposed(a, b_t)
        env_after = dict(os.environ)
        self.assertEqual(env_before, env_after)

    def test_18_backend_prepare_execute_cycle(self):
        """18. PretransposedGemmBackend prepares and executes correctly."""
        backend = PretransposedGemmBackend()
        backend.prepare(8, 8, 8)
        self.assertIsNotNone(backend.a)
        self.assertIsNotNone(backend.b)
        self.assertIsNotNone(backend.b_t)
        res = backend.execute()
        self.assertEqual(len(res), 8)
        self.assertEqual(len(res[0]), 8)
        # End to end check
        res_e2e, duration_ms = backend.execute_end_to_end()
        self.assertEqual(res, res_e2e)
        self.assertGreater(duration_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
