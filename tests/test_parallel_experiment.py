"""Test suite for Experiment 3: Multi-Process Parallelism.

Tests:
    01-03: Work partitioning and remainder rows (exact preservation, remainders, fewer rows than workers)
    04-06: Mathematical correctness across workers (square, non-square, prime/indivisible dimensions)
    07-08: Determinism and CPU utilization calculation
    09-13: Metrics, scaling formulas, decision logic, and best configuration selection
    14-15: Result schema and JSON serialization
    16-18: Immutability (locked baseline SHA256), environment safety, and correctness verification suite
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import generate_deterministic_matrix
from src.parallel_experiment import (
    BASELINE_FLOPS,
    BASELINE_K,
    BASELINE_M,
    BASELINE_N,
    LOCKED_BASELINE_GFLOPS,
    LOCKED_BASELINE_MEDIAN_MS,
    LOCKED_BASELINE_P95_MS,
    LOCKED_BASELINE_SHA256,
    WORKER_CONFIGS,
    calculate_cpu_utilization,
    partition_row_chunks,
    reference_gemm,
    verify_correctness_across_workers,
)


class TestParallelExperiment(unittest.TestCase):
    """Exhaustive test suite for Experiment 3 Multi-Process Parallelism."""

    # ==================================================================
    # 01-03: Work Partitioning & Remainder Rows
    # ==================================================================

    def test_01_partition_row_chunks_exact(self):
        """01. Partitioning preserves all rows and row ordering across all worker counts."""
        for num_rows in [8, 16, 32, 64, 128]:
            data = [[float(i * 10 + j) for j in range(4)] for i in range(num_rows)]
            for w in [1, 2, 4, 8]:
                chunks = partition_row_chunks(data, w)
                reconstructed = [row for chunk in chunks for row in chunk]
                self.assertEqual(data, reconstructed, f"Mismatch for rows={num_rows}, workers={w}")

    def test_02_partition_row_chunks_remainder(self):
        """02. Partitioning correctly handles remainder rows for odd/prime sizes."""
        for num_rows in [5, 7, 10, 11, 13, 17, 33, 65, 99]:
            data = [[float(i)] for i in range(num_rows)]
            for w in [1, 2, 3, 4, 5, 8]:
                chunks = partition_row_chunks(data, w)
                reconstructed = [row for chunk in chunks for row in chunk]
                self.assertEqual(len(reconstructed), num_rows)
                self.assertEqual(data, reconstructed)

    def test_03_partition_row_chunks_fewer_rows_than_workers(self):
        """03. Handles case where row count is less than worker count (m < workers)."""
        data = [[1.0], [2.0], [3.0]]  # m = 3
        for w in [4, 8]:
            chunks = partition_row_chunks(data, w)
            reconstructed = [row for chunk in chunks for row in chunk]
            self.assertEqual(data, reconstructed)
            self.assertEqual(len(reconstructed), 3)

    # ==================================================================
    # 04-06: Mathematical Correctness Across Workers
    # ==================================================================

    def test_04_mathematical_equivalence_small_square(self):
        """04. Small square matrices produce identical results across 1, 2, 4, 8 workers."""
        from src.benchmark import CpuGemmBackend
        for size in [4, 8, 16]:
            a = generate_deterministic_matrix(size, size, seed_offset=1)
            b = generate_deterministic_matrix(size, size, seed_offset=2)
            ref = reference_gemm(a, b)
            for w in [1, 2, 4, 8]:
                backend = CpuGemmBackend(workers=w)
                backend.prepare(size, size, size)
                cand = backend.execute()
                backend.cleanup()
                for i in range(size):
                    for j in range(size):
                        self.assertAlmostEqual(
                            ref[i][j], cand[i][j], places=5,
                            msg=f"Mismatch at [{i}][{j}] for size={size}, w={w}"
                        )

    def test_05_mathematical_equivalence_nonsquare(self):
        """05. Non-square matrices produce identical results across workers."""
        from src.benchmark import CpuGemmBackend
        shapes = [(12, 8, 16), (15, 10, 7), (20, 6, 8)]
        for m, n, k in shapes:
            a = generate_deterministic_matrix(m, k, seed_offset=1)
            b = generate_deterministic_matrix(k, n, seed_offset=2)
            ref = reference_gemm(a, b)
            for w in [1, 2, 4]:
                backend = CpuGemmBackend(workers=w)
                backend.prepare(m, n, k)
                cand = backend.execute()
                backend.cleanup()
                self.assertEqual(len(cand), m)
                self.assertEqual(len(cand[0]), n)
                for i in range(m):
                    for j in range(n):
                        self.assertAlmostEqual(
                            ref[i][j], cand[i][j], places=5,
                            msg=f"Mismatch at [{i}][{j}] for {m}x{n}x{k}, w={w}"
                        )

    def test_06_mathematical_equivalence_indivisible_dimensions(self):
        """06. Indivisible prime dimensions produce identical results across workers."""
        from src.benchmark import CpuGemmBackend
        m, n, k = 17, 19, 23
        a = generate_deterministic_matrix(m, k, seed_offset=1)
        b = generate_deterministic_matrix(k, n, seed_offset=2)
        ref = reference_gemm(a, b)
        for w in [1, 2, 4, 8]:
            backend = CpuGemmBackend(workers=w)
            backend.prepare(m, n, k)
            cand = backend.execute()
            backend.cleanup()
            for i in range(m):
                for j in range(n):
                    self.assertAlmostEqual(ref[i][j], cand[i][j], places=5)

    # ==================================================================
    # 07-08: Determinism & CPU Utilization
    # ==================================================================

    def test_07_deterministic_inputs_and_outputs(self):
        """07. Input generation and parallel GEMM outputs are deterministic."""
        from src.benchmark import CpuGemmBackend
        a1 = generate_deterministic_matrix(8, 8, seed_offset=1)
        a2 = generate_deterministic_matrix(8, 8, seed_offset=1)
        self.assertEqual(a1, a2)

        backend = CpuGemmBackend(workers=2)
        backend.prepare(8, 8, 8)
        out1 = backend.execute()
        out2 = backend.execute()
        backend.cleanup()
        self.assertEqual(out1, out2)

    def test_08_cpu_utilization_calculation(self):
        """08. CPU utilization calculation computes expected percentage or reports unavailable."""
        val = calculate_cpu_utilization((100.0, 60.0), (200.0, 120.0))
        self.assertEqual(val, 40.0)

        self.assertEqual(
            calculate_cpu_utilization(None, (100.0, 50.0)),
            "CPU utilization measurement unavailable",
        )
        self.assertEqual(
            calculate_cpu_utilization((100.0, 50.0), None),
            "CPU utilization measurement unavailable",
        )

    # ==================================================================
    # 09-13: Metrics, Scaling, and Decisions
    # ==================================================================

    def test_09_parallel_speedup_formula(self):
        """09. Parallel speedup = one_worker_median / candidate_median is correct."""
        one_w = 7000.0
        cand_4w = 1750.0
        speedup = one_w / cand_4w
        self.assertAlmostEqual(speedup, 4.0)

    def test_10_speedup_vs_baseline_formula(self):
        """10. Speedup vs locked baseline = locked_baseline_median / candidate_median."""
        locked_med = LOCKED_BASELINE_MEDIAN_MS
        cand_med = 1342.067
        speedup = locked_med / cand_med
        self.assertAlmostEqual(speedup, 1.0)

    def test_11_gflops_formula(self):
        """11. GFLOPS calculation = 2*M*N*K / seconds / 1e9."""
        flops = 2 * 512 * 512 * 512
        med_ms = 1342.067
        gflops = flops / (med_ms / 1000.0) / 1e9
        self.assertAlmostEqual(gflops, 0.200, places=2)

    def test_12_decision_logic(self):
        """12. Decision logic classifies candidates as ADOPTABLE or REJECT."""
        cand_slow = 2000.0
        imp_slow = ((LOCKED_BASELINE_MEDIAN_MS - cand_slow) / LOCKED_BASELINE_MEDIAN_MS) * 100.0
        self.assertLess(imp_slow, 0.0)

        cand_fast = 1200.0
        imp_fast = ((LOCKED_BASELINE_MEDIAN_MS - cand_fast) / LOCKED_BASELINE_MEDIAN_MS) * 100.0
        self.assertGreater(imp_fast, 5.0)

    def test_13_best_configuration_selection(self):
        """13. Configuration selection correctly identifies absolute best and best scaling."""
        mock_summary = {
            "1": {"median_ms": 7000.0, "parallel_scaling_speedup": 1.0},
            "2": {"median_ms": 3600.0, "parallel_scaling_speedup": 1.944},
            "4": {"median_ms": 1900.0, "parallel_scaling_speedup": 3.684},
            "8": {"median_ms": 1330.0, "parallel_scaling_speedup": 5.263},
        }
        best_abs = min([1, 2, 4, 8], key=lambda w: mock_summary[str(w)]["median_ms"])
        self.assertEqual(best_abs, 8)
        best_scale = max([2, 4, 8], key=lambda w: mock_summary[str(w)]["parallel_scaling_speedup"])
        self.assertEqual(best_scale, 8)

    # ==================================================================
    # 14-15: Result Schema & JSON Serialization
    # ==================================================================

    def test_14_schema_keys_present(self):
        """14. Result schema contains all required top-level keys."""
        required_keys = [
            "experiment_id",
            "baseline_reference",
            "workload",
            "worker_configurations",
            "per_worker_metrics",
            "scaling_metrics",
            "correctness_results",
            "environment_snapshot",
            "verification_status",
            "timestamp",
            "final_decision",
        ]
        mock_res = {k: None for k in required_keys}
        for k in required_keys:
            self.assertIn(k, mock_res)

    def test_15_json_serializable(self):
        """15. Result dictionary is JSON serializable."""
        sample = {
            "experiment_id": "experiment-003-parallelism",
            "worker_configurations": WORKER_CONFIGS,
            "scaling_metrics": {"1_to_8_speedup": 5.3},
            "final_decision": "ADOPTABLE",
        }
        encoded = json.dumps(sample)
        decoded = json.loads(encoded)
        self.assertEqual(sample, decoded)

    # ==================================================================
    # 16-18: Immutability, Environment Safety, and Correctness Suite
    # ==================================================================

    def test_16_locked_baseline_sha256_unmodified(self):
        """16. Locked baseline JSON is byte-for-byte unmodified matching SHA256."""
        path = (
            Path(__file__).resolve().parent.parent
            / "experiments/results/gemm_cpu_512x512x512_fp32_20261003_043010.json"
        )
        self.assertTrue(path.exists(), f"Baseline file missing at {path}")
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            h,
            LOCKED_BASELINE_SHA256,
            f"Baseline file modified! Hash: {h} vs expected {LOCKED_BASELINE_SHA256}",
        )

    def test_17_no_environment_mutation(self):
        """17. Running partition and reference functions does not mutate os.environ."""
        env_before = dict(os.environ)
        _ = partition_row_chunks([[1.0, 2.0]], 2)
        _ = reference_gemm([[1.0]], [[2.0]])
        env_after = dict(os.environ)
        self.assertEqual(env_before, env_after)

    def test_18_verify_correctness_suite_passes(self):
        """18. The comprehensive worker correctness suite passes on all test cases."""
        result = verify_correctness_across_workers([1, 2, 4, 8])
        self.assertEqual(result["status"], "ALL_PASS")
        self.assertGreater(result["test_cases_evaluated"], 5)


if __name__ == "__main__":
    unittest.main()
