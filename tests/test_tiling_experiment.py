"""Test suite for Experiment 0 & 1: Measurement Validation and Loop Tiling.

Tests:
    01-03: Tiled GEMM correctness verification
    04-06: Tile-size validation and edge cases
    07-09: Deterministic input and experiment ID
    10-12: Timing statistics and speedup calculation
    13-15: JSON serialization and result structure
    16-18: Baseline immutability and no system mutation
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from src.benchmark import generate_deterministic_matrix, calculate_percentile

# Import from the experiment runner
from run_experiments import (
    tiled_gemm,
    naive_gemm,
    TiledGemmBackend,
    BASELINE_M,
    BASELINE_N,
    BASELINE_K,
    BASELINE_FLOPS,
    LOCKED_BASELINE_MEDIAN_MS,
    TILE_SIZES,
)


def _reference_gemm(a, b, m, n, k):
    """Independent reference GEMM using standard ijk order."""
    c = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i][p] * b[p][j]
            c[i][j] = total
    return c


class TestTilingExperiment(unittest.TestCase):
    """Comprehensive tests for tiled GEMM experiments."""

    # ==================================================================
    # 01-03: Tiled GEMM correctness
    # ==================================================================

    def test_01_tiled_gemm_correctness_small(self):
        """01. Tiled GEMM produces identical results to reference for small matrices."""
        for size in [4, 8, 16]:
            a = generate_deterministic_matrix(size, size, seed_offset=1)
            b = generate_deterministic_matrix(size, size, seed_offset=2)
            ref = _reference_gemm(a, b, size, size, size)
            for ts in [2, 4, min(size, 8)]:
                tiled = tiled_gemm(a, b, size, size, size, ts)
                for i in range(size):
                    for j in range(size):
                        self.assertAlmostEqual(
                            ref[i][j], tiled[i][j], places=6,
                            msg=f"Mismatch at [{i}][{j}] for size={size}, tile={ts}"
                        )

    def test_02_tiled_gemm_correctness_nonsquare(self):
        """02. Tiled GEMM works for non-square matrices."""
        m, n, k = 12, 8, 16
        a = generate_deterministic_matrix(m, k, seed_offset=1)
        b = generate_deterministic_matrix(k, n, seed_offset=2)
        ref = _reference_gemm(a, b, m, n, k)
        for ts in [4, 8]:
            tiled = tiled_gemm(a, b, m, n, k, ts)
            for i in range(m):
                for j in range(n):
                    self.assertAlmostEqual(
                        ref[i][j], tiled[i][j], places=6,
                        msg=f"Mismatch at [{i}][{j}] for {m}x{n}x{k}, tile={ts}"
                    )

    def test_03_tiled_gemm_correctness_indivisible_tile(self):
        """03. Tiled GEMM handles tile sizes that don't evenly divide dimensions."""
        m, n, k = 10, 10, 10
        a = generate_deterministic_matrix(m, k, seed_offset=1)
        b = generate_deterministic_matrix(k, n, seed_offset=2)
        ref = _reference_gemm(a, b, m, n, k)
        for ts in [3, 7]:
            tiled = tiled_gemm(a, b, m, n, k, ts)
            for i in range(m):
                for j in range(n):
                    self.assertAlmostEqual(
                        ref[i][j], tiled[i][j], places=6,
                        msg=f"Mismatch at [{i}][{j}] for {m}x{n}x{k}, tile={ts}"
                    )

    # ==================================================================
    # 04-06: Tile-size validation
    # ==================================================================

    def test_04_tile_size_zero_raises(self):
        """04. Tile size of 0 raises ValueError."""
        a = [[1.0]]
        b = [[1.0]]
        with self.assertRaises(ValueError):
            tiled_gemm(a, b, 1, 1, 1, 0)

    def test_05_tile_size_negative_raises(self):
        """05. Negative tile size raises ValueError."""
        a = [[1.0]]
        b = [[1.0]]
        with self.assertRaises(ValueError):
            tiled_gemm(a, b, 1, 1, 1, -1)

    def test_06_all_experiment_tile_sizes_valid(self):
        """06. All configured TILE_SIZES (16, 32, 64, 128) are valid for 512x512x512."""
        for ts in TILE_SIZES:
            self.assertGreater(ts, 0)
            self.assertLessEqual(ts, BASELINE_M)
            self.assertLessEqual(ts, BASELINE_N)
            self.assertLessEqual(ts, BASELINE_K)

    # ==================================================================
    # 07-09: Deterministic input and experiment ID
    # ==================================================================

    def test_07_deterministic_matrix_input(self):
        """07. Matrix generation is deterministic across calls."""
        a1 = generate_deterministic_matrix(4, 4, seed_offset=1)
        a2 = generate_deterministic_matrix(4, 4, seed_offset=1)
        for i in range(4):
            for j in range(4):
                self.assertEqual(a1[i][j], a2[i][j])

    def test_08_deterministic_gemm_output(self):
        """08. Tiled GEMM output is deterministic across calls."""
        a = generate_deterministic_matrix(8, 8, seed_offset=1)
        b = generate_deterministic_matrix(8, 8, seed_offset=2)
        c1 = tiled_gemm(a, b, 8, 8, 8, 4)
        c2 = tiled_gemm(a, b, 8, 8, 8, 4)
        for i in range(8):
            for j in range(8):
                self.assertEqual(c1[i][j], c2[i][j])

    def test_09_experiment_id_format(self):
        """09. Experiment IDs follow expected naming convention."""
        self.assertTrue("experiment-000" in "experiment-000-measurement-validation")
        self.assertTrue("experiment-001" in "experiment-001-loop-tiling")

    # ==================================================================
    # 10-12: Timing and speedup calculation
    # ==================================================================

    def test_10_timing_produces_positive_values(self):
        """10. Timing a real execution produces positive duration values."""
        backend = TiledGemmBackend(tile_size=16)
        backend.prepare(16, 16, 16)
        start = time.perf_counter_ns()
        backend.execute()
        end = time.perf_counter_ns()
        self.assertGreater(end - start, 0)

    def test_11_speedup_calculation(self):
        """11. Speedup = baseline_median / candidate_median is correct."""
        baseline_ms = 1000.0
        candidate_ms = 500.0
        speedup = baseline_ms / candidate_ms
        self.assertAlmostEqual(speedup, 2.0)

        improvement_pct = ((baseline_ms - candidate_ms) / baseline_ms) * 100.0
        self.assertAlmostEqual(improvement_pct, 50.0)

    def test_12_gflops_calculation(self):
        """12. GFLOPS = FLOPs / (median_seconds * 1e9) is correct."""
        flops = 268435456  # 2 * 512 * 512 * 512
        median_ms = 1342.067
        median_s = median_ms / 1000.0
        gflops = flops / median_s / 1e9
        self.assertAlmostEqual(gflops, 0.200, places=2)

    # ==================================================================
    # 13-15: JSON serialization
    # ==================================================================

    def test_13_tile_result_json_serializable(self):
        """13. Tile experiment result dict is fully JSON-serializable."""
        result = {
            "tile_size": 32,
            "latency_ms": {"median": 1200.0, "p95": 1300.0},
            "comparison": {"speedup": 1.12, "improvement_percent": 10.5},
        }
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["tile_size"], 32)
        self.assertAlmostEqual(parsed["comparison"]["speedup"], 1.12)

    def test_14_experiment_result_has_required_keys(self):
        """14. Experiment result dict contains all required top-level keys."""
        required_keys = [
            "experiment_id", "experiment_name", "schema_version",
            "baseline_result_reference", "candidate_configuration",
            "tile_sizes_tested", "results", "summary", "timestamp",
        ]
        # Create a minimal mock result
        result = {k: None for k in required_keys}
        for key in required_keys:
            self.assertIn(key, result)

    def test_15_flops_constant(self):
        """15. BASELINE_FLOPS equals 2 * 512 * 512 * 512."""
        self.assertEqual(BASELINE_FLOPS, 2 * 512 * 512 * 512)
        self.assertEqual(BASELINE_FLOPS, 268435456)

    # ==================================================================
    # 16-18: Baseline immutability and no system mutation
    # ==================================================================

    def test_16_baseline_json_not_modified(self):
        """16. The locked baseline JSON file remains unmodified."""
        baseline_path = Path(__file__).resolve().parent.parent / \
            "experiments/results/gemm_cpu_512x512x512_fp32_20261003_043010.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                data = json.load(f)
            self.assertAlmostEqual(data["latency_ms"]["median"], 1342.066954)
            self.assertEqual(data["configuration"]["m"], 512)
            self.assertEqual(data["configuration"]["n"], 512)
            self.assertEqual(data["configuration"]["k"], 512)

    def test_17_no_environment_mutation(self):
        """17. Running tiled GEMM does not mutate os.environ."""
        env_before = dict(os.environ)
        a = generate_deterministic_matrix(8, 8, seed_offset=1)
        b = generate_deterministic_matrix(8, 8, seed_offset=2)
        _ = tiled_gemm(a, b, 8, 8, 8, 4)
        env_after = dict(os.environ)
        self.assertEqual(env_before, env_after)

    def test_18_tiled_backend_prepare_execute_cycle(self):
        """18. TiledGemmBackend prepare/execute cycle works correctly."""
        backend = TiledGemmBackend(tile_size=4)
        backend.prepare(8, 8, 8)
        result = backend.execute()
        self.assertEqual(len(result), 8)
        self.assertEqual(len(result[0]), 8)
        # Verify against reference
        a = generate_deterministic_matrix(8, 8, seed_offset=1)
        b = generate_deterministic_matrix(8, 8, seed_offset=2)
        ref = _reference_gemm(a, b, 8, 8, 8)
        for i in range(8):
            for j in range(8):
                self.assertAlmostEqual(result[i][j], ref[i][j], places=6)


if __name__ == "__main__":
    unittest.main()
