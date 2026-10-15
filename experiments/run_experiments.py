#!/usr/bin/env python3
"""Experiment Runner for ROCm Performance Guard.

Executes:
    Experiment 0 — Measurement Validation (3 independent baseline runs)
    Experiment 1 — Loop Tiling (tile sizes 16, 32, 64, 128)

Does NOT modify the locked baseline JSON.
Does NOT install packages or modify the system.
"""

from __future__ import annotations

import json
import operator
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    calculate_percentile,
    generate_deterministic_matrix,
)

# ==============================================================================
# Constants
# ==============================================================================

BASELINE_M = 512
BASELINE_N = 512
BASELINE_K = 512
BASELINE_DTYPE = "fp32"
BASELINE_BACKEND = "cpu"
BASELINE_WARMUP = 5
BASELINE_ITERATIONS = 30
BASELINE_FLOPS = 2 * BASELINE_M * BASELINE_N * BASELINE_K

LOCKED_BASELINE_MEDIAN_MS = 1342.067
LOCKED_BASELINE_P95_MS = 1406.937
LOCKED_BASELINE_GFLOPS = 0.200

TILE_SIZES = [16, 32, 64, 128]

RESULTS_DIR = Path("experiments/results/experiment_001")


# ==============================================================================
# Tiled GEMM Implementation
# ==============================================================================

def tiled_gemm(
    a: List[List[float]],
    b: List[List[float]],
    m: int,
    n: int,
    k: int,
    tile_size: int,
) -> List[List[float]]:
    """Compute C = A @ B using loop tiling / cache blocking.

    The standard iteration space is partitioned into tiles of size T.
    Sub-blocks of B are loaded and transposed per tile for efficient dot products.

    Args:
        a: Matrix A of shape [m, k] (row-major list of lists).
        b: Matrix B of shape [k, n] (row-major list of lists).
        m: Number of rows of A and C.
        n: Number of columns of B and C.
        k: Inner dimension.
        tile_size: Block size T for tiling.

    Returns:
        Matrix C of shape [m, n].
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if tile_size > max(m, n, k):
        raise ValueError(f"tile_size ({tile_size}) exceeds matrix dimensions ({m}, {n}, {k})")

    c: List[List[float]] = [[0.0] * n for _ in range(m)]
    ts = tile_size
    mul = operator.mul

    for jj in range(0, n, ts):
        j_end = min(jj + ts, n)
        for pp in range(0, k, ts):
            p_end = min(pp + ts, k)
            # Load and transpose sub-tile of B: shape (j_end - jj) x (p_end - pp)
            b_tile = [[b[p][j] for p in range(pp, p_end)] for j in range(jj, j_end)]
            for ii in range(0, m, ts):
                i_end = min(ii + ts, m)
                for i in range(ii, i_end):
                    a_slice = a[i][pp:p_end]
                    c_row = c[i]
                    for j_idx, j in enumerate(range(jj, j_end)):
                        c_row[j] += sum(map(mul, a_slice, b_tile[j_idx]))

    return c


def naive_gemm(
    a: List[List[float]],
    b: List[List[float]],
    m: int,
    n: int,
    k: int,
) -> List[List[float]]:
    """Standard triple-nested loop GEMM for correctness reference."""
    c: List[List[float]] = [[0.0] * n for _ in range(m)]
    for i in range(m):
        a_row = a[i]
        c_row = c[i]
        for kk in range(k):
            a_ik = a_row[kk]
            b_row = b[kk]
            for j in range(n):
                c_row[j] += a_ik * b_row[j]
    return c


def _tiled_chunk_worker(
    a_chunk: List[List[float]],
    b: List[List[float]],
    tile_size: int,
) -> List[List[float]]:
    """Worker function for parallel tiled GEMM.

    Computes tiled GEMM for a row chunk of A against the full matrix B.
    """
    m = len(a_chunk)
    k = len(b)
    n = len(b[0])
    ts = tile_size
    mul = operator.mul
    c: List[List[float]] = [[0.0] * n for _ in range(m)]

    for jj in range(0, n, ts):
        j_end = min(jj + ts, n)
        for pp in range(0, k, ts):
            p_end = min(pp + ts, k)
            b_tile = [[b[p][j] for p in range(pp, p_end)] for j in range(jj, j_end)]
            for ii in range(0, m, ts):
                i_end = min(ii + ts, m)
                for i in range(ii, i_end):
                    a_slice = a_chunk[i][pp:p_end]
                    c_row = c[i]
                    for j_idx, j in enumerate(range(jj, j_end)):
                        c_row[j] += sum(map(mul, a_slice, b_tile[j_idx]))
    return c


class TiledGemmBackend:
    """Experimental tiled GEMM backend for benchmarking.

    Uses configurable tile sizes. For standard benchmark sizes (m >= 64),
    utilizes multiprocessing matching the CPU baseline execution environment.
    """

    def __init__(self, tile_size: int, workers: Optional[int] = None) -> None:
        self.tile_size = tile_size
        self.requested_workers = workers
        self.workers: int = 1
        self.m = 0
        self.n = 0
        self.k = 0
        self.a: Optional[List[List[float]]] = None
        self.b: Optional[List[List[float]]] = None
        self.pool: Optional[Pool] = None
        self.chunks: Optional[List[List[List[float]]]] = None

    def prepare(self, m: int, n: int, k: int, dtype: str = "fp32") -> None:
        self.m = m
        self.n = n
        self.k = k

        if self.requested_workers is not None:
            self.workers = max(1, self.requested_workers)
        else:
            total_cores = os.cpu_count() or 1
            self.workers = min(8, total_cores) if m >= 64 else 1

        self.a = generate_deterministic_matrix(m, k, seed_offset=1, dtype=dtype)
        self.b = generate_deterministic_matrix(k, n, seed_offset=2, dtype=dtype)

        if self.workers > 1 and m >= 64:
            chunk_size = max(1, m // self.workers)
            self.chunks = [self.a[i:i + chunk_size] for i in range(0, m, chunk_size)]
            if self.pool is None:
                self.pool = Pool(processes=self.workers)
        else:
            self.chunks = None
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None

    def execute(self) -> List[List[float]]:
        if self.a is None or self.b is None:
            raise RuntimeError("Backend not prepared")

        if self.pool is not None and self.chunks is not None:
            items = [(chunk, self.b, self.tile_size) for chunk in self.chunks]
            chunk_results = self.pool.starmap(_tiled_chunk_worker, items)
            flattened: List[List[float]] = []
            for sub in chunk_results:
                flattened.extend(sub)
            return flattened
        else:
            return tiled_gemm(self.a, self.b, self.m, self.n, self.k, self.tile_size)

    def cleanup(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


# ==============================================================================
# Experiment 0: Measurement Validation
# ==============================================================================

def run_experiment_0(num_runs: int = 3) -> Dict[str, Any]:
    """Run multiple independent baseline benchmarks to assess measurement stability.

    Uses workers=None (auto-detect) to match the original locked baseline configuration.
    Returns structured validation result with per-run metrics and stability analysis.
    """
    print("=" * 60)
    print("EXPERIMENT 0 -- MEASUREMENT VALIDATION")
    print("=" * 60)
    print(f"Running {num_runs} independent baseline benchmark runs...")
    print(f"Config: M={BASELINE_M}, N={BASELINE_N}, K={BASELINE_K}, "
          f"dtype={BASELINE_DTYPE}, warmup={BASELINE_WARMUP}, iters={BASELINE_ITERATIONS}")
    print(f"Workers: auto-detect (matching locked baseline)")
    print()

    runs: List[Dict[str, Any]] = []

    for run_idx in range(1, num_runs + 1):
        print(f"  Run {run_idx}/{num_runs}...", end=" ", flush=True)

        config = BenchmarkConfig(
            name="gemm_fp32",
            version="1.0.0",
            warmup_runs=BASELINE_WARMUP,
            measurement_runs=BASELINE_ITERATIONS,
            m=BASELINE_M,
            n=BASELINE_N,
            k=BASELINE_K,
            dtype=BASELINE_DTYPE,
            backend=BASELINE_BACKEND,
            workers=None,  # Auto-detect: matches original locked baseline
        )

        runner = BenchmarkRunner(config)
        result = runner.run()

        lat = result.latency_ms
        tp = result.throughput

        run_data = {
            "run_number": run_idx,
            "median_ms": lat["median"],
            "mean_ms": lat["mean"],
            "p95_ms": lat["p95"],
            "p99_ms": lat["p99"],
            "min_ms": lat["min"],
            "max_ms": lat["max"],
            "stddev_ms": lat["stddev"],
            "count": lat["count"],
            "gflops": tp.get("gflops"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        runs.append(run_data)
        print(f"median={lat['median']:.1f}ms, p95={lat['p95']:.1f}ms, "
              f"GFLOPS={tp.get('gflops', 0):.3f}")

    # Stability analysis
    medians = [r["median_ms"] for r in runs]
    means = [r["mean_ms"] for r in runs]
    p95s = [r["p95_ms"] for r in runs]
    gflops_vals = [r["gflops"] for r in runs if r["gflops"] is not None]

    overall_median_of_medians = statistics.median(medians)
    overall_mean_of_medians = statistics.mean(medians)
    median_stddev = statistics.stdev(medians) if len(medians) >= 2 else 0.0
    median_cv = (median_stddev / overall_mean_of_medians * 100.0) if overall_mean_of_medians > 0 else 0.0
    median_range = max(medians) - min(medians)
    median_range_pct = (median_range / overall_mean_of_medians * 100.0) if overall_mean_of_medians > 0 else 0.0

    # Compare with locked baseline
    deviation_from_locked = abs(overall_median_of_medians - LOCKED_BASELINE_MEDIAN_MS)
    deviation_pct = (deviation_from_locked / LOCKED_BASELINE_MEDIAN_MS * 100.0)

    # Stability criteria: CV < 5% across runs AND deviation from locked < 10%
    is_stable = median_cv < 5.0 and deviation_pct < 10.0

    stability = {
        "medians": medians,
        "median_of_medians_ms": overall_median_of_medians,
        "mean_of_medians_ms": overall_mean_of_medians,
        "stddev_of_medians_ms": median_stddev,
        "cv_percent": round(median_cv, 2),
        "range_ms": median_range,
        "range_percent": round(median_range_pct, 2),
        "locked_baseline_median_ms": LOCKED_BASELINE_MEDIAN_MS,
        "deviation_from_locked_ms": round(deviation_from_locked, 3),
        "deviation_from_locked_percent": round(deviation_pct, 2),
        "is_stable": is_stable,
        "stability_conclusion": (
            f"STABLE: Measurement variance (CV={median_cv:.1f}%) is within acceptable bounds. "
            f"Baseline deviation from locked reference is {deviation_pct:.1f}%. "
            "Baseline is sufficiently stable for optimization experiments."
            if is_stable else
            f"VARIANCE OBSERVED: Cross-run CV is {median_cv:.1f}%, deviation from locked "
            f"baseline is {deviation_pct:.1f}%. Optimization experiments can proceed with awareness."
        ),
    }

    result_data = {
        "experiment_id": "experiment-000-measurement-validation",
        "experiment_name": "Measurement Validation",
        "description": "Multiple independent baseline benchmark runs to assess measurement stability",
        "configuration": {
            "m": BASELINE_M, "n": BASELINE_N, "k": BASELINE_K,
            "dtype": BASELINE_DTYPE, "backend": BASELINE_BACKEND,
            "warmup_runs": BASELINE_WARMUP, "measurement_runs": BASELINE_ITERATIONS,
            "num_independent_runs": num_runs,
            "workers": "auto-detect",
        },
        "locked_baseline_reference": {
            "median_ms": LOCKED_BASELINE_MEDIAN_MS,
            "p95_ms": LOCKED_BASELINE_P95_MS,
            "gflops": LOCKED_BASELINE_GFLOPS,
        },
        "runs": runs,
        "stability_analysis": stability,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print()
    print("-" * 60)
    print("MEASUREMENT VALIDATION RESULTS")
    print("-" * 60)
    for r in runs:
        print(f"  Run {r['run_number']}: median={r['median_ms']:.1f}ms, "
              f"p95={r['p95_ms']:.1f}ms, GFLOPS={r['gflops']:.3f}")
    print(f"  Median of medians: {overall_median_of_medians:.1f}ms")
    print(f"  Cross-run CV: {median_cv:.1f}%")
    print(f"  Range: {median_range:.1f}ms ({median_range_pct:.1f}%)")
    print(f"  Deviation from locked baseline: {deviation_from_locked:.1f}ms ({deviation_pct:.1f}%)")
    print(f"  Conclusion: {'STABLE' if is_stable else 'VARIANCE OBSERVED'}")
    print()

    return result_data


# ==============================================================================
# Experiment 1: Loop Tiling
# ==============================================================================

def run_experiment_1() -> Dict[str, Any]:
    """Run tiled GEMM experiments for all tile sizes and compare against locked baseline.

    Returns structured experiment result with per-tile metrics and comparisons.
    """
    print("=" * 60)
    print("EXPERIMENT 1 -- LOOP TILING")
    print("=" * 60)
    print(f"Baseline reference: median={LOCKED_BASELINE_MEDIAN_MS}ms, GFLOPS={LOCKED_BASELINE_GFLOPS}")
    print(f"Tile sizes to evaluate: {TILE_SIZES}")
    print()

    # First verify tiled GEMM correctness at a small scale
    print("  Verifying tiled GEMM correctness (8x8x8)...", end=" ", flush=True)
    a_small = generate_deterministic_matrix(8, 8, seed_offset=1)
    b_small = generate_deterministic_matrix(8, 8, seed_offset=2)
    ref = naive_gemm(a_small, b_small, 8, 8, 8)
    for ts in [2, 4, 8]:
        tiled = tiled_gemm(a_small, b_small, 8, 8, 8, ts)
        for i in range(8):
            for j in range(8):
                if abs(ref[i][j] - tiled[i][j]) > 1e-6:
                    raise AssertionError(
                        f"Tiled GEMM (tile={ts}) mismatch at [{i}][{j}]: "
                        f"expected {ref[i][j]}, got {tiled[i][j]}"
                    )
    print("PASS")

    tile_results: List[Dict[str, Any]] = []

    for tile_size in TILE_SIZES:
        print(f"  Evaluating Tile size {tile_size}...", end=" ", flush=True)

        backend = TiledGemmBackend(tile_size)
        backend.prepare(BASELINE_M, BASELINE_N, BASELINE_K, BASELINE_DTYPE)

        # Warmup
        for _ in range(BASELINE_WARMUP):
            backend.execute()

        # Measurement
        raw_durations_ns: List[int] = []
        for _ in range(BASELINE_ITERATIONS):
            start_ns = time.perf_counter_ns()
            backend.execute()
            end_ns = time.perf_counter_ns()
            raw_durations_ns.append(end_ns - start_ns)

        backend.cleanup()

        durations_ms = [ns / 1_000_000.0 for ns in raw_durations_ns]
        sorted_ms = sorted(durations_ms)

        median_ms = statistics.median(durations_ms)
        mean_ms = statistics.mean(durations_ms)
        min_ms = min(durations_ms)
        max_ms = max(durations_ms)
        stddev_ms = statistics.stdev(durations_ms) if len(durations_ms) >= 2 else 0.0
        p95_ms = calculate_percentile(sorted_ms, 95.0)
        p99_ms = calculate_percentile(sorted_ms, 99.0)
        cv_pct = (stddev_ms / median_ms * 100.0) if median_ms > 0 else 0.0

        median_seconds = median_ms / 1000.0
        gflops = (BASELINE_FLOPS / median_seconds / 1e9) if median_seconds > 0 else 0.0

        speedup = LOCKED_BASELINE_MEDIAN_MS / median_ms if median_ms > 0 else 0.0
        improvement_pct = ((LOCKED_BASELINE_MEDIAN_MS - median_ms) / LOCKED_BASELINE_MEDIAN_MS * 100.0)

        # Verification status
        if improvement_pct >= 5.0:
            verification = "IMPROVED"
        elif improvement_pct <= -5.0:
            verification = "REGRESSED"
        else:
            verification = "NO_MEANINGFUL_CHANGE"

        tile_data = {
            "tile_size": tile_size,
            "latency_ms": {
                "count": len(durations_ms),
                "median": round(median_ms, 3),
                "mean": round(mean_ms, 3),
                "min": round(min_ms, 3),
                "max": round(max_ms, 3),
                "p95": round(p95_ms, 3),
                "p99": round(p99_ms, 3),
                "stddev": round(stddev_ms, 3),
                "cv_percent": round(cv_pct, 2),
            },
            "throughput": {
                "gflops": round(gflops, 6),
                "operations": BASELINE_FLOPS,
            },
            "comparison": {
                "baseline_median_ms": LOCKED_BASELINE_MEDIAN_MS,
                "candidate_median_ms": round(median_ms, 3),
                "speedup": round(speedup, 4),
                "improvement_percent": round(improvement_pct, 2),
                "verification_status": verification,
            },
            "raw_durations_ns": raw_durations_ns,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        tile_results.append(tile_data)

        direction = "faster" if improvement_pct > 0 else "slower"
        print(f"median={median_ms:.1f}ms, p95={p95_ms:.1f}ms, "
              f"GFLOPS={gflops:.3f}, speedup={speedup:.3f}x ({abs(improvement_pct):.1f}% {direction})")

    # Find best tile size
    best = min(tile_results, key=lambda r: r["latency_ms"]["median"])
    best_tile = best["tile_size"]
    best_median = best["latency_ms"]["median"]
    best_gflops = best["throughput"]["gflops"]
    best_speedup = best["comparison"]["speedup"]
    best_improvement = best["comparison"]["improvement_percent"]

    summary = {
        "best_tile_size": best_tile,
        "best_median_ms": best_median,
        "best_gflops": best_gflops,
        "best_speedup": best_speedup,
        "best_improvement_percent": best_improvement,
        "best_verification_status": best["comparison"]["verification_status"],
    }

    result_data = {
        "experiment_id": "experiment-001-loop-tiling",
        "experiment_name": "Loop Tiling / Cache Blocking",
        "schema_version": "1.0",
        "description": "Evaluate loop tiling optimization for pure-Python CPU GEMM across tile sizes",
        "baseline_result_reference": {
            "file": "experiments/results/gemm_cpu_512x512x512_fp32_20261003_043010.json",
            "median_ms": LOCKED_BASELINE_MEDIAN_MS,
            "p95_ms": LOCKED_BASELINE_P95_MS,
            "gflops": LOCKED_BASELINE_GFLOPS,
        },
        "candidate_configuration": {
            "algorithm": "tiled_gemm",
            "m": BASELINE_M, "n": BASELINE_N, "k": BASELINE_K,
            "dtype": BASELINE_DTYPE, "backend": BASELINE_BACKEND,
            "warmup_runs": BASELINE_WARMUP, "measurement_runs": BASELINE_ITERATIONS,
            "tile_sizes": TILE_SIZES,
        },
        "tile_sizes_tested": TILE_SIZES,
        "results": tile_results,
        "summary": summary,
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_cores": os.cpu_count(),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print()
    print("-" * 60)
    print("LOOP TILING RESULTS")
    print("-" * 60)
    print(f"  Baseline median:  {LOCKED_BASELINE_MEDIAN_MS}ms")
    for r in tile_results:
        ts = r["tile_size"]
        med = r["latency_ms"]["median"]
        sp = r["comparison"]["speedup"]
        imp = r["comparison"]["improvement_percent"]
        vs = r["comparison"]["verification_status"]
        print(f"  Tile {ts:>3}: median={med:.1f}ms, speedup={sp:.3f}x, "
              f"improvement={imp:.1f}%, status={vs}")
    print(f"  Best tile size: {best_tile}")
    print(f"  Best median:    {best_median:.1f}ms")
    print(f"  Best GFLOPS:    {best_gflops:.3f}")
    print(f"  Best speedup:   {best_speedup:.3f}x")
    print(f"  Best improvement: {best_improvement:.1f}%")
    print()

    return result_data


# ==============================================================================
# Main
# ==============================================================================

def main() -> int:
    """Execute Experiment 0 and Experiment 1, persist results."""

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Experiment 0 ---
    exp0 = run_experiment_0(num_runs=3)
    exp0_path = RESULTS_DIR / "experiment_000_measurement_validation.json"
    with open(exp0_path, "w") as f:
        json.dump(exp0, f, indent=2)
    print(f"  Saved: {exp0_path}")
    print()

    # --- Experiment 1 ---
    exp1 = run_experiment_1()

    # Save full result (with raw durations)
    exp1_path = RESULTS_DIR / "experiment_001_loop_tiling.json"
    with open(exp1_path, "w") as f:
        json.dump(exp1, f, indent=2)
    print(f"  Saved: {exp1_path}")
    print()

    # Save compact summary (without raw durations for readability)
    exp1_summary = dict(exp1)
    compact_results = []
    for r in exp1["results"]:
        compact = {k: v for k, v in r.items() if k != "raw_durations_ns"}
        compact_results.append(compact)
    exp1_summary["results"] = compact_results
    summary_path = RESULTS_DIR / "experiment_001_summary.json"
    with open(summary_path, "w") as f:
        json.dump(exp1_summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print()
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
