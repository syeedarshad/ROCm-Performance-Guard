#!/usr/bin/env python3
"""Experiment 2: Pre-Transposed Matrix B Evaluation for ROCm Performance Guard.

Investigates whether pre-transposing matrix B once before GEMM execution improves
steady-state CPU memory access locality and overall throughput.

Rules:
- Does NOT modify the locked baseline.
- Does NOT combine with multiprocessing (pure single-threaded isolated variable).
- Does NOT install external dependencies or packages.
- Measures and separates steady-state GEMM timing from one-time transpose cost.
- Reports both steady-state and end-to-end performance views.
"""

from __future__ import annotations

import json
import operator
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark import (
    calculate_percentile,
    generate_deterministic_matrix,
)

# ==============================================================================
# Constants & Locked Baseline Reference
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

EXPERIMENT_002_DIR = Path("experiments/results/experiment_002")


# ==============================================================================
# Matrix Transpose & GEMM Kernels
# ==============================================================================

def transpose_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Transpose a 2D matrix of shape [rows, cols] into [cols, rows].

    Handles arbitrary square and non-square dimensions.

    Args:
        matrix: 2D row-major list of lists.

    Returns:
        Transposed 2D list of lists.
    """
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")
    return list(map(list, zip(*matrix)))


def gemm_pretransposed(
    a: List[List[float]],
    b_t: List[List[float]],
) -> List[List[float]]:
    """Compute C = A @ B using pre-transposed B (B^T).

    Since B is pre-transposed, each column of B is now a row of B^T.
    The dot product between row A[i] and column B[:, j] becomes a dot product
    between row A[i] and row B^T[j], enabling sequential memory access.

    Args:
        a: Matrix A of shape [m, k].
        b_t: Pre-transposed matrix B of shape [n, k].

    Returns:
        Matrix C of shape [m, n].
    """
    if not a or not a[0] or not b_t or not b_t[0]:
        raise ValueError("Matrices cannot be empty")

    m = len(a)
    k_a = len(a[0])
    n = len(b_t)
    k_b = len(b_t[0])

    if k_a != k_b:
        raise ValueError(
            f"Inner dimension mismatch: A is [m x {k_a}], B^T is [n x {k_b}] (expected {k_a})"
        )

    mul = operator.mul
    return [[sum(map(mul, a_row, b_col)) for b_col in b_t] for a_row in a]


def reference_gemm(
    a: List[List[float]],
    b: List[List[float]],
) -> List[List[float]]:
    """Standard triple-nested loop GEMM for correctness verification.

    Computes C = A @ B using naive column-major inner-loop access on B.
    """
    if not a or not a[0] or not b or not b[0]:
        raise ValueError("Matrices cannot be empty")

    m = len(a)
    k = len(a[0])
    n = len(b[0])

    c = [[0.0] * n for _ in range(m)]
    for i in range(m):
        a_row = a[i]
        c_row = c[i]
        for p in range(k):
            a_ip = a_row[p]
            b_row = b[p]
            for j in range(n):
                c_row[j] += a_ip * b_row[j]
    return c


# ==============================================================================
# Experimental Backend
# ==============================================================================

class PretransposedGemmBackend:
    """Experimental single-threaded backend for pre-transposed GEMM.

    Isolates the effect of memory-access layout (pre-transposed B) without
    confounding by multiprocessing or tiling.
    """

    def __init__(self) -> None:
        self.m = 0
        self.n = 0
        self.k = 0
        self.dtype = "fp32"
        self.a: Optional[List[List[float]]] = None
        self.b: Optional[List[List[float]]] = None
        self.b_t: Optional[List[List[float]]] = None
        self.transpose_time_ms: float = 0.0

    def prepare(self, m: int, n: int, k: int, dtype: str = "fp32") -> None:
        """Prepare deterministic inputs and measure one-time transpose cost."""
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype

        # Generate identical deterministic inputs as baseline
        self.a = generate_deterministic_matrix(m, k, seed_offset=1, dtype=dtype)
        self.b = generate_deterministic_matrix(k, n, seed_offset=2, dtype=dtype)

        # Measure transpose time separately
        t0 = time.perf_counter_ns()
        self.b_t = transpose_matrix(self.b)
        t1 = time.perf_counter_ns()
        self.transpose_time_ms = (t1 - t0) / 1_000_000.0

    def measure_transpose_cost(self, repeats: int = 10) -> float:
        """Benchmark transpose cost over multiple repetitions to ensure accuracy."""
        if self.b is None:
            raise RuntimeError("Backend not prepared")
        times: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            _ = transpose_matrix(self.b)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1_000_000.0)
        self.transpose_time_ms = statistics.median(times)
        return self.transpose_time_ms

    def execute(self) -> List[List[float]]:
        """Execute steady-state GEMM excluding transpose cost."""
        if self.a is None or self.b_t is None:
            raise RuntimeError("Backend not prepared")
        return gemm_pretransposed(self.a, self.b_t)

    def execute_end_to_end(self) -> Tuple[List[List[float]], float]:
        """Execute GEMM including the transpose cost, returning (result, total_ms)."""
        if self.a is None or self.b is None:
            raise RuntimeError("Backend not prepared")
        t0 = time.perf_counter_ns()
        b_t = transpose_matrix(self.b)
        res = gemm_pretransposed(self.a, b_t)
        t1 = time.perf_counter_ns()
        return res, (t1 - t0) / 1_000_000.0


# ==============================================================================
# Experiment Runner
# ==============================================================================

def run_transpose_experiment(
    num_runs: int = 3,
    warmup: int = BASELINE_WARMUP,
    iterations: int = BASELINE_ITERATIONS,
    m: int = BASELINE_M,
    n: int = BASELINE_N,
    k: int = BASELINE_K,
    dtype: str = BASELINE_DTYPE,
) -> Dict[str, Any]:
    """Execute Experiment 2: Pre-Transposed B evaluation.

    Runs multiple independent benchmark runs under identical protocol:
    warmup iterations followed by measured iterations.
    Separates steady-state timing from transpose cost, and computes
    both steady-state and end-to-end performance views.
    """
    print("=" * 65)
    print("EXPERIMENT 2 -- PRE-TRANSPOSED MATRIX B EVALUATION")
    print("=" * 65)
    print(f"Workload: M={m}, N={n}, K={k}, dtype={dtype}")
    print(f"Protocol: {warmup} warmup runs, {iterations} measured runs, {num_runs} independent runs")
    print(f"Execution Model: Single-threaded pure Python (no multiprocessing, no tiling)")
    print(f"Locked Baseline Reference: median={LOCKED_BASELINE_MEDIAN_MS}ms, GFLOPS={LOCKED_BASELINE_GFLOPS}")
    print()

    # Verify mathematical correctness on smaller scale first
    print("  Verifying mathematical equivalence against naive reference...", end=" ", flush=True)
    a_test = generate_deterministic_matrix(16, 16, seed_offset=1)
    b_test = generate_deterministic_matrix(16, 16, seed_offset=2)
    b_test_t = transpose_matrix(b_test)
    ref = reference_gemm(a_test, b_test)
    cand = gemm_pretransposed(a_test, b_test_t)
    for i in range(16):
        for j in range(16):
            if abs(ref[i][j] - cand[i][j]) > 1e-5:
                raise AssertionError(f"Correctness mismatch at [{i}][{j}]")
    print("PASS")

    backend = PretransposedGemmBackend()
    backend.prepare(m, n, k, dtype)
    transpose_cost_ms = backend.measure_transpose_cost(repeats=10)
    print(f"  Measured Pre-Transpose Cost: {transpose_cost_ms:.3f} ms")
    print()

    flops = 2 * m * n * k
    runs_data: List[Dict[str, Any]] = []

    for run_idx in range(1, num_runs + 1):
        print(f"  Run {run_idx}/{num_runs}...", end=" ", flush=True)

        # Warmup
        for _ in range(warmup):
            backend.execute()

        # Measured steady-state iterations
        raw_durations_ns: List[int] = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            backend.execute()
            t1 = time.perf_counter_ns()
            raw_durations_ns.append(t1 - t0)

        durations_ms = [ns / 1_000_000.0 for ns in raw_durations_ns]
        sorted_ms = sorted(durations_ms)

        med_ms = statistics.median(durations_ms)
        mean_ms = statistics.mean(durations_ms)
        p95_ms = calculate_percentile(sorted_ms, 95.0)
        p99_ms = calculate_percentile(sorted_ms, 99.0)
        min_ms = min(durations_ms)
        max_ms = max(durations_ms)
        stddev_ms = statistics.stdev(durations_ms) if len(durations_ms) >= 2 else 0.0
        cv_pct = (stddev_ms / med_ms * 100.0) if med_ms > 0 else 0.0

        gflops = (flops / (med_ms / 1000.0) / 1e9) if med_ms > 0 else 0.0

        run_info = {
            "run_number": run_idx,
            "steady_state_median_ms": round(med_ms, 3),
            "steady_state_mean_ms": round(mean_ms, 3),
            "steady_state_p95_ms": round(p95_ms, 3),
            "steady_state_p99_ms": round(p99_ms, 3),
            "steady_state_min_ms": round(min_ms, 3),
            "steady_state_max_ms": round(max_ms, 3),
            "steady_state_stddev_ms": round(stddev_ms, 3),
            "steady_state_cv_percent": round(cv_pct, 2),
            "steady_state_gflops": round(gflops, 6),
            "transpose_time_ms": round(transpose_cost_ms, 3),
            "end_to_end_median_ms": round(med_ms + transpose_cost_ms, 3),
            "raw_durations_ns": raw_durations_ns,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        runs_data.append(run_info)
        print(f"steady_median={med_ms:.1f}ms, p95={p95_ms:.1f}ms, GFLOPS={gflops:.3f}")

    # Aggregate Steady-State Metrics
    all_steady_medians = [r["steady_state_median_ms"] for r in runs_data]
    overall_steady_median = statistics.median(all_steady_medians)
    overall_steady_mean = statistics.mean(all_steady_medians)
    steady_stddev = statistics.stdev(all_steady_medians) if len(all_steady_medians) >= 2 else 0.0
    steady_cv = (steady_stddev / overall_steady_mean * 100.0) if overall_steady_mean > 0 else 0.0

    all_p95s = [r["steady_state_p95_ms"] for r in runs_data]
    overall_steady_p95 = statistics.median(all_p95s)
    all_p99s = [r["steady_state_p99_ms"] for r in runs_data]
    overall_steady_p99 = statistics.median(all_p99s)
    overall_steady_min = min(r["steady_state_min_ms"] for r in runs_data)
    overall_steady_max = max(r["steady_state_max_ms"] for r in runs_data)
    overall_steady_gflops = (flops / (overall_steady_median / 1000.0) / 1e9) if overall_steady_median > 0 else 0.0

    steady_speedup = LOCKED_BASELINE_MEDIAN_MS / overall_steady_median if overall_steady_median > 0 else 0.0
    steady_improvement_pct = ((LOCKED_BASELINE_MEDIAN_MS - overall_steady_median) / LOCKED_BASELINE_MEDIAN_MS * 100.0)

    # Aggregate End-to-End Metrics (steady latency + transpose cost)
    overall_e2e_median = overall_steady_median + transpose_cost_ms
    overall_e2e_mean = overall_steady_mean + transpose_cost_ms
    overall_e2e_p95 = overall_steady_p95 + transpose_cost_ms
    overall_e2e_p99 = overall_steady_p99 + transpose_cost_ms
    overall_e2e_min = overall_steady_min + transpose_cost_ms
    overall_e2e_max = overall_steady_max + transpose_cost_ms
    overall_e2e_gflops = (flops / (overall_e2e_median / 1000.0) / 1e9) if overall_e2e_median > 0 else 0.0

    e2e_speedup = LOCKED_BASELINE_MEDIAN_MS / overall_e2e_median if overall_e2e_median > 0 else 0.0
    e2e_improvement_pct = ((LOCKED_BASELINE_MEDIAN_MS - overall_e2e_median) / LOCKED_BASELINE_MEDIAN_MS * 100.0)

    # Verification Decision (against locked baseline)
    def determine_status(imp_pct: float) -> str:
        if imp_pct >= 5.0:
            return "IMPROVED"
        elif imp_pct <= -5.0:
            return "REGRESSED"
        else:
            return "NO_MEANINGFUL_CHANGE"

    steady_status = determine_status(steady_improvement_pct)
    e2e_status = determine_status(e2e_improvement_pct)

    # Final Decision
    final_decision = "REJECT" if steady_status == "REGRESSED" else ("ADOPT" if steady_status == "IMPROVED" else "INCONCLUSIVE")

    result = {
        "experiment_id": "experiment-002-pretransposed-b",
        "experiment_name": "Pre-Transposed Matrix B Memory Locality",
        "schema_version": "1.0",
        "description": "Evaluate pre-transposing matrix B to optimize memory-access locality in CPU GEMM",
        "baseline_reference": {
            "file": "experiments/results/gemm_cpu_512x512x512_fp32_20261003_043010.json",
            "median_ms": LOCKED_BASELINE_MEDIAN_MS,
            "p95_ms": LOCKED_BASELINE_P95_MS,
            "gflops": LOCKED_BASELINE_GFLOPS,
            "backend": BASELINE_BACKEND,
            "note": "Locked baseline utilized multi-worker execution on Ryzen 7 7730U",
        },
        "workload": {
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype,
            "backend": BASELINE_BACKEND,
            "warmup_runs": warmup,
            "measurement_runs": iterations,
            "num_independent_runs": num_runs,
            "flops": flops,
        },
        "candidate_configuration": {
            "name": "pretransposed_b",
            "optimization": "pre-transposed matrix B layout",
            "parallelism": "single_threaded (no multiprocessing)",
            "loop_tiling": "none",
            "transpose_implementation": "list(map(list, zip(*b)))",
            "kernel": "sum(map(operator.mul, a_row, b_col)) over transposed B",
        },
        "transpose_time_ms": round(transpose_cost_ms, 3),
        "steady_state_metrics": {
            "median_ms": round(overall_steady_median, 3),
            "mean_ms": round(overall_steady_mean, 3),
            "p95_ms": round(overall_steady_p95, 3),
            "p99_ms": round(overall_steady_p99, 3),
            "min_ms": round(overall_steady_min, 3),
            "max_ms": round(overall_steady_max, 3),
            "stddev_ms": round(steady_stddev, 3),
            "cv_percent": round(steady_cv, 2),
            "gflops": round(overall_steady_gflops, 6),
        },
        "end_to_end_metrics": {
            "median_ms": round(overall_e2e_median, 3),
            "mean_ms": round(overall_e2e_mean, 3),
            "p95_ms": round(overall_e2e_p95, 3),
            "p99_ms": round(overall_e2e_p99, 3),
            "min_ms": round(overall_e2e_min, 3),
            "max_ms": round(overall_e2e_max, 3),
            "stddev_ms": round(steady_stddev, 3),
            "gflops": round(overall_e2e_gflops, 6),
        },
        "speedup": {
            "steady_state": round(steady_speedup, 4),
            "end_to_end": round(e2e_speedup, 4),
        },
        "improvement_percentage": {
            "steady_state": round(steady_improvement_pct, 2),
            "end_to_end": round(e2e_improvement_pct, 2),
        },
        "verification_status": {
            "steady_state": steady_status,
            "end_to_end": e2e_status,
        },
        "final_decision": final_decision,
        "runs": runs_data,
        "environment_snapshot": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_cores": os.cpu_count(),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print()
    print("-" * 65)
    print("EXPERIMENT 2 RESULTS SUMMARY")
    print("-" * 65)
    print(f"  Transpose Cost:          {transpose_cost_ms:.3f} ms")
    print(f"  Steady-State Median:     {overall_steady_median:.1f} ms")
    print(f"  Steady-State P95:        {overall_steady_p95:.1f} ms")
    print(f"  Steady-State GFLOPS:     {overall_steady_gflops:.3f}")
    print(f"  Steady-State Speedup:    {steady_speedup:.4f}x ({steady_improvement_pct:.1f}%)")
    print(f"  Steady-State Status:     {steady_status}")
    print()
    print(f"  End-to-End Median:       {overall_e2e_median:.1f} ms")
    print(f"  End-to-End P95:          {overall_e2e_p95:.1f} ms")
    print(f"  End-to-End GFLOPS:       {overall_e2e_gflops:.3f}")
    print(f"  End-to-End Speedup:      {e2e_speedup:.4f}x ({e2e_improvement_pct:.1f}%)")
    print(f"  End-to-End Status:       {e2e_status}")
    print(f"  Final Decision:          {final_decision}")
    print("-" * 65)
    print()

    return result


def main() -> int:
    """Execute Experiment 2 and persist JSON results."""
    EXPERIMENT_002_DIR.mkdir(parents=True, exist_ok=True)

    result = run_transpose_experiment(num_runs=3)

    # Save full result
    full_path = EXPERIMENT_002_DIR / "experiment_002_pretransposed_b.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved full result: {full_path}")

    # Save compact summary
    summary_path = EXPERIMENT_002_DIR / "experiment_002_summary.json"
    summary_data = {
        "experiment_id": result["experiment_id"],
        "experiment_name": result["experiment_name"],
        "timestamp": result["timestamp"],
        "baseline_reference": result["baseline_reference"],
        "workload": result["workload"],
        "candidate_configuration": result["candidate_configuration"],
        "transpose_time_ms": result["transpose_time_ms"],
        "steady_state_metrics": result["steady_state_metrics"],
        "end_to_end_metrics": result["end_to_end_metrics"],
        "speedup": result["speedup"],
        "improvement_percentage": result["improvement_percentage"],
        "verification_status": result["verification_status"],
        "final_decision": result["final_decision"],
        "environment_snapshot": result["environment_snapshot"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
