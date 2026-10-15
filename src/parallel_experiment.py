#!/usr/bin/env python3
"""Experiment 3: Multi-Process Parallelism Evaluation for ROCm Performance Guard.

Evaluates scaling of pure-Python GEMM performance across worker process counts:
1 worker, 2 workers, 4 workers, and 8 workers.

Rules:
- Does NOT modify the locked baseline JSON.
- Does NOT combine with loop tiling, Strassen, or external packages.
- Reuses the existing benchmark infrastructure and CpuGemmBackend worker-count mechanism.
- Measures both speedup relative to locked baseline and parallel scaling relative to 1 worker.
- Evaluates CPU utilization using available OS standard library facilities.
- Verifies mathematical correctness and remainder row handling across all worker counts.
"""

from __future__ import annotations

import hashlib
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
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    CpuGemmBackend,
    calculate_percentile,
    generate_deterministic_matrix,
)

# ==============================================================================
# Constants & Reference Baselines
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
LOCKED_BASELINE_SHA256 = "ded6b85ff3ec29349018c25ae7e0d7aedcb3ea719380dc5dec57d1f31d2376ad"

WORKER_CONFIGS = [1, 2, 4, 8]

RESULTS_DIR = Path("experiments/results/experiment_003")


# ==============================================================================
# CPU Utilization Monitoring (Builtin OS Facilities)
# ==============================================================================

def read_system_cpu_ticks() -> Optional[Tuple[float, float]]:
    """Read total and idle CPU ticks from /proc/stat if available.

    Returns:
        (total_ticks, idle_ticks) or None if /proc/stat is unavailable.
    """
    try:
        if not os.path.exists("/proc/stat"):
            return None
        with open("/proc/stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = [float(x) for x in line.strip().split()[1:]]
                    idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
                    total = sum(parts)
                    return total, idle
    except Exception:
        return None
    return None


def calculate_cpu_utilization(
    start_ticks: Optional[Tuple[float, float]],
    end_ticks: Optional[Tuple[float, float]],
) -> Any:
    """Compute CPU utilization percentage between two /proc/stat samples."""
    if start_ticks is None or end_ticks is None:
        return "CPU utilization measurement unavailable"
    tot_delta = end_ticks[0] - start_ticks[0]
    idle_delta = end_ticks[1] - start_ticks[1]
    if tot_delta <= 0:
        return "CPU utilization measurement unavailable"
    usage = (1.0 - (idle_delta / tot_delta)) * 100.0
    return round(max(0.0, min(100.0, usage)), 1)


# ==============================================================================
# Correctness & Partition Verification
# ==============================================================================

def reference_gemm(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Mathematical reference GEMM: C[i][j] = sum(A[i][k] * B[k][j])."""
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    c = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            tot = 0.0
            for p in range(k):
                tot += a[i][p] * b[p][j]
            c[i][j] = tot
    return c


def partition_row_chunks(matrix: List[Any], workers: int) -> List[List[Any]]:
    """Partition matrix rows across worker count preserving all rows."""
    if not matrix:
        return []
    if workers <= 1:
        return [matrix]
    m = len(matrix)
    chunk_size = max(1, m // workers)
    return [matrix[i:i + chunk_size] for i in range(0, m, chunk_size)]


def verify_correctness_across_workers(worker_counts: List[int]) -> Dict[str, Any]:
    """Verify that all worker configurations produce identical mathematical output.

    Tests square, non-square, and indivisible matrix sizes including remainder rows.
    """
    test_cases = [
        ("small_square", 8, 8, 8),
        ("medium_square", 16, 16, 16),
        ("non_square_1", 12, 8, 16),
        ("non_square_2", 15, 10, 7),
        ("indivisible_prime", 17, 19, 23),
        ("remainder_rows_10", 10, 8, 8),
        ("remainder_rows_65", 65, 32, 32),
    ]

    results: Dict[str, Any] = {"test_cases_evaluated": len(test_cases), "details": []}

    for name, m, n, k in test_cases:
        a = generate_deterministic_matrix(m, k, seed_offset=1)
        b = generate_deterministic_matrix(k, n, seed_offset=2)
        ref = reference_gemm(a, b)

        case_detail = {"name": name, "dimensions": f"{m}x{n}x{k}", "workers_checked": []}

        for w in worker_counts:
            # Check partition reconstruction
            chunks = partition_row_chunks(a, w)
            reconstructed = [row for chunk in chunks for row in chunk]
            if reconstructed != a:
                raise AssertionError(f"Row chunk partitioning lost or reordered rows for {name}, w={w}")

            # Execute via backend
            backend = CpuGemmBackend(workers=w)
            backend.prepare(m, n, k, dtype="fp32")
            out = backend.execute()
            backend.cleanup()

            # Verify dimensions
            if len(out) != m or len(out[0]) != n:
                raise AssertionError(f"Output shape mismatch for {name}, w={w}: got {len(out)}x{len(out[0])}")

            # Verify values
            for i in range(m):
                for j in range(n):
                    if abs(ref[i][j] - out[i][j]) > 1e-5:
                        raise AssertionError(
                            f"Numerical mismatch at [{i}][{j}] for {name}, w={w}: ref={ref[i][j]}, cand={out[i][j]}"
                        )

            case_detail["workers_checked"].append(w)

        results["details"].append(case_detail)

    results["status"] = "ALL_PASS"
    return results


# ==============================================================================
# Experiment 3 Execution
# ==============================================================================

def run_single_worker_benchmark(
    workers: int,
    run_idx: int,
    m: int = BASELINE_M,
    n: int = BASELINE_N,
    k: int = BASELINE_K,
    dtype: str = BASELINE_DTYPE,
    warmup: int = BASELINE_WARMUP,
    iterations: int = BASELINE_ITERATIONS,
) -> Dict[str, Any]:
    """Execute one independent benchmark run for a specific worker count."""
    config = BenchmarkConfig(
        name="gemm_fp32",
        version="1.0.0",
        warmup_runs=warmup,
        measurement_runs=iterations,
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        backend=BASELINE_BACKEND,
        workers=workers,
    )

    t_cpu_start = read_system_cpu_ticks()
    runner = BenchmarkRunner(config)
    result = runner.run()
    t_cpu_end = read_system_cpu_ticks()

    cpu_util = calculate_cpu_utilization(t_cpu_start, t_cpu_end)

    lat = result.latency_ms
    tp = result.throughput

    return {
        "run_number": run_idx,
        "workers": workers,
        "median_ms": lat["median"],
        "mean_ms": lat["mean"],
        "p95_ms": lat["p95"],
        "p99_ms": lat["p99"],
        "min_ms": lat["min"],
        "max_ms": lat["max"],
        "stddev_ms": lat["stddev"],
        "count": lat["count"],
        "gflops": tp.get("gflops"),
        "cpu_utilization": cpu_util,
        "raw_durations_ns": result.raw_durations_ns,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_experiment_3(
    worker_configs: Optional[List[int]] = None,
    num_runs: int = 3,
    warmup: int = BASELINE_WARMUP,
    iterations: int = BASELINE_ITERATIONS,
    m: int = BASELINE_M,
    n: int = BASELINE_N,
    k: int = BASELINE_K,
    dtype: str = BASELINE_DTYPE,
) -> Dict[str, Any]:
    """Execute Experiment 3: Multi-Process Parallelism.

    Evaluates worker counts, computes speedup vs locked baseline and scaling
    relative to 1-worker execution.
    """
    if worker_configs is None:
        worker_configs = WORKER_CONFIGS

    print("=" * 65)
    print("EXPERIMENT 3 -- MULTI-PROCESS PARALLELISM")
    print("=" * 65)
    print(f"Workload: M={m}, N={n}, K={k}, dtype={dtype}")
    print(f"Configurations: {worker_configs} workers")
    print(f"Protocol: {warmup} warmup runs, {iterations} measured runs, {num_runs} independent runs")
    print(f"Locked Baseline Reference: median={LOCKED_BASELINE_MEDIAN_MS}ms, GFLOPS={LOCKED_BASELINE_GFLOPS}")
    print()

    # Step 1: Correctness verification
    print("  Verifying mathematical equivalence across all worker counts...", end=" ", flush=True)
    correctness_results = verify_correctness_across_workers(worker_configs)
    print("PASS")
    print()

    flops = 2 * m * n * k
    per_worker_summary: Dict[str, Any] = {}
    runs_by_worker: Dict[int, List[Dict[str, Any]]] = {w: [] for w in worker_configs}

    # Step 2: Benchmark each worker count
    for w in worker_configs:
        print(f"Evaluating {w} Worker{'s' if w > 1 else ''}:")
        for r in range(1, num_runs + 1):
            print(f"  Run {r}/{num_runs}...", end=" ", flush=True)
            run_data = run_single_worker_benchmark(
                workers=w,
                run_idx=r,
                m=m,
                n=n,
                k=k,
                dtype=dtype,
                warmup=warmup,
                iterations=iterations,
            )
            runs_by_worker[w].append(run_data)
            print(f"median={run_data['median_ms']:.1f}ms, p95={run_data['p95_ms']:.1f}ms, GFLOPS={run_data['gflops']:.3f}")

        # Aggregate metrics across runs for this worker count
        w_medians = [run["median_ms"] for run in runs_by_worker[w]]
        overall_med = statistics.median(w_medians)
        overall_mean = statistics.mean(w_medians)
        stddev_med = statistics.stdev(w_medians) if len(w_medians) >= 2 else 0.0
        cv_pct = (stddev_med / overall_mean * 100.0) if overall_mean > 0 else 0.0

        all_p95s = [run["p95_ms"] for run in runs_by_worker[w]]
        overall_p95 = statistics.median(all_p95s)
        all_p99s = [run["p99_ms"] for run in runs_by_worker[w]]
        overall_p99 = statistics.median(all_p99s)
        overall_min = min(run["min_ms"] for run in runs_by_worker[w])
        overall_max = max(run["max_ms"] for run in runs_by_worker[w])

        gflops = (flops / (overall_med / 1000.0) / 1e9) if overall_med > 0 else 0.0

        # Speedup vs locked baseline
        speedup_vs_baseline = LOCKED_BASELINE_MEDIAN_MS / overall_med if overall_med > 0 else 0.0
        improvement_pct = ((LOCKED_BASELINE_MEDIAN_MS - overall_med) / LOCKED_BASELINE_MEDIAN_MS * 100.0)

        # Decision vs locked baseline
        if improvement_pct >= 5.0:
            decision = "ADOPTABLE"
        elif improvement_pct <= -5.0:
            decision = "REJECT"
        else:
            decision = "ADOPTABLE" if overall_med <= LOCKED_BASELINE_MEDIAN_MS else "REJECT"

        # CPU utilization
        cpu_utils = [r["cpu_utilization"] for r in runs_by_worker[w] if isinstance(r["cpu_utilization"], (int, float))]
        avg_cpu_util = round(statistics.mean(cpu_utils), 1) if cpu_utils else "CPU utilization measurement unavailable"

        per_worker_summary[str(w)] = {
            "workers": w,
            "median_ms": round(overall_med, 3),
            "mean_ms": round(overall_mean, 3),
            "p95_ms": round(overall_p95, 3),
            "p99_ms": round(overall_p99, 3),
            "min_ms": round(overall_min, 3),
            "max_ms": round(overall_max, 3),
            "stddev_ms": round(stddev_med, 3),
            "cv_percent": round(cv_pct, 2),
            "gflops": round(gflops, 6),
            "speedup_vs_baseline": round(speedup_vs_baseline, 4),
            "improvement_percent_vs_baseline": round(improvement_pct, 2),
            "decision": decision,
            "cpu_utilization": avg_cpu_util,
            "runs": runs_by_worker[w],
        }
        print(f"  --> Summary {w}W: median={overall_med:.1f}ms, speedup={speedup_vs_baseline:.4f}x ({improvement_pct:+.1f}%), decision={decision}")
        print()

    # Step 3: Scaling metrics relative to 1-worker baseline
    one_worker_med = per_worker_summary["1"]["median_ms"]
    scaling_metrics: Dict[str, Any] = {"reference_1_worker_median_ms": one_worker_med}

    for w in worker_configs:
        cand_med = per_worker_summary[str(w)]["median_ms"]
        scaling_speedup = one_worker_med / cand_med if cand_med > 0 else 0.0
        efficiency = (scaling_speedup / w) * 100.0 if w > 0 else 0.0
        scaling_metrics[f"1_to_{w}_speedup"] = round(scaling_speedup, 4)
        scaling_metrics[f"efficiency_{w}_percent"] = round(efficiency, 2)
        per_worker_summary[str(w)]["parallel_scaling_speedup"] = round(scaling_speedup, 4)
        per_worker_summary[str(w)]["parallel_efficiency_percent"] = round(efficiency, 2)

    # Step 4: Identify best absolute and best scaling configurations
    best_abs = min(worker_configs, key=lambda w: per_worker_summary[str(w)]["median_ms"])
    best_scale = max(
        [w for w in worker_configs if w > 1],
        key=lambda w: per_worker_summary[str(w)]["parallel_scaling_speedup"],
    )

    best_abs_med = per_worker_summary[str(best_abs)]["median_ms"]
    best_abs_speedup = per_worker_summary[str(best_abs)]["speedup_vs_baseline"]
    best_abs_imp = per_worker_summary[str(best_abs)]["improvement_percent_vs_baseline"]

    # Final overall decision: ADOPT if best configuration matches or beats locked baseline
    final_decision = "ADOPTABLE" if best_abs_med <= LOCKED_BASELINE_MEDIAN_MS else "REJECT"

    result_data = {
        "experiment_id": "experiment-003-parallelism",
        "experiment_name": "Multi-Process Parallelism Scaling",
        "schema_version": "1.0",
        "description": "Evaluate CPU worker process scaling for GEMM FP32 on AMD Ryzen 7 7730U",
        "baseline_reference": {
            "file": "experiments/results/gemm_cpu_512x512x512_fp32_20261003_043010.json",
            "median_ms": LOCKED_BASELINE_MEDIAN_MS,
            "p95_ms": LOCKED_BASELINE_P95_MS,
            "gflops": LOCKED_BASELINE_GFLOPS,
            "sha256": LOCKED_BASELINE_SHA256,
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
        "worker_configurations": worker_configs,
        "per_worker_metrics": per_worker_summary,
        "scaling_metrics": scaling_metrics,
        "best_configurations": {
            "best_absolute_worker_count": best_abs,
            "best_absolute_median_ms": best_abs_med,
            "best_absolute_speedup_vs_baseline": best_abs_speedup,
            "best_absolute_improvement_percent": best_abs_imp,
            "best_scaling_worker_count": best_scale,
            "best_scaling_speedup_vs_1w": per_worker_summary[str(best_scale)]["parallel_scaling_speedup"],
        },
        "correctness_results": correctness_results,
        "environment_snapshot": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_cores": os.cpu_count(),
        },
        "verification_status": final_decision,
        "final_decision": final_decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Print Comparison Table
    print()
    print("=" * 80)
    print("EXPERIMENT 3 -- FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Workers':<16} | {'Median (ms)':<12} | {'P95 (ms)':<10} | {'GFLOPS':<8} | {'Speedup':<9} | {'Improvement':<12} | {'Decision':<10}")
    print("-" * 80)
    for w in worker_configs:
        m_info = per_worker_summary[str(w)]
        w_lbl = f"{w} worker{'s' if w > 1 else ''}"
        med_str = f"{m_info['median_ms']:.1f}"
        p95_str = f"{m_info['p95_ms']:.1f}"
        gfl_str = f"{m_info['gflops']:.3f}"
        spd_str = f"{m_info['speedup_vs_baseline']:.4f}x"
        imp_str = f"{m_info['improvement_percent_vs_baseline']:+.2f}%"
        dec_str = m_info["decision"]
        print(f"{w_lbl:<16} | {med_str:<12} | {p95_str:<10} | {gfl_str:<8} | {spd_str:<9} | {imp_str:<12} | {dec_str:<10}")

    print(f"{'Locked baseline':<16} | {LOCKED_BASELINE_MEDIAN_MS:<12.1f} | {LOCKED_BASELINE_P95_MS:<10.1f} | {LOCKED_BASELINE_GFLOPS:<8.3f} | {'1.0000x':<9} | {'0.00%':<12} | {'Reference':<10}")
    print("-" * 80)
    print()
    print("SCALING ANALYSIS (Relative to 1 Worker):")
    for w in worker_configs:
        if w > 1:
            sc_spd = per_worker_summary[str(w)]["parallel_scaling_speedup"]
            eff = per_worker_summary[str(w)]["parallel_efficiency_percent"]
            print(f"  1 -> {w} workers: {sc_spd:.4f}x speedup ({eff:.1f}% parallel efficiency)")
    print()
    print(f"BEST ABSOLUTE CONFIGURATION: {best_abs} Workers ({best_abs_med:.1f} ms, {best_abs_speedup:.4f}x vs baseline)")
    print(f"BEST SCALING CONFIGURATION:  {best_scale} Workers ({per_worker_summary[str(best_scale)]['parallel_scaling_speedup']:.4f}x vs 1 worker)")
    print(f"FINAL DECISION:             {final_decision}")
    print("=" * 80)
    print()

    return result_data


def main() -> int:
    """Execute Experiment 3 and persist results to experiments/results/experiment_003/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = run_experiment_3(worker_configs=[1, 2, 4, 8], num_runs=3)

    # Save full result
    full_path = RESULTS_DIR / "experiment_003_parallelism.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved full result: {full_path}")

    # Save compact summary (without raw_durations_ns)
    summary_path = RESULTS_DIR / "experiment_003_summary.json"
    summary_data = {
        "experiment_id": result["experiment_id"],
        "experiment_name": result["experiment_name"],
        "timestamp": result["timestamp"],
        "baseline_reference": result["baseline_reference"],
        "workload": result["workload"],
        "worker_configurations": result["worker_configurations"],
        "per_worker_summary": {},
        "scaling_metrics": result["scaling_metrics"],
        "best_configurations": result["best_configurations"],
        "verification_status": result["verification_status"],
        "final_decision": result["final_decision"],
        "environment_snapshot": result["environment_snapshot"],
    }
    for w_str, m_info in result["per_worker_metrics"].items():
        compact_m = dict(m_info)
        compact_runs = []
        for r in m_info.get("runs", []):
            cr = {k: v for k, v in r.items() if k != "raw_durations_ns"}
            compact_runs.append(cr)
        compact_m["runs"] = compact_runs
        summary_data["per_worker_summary"][w_str] = compact_m

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
