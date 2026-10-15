"""Experiment Planner V1 for ROCm Performance Guard.

Analyzes an existing baseline benchmark result and generates evidence-based,
CONTROLLED experiment candidates for CPU GEMM optimization.

Core contract:
    - Reads and analyzes an existing baseline result (JSON or BenchmarkResult object).
    - Generates ranked experiment candidates with rationale derived from observed evidence.
    - NEVER executes optimizations automatically.
    - NEVER mutates system state, files, or environment variables.
    - Pure standard-library Python. No external dependencies.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from src.benchmark import BenchmarkResult
except ImportError:
    BenchmarkResult = Any  # type: ignore[misc,assignment]


# ==============================================================================
# Enums
# ==============================================================================

class ExperimentStatus(str, Enum):
    """Classification status for experiment planning outcome."""

    EXPERIMENTS_GENERATED = "EXPERIMENTS_GENERATED"
    NO_EXPERIMENTS_POSSIBLE = "NO_EXPERIMENTS_POSSIBLE"
    INSUFFICIENT_BASELINE = "INSUFFICIENT_BASELINE"
    BASELINE_ALREADY_OPTIMAL = "BASELINE_ALREADY_OPTIMAL"


class ExperimentPriority(str, Enum):
    """Investigation priority for experiment candidates."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ExperimentRisk(str, Enum):
    """Risk assessment for proposed experiments."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ExperimentCategory(str, Enum):
    """Category classification for experiment candidates."""

    ALGORITHM = "algorithm"
    PARALLELISM = "parallelism"
    MEMORY_ACCESS = "memory_access"
    MEASUREMENT = "measurement"


DEFAULT_EXPERIMENT_LIMITATIONS: List[str] = [
    "Experiment candidates are hypotheses requiring controlled empirical measurement.",
    "Expected speedup estimates are approximate upper bounds; actual improvements depend on hardware, cache hierarchy, and OS scheduling.",
    "No experiments are executed automatically; the plan is advisory only.",
    "Each experiment must be run under identical hardware and environment conditions as the baseline for valid comparison.",
    "Pure-Python CPU GEMM performance is dominated by interpreter overhead; algorithmic gains may be masked.",
]


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class ExperimentCandidate:
    """A single, evidence-backed experiment proposal for CPU GEMM optimization.

    Attributes:
        experiment_id: Unique identifier for this experiment.
        category: Classification (algorithm, parallelism, memory_access, measurement).
        title: Short human-readable title.
        hypothesis: Specific, falsifiable hypothesis statement.
        description: Detailed description of the proposed change.
        rationale: Evidence-based reasoning connecting baseline observations to this experiment.
        priority: Investigation priority (HIGH, MEDIUM, LOW).
        risk: Risk assessment (LOW, MEDIUM, HIGH).
        independent_variable: The single variable being manipulated.
        control_variables: Variables held constant to ensure a controlled experiment.
        dependent_variable: The metric being measured.
        proposed_values: Specific values to test for the independent variable.
        expected_direction: Whether the experiment aims to reduce latency or increase throughput.
        estimated_speedup_range: Approximate expected improvement range.
        measurement_requirements: Requirements for valid measurement.
        acceptance_criteria: How to determine if the experiment succeeded.
        rollback_plan: How to revert to the baseline configuration if needed.
    """

    experiment_id: str
    category: str
    title: str
    hypothesis: str
    description: str
    rationale: str
    priority: str
    risk: str
    independent_variable: str
    control_variables: List[str]
    dependent_variable: str
    proposed_values: List[Any]
    expected_direction: str
    estimated_speedup_range: str
    measurement_requirements: Dict[str, Any]
    acceptance_criteria: str
    rollback_plan: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to clean JSON-serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "category": self.category,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "description": self.description,
            "rationale": self.rationale,
            "priority": self.priority,
            "risk": self.risk,
            "independent_variable": self.independent_variable,
            "control_variables": list(self.control_variables),
            "dependent_variable": self.dependent_variable,
            "proposed_values": list(self.proposed_values),
            "expected_direction": self.expected_direction,
            "estimated_speedup_range": self.estimated_speedup_range,
            "measurement_requirements": dict(self.measurement_requirements),
            "acceptance_criteria": self.acceptance_criteria,
            "rollback_plan": self.rollback_plan,
        }


@dataclass
class ExperimentPlan:
    """Comprehensive, JSON-serializable experiment plan for CPU GEMM optimization.

    Attributes:
        schema_version: Schema version for forward compatibility.
        status: Overall plan status.
        summary: Human-readable summary of the plan.
        baseline_analysis: Extracted and derived analysis of the baseline result.
        candidates: Ordered list of experiment candidates.
        execution_order: Recommended order of execution (list of experiment_ids).
        limitations: Epistemic limitations and caveats.
        generated_at: ISO 8601 timestamp of plan generation.
    """

    schema_version: str
    status: ExperimentStatus
    summary: str
    baseline_analysis: Dict[str, Any]
    candidates: List[ExperimentCandidate] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=lambda: list(DEFAULT_EXPERIMENT_LIMITATIONS))
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "status": self.status.value if isinstance(self.status, ExperimentStatus) else str(self.status),
            "summary": self.summary,
            "baseline_analysis": dict(self.baseline_analysis),
            "candidates": [c.to_dict() for c in self.candidates],
            "execution_order": list(self.execution_order),
            "limitations": list(self.limitations),
            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert plan to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ==============================================================================
# Baseline Analysis Helpers
# ==============================================================================

def _to_dict_safe(obj: Any) -> Optional[Dict[str, Any]]:
    """Safely normalize a BenchmarkResult or dict to a dictionary."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return None


def _extract_baseline_metrics(baseline_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key performance metrics from a baseline benchmark result.

    Returns a structured analysis dict with:
        - workload: type and dimensions
        - latency: summary statistics
        - throughput: operations per second, GFLOPS
        - configuration: benchmark settings
        - observations: derived performance characteristics
    """
    config = baseline_dict.get("configuration", {})
    latency = baseline_dict.get("latency_ms", {})
    throughput = baseline_dict.get("throughput", {})
    benchmark = baseline_dict.get("benchmark", {})
    hw = baseline_dict.get("hardware_snapshot", {})

    m = config.get("m", 0)
    n = config.get("n", 0)
    k = config.get("k", 0)
    dtype = config.get("dtype", "unknown")
    backend = config.get("backend", "unknown")
    warmup = config.get("warmup_runs", 0)
    iterations = config.get("measurement_runs", 0)
    operation_count = config.get("operation_count", 0)

    median_ms = latency.get("median", 0.0)
    mean_ms = latency.get("mean", 0.0)
    stddev_ms = latency.get("stddev", 0.0)
    p95_ms = latency.get("p95", 0.0)
    p99_ms = latency.get("p99", 0.0)
    min_ms = latency.get("min", 0.0)
    max_ms = latency.get("max", 0.0)
    count = latency.get("count", 0)

    gflops = throughput.get("gflops")
    ops_per_sec = throughput.get("operations_per_second")

    # Derived observations
    observations: List[str] = []

    # Coefficient of variation
    cv = 0.0
    if median_ms > 0.0:
        cv = (stddev_ms / median_ms) * 100.0

    if cv > 5.0:
        observations.append(
            f"High measurement variance detected: CV={cv:.1f}% (stddev={stddev_ms:.1f}ms, median={median_ms:.1f}ms). "
            "This may indicate background system noise or thermal throttling."
        )
    elif cv > 2.0:
        observations.append(
            f"Moderate measurement variance: CV={cv:.1f}%. "
            "Results are reasonably stable but more iterations could reduce noise."
        )
    else:
        observations.append(
            f"Low measurement variance: CV={cv:.1f}%. Baseline measurements appear stable."
        )

    # Tail latency check
    if median_ms > 0.0 and p95_ms > 0.0:
        tail_ratio = p95_ms / median_ms
        if tail_ratio > 1.15:
            observations.append(
                f"Significant tail latency: p95/median ratio={tail_ratio:.2f}. "
                "Occasional slow iterations may indicate OS scheduling jitter or cache pressure."
            )

    # GFLOPS context for pure-Python
    if gflops is not None and gflops < 1.0:
        observations.append(
            f"Throughput is {gflops:.3f} GFLOPS, which is expected for pure-Python CPU GEMM. "
            "Algorithmic and access-pattern optimizations may yield measurable improvements."
        )

    # Large matrix observation
    total_elements = m * n + n * k + m * k
    if total_elements > 500_000:
        observations.append(
            f"Total matrix elements ({total_elements:,}) exceed typical L2 cache capacity. "
            "Loop tiling may improve cache locality."
        )

    # CPU core count
    cpu_info = hw.get("cpu", {})
    logical_cores = cpu_info.get("logical_cores", os.cpu_count() or 1)
    physical_cores = cpu_info.get("physical_cores", logical_cores)

    if logical_cores > 1 and backend == "cpu":
        observations.append(
            f"System has {physical_cores} physical / {logical_cores} logical CPU cores. "
            f"Parallel work distribution may reduce latency."
        )

    analysis: Dict[str, Any] = {
        "workload": {
            "type": config.get("workload", "gemm"),
            "name": benchmark.get("name", "unknown"),
            "version": benchmark.get("version", "unknown"),
            "dimensions": {"m": m, "n": n, "k": k},
            "dtype": dtype,
            "backend": backend,
            "total_elements": total_elements,
            "operation_count": operation_count,
        },
        "latency": {
            "median_ms": median_ms,
            "mean_ms": mean_ms,
            "stddev_ms": stddev_ms,
            "cv_percent": round(cv, 2),
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "count": count,
        },
        "throughput": {
            "gflops": gflops,
            "operations_per_second": ops_per_sec,
        },
        "configuration": {
            "warmup_runs": warmup,
            "measurement_runs": iterations,
        },
        "hardware": {
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "cpu_model": cpu_info.get("model_name", "unknown"),
        },
        "observations": observations,
    }

    return analysis


def _compute_flops(m: int, n: int, k: int) -> int:
    """Compute floating-point operation count for GEMM C[m,n] = A[m,k] @ B[k,n]."""
    return 2 * m * n * k


# ==============================================================================
# Experiment Candidate Generators
# ==============================================================================

def _generate_tiling_candidate(
    analysis: Dict[str, Any],
    idx: int,
) -> ExperimentCandidate:
    """Generate loop tiling / blocking experiment candidate.

    Loop tiling partitions the matrix into sub-blocks that fit in L1/L2 cache,
    reducing cache misses for the inner loop.
    """
    dims = analysis["workload"]["dimensions"]
    m, n, k = dims["m"], dims["n"], dims["k"]
    total_elements = analysis["workload"]["total_elements"]
    median_ms = analysis["latency"]["median_ms"]

    # Tile sizes to test: powers of 2 that divide evenly, or common sizes
    candidate_tiles = []
    for ts in [16, 32, 64, 128]:
        if m >= ts and n >= ts and k >= ts:
            candidate_tiles.append(ts)

    if not candidate_tiles:
        candidate_tiles = [min(m, n, k)]

    return ExperimentCandidate(
        experiment_id=f"tile-{idx:03d}",
        category=ExperimentCategory.ALGORITHM.value,
        title="Loop Tiling / Cache Blocking for GEMM Inner Loops",
        hypothesis=(
            f"Partitioning the {m}x{n}x{k} GEMM into tile-sized sub-blocks "
            "will reduce L2 cache misses and lower median latency compared to "
            f"the baseline ({median_ms:.1f}ms)."
        ),
        description=(
            "Implement blocked matrix multiplication where the triple-nested loop "
            "is partitioned into tiles of size T. For each tile, the sub-block of "
            "A[i:i+T, p:p+T] and B[p:p+T, j:j+T] fits within cache, reducing "
            "main-memory bandwidth pressure. The total FLOPs remain identical; "
            "only the access pattern changes."
        ),
        rationale=(
            f"The baseline GEMM processes {total_elements:,} total matrix elements. "
            f"With {m}x{k} (A), {k}x{n} (B), and {m}x{n} (C) matrices, "
            "the working set exceeds typical L2 cache size. "
            "The standard ijk loop order exhibits poor spatial locality in the "
            "inner dimension. Tiling converts O(N^3) cache misses to O(N^3/T) "
            "by reusing data within each cache-resident tile."
        ),
        priority=ExperimentPriority.HIGH.value,
        risk=ExperimentRisk.LOW.value,
        independent_variable="tile_size",
        control_variables=[
            f"matrix_dimensions: m={m}, n={n}, k={k}",
            f"dtype: {analysis['workload']['dtype']}",
            f"backend: {analysis['workload']['backend']}",
            f"warmup_runs: {analysis['configuration']['warmup_runs']}",
            f"measurement_runs: {analysis['configuration']['measurement_runs']}",
        ],
        dependent_variable="median_latency_ms",
        proposed_values=candidate_tiles,
        expected_direction="LOWER_LATENCY",
        estimated_speedup_range="1.05x-1.3x",
        measurement_requirements={
            "warmup_runs": analysis["configuration"]["warmup_runs"],
            "measurement_runs": analysis["configuration"]["measurement_runs"],
            "identical_hardware": True,
            "identical_environment": True,
            "comparison_metric": "median_latency_ms",
        },
        acceptance_criteria=(
            "Accept if median latency decreases by >=5% compared to baseline "
            f"({median_ms:.1f}ms) with p95 not increasing by more than 10%."
        ),
        rollback_plan=(
            "Revert to standard non-tiled GEMM implementation. "
            "No system state is modified by this experiment."
        ),
    )


def _generate_transpose_candidate(
    analysis: Dict[str, Any],
    idx: int,
) -> ExperimentCandidate:
    """Generate transposed B matrix access experiment candidate.

    Pre-transposing B converts column-major access in the inner loop to
    row-major (contiguous) access, improving cache line utilization.
    """
    dims = analysis["workload"]["dimensions"]
    m, n, k = dims["m"], dims["n"], dims["k"]
    median_ms = analysis["latency"]["median_ms"]

    return ExperimentCandidate(
        experiment_id=f"transpose-{idx:03d}",
        category=ExperimentCategory.MEMORY_ACCESS.value,
        title="Pre-Transposed B Matrix for Contiguous Inner-Loop Access",
        hypothesis=(
            "Pre-transposing matrix B so that the inner dot-product loop accesses "
            "contiguous memory will improve cache line utilization and reduce "
            f"median latency below the baseline ({median_ms:.1f}ms)."
        ),
        description=(
            "In standard C[i][j] += A[i][p] * B[p][j], accessing B[p][j] for "
            "varying p with fixed j is a strided (column) access pattern. "
            "By pre-computing B_T[j][p] = B[p][j], the inner loop becomes "
            "C[i][j] += A[i][p] * B_T[j][p], where both A[i] and B_T[j] are "
            "accessed contiguously. The transpose cost is O(k*n), amortized "
            "over O(m*n*k) multiplications."
        ),
        rationale=(
            "The existing CpuGemmBackend already uses a B-transposed layout. "
            "This experiment validates the impact by comparing transposed vs. "
            "naive column-access patterns to quantify the cache benefit. "
            "On the Ryzen 7 7730U with 512KB L2 per core, row-contiguous access "
            "ensures each cache line (64 bytes = 8 doubles / 16 floats) is fully utilized."
        ),
        priority=ExperimentPriority.MEDIUM.value,
        risk=ExperimentRisk.LOW.value,
        independent_variable="b_matrix_layout",
        control_variables=[
            f"matrix_dimensions: m={m}, n={n}, k={k}",
            f"dtype: {analysis['workload']['dtype']}",
            f"backend: {analysis['workload']['backend']}",
            f"warmup_runs: {analysis['configuration']['warmup_runs']}",
            f"measurement_runs: {analysis['configuration']['measurement_runs']}",
        ],
        dependent_variable="median_latency_ms",
        proposed_values=["row_major_transposed", "column_major_naive"],
        expected_direction="LOWER_LATENCY",
        estimated_speedup_range="1.1x-2.0x",
        measurement_requirements={
            "warmup_runs": analysis["configuration"]["warmup_runs"],
            "measurement_runs": analysis["configuration"]["measurement_runs"],
            "identical_hardware": True,
            "identical_environment": True,
            "comparison_metric": "median_latency_ms",
        },
        acceptance_criteria=(
            "Accept if the transposed layout achieves >=5% lower median latency "
            "than the naive column-access layout under identical conditions."
        ),
        rollback_plan=(
            "Use whichever layout produces lower latency. "
            "No system state is modified by this experiment."
        ),
    )


def _generate_parallelism_candidate(
    analysis: Dict[str, Any],
    idx: int,
) -> ExperimentCandidate:
    """Generate multi-process parallelism experiment candidate.

    Distribute row chunks across CPU cores using multiprocessing.Pool.
    """
    dims = analysis["workload"]["dimensions"]
    m, n, k = dims["m"], dims["n"], dims["k"]
    median_ms = analysis["latency"]["median_ms"]
    physical_cores = analysis["hardware"]["physical_cores"]
    logical_cores = analysis["hardware"]["logical_cores"]

    worker_counts = []
    for w in [1, 2, 4, 8]:
        if w <= logical_cores:
            worker_counts.append(w)
    if physical_cores not in worker_counts and physical_cores <= logical_cores:
        worker_counts.append(physical_cores)
        worker_counts.sort()

    return ExperimentCandidate(
        experiment_id=f"parallel-{idx:03d}",
        category=ExperimentCategory.PARALLELISM.value,
        title="Multi-Process Row-Chunk Distribution for CPU GEMM",
        hypothesis=(
            f"Distributing the {m} row computations across multiple CPU worker "
            f"processes will reduce median latency from {median_ms:.1f}ms by "
            f"utilizing the available {physical_cores} physical / {logical_cores} logical cores."
        ),
        description=(
            "Partition matrix A into W row-chunks (where W = number of workers). "
            "Each worker independently computes C_chunk[i][j] = sum(A_chunk[i][p] * B_T[j][p]) "
            "for its assigned rows. Results are concatenated. "
            "Inter-process data serialization (pickle) overhead is measured and reported."
        ),
        rationale=(
            f"The baseline runs on a system with {physical_cores} physical cores "
            f"({logical_cores} logical). Pure-Python GEMM is CPU-bound and embarrassingly "
            "parallel across rows. The GIL prevents threading from utilizing multiple cores, "
            "but multiprocessing.Pool bypasses this limitation. For M={m}, chunk size "
            f"at W={min(physical_cores, m)} is {max(1, m // min(physical_cores, m))} rows/worker."
        ),
        priority=ExperimentPriority.HIGH.value,
        risk=ExperimentRisk.LOW.value,
        independent_variable="worker_count",
        control_variables=[
            f"matrix_dimensions: m={m}, n={n}, k={k}",
            f"dtype: {analysis['workload']['dtype']}",
            "b_matrix_layout: transposed",
            f"warmup_runs: {analysis['configuration']['warmup_runs']}",
            f"measurement_runs: {analysis['configuration']['measurement_runs']}",
        ],
        dependent_variable="median_latency_ms",
        proposed_values=worker_counts,
        expected_direction="LOWER_LATENCY",
        estimated_speedup_range=f"1.5x-{min(physical_cores, m):.0f}x",
        measurement_requirements={
            "warmup_runs": analysis["configuration"]["warmup_runs"],
            "measurement_runs": analysis["configuration"]["measurement_runs"],
            "identical_hardware": True,
            "identical_environment": True,
            "comparison_metric": "median_latency_ms",
            "note": "Include process spawn overhead in measurement; do NOT pre-create pool before timing.",
        },
        acceptance_criteria=(
            "Accept if median latency decreases by >=10% compared to single-worker baseline. "
            "Report scaling efficiency = (1-worker time) / (W-worker time) / W."
        ),
        rollback_plan=(
            "Fall back to single-worker execution. Pool processes are terminated after each run. "
            "No system state is modified."
        ),
    )


def _generate_strassen_candidate(
    analysis: Dict[str, Any],
    idx: int,
) -> ExperimentCandidate:
    """Generate Strassen algorithm experiment candidate.

    Strassen reduces multiplication count from O(n^3) to O(n^{2.807}) at the cost
    of more additions and increased memory usage.
    """
    dims = analysis["workload"]["dimensions"]
    m, n, k = dims["m"], dims["n"], dims["k"]
    median_ms = analysis["latency"]["median_ms"]

    is_square = (m == n == k)
    is_power_of_2 = is_square and (m & (m - 1) == 0) and m > 0
    min_dim = min(m, n, k)

    crossover_candidates = [32, 64, 128, 256]
    valid_crossovers = [c for c in crossover_candidates if c <= min_dim]
    if not valid_crossovers:
        valid_crossovers = [min_dim]

    risk = ExperimentRisk.MEDIUM.value
    priority = ExperimentPriority.LOW.value
    if is_square and min_dim >= 256:
        priority = ExperimentPriority.MEDIUM.value

    padding_note = (
        "Matrices are already square and power-of-2, ideal for Strassen. "
        if is_power_of_2 else
        "Non-square or non-power-of-2 matrices require padding, adding overhead. "
    )

    return ExperimentCandidate(
        experiment_id=f"strassen-{idx:03d}",
        category=ExperimentCategory.ALGORITHM.value,
        title="Strassen Algorithm with Crossover Threshold for Large GEMM",
        hypothesis=(
            f"Applying Strassen's algorithm (O(n^2.807)) with a crossover threshold "
            f"to standard GEMM for sub-matrices below threshold will reduce total "
            f"multiplication count and lower median latency from {median_ms:.1f}ms "
            f"for the {m}x{n}x{k} workload."
        ),
        description=(
            "Strassen's algorithm reduces 8 recursive multiplications to 7 at each level, "
            "at the cost of 18 additions/subtractions per level. For pure-Python, the "
            "reduced multiplication count may not offset the overhead of recursive calls, "
            "temporary matrix allocation, and addition operations. A crossover threshold "
            "determines when to fall back to standard GEMM for small sub-blocks. "
            + padding_note
        ),
        rationale=(
            f"Standard GEMM requires {_compute_flops(m, n, k):,} multiply-add operations. "
            f"Strassen at one level reduces this to ~{int(7 * (m // 2) * (n // 2) * (k // 2) * 2):,} "
            "multiplications (7/8 = 87.5% of standard). However, pure-Python function call "
            "and list allocation overhead may dominate. The experiment measures whether the "
            "asymptotic advantage materializes at this matrix size."
        ),
        priority=priority,
        risk=risk,
        independent_variable="strassen_crossover_threshold",
        control_variables=[
            f"matrix_dimensions: m={m}, n={n}, k={k}",
            f"dtype: {analysis['workload']['dtype']}",
            f"backend: {analysis['workload']['backend']}",
            f"warmup_runs: {analysis['configuration']['warmup_runs']}",
            f"measurement_runs: {analysis['configuration']['measurement_runs']}",
        ],
        dependent_variable="median_latency_ms",
        proposed_values=valid_crossovers,
        expected_direction="LOWER_LATENCY",
        estimated_speedup_range="0.8x-1.15x",
        measurement_requirements={
            "warmup_runs": analysis["configuration"]["warmup_runs"],
            "measurement_runs": analysis["configuration"]["measurement_runs"],
            "identical_hardware": True,
            "identical_environment": True,
            "comparison_metric": "median_latency_ms",
            "note": "Memory allocation overhead must be included in timing.",
        },
        acceptance_criteria=(
            "Accept if median latency decreases by >=5% compared to baseline. "
            "Due to Python overhead, this experiment is expected to have marginal or "
            "negative benefit. A negative result is informative."
        ),
        rollback_plan=(
            "Revert to standard triple-nested-loop GEMM. "
            "No system state is modified by this experiment."
        ),
    )


def _generate_measurement_candidate(
    analysis: Dict[str, Any],
    idx: int,
) -> ExperimentCandidate:
    """Generate measurement methodology refinement experiment candidate.

    Tests whether increasing iteration count or warmup reduces variance and
    produces a more stable median estimate.
    """
    dims = analysis["workload"]["dimensions"]
    m, n, k = dims["m"], dims["n"], dims["k"]
    median_ms = analysis["latency"]["median_ms"]
    cv = analysis["latency"]["cv_percent"]
    current_warmup = analysis["configuration"]["warmup_runs"]
    current_iters = analysis["configuration"]["measurement_runs"]

    proposed_configs = []
    proposed_configs.append({
        "warmup_runs": current_warmup,
        "measurement_runs": current_iters * 2,
        "label": f"{current_iters * 2} iterations (2x baseline)",
    })
    proposed_configs.append({
        "warmup_runs": current_warmup * 2,
        "measurement_runs": current_iters,
        "label": f"{current_warmup * 2} warmup iterations (2x baseline)",
    })
    proposed_configs.append({
        "warmup_runs": current_warmup * 2,
        "measurement_runs": current_iters * 2,
        "label": f"{current_warmup * 2} warmup + {current_iters * 2} iterations (2x both)",
    })

    priority = ExperimentPriority.MEDIUM.value if cv > 3.0 else ExperimentPriority.LOW.value

    cv_note = (
        f"CV > 3% suggests measurable noise; additional iterations may help distinguish "
        "real performance differences from measurement artifacts. "
        if cv > 3.0 else
        f"CV is already low ({cv:.1f}%), but this experiment establishes whether "
        "the current methodology is sufficient for detecting small (2-5%) changes. "
    )

    return ExperimentCandidate(
        experiment_id=f"measurement-{idx:03d}",
        category=ExperimentCategory.MEASUREMENT.value,
        title="Measurement Methodology Refinement for Variance Reduction",
        hypothesis=(
            f"Increasing warmup and/or measurement iterations from the current "
            f"{current_warmup}/{current_iters} will reduce the coefficient of "
            f"variation (currently {cv:.1f}%) and produce a more reliable median estimate."
        ),
        description=(
            "The baseline measurement uses a fixed warmup and iteration count. "
            "This experiment tests whether additional warmup (allowing JIT, caches, and "
            "OS scheduling to stabilize) and/or additional measurement iterations "
            "(reducing sampling noise) produce materially different median estimates "
            "and lower variance. This does NOT optimize the workload; it validates "
            "the measurement methodology."
        ),
        rationale=(
            f"Current CV={cv:.1f}% with {current_iters} iterations and {current_warmup} warmup. "
            + cv_note
            + f"Median latency: {median_ms:.1f}ms, stddev: {analysis['latency']['stddev_ms']:.1f}ms."
        ),
        priority=priority,
        risk=ExperimentRisk.LOW.value,
        independent_variable="measurement_configuration",
        control_variables=[
            f"matrix_dimensions: m={m}, n={n}, k={k}",
            f"dtype: {analysis['workload']['dtype']}",
            f"backend: {analysis['workload']['backend']}",
            "algorithm: unchanged (baseline implementation)",
        ],
        dependent_variable="coefficient_of_variation_percent",
        proposed_values=proposed_configs,
        expected_direction="LOWER_VARIANCE",
        estimated_speedup_range="N/A (methodology validation)",
        measurement_requirements={
            "identical_hardware": True,
            "identical_environment": True,
            "comparison_metric": "cv_percent",
            "note": "Compare CV and median stability across configurations, not absolute latency.",
        },
        acceptance_criteria=(
            f"Accept if CV drops below 2% or is reduced by >=30% relative to baseline ({cv:.1f}%). "
            "Report whether median estimate shifts by more than 1 stddev."
        ),
        rollback_plan=(
            "Use whichever measurement configuration provides the best balance of "
            "accuracy and total benchmark duration. No system state is modified."
        ),
    )


# ==============================================================================
# Core Planner Function
# ==============================================================================

def generate_experiment_plan(
    baseline: Union[Dict[str, Any], Any],
) -> ExperimentPlan:
    """Analyze a baseline benchmark result and generate controlled experiment candidates.

    This function:
        1. Accepts a baseline BenchmarkResult (dict or object with .to_dict()).
        2. Extracts and analyzes performance characteristics.
        3. Generates ranked experiment candidates based on observed evidence.
        4. Returns a structured ExperimentPlan -- NEVER executes experiments.

    Args:
        baseline: A baseline BenchmarkResult dict or object. Must contain at minimum:
            - configuration.m, configuration.n, configuration.k (matrix dimensions)
            - latency_ms.median, latency_ms.count (measurement data)

    Returns:
        ExperimentPlan with status, analysis, and ordered candidates.

    Raises:
        TypeError: If baseline cannot be converted to a dictionary.
    """
    # 1. Normalize to dict
    baseline_dict = _to_dict_safe(baseline)
    if baseline_dict is None:
        raise TypeError(
            f"Cannot convert baseline of type {type(baseline).__name__} to dictionary. "
            "Expected a BenchmarkResult instance or dict."
        )

    # 2. Validate minimum required fields
    config = baseline_dict.get("configuration", {})
    latency = baseline_dict.get("latency_ms", {})

    m = config.get("m")
    n = config.get("n")
    k = config.get("k")

    if m is None or n is None or k is None:
        return ExperimentPlan(
            schema_version="1.0",
            status=ExperimentStatus.INSUFFICIENT_BASELINE,
            summary=(
                "Baseline benchmark result is missing required matrix dimension fields "
                "(configuration.m, configuration.n, configuration.k)."
            ),
            baseline_analysis={"error": "missing_dimensions", "raw": baseline_dict},
        )

    median_ms = latency.get("median")
    count = latency.get("count", 0)

    if median_ms is None or count < 1:
        return ExperimentPlan(
            schema_version="1.0",
            status=ExperimentStatus.INSUFFICIENT_BASELINE,
            summary=(
                "Baseline benchmark result has no valid latency measurements. "
                "At least 1 measured iteration with median latency is required."
            ),
            baseline_analysis={"error": "no_measurements", "raw": baseline_dict},
        )

    # 3. Extract structured analysis
    analysis = _extract_baseline_metrics(baseline_dict)

    # 4. Generate experiment candidates
    candidates: List[ExperimentCandidate] = []
    idx = 1

    # 4a. Loop tiling (high priority for large matrices)
    min_dim = min(m, n, k)
    if min_dim >= 16:
        candidates.append(_generate_tiling_candidate(analysis, idx))
        idx += 1

    # 4b. Transpose access pattern
    if k >= 4 and n >= 4:
        candidates.append(_generate_transpose_candidate(analysis, idx))
        idx += 1

    # 4c. Multi-process parallelism
    physical_cores = analysis["hardware"]["physical_cores"]
    if physical_cores > 1 and m >= 4:
        candidates.append(_generate_parallelism_candidate(analysis, idx))
        idx += 1

    # 4d. Strassen algorithm
    if min_dim >= 32:
        candidates.append(_generate_strassen_candidate(analysis, idx))
        idx += 1

    # 4e. Measurement methodology refinement
    candidates.append(_generate_measurement_candidate(analysis, idx))
    idx += 1

    # 5. Determine execution order based on priority and risk
    priority_order = {
        ExperimentPriority.HIGH.value: 0,
        ExperimentPriority.MEDIUM.value: 1,
        ExperimentPriority.LOW.value: 2,
    }
    risk_order = {
        ExperimentRisk.LOW.value: 0,
        ExperimentRisk.MEDIUM.value: 1,
        ExperimentRisk.HIGH.value: 2,
    }

    sorted_candidates = sorted(
        candidates,
        key=lambda c: (priority_order.get(c.priority, 9), risk_order.get(c.risk, 9)),
    )

    execution_order = [c.experiment_id for c in sorted_candidates]

    # 6. Build plan
    if not sorted_candidates:
        return ExperimentPlan(
            schema_version="1.0",
            status=ExperimentStatus.NO_EXPERIMENTS_POSSIBLE,
            summary="No applicable experiment candidates could be generated for this baseline.",
            baseline_analysis=analysis,
        )

    gflops_val = analysis["throughput"]["gflops"]
    if gflops_val is not None:
        summary_text = (
            f"{len(sorted_candidates)} controlled experiment(s) generated for "
            f"GEMM {m}x{n}x{k} {config.get('dtype', 'fp32').upper()} CPU baseline "
            f"(median={median_ms:.1f}ms, {gflops_val:.3f} GFLOPS). "
            f"Experiments are ranked by priority and risk."
        )
    else:
        summary_text = (
            f"{len(sorted_candidates)} controlled experiment(s) generated for "
            f"GEMM {m}x{n}x{k} {config.get('dtype', 'fp32').upper()} CPU baseline "
            f"(median={median_ms:.1f}ms). "
            f"Experiments are ranked by priority and risk."
        )

    return ExperimentPlan(
        schema_version="1.0",
        status=ExperimentStatus.EXPERIMENTS_GENERATED,
        summary=summary_text,
        baseline_analysis=analysis,
        candidates=sorted_candidates,
        execution_order=execution_order,
    )


def load_baseline_from_file(filepath: str) -> Dict[str, Any]:
    """Load a baseline benchmark result from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed benchmark result as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {filepath}")
    with open(path, "r") as f:
        return json.load(f)


# Convenient alias
plan_experiments = generate_experiment_plan


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ROCm Performance Guard -- Experiment Planner V1")
    print("=" * 60)

    # Try to load the latest baseline from experiments/results/
    results_dir = Path("experiments/results")
    baseline_files = sorted(results_dir.glob("*.json")) if results_dir.exists() else []

    if baseline_files:
        latest = baseline_files[-1]
        print(f"Loading baseline: {latest}")
        baseline_data = load_baseline_from_file(str(latest))
    else:
        # Synthetic demo baseline
        print("No baseline found in experiments/results/. Using synthetic demo.")
        baseline_data = {
            "schema_version": "1.0",
            "fingerprint": "demo0000",
            "benchmark": {"name": "gemm_fp32", "version": "1.0.0"},
            "configuration": {
                "workload": "gemm",
                "m": 512, "n": 512, "k": 512,
                "dtype": "fp32", "backend": "cpu",
                "warmup_runs": 5, "measurement_runs": 30,
                "operation_count": 268435456,
            },
            "latency_ms": {
                "count": 30, "min": 1200.0, "mean": 1350.0,
                "median": 1340.0, "p95": 1400.0, "p99": 1430.0,
                "max": 1440.0, "stddev": 40.0,
            },
            "throughput": {
                "operations_per_second": 200000000.0,
                "gflops": 0.200,
                "operations": 268435456,
            },
            "hardware_snapshot": {
                "cpu": {
                    "model_name": "AMD Ryzen 7 7730U with Radeon Graphics",
                    "physical_cores": 8,
                    "logical_cores": 16,
                },
            },
        }

    plan = generate_experiment_plan(baseline_data)

    print(f"\nStatus: {plan.status.value}")
    print(f"Summary: {plan.summary}")
    print(f"Candidates: {len(plan.candidates)}")
    print(f"Execution Order: {plan.execution_order}")
    print()
    print(plan.to_json(indent=2))

    sys.exit(0)
