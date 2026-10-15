"""Benchmark Engine V1 for ROCm Performance Guard.

Provides reliable, reproducible performance timing, matrix multiplication (GEMM)
workloads, statistical analysis, and hardware/environment snapshotting.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import statistics
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class BenchmarkError(Exception):
    """Base exception for benchmark engine errors."""


class BenchmarkExecutionError(BenchmarkError):
    """Raised when a benchmark workload fails during execution."""


class BackendUnavailableError(BenchmarkError):
    """Raised when the requested execution backend is not available on this system."""


def calculate_percentile(sorted_values: List[float], percentile: float) -> float:
    """Calculate the p-th percentile of a sorted dataset using linear interpolation.

    Method:
        Uses the standard NIST / Type 7 linear interpolation method (equivalent to
        numpy.percentile with method='linear' and Excel PERCENTILE.INC).
        For a sorted dataset of length N:
            index = (percentile / 100.0) * (N - 1)
            lower = int(index)
            upper = min(lower + 1, N - 1)
            weight = index - lower
            result = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    Args:
        sorted_values: Ascending sorted list of numerical floats.
        percentile: Float between 0.0 and 100.0.

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If sorted_values is empty or percentile is outside [0.0, 100.0].
    """
    if not sorted_values:
        raise ValueError("Cannot calculate percentile on an empty dataset")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"Percentile must be between 0.0 and 100.0, got {percentile}")

    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    index = (percentile / 100.0) * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    weight = index - lower

    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def generate_deterministic_matrix(rows: int, cols: int, seed_offset: int = 1, dtype: str = "fp32") -> List[List[float]]:
    """Generate a deterministic matrix with bounded float values.

    Does not use random or uncontrolled inputs.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        seed_offset: Prime offset to generate distinct matrices A and B.
        dtype: Data type representation ("fp32" or "float32").

    Returns:
        2D list of shape rows x cols with deterministic floating point values in [0.0, 1.0].
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Matrix dimensions must be positive integers, got rows={rows}, cols={cols}")

    matrix: List[List[float]] = []
    p1 = 17 if seed_offset == 1 else 23
    p2 = 31 if seed_offset == 1 else 41
    mod = 1000

    for i in range(rows):
        row = [float((i * p1 + j * p2 + seed_offset) % mod) / float(mod) for j in range(cols)]
        matrix.append(row)

    return matrix


def _compute_chunk_worker(chunk: List[List[float]], b_transposed: List[List[float]]) -> List[List[float]]:
    """Worker function for multiprocessing pool chunk execution."""
    mul = operator.mul
    result: List[List[float]] = []
    for a_row in chunk:
        result.append([sum(map(mul, a_row, b_col)) for b_col in b_transposed])
    return result


class BaseGemmBackend(ABC):
    """Abstract interface for GEMM execution backends."""

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is confirmed usable on the host environment."""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return structured capability and hardware status."""

    @abstractmethod
    def prepare(self, m: int, n: int, k: int, dtype: str = "fp32") -> None:
        """Prepare matrices and execution context before benchmarking."""

    @abstractmethod
    def execute(self) -> Any:
        """Execute one iteration of matrix multiplication C = A @ B."""

    def cleanup(self) -> None:
        """Release any background workers or hardware allocations."""


class CpuGemmBackend(BaseGemmBackend):
    """Pure Python CPU backend for GEMM execution.

    Supports single-worker and multi-process chunked evaluation without external dependencies.
    """

    name = "cpu"

    def __init__(self, workers: Optional[int] = None) -> None:
        self.requested_workers = workers
        self.workers: int = 1
        self.m: int = 0
        self.n: int = 0
        self.k: int = 0
        self.dtype: str = "fp32"
        self.a: Optional[List[List[float]]] = None
        self.b_t: Optional[List[List[float]]] = None
        self.pool: Optional[Pool] = None
        self.chunks: Optional[List[List[List[float]]]] = None

    def is_available(self) -> bool:
        return True

    def get_status(self) -> Dict[str, Any]:
        cpu_count = os.cpu_count() or 1
        return {
            "status": "AVAILABLE",
            "backend": "cpu",
            "device": "CPU",
            "available_cores": cpu_count,
            "configured_workers": self.workers,
        }

    def prepare(self, m: int, n: int, k: int, dtype: str = "fp32") -> None:
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype

        if self.requested_workers is not None:
            self.workers = max(1, self.requested_workers)
        else:
            # For small matrices (e.g. tests M < 64), single-worker is fastest
            # For standard benchmark (M >= 64), use multi-worker
            total_cores = os.cpu_count() or 1
            self.workers = min(8, total_cores) if m >= 64 else 1

        self.a = generate_deterministic_matrix(m, k, seed_offset=1, dtype=dtype)
        b = generate_deterministic_matrix(k, n, seed_offset=2, dtype=dtype)
        self.b_t = [[b[p][j] for p in range(k)] for j in range(n)]

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
        if self.a is None or self.b_t is None:
            raise BenchmarkExecutionError("Backend not prepared before execution.")

        if self.pool is not None and self.chunks is not None:
            items = [(chunk, self.b_t) for chunk in self.chunks]
            chunk_results = self.pool.starmap(_compute_chunk_worker, items)
            flattened: List[List[float]] = []
            for sub in chunk_results:
                flattened.extend(sub)
            return flattened
        else:
            mul = operator.mul
            return [[sum(map(mul, a_row, b_col)) for b_col in self.b_t] for a_row in self.a]

    def cleanup(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


class RocmGemmBackend(BaseGemmBackend):
    """ROCm/HIP GPU acceleration backend.

    Defensively inspects system capabilities and reports UNAVAILABLE when ROCm compute is absent.
    """

    name = "rocm"

    def is_available(self) -> bool:
        # Inspect ROCm device node and tooling availability
        kfd_exists = os.path.exists("/dev/kfd")
        import shutil
        has_tools = bool(shutil.which("rocminfo") or shutil.which("hipcc"))
        return kfd_exists and has_tools

    def get_status(self) -> Dict[str, Any]:
        avail = self.is_available()
        return {
            "status": "AVAILABLE" if avail else "UNAVAILABLE",
            "backend": "rocm",
            "device": "AMD ROCm / HIP GPU",
            "reason": None if avail else "ROCm tools (rocminfo, hipcc) or /dev/kfd compute interface not available on host/WSL.",
        }

    def prepare(self, m: int, n: int, k: int, dtype: str = "fp32") -> None:
        if not self.is_available():
            raise BackendUnavailableError(
                "ROCm/HIP backend is UNAVAILABLE on this host environment. "
                "No ROCm compute devices or tools detected."
            )
        raise NotImplementedError("ROCm GPU execution requires confirmed ROCm device support.")

    def execute(self) -> Any:
        raise BackendUnavailableError("ROCm/HIP backend is UNAVAILABLE on this host environment.")


def get_backend(name: str = "cpu", workers: Optional[int] = None) -> BaseGemmBackend:
    """Factory creating configured GEMM backend instances."""
    norm = name.lower().strip()
    if norm in ("cpu", "default"):
        return CpuGemmBackend(workers=workers)
    if norm in ("rocm", "hip"):
        return RocmGemmBackend()
    raise ValueError(f"Unknown benchmark backend: '{name}'. Supported backends: 'cpu', 'rocm'")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        name: Unique identifier for the benchmark.
        version: Version string for the benchmark workload definition.
        warmup_runs: Number of untimed initial runs to eliminate cold-start effects.
        measurement_runs: Number of timed iterations to collect.
        operation_count: Optional count of logical operations performed per run (for throughput).
        m: Matrix rows for A and C.
        n: Matrix columns for B and C.
        k: Inner matrix dimension (columns of A, rows of B).
        dtype: Numerical data type ("fp32", "float32").
        backend: Execution backend identifier ("cpu", "rocm").
        workers: Optional worker count for CPU multiprocessing.
        metadata: Optional dictionary of arbitrary workload/run descriptors.
    """

    name: str = "gemm_fp32"
    version: str = "1.0.0"
    warmup_runs: int = 5
    measurement_runs: int = 30
    operation_count: Optional[int] = None
    m: int = 512
    n: int = 512
    k: int = 512
    dtype: str = "fp32"
    backend: str = "cpu"
    workers: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters and derive default values."""
        if not self.name or not self.name.strip():
            raise ValueError("Benchmark name must not be empty")
        if not self.version or not self.version.strip():
            raise ValueError("Benchmark version must not be empty")
        if self.warmup_runs < 0:
            raise ValueError(f"warmup_runs must be non-negative (>= 0), got {self.warmup_runs}")
        if self.measurement_runs <= 0:
            raise ValueError(f"measurement_runs must be strictly positive (> 0), got {self.measurement_runs}")
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError(f"Matrix dimensions M, N, K must be positive integers, got M={self.m}, N={self.n}, K={self.k}")

        norm_dtype = self.dtype.lower().strip()
        if norm_dtype not in ("fp32", "float32"):
            raise ValueError(f"Unsupported dtype '{self.dtype}'. Initially supported: 'fp32'")
        self.dtype = norm_dtype

        norm_backend = self.backend.lower().strip()
        if norm_backend not in ("cpu", "rocm", "hip"):
            raise ValueError(f"Unsupported backend '{self.backend}'. Supported: 'cpu', 'rocm'")
        self.backend = norm_backend

        # Derive operation count for GEMM if not explicitly set
        # 2 * M * N * K FLOPs
        if self.operation_count is None:
            self.operation_count = 2 * self.m * self.n * self.k
        elif self.operation_count <= 0:
            raise ValueError(f"operation_count must be strictly positive (> 0) if provided, got {self.operation_count}")

    def fingerprint(self) -> str:
        """Generate a deterministic fingerprint hash based solely on configuration parameters."""
        canonical = {
            "name": self.name.strip(),
            "version": self.version.strip(),
            "warmup_runs": self.warmup_runs,
            "measurement_runs": self.measurement_runs,
            "operation_count": self.operation_count,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "dtype": self.dtype,
            "backend": self.backend,
        }
        encoded = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class BenchmarkResult:
    """Structured, JSON-serializable benchmark execution result."""

    schema_version: str
    fingerprint: str
    benchmark: Dict[str, str]
    configuration: Dict[str, Any]
    latency_ms: Dict[str, Any]
    throughput: Dict[str, Any]
    raw_durations_ns: List[int] = field(default_factory=list, repr=False)
    hardware_snapshot: Optional[Dict[str, Any]] = None
    environment_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to a clean, serializable dictionary."""
        data: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "benchmark": self.benchmark,
            "configuration": self.configuration,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
        }
        if self.hardware_snapshot is not None:
            data["hardware_snapshot"] = self.hardware_snapshot
        if self.environment_snapshot is not None:
            data["environment_snapshot"] = self.environment_snapshot
        return data

    def to_json(self, indent: int = 2) -> str:
        """Format the result as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, directory: str = "experiments/results") -> str:
        """Save benchmark result to JSON file with deterministic timestamped filename."""
        return save_benchmark_result(self, directory=directory)


def save_benchmark_result(result: BenchmarkResult, directory: str = "experiments/results") -> str:
    """Persist benchmark result as JSON without overwriting existing files."""
    path_dir = Path(directory)
    path_dir.mkdir(parents=True, exist_ok=True)

    cfg = result.configuration
    m = cfg.get("m", 512)
    n = cfg.get("n", 512)
    k = cfg.get("k", 512)
    dtype = cfg.get("dtype", "fp32")
    backend = cfg.get("backend", "cpu")

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"gemm_{backend}_{m}x{n}x{k}_{dtype}_{ts_str}"
    target_path = path_dir / f"{base_name}.json"

    counter = 1
    while target_path.exists():
        target_path = path_dir / f"{base_name}_{counter}.json"
        counter += 1

    target_path.write_text(result.to_json(indent=2), encoding="utf-8")
    return str(target_path)


class BenchmarkRunner:
    """Coordinates execution of warmup iterations, timing, and metric aggregation."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(
        self,
        workload: Optional[Callable[[], Any]] = None,
        backend: Optional[BaseGemmBackend] = None,
        hardware_snapshot: Optional[Dict[str, Any]] = None,
        environment_snapshot: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Execute benchmark using either callable workload or configured GEMM backend."""
        # Setup workload callable
        exec_fn: Callable[[], Any]
        active_backend: Optional[BaseGemmBackend] = None

        if workload is not None:
            exec_fn = workload
        else:
            active_backend = backend or get_backend(self.config.backend, workers=self.config.workers)
            if not active_backend.is_available():
                status = active_backend.get_status()
                raise BackendUnavailableError(
                    f"Backend '{active_backend.name}' is UNAVAILABLE: {status.get('reason')}"
                )
            active_backend.prepare(
                m=self.config.m,
                n=self.config.n,
                k=self.config.k,
                dtype=self.config.dtype,
            )
            exec_fn = active_backend.execute

        try:
            # 1. Warmup phase (untimed, discarded)
            for idx in range(self.config.warmup_runs):
                try:
                    exec_fn()
                except Exception as err:
                    raise BenchmarkExecutionError(
                        f"Workload failed during warmup run {idx + 1}/{self.config.warmup_runs}: {err}"
                    ) from err

            # 2. Measurement phase (nanosecond precision)
            raw_durations_ns: List[int] = []
            for idx in range(self.config.measurement_runs):
                start_ns = time.perf_counter_ns()
                try:
                    exec_fn()
                except Exception as err:
                    raise BenchmarkExecutionError(
                        f"Workload failed during measurement run {idx + 1}/{self.config.measurement_runs}: {err}"
                    ) from err
                end_ns = time.perf_counter_ns()
                raw_durations_ns.append(end_ns - start_ns)
        finally:
            if active_backend is not None:
                active_backend.cleanup()

        # 3. Statistical calculations in milliseconds
        durations_ms = [ns / 1_000_000.0 for ns in raw_durations_ns]
        sorted_ms = sorted(durations_ms)
        count = len(durations_ms)

        min_ms = min(durations_ms)
        max_ms = max(durations_ms)
        mean_ms = statistics.mean(durations_ms)
        median_ms = statistics.median(durations_ms)
        stddev_ms = statistics.stdev(durations_ms) if count >= 2 else 0.0
        p95_ms = calculate_percentile(sorted_ms, 95.0)
        p99_ms = calculate_percentile(sorted_ms, 99.0)

        # 4. Throughput calculation based on median latency
        ops_per_second: Optional[float] = None
        gflops: Optional[float] = None

        if self.config.operation_count is not None:
            median_seconds = median_ms / 1000.0
            if median_seconds > 0:
                ops_per_second = float(self.config.operation_count) / median_seconds
                gflops = ops_per_second / 1e9

        latency_summary = {
            "count": count,
            "min": min_ms,
            "mean": mean_ms,
            "median": median_ms,
            "p95": p95_ms,
            "p99": p99_ms,
            "max": max_ms,
            "stddev": stddev_ms,
        }

        throughput_summary: Dict[str, Any] = {
            "operations_per_second": ops_per_second,
            "gflops": gflops,
            "operations": self.config.operation_count,
        }

        metadata = dict(self.config.metadata)
        metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        metadata.setdefault("matrix_dimensions", {"m": self.config.m, "n": self.config.n, "k": self.config.k})
        metadata.setdefault("dtype", self.config.dtype)
        metadata.setdefault("backend", self.config.backend)

        return BenchmarkResult(
            schema_version="1.0",
            fingerprint=self.config.fingerprint(),
            benchmark={
                "name": self.config.name,
                "version": self.config.version,
            },
            configuration={
                "workload": "gemm",
                "m": self.config.m,
                "n": self.config.n,
                "k": self.config.k,
                "dtype": self.config.dtype,
                "backend": self.config.backend,
                "warmup_runs": self.config.warmup_runs,
                "measurement_runs": self.config.measurement_runs,
                "operation_count": self.config.operation_count,
                "metadata": metadata,
            },
            latency_ms=latency_summary,
            throughput=throughput_summary,
            raw_durations_ns=raw_durations_ns,
            hardware_snapshot=hardware_snapshot,
            environment_snapshot=environment_snapshot,
        )


def demo_cpu_workload() -> int:
    """Deterministic pure-Python CPU demonstration workload (sum of squares)."""
    total = 0
    for i in range(10_000):
        total += i * i
    return total


def run_gemm_benchmark(
    m: int = 512,
    n: int = 512,
    k: int = 512,
    warmup: int = 5,
    iterations: int = 30,
    dtype: str = "fp32",
    backend: str = "cpu",
    workers: Optional[int] = None,
    save_result: bool = True,
    output_dir: str = "experiments/results",
) -> Tuple[BenchmarkResult, Optional[str]]:
    """Run production GEMM benchmark with optional snapshots and file persistence."""
    # Attempt automatic snapshot collection from existing modules
    hw_snap = None
    env_snap = None
    try:
        from src.hardware import get_hardware_info
        hw_snap = get_hardware_info()
    except Exception:
        pass

    try:
        from src.environment import get_environment_info
        env_snap = get_environment_info()
    except Exception:
        pass

    config = BenchmarkConfig(
        name=f"gemm_{dtype.lower()}",
        version="1.0.0",
        warmup_runs=warmup,
        measurement_runs=iterations,
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        backend=backend,
        workers=workers,
    )

    runner = BenchmarkRunner(config)
    result = runner.run(
        hardware_snapshot=hw_snap,
        environment_snapshot=env_snap,
    )

    saved_path: Optional[str] = None
    if save_result:
        saved_path = save_benchmark_result(result, directory=output_dir)

    return result, saved_path


def print_benchmark_summary(result: BenchmarkResult, saved_path: Optional[str] = None) -> None:
    """Print human-readable benchmark summary followed by full machine-readable JSON."""
    cfg = result.configuration
    m = cfg.get("m", 512)
    n = cfg.get("n", 512)
    k = cfg.get("k", 512)
    dtype = cfg.get("dtype", "fp32").upper()
    backend = cfg.get("backend", "cpu").upper()
    lat = result.latency_ms
    tp = result.throughput

    ops = tp.get("operations") or (2 * m * n * k)
    gflops = tp.get("gflops")
    gflops_str = f"{gflops:.3f} GFLOPS" if gflops is not None else "N/A"

    print("=" * 60)
    print("ROCm Performance Guard - GEMM Benchmark V1")
    print("=" * 60)
    print(f"Workload           : Matrix Multiplication (C = A @ B)")
    print(f"Matrix Dimensions  : M={m}, N={n}, K={k}")
    print(f"Data Type          : {dtype}")
    print(f"Backend            : {backend}")
    print(f"Iterations         : {lat.get('count')} measured ({cfg.get('warmup_runs')} warmup)")
    print("-" * 60)
    print("Latency Statistics :")
    print(f"  Median Latency   : {lat.get('median'):.3f} ms")
    print(f"  Mean Latency     : {lat.get('mean'):.3f} ms")
    print(f"  Min Latency      : {lat.get('min'):.3f} ms")
    print(f"  Max Latency      : {lat.get('max'):.3f} ms")
    print(f"  p95 Latency      : {lat.get('p95'):.3f} ms")
    print(f"  p99 Latency      : {lat.get('p99'):.3f} ms")
    print(f"  Std Deviation    : {lat.get('stddev'):.3f} ms")
    print("Throughput :")
    print(f"  FLOPs / Run      : {ops:,} FLOPs (2 * M * N * K)")
    print(f"  Throughput       : {gflops_str} (median-based)")
    if saved_path:
        print(f"Saved Result       : {saved_path}")
    print("=" * 60)
    print("Benchmark JSON:")
    print(result.to_json(indent=2))


def main() -> int:
    """CLI entry point for Benchmark Engine V1."""
    parser = argparse.ArgumentParser(
        description="ROCm Performance Guard ? Production GEMM Benchmark V1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--m", type=int, default=512, help="Matrix rows of A and C")
    parser.add_argument("--n", type=int, default=512, help="Matrix columns of B and C")
    parser.add_argument("--k", type=int, default=512, help="Inner dimension (cols of A, rows of B)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of untimed warmup iterations")
    parser.add_argument("--iterations", type=int, default=30, help="Number of timed measurement iterations")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "float32"], help="Numerical data precision")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "rocm", "hip"], help="Execution backend")
    parser.add_argument("--workers", type=int, default=None, help="Worker count for CPU execution (default: auto)")
    parser.add_argument("--no-save", action="store_true", help="Disable saving result to experiments/results/")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Directory to save benchmark results")

    args = parser.parse_args()

    # Verify backend availability before running
    target_backend = get_backend(args.backend, workers=args.workers)
    if not target_backend.is_available():
        status = target_backend.get_status()
        print("=" * 60, file=sys.stderr)
        print(f"ERROR: Backend '{args.backend}' is UNAVAILABLE", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Reason: {status.get('reason')}", file=sys.stderr)
        print("\nStructured Status:", file=sys.stderr)
        print(json.dumps(status, indent=2), file=sys.stderr)
        return 1

    result, saved_path = run_gemm_benchmark(
        m=args.m,
        n=args.n,
        k=args.k,
        warmup=args.warmup,
        iterations=args.iterations,
        dtype=args.dtype,
        backend=args.backend,
        workers=args.workers,
        save_result=not args.no_save,
        output_dir=args.output_dir,
    )

    print_benchmark_summary(result, saved_path=saved_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
