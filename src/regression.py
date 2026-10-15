"""Regression Detection V1 for ROCm Performance Guard.

Provides deterministic, explainable, and threshold-based performance comparison
between a baseline BenchmarkResult and a current BenchmarkResult.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

# Attempt import from src.benchmark for type hinting, gracefully handling both module/package imports
try:
    from src.benchmark import BenchmarkResult
except ImportError:
    BenchmarkResult = Any  # type: ignore[misc,assignment]


class RegressionStatus(str, Enum):
    """Classification status for a performance comparison."""

    NO_REGRESSION = "NO_REGRESSION"
    POSSIBLE_REGRESSION = "POSSIBLE_REGRESSION"
    REGRESSION = "REGRESSION"
    IMPROVEMENT = "IMPROVEMENT"
    POSSIBLE_IMPROVEMENT = "POSSIBLE_IMPROVEMENT"
    INCOMPARABLE = "INCOMPARABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class MetricDirection(str, Enum):
    """Direction of performance change."""

    BETTER = "BETTER"
    WORSE = "WORSE"
    NEUTRAL = "NEUTRAL"


@dataclass
class RegressionConfig:
    """Configuration thresholds and parameters for regression analysis.

    Attributes:
        metric: Metric to evaluate ("median_latency_ms", "mean_latency_ms",
                "p95_latency_ms", "p99_latency_ms", "throughput").
        possible_regression_threshold_percent: Threshold percentage for possible regression.
        regression_threshold_percent: Threshold percentage for confirmed regression.
        min_measurement_count: Minimum number of measurements required in both runs.
    """

    metric: str = "median_latency_ms"
    possible_regression_threshold_percent: float = 2.0
    regression_threshold_percent: float = 5.0
    min_measurement_count: int = 5

    def __post_init__(self) -> None:
        """Validate threshold parameters."""
        if self.possible_regression_threshold_percent < 0:
            raise ValueError("possible_regression_threshold_percent must be non-negative")
        if self.regression_threshold_percent < self.possible_regression_threshold_percent:
            raise ValueError("regression_threshold_percent must be >= possible_regression_threshold_percent")
        if self.min_measurement_count <= 0:
            raise ValueError("min_measurement_count must be strictly positive (> 0)")


@dataclass
class RegressionResult:
    """Structured, JSON-serializable regression analysis outcome."""

    schema_version: str
    status: RegressionStatus
    metric: Optional[Dict[str, Any]]
    evidence: Dict[str, Any]
    thresholds: Dict[str, float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert regression result to a clean dictionary."""
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "metric": self.metric,
            "evidence": self.evidence,
            "thresholds": self.thresholds,
            "reason": self.reason,
        }

    def to_json(self, indent: int = 2) -> str:
        """Format the result as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _to_result_dict(bm: Union[BenchmarkResult, Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize a BenchmarkResult instance or dictionary to a dictionary."""
    if hasattr(bm, "to_dict") and callable(bm.to_dict):
        return bm.to_dict()
    if isinstance(bm, dict):
        return bm
    raise TypeError(f"Expected BenchmarkResult instance or dict, got {type(bm).__name__}")


def _extract_metric_value(bm_dict: Dict[str, Any], metric_name: str) -> Tuple[Optional[float], bool]:
    """Extract a metric value and determine if higher is better.

    Returns:
        Tuple of (value, higher_is_better).
    """
    normalized = metric_name.lower().strip()
    lat = bm_dict.get("latency_ms", {})
    tp = bm_dict.get("throughput", {})

    if normalized in ("median", "median_latency_ms", "median_ms"):
        val = lat.get("median")
        return (float(val) if val is not None else None, False)

    if normalized in ("mean", "mean_latency_ms", "mean_ms"):
        val = lat.get("mean")
        return (float(val) if val is not None else None, False)

    if normalized in ("p95", "p95_latency_ms", "p95_ms"):
        val = lat.get("p95")
        return (float(val) if val is not None else None, False)

    if normalized in ("p99", "p99_latency_ms", "p99_ms"):
        val = lat.get("p99")
        return (float(val) if val is not None else None, False)

    if normalized in ("throughput", "throughput_ops_per_sec", "operations_per_second"):
        val = tp.get("operations_per_second")
        return (float(val) if val is not None else None, True)

    raise ValueError(f"Unsupported metric: '{metric_name}'")


def compare_benchmarks(
    baseline: Union[BenchmarkResult, Dict[str, Any]],
    current: Union[BenchmarkResult, Dict[str, Any]],
    config: Optional[RegressionConfig] = None,
) -> RegressionResult:
    """Compare a baseline benchmark result with a current benchmark result.

    Args:
        baseline: BenchmarkResult instance or dict from trusted baseline.
        current: BenchmarkResult instance or dict from current run.
        config: Optional RegressionConfig specifying thresholds and metrics.

    Returns:
        RegressionResult classifying the performance change.
    """
    if config is None:
        config = RegressionConfig()

    base_dict = _to_result_dict(baseline)
    curr_dict = _to_result_dict(current)

    thresholds_dict = {
        "possible_percent": config.possible_regression_threshold_percent,
        "regression_percent": config.regression_threshold_percent,
    }

    # Extract metadata for comparability
    base_bm = base_dict.get("benchmark", {})
    curr_bm = curr_dict.get("benchmark", {})
    base_name = base_bm.get("name")
    curr_name = curr_bm.get("name")
    base_ver = base_bm.get("version")
    curr_ver = curr_bm.get("version")
    base_fp = base_dict.get("fingerprint")
    curr_fp = curr_dict.get("fingerprint")

    evidence_dict: Dict[str, Any] = {
        "baseline_measurements": base_dict.get("latency_ms", {}).get("count", 0),
        "current_measurements": curr_dict.get("latency_ms", {}).get("count", 0),
        "baseline_stddev": base_dict.get("latency_ms", {}).get("stddev"),
        "current_stddev": curr_dict.get("latency_ms", {}).get("stddev"),
    }

    # 1. Verify comparability
    if base_name != curr_name:
        return RegressionResult(
            schema_version="1.0",
            status=RegressionStatus.INCOMPARABLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            reason=f"Benchmark names differ: baseline '{base_name}' != current '{curr_name}'.",
        )

    if base_ver != curr_ver:
        return RegressionResult(
            schema_version="1.0",
            status=RegressionStatus.INCOMPARABLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            reason=f"Benchmark versions differ: baseline '{base_ver}' != current '{curr_ver}'.",
        )

    if base_fp and curr_fp and base_fp != curr_fp:
        return RegressionResult(
            schema_version="1.0",
            status=RegressionStatus.INCOMPARABLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            reason=f"Benchmark configuration fingerprints differ: baseline '{base_fp}' != current '{curr_fp}'.",
        )

    # 2. Verify measurement count sufficiency
    base_count = evidence_dict["baseline_measurements"]
    curr_count = evidence_dict["current_measurements"]

    if base_count < config.min_measurement_count or curr_count < config.min_measurement_count:
        return RegressionResult(
            schema_version="1.0",
            status=RegressionStatus.INSUFFICIENT_DATA,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            reason=(
                f"Insufficient measurements for reliable analysis: "
                f"baseline has {base_count}, current has {curr_count} "
                f"(minimum required: {config.min_measurement_count})."
            ),
        )

    # 3. Extract metric values
    base_val, higher_is_better = _extract_metric_value(base_dict, config.metric)
    curr_val, _ = _extract_metric_value(curr_dict, config.metric)

    if base_val is None or curr_val is None:
        missing = "baseline" if base_val is None else "current"
        return RegressionResult(
            schema_version="1.0",
            status=RegressionStatus.INSUFFICIENT_DATA,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            reason=f"Metric '{config.metric}' is missing or unavailable in {missing} benchmark result.",
        )

    if base_val <= 0:
        return RegressionResult(
            schema_version="1.0",
            status=RegressionStatus.INSUFFICIENT_DATA,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            reason=f"Baseline metric '{config.metric}' value is <= 0 ({base_val}); cannot compute percentage change.",
        )

    # 4. Compute changes
    abs_change = curr_val - base_val
    pct_change = ((curr_val - base_val) / base_val) * 100.0

    # Determine direction
    if abs_change == 0.0:
        direction = MetricDirection.NEUTRAL
    elif higher_is_better:
        direction = MetricDirection.BETTER if abs_change > 0 else MetricDirection.WORSE
    else:
        direction = MetricDirection.BETTER if abs_change < 0 else MetricDirection.WORSE

    metric_summary = {
        "name": config.metric,
        "baseline": base_val,
        "current": curr_val,
        "absolute_change": abs_change,
        "percentage_change": pct_change,
        "direction": direction.value,
        "higher_is_better": higher_is_better,
    }

    # 5. Evaluate thresholds and classify
    # Calculate magnitude of performance degradation or improvement
    if higher_is_better:
        # Throughput: drop is degradation, gain is improvement
        degradation_pct = -pct_change if pct_change < 0 else 0.0
        improvement_pct = pct_change if pct_change > 0 else 0.0
    else:
        # Latency: increase is degradation, decrease is improvement
        degradation_pct = pct_change if pct_change > 0 else 0.0
        improvement_pct = -pct_change if pct_change < 0 else 0.0

    t_possible = config.possible_regression_threshold_percent
    t_regression = config.regression_threshold_percent

    if degradation_pct > t_regression:
        status = RegressionStatus.REGRESSION
        reason = (
            f"Performance degradation of {degradation_pct:.2f}% in '{config.metric}' "
            f"exceeded regression threshold ({t_regression:.1f}%)."
        )
    elif degradation_pct > t_possible:
        status = RegressionStatus.POSSIBLE_REGRESSION
        reason = (
            f"Performance degradation of {degradation_pct:.2f}% in '{config.metric}' "
            f"exceeded possible regression threshold ({t_possible:.1f}%)."
        )
    elif improvement_pct > t_regression:
        status = RegressionStatus.IMPROVEMENT
        reason = (
            f"Performance improvement of {improvement_pct:.2f}% in '{config.metric}' "
            f"exceeded improvement threshold ({t_regression:.1f}%)."
        )
    elif improvement_pct > t_possible:
        status = RegressionStatus.POSSIBLE_IMPROVEMENT
        reason = (
            f"Performance improvement of {improvement_pct:.2f}% in '{config.metric}' "
            f"exceeded possible improvement threshold ({t_possible:.1f}%)."
        )
    else:
        status = RegressionStatus.NO_REGRESSION
        reason = (
            f"Observed change of {pct_change:+.2f}% in '{config.metric}' "
            f"is within acceptable variance threshold (±{t_possible:.1f}%)."
        )

    return RegressionResult(
        schema_version="1.0",
        status=status,
        metric=metric_summary,
        evidence=evidence_dict,
        thresholds=thresholds_dict,
        reason=reason,
    )


if __name__ == "__main__":
    # Synthetic demonstration of regression analysis
    demo_baseline = {
        "schema_version": "1.0",
        "fingerprint": "a1b2c3d4e5f60000",
        "benchmark": {"name": "demo_kernel", "version": "1.0.0"},
        "configuration": {"warmup_runs": 5, "measurement_runs": 30, "operation_count": 1000},
        "latency_ms": {
            "count": 30,
            "min": 48.2,
            "mean": 50.1,
            "median": 50.0,
            "p95": 52.4,
            "p99": 54.1,
            "max": 55.0,
            "stddev": 1.4,
        },
        "throughput": {"operations_per_second": 20000.0},
    }

    demo_current = {
        "schema_version": "1.0",
        "fingerprint": "a1b2c3d4e5f60000",
        "benchmark": {"name": "demo_kernel", "version": "1.0.0"},
        "configuration": {"warmup_runs": 5, "measurement_runs": 30, "operation_count": 1000},
        "latency_ms": {
            "count": 30,
            "min": 58.1,
            "mean": 60.3,
            "median": 60.0,
            "p95": 62.8,
            "p99": 64.9,
            "max": 66.0,
            "stddev": 1.8,
        },
        "throughput": {"operations_per_second": 16666.67},
    }

    print("=== ROCm Performance Guard — Regression Detection V1 Demo ===")
    res = compare_benchmarks(demo_baseline, demo_current)
    print(res.to_json(indent=2))