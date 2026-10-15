"""Verification Engine V1 for ROCm Performance Guard.

Evaluates whether a candidate optimization experiment produces an empirical performance
improvement when evaluated against a known-good baseline under identical benchmark methodology.
Does not mutate environment or persist system changes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from src.benchmark import BenchmarkResult
except ImportError:
    BenchmarkResult = Any  # type: ignore[misc,assignment]

try:
    from src.optimizer import OptimizationCandidate
except ImportError:
    OptimizationCandidate = Any  # type: ignore[misc,assignment]


class VerificationStatus(str, Enum):
    """Classification status for optimization experiment verification."""

    IMPROVED = "IMPROVED"
    NO_MEANINGFUL_CHANGE = "NO_MEANINGFUL_CHANGE"
    REGRESSED = "REGRESSED"
    INCOMPARABLE = "INCOMPARABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NO_VERIFICATION_POSSIBLE = "NO_VERIFICATION_POSSIBLE"


class MetricDirection(str, Enum):
    """Direction of performance change."""

    BETTER = "BETTER"
    WORSE = "WORSE"
    NEUTRAL = "NEUTRAL"


DEFAULT_VERIFICATION_LIMITATIONS: List[str] = [
    "Threshold-based verification does not establish statistical significance.",
    "Verification compares empirical benchmark samples; results are subject to system noise and variance.",
    "Evaluations do not apply, modify, or persist system or environment changes automatically.",
    "Verification evidence indicates observed performance delta; it does not establish definitive causality.",
]


@dataclass
class VerificationConfig:
    """Configuration thresholds and parameters for optimization verification.

    Engineering heuristics:
        Thresholds are heuristic indicators for decision-making and do NOT
        guarantee statistical significance or causal relationships.

    Attributes:
        metric: Primary metric to evaluate (e.g. "median_latency_ms", "mean_latency_ms",
                "p95_latency_ms", "p99_latency_ms", "throughput").
        improvement_threshold_percent: Minimum percentage change in desirable direction to declare IMPROVED.
        regression_threshold_percent: Percentage change in undesirable direction to declare REGRESSED.
        minimum_measurements: Minimum number of measurement samples required in both baseline and candidate.
    """

    metric: str = "median_latency_ms"
    improvement_threshold_percent: float = 5.0
    regression_threshold_percent: float = 5.0
    minimum_measurements: int = 5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.metric or not str(self.metric).strip():
            raise ValueError("Metric name must not be empty")
        if self.improvement_threshold_percent < 0:
            raise ValueError(
                f"improvement_threshold_percent must be non-negative (>= 0), got {self.improvement_threshold_percent}"
            )
        if self.regression_threshold_percent < 0:
            raise ValueError(
                f"regression_threshold_percent must be non-negative (>= 0), got {self.regression_threshold_percent}"
            )
        if self.minimum_measurements <= 0:
            raise ValueError(
                f"minimum_measurements must be strictly positive (> 0), got {self.minimum_measurements}"
            )


@dataclass
class VerificationResult:
    """Structured, JSON-serializable outcome of an optimization verification evaluation."""

    schema_version: str
    status: VerificationStatus
    metric: Optional[Dict[str, Any]]
    evidence: Dict[str, Any]
    thresholds: Dict[str, float]
    explanation: str
    candidate_id: Optional[str] = None
    limitations: List[str] = field(default_factory=lambda: list(DEFAULT_VERIFICATION_LIMITATIONS))

    def to_dict(self) -> Dict[str, Any]:
        """Convert verification result to a clean dictionary."""
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status.value if isinstance(self.status, VerificationStatus) else str(self.status),
            "metric": self.metric,
            "evidence": self.evidence,
            "thresholds": self.thresholds,
            "explanation": self.explanation,
        }
        if self.candidate_id is not None:
            d["candidate_id"] = self.candidate_id
        d["limitations"] = list(self.limitations)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Format the result as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _to_result_dict(bm: Union[BenchmarkResult, Dict[str, Any], None]) -> Optional[Dict[str, Any]]:
    """Normalize a BenchmarkResult instance or dictionary to a dictionary."""
    if bm is None:
        return None
    if hasattr(bm, "to_dict") and callable(bm.to_dict):
        return bm.to_dict()
    if isinstance(bm, dict):
        return bm
    raise TypeError(f"Expected BenchmarkResult instance, dict, or None, got {type(bm).__name__}")


def _extract_candidate_id(
    candidate_info: Optional[Union[OptimizationCandidate, Dict[str, Any], str]]
) -> Optional[str]:
    """Safely extract candidate_id from diverse candidate metadata sources."""
    if candidate_info is None:
        return None
    if isinstance(candidate_info, str):
        cid = candidate_info.strip()
        return cid if cid else None
    if hasattr(candidate_info, "candidate_id"):
        cid_val = getattr(candidate_info, "candidate_id")
        return str(cid_val).strip() if cid_val else None
    if isinstance(candidate_info, dict):
        cid_val = candidate_info.get("candidate_id")
        return str(cid_val).strip() if cid_val else None
    return None


def _extract_metric_value(bm_dict: Dict[str, Any], metric_name: str) -> Tuple[Optional[float], bool, str]:
    """Extract a metric value, direction rule, and human-readable label.

    Returns:
        Tuple of (value, higher_is_better, display_name).

    Raises:
        KeyError: If metric_name is unrecognized.
    """
    normalized = metric_name.lower().strip()
    lat = bm_dict.get("latency_ms", {}) if isinstance(bm_dict.get("latency_ms"), dict) else {}
    tp = bm_dict.get("throughput", {}) if isinstance(bm_dict.get("throughput"), dict) else {}

    if normalized in ("median", "median_latency_ms", "median_ms"):
        val = lat.get("median")
        return (float(val) if val is not None else None, False, "median latency")

    if normalized in ("mean", "mean_latency_ms", "mean_ms"):
        val = lat.get("mean")
        return (float(val) if val is not None else None, False, "mean latency")

    if normalized in ("p95", "p95_latency_ms", "p95_ms"):
        val = lat.get("p95")
        return (float(val) if val is not None else None, False, "p95 latency")

    if normalized in ("p99", "p99_latency_ms", "p99_ms"):
        val = lat.get("p99")
        return (float(val) if val is not None else None, False, "p99 latency")

    if normalized in ("throughput", "throughput_ops_per_sec", "operations_per_second"):
        val = tp.get("operations_per_second")
        return (float(val) if val is not None else None, True, "throughput")

    raise KeyError(f"Unsupported metric: '{metric_name}'")


def verify_optimization(
    baseline: Union[BenchmarkResult, Dict[str, Any], None],
    candidate: Union[BenchmarkResult, Dict[str, Any], None],
    candidate_info: Optional[Union[OptimizationCandidate, Dict[str, Any], str]] = None,
    config: Optional[VerificationConfig] = None,
) -> VerificationResult:
    """Verify whether a candidate optimization experiment produces an empirical improvement.

    Compares a candidate experiment against a known-good baseline using the same benchmark methodology.
    Does not apply changes or mutate environment.

    Args:
        baseline: BenchmarkResult or dict from the trusted baseline run.
        candidate: BenchmarkResult or dict from the candidate experiment run.
        candidate_info: Optional OptimizationCandidate, candidate dict, or candidate_id string.
        config: Optional VerificationConfig specifying thresholds and metrics.

    Returns:
        VerificationResult with classification, metric comparison, evidence, and explanation.
    """
    if config is None:
        config = VerificationConfig()

    candidate_id = _extract_candidate_id(candidate_info)

    thresholds_dict = {
        "improvement_percent": float(config.improvement_threshold_percent),
        "regression_percent": float(config.regression_threshold_percent),
    }

    # 1. Validate benchmark inputs presence
    if baseline is None or candidate is None:
        missing_party = (
            "Baseline and candidate"
            if baseline is None and candidate is None
            else "Baseline"
            if baseline is None
            else "Candidate"
        )
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence={
                "baseline_measurements": 0,
                "candidate_measurements": 0,
                "baseline_stddev": None,
                "candidate_stddev": None,
            },
            thresholds=thresholds_dict,
            explanation=f"{missing_party} benchmark result is missing or None; verification cannot proceed.",
            candidate_id=candidate_id,
        )

    base_dict = _to_result_dict(baseline)
    cand_dict = _to_result_dict(candidate)

    if base_dict is None or cand_dict is None:
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence={
                "baseline_measurements": 0,
                "candidate_measurements": 0,
                "baseline_stddev": None,
                "candidate_stddev": None,
            },
            thresholds=thresholds_dict,
            explanation="Invalid or empty benchmark dictionary supplied.",
            candidate_id=candidate_id,
        )

    # Extract metadata and measurements
    base_bm = base_dict.get("benchmark", {}) if isinstance(base_dict.get("benchmark"), dict) else {}
    cand_bm = cand_dict.get("benchmark", {}) if isinstance(cand_dict.get("benchmark"), dict) else {}

    base_name = base_bm.get("name")
    cand_name = cand_bm.get("name")
    base_ver = base_bm.get("version")
    cand_ver = cand_bm.get("version")
    base_fp = base_dict.get("fingerprint")
    cand_fp = cand_dict.get("fingerprint")

    base_lat = base_dict.get("latency_ms", {}) if isinstance(base_dict.get("latency_ms"), dict) else {}
    cand_lat = cand_dict.get("latency_ms", {}) if isinstance(cand_dict.get("latency_ms"), dict) else {}

    base_count = base_lat.get("count", 0)
    cand_count = cand_lat.get("count", 0)

    # Fallback to raw_durations_ns length if count not set in latency_ms
    if not base_count and isinstance(base_dict.get("raw_durations_ns"), list):
        base_count = len(base_dict["raw_durations_ns"])
    if not cand_count and isinstance(cand_dict.get("raw_durations_ns"), list):
        cand_count = len(cand_dict["raw_durations_ns"])

    base_stddev = base_lat.get("stddev")
    cand_stddev = cand_lat.get("stddev")

    evidence_dict: Dict[str, Any] = {
        "baseline_measurements": int(base_count) if isinstance(base_count, (int, float)) else 0,
        "candidate_measurements": int(cand_count) if isinstance(cand_count, (int, float)) else 0,
        "baseline_stddev": base_stddev,
        "candidate_stddev": cand_stddev,
    }

    # 2. Comparability verification
    if base_name != cand_name or not base_name:
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.INCOMPARABLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Benchmark names differ: baseline '{base_name}' != candidate '{cand_name}'. Workloads are not comparable.",
            candidate_id=candidate_id,
        )

    if base_ver != cand_ver or not base_ver:
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.INCOMPARABLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Benchmark versions differ: baseline '{base_ver}' != candidate '{cand_ver}'. Workloads are not comparable.",
            candidate_id=candidate_id,
        )

    if base_fp and cand_fp and base_fp != cand_fp:
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.INCOMPARABLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=(
                f"Benchmark configuration fingerprints differ: baseline '{base_fp}' != candidate '{cand_fp}'. "
                "Candidate workload configuration differs from baseline; improvement cannot be attributed to an optimization experiment."
            ),
            candidate_id=candidate_id,
        )

    # 3. Measurement sufficiency verification
    if evidence_dict["baseline_measurements"] < config.minimum_measurements:
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.INSUFFICIENT_DATA,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=(
                f"Baseline contains only {evidence_dict['baseline_measurements']} measurement(s); "
                f"at least {config.minimum_measurements} are required."
            ),
            candidate_id=candidate_id,
        )

    if evidence_dict["candidate_measurements"] < config.minimum_measurements:
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.INSUFFICIENT_DATA,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=(
                f"Candidate contains only {evidence_dict['candidate_measurements']} measurement(s); "
                f"at least {config.minimum_measurements} are required."
            ),
            candidate_id=candidate_id,
        )

    # 4. Metric extraction and validation
    try:
        raw_base_val, higher_is_better, display_name = _extract_metric_value(base_dict, config.metric)
        raw_cand_val, _, _ = _extract_metric_value(cand_dict, config.metric)
    except KeyError:
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Requested metric '{config.metric}' is not supported.",
            candidate_id=candidate_id,
        )

    if raw_base_val is None or raw_cand_val is None:
        missing = "baseline" if raw_base_val is None else "candidate"
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Requested metric '{config.metric}' is missing or None in {missing} benchmark result.",
            candidate_id=candidate_id,
        )

    try:
        base_val = float(raw_base_val)
        cand_val = float(raw_cand_val)
    except (TypeError, ValueError):
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Metric '{config.metric}' contains non-numeric values.",
            candidate_id=candidate_id,
        )

    if math.isnan(base_val) or math.isinf(base_val):
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Baseline metric '{config.metric}' has invalid value ({base_val}).",
            candidate_id=candidate_id,
        )

    if math.isnan(cand_val) or math.isinf(cand_val):
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Candidate metric '{config.metric}' has invalid value ({cand_val}).",
            candidate_id=candidate_id,
        )

    if not higher_is_better and (base_val < 0.0 or cand_val < 0.0):
        neg_val = base_val if base_val < 0.0 else cand_val
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Negative latency value ({neg_val} ms) is invalid.",
            candidate_id=candidate_id,
        )

    if higher_is_better and (base_val < 0.0 or cand_val < 0.0):
        neg_val = base_val if base_val < 0.0 else cand_val
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Negative throughput value ({neg_val}) is invalid.",
            candidate_id=candidate_id,
        )

    if base_val == 0.0:
        return VerificationResult(
            schema_version="1.0",
            status=VerificationStatus.NO_VERIFICATION_POSSIBLE,
            metric=None,
            evidence=evidence_dict,
            thresholds=thresholds_dict,
            explanation=f"Baseline metric '{config.metric}' is zero; cannot compute percentage change without division by zero.",
            candidate_id=candidate_id,
        )

    # 5. Delta and percentage calculations
    abs_change = cand_val - base_val
    pct_change = ((cand_val - base_val) / base_val) * 100.0

    if abs_change == 0.0:
        direction = MetricDirection.NEUTRAL
    elif higher_is_better:
        direction = MetricDirection.BETTER if abs_change > 0 else MetricDirection.WORSE
    else:
        direction = MetricDirection.BETTER if abs_change < 0 else MetricDirection.WORSE

    metric_summary: Dict[str, Any] = {
        "name": config.metric,
        "baseline": base_val,
        "candidate": cand_val,
        "absolute_change": abs_change,
        "percentage_change": pct_change,
        "direction": direction.value,
    }

    # Improvement vs degradation percentage
    if higher_is_better:
        improvement_pct = pct_change if pct_change > 0.0 else 0.0
        degradation_pct = -pct_change if pct_change < 0.0 else 0.0
    else:
        improvement_pct = -pct_change if pct_change < 0.0 else 0.0
        degradation_pct = pct_change if pct_change > 0.0 else 0.0

    t_improve = config.improvement_threshold_percent
    t_regress = config.regression_threshold_percent

    # 6. Status classification and explanation
    if improvement_pct > t_improve:
        status = VerificationStatus.IMPROVED
        explanation = (
            f"Candidate {display_name} improved by {improvement_pct:.1f}%, "
            f"exceeding the configured {t_improve:.1f}% improvement threshold."
        )
    elif degradation_pct > t_regress:
        status = VerificationStatus.REGRESSED
        explanation = (
            f"Candidate {display_name} worsened by {degradation_pct:.1f}%, "
            f"exceeding the configured {t_regress:.1f}% regression threshold."
        )
    else:
        status = VerificationStatus.NO_MEANINGFUL_CHANGE
        if improvement_pct > 0.0:
            explanation = (
                f"Candidate {display_name} improved by {improvement_pct:.1f}%, "
                f"which is below the configured {t_improve:.1f}% improvement threshold."
            )
        elif degradation_pct > 0.0:
            explanation = (
                f"Candidate {display_name} changed by {degradation_pct:.1f}%, "
                f"which is below the configured {t_regress:.1f}% regression threshold."
            )
        else:
            explanation = f"Candidate {display_name} showed no change compared with baseline."

    return VerificationResult(
        schema_version="1.0",
        status=status,
        metric=metric_summary,
        evidence=evidence_dict,
        thresholds=thresholds_dict,
        explanation=explanation,
        candidate_id=candidate_id,
        limitations=list(DEFAULT_VERIFICATION_LIMITATIONS),
    )


# Convenient alias
verify = verify_optimization


if __name__ == "__main__":
    # Deterministic synthetic demonstration of Verification Engine V1
    demo_baseline = {
        "schema_version": "1.0",
        "fingerprint": "a1b2c3d4e5f60000",
        "benchmark": {"name": "gemm_fp16", "version": "1.0.0"},
        "configuration": {"warmup_runs": 5, "measurement_runs": 30, "operation_count": 1000},
        "latency_ms": {"count": 30, "median": 100.0, "mean": 100.5, "stddev": 1.5},
        "throughput": {"operations_per_second": 10000.0},
    }

    demo_candidate = {
        "schema_version": "1.0",
        "fingerprint": "a1b2c3d4e5f60000",
        "benchmark": {"name": "gemm_fp16", "version": "1.0.0"},
        "configuration": {"warmup_runs": 5, "measurement_runs": 30, "operation_count": 1000},
        "latency_ms": {"count": 30, "median": 90.0, "mean": 90.3, "stddev": 1.2},
        "throughput": {"operations_per_second": 11111.1},
    }

    demo_config = VerificationConfig(
        metric="median_latency_ms",
        improvement_threshold_percent=5.0,
        regression_threshold_percent=5.0,
        minimum_measurements=5,
    )

    demo_result = verify_optimization(
        baseline=demo_baseline,
        candidate=demo_candidate,
        candidate_info="runtime-001",
        config=demo_config,
    )

    print(demo_result.to_json(indent=2))
