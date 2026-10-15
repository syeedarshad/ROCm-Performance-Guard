"""Optimization Engine V1 for ROCm Performance Guard.

Converts evidence from regression and diagnosis engines into ranked, conservative,
evidence-based optimization experiments without making automatic or destructive system changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from src.benchmark import BenchmarkResult
except ImportError:
    BenchmarkResult = Any  # type: ignore[misc,assignment]

try:
    from src.regression import RegressionResult, RegressionStatus, compare_benchmarks
except ImportError:
    RegressionResult = Any  # type: ignore[misc,assignment]
    RegressionStatus = Any  # type: ignore[misc,assignment]
    compare_benchmarks = None  # type: ignore[assignment]

try:
    from src.diagnosis import (
        DetectedChange,
        DiagnosisResult,
        DiagnosisStatus,
        diagnose_regression,
    )
except ImportError:
    DetectedChange = Any  # type: ignore[misc,assignment]
    DiagnosisResult = Any  # type: ignore[misc,assignment]
    DiagnosisStatus = Any  # type: ignore[misc,assignment]
    diagnose_regression = None  # type: ignore[assignment]


class OptimizationStatus(str, Enum):
    """Status classification for optimization planning."""

    ACTIONABLE_OPTIMIZATIONS_FOUND = "ACTIONABLE_OPTIMIZATIONS_FOUND"
    NO_ACTIONABLE_OPTIMIZATION = "NO_ACTIONABLE_OPTIMIZATION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFIGURATION_MUST_BE_RESTORED = "CONFIGURATION_MUST_BE_RESTORED"
    NO_REGRESSION_TO_OPTIMIZE = "NO_REGRESSION_TO_OPTIMIZE"
    IMPROVEMENT_TO_VALIDATE = "IMPROVEMENT_TO_VALIDATE"


class OptimizationPriority(str, Enum):
    """Investigation priority for optimization candidates."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class OptimizationRisk(str, Enum):
    """Safety and risk assessment for proposed experiments."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class OptimizationCategory(str, Enum):
    """Category classification for optimization candidates."""

    RUNTIME = "runtime"
    SOFTWARE = "software"
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"
    WORKLOAD = "workload"
    HARDWARE = "hardware"


DEFAULT_OPTIMIZATION_LIMITATIONS: List[str] = [
    "Recommendations are evidence-based controlled experiments; improvements require empirical verification.",
    "Candidate optimizations must be empirically verified using identical benchmark methodology.",
    "No system files, drivers, or permanent environment configurations are modified automatically.",
    "Unmeasured background system load, thermal throttling, or OS scheduling variance may influence results.",
]


@dataclass
class OptimizationCandidate:
    """A single, evidence-backed planned optimization experiment."""

    candidate_id: str
    category: str
    title: str
    description: str
    rationale: str
    priority: str
    risk: str
    expected_direction: str
    target: str
    proposed_change: str
    evidence: List[str]
    verification_plan: str
    action: str = "TEST"
    current_value: Optional[Any] = None
    candidate_values: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to clean JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "priority": self.priority,
            "risk": self.risk,
            "expected_direction": self.expected_direction,
            "target": self.target,
            "proposed_change": self.proposed_change,
            "evidence": list(self.evidence),
            "verification_plan": self.verification_plan,
            "action": self.action,
        }
        if self.current_value is not None:
            d["current_value"] = self.current_value
        if self.candidate_values is not None:
            d["candidate_values"] = list(self.candidate_values)
        return d


@dataclass
class OptimizationPlan:
    """Comprehensive, JSON-serializable optimization plan with verification requirements."""

    schema_version: str
    status: OptimizationStatus
    summary: str
    baseline: Dict[str, Any]
    candidates: List[OptimizationCandidate] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=lambda: list(DEFAULT_OPTIMIZATION_LIMITATIONS))

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "summary": self.summary,
            "baseline": dict(self.baseline),
            "candidates": [c.to_dict() for c in self.candidates],
            "verification": dict(self.verification),
            "limitations": list(self.limitations),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert plan to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _to_dict_safe(obj: Any) -> Optional[Dict[str, Any]]:
    """Safely extract dictionary representation from object or dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return None


def _format_verification_plan(metric_name: str) -> str:
    """Generate standard verification instructions for experiments."""
    return (
        f"Run identical benchmark using candidate configuration; "
        f"evaluate '{metric_name}' across sufficient repetitions and warmup runs; "
        f"compare against current baseline and reject changes that worsen performance."
    )


def _sort_candidates(candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
    """Sort candidates deterministically by priority, evidence strength, risk, and ID.

    Order:
      1. Priority DESC (HIGH=3, MEDIUM=2, LOW=1)
      2. Evidence count DESC
      3. Risk ASC (LOW=1, MEDIUM=2, HIGH=3) -> lower risk first
      4. Candidate ID ASC (alphabetical tie-breaker)
    """
    def key_fn(c: OptimizationCandidate) -> Tuple[int, int, int, str]:
        prio_val = 3 if c.priority == "HIGH" else (2 if c.priority == "MEDIUM" else 1)
        risk_val = 1 if c.risk == "LOW" else (2 if c.risk == "MEDIUM" else 3)
        ev_count = len(c.evidence)
        return (-prio_val, -ev_count, risk_val, c.candidate_id)

    return sorted(candidates, key=key_fn)


def _deduplicate_candidates(candidates: List[OptimizationCandidate]) -> List[OptimizationCandidate]:
    """Remove duplicate candidates sharing identical target and category, merging evidence."""
    seen: Dict[Tuple[str, str], OptimizationCandidate] = {}
    ordered_keys: List[Tuple[str, str]] = []

    for c in candidates:
        key = (c.category, c.target)
        if key not in seen:
            seen[key] = c
            ordered_keys.append(key)
        else:
            # Merge evidence and keep higher priority
            existing = seen[key]
            merged_evidence = list(dict.fromkeys(existing.evidence + c.evidence))
            existing.evidence = merged_evidence
            if c.priority == "HIGH" and existing.priority != "HIGH":
                existing.priority = "HIGH"

    return [seen[k] for k in ordered_keys]


def generate_optimization_plan(
    diagnosis: Optional[Union[DiagnosisResult, Dict[str, Any]]] = None,
    regression: Optional[Union[RegressionResult, Dict[str, Any]]] = None,
    baseline_benchmark: Optional[Union[BenchmarkResult, Dict[str, Any]]] = None,
    current_benchmark: Optional[Union[BenchmarkResult, Dict[str, Any]]] = None,
    baseline_hardware: Optional[Dict[str, Any]] = None,
    current_hardware: Optional[Dict[str, Any]] = None,
    baseline_environment: Optional[Dict[str, Any]] = None,
    current_environment: Optional[Dict[str, Any]] = None,
) -> OptimizationPlan:
    """Generate evidence-based optimization recommendations and controlled experiment plans.

    Args:
        diagnosis: Optional DiagnosisResult object or dict.
        regression: Optional RegressionResult object or dict.
        baseline_benchmark: Optional baseline benchmark result.
        current_benchmark: Optional current benchmark result.
        baseline_hardware: Optional baseline hardware snapshot.
        current_hardware: Optional current hardware snapshot.
        baseline_environment: Optional baseline environment snapshot.
        current_environment: Optional current environment snapshot.

    Returns:
        OptimizationPlan: Transparent, ranked, evidence-backed candidate experiments.
    """
    diag_dict = _to_dict_safe(diagnosis)
    reg_dict = _to_dict_safe(regression)
    base_bm_dict = _to_dict_safe(baseline_benchmark) or {}
    curr_bm_dict = _to_dict_safe(current_benchmark) or {}

    # If regression result is missing but benchmarks are provided, compute it
    if reg_dict is None and base_bm_dict and curr_bm_dict and compare_benchmarks is not None:
        computed_reg = compare_benchmarks(base_bm_dict, curr_bm_dict)
        reg_dict = computed_reg.to_dict()

    # If diagnosis is missing but benchmarks are provided, compute it
    if diag_dict is None and base_bm_dict and curr_bm_dict and diagnose_regression is not None:
        computed_diag = diagnose_regression(
            baseline_benchmark=base_bm_dict,
            current_benchmark=curr_bm_dict,
            baseline_hardware=baseline_hardware,
            current_hardware=current_hardware,
            baseline_environment=baseline_environment,
            current_environment=current_environment,
            regression_result=reg_dict,
        )
        diag_dict = computed_diag.to_dict()

    # Extract regression status and metric information
    if reg_dict:
        reg_status_str = str(reg_dict.get("status", "UNKNOWN"))
    elif diag_dict and diag_dict.get("regression_status"):
        reg_status_str = str(diag_dict.get("regression_status"))
    else:
        reg_status_str = "UNKNOWN"

    diag_status_str = str(diag_dict.get("status", "UNKNOWN")) if diag_dict else "UNKNOWN"

    # Baseline metric info
    metric_info = (reg_dict or {}).get("metric", {})
    metric_name = metric_info.get("name") or "median_latency_ms"
    metric_baseline = metric_info.get("baseline")
    metric_current = metric_info.get("current")

    if metric_baseline is None and "latency_ms" in base_bm_dict:
        metric_baseline = base_bm_dict["latency_ms"].get("median")
    if metric_current is None and "latency_ms" in curr_bm_dict:
        metric_current = curr_bm_dict["latency_ms"].get("median")

    baseline_summary = {
        "metric": metric_name,
        "value": metric_baseline,
        "current_value": metric_current,
    }

    verification_meta = {
        "required": True,
        "method": _format_verification_plan(metric_name),
    }

    # 1. Check for missing snapshots / evidence
    if diag_dict is None and not base_bm_dict and not curr_bm_dict:
        return OptimizationPlan(
            schema_version="1.0",
            status=OptimizationStatus.INSUFFICIENT_EVIDENCE,
            summary="Insufficient evidence to generate actionable optimization candidates without unsupported speculation.",
            baseline=baseline_summary,
            candidates=[],
            verification=verification_meta,
        )

    # 2. No regression to optimize
    if reg_status_str == "NO_REGRESSION" or diag_status_str == "NO_REGRESSION_TO_DIAGNOSE":
        return OptimizationPlan(
            schema_version="1.0",
            status=OptimizationStatus.NO_REGRESSION_TO_OPTIMIZE,
            summary="No performance regression detected; optimization planning is not required.",
            baseline=baseline_summary,
            candidates=[],
            verification=verification_meta,
        )

    # 3. Benchmark configuration comparability mismatch
    if diag_status_str == "CONFIGURATION_CHANGED" or reg_status_str == "INCOMPARABLE":
        config_changes = (diag_dict or {}).get("changes", [])
        evidence_list = [c.get("explanation", c.get("reason", "Benchmark configuration changed.")) for c in config_changes]
        if not evidence_list:
            evidence_list = ["Benchmark configuration parameters or fingerprints differ between runs."]

        candidate = OptimizationCandidate(
            candidate_id="config-001",
            category=OptimizationCategory.CONFIGURATION.value,
            title="Restore comparable benchmark configuration",
            description="Workload execution parameters differ from the baseline, invalidating software regression attribution.",
            rationale="The benchmark configuration changed; valid optimization requires identical workload parameters.",
            priority=OptimizationPriority.HIGH.value,
            risk=OptimizationRisk.LOW.value,
            expected_direction="VALIDATE",
            target="benchmark.configuration",
            proposed_change="Restore benchmark configuration and workload parameters to baseline values before optimizing.",
            evidence=evidence_list,
            verification_plan=_format_verification_plan(metric_name),
            action="RESTORE",
        )

        return OptimizationPlan(
            schema_version="1.0",
            status=OptimizationStatus.CONFIGURATION_MUST_BE_RESTORED,
            summary="Benchmark configuration changed; comparability must be restored before performance optimization can proceed.",
            baseline=baseline_summary,
            candidates=[candidate],
            verification=verification_meta,
        )

    # 4. Insufficient evidence
    if diag_status_str == "INSUFFICIENT_EVIDENCE":
        return OptimizationPlan(
            schema_version="1.0",
            status=OptimizationStatus.INSUFFICIENT_EVIDENCE,
            summary="Insufficient evidence to generate actionable optimization candidates without unsupported speculation.",
            baseline=baseline_summary,
            candidates=[],
            verification=verification_meta,
        )

    # 5. Generate candidates from diagnosis changes
    detected_changes: List[Dict[str, Any]] = (diag_dict or {}).get("changes", [])
    raw_candidates: List[OptimizationCandidate] = []
    idx_counter: int = 1

    is_improvement = "IMPROVEMENT" in reg_status_str
    direction = "VALIDATE" if is_improvement else "IMPROVE"

    for ch in detected_changes:
        field_str = ch.get("field", "")
        f_lower = field_str.lower()
        base_val = ch.get("baseline")
        curr_val = ch.get("current")
        ch_type = ch.get("change_type", "modified")
        reason_str = ch.get("explanation") or ch.get("reason") or f"Detected change in {field_str}."

        # A. GPU Identity change
        if "gpu.host_devices" in f_lower or "gpu_name" in f_lower:
            raw_candidates.append(OptimizationCandidate(
                candidate_id=f"hardware-{idx_counter:03d}",
                category=OptimizationCategory.HARDWARE.value,
                title="Validate intended AMD accelerator target",
                description="GPU hardware device identity differs from the known baseline device.",
                rationale="A different GPU device was detected compared to the baseline run.",
                priority=OptimizationPriority.HIGH.value,
                risk=OptimizationRisk.LOW.value,
                expected_direction=direction,
                target=field_str,
                proposed_change="Validate that the intended AMD accelerator is being used and verify compute target.",
                evidence=[reason_str],
                verification_plan=_format_verification_plan(metric_name),
                action="VALIDATE",
                current_value=curr_val,
                candidate_values=[base_val, curr_val],
            ))
            idx_counter += 1

        # B. GPU or ROCm availability change
        elif "gpu.linux_has_gpu" in f_lower or "wsl_has_host_gpu" in f_lower or "rocm.is_available" in f_lower:
            raw_candidates.append(OptimizationCandidate(
                candidate_id=f"hardware-{idx_counter:03d}",
                category=OptimizationCategory.HARDWARE.value,
                title="Restore accelerator and ROCm runtime availability",
                description="Accelerator or ROCm tooling availability changed between baseline and current runs.",
                rationale="The diagnosis engine identified that accelerator or ROCm availability changed.",
                priority=OptimizationPriority.HIGH.value,
                risk=OptimizationRisk.LOW.value,
                expected_direction=direction,
                target=field_str,
                proposed_change="Restore and validate accelerator runtime availability before further optimization.",
                evidence=[reason_str],
                verification_plan=_format_verification_plan(metric_name),
                action="VALIDATE",
                current_value=curr_val,
                candidate_values=[base_val, curr_val],
            ))
            idx_counter += 1

        # C. ROCm or HIP version change
        elif "rocm" in f_lower and ("version" in f_lower or "hip" in f_lower):
            raw_candidates.append(OptimizationCandidate(
                candidate_id=f"runtime-{idx_counter:03d}",
                category=OptimizationCategory.RUNTIME.value,
                title="Validate ROCm runtime compatibility against baseline",
                description="Evaluate whether the current ROCm runtime configuration is contributing to the observed performance difference.",
                rationale="The diagnosis engine detected a ROCm/HIP version change associated with a performance delta.",
                priority=OptimizationPriority.HIGH.value,
                risk=OptimizationRisk.LOW.value,
                expected_direction=direction,
                target=field_str,
                proposed_change=f"Compare current runtime ({curr_val}) against the known-good baseline ({base_val}) under identical workload conditions.",
                evidence=[reason_str],
                verification_plan=_format_verification_plan(metric_name),
                action="TEST",
                current_value=curr_val,
                candidate_values=[base_val, curr_val],
            ))
            idx_counter += 1

        # D. Python packages (torch, transformers, onnxruntime, etc.)
        elif "python_packages" in f_lower:
            pkg_name = field_str.split(".")[-1]
            prio = OptimizationPriority.HIGH.value if pkg_name in ("torch", "triton") else OptimizationPriority.MEDIUM.value
            raw_candidates.append(OptimizationCandidate(
                candidate_id=f"software-{idx_counter:03d}",
                category=OptimizationCategory.SOFTWARE.value,
                title=f"Compare {pkg_name} package version against baseline",
                description=f"Evaluate performance impact of package '{pkg_name}' version difference ({base_val} vs {curr_val}).",
                rationale=f"Package '{pkg_name}' was {ch_type} between benchmark runs.",
                priority=prio,
                risk=OptimizationRisk.LOW.value,
                expected_direction=direction,
                target=field_str,
                proposed_change=f"Compare current package version ({curr_val}) against the known-good baseline ({base_val}) under identical workload conditions.",
                evidence=[reason_str],
                verification_plan=_format_verification_plan(metric_name),
                action="TEST",
                current_value=curr_val,
                candidate_values=[base_val, curr_val],
            ))
            idx_counter += 1

        # E. Environment variables
        elif "environment_variables" in f_lower:
            var_name = field_str.split(".")[-1]
            is_compute_var = any(k in var_name for k in ("ROCM", "HIP", "HSA", "CUDA", "DEVICE", "MIOPEN"))
            is_thread_var = "THREAD" in var_name

            prio = OptimizationPriority.HIGH.value if is_compute_var else (
                OptimizationPriority.MEDIUM.value if is_thread_var else OptimizationPriority.LOW.value
            )
            risk = OptimizationRisk.MEDIUM.value if is_thread_var else OptimizationRisk.LOW.value

            raw_candidates.append(OptimizationCandidate(
                candidate_id=f"environment-{idx_counter:03d}",
                category=OptimizationCategory.ENVIRONMENT.value,
                title=f"Evaluate {var_name} setting in controlled experiment",
                description=f"Measure workload sensitivity to environment variable '{var_name}'.",
                rationale=f"Environment variable '{var_name}' changed from '{base_val}' to '{curr_val}'.",
                priority=prio,
                risk=risk,
                expected_direction=direction,
                target=var_name,
                proposed_change=f"Evaluate baseline value ('{base_val}') and current value ('{curr_val}') under identical workload conditions without modifying permanent environment.",
                evidence=[reason_str],
                verification_plan=_format_verification_plan(metric_name),
                action="TEST",
                current_value=curr_val,
                candidate_values=[base_val, curr_val],
            ))
            idx_counter += 1

        # F. CPU performance / governor
        elif "cpu_performance" in f_lower or "governor" in f_lower:
            raw_candidates.append(OptimizationCandidate(
                candidate_id=f"runtime-{idx_counter:03d}",
                category=OptimizationCategory.RUNTIME.value,
                title="Evaluate CPU scaling governor configuration",
                description="Measure impact of CPU power governor on execution latency.",
                rationale="CPU performance governor configuration differs from baseline run.",
                priority=OptimizationPriority.MEDIUM.value,
                risk=OptimizationRisk.LOW.value,
                expected_direction=direction,
                target=field_str,
                proposed_change=f"Test workload execution under governor setting '{base_val}' compared to '{curr_val}'.",
                evidence=[reason_str],
                verification_plan=_format_verification_plan(metric_name),
                action="TEST",
                current_value=curr_val,
                candidate_values=[base_val, curr_val],
            ))
            idx_counter += 1

        # G. Workload parameter change present in benchmark config
        elif "configuration" in f_lower:
            param_name = field_str.split(".")[-1]
            raw_candidates.append(OptimizationCandidate(
                candidate_id=f"workload-{idx_counter:03d}",
                category=OptimizationCategory.WORKLOAD.value,
                title=f"Evaluate workload parameter '{param_name}' alternatives",
                description=f"Workload configuration parameter '{param_name}' differs from baseline.",
                rationale=f"Parameter '{param_name}' is configurable and directly alters workload execution characteristics.",
                priority=OptimizationPriority.MEDIUM.value,
                risk=OptimizationRisk.MEDIUM.value,
                expected_direction=direction,
                target=field_str,
                proposed_change=f"Evaluate alternative values ({base_val} vs {curr_val}) under identical hardware and software conditions.",
                evidence=[reason_str],
                verification_plan=_format_verification_plan(metric_name),
                action="TEST",
                current_value=curr_val,
                candidate_values=[base_val, curr_val],
            ))
            idx_counter += 1

    # 6. Deduplicate and sort candidates deterministically
    deduped = _deduplicate_candidates(raw_candidates)
    sorted_candidates = _sort_candidates(deduped)

    # 7. Check if any actionable candidate was found
    if not sorted_candidates:
        return OptimizationPlan(
            schema_version="1.0",
            status=OptimizationStatus.NO_ACTIONABLE_OPTIMIZATION,
            summary="No actionable optimization candidates could be derived from the available evidence.",
            baseline=baseline_summary,
            candidates=[],
            verification=verification_meta,
        )

    # 8. Formulate plan status and summary
    if is_improvement:
        status = OptimizationStatus.IMPROVEMENT_TO_VALIDATE
        summary = f"Performance improved; {len(sorted_candidates)} controlled validation experiment(s) are recommended to isolate and verify contributing factors."
    else:
        status = OptimizationStatus.ACTIONABLE_OPTIMIZATIONS_FOUND
        summary = f"{len(sorted_candidates)} controlled optimization experiment(s) are recommended based on observed evidence."

    return OptimizationPlan(
        schema_version="1.0",
        status=status,
        summary=summary,
        baseline=baseline_summary,
        candidates=sorted_candidates,
        verification=verification_meta,
        limitations=list(DEFAULT_OPTIMIZATION_LIMITATIONS),
    )


# Convenient alias
optimize = generate_optimization_plan


if __name__ == "__main__":
    # Synthetic demonstration of Optimization Engine V1
    demo_base_bm = {
        "schema_version": "1.0",
        "fingerprint": "a1b2c3d4e5f60000",
        "benchmark": {"name": "gemm_fp16", "version": "1.0.0"},
        "configuration": {"warmup_runs": 5, "measurement_runs": 30, "operation_count": 1000},
        "latency_ms": {"count": 30, "median": 50.0, "stddev": 1.2},
    }

    demo_curr_bm = {
        "schema_version": "1.0",
        "fingerprint": "a1b2c3d4e5f60000",
        "benchmark": {"name": "gemm_fp16", "version": "1.0.0"},
        "configuration": {"warmup_runs": 5, "measurement_runs": 30, "operation_count": 1000},
        "latency_ms": {"count": 30, "median": 60.0, "stddev": 1.9},
    }

    demo_base_env = {
        "python": {"version": "3.13.1"},
        "rocm": {"version": "6.2.0", "available": True, "hipconfig": {"version": "6.2.0"}},
        "python_packages": {
            "torch": {"installed": True, "version": "2.4.0+rocm6.2"},
        },
        "environment_variables": {"ROCM_PATH": "/opt/rocm-6.2.0"},
    }

    demo_curr_env = {
        "python": {"version": "3.13.1"},
        "rocm": {"version": "7.0.0", "available": True, "hipconfig": {"version": "7.0.0"}},
        "python_packages": {
            "torch": {"installed": True, "version": "2.5.0+rocm7.0"},
        },
        "environment_variables": {"ROCM_PATH": "/opt/rocm-7.0.0"},
    }

    print("=== ROCm Performance Guard — Optimization Engine V1 Demo ===")
    plan = generate_optimization_plan(
        baseline_benchmark=demo_base_bm,
        current_benchmark=demo_curr_bm,
        baseline_environment=demo_base_env,
        current_environment=demo_curr_env,
    )
    print(plan.to_json(indent=2))
