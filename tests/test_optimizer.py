"""Tests for Optimization Engine V1 (src/optimizer.py).

Verifies deterministic, explainable, and safe generation of optimization candidates
and controlled experiment plans without modifying the user's system.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from typing import Any, Dict

from src.diagnosis import DetectedChange, DiagnosisResult, DiagnosisStatus
from src.optimizer import (
    OptimizationCandidate,
    OptimizationCategory,
    OptimizationPlan,
    OptimizationPriority,
    OptimizationRisk,
    OptimizationStatus,
    generate_optimization_plan,
)
from src.regression import RegressionResult, RegressionStatus


def _create_sample_benchmark(
    name: str = "gemm_fp16",
    version: str = "1.0.0",
    fingerprint: str = "fp123456",
    warmup_runs: int = 5,
    measurement_runs: int = 30,
    operation_count: int = 1000,
    median_latency_ms: float = 50.0,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Helper to generate a deterministic synthetic BenchmarkResult dictionary."""
    return {
        "schema_version": "1.0",
        "fingerprint": fingerprint,
        "benchmark": {"name": name, "version": version},
        "configuration": {
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "operation_count": operation_count,
            "batch_size": batch_size,
        },
        "latency_ms": {
            "count": measurement_runs,
            "median": median_latency_ms,
            "mean": median_latency_ms + 0.1,
            "stddev": 1.0,
        },
        "throughput": {"operations_per_second": (operation_count / (median_latency_ms / 1000.0))},
    }


def _create_sample_diagnosis(
    status: DiagnosisStatus = DiagnosisStatus.POTENTIAL_CAUSES_FOUND,
    regression_status: str = "REGRESSION",
    changes: list = None,
) -> DiagnosisResult:
    """Helper to generate a deterministic DiagnosisResult object."""
    return DiagnosisResult(
        schema_version="1.0",
        status=status,
        regression_status=regression_status,
        summary="Synthetic diagnosis for optimizer testing.",
        changes=changes or [],
    )


class TestOptimizationEngine(unittest.TestCase):
    """Unit tests covering all requirements for Optimization Engine V1."""

    def test_rocm_change_generates_high_priority_runtime_candidate(self) -> None:
        """1. Regression + ROCm change -> HIGH priority runtime candidate."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange(
                    category="software",
                    field="rocm.version",
                    baseline="6.2.0",
                    current="7.0.0",
                    priority="HIGH",
                    investigation_priority_score=3,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="ROCm version changed from 6.2.0 to 7.0.0.",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.ACTIONABLE_OPTIMIZATIONS_FOUND)
        rocm_cands = [c for c in plan.candidates if "rocm" in c.target.lower()]
        self.assertTrue(len(rocm_cands) > 0)
        self.assertEqual(rocm_cands[0].category, OptimizationCategory.RUNTIME.value)
        self.assertEqual(rocm_cands[0].priority, OptimizationPriority.HIGH.value)
        self.assertEqual(rocm_cands[0].risk, OptimizationRisk.LOW.value)

    def test_gpu_availability_change_candidate(self) -> None:
        """2. Regression + GPU availability change -> HIGH priority candidate."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange(
                    category="hardware",
                    field="gpu.linux_has_gpu",
                    baseline=True,
                    current=False,
                    priority="HIGH",
                    investigation_priority_score=3,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="GPU availability changed from True to False.",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.ACTIONABLE_OPTIMIZATIONS_FOUND)
        gpu_cands = [c for c in plan.candidates if "gpu" in c.target.lower()]
        self.assertTrue(len(gpu_cands) > 0)
        self.assertEqual(gpu_cands[0].priority, OptimizationPriority.HIGH.value)

    def test_gpu_identity_change_candidate(self) -> None:
        """3. Regression + GPU identity change -> HIGH priority hardware candidate."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange(
                    category="hardware",
                    field="gpu.host_devices",
                    baseline=["AMD Radeon Graphics"],
                    current=["AMD Radeon RX 7900 XTX"],
                    priority="HIGH",
                    investigation_priority_score=3,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="Host GPU devices differed.",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.ACTIONABLE_OPTIMIZATIONS_FOUND)
        c = plan.candidates[0]
        self.assertEqual(c.category, OptimizationCategory.HARDWARE.value)
        self.assertEqual(c.priority, OptimizationPriority.HIGH.value)
        self.assertIn("intended AMD accelerator", c.title)

    def test_relevant_package_change_candidate(self) -> None:
        """4. Regression + relevant package change -> software candidate."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange(
                    category="software",
                    field="python_packages.transformers",
                    baseline="4.40.0",
                    current="4.44.0",
                    priority="MEDIUM",
                    investigation_priority_score=2,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="Package transformers changed.",
                    change_type="upgraded",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.ACTIONABLE_OPTIMIZATIONS_FOUND)
        self.assertEqual(plan.candidates[0].category, OptimizationCategory.SOFTWARE.value)
        self.assertIn("transformers", plan.candidates[0].target)

    def test_environment_variable_change_candidate(self) -> None:
        """5. Regression + environment variable change -> environment candidate."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange(
                    category="environment",
                    field="environment_variables.OMP_NUM_THREADS",
                    baseline="8",
                    current="16",
                    priority="MEDIUM",
                    investigation_priority_score=2,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="OMP_NUM_THREADS changed.",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.ACTIONABLE_OPTIMIZATIONS_FOUND)
        self.assertEqual(plan.candidates[0].category, OptimizationCategory.ENVIRONMENT.value)
        self.assertEqual(plan.candidates[0].target, "OMP_NUM_THREADS")

    def test_benchmark_configuration_changed_status(self) -> None:
        """6. Benchmark configuration changed -> CONFIGURATION_MUST_BE_RESTORED."""
        diag = _create_sample_diagnosis(
            status=DiagnosisStatus.CONFIGURATION_CHANGED,
            changes=[
                DetectedChange(
                    category="configuration",
                    field="configuration.batch_size",
                    baseline=1,
                    current=8,
                    priority="HIGH",
                    investigation_priority_score=3,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="Batch size changed from 1 to 8.",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.CONFIGURATION_MUST_BE_RESTORED)
        self.assertIn("comparability must be restored", plan.summary)
        self.assertEqual(plan.candidates[0].priority, OptimizationPriority.HIGH.value)
        self.assertEqual(plan.candidates[0].risk, OptimizationRisk.LOW.value)

    def test_regression_with_no_diagnosis_evidence(self) -> None:
        """7. Regression + no diagnosis evidence -> INSUFFICIENT_EVIDENCE."""
        diag = _create_sample_diagnosis(
            status=DiagnosisStatus.INSUFFICIENT_EVIDENCE,
            changes=[],
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(len(plan.candidates), 0)

    def test_no_regression_to_optimize(self) -> None:
        """8. No regression -> NO_REGRESSION_TO_OPTIMIZE."""
        diag = _create_sample_diagnosis(
            status=DiagnosisStatus.NO_REGRESSION_TO_DIAGNOSE,
            regression_status="NO_REGRESSION",
            changes=[],
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.NO_REGRESSION_TO_OPTIMIZE)
        self.assertEqual(len(plan.candidates), 0)

    def test_improvement_validation(self) -> None:
        """9. Improvement -> IMPROVEMENT_TO_VALIDATE."""
        diag = _create_sample_diagnosis(
            status=DiagnosisStatus.POTENTIAL_CAUSES_FOUND,
            regression_status="IMPROVEMENT",
            changes=[
                DetectedChange(
                    category="software",
                    field="rocm.version",
                    baseline="6.2.0",
                    current="7.0.0",
                    priority="HIGH",
                    investigation_priority_score=3,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="ROCm version upgraded.",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.IMPROVEMENT_TO_VALIDATE)
        self.assertIn("Performance improved", plan.summary)
        self.assertEqual(plan.candidates[0].expected_direction, "VALIDATE")

    def test_multiple_candidates_deterministic_ranking(self) -> None:
        """10. Multiple candidates -> sorted deterministically by priority, evidence, and risk."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange(
                    category="environment",
                    field="environment_variables.OMP_NUM_THREADS",
                    baseline="8",
                    current="16",
                    priority="MEDIUM",
                    investigation_priority_score=2,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="Thread count changed.",
                ),
                DetectedChange(
                    category="software",
                    field="rocm.version",
                    baseline="6.2.0",
                    current="7.0.0",
                    priority="HIGH",
                    investigation_priority_score=3,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="ROCm version changed.",
                ),
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        priorities = [c.priority for c in plan.candidates]
        self.assertEqual(priorities[0], OptimizationPriority.HIGH.value)
        self.assertEqual(priorities[1], OptimizationPriority.MEDIUM.value)

    def test_candidate_contains_verification_plan(self) -> None:
        """11. Every candidate contains an explicit verification plan."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange(
                    category="software",
                    field="rocm.version",
                    baseline="6.2.0",
                    current="7.0.0",
                    priority="HIGH",
                    investigation_priority_score=3,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="ROCm changed.",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        c = plan.candidates[0]
        self.assertTrue(len(c.verification_plan) > 0)
        self.assertIn("identical benchmark", c.verification_plan)
        self.assertIn("reject changes", c.verification_plan)

    def test_candidate_is_json_serializable(self) -> None:
        """12. OptimizationCandidate is JSON serializable."""
        c = OptimizationCandidate(
            candidate_id="test-001",
            category="runtime",
            title="Test Candidate",
            description="Test description",
            rationale="Test rationale",
            priority="HIGH",
            risk="LOW",
            expected_direction="IMPROVE",
            target="rocm_version",
            proposed_change="Test proposed change",
            evidence=["Evidence 1"],
            verification_plan="Run benchmark",
            candidate_values=["v1", "v2"],
        )
        dumped = json.dumps(c.to_dict())
        parsed = json.loads(dumped)
        self.assertEqual(parsed["candidate_id"], "test-001")
        self.assertEqual(parsed["candidate_values"], ["v1", "v2"])

    def test_optimization_plan_is_json_serializable(self) -> None:
        """13. OptimizationPlan is strictly JSON serializable."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange(
                    category="software",
                    field="rocm.version",
                    baseline="6.2.0",
                    current="7.0.0",
                    priority="HIGH",
                    investigation_priority_score=3,
                    role="POTENTIAL_CONTRIBUTOR",
                    explanation="ROCm changed.",
                )
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        dumped = plan.to_json()
        parsed = json.loads(dumped)
        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(parsed["status"], "ACTIONABLE_OPTIMIZATIONS_FOUND")

    def test_priority_ordering(self) -> None:
        """14. Priority levels sort HIGH before MEDIUM before LOW."""
        c_low = OptimizationCandidate("low-1", "env", "t", "d", "r", "LOW", "LOW", "IMPROVE", "t", "p", ["e"], "v")
        c_med = OptimizationCandidate("med-1", "sw", "t", "d", "r", "MEDIUM", "LOW", "IMPROVE", "t", "p", ["e"], "v")
        c_high = OptimizationCandidate("high-1", "rt", "t", "d", "r", "HIGH", "LOW", "IMPROVE", "t", "p", ["e"], "v")

        from src.optimizer import _sort_candidates
        sorted_cands = _sort_candidates([c_low, c_high, c_med])
        self.assertEqual([c.priority for c in sorted_cands], ["HIGH", "MEDIUM", "LOW"])

    def test_risk_ordering_within_same_priority(self) -> None:
        """15. Lower risk experiments sort before higher risk within same priority."""
        c_med_risk = OptimizationCandidate("cand-2", "env", "t", "d", "r", "HIGH", "MEDIUM", "IMPROVE", "t", "p", ["e"], "v")
        c_low_risk = OptimizationCandidate("cand-1", "env", "t", "d", "r", "HIGH", "LOW", "IMPROVE", "t", "p", ["e"], "v")

        from src.optimizer import _sort_candidates
        sorted_cands = _sort_candidates([c_med_risk, c_low_risk])
        self.assertEqual([c.risk for c in sorted_cands], ["LOW", "MEDIUM"])

    def test_candidate_ids_are_deterministic(self) -> None:
        """16. Generated candidate IDs are deterministic across multiple calls."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("software", "rocm.version", "6.2.0", "7.0.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "e1"),
                DetectedChange("environment", "environment_variables.ROCM_PATH", "p1", "p2", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "e2"),
            ]
        )
        plan1 = generate_optimization_plan(diagnosis=diag)
        plan2 = generate_optimization_plan(diagnosis=diag)
        self.assertEqual([c.candidate_id for c in plan1.candidates], [c.candidate_id for c in plan2.candidates])

    def test_no_candidate_modifies_environment(self) -> None:
        """17. Verifies that optimizer never mutates os.environ or system state."""
        orig_env = dict(os.environ)
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("environment", "environment_variables.OMP_NUM_THREADS", "4", "16", "MEDIUM", 2, "POTENTIAL_CONTRIBUTOR", "e"),
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(os.environ, orig_env)
        # Verify candidate only proposed a test action
        self.assertEqual(plan.candidates[0].action, "TEST")

    def test_missing_snapshots_handled_safely(self) -> None:
        """18. Missing snapshots do not crash plan generation and report INSUFFICIENT_EVIDENCE."""
        plan = generate_optimization_plan(diagnosis=None, baseline_benchmark=None, current_benchmark=None)
        self.assertEqual(plan.status, OptimizationStatus.INSUFFICIENT_EVIDENCE)

    def test_missing_diagnosis_evidence_handled_safely(self) -> None:
        """19. Diagnosis with empty changes yields NO_ACTIONABLE_OPTIMIZATION or INSUFFICIENT_EVIDENCE."""
        diag = _create_sample_diagnosis(status=DiagnosisStatus.POTENTIAL_CAUSES_FOUND, changes=[])
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.status, OptimizationStatus.NO_ACTIONABLE_OPTIMIZATION)

    def test_package_upgrade_detection(self) -> None:
        """20. Package upgrade produces candidate with compare baseline recommendation."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("software", "python_packages.torch", "2.4.0", "2.5.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "upgraded", change_type="upgraded")
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertIn("torch", plan.candidates[0].target)
        self.assertIn("Compare current package version", plan.candidates[0].proposed_change)

    def test_package_downgrade_detection(self) -> None:
        """21. Package downgrade produces candidate with compare baseline recommendation."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("software", "python_packages.torch", "2.5.0", "2.4.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "downgraded", change_type="downgraded")
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertIn("torch", plan.candidates[0].target)
        self.assertIn("Compare current package version", plan.candidates[0].proposed_change)

    def test_package_addition_and_removal(self) -> None:
        """22. Package addition/removal handled cleanly."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("software", "python_packages.onnxruntime", None, "1.18.0", "MEDIUM", 2, "POTENTIAL_CONTRIBUTOR", "added", change_type="added")
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertIn("onnxruntime", plan.candidates[0].target)

    def test_relevant_environment_variable_candidate(self) -> None:
        """23. Compute-related environment variable produces HIGH priority candidate."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("environment", "environment_variables.ROCM_PATH", "/opt/rocm-6.2.0", "/opt/rocm-7.0.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "e")
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.candidates[0].priority, OptimizationPriority.HIGH.value)
        self.assertEqual(plan.candidates[0].target, "ROCM_PATH")

    def test_workload_configuration_parameter_candidate(self) -> None:
        """24. Workload configuration parameter difference generates controlled experiment."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("configuration", "configuration.batch_size", 1, 8, "MEDIUM", 2, "POTENTIAL_CONTRIBUTOR", "e")
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        self.assertEqual(plan.candidates[0].category, OptimizationCategory.WORKLOAD.value)
        self.assertEqual(plan.candidates[0].risk, OptimizationRisk.MEDIUM.value)

    def test_current_and_baseline_benchmark_comparison_flow(self) -> None:
        """25. Generates plan directly from baseline and current benchmark results."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)
        base_env = {"rocm": {"version": "6.2.0", "available": True}}
        curr_env = {"rocm": {"version": "7.0.0", "available": True}}

        plan = generate_optimization_plan(
            baseline_benchmark=base_bm,
            current_benchmark=curr_bm,
            baseline_environment=base_env,
            current_environment=curr_env,
        )
        self.assertEqual(plan.status, OptimizationStatus.ACTIONABLE_OPTIMIZATIONS_FOUND)
        self.assertEqual(plan.baseline["value"], 50.0)
        self.assertEqual(plan.baseline["current_value"], 65.0)

    def test_latency_optimization_metric_propagation(self) -> None:
        """26. Propagates latency metric name into verification plan."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)
        plan = generate_optimization_plan(
            baseline_benchmark=base_bm,
            current_benchmark=curr_bm,
            regression={"status": "REGRESSION", "metric": {"name": "median_latency_ms", "baseline": 50.0, "current": 65.0}},
            diagnosis=_create_sample_diagnosis(changes=[DetectedChange("software", "rocm.version", "6.2", "7.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "e")]),
        )
        self.assertEqual(plan.baseline["metric"], "median_latency_ms")
        self.assertIn("median_latency_ms", plan.candidates[0].verification_plan)

    def test_throughput_optimization_metric_propagation(self) -> None:
        """27. Propagates throughput metric name into verification plan."""
        diag = _create_sample_diagnosis(changes=[DetectedChange("software", "rocm.version", "6.2", "7.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "e")])
        plan = generate_optimization_plan(
            diagnosis=diag,
            regression={"status": "REGRESSION", "metric": {"name": "throughput", "baseline": 100.0, "current": 80.0}},
        )
        self.assertEqual(plan.baseline["metric"], "throughput")
        self.assertIn("throughput", plan.candidates[0].verification_plan)

    def test_candidate_explanation_references_evidence(self) -> None:
        """28. Candidate explanation and rationale reference observed evidence."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("software", "rocm.version", "6.2.0", "7.0.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "Observed ROCm version delta from 6.2.0 to 7.0.0.")
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        c = plan.candidates[0]
        self.assertTrue(len(c.evidence) > 0)
        self.assertIn("6.2.0", c.evidence[0])

    def test_no_unsupported_causal_claims_in_plan(self) -> None:
        """29. Verifies absence of forbidden false-promise statements across all plan text."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("software", "rocm.version", "6.2.0", "7.0.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "ROCm changed.")
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        plan_json = plan.to_json().lower()

        forbidden_phrases = [
            "this will fix",
            "is guaranteed",
            "guaranteed fix",
            "definitely caused",
            "statistically significant",
        ]
        for phrase in forbidden_phrases:
            self.assertNotIn(phrase, plan_json)

    def test_duplicate_candidates_are_removed(self) -> None:
        """30. Duplicate candidates targeting identical category and target are consolidated."""
        diag = _create_sample_diagnosis(
            changes=[
                DetectedChange("software", "rocm.version", "6.2.0", "7.0.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "Evidence A"),
                DetectedChange("software", "rocm.version", "6.2.0", "7.0.0", "HIGH", 3, "POTENTIAL_CONTRIBUTOR", "Evidence B"),
            ]
        )
        plan = generate_optimization_plan(diagnosis=diag)
        rocm_cands = [c for c in plan.candidates if c.target == "rocm.version"]
        self.assertEqual(len(rocm_cands), 1)
        self.assertEqual(len(rocm_cands[0].evidence), 2)

    def test_cli_execution(self) -> None:
        """31. Verify python -m src.optimizer executes and returns valid JSON demonstration."""
        cmd = [sys.executable, "-m", "src.optimizer"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, f"CLI execution failed: {result.stderr}")
        self.assertIn("=== ROCm Performance Guard — Optimization Engine V1 Demo ===", result.stdout)

        lines = result.stdout.splitlines()
        json_lines = [l for l in lines if not l.startswith("===")]
        json_str = "\n".join(json_lines).strip()
        data = json.loads(json_str)

        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["status"], "ACTIONABLE_OPTIMIZATIONS_FOUND")
        self.assertTrue(len(data["candidates"]) > 0)


if __name__ == "__main__":
    unittest.main()
