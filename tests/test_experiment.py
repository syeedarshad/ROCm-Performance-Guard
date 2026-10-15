"""Test suite for Experiment Planner V1 (src/experiment.py).

Tests:
    01-05: Data class construction and serialization
    06-10: Baseline analysis and metric extraction
    11-15: Experiment candidate generation (individual generators)
    16-20: Core planner logic and edge cases
    21-25: Candidate ordering, priorities, and plan structure
    26-30: Integration with real baseline files and BenchmarkResult objects
    31-33: CLI execution and no-mutation verification
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import (
    DEFAULT_EXPERIMENT_LIMITATIONS,
    ExperimentCandidate,
    ExperimentCategory,
    ExperimentPlan,
    ExperimentPriority,
    ExperimentRisk,
    ExperimentStatus,
    _compute_flops,
    _extract_baseline_metrics,
    _generate_measurement_candidate,
    _generate_parallelism_candidate,
    _generate_strassen_candidate,
    _generate_tiling_candidate,
    _generate_transpose_candidate,
    _to_dict_safe,
    generate_experiment_plan,
    load_baseline_from_file,
    plan_experiments,
)


def _make_baseline(
    m: int = 512,
    n: int = 512,
    k: int = 512,
    median: float = 1342.0,
    mean: float = 1348.0,
    stddev: float = 40.0,
    p95: float = 1407.0,
    p99: float = 1429.0,
    min_ms: float = 1269.0,
    max_ms: float = 1437.0,
    count: int = 30,
    warmup: int = 5,
    iterations: int = 30,
    dtype: str = "fp32",
    backend: str = "cpu",
    gflops: float = 0.200,
    physical_cores: int = 8,
    logical_cores: int = 16,
) -> Dict[str, Any]:
    """Create a synthetic baseline benchmark result for testing."""
    operation_count = 2 * m * n * k
    return {
        "schema_version": "1.0",
        "fingerprint": "test_fp_0000",
        "benchmark": {"name": f"gemm_{dtype}", "version": "1.0.0"},
        "configuration": {
            "workload": "gemm",
            "m": m,
            "n": n,
            "k": k,
            "dtype": dtype,
            "backend": backend,
            "warmup_runs": warmup,
            "measurement_runs": iterations,
            "operation_count": operation_count,
        },
        "latency_ms": {
            "count": count,
            "min": min_ms,
            "mean": mean,
            "median": median,
            "p95": p95,
            "p99": p99,
            "max": max_ms,
            "stddev": stddev,
        },
        "throughput": {
            "operations_per_second": operation_count / (median / 1000.0) if median > 0 else None,
            "gflops": gflops,
            "operations": operation_count,
        },
        "hardware_snapshot": {
            "cpu": {
                "vendor": "AuthenticAMD",
                "model_name": "AMD Ryzen 7 7730U with Radeon Graphics",
                "physical_cores": physical_cores,
                "logical_cores": logical_cores,
            },
        },
    }


class TestExperimentPlanner(unittest.TestCase):
    """Comprehensive test suite for Experiment Planner V1."""

    # ==================================================================
    # 01-05: Data class construction and serialization
    # ==================================================================

    def test_01_experiment_candidate_construction(self):
        """01. ExperimentCandidate can be constructed with all required fields."""
        c = ExperimentCandidate(
            experiment_id="test-001",
            category="algorithm",
            title="Test Candidate",
            hypothesis="Testing works.",
            description="A test candidate.",
            rationale="For testing purposes.",
            priority="HIGH",
            risk="LOW",
            independent_variable="test_var",
            control_variables=["var1", "var2"],
            dependent_variable="median_latency_ms",
            proposed_values=[1, 2, 3],
            expected_direction="LOWER_LATENCY",
            estimated_speedup_range="1.0x-1.5x",
            measurement_requirements={"warmup_runs": 5},
            acceptance_criteria="Accept if improved.",
            rollback_plan="Revert to baseline.",
        )
        self.assertEqual(c.experiment_id, "test-001")
        self.assertEqual(c.category, "algorithm")
        self.assertEqual(c.priority, "HIGH")
        self.assertEqual(c.risk, "LOW")
        self.assertEqual(len(c.proposed_values), 3)

    def test_02_experiment_candidate_to_dict(self):
        """02. ExperimentCandidate.to_dict() produces JSON-serializable output."""
        c = ExperimentCandidate(
            experiment_id="test-002",
            category="parallelism",
            title="Test Dict",
            hypothesis="Serializable.",
            description="Desc.",
            rationale="Reason.",
            priority="MEDIUM",
            risk="MEDIUM",
            independent_variable="workers",
            control_variables=["m=512"],
            dependent_variable="median_latency_ms",
            proposed_values=[1, 2, 4],
            expected_direction="LOWER_LATENCY",
            estimated_speedup_range="1.5x-4x",
            measurement_requirements={"identical_hardware": True},
            acceptance_criteria="10% improvement.",
            rollback_plan="Single worker.",
        )
        d = c.to_dict()
        self.assertIsInstance(d, dict)
        json_str = json.dumps(d)
        self.assertIn("test-002", json_str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["experiment_id"], "test-002")
        self.assertEqual(parsed["proposed_values"], [1, 2, 4])

    def test_03_experiment_plan_construction(self):
        """03. ExperimentPlan can be constructed with all required fields."""
        plan = ExperimentPlan(
            schema_version="1.0",
            status=ExperimentStatus.EXPERIMENTS_GENERATED,
            summary="Test plan.",
            baseline_analysis={"test": True},
        )
        self.assertEqual(plan.schema_version, "1.0")
        self.assertEqual(plan.status, ExperimentStatus.EXPERIMENTS_GENERATED)
        self.assertIsInstance(plan.candidates, list)
        self.assertIsInstance(plan.limitations, list)
        self.assertTrue(len(plan.generated_at) > 0)

    def test_04_experiment_plan_to_json(self):
        """04. ExperimentPlan.to_json() produces valid, parseable JSON."""
        plan = ExperimentPlan(
            schema_version="1.0",
            status=ExperimentStatus.NO_EXPERIMENTS_POSSIBLE,
            summary="No experiments.",
            baseline_analysis={"workload": "gemm"},
        )
        json_str = plan.to_json(indent=2)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(parsed["status"], "NO_EXPERIMENTS_POSSIBLE")
        self.assertIn("limitations", parsed)

    def test_05_enum_values(self):
        """05. All enums have expected string values."""
        self.assertEqual(ExperimentStatus.EXPERIMENTS_GENERATED.value, "EXPERIMENTS_GENERATED")
        self.assertEqual(ExperimentStatus.INSUFFICIENT_BASELINE.value, "INSUFFICIENT_BASELINE")
        self.assertEqual(ExperimentPriority.HIGH.value, "HIGH")
        self.assertEqual(ExperimentRisk.LOW.value, "LOW")
        self.assertEqual(ExperimentCategory.ALGORITHM.value, "algorithm")
        self.assertEqual(ExperimentCategory.PARALLELISM.value, "parallelism")
        self.assertEqual(ExperimentCategory.MEMORY_ACCESS.value, "memory_access")
        self.assertEqual(ExperimentCategory.MEASUREMENT.value, "measurement")

    # ==================================================================
    # 06-10: Baseline analysis and metric extraction
    # ==================================================================

    def test_06_extract_baseline_metrics_standard(self):
        """06. _extract_baseline_metrics returns structured analysis for standard baseline."""
        baseline = _make_baseline()
        analysis = _extract_baseline_metrics(baseline)
        self.assertEqual(analysis["workload"]["dimensions"]["m"], 512)
        self.assertEqual(analysis["workload"]["dimensions"]["n"], 512)
        self.assertEqual(analysis["workload"]["dimensions"]["k"], 512)
        self.assertEqual(analysis["latency"]["median_ms"], 1342.0)
        self.assertEqual(analysis["latency"]["count"], 30)
        self.assertIsInstance(analysis["observations"], list)
        self.assertTrue(len(analysis["observations"]) > 0)

    def test_07_extract_baseline_high_variance(self):
        """07. High CV triggers appropriate observation."""
        baseline = _make_baseline(median=100.0, stddev=10.0)
        analysis = _extract_baseline_metrics(baseline)
        cv = analysis["latency"]["cv_percent"]
        self.assertGreater(cv, 5.0)
        obs = " ".join(analysis["observations"])
        self.assertIn("variance", obs.lower())

    def test_08_extract_baseline_low_variance(self):
        """08. Low CV triggers low-variance observation."""
        baseline = _make_baseline(median=1000.0, stddev=5.0)
        analysis = _extract_baseline_metrics(baseline)
        cv = analysis["latency"]["cv_percent"]
        self.assertLess(cv, 2.0)
        obs = " ".join(analysis["observations"])
        self.assertIn("Low measurement variance", obs)

    def test_09_extract_baseline_tail_latency(self):
        """09. Significant tail latency is detected when p95/median > 1.15."""
        baseline = _make_baseline(median=100.0, p95=120.0, stddev=5.0)
        analysis = _extract_baseline_metrics(baseline)
        obs = " ".join(analysis["observations"])
        self.assertIn("tail latency", obs.lower())

    def test_10_compute_flops(self):
        """10. _compute_flops returns 2*m*n*k."""
        self.assertEqual(_compute_flops(512, 512, 512), 2 * 512 * 512 * 512)
        self.assertEqual(_compute_flops(1, 1, 1), 2)
        self.assertEqual(_compute_flops(100, 200, 300), 2 * 100 * 200 * 300)

    # ==================================================================
    # 11-15: Experiment candidate generation
    # ==================================================================

    def test_11_tiling_candidate_generation(self):
        """11. Tiling candidate generated with correct fields for 512x512x512."""
        baseline = _make_baseline()
        analysis = _extract_baseline_metrics(baseline)
        c = _generate_tiling_candidate(analysis, 1)
        self.assertEqual(c.experiment_id, "tile-001")
        self.assertEqual(c.category, "algorithm")
        self.assertEqual(c.priority, "HIGH")
        self.assertEqual(c.risk, "LOW")
        self.assertEqual(c.independent_variable, "tile_size")
        self.assertEqual(c.dependent_variable, "median_latency_ms")
        self.assertTrue(len(c.proposed_values) > 0)
        for ts in c.proposed_values:
            self.assertIn(ts, [16, 32, 64, 128])

    def test_12_transpose_candidate_generation(self):
        """12. Transpose candidate generated with correct fields."""
        baseline = _make_baseline()
        analysis = _extract_baseline_metrics(baseline)
        c = _generate_transpose_candidate(analysis, 2)
        self.assertEqual(c.experiment_id, "transpose-002")
        self.assertEqual(c.category, "memory_access")
        self.assertIn("row_major_transposed", c.proposed_values)
        self.assertIn("column_major_naive", c.proposed_values)
        self.assertEqual(c.expected_direction, "LOWER_LATENCY")

    def test_13_parallelism_candidate_generation(self):
        """13. Parallelism candidate generated with worker counts."""
        baseline = _make_baseline(physical_cores=8, logical_cores=16)
        analysis = _extract_baseline_metrics(baseline)
        c = _generate_parallelism_candidate(analysis, 3)
        self.assertEqual(c.experiment_id, "parallel-003")
        self.assertEqual(c.category, "parallelism")
        self.assertIn(1, c.proposed_values)
        self.assertTrue(any(w > 1 for w in c.proposed_values))
        self.assertEqual(c.priority, "HIGH")

    def test_14_strassen_candidate_generation(self):
        """14. Strassen candidate generated for square power-of-2 matrix."""
        baseline = _make_baseline(m=512, n=512, k=512)
        analysis = _extract_baseline_metrics(baseline)
        c = _generate_strassen_candidate(analysis, 4)
        self.assertEqual(c.experiment_id, "strassen-004")
        self.assertEqual(c.category, "algorithm")
        self.assertEqual(c.risk, "MEDIUM")
        self.assertTrue(len(c.proposed_values) > 0)
        for val in c.proposed_values:
            self.assertLessEqual(val, 512)

    def test_15_measurement_candidate_generation(self):
        """15. Measurement refinement candidate generated with correct structure."""
        baseline = _make_baseline(warmup=5, iterations=30)
        analysis = _extract_baseline_metrics(baseline)
        c = _generate_measurement_candidate(analysis, 5)
        self.assertEqual(c.experiment_id, "measurement-005")
        self.assertEqual(c.category, "measurement")
        self.assertEqual(c.expected_direction, "LOWER_VARIANCE")
        self.assertEqual(len(c.proposed_values), 3)
        for cfg in c.proposed_values:
            self.assertIn("warmup_runs", cfg)
            self.assertIn("measurement_runs", cfg)
            self.assertIn("label", cfg)

    # ==================================================================
    # 16-20: Core planner logic and edge cases
    # ==================================================================

    def test_16_standard_plan_generation(self):
        """16. Standard baseline produces EXPERIMENTS_GENERATED with multiple candidates."""
        baseline = _make_baseline()
        plan = generate_experiment_plan(baseline)
        self.assertEqual(plan.status, ExperimentStatus.EXPERIMENTS_GENERATED)
        self.assertTrue(len(plan.candidates) >= 3)
        self.assertEqual(len(plan.execution_order), len(plan.candidates))
        self.assertIn("baseline_analysis", plan.to_dict())

    def test_17_missing_dimensions_returns_insufficient(self):
        """17. Missing matrix dimensions returns INSUFFICIENT_BASELINE."""
        baseline = {
            "configuration": {"warmup_runs": 5},
            "latency_ms": {"count": 30, "median": 1000.0},
        }
        plan = generate_experiment_plan(baseline)
        self.assertEqual(plan.status, ExperimentStatus.INSUFFICIENT_BASELINE)
        self.assertIn("missing", plan.summary.lower())

    def test_18_missing_latency_returns_insufficient(self):
        """18. Missing latency data returns INSUFFICIENT_BASELINE."""
        baseline = {
            "configuration": {"m": 512, "n": 512, "k": 512},
            "latency_ms": {},
        }
        plan = generate_experiment_plan(baseline)
        self.assertEqual(plan.status, ExperimentStatus.INSUFFICIENT_BASELINE)

    def test_19_zero_count_returns_insufficient(self):
        """19. Zero measurement count returns INSUFFICIENT_BASELINE."""
        baseline = {
            "configuration": {"m": 512, "n": 512, "k": 512},
            "latency_ms": {"count": 0, "median": 100.0},
        }
        plan = generate_experiment_plan(baseline)
        self.assertEqual(plan.status, ExperimentStatus.INSUFFICIENT_BASELINE)

    def test_20_type_error_on_none(self):
        """20. None baseline raises TypeError."""
        with self.assertRaises(TypeError):
            generate_experiment_plan(None)

    # ==================================================================
    # 21-25: Candidate ordering, priorities, and plan structure
    # ==================================================================

    def test_21_candidates_sorted_by_priority(self):
        """21. Candidates are sorted by priority (HIGH first) then risk (LOW first)."""
        baseline = _make_baseline()
        plan = generate_experiment_plan(baseline)
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        for i in range(len(plan.candidates) - 1):
            c1 = plan.candidates[i]
            c2 = plan.candidates[i + 1]
            key1 = (priority_order.get(c1.priority, 9), risk_order.get(c1.risk, 9))
            key2 = (priority_order.get(c2.priority, 9), risk_order.get(c2.risk, 9))
            self.assertLessEqual(key1, key2, f"{c1.experiment_id} should come before {c2.experiment_id}")

    def test_22_execution_order_matches_candidates(self):
        """22. Execution order list matches candidate IDs in order."""
        baseline = _make_baseline()
        plan = generate_experiment_plan(baseline)
        expected_order = [c.experiment_id for c in plan.candidates]
        self.assertEqual(plan.execution_order, expected_order)

    def test_23_all_candidates_have_unique_ids(self):
        """23. All experiment IDs are unique within a plan."""
        baseline = _make_baseline()
        plan = generate_experiment_plan(baseline)
        ids = [c.experiment_id for c in plan.candidates]
        self.assertEqual(len(ids), len(set(ids)))

    def test_24_plan_json_is_complete(self):
        """24. Plan JSON contains all required top-level keys."""
        baseline = _make_baseline()
        plan = generate_experiment_plan(baseline)
        d = plan.to_dict()
        required_keys = [
            "schema_version", "status", "summary", "baseline_analysis",
            "candidates", "execution_order", "limitations", "generated_at",
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_25_limitations_present(self):
        """25. Default limitations are included in generated plan."""
        baseline = _make_baseline()
        plan = generate_experiment_plan(baseline)
        self.assertTrue(len(plan.limitations) >= 3)
        for lim in DEFAULT_EXPERIMENT_LIMITATIONS:
            self.assertIn(lim, plan.limitations)

    # ==================================================================
    # 26-30: Integration and additional scenarios
    # ==================================================================

    def test_26_small_matrix_generates_fewer_candidates(self):
        """26. Small matrix (4x4x4) skips tiling and Strassen."""
        baseline = _make_baseline(m=4, n=4, k=4, median=0.01, gflops=None)
        plan = generate_experiment_plan(baseline)
        self.assertEqual(plan.status, ExperimentStatus.EXPERIMENTS_GENERATED)
        ids = [c.experiment_id for c in plan.candidates]
        # No tiling for m<16
        tiling_present = any("tile" in eid for eid in ids)
        self.assertFalse(tiling_present, "Tiling should not be generated for 4x4x4")
        # No Strassen for min_dim<32
        strassen_present = any("strassen" in eid for eid in ids)
        self.assertFalse(strassen_present, "Strassen should not be generated for 4x4x4")

    def test_27_single_core_skips_parallelism(self):
        """27. Single-core system skips parallelism candidate."""
        baseline = _make_baseline(physical_cores=1, logical_cores=1)
        plan = generate_experiment_plan(baseline)
        ids = [c.experiment_id for c in plan.candidates]
        parallel_present = any("parallel" in eid for eid in ids)
        self.assertFalse(parallel_present, "Parallelism should not be generated for single-core")

    def test_28_plan_experiments_alias(self):
        """28. plan_experiments is an alias for generate_experiment_plan."""
        self.assertIs(plan_experiments, generate_experiment_plan)

    def test_29_to_dict_safe_with_object(self):
        """29. _to_dict_safe handles objects with to_dict method."""
        class MockResult:
            def to_dict(self):
                return {"test": True}

        result = _to_dict_safe(MockResult())
        self.assertEqual(result, {"test": True})

        # None returns None
        self.assertIsNone(_to_dict_safe(None))

        # Dict passes through
        d = {"a": 1}
        self.assertIs(_to_dict_safe(d), d)

    def test_30_load_baseline_from_file(self):
        """30. load_baseline_from_file reads and parses JSON correctly."""
        baseline = _make_baseline()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(baseline, f)
            tmp_path = f.name

        try:
            loaded = load_baseline_from_file(tmp_path)
            self.assertEqual(loaded["configuration"]["m"], 512)
            self.assertEqual(loaded["latency_ms"]["median"], 1342.0)
        finally:
            os.unlink(tmp_path)

    # ==================================================================
    # 31-33: CLI, file-not-found, and no-mutation
    # ==================================================================

    def test_31_load_baseline_file_not_found(self):
        """31. load_baseline_from_file raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            load_baseline_from_file("/nonexistent/path/baseline.json")

    def test_32_no_mutation_of_environment(self):
        """32. Experiment planning never mutates os.environ."""
        env_before = dict(os.environ)
        baseline = _make_baseline()
        plan = generate_experiment_plan(baseline)
        env_after = dict(os.environ)
        self.assertEqual(env_before, env_after)

    def test_33_cli_execution(self):
        """33. python -m src.experiment executes and outputs valid JSON."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "src.experiment"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        # Output should contain JSON
        output = result.stdout
        self.assertIn("EXPERIMENTS_GENERATED", output)
        # Find the JSON portion and verify it parses
        json_start = output.find("{")
        if json_start >= 0:
            json_text = output[json_start:]
            parsed = json.loads(json_text)
            self.assertEqual(parsed["schema_version"], "1.0")
            self.assertIn("candidates", parsed)


if __name__ == "__main__":
    unittest.main()
