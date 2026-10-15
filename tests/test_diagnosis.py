"""Tests for Root-Cause and Evidence Diagnosis Engine V1 (src/diagnosis.py).

Verifies deterministic, explainable, and conservative diagnosis of performance
regressions without claiming unsupported causality.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from typing import Any, Dict

from src.diagnosis import (
    ChangePriority,
    ChangeRole,
    DetectedChange,
    DiagnosisResult,
    DiagnosisStatus,
    diff_dicts,
    diagnose_regression,
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
    stddev: float = 1.0,
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
        },
        "latency_ms": {
            "count": measurement_runs,
            "min": median_latency_ms - 2.0,
            "mean": median_latency_ms + 0.1,
            "median": median_latency_ms,
            "p95": median_latency_ms + 2.0,
            "p99": median_latency_ms + 3.0,
            "max": median_latency_ms + 4.0,
            "stddev": stddev,
        },
        "throughput": {"operations_per_second": (operation_count / (median_latency_ms / 1000.0))},
    }


def _create_sample_hardware(
    gpu_name: str = "AMD Radeon (TM) Graphics",
    rocm_version: str = "6.2.0",
    rocm_available: bool = True,
    linux_has_gpu: bool = True,
    cpu_model: str = "AMD Ryzen 7 7730U",
    logical_cores: int = 16,
    linux_total_mb: int = 7620,
    is_wsl: bool = True,
) -> Dict[str, Any]:
    """Helper to generate a deterministic synthetic HardwareInfo dictionary."""
    return {
        "cpu": {
            "vendor": "AuthenticAMD",
            "model_name": cpu_model,
            "physical_cores": 8,
            "logical_cores": logical_cores,
        },
        "memory": {
            "linux_total_mb": linux_total_mb,
            "host_total_mb": 16384,
        },
        "os": {
            "system": "Linux",
            "kernel_release": "5.15.167.4-microsoft-standard-WSL2",
        },
        "wsl": {
            "is_wsl": is_wsl,
            "wsl_version": 2,
        },
        "gpu": {
            "linux_has_gpu": linux_has_gpu,
            "wsl_has_host_gpu": True,
            "host_devices": [{"name": gpu_name}],
        },
        "amd": {
            "is_amd_hardware": True,
            "gpu_is_amd": True,
        },
        "rocm": {
            "is_available": rocm_available,
            "version": rocm_version,
            "kfd_available": True,
        },
    }


def _create_sample_environment(
    python_version: str = "3.13.1",
    rocm_version: str = "6.2.0",
    rocm_available: bool = True,
    torch_version: str = "2.4.0+rocm6.2",
    torch_installed: bool = True,
    rocm_path: str = "/opt/rocm-6.2.0",
    omp_num_threads: str = "8",
    unrelated_var: str = "default_value",
) -> Dict[str, Any]:
    """Helper to generate a deterministic synthetic EnvironmentInfo dictionary."""
    return {
        "python": {
            "version": python_version,
            "implementation": "CPython",
        },
        "os": {
            "distribution": "Ubuntu 24.04.5 LTS",
            "kernel_release": "5.15.167.4-microsoft-standard-WSL2",
        },
        "rocm": {
            "available": rocm_available,
            "version": rocm_version,
            "hipconfig": {"version": rocm_version},
        },
        "python_packages": {
            "torch": {"installed": torch_installed, "version": torch_version if torch_installed else None},
            "transformers": {"installed": True, "version": "4.44.0"},
            "unrelated_pkg": {"installed": True, "version": "1.0.0"},
        },
        "environment_variables": {
            "ROCM_PATH": rocm_path,
            "OMP_NUM_THREADS": omp_num_threads,
            "UNRELATED_VAR": unrelated_var,
        },
        "cpu_performance": {
            "scaling_governor": "powersave",
        },
        "build_tools": {
            "gcc": {"available": True, "version": "13.3.0"},
        },
    }


class TestDiagnosisEngine(unittest.TestCase):
    """Unit tests for the Root-Cause and Evidence Diagnosis Engine V1."""

    def test_rocm_version_change_is_high_priority(self) -> None:
        """1. Regression + ROCm version change -> ROCm appears as HIGH priority potential contributor."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)  # +30% regression

        base_env = _create_sample_environment(rocm_version="6.2.0")
        curr_env = _create_sample_environment(rocm_version="7.0.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        self.assertEqual(res.status, DiagnosisStatus.POTENTIAL_CAUSES_FOUND)
        self.assertEqual(res.regression_status, "REGRESSION")

        rocm_changes = [c for c in res.changes if "rocm.version" in c.field]
        self.assertTrue(len(rocm_changes) > 0)
        top_rocm = rocm_changes[0]
        self.assertEqual(top_rocm.priority, ChangePriority.HIGH.value)
        self.assertEqual(top_rocm.investigation_priority_score, 3)
        self.assertEqual(top_rocm.role, ChangeRole.POTENTIAL_CONTRIBUTOR.value)

    def test_gpu_identity_change_is_high_priority(self) -> None:
        """2. Regression + GPU identity change -> GPU change appears HIGH priority."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_hw = _create_sample_hardware(gpu_name="AMD Radeon (TM) Graphics")
        curr_hw = _create_sample_hardware(gpu_name="AMD Radeon RX 7900 XTX")

        res = diagnose_regression(base_bm, curr_bm, baseline_hardware=base_hw, current_hardware=curr_hw)
        self.assertEqual(res.status, DiagnosisStatus.POTENTIAL_CAUSES_FOUND)

        gpu_changes = [c for c in res.changes if "gpu.host_devices" in c.field]
        self.assertEqual(len(gpu_changes), 1)
        self.assertEqual(gpu_changes[0].priority, ChangePriority.HIGH.value)
        self.assertEqual(gpu_changes[0].investigation_priority_score, 3)
        self.assertEqual(gpu_changes[0].role, ChangeRole.POTENTIAL_CONTRIBUTOR.value)

    def test_python_version_change_is_medium_priority(self) -> None:
        """3. Regression + Python version change -> Python appears MEDIUM priority."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(python_version="3.12.0")
        curr_env = _create_sample_environment(python_version="3.13.1")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        self.assertEqual(res.status, DiagnosisStatus.POTENTIAL_CAUSES_FOUND)

        py_changes = [c for c in res.changes if "python.version" in c.field]
        self.assertEqual(len(py_changes), 1)
        self.assertEqual(py_changes[0].priority, ChangePriority.MEDIUM.value)
        self.assertEqual(py_changes[0].investigation_priority_score, 2)
        self.assertEqual(py_changes[0].role, ChangeRole.POTENTIAL_CONTRIBUTOR.value)

    def test_relevant_package_upgrade_detected(self) -> None:
        """4. Regression + relevant package upgrade -> package change detected with correct priority."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(torch_version="2.4.0")
        curr_env = _create_sample_environment(torch_version="2.5.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        torch_changes = [c for c in res.changes if "torch" in c.field]
        self.assertEqual(len(torch_changes), 1)
        self.assertEqual(torch_changes[0].change_type, "upgraded")
        self.assertEqual(torch_changes[0].priority, ChangePriority.MEDIUM.value)

    def test_benchmark_configuration_change(self) -> None:
        """5. Regression + benchmark configuration change -> CONFIGURATION_CHANGED."""
        base_bm = _create_sample_benchmark(warmup_runs=5, measurement_runs=30)
        curr_bm = _create_sample_benchmark(warmup_runs=10, measurement_runs=30)

        res = diagnose_regression(base_bm, curr_bm)
        self.assertEqual(res.status, DiagnosisStatus.CONFIGURATION_CHANGED)
        self.assertIn("Benchmark configuration changed", res.summary)
        self.assertTrue(any("configuration" in c.category for c in res.changes))

    def test_regression_with_no_relevant_changes(self) -> None:
        """6. Regression + no relevant changes -> INSUFFICIENT_EVIDENCE."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        env = _create_sample_environment()
        hw = _create_sample_hardware()

        res = diagnose_regression(base_bm, curr_bm, baseline_hardware=hw, current_hardware=hw, baseline_environment=env, current_environment=env)
        self.assertEqual(res.status, DiagnosisStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn("no relevant configuration or environment changes", res.summary)
        self.assertEqual(len(res.changes), 0)

    def test_no_regression_to_diagnose(self) -> None:
        """7. No regression -> NO_REGRESSION_TO_DIAGNOSE."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=50.2)  # within variance

        env1 = _create_sample_environment(rocm_version="6.2.0")
        env2 = _create_sample_environment(rocm_version="7.0.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=env1, current_environment=env2)
        self.assertEqual(res.status, DiagnosisStatus.NO_REGRESSION_TO_DIAGNOSE)
        self.assertEqual(len(res.changes), 0)

    def test_improvement_diagnosis(self) -> None:
        """8. Improvement -> diagnosis still works and reports observed changes."""
        base_bm = _create_sample_benchmark(median_latency_ms=60.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=45.0)  # -25% improvement

        base_env = _create_sample_environment(rocm_version="6.2.0")
        curr_env = _create_sample_environment(rocm_version="7.0.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        self.assertEqual(res.status, DiagnosisStatus.POTENTIAL_CAUSES_FOUND)
        self.assertIn("Performance improved", res.summary)
        self.assertTrue(len(res.changes) > 0)

    def test_missing_environment_snapshots(self) -> None:
        """9. Missing environment and hardware snapshots -> INSUFFICIENT_EVIDENCE."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=None, current_environment=None)
        self.assertEqual(res.status, DiagnosisStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn("snapshots were not supplied", res.summary)

    def test_missing_hardware_snapshots_graceful(self) -> None:
        """10. Missing hardware snapshot only -> graceful degradation, detects environment changes."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(rocm_version="6.2.0")
        curr_env = _create_sample_environment(rocm_version="7.0.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_hardware=None, current_hardware=None, baseline_environment=base_env, current_environment=curr_env)
        self.assertEqual(res.status, DiagnosisStatus.POTENTIAL_CAUSES_FOUND)
        self.assertTrue(any("rocm" in c.field for c in res.changes))

    def test_multiple_changes_deterministic_ranking(self) -> None:
        """11. Multiple changes across tiers -> deterministic ranking (HIGH before MEDIUM before LOW)."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(
            python_version="3.12.0",
            rocm_version="6.2.0",
            torch_version="2.4.0",
            unrelated_var="v1",
        )
        curr_env = _create_sample_environment(
            python_version="3.13.1",
            rocm_version="7.0.0",
            torch_version="2.5.0",
            unrelated_var="v2",
        )

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        priorities = [c.priority for c in res.changes]

        # Verify that all HIGH priority changes come before MEDIUM, and MEDIUM before LOW
        high_idx = [i for i, p in enumerate(priorities) if p == ChangePriority.HIGH.value]
        med_idx = [i for i, p in enumerate(priorities) if p == ChangePriority.MEDIUM.value]
        low_idx = [i for i, p in enumerate(priorities) if p == ChangePriority.LOW.value]

        if high_idx and med_idx:
            self.assertLess(max(high_idx), min(med_idx))
        if med_idx and low_idx:
            self.assertLess(max(med_idx), min(low_idx))

    def test_unrelated_environment_changes_low_priority(self) -> None:
        """12. Unrelated environment variables/tools have LOW priority and OBSERVED_CHANGE role."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(unrelated_var="val1")
        curr_env = _create_sample_environment(unrelated_var="val2")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        unrelated = [c for c in res.changes if "UNRELATED_VAR" in c.field]
        self.assertEqual(len(unrelated), 1)
        self.assertEqual(unrelated[0].priority, ChangePriority.LOW.value)
        self.assertEqual(unrelated[0].investigation_priority_score, 1)
        self.assertEqual(unrelated[0].role, ChangeRole.OBSERVED_CHANGE.value)

    def test_package_added(self) -> None:
        """13. Package added -> detected with change_type='added'."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(torch_installed=False)
        curr_env = _create_sample_environment(torch_installed=True, torch_version="2.5.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        torch_changes = [c for c in res.changes if "torch" in c.field]
        self.assertEqual(len(torch_changes), 1)
        self.assertEqual(torch_changes[0].change_type, "added")

    def test_package_removed(self) -> None:
        """14. Package removed -> detected with change_type='removed'."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(torch_installed=True, torch_version="2.4.0")
        curr_env = _create_sample_environment(torch_installed=False)

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        torch_changes = [c for c in res.changes if "torch" in c.field]
        self.assertEqual(len(torch_changes), 1)
        self.assertEqual(torch_changes[0].change_type, "removed")

    def test_package_upgraded(self) -> None:
        """15. Package upgraded -> detected with change_type='upgraded'."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(torch_version="2.4.0")
        curr_env = _create_sample_environment(torch_version="2.5.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        torch_changes = [c for c in res.changes if "torch" in c.field]
        self.assertEqual(torch_changes[0].change_type, "upgraded")

    def test_package_downgraded(self) -> None:
        """16. Package downgraded -> detected with change_type='downgraded'."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(torch_version="2.5.0")
        curr_env = _create_sample_environment(torch_version="2.4.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        torch_changes = [c for c in res.changes if "torch" in c.field]
        self.assertEqual(torch_changes[0].change_type, "downgraded")

    def test_environment_variable_changed(self) -> None:
        """17. Environment variable changed -> detected with proper classification."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(omp_num_threads="8")
        curr_env = _create_sample_environment(omp_num_threads="2")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        omp_changes = [c for c in res.changes if "OMP_NUM_THREADS" in c.field]
        self.assertEqual(len(omp_changes), 1)
        self.assertEqual(omp_changes[0].priority, ChangePriority.MEDIUM.value)
        self.assertEqual(omp_changes[0].role, ChangeRole.POTENTIAL_CONTRIBUTOR.value)

    def test_gpu_availability_changed(self) -> None:
        """18. GPU availability changed -> detected with HIGH priority."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_hw = _create_sample_hardware(linux_has_gpu=True)
        curr_hw = _create_sample_hardware(linux_has_gpu=False)

        res = diagnose_regression(base_bm, curr_bm, baseline_hardware=base_hw, current_hardware=curr_hw)
        gpu_changes = [c for c in res.changes if "linux_has_gpu" in c.field]
        self.assertEqual(len(gpu_changes), 1)
        self.assertEqual(gpu_changes[0].priority, ChangePriority.HIGH.value)

    def test_rocm_availability_changed(self) -> None:
        """19. ROCm availability changed -> detected with HIGH priority."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_hw = _create_sample_hardware(rocm_available=True)
        curr_hw = _create_sample_hardware(rocm_available=False)

        res = diagnose_regression(base_bm, curr_bm, baseline_hardware=base_hw, current_hardware=curr_hw)
        rocm_changes = [c for c in res.changes if "rocm.is_available" in c.field]
        self.assertEqual(len(rocm_changes), 1)
        self.assertEqual(rocm_changes[0].priority, ChangePriority.HIGH.value)

    def test_result_is_json_serializable(self) -> None:
        """20. Result is strictly JSON serializable."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)
        base_env = _create_sample_environment(rocm_version="6.2.0")
        curr_env = _create_sample_environment(rocm_version="7.0.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        json_str = res.to_json(indent=2)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(parsed["status"], "POTENTIAL_CAUSES_FOUND")
        self.assertIsInstance(parsed["changes"], list)
        self.assertIsInstance(parsed["limitations"], list)

    def test_explanations_contain_no_unsupported_causal_claims(self) -> None:
        """21. Explanations contain no unsupported causal claims (e.g. 'definitely caused', 'statistically significant')."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)
        base_env = _create_sample_environment(rocm_version="6.2.0")
        curr_env = _create_sample_environment(rocm_version="7.0.0")

        res = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)

        forbidden_phrases = [
            "definitely caused",
            "proves causality",
            "proven cause",
            "statistically significant",
        ]

        for change in res.changes:
            expl_lower = change.explanation.lower()
            for phrase in forbidden_phrases:
                self.assertNotIn(phrase, expl_lower)

        # Check limitations explicitly note causality limits
        lim_text = " ".join(res.limitations).lower()
        self.assertIn("do not prove causality", lim_text)

    def test_benchmark_configuration_parameter_difference(self) -> None:
        """22. Workload configuration parameter difference is detected."""
        base_bm = _create_sample_benchmark()
        curr_bm = _create_sample_benchmark()
        base_bm["configuration"]["batch_size"] = 1
        curr_bm["configuration"]["batch_size"] = 8

        res = diagnose_regression(base_bm, curr_bm)
        self.assertEqual(res.status, DiagnosisStatus.CONFIGURATION_CHANGED)
        batch_changes = [c for c in res.changes if "batch_size" in c.field]
        self.assertEqual(len(batch_changes), 1)
        self.assertEqual(batch_changes[0].baseline, 1)
        self.assertEqual(batch_changes[0].current, 8)

    def test_recursive_nested_dictionary_comparison(self) -> None:
        """23. Safe recursive nested dictionary comparison utility."""
        base = {
            "runtime": {
                "engine": {"version": "1.0", "flags": ["-O3"]},
                "unchanged": 42,
            },
            "timestamp": "2026-10-01T00:00:00",
        }
        curr = {
            "runtime": {
                "engine": {"version": "2.0", "flags": ["-O3", "-fPIC"]},
                "unchanged": 42,
            },
            "timestamp": "2026-10-02T00:00:00",
        }

        diffs = diff_dicts(base, curr)
        fields = [d["field"] for d in diffs]
        self.assertIn("runtime.engine.version", fields)
        self.assertIn("runtime.engine.flags", fields)
        self.assertNotIn("runtime.unchanged", fields)
        self.assertNotIn("timestamp", fields)  # timestamp ignored

    def test_empty_snapshots_handled_gracefully(self) -> None:
        """24. Empty snapshots do not cause crashes."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        res = diagnose_regression(base_bm, curr_bm, baseline_hardware={}, current_hardware={}, baseline_environment={}, current_environment={})
        self.assertIn(res.status, (DiagnosisStatus.INSUFFICIENT_EVIDENCE, DiagnosisStatus.NO_CHANGES_DETECTED))

    def test_deterministic_ordering_of_diagnosis_results(self) -> None:
        """25. Repeated diagnosis produces identical ordering of detected changes."""
        base_bm = _create_sample_benchmark(median_latency_ms=50.0)
        curr_bm = _create_sample_benchmark(median_latency_ms=65.0)

        base_env = _create_sample_environment(
            python_version="3.12.0",
            rocm_version="6.2.0",
            torch_version="2.4.0",
            unrelated_var="v1",
        )
        curr_env = _create_sample_environment(
            python_version="3.13.1",
            rocm_version="7.0.0",
            torch_version="2.5.0",
            unrelated_var="v2",
        )

        res1 = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)
        res2 = diagnose_regression(base_bm, curr_bm, baseline_environment=base_env, current_environment=curr_env)

        fields1 = [c.field for c in res1.changes]
        fields2 = [c.field for c in res2.changes]
        self.assertEqual(fields1, fields2)

    def test_cli_execution(self) -> None:
        """26. Verify python -m src.diagnosis runs and returns valid JSON demonstration."""
        cmd = [sys.executable, "-m", "src.diagnosis"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        self.assertIn("=== ROCm Performance Guard — Evidence Diagnosis Engine V1 Demo ===", result.stdout)

        # Parse JSON portion of CLI output
        lines = result.stdout.splitlines()
        json_lines = [l for l in lines if not l.startswith("===")]
        json_str = "\n".join(json_lines).strip()
        data = json.loads(json_str)

        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["status"], "POTENTIAL_CAUSES_FOUND")
        self.assertIn("rocm.version", [c["field"] for c in data["changes"]])


if __name__ == "__main__":
    unittest.main()
