"""Tests for Verification Engine V1.

Validates deterministic, explainable, and threshold-based optimization verification
without automatic system mutations, statistical exaggeration, or causal claims.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import unittest
from typing import Any, Dict

from src.benchmark import BenchmarkResult
from src.optimizer import OptimizationCandidate
from src.verification import (
    DEFAULT_VERIFICATION_LIMITATIONS,
    MetricDirection,
    VerificationConfig,
    VerificationResult,
    VerificationStatus,
    verify,
    verify_optimization,
)


def make_benchmark_fixture(
    name: str = "gemm_fp16",
    version: str = "1.0.0",
    fingerprint: str = "fp_abc123",
    count: int = 30,
    median: float = 100.0,
    mean: float = 101.0,
    p95: float = 105.0,
    p99: float = 108.0,
    stddev: float = 1.5,
    throughput_ops: float = 10000.0,
) -> Dict[str, Any]:
    """Helper creating standard synthetic benchmark results for verification tests."""
    return {
        "schema_version": "1.0",
        "fingerprint": fingerprint,
        "benchmark": {"name": name, "version": version},
        "configuration": {"warmup_runs": 5, "measurement_runs": count, "operation_count": 1000},
        "latency_ms": {
            "count": count,
            "median": median,
            "mean": mean,
            "p95": p95,
            "p99": p99,
            "stddev": stddev,
        },
        "throughput": {
            "operations_per_second": throughput_ops,
        },
    }


class TestVerificationEngine(unittest.TestCase):
    """Comprehensive test suite for Verification Engine V1."""

    def test_01_latency_improvement_above_threshold(self) -> None:
        """1. Latency improvement exceeding threshold yields IMPROVED."""
        baseline = make_benchmark_fixture(median=100.0)
        candidate = make_benchmark_fixture(median=90.0)  # -10% latency (improvement)
        result = verify_optimization(baseline, candidate, config=VerificationConfig(improvement_threshold_percent=5.0))

        self.assertEqual(result.status, VerificationStatus.IMPROVED)
        self.assertIsNotNone(result.metric)
        self.assertEqual(result.metric["direction"], MetricDirection.BETTER.value)
        self.assertAlmostEqual(result.metric["percentage_change"], -10.0)
        self.assertIn("improved by 10.0%", result.explanation)

    def test_02_latency_improvement_below_threshold(self) -> None:
        """2. Minor latency improvement below threshold yields NO_MEANINGFUL_CHANGE."""
        baseline = make_benchmark_fixture(median=100.0)
        candidate = make_benchmark_fixture(median=98.0)  # -2% latency
        result = verify_optimization(baseline, candidate, config=VerificationConfig(improvement_threshold_percent=5.0))

        self.assertEqual(result.status, VerificationStatus.NO_MEANINGFUL_CHANGE)
        self.assertIsNotNone(result.metric)
        self.assertEqual(result.metric["direction"], MetricDirection.BETTER.value)
        self.assertIn("below the configured", result.explanation)

    def test_03_latency_regression_above_threshold(self) -> None:
        """3. Latency increase exceeding regression threshold yields REGRESSED."""
        baseline = make_benchmark_fixture(median=100.0)
        candidate = make_benchmark_fixture(median=110.0)  # +10% latency (regression)
        result = verify_optimization(baseline, candidate, config=VerificationConfig(regression_threshold_percent=5.0))

        self.assertEqual(result.status, VerificationStatus.REGRESSED)
        self.assertIsNotNone(result.metric)
        self.assertEqual(result.metric["direction"], MetricDirection.WORSE.value)
        self.assertAlmostEqual(result.metric["percentage_change"], 10.0)
        self.assertIn("worsened by 10.0%", result.explanation)

    def test_04_throughput_improvement_above_threshold(self) -> None:
        """4. Throughput gain exceeding threshold yields IMPROVED."""
        baseline = make_benchmark_fixture(throughput_ops=100.0)
        candidate = make_benchmark_fixture(throughput_ops=115.0)  # +15% throughput
        cfg = VerificationConfig(metric="throughput", improvement_threshold_percent=5.0)
        result = verify_optimization(baseline, candidate, config=cfg)

        self.assertEqual(result.status, VerificationStatus.IMPROVED)
        self.assertIsNotNone(result.metric)
        self.assertEqual(result.metric["direction"], MetricDirection.BETTER.value)
        self.assertAlmostEqual(result.metric["percentage_change"], 15.0)
        self.assertIn("improved by 15.0%", result.explanation)

    def test_05_throughput_improvement_below_threshold(self) -> None:
        """5. Throughput gain below threshold yields NO_MEANINGFUL_CHANGE."""
        baseline = make_benchmark_fixture(throughput_ops=100.0)
        candidate = make_benchmark_fixture(throughput_ops=102.0)  # +2% throughput
        cfg = VerificationConfig(metric="throughput", improvement_threshold_percent=5.0)
        result = verify_optimization(baseline, candidate, config=cfg)

        self.assertEqual(result.status, VerificationStatus.NO_MEANINGFUL_CHANGE)
        self.assertIsNotNone(result.metric)
        self.assertEqual(result.metric["direction"], MetricDirection.BETTER.value)
        self.assertIn("below the configured", result.explanation)

    def test_06_throughput_regression(self) -> None:
        """6. Throughput reduction exceeding regression threshold yields REGRESSED."""
        baseline = make_benchmark_fixture(throughput_ops=100.0)
        candidate = make_benchmark_fixture(throughput_ops=90.0)  # -10% throughput
        cfg = VerificationConfig(metric="throughput", regression_threshold_percent=5.0)
        result = verify_optimization(baseline, candidate, config=cfg)

        self.assertEqual(result.status, VerificationStatus.REGRESSED)
        self.assertIsNotNone(result.metric)
        self.assertEqual(result.metric["direction"], MetricDirection.WORSE.value)
        self.assertAlmostEqual(result.metric["percentage_change"], -10.0)
        self.assertIn("worsened by 10.0%", result.explanation)

    def test_07_identical_results(self) -> None:
        """7. Identical benchmark metrics yield NO_MEANINGFUL_CHANGE with neutral direction."""
        baseline = make_benchmark_fixture(median=100.0)
        candidate = make_benchmark_fixture(median=100.0)
        result = verify_optimization(baseline, candidate)

        self.assertEqual(result.status, VerificationStatus.NO_MEANINGFUL_CHANGE)
        self.assertIsNotNone(result.metric)
        self.assertEqual(result.metric["direction"], MetricDirection.NEUTRAL.value)
        self.assertEqual(result.metric["absolute_change"], 0.0)
        self.assertEqual(result.metric["percentage_change"], 0.0)

    def test_08_incompatible_benchmark_name(self) -> None:
        """8. Mismatched benchmark names yield INCOMPARABLE."""
        baseline = make_benchmark_fixture(name="gemm_fp16")
        candidate = make_benchmark_fixture(name="bert_latency")
        result = verify_optimization(baseline, candidate)

        self.assertEqual(result.status, VerificationStatus.INCOMPARABLE)
        self.assertIsNone(result.metric)
        self.assertIn("Benchmark names differ", result.explanation)

    def test_09_incompatible_benchmark_version(self) -> None:
        """9. Mismatched benchmark versions yield INCOMPARABLE."""
        baseline = make_benchmark_fixture(version="1.0.0")
        candidate = make_benchmark_fixture(version="2.0.0")
        result = verify_optimization(baseline, candidate)

        self.assertEqual(result.status, VerificationStatus.INCOMPARABLE)
        self.assertIsNone(result.metric)
        self.assertIn("Benchmark versions differ", result.explanation)

    def test_10_incompatible_configuration_fingerprint(self) -> None:
        """10. Mismatched configuration fingerprints yield INCOMPARABLE."""
        baseline = make_benchmark_fixture(fingerprint="batch_size_1_fp")
        candidate = make_benchmark_fixture(fingerprint="batch_size_8_fp")
        result = verify_optimization(baseline, candidate)

        self.assertEqual(result.status, VerificationStatus.INCOMPARABLE)
        self.assertIsNone(result.metric)
        self.assertIn("configuration fingerprints differ", result.explanation)

    def test_11_insufficient_baseline_measurements(self) -> None:
        """11. Measurement count below threshold in baseline yields INSUFFICIENT_DATA."""
        baseline = make_benchmark_fixture(count=3)
        candidate = make_benchmark_fixture(count=30)
        cfg = VerificationConfig(minimum_measurements=5)
        result = verify_optimization(baseline, candidate, config=cfg)

        self.assertEqual(result.status, VerificationStatus.INSUFFICIENT_DATA)
        self.assertIsNone(result.metric)
        self.assertIn("Baseline contains only 3", result.explanation)

    def test_12_insufficient_candidate_measurements(self) -> None:
        """12. Measurement count below threshold in candidate yields INSUFFICIENT_DATA."""
        baseline = make_benchmark_fixture(count=30)
        candidate = make_benchmark_fixture(count=4)
        cfg = VerificationConfig(minimum_measurements=5)
        result = verify_optimization(baseline, candidate, config=cfg)

        self.assertEqual(result.status, VerificationStatus.INSUFFICIENT_DATA)
        self.assertIsNone(result.metric)
        self.assertIn("Candidate contains only 4", result.explanation)

    def test_13_zero_baseline(self) -> None:
        """13. Zero baseline value avoids division by zero and yields NO_VERIFICATION_POSSIBLE."""
        baseline = make_benchmark_fixture(median=0.0)
        candidate = make_benchmark_fixture(median=10.0)
        result = verify_optimization(baseline, candidate)

        self.assertEqual(result.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIsNone(result.metric)
        self.assertIn("zero", result.explanation)

    def test_14_missing_requested_metric(self) -> None:
        """14. Requesting an unrecorded or unsupported metric yields NO_VERIFICATION_POSSIBLE."""
        baseline = make_benchmark_fixture()
        candidate = make_benchmark_fixture()
        cfg = VerificationConfig(metric="non_existent_metric")
        result = verify_optimization(baseline, candidate, config=cfg)

        self.assertEqual(result.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIsNone(result.metric)
        self.assertIn("not supported", result.explanation)

    def test_15_invalid_latency_value(self) -> None:
        """15. NaN latency value is safely rejected with NO_VERIFICATION_POSSIBLE."""
        baseline = make_benchmark_fixture(median=float("nan"))
        candidate = make_benchmark_fixture(median=90.0)
        result = verify_optimization(baseline, candidate)

        self.assertEqual(result.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIsNone(result.metric)
        self.assertIn("invalid value", result.explanation)

    def test_16_invalid_throughput_value(self) -> None:
        """16. NaN throughput value is safely rejected with NO_VERIFICATION_POSSIBLE."""
        baseline = make_benchmark_fixture(throughput_ops=100.0)
        candidate = make_benchmark_fixture(throughput_ops=float("nan"))
        cfg = VerificationConfig(metric="throughput")
        result = verify_optimization(baseline, candidate, config=cfg)

        self.assertEqual(result.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIsNone(result.metric)
        self.assertIn("invalid value", result.explanation)

    def test_17_configurable_improvement_threshold(self) -> None:
        """17. Custom improvement threshold modifies classification accordingly."""
        baseline = make_benchmark_fixture(median=100.0)
        candidate = make_benchmark_fixture(median=92.0)  # -8% improvement

        # With 5% threshold -> IMPROVED
        res_5 = verify_optimization(baseline, candidate, config=VerificationConfig(improvement_threshold_percent=5.0))
        self.assertEqual(res_5.status, VerificationStatus.IMPROVED)

        # With 10% threshold -> NO_MEANINGFUL_CHANGE
        res_10 = verify_optimization(baseline, candidate, config=VerificationConfig(improvement_threshold_percent=10.0))
        self.assertEqual(res_10.status, VerificationStatus.NO_MEANINGFUL_CHANGE)

    def test_18_configurable_regression_threshold(self) -> None:
        """18. Custom regression threshold modifies classification accordingly."""
        baseline = make_benchmark_fixture(median=100.0)
        candidate = make_benchmark_fixture(median=108.0)  # +8% regression

        # With 5% threshold -> REGRESSED
        res_5 = verify_optimization(baseline, candidate, config=VerificationConfig(regression_threshold_percent=5.0))
        self.assertEqual(res_5.status, VerificationStatus.REGRESSED)

        # With 10% threshold -> NO_MEANINGFUL_CHANGE
        res_10 = verify_optimization(baseline, candidate, config=VerificationConfig(regression_threshold_percent=10.0))
        self.assertEqual(res_10.status, VerificationStatus.NO_MEANINGFUL_CHANGE)

    def test_19_configurable_minimum_measurement_count(self) -> None:
        """19. Custom minimum measurement count modifies sufficiency validation."""
        baseline = make_benchmark_fixture(count=8)
        candidate = make_benchmark_fixture(count=8)

        # With min=5 -> valid
        res_5 = verify_optimization(baseline, candidate, config=VerificationConfig(minimum_measurements=5))
        self.assertEqual(res_5.status, VerificationStatus.NO_MEANINGFUL_CHANGE)

        # With min=10 -> INSUFFICIENT_DATA
        res_10 = verify_optimization(baseline, candidate, config=VerificationConfig(minimum_measurements=10))
        self.assertEqual(res_10.status, VerificationStatus.INSUFFICIENT_DATA)

    def test_20_configurable_metric_selection(self) -> None:
        """20. Configurable metric selection supports mean, p95, p99, and throughput."""
        base = make_benchmark_fixture(mean=100.0, p95=120.0, p99=140.0, throughput_ops=500.0)
        cand = make_benchmark_fixture(mean=85.0, p95=105.0, p99=120.0, throughput_ops=600.0)

        res_mean = verify_optimization(base, cand, config=VerificationConfig(metric="mean_latency_ms"))
        self.assertEqual(res_mean.status, VerificationStatus.IMPROVED)
        self.assertEqual(res_mean.metric["name"], "mean_latency_ms")

        res_p95 = verify_optimization(base, cand, config=VerificationConfig(metric="p95_latency_ms"))
        self.assertEqual(res_p95.status, VerificationStatus.IMPROVED)
        self.assertEqual(res_p95.metric["name"], "p95_latency_ms")

        res_p99 = verify_optimization(base, cand, config=VerificationConfig(metric="p99_latency_ms"))
        self.assertEqual(res_p99.status, VerificationStatus.IMPROVED)

        res_tp = verify_optimization(base, cand, config=VerificationConfig(metric="throughput"))
        self.assertEqual(res_tp.status, VerificationStatus.IMPROVED)

    def test_21_candidate_id_preserved(self) -> None:
        """21. candidate_id is preserved from string or OptimizationCandidate object."""
        base = make_benchmark_fixture(median=100.0)
        cand = make_benchmark_fixture(median=90.0)

        # From string
        res_str = verify_optimization(base, cand, candidate_info="runtime-001")
        self.assertEqual(res_str.candidate_id, "runtime-001")

        # From OptimizationCandidate instance
        opt_cand = OptimizationCandidate(
            candidate_id="environment-003",
            category="environment",
            title="Test ROCM_PATH",
            description="desc",
            rationale="rat",
            priority="HIGH",
            risk="LOW",
            expected_direction="IMPROVE",
            target="ROCM_PATH",
            proposed_change="test",
            evidence=["ev1"],
            verification_plan="plan",
        )
        res_obj = verify_optimization(base, cand, candidate_info=opt_cand)
        self.assertEqual(res_obj.candidate_id, "environment-003")

    def test_22_result_json_serializable(self) -> None:
        """22. VerificationResult produces strictly valid and parseable JSON."""
        base = make_benchmark_fixture(median=100.0)
        cand = make_benchmark_fixture(median=90.0)
        result = verify_optimization(base, cand, candidate_info="runtime-001")

        json_str = result.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(parsed["status"], "IMPROVED")
        self.assertEqual(parsed["candidate_id"], "runtime-001")

    def test_23_explanation_is_generated(self) -> None:
        """23. Explanations are non-empty and clearly describe the observed delta."""
        base = make_benchmark_fixture(median=100.0)
        cand = make_benchmark_fixture(median=85.0)
        result = verify_optimization(base, cand)

        self.assertTrue(len(result.explanation) > 10)
        self.assertIn("improved by 15.0%", result.explanation)

    def test_24_limitations_are_present(self) -> None:
        """24. Results contain explicit limitations including statistical disclaimer."""
        base = make_benchmark_fixture()
        cand = make_benchmark_fixture()
        result = verify_optimization(base, cand)

        self.assertTrue(len(result.limitations) >= 1)
        self.assertTrue(
            any("statistical significance" in lim.lower() for lim in result.limitations)
        )

    def test_25_no_statistical_significance_claim(self) -> None:
        """25. Verification does not claim causation or statistical significance."""
        base = make_benchmark_fixture(median=100.0)
        cand = make_benchmark_fixture(median=80.0)
        result = verify_optimization(base, cand)

        forbidden_phrases = [
            "statistically significant",
            "p-value",
            "definitely caused",
            "proven to cause",
            "guaranteed",
            "definitely fixed",
        ]
        explanation_lower = result.explanation.lower()
        for phrase in forbidden_phrases:
            self.assertNotIn(phrase, explanation_lower)

    def test_26_baseline_remains_unchanged(self) -> None:
        """26. The baseline structure is not modified during verification evaluation."""
        base = make_benchmark_fixture(median=100.0)
        base_copy = copy.deepcopy(base)
        cand = make_benchmark_fixture(median=90.0)

        _ = verify_optimization(base, cand)
        self.assertEqual(base, base_copy)

    def test_27_same_candidate_evaluated_against_same_baseline(self) -> None:
        """27. Multiple candidate evaluations compare independently against the same baseline."""
        baseline = make_benchmark_fixture(median=100.0)
        cand_a = make_benchmark_fixture(median=90.0)  # -10%
        cand_b = make_benchmark_fixture(median=85.0)  # -15%

        res_a = verify_optimization(baseline, cand_a, candidate_info="candidate-A")
        res_b = verify_optimization(baseline, cand_b, candidate_info="candidate-B")

        self.assertEqual(res_a.metric["baseline"], 100.0)
        self.assertEqual(res_b.metric["baseline"], 100.0)
        self.assertAlmostEqual(res_a.metric["percentage_change"], -10.0)
        self.assertAlmostEqual(res_b.metric["percentage_change"], -15.0)

    def test_28_latency_direction_is_correct(self) -> None:
        """28. Lower latency is BETTER; higher latency is WORSE."""
        base = make_benchmark_fixture(median=100.0)

        lower_lat = make_benchmark_fixture(median=90.0)
        res_better = verify_optimization(base, lower_lat)
        self.assertEqual(res_better.metric["direction"], MetricDirection.BETTER.value)

        higher_lat = make_benchmark_fixture(median=110.0)
        res_worse = verify_optimization(base, higher_lat)
        self.assertEqual(res_worse.metric["direction"], MetricDirection.WORSE.value)

    def test_29_throughput_direction_is_correct(self) -> None:
        """29. Higher throughput is BETTER; lower throughput is WORSE."""
        base = make_benchmark_fixture(throughput_ops=100.0)
        cfg = VerificationConfig(metric="throughput")

        higher_tp = make_benchmark_fixture(throughput_ops=110.0)
        res_better = verify_optimization(base, higher_tp, config=cfg)
        self.assertEqual(res_better.metric["direction"], MetricDirection.BETTER.value)

        lower_tp = make_benchmark_fixture(throughput_ops=90.0)
        res_worse = verify_optimization(base, lower_tp, config=cfg)
        self.assertEqual(res_worse.metric["direction"], MetricDirection.WORSE.value)

    def test_30_deterministic_result(self) -> None:
        """30. Multiple calls with identical inputs yield identical output."""
        base = make_benchmark_fixture(median=100.0)
        cand = make_benchmark_fixture(median=90.0)

        res_1 = verify_optimization(base, cand, candidate_info="runtime-001")
        res_2 = verify_optimization(base, cand, candidate_info="runtime-001")

        self.assertEqual(res_1.to_dict(), res_2.to_dict())

    def test_31_missing_benchmark_result_handled_safely(self) -> None:
        """31. Missing (None) baseline or candidate returns NO_VERIFICATION_POSSIBLE."""
        valid_bm = make_benchmark_fixture()

        res_base_none = verify_optimization(None, valid_bm)
        self.assertEqual(res_base_none.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIn("Baseline", res_base_none.explanation)

        res_cand_none = verify_optimization(valid_bm, None)
        self.assertEqual(res_cand_none.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIn("Candidate", res_cand_none.explanation)

        res_both_none = verify_optimization(None, None)
        self.assertEqual(res_both_none.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)

    def test_32_none_metric_handled_safely(self) -> None:
        """32. None metric value in benchmark structure returns NO_VERIFICATION_POSSIBLE."""
        base = make_benchmark_fixture()
        base["latency_ms"]["median"] = None
        cand = make_benchmark_fixture(median=90.0)

        result = verify_optimization(base, cand)
        self.assertEqual(result.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIn("missing or None", result.explanation)

    def test_33_infinity_handled_safely(self) -> None:
        """33. Infinite metric values return NO_VERIFICATION_POSSIBLE without crashing."""
        base = make_benchmark_fixture(median=100.0)
        cand = make_benchmark_fixture(median=float("inf"))

        result = verify_optimization(base, cand)
        self.assertEqual(result.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIn("invalid value", result.explanation)

    def test_34_negative_latency_rejected(self) -> None:
        """34. Negative latency values are rejected as invalid."""
        base = make_benchmark_fixture(median=-10.0)
        cand = make_benchmark_fixture(median=90.0)

        result = verify_optimization(base, cand)
        self.assertEqual(result.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIn("Negative latency", result.explanation)

    def test_35_negative_throughput_rejected(self) -> None:
        """35. Negative throughput values are rejected as invalid."""
        base = make_benchmark_fixture(throughput_ops=100.0)
        cand = make_benchmark_fixture(throughput_ops=-50.0)
        cfg = VerificationConfig(metric="throughput")

        result = verify_optimization(base, cand, config=cfg)
        self.assertEqual(result.status, VerificationStatus.NO_VERIFICATION_POSSIBLE)
        self.assertIn("Negative throughput", result.explanation)

    def test_36_no_automatic_action(self) -> None:
        """36. Verifies that verification never mutates os.environ or files."""
        environ_before = dict(os.environ)

        base = make_benchmark_fixture(median=100.0)
        cand = make_benchmark_fixture(median=90.0)
        _ = verify_optimization(base, cand, candidate_info="test-candidate")

        self.assertEqual(dict(os.environ), environ_before)

    def test_37_cli_execution(self) -> None:
        """37. Verify python -m src.verification executes and outputs valid JSON."""
        cmd = [sys.executable, "-m", "src.verification"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["status"], "IMPROVED")
        self.assertEqual(data["candidate_id"], "runtime-001")
        self.assertAlmostEqual(data["metric"]["percentage_change"], -10.0)

    def test_38_benchmark_result_instance_acceptance(self) -> None:
        """38. Accepts BenchmarkResult dataclass instances directly."""
        bm_base = BenchmarkResult(
            schema_version="1.0",
            fingerprint="fp1",
            benchmark={"name": "gemm_fp16", "version": "1.0.0"},
            configuration={"warmup_runs": 5, "measurement_runs": 30},
            latency_ms={"count": 30, "median": 100.0, "mean": 101.0, "stddev": 1.5},
            throughput={"operations_per_second": 1000.0},
        )
        bm_cand = BenchmarkResult(
            schema_version="1.0",
            fingerprint="fp1",
            benchmark={"name": "gemm_fp16", "version": "1.0.0"},
            configuration={"warmup_runs": 5, "measurement_runs": 30},
            latency_ms={"count": 30, "median": 85.0, "mean": 86.0, "stddev": 1.2},
            throughput={"operations_per_second": 1176.0},
        )

        res = verify_optimization(bm_base, bm_cand)
        self.assertEqual(res.status, VerificationStatus.IMPROVED)
        self.assertAlmostEqual(res.metric["percentage_change"], -15.0)


if __name__ == "__main__":
    unittest.main()
