"""Unit tests for Regression Detection V1."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

from src.benchmark import BenchmarkConfig, BenchmarkResult
from src.regression import (
    MetricDirection,
    RegressionConfig,
    RegressionResult,
    RegressionStatus,
    compare_benchmarks,
)


def _make_benchmark_fixture(
    name: str = "test_kernel",
    version: str = "1.0.0",
    fingerprint: str = "fp_1234567890abcdef",
    count: int = 30,
    median: float = 100.0,
    mean: float = 100.0,
    min_val: float = 95.0,
    max_val: float = 105.0,
    stddev: float = 2.0,
    throughput_ops_sec: float | None = 1000.0,
) -> dict:
    """Helper to generate a deterministic BenchmarkResult dictionary."""
    return {
        "schema_version": "1.0",
        "fingerprint": fingerprint,
        "benchmark": {"name": name, "version": version},
        "configuration": {
            "warmup_runs": 5,
            "measurement_runs": count,
            "operation_count": 100 if throughput_ops_sec is not None else None,
        },
        "latency_ms": {
            "count": count,
            "min": min_val,
            "mean": mean,
            "median": median,
            "p95": median * 1.03,
            "p99": median * 1.05,
            "max": max_val,
            "stddev": stddev,
        },
        "throughput": {"operations_per_second": throughput_ops_sec},
    }


class TestRegressionDetection(unittest.TestCase):
    """Test suite for regression analysis and threshold classification."""

    def test_identical_latency_results_no_regression(self) -> None:
        """Verify identical baseline and current results return NO_REGRESSION."""
        base = _make_benchmark_fixture(median=100.0)
        curr = _make_benchmark_fixture(median=100.0)
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.NO_REGRESSION)
        self.assertEqual(res.metric["percentage_change"], 0.0)
        self.assertEqual(res.metric["absolute_change"], 0.0)
        self.assertEqual(res.metric["direction"], MetricDirection.NEUTRAL.value)

    def test_small_latency_increase_within_variance(self) -> None:
        """Verify small latency increase within threshold (e.g. +1.0%) reports NO_REGRESSION."""
        base = _make_benchmark_fixture(median=100.0)
        curr = _make_benchmark_fixture(median=101.0)  # +1%
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.NO_REGRESSION)
        self.assertAlmostEqual(res.metric["percentage_change"], 1.0)
        self.assertEqual(res.metric["direction"], MetricDirection.WORSE.value)

    def test_moderate_latency_increase_possible_regression(self) -> None:
        """Verify moderate latency increase between 2% and 5% reports POSSIBLE_REGRESSION."""
        base = _make_benchmark_fixture(median=100.0)
        curr = _make_benchmark_fixture(median=103.5)  # +3.5%
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.POSSIBLE_REGRESSION)
        self.assertAlmostEqual(res.metric["percentage_change"], 3.5)
        self.assertEqual(res.metric["direction"], MetricDirection.WORSE.value)
        self.assertIn("exceeded possible regression threshold", res.reason)

    def test_large_latency_increase_regression(self) -> None:
        """Verify large latency increase > 5% reports REGRESSION."""
        base = _make_benchmark_fixture(median=100.0)
        curr = _make_benchmark_fixture(median=110.0)  # +10.0%
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.REGRESSION)
        self.assertAlmostEqual(res.metric["percentage_change"], 10.0)
        self.assertEqual(res.metric["direction"], MetricDirection.WORSE.value)
        self.assertIn("exceeded regression threshold", res.reason)

    def test_small_latency_decrease_no_regression(self) -> None:
        """Verify minor latency decrease (<= 2%) is classified as acceptable NO_REGRESSION."""
        base = _make_benchmark_fixture(median=100.0)
        curr = _make_benchmark_fixture(median=99.0)  # -1%
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.NO_REGRESSION)
        self.assertEqual(res.metric["direction"], MetricDirection.BETTER.value)

    def test_meaningful_latency_decrease_improvement(self) -> None:
        """Verify substantial latency reduction (> 5%) reports IMPROVEMENT."""
        base = _make_benchmark_fixture(median=100.0)
        curr = _make_benchmark_fixture(median=85.0)  # -15%
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.IMPROVEMENT)
        self.assertAlmostEqual(res.metric["percentage_change"], -15.0)
        self.assertEqual(res.metric["direction"], MetricDirection.BETTER.value)
        self.assertIn("exceeded improvement threshold", res.reason)

    def test_moderate_latency_decrease_possible_improvement(self) -> None:
        """Verify moderate latency reduction (between 2% and 5%) reports POSSIBLE_IMPROVEMENT."""
        base = _make_benchmark_fixture(median=100.0)
        curr = _make_benchmark_fixture(median=96.5)  # -3.5%
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.POSSIBLE_IMPROVEMENT)
        self.assertAlmostEqual(res.metric["percentage_change"], -3.5)
        self.assertEqual(res.metric["direction"], MetricDirection.BETTER.value)

    def test_throughput_decrease_regression(self) -> None:
        """Verify throughput reduction (> 5%) reports REGRESSION with direction WORSE."""
        cfg = RegressionConfig(metric="throughput")
        base = _make_benchmark_fixture(throughput_ops_sec=1000.0)
        curr = _make_benchmark_fixture(throughput_ops_sec=900.0)  # -10% throughput
        res = compare_benchmarks(base, curr, config=cfg)

        self.assertEqual(res.status, RegressionStatus.REGRESSION)
        self.assertAlmostEqual(res.metric["percentage_change"], -10.0)
        self.assertEqual(res.metric["direction"], MetricDirection.WORSE.value)
        self.assertTrue(res.metric["higher_is_better"])

    def test_throughput_increase_improvement(self) -> None:
        """Verify throughput increase (> 5%) reports IMPROVEMENT with direction BETTER."""
        cfg = RegressionConfig(metric="throughput")
        base = _make_benchmark_fixture(throughput_ops_sec=1000.0)
        curr = _make_benchmark_fixture(throughput_ops_sec=1150.0)  # +15% throughput
        res = compare_benchmarks(base, curr, config=cfg)

        self.assertEqual(res.status, RegressionStatus.IMPROVEMENT)
        self.assertAlmostEqual(res.metric["percentage_change"], 15.0)
        self.assertEqual(res.metric["direction"], MetricDirection.BETTER.value)
        self.assertTrue(res.metric["higher_is_better"])

    def test_incompatible_benchmark_names(self) -> None:
        """Verify differing benchmark names report INCOMPARABLE."""
        base = _make_benchmark_fixture(name="kernel_gemm")
        curr = _make_benchmark_fixture(name="kernel_conv")
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.INCOMPARABLE)
        self.assertIn("names differ", res.reason)

    def test_incompatible_versions(self) -> None:
        """Verify differing benchmark versions report INCOMPARABLE."""
        base = _make_benchmark_fixture(version="1.0.0")
        curr = _make_benchmark_fixture(version="2.0.0")
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.INCOMPARABLE)
        self.assertIn("versions differ", res.reason)

    def test_incompatible_configuration_fingerprints(self) -> None:
        """Verify differing configuration fingerprints report INCOMPARABLE."""
        base = _make_benchmark_fixture(fingerprint="fp_aaaa")
        curr = _make_benchmark_fixture(fingerprint="fp_bbbb")
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.INCOMPARABLE)
        self.assertIn("fingerprints differ", res.reason)

    def test_insufficient_measurement_count(self) -> None:
        """Verify counts below min_measurement_count report INSUFFICIENT_DATA."""
        cfg = RegressionConfig(min_measurement_count=10)
        base = _make_benchmark_fixture(count=5)  # 5 < 10
        curr = _make_benchmark_fixture(count=15)
        res = compare_benchmarks(base, curr, config=cfg)

        self.assertEqual(res.status, RegressionStatus.INSUFFICIENT_DATA)
        self.assertIn("Insufficient measurements", res.reason)

    def test_zero_baseline_handled_safely(self) -> None:
        """Verify zero baseline metric value avoids division by zero and reports INSUFFICIENT_DATA."""
        base = _make_benchmark_fixture(median=0.0)
        curr = _make_benchmark_fixture(median=10.0)
        res = compare_benchmarks(base, curr)

        self.assertEqual(res.status, RegressionStatus.INSUFFICIENT_DATA)
        self.assertIn("cannot compute percentage change", res.reason)

    def test_missing_throughput_handled_safely(self) -> None:
        """Verify missing throughput metric reports INSUFFICIENT_DATA gracefully."""
        cfg = RegressionConfig(metric="throughput")
        base = _make_benchmark_fixture(throughput_ops_sec=None)
        curr = _make_benchmark_fixture(throughput_ops_sec=1000.0)
        res = compare_benchmarks(base, curr, config=cfg)

        self.assertEqual(res.status, RegressionStatus.INSUFFICIENT_DATA)
        self.assertIn("is missing", res.reason)

    def test_percentage_and_absolute_change_calculations(self) -> None:
        """Verify mathematical calculations of absolute and percentage changes."""
        base = _make_benchmark_fixture(median=80.0)
        curr = _make_benchmark_fixture(median=100.0)
        res = compare_benchmarks(base, curr)

        self.assertAlmostEqual(res.metric["absolute_change"], 20.0)
        self.assertAlmostEqual(res.metric["percentage_change"], 25.0)

    def test_result_is_json_serializable(self) -> None:
        """Verify RegressionResult can be serialized to JSON without error."""
        base = _make_benchmark_fixture(median=50.0)
        curr = _make_benchmark_fixture(median=60.0)
        res = compare_benchmarks(base, curr)

        json_str = res.to_json()
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["status"], "REGRESSION")
        self.assertIn("thresholds", parsed)
        self.assertIn("evidence", parsed)

    def test_configurable_thresholds(self) -> None:
        """Verify custom threshold configuration modifies classification accordingly."""
        # 10% increase with default (2%, 5%) is REGRESSION
        base = _make_benchmark_fixture(median=100.0)
        curr = _make_benchmark_fixture(median=110.0)

        # Custom high threshold: possible=15%, regression=20%
        custom_cfg = RegressionConfig(
            possible_regression_threshold_percent=15.0,
            regression_threshold_percent=20.0,
        )
        res = compare_benchmarks(base, curr, config=custom_cfg)
        self.assertEqual(res.status, RegressionStatus.NO_REGRESSION)

    def test_default_configuration_values(self) -> None:
        """Verify RegressionConfig default parameters."""
        cfg = RegressionConfig()
        self.assertEqual(cfg.metric, "median_latency_ms")
        self.assertEqual(cfg.possible_regression_threshold_percent, 2.0)
        self.assertEqual(cfg.regression_threshold_percent, 5.0)
        self.assertEqual(cfg.min_measurement_count, 5)

    def test_reason_is_clear_and_defensible(self) -> None:
        """Verify reason describes the threshold-based evidence without false statistical claims."""
        base = _make_benchmark_fixture(median=50.0)
        curr = _make_benchmark_fixture(median=60.0)
        res = compare_benchmarks(base, curr)

        self.assertIn("Performance degradation of 20.00%", res.reason)
        self.assertIn("exceeded regression threshold", res.reason)
        # Verify it does NOT claim unproven statistical significance
        self.assertNotIn("Statistically significant", res.reason)

    def test_benchmark_result_instance_acceptance(self) -> None:
        """Verify compare_benchmarks accepts actual BenchmarkResult dataclass instances."""
        bm_res_base = BenchmarkResult(
            schema_version="1.0",
            fingerprint="fp_same",
            benchmark={"name": "bm_dataclass", "version": "1.0"},
            configuration={"warmup_runs": 1, "measurement_runs": 10, "operation_count": None},
            latency_ms={"count": 10, "median": 20.0, "mean": 20.0, "min": 19.0, "max": 21.0, "stddev": 0.5, "p95": 20.9, "p99": 21.0},
            throughput={"operations_per_second": None},
        )
        bm_res_curr = BenchmarkResult(
            schema_version="1.0",
            fingerprint="fp_same",
            benchmark={"name": "bm_dataclass", "version": "1.0"},
            configuration={"warmup_runs": 1, "measurement_runs": 10, "operation_count": None},
            latency_ms={"count": 10, "median": 25.0, "mean": 25.0, "min": 24.0, "max": 26.0, "stddev": 0.6, "p95": 25.9, "p99": 26.0},
            throughput={"operations_per_second": None},
        )
        res = compare_benchmarks(bm_res_base, bm_res_curr)
        self.assertEqual(res.status, RegressionStatus.REGRESSION)
        self.assertAlmostEqual(res.metric["percentage_change"], 25.0)

    def test_cli_execution(self) -> None:
        """Verify python -m src.regression executes and returns valid JSON demonstration."""
        res = subprocess.run(
            [sys.executable, "-m", "src.regression"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"CLI exited with error: {res.stderr}")
        self.assertIn("ROCm Performance Guard", res.stdout)
        self.assertIn('"status": "REGRESSION"', res.stdout)


if __name__ == "__main__":
    unittest.main()