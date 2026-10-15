"""Comprehensive test suite for Benchmark Engine V1 and GEMM workload."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.benchmark import (
    BackendUnavailableError,
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkExecutionError,
    BenchmarkResult,
    BenchmarkRunner,
    CpuGemmBackend,
    RocmGemmBackend,
    calculate_percentile,
    demo_cpu_workload,
    generate_deterministic_matrix,
    get_backend,
    run_gemm_benchmark,
    save_benchmark_result,
)


class TestBenchmarkEngine(unittest.TestCase):
    """Test suite for Benchmark Engine V1, GEMM workload, and execution backends."""

    def test_01_deterministic_matrix_generation(self) -> None:
        """1. Matrix generation produces deterministic, bounded float values."""
        m1 = generate_deterministic_matrix(4, 8, seed_offset=1)
        m2 = generate_deterministic_matrix(4, 8, seed_offset=1)

        self.assertEqual(len(m1), 4)
        self.assertEqual(len(m1[0]), 8)
        # Exact mathematical determinism across invocations
        self.assertEqual(m1, m2)

        # Values bounded in [0.0, 1.0]
        for row in m1:
            for val in row:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

        # Different seed offsets yield distinct matrices
        m3 = generate_deterministic_matrix(4, 8, seed_offset=2)
        self.assertNotEqual(m1, m3)

    def test_02_gemm_correctness(self) -> None:
        """2. GEMM matrix multiplication matches mathematically verified product."""
        # 2x2 test matrices:
        # A = [[1.0, 2.0], [3.0, 4.0]]
        # B = [[5.0, 6.0], [7.0, 8.0]]
        # C = A @ B = [[19.0, 22.0], [43.0, 50.0]]
        backend = CpuGemmBackend(workers=1)
        backend.a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        backend.b_t = [[b[p][j] for p in range(2)] for j in range(2)]

        c = backend.execute()
        self.assertEqual(len(c), 2)
        self.assertEqual(len(c[0]), 2)
        self.assertAlmostEqual(c[0][0], 19.0)
        self.assertAlmostEqual(c[0][1], 22.0)
        self.assertAlmostEqual(c[1][0], 43.0)
        self.assertAlmostEqual(c[1][1], 50.0)

    def test_03_matrix_dimension_validation(self) -> None:
        """3. Invalid matrix dimensions (<= 0) raise ValueError."""
        with self.assertRaises(ValueError):
            generate_deterministic_matrix(0, 10)
        with self.assertRaises(ValueError):
            generate_deterministic_matrix(10, -5)

        with self.assertRaises(ValueError):
            BenchmarkConfig(m=0, n=512, k=512)
        with self.assertRaises(ValueError):
            BenchmarkConfig(m=512, n=-10, k=512)
        with self.assertRaises(ValueError):
            BenchmarkConfig(m=512, n=512, k=0)

    def test_04_timing_statistics_accuracy(self) -> None:
        """4. Statistical metrics (min, mean, median, max, stddev) are computed accurately."""
        durations_ns = [10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]
        timestamps: list[int] = []
        for d in durations_ns:
            timestamps.extend([1_000_000_000, 1_000_000_000 + d])

        cfg = BenchmarkConfig(name="stat_test", version="1.0.0", warmup_runs=0, measurement_runs=5, m=8, n=8, k=8)
        runner = BenchmarkRunner(cfg)

        with patch("time.perf_counter_ns", side_effect=timestamps):
            result = runner.run(workload=lambda: None)

        lat = result.latency_ms
        self.assertEqual(lat["count"], 5)
        self.assertAlmostEqual(lat["min"], 10.0)
        self.assertAlmostEqual(lat["max"], 50.0)
        self.assertAlmostEqual(lat["mean"], 30.0)
        self.assertAlmostEqual(lat["median"], 30.0)
        self.assertAlmostEqual(lat["stddev"], 15.8113883, places=4)

    def test_05_median_calculation(self) -> None:
        """5. Median latency calculation handles odd and even iteration counts."""
        # Odd count: [10.0, 20.0, 30.0] -> median is 20.0
        d_odd = [10_000_000, 20_000_000, 30_000_000]
        ts_odd: list[int] = []
        for d in d_odd:
            ts_odd.extend([1_000_000_000, 1_000_000_000 + d])

        cfg_odd = BenchmarkConfig(warmup_runs=0, measurement_runs=3, m=4, n=4, k=4)
        with patch("time.perf_counter_ns", side_effect=ts_odd):
            res_odd = BenchmarkRunner(cfg_odd).run(workload=lambda: None)
        self.assertAlmostEqual(res_odd.latency_ms["median"], 20.0)

        # Even count: [10.0, 20.0, 30.0, 40.0] -> median is 25.0
        d_even = [10_000_000, 20_000_000, 30_000_000, 40_000_000]
        ts_even: list[int] = []
        for d in d_even:
            ts_even.extend([1_000_000_000, 1_000_000_000 + d])

        cfg_even = BenchmarkConfig(warmup_runs=0, measurement_runs=4, m=4, n=4, k=4)
        with patch("time.perf_counter_ns", side_effect=ts_even):
            res_even = BenchmarkRunner(cfg_even).run(workload=lambda: None)
        self.assertAlmostEqual(res_even.latency_ms["median"], 25.0)

    def test_06_p95_calculation(self) -> None:
        """6. p95 calculation follows standard NIST linear interpolation."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        p95 = calculate_percentile(data, 95.0)
        # index = 0.95 * 4 = 3.8 -> 40.0 * 0.2 + 50.0 * 0.8 = 48.0
        self.assertAlmostEqual(p95, 48.0)

        data_single = [42.0]
        self.assertEqual(calculate_percentile(data_single, 95.0), 42.0)

    def test_07_gflops_calculation(self) -> None:
        """7. GFLOPS is calculated accurately from median latency and 2 * M * N * K."""
        # For M=64, N=64, K=64:
        # operations = 2 * 64 * 64 * 64 = 524,288 FLOPs
        # If median latency = 10.0 ms = 0.01 s:
        # ops_per_second = 524,288 / 0.01 = 52,428,800 ops/sec
        # GFLOPS = 52,428,800 / 1e9 = 0.0524288 GFLOPS
        durations = [10_000_000] * 5
        ts: list[int] = []
        for d in durations:
            ts.extend([1_000_000_000, 1_000_000_000 + d])

        cfg = BenchmarkConfig(warmup_runs=0, measurement_runs=5, m=64, n=64, k=64)
        with patch("time.perf_counter_ns", side_effect=ts):
            res = BenchmarkRunner(cfg).run(workload=lambda: None)

        self.assertEqual(res.throughput["operations"], 524288)
        self.assertAlmostEqual(res.throughput["operations_per_second"], 52428800.0)
        self.assertAlmostEqual(res.throughput["gflops"], 0.0524288)

    def test_08_cpu_backend_execution(self) -> None:
        """8. CPU backend prepares and executes successfully."""
        backend = get_backend("cpu", workers=1)
        self.assertTrue(backend.is_available())
        backend.prepare(m=16, n=16, k=16, dtype="fp32")
        result = backend.execute()
        self.assertEqual(len(result), 16)
        self.assertEqual(len(result[0]), 16)
        backend.cleanup()

    def test_09_unavailable_rocm_backend(self) -> None:
        """9. RocmGemmBackend reports UNAVAILABLE and raises BackendUnavailableError on host without ROCm."""
        rocm_backend = get_backend("rocm")
        # On current WSL2 test environment, ROCm is unavailable
        self.assertFalse(rocm_backend.is_available())
        status = rocm_backend.get_status()
        self.assertEqual(status["status"], "UNAVAILABLE")
        self.assertIn("reason", status)

        with self.assertRaises(BackendUnavailableError):
            rocm_backend.prepare(m=64, n=64, k=64)

        with self.assertRaises(BackendUnavailableError):
            rocm_backend.execute()

    def test_10_cli_execution(self) -> None:
        """10. CLI executes successfully and outputs valid summary and JSON."""
        cmd = [
            sys.executable,
            "-m",
            "src.benchmark",
            "--m", "32",
            "--n", "32",
            "--k", "32",
            "--warmup", "1",
            "--iterations", "2",
            "--backend", "cpu",
            "--no-save",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ROCm Performance Guard", res.stdout)
        self.assertIn("Benchmark JSON:", res.stdout)

        json_part = res.stdout.split("Benchmark JSON:\n")[-1].strip()
        data = json.loads(json_part)
        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(data["configuration"]["m"], 32)
        self.assertEqual(data["configuration"]["n"], 32)
        self.assertEqual(data["configuration"]["k"], 32)
        self.assertIn("gflops", data["throughput"])

    def test_11_cli_help(self) -> None:
        """11. CLI --help displays usage options and exits with 0."""
        cmd = [sys.executable, "-m", "src.benchmark", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--m", res.stdout)
        self.assertIn("--backend", res.stdout)

    def test_12_cli_unavailable_rocm(self) -> None:
        """12. CLI reports structured error when requesting unavailable backend."""
        cmd = [sys.executable, "-m", "src.benchmark", "--backend", "rocm"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 1)
        self.assertIn("UNAVAILABLE", res.stderr)

    def test_13_json_serialization(self) -> None:
        """13. BenchmarkResult produces valid and complete JSON."""
        cfg = BenchmarkConfig(m=16, n=16, k=16, warmup_runs=1, measurement_runs=2)
        res, _ = run_gemm_benchmark(m=16, n=16, k=16, warmup=1, iterations=2, save_result=False)

        json_str = res.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(parsed["benchmark"]["name"], "gemm_fp32")
        self.assertEqual(parsed["configuration"]["m"], 16)
        self.assertIn("median", parsed["latency_ms"])

    def test_14_result_persistence(self) -> None:
        """14. Benchmark result persists as JSON without overwriting existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BenchmarkConfig(m=8, n=8, k=8, warmup_runs=1, measurement_runs=2)
            runner = BenchmarkRunner(cfg)
            res1 = runner.run(workload=lambda: None)
            res2 = runner.run(workload=lambda: None)

            path1 = save_benchmark_result(res1, directory=tmpdir)
            path2 = save_benchmark_result(res2, directory=tmpdir)

            self.assertTrue(os.path.isfile(path1))
            self.assertTrue(os.path.isfile(path2))
            self.assertNotEqual(path1, path2)

            # File is valid JSON
            with open(path1, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["schema_version"], "1.0")

    def test_15_invalid_arguments(self) -> None:
        """15. Invalid configuration parameters raise descriptive ValueError."""
        with self.assertRaises(ValueError):
            BenchmarkConfig(warmup_runs=-1)
        with self.assertRaises(ValueError):
            BenchmarkConfig(measurement_runs=0)
        with self.assertRaises(ValueError):
            BenchmarkConfig(dtype="int8")
        with self.assertRaises(ValueError):
            BenchmarkConfig(backend="cuda_unsupported")

    def test_16_no_system_mutation(self) -> None:
        """16. Benchmark execution does not modify os.environ."""
        environ_before = dict(os.environ)
        _ = run_gemm_benchmark(m=16, n=16, k=16, warmup=1, iterations=2, save_result=False)
        self.assertEqual(dict(os.environ), environ_before)

    def test_17_deterministic_configuration_fingerprint(self) -> None:
        """17. Fingerprint is purely a deterministic function of benchmark configuration."""
        cfg1 = BenchmarkConfig(m=128, n=128, k=128, warmup_runs=3, measurement_runs=10)
        cfg2 = BenchmarkConfig(m=128, n=128, k=128, warmup_runs=3, measurement_runs=10)
        cfg3 = BenchmarkConfig(m=256, n=128, k=128, warmup_runs=3, measurement_runs=10)

        self.assertEqual(cfg1.fingerprint(), cfg2.fingerprint())
        self.assertEqual(len(cfg1.fingerprint()), 16)
        self.assertNotEqual(cfg1.fingerprint(), cfg3.fingerprint())

    def test_18_percentile_small_datasets(self) -> None:
        """18. Percentile calculations on small datasets execute deterministically without errors."""
        self.assertEqual(calculate_percentile([100.0], 95.0), 100.0)
        with self.assertRaises(ValueError):
            calculate_percentile([], 95.0)
        with self.assertRaises(ValueError):
            calculate_percentile([1.0, 2.0], -1.0)
        with self.assertRaises(ValueError):
            calculate_percentile([1.0, 2.0], 101.0)

    def test_19_demo_workload_execution(self) -> None:
        """19. Legacy demo CPU workload completes and returns expected mathematical result."""
        result = demo_cpu_workload()
        self.assertEqual(result, 333283335000)

    def test_20_optional_snapshots_integration(self) -> None:
        """20. Snapshots are correctly attached to BenchmarkResult."""
        cfg = BenchmarkConfig(name="snapshot_test", version="1.0.0", warmup_runs=0, measurement_runs=1, m=8, n=8, k=8)
        hw_snap = {"cpu": {"vendor": "AMD"}}
        env_snap = {"python": {"version": "3.13.16"}}

        runner = BenchmarkRunner(cfg)
        result = runner.run(workload=lambda: None, hardware_snapshot=hw_snap, environment_snapshot=env_snap)

        res_dict = result.to_dict()
        self.assertEqual(res_dict.get("hardware_snapshot"), hw_snap)
        self.assertEqual(res_dict.get("environment_snapshot"), env_snap)


if __name__ == "__main__":
    unittest.main()
