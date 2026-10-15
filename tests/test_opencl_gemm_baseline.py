"""OpenCL timing and correctness-contract tests; no GPU dependency."""
import ctypes as ct
import math
import unittest
from unittest.mock import patch

from experiments import opencl_gemm_baseline as gemm


class TestOpenCLGemmBaseline(unittest.TestCase):
    def test_fp32_inputs_match_benchmark_generator(self):
        a, b = gemm.deterministic_inputs()
        self.assertEqual((len(a), len(b)), (512 * 512, 512 * 512))
        self.assertEqual(ct.sizeof(a), 512 * 512 * 4)
        for row, col in ((0, 0), (7, 11), (511, 511)):
            self.assertEqual(a[row * 512 + col],
                             ct.c_float(((17 * row + 31 * col + 1) % 1000) / 1000).value)
            self.assertEqual(b[row * 512 + col],
                             ct.c_float(((23 * row + 41 * col + 2) % 1000) / 1000).value)

    def test_correctness_accepts_and_rejects(self):
        self.assertEqual(gemm.correctness([1., 0.], [1.0001, 0.0001])["result"], "PASS")
        self.assertEqual(gemm.correctness([1.], [1.01])["result"], "FAIL")

    def test_nonfinite_output_rejected(self):
        self.assertEqual(gemm.correctness([1.], [math.nan])["result"], "FAIL")

    def test_event_nanoseconds_to_milliseconds(self):
        self.assertEqual(gemm.event_latency_ms(100, 2000100), 2.0)
        for start, end in ((1, 1), (2, 1), (-1, 2)):
            with self.assertRaises(ValueError):
                gemm.event_latency_ms(start, end)

    def test_statistics_and_p95(self):
        result = gemm.timing_summary([1., 2., 3., 4.])
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["p95"], 3.85)
        self.assertEqual((result["min"], result["max"], result["count"]), (1., 4., 4))

    def test_invalid_timing_rejected(self):
        for values in ([], [0.], [-1.], [math.inf], [math.nan]):
            with self.assertRaises(ValueError):
                gemm.timing_summary(values)

    def test_gflops_uses_kernel_milliseconds(self):
        self.assertAlmostEqual(gemm.gflops(10.), 26.8435456)
        with self.assertRaises(ValueError):
            gemm.gflops(0)

    def fake_backend(self, profile_error=0, wait_error=0):
        backend = object.__new__(gemm.WindowsOpenCL)
        backend.queue, backend.kernel = 1, 2
        backend.buffers = [3, 4, 5]
        backend.output = (ct.c_float * 4)()
        backend.global_size = (gemm.S * 2)(512, 512)
        calls = []
        counter = [0]
        def enqueue(label):
            def inner(*args):
                counter[0] += 1
                ct.cast(args[-1], ct.POINTER(gemm.P))[0] = counter[0]
                calls.append(label)
                return 0
            return inner
        backend.write = enqueue("write")
        backend.enqueue = enqueue("kernel")
        backend.read = enqueue("read")
        def wait(*args):
            calls.append("wait")
            return wait_error
        backend.wait = wait
        def profile(event, key, size, out, unused):
            self.assertIn("wait", calls)
            calls.append("profile")
            ct.cast(out, ct.POINTER(gemm.Q))[0] = event.value * 10000 + (2000 if key == 0x1283 else 0)
            return profile_error
        backend.profile = profile
        backend.release_event = lambda event: calls.append("release") or 0
        return backend, calls

    def test_wait_precedes_profiling_and_timings_are_separate(self):
        backend, calls = self.fake_backend()
        host = (ct.c_float * 4)(1, 2, 3, 4)
        with patch.object(gemm.time, "perf_counter_ns", side_effect=[1000, 11000]):
            sample = backend.sample(host, host)
        self.assertEqual(calls[:5], ["write", "write", "kernel", "read", "wait"])
        self.assertEqual(sample["kernel_ms"], 0.002)
        self.assertEqual(sample["h2d_ms"], 0.004)
        self.assertEqual(sample["d2h_ms"], 0.002)
        self.assertEqual(sample["end_to_end_ms"], 0.01)
        self.assertEqual(calls.count("release"), 4)

    def test_profile_failure_has_no_wall_clock_fallback(self):
        backend, calls = self.fake_backend(profile_error=-7)
        host = (ct.c_float * 4)()
        with self.assertRaisesRegex(RuntimeError, "event START"):
            backend.sample(host, host)
        self.assertEqual(calls.count("release"), 4)

    def test_wait_failure_prevents_profiling(self):
        backend, calls = self.fake_backend(wait_error=-14)
        host = (ct.c_float * 4)()
        with self.assertRaisesRegex(RuntimeError, "wait for read"):
            backend.sample(host, host)
        self.assertNotIn("profile", calls)
        self.assertEqual(calls.count("release"), 4)


if __name__ == "__main__":
    unittest.main()
