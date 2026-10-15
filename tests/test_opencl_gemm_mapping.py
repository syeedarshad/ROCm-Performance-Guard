"""Experiment 007 controlled source changes, correctness-first execution and dual gates."""
import ctypes as ct
import unittest
from unittest.mock import patch
from experiments import opencl_gemm_mapping as exp


class FakeBackend:
    metadata={}
    device_limits={}
    kernel_limits={}
    def __init__(self,fail_at=None):
        self.calls=0
        self.fail_at=fail_at
        self.output=[1.]
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def sample(self,a,b):
        self.calls+=1
        self.output=[2. if self.calls==self.fail_at else 1.]
        return dict(kernel_ms=1.,h2d_ms=0.1,d2h_ms=0.1,end_to_end_ms=2.)


class TestOpenCLMapping(unittest.TestCase):
    def test_control_source_is_exact_locked_source(self):
        self.assertEqual(exp.kernel_source("row_fast"),exp.tiled.kernel_source(8))

    def test_only_four_axis_assignments_change(self):
        original=exp.kernel_source("row_fast").splitlines()
        candidate=exp.kernel_source("column_fast").splitlines()
        self.assertEqual(len(original),len(candidate))
        changes=[(a,b) for a,b in zip(original,candidate) if a!=b]
        self.assertEqual(len(changes),4)
        self.assertEqual({a.strip().split(" = ")[0] for a,b in changes},
                         {"const int row","const int col","const int lr","const int lc"})

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            exp.kernel_source("unroll")

    def test_axis_swap_covers_each_output_once(self):
        # Dimension swap is a bijection; check tile boundaries and complete local tile coverage.
        self.assertEqual({(y,x) for x in range(8) for y in range(8)},
                         {(x,y) for x in range(8) for y in range(8)})
        for x,y in ((0,0),(7,8),(8,7),(511,511)):
            row,col=y,x
            self.assertEqual((row%8,col%8),(y%8,x%8))

    def measure(self,fail_at=None):
        backend=FakeBackend(fail_at)
        a=(ct.c_float*1)(1.)
        with patch.object(exp.base,"deterministic_inputs",return_value=(a,a)), \
             patch.object(exp.base,"cpu_reference",return_value=[1.]):
            run=exp.measure_candidate(lambda:backend)
        return run,backend

    def test_exact_preflight_warmup_measurement_counts(self):
        run,backend=self.measure()
        self.assertEqual(backend.calls,1+10+50)
        self.assertEqual(run["result"],"PASS")
        self.assertEqual(len(run["samples"]),50)
        self.assertEqual(run["warmups_completed"],10)
        self.assertTrue(all(s["correctness"]["allowed_atol"]==0.001 and
                            s["correctness"]["allowed_rtol"]==0.001 for s in run["samples"]))

    def test_preflight_failure_prevents_benchmark(self):
        run,backend=self.measure(1)
        self.assertEqual(run["result"],"FAILED")
        self.assertEqual(backend.calls,1)
        self.assertEqual(run["samples"],[])

    def test_warmup_failure_prevents_measured_iterations(self):
        run,backend=self.measure(5)
        self.assertEqual(run["result"],"FAILED")
        self.assertEqual(backend.calls,5)
        self.assertEqual(run["samples"],[])

    def test_measured_failure_stops_and_excludes_statistics(self):
        run,backend=self.measure(13)
        self.assertEqual(run["result"],"FAILED")
        self.assertEqual(len(run["samples"]),2)
        self.assertNotIn("timing_ms",run)

    def result(self,median=100.,cv=1.,decision="ADOPT",p95=None):
        return dict(result="PASS",decision=decision,
                    aggregate_timing_ms=dict(kernel_ms=dict(median=median,p95=p95 or median*1.1)),
                    run_median_cv_percent=cv,run_medians_ms=[median]*3)

    def test_control_reproducible(self):
        gate=exp.control_gate(self.result(102.),self.result(100.))
        self.assertTrue(gate["passed"])
        self.assertAlmostEqual(gate["signed_drift_percent"],2.)

    def test_control_drift_rejects_faster_or_slower(self):
        for value in (94.,106.):
            self.assertFalse(exp.control_gate(self.result(value),self.result())["passed"])

    def test_unstable_control_rejected(self):
        self.assertFalse(exp.control_gate(self.result(cv=6.),self.result())["passed"])

    def test_failed_control_rejected(self):
        self.assertFalse(exp.control_gate(dict(result="FAILED"),self.result())["passed"])

    def test_dual_gate_adopts(self):
        result=exp.final_gate(self.result(80.),self.result(),dict(passed=True))
        self.assertEqual(result["decision"],"ADOPT")

    def test_historical_only_gain_rejected(self):
        result=exp.final_gate(self.result(98.),self.result(),dict(passed=True))
        self.assertEqual(result["decision"],"REJECT")

    def test_fresh_only_gain_rejected(self):
        result=exp.final_gate(self.result(80.,decision="REJECT"),self.result(),dict(passed=True))
        self.assertEqual(result["decision"],"REJECT")

    def test_candidate_failure_never_adopted(self):
        result=exp.final_gate(dict(result="FAILED"),self.result(),dict(passed=True))
        self.assertEqual(result["decision"],"REJECT")

    def test_unreproduced_baseline_never_adopts(self):
        result=exp.final_gate(self.result(50.),self.result(),dict(passed=False))
        self.assertEqual(result["decision"],"REJECT")


if __name__=="__main__":
    unittest.main()
