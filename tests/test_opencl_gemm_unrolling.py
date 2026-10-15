"""Experiment 008 source isolation, execution provenance and candidate selection."""
import hashlib
import re
import unittest
from unittest.mock import patch
from experiments import opencl_gemm_unrolling as exp


class TestOpenCLUnrolling(unittest.TestCase):
    def test_control_is_byte_identical_kernel(self):
        self.assertEqual(exp.kernel_source(1),exp.mapping.kernel_source("column_fast"))

    def test_only_inner_loop_changes(self):
        original=exp.kernel_source(1)
        before,after=original.split(exp.ORIGINAL_INNER)
        for factor in exp.FACTORS:
            self.assertEqual(exp.kernel_source(factor),before+exp.inner_source(factor)+after)

    def test_partial_unroll_preserves_accumulation_order(self):
        for factor in (2,4):
            fragment=exp.inner_source(factor)
            offsets=[int(x) for x in re.findall(r"tile_a\[lr\]\[k \+ (\d+)\]",fragment)]
            self.assertEqual([k+i for k in range(0,8,factor) for i in offsets],list(range(8)))
            self.assertIn(f"k += {factor}",fragment)

    def test_full_unroll_preserves_all_eight_steps(self):
        fragment=exp.inner_source(8)
        self.assertNotIn("for (",fragment)
        self.assertEqual([int(x) for x in re.findall(r"tile_a\[lr\]\[(\d+)\]",fragment)],list(range(8)))
        self.assertEqual([int(x) for x in re.findall(r"tile_b\[(\d+)\]\[lc\]",fragment)],list(range(8)))

    def test_barriers_and_mapping_unchanged(self):
        for factor in exp.FACTORS:
            source=exp.kernel_source(factor)
            self.assertEqual(source.count("barrier(CLK_LOCAL_MEM_FENCE)"),2)
            self.assertIn("row = get_global_id(1)",source)
            self.assertIn("col = get_global_id(0)",source)
            self.assertIn("#pragma OPENCL FP_CONTRACT OFF",source)

    def test_unsupported_factors_rejected(self):
        for factor in (0,3,16,-1):
            with self.assertRaises(ValueError): exp.kernel_source(factor)
            with self.assertRaises(ValueError): exp.measure(factor)

    def test_control_uses_exact_previous_backend_and_routine(self):
        with patch.object(exp.mapping,"measure_candidate",return_value={"result":"PASS"}) as runner:
            run=exp.measure(1)
        runner.assert_called_once_with(exp.mapping.ColumnFastOpenCL)
        self.assertEqual(run["variant"],"compiler_default")
        self.assertEqual(run["kernel_sha256"],hashlib.sha256(exp.kernel_source(1).encode()).hexdigest())

    def test_candidate_source_provenance_not_control_hash(self):
        for factor in exp.FACTORS:
            with patch.object(exp.mapping,"measure_candidate",return_value={"result":"PASS"}):
                run=exp.measure(factor)
            self.assertEqual(run["unroll_factor"],factor)
            self.assertEqual(run["kernel_sha256"],hashlib.sha256(exp.kernel_source(factor).encode()).hexdigest())
            self.assertNotEqual(run["kernel_sha256"],hashlib.sha256(exp.kernel_source(1).encode()).hexdigest())

    def test_failed_measurement_remains_failed(self):
        with patch.object(exp.mapping,"measure_candidate",return_value={"result":"FAILED","reason":"incorrect"}):
            run=exp.measure(2)
        self.assertEqual(run["result"],"FAILED")

    def candidate(self,factor,latency,result="PASS",decision="ADOPT"):
        return dict(unroll_factor=factor,result=result,decision=decision,
                    aggregate_timing_ms=dict(kernel_ms=dict(median=latency)))

    def test_best_only_among_adopted_correct_candidates(self):
        fastest_failed=self.candidate(8,.1,result="FAILED")
        fastest_rejected=self.candidate(4,.2,decision="REJECT")
        accepted=self.candidate(2,2.)
        self.assertIs(exp.best_candidate([fastest_failed,fastest_rejected,accepted]),accepted)

    def test_best_adopted_latency(self):
        candidates=[self.candidate(2,2.),self.candidate(4,1.5),self.candidate(8,1.8)]
        self.assertEqual(exp.best_candidate(candidates)["unroll_factor"],4)

    def test_no_gain_means_no_adoption(self):
        self.assertIsNone(exp.best_candidate([self.candidate(2,4.,decision="REJECT")]))
        self.assertIsNone(exp.best_candidate([]))

    def test_failed_protocol_has_no_performance_adoption(self):
        locked=dict(runs=[dict(input_sha256={},device={})])
        result=exp.summarize(2,[],locked,{},dict(passed=True))
        self.assertEqual(result["result"],"FAILED")
        self.assertEqual(result["decision"],"REJECT")
        self.assertNotIn("aggregate_timing_ms",result)

    def test_control_gate_still_requires_reproducibility(self):
        control=dict(result="PASS",aggregate_timing_ms=dict(kernel_ms=dict(median=4.)),
                     run_median_cv_percent=.2)
        locked=dict(aggregate_timing_ms=dict(kernel_ms=dict(median=3.71454)))
        self.assertFalse(exp.mapping.control_gate(control,locked)["passed"])

    def test_correctness_failure_bypasses_latency_adoption(self):
        result=exp.mapping.final_gate(dict(result="FAILED"),{},dict(passed=True))
        self.assertEqual(result["decision"],"REJECT")


if __name__=="__main__":
    unittest.main()
