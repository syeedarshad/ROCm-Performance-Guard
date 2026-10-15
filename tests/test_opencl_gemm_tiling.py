"""Experiment 006 legality, launch configuration and adoption gates."""
import ctypes as ct
import unittest
from experiments import opencl_gemm_tiling as exp


class TestOpenCLTiling(unittest.TestCase):
    def device(self):
        return dict(max_work_group_size=256,max_work_item_sizes=[1024,1024,1024],
                    local_memory_bytes=32768)

    def kernel(self):
        return dict(max_work_group_size=256,preferred_work_group_multiple=64,
                    local_memory_bytes=2048,compile_work_group_size=[16,16,1])

    def test_device_legal_tiles(self):
        for tile in (8,16):
            self.assertEqual(exp.tile_legality(tile,self.device()),[])

    def test_32_rejected_by_total_workgroup_limit(self):
        self.assertIn("Work-group exceeds device work-item limit",
                      exp.tile_legality(32,self.device()))

    def test_per_dimension_limit(self):
        device=self.device();device["max_work_item_sizes"]=[8,1024,1024]
        self.assertIn("Per-dimension work-item limit exceeded",exp.tile_legality(16,device))

    def test_local_memory_limit(self):
        device=self.device();device["local_memory_bytes"]=256
        self.assertTrue(exp.tile_legality(8,device))

    def test_compiled_kernel_group_limit(self):
        kernel=self.kernel();kernel["max_work_group_size"]=128
        self.assertIn("Work-group exceeds compiled kernel limit",
                      exp.tile_legality(16,self.device(),kernel))

    def test_compiled_kernel_memory_limit(self):
        kernel=self.kernel();kernel["local_memory_bytes"]=65536
        self.assertIn("Compiled kernel local memory exceeds device limit",
                      exp.tile_legality(16,self.device(),kernel))

    def test_required_workgroup_mismatch(self):
        kernel=self.kernel();kernel["compile_work_group_size"]=[8,8,1]
        self.assertTrue(exp.tile_legality(16,self.device(),kernel))

    def test_unregistered_tiles_rejected(self):
        for tile in (0,-8,7,64):
            self.assertTrue(exp.tile_legality(tile,self.device()))
            with self.assertRaises(ValueError):
                exp.kernel_source(tile)

    def test_launch_supplies_tile_without_changing_other_arguments(self):
        backend=object.__new__(exp.TiledOpenCL)
        backend.local_size=(exp.base.S*2)(16,16)
        calls=[]
        backend.raw_enqueue=lambda *args:calls.append(args) or 0
        args=(1,2,2,None,(512,512),None,0,None,9)
        self.assertEqual(backend.enqueue_tiled(*args),0)
        self.assertEqual(list(calls[0][5]),[16,16])
        self.assertEqual(calls[0][:5],args[:5])
        self.assertEqual(calls[0][6:],args[6:])

    def decision(self,**changes):
        args=dict(baseline_ms=100.,baseline_p95=110.,baseline_cv=4.12,
                  candidate_ms=50.,candidate_p95=55.,candidate_cv=1.,
                  run_medians=[49.,50.,51.],correct=True)
        args.update(changes)
        return exp.decide(**args)

    def test_meaningful_repeatable_gain_adopted(self):
        result=self.decision()
        self.assertEqual(result["decision"],"ADOPT")
        self.assertEqual(result["speedup_vs_baseline"],2.)
        self.assertEqual(result["improvement_percent"],50.)

    def test_small_gain_rejected(self):
        self.assertEqual(self.decision(candidate_ms=96.)["decision"],"REJECT")

    def test_regression_rejected(self):
        result=self.decision(candidate_ms=120.)
        self.assertEqual(result["decision"],"REJECT")
        self.assertEqual(result["verification_status"],"REGRESSED")

    def test_variability_margin_rejects_uncertain_gain(self):
        result=self.decision(candidate_ms=90.,candidate_cv=6.,run_medians=[89.,90.,91.])
        self.assertFalse(result["checks"]["exceeds_variability"])
        self.assertEqual(result["decision"],"REJECT")

    def test_p95_guard(self):
        self.assertEqual(self.decision(candidate_p95=130.)["decision"],"REJECT")

    def test_every_run_must_improve(self):
        self.assertEqual(self.decision(run_medians=[49.,50.,99.])["decision"],"REJECT")

    def test_incorrect_candidate_never_adopted(self):
        result=self.decision(correct=False,candidate_ms=0.01)
        self.assertEqual(result["decision"],"REJECT")
        self.assertEqual(result["verification_status"],"FAILED")

    def test_incomplete_runs_excluded_from_performance(self):
        locked=dict(runs=[dict(input_sha256={},device={})])
        result=exp.summarize(8,[],locked)
        self.assertEqual(result["result"],"FAILED")
        self.assertNotIn("aggregate_timing_ms",result)

    def test_nonfinite_timings_rejected(self):
        with self.assertRaises(ValueError):
            self.decision(candidate_ms=float("nan"))
        with self.assertRaises(ValueError):
            self.decision(candidate_cv=float("inf"))


if __name__=="__main__":
    unittest.main()
