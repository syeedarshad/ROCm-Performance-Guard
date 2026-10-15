"""Experiment 008: explicit inner-loop unroll factor of locked 8x8 column-first GEMM."""
from __future__ import annotations
import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments import opencl_gemm_baseline as base
from experiments import opencl_gemm_tiling as tiled
from experiments import opencl_gemm_mapping as mapping

OUTPUT=ROOT/"experiments/results/experiment_008"
MANIFEST=OUTPUT/"experiment_008_preservation_manifest.json"
LOCKED=ROOT/"experiments/results/experiment_007/experiment_007_20261003_165754_114432_summary.json"
LOCKED_SHA256="2c10966cc5669870b202f02efcccce82a5457be6b9c221924239da1336a5ba53"
FACTORS=(2,4,8)
TILE=8
RULE=dict(control_max_absolute_drift_percent=5.0,control_max_run_cv_percent=5.0,
          candidate_gate=dict(tiled.RULE),
          additional_gate="Pass against locked Experiment 007 AND fresh exact control",
          failure_action="REJECT; failed control prevents candidate measurements",
          candidate_order="2,4,8 then 4,8,2 then 8,2,4")


def verify_preserved():
    for item in json.loads(MANIFEST.read_text())["files"]:
        if base.digest(ROOT/item["path"])!=item["sha256"]:
            raise RuntimeError("Immutable file changed: "+item["path"])
    if base.digest(LOCKED)!=LOCKED_SHA256:
        raise RuntimeError("Experiment 007 checksum mismatch")
    base.verify_locked_baseline()


def locked_reference():
    record=json.loads(LOCKED.read_text())["candidate"]
    if record["result"]!="PASS" or record["decision"]!="ADOPT":
        raise RuntimeError("Experiment 007 candidate is not a correct adopted reference")
    return dict(record,kernel_median_between_runs_cv_percent=record["run_median_cv_percent"])


ORIGINAL_INNER = """        for (int k = 0; k < TILE; ++k)
            value += tile_a[lr][k] * tile_b[k][lc];"""


def inner_source(factor):
    if factor not in FACTORS:
        raise ValueError("Unroll factor must be 2, 4 or 8")
    if factor==8:
        return "\n".join(f"        value += tile_a[lr][{i}] * tile_b[{i}][lc];" for i in range(8))
    statements="\n".join(
        f"            value += tile_a[lr][k + {i}] * tile_b[k + {i}][lc];"
        for i in range(factor))
    return f"        for (int k = 0; k < TILE; k += {factor}) {{\n{statements}\n        }}"


def kernel_source(factor):
    source=mapping.kernel_source("column_fast")
    if factor==1:  # Exact compiler-default loop, not a forced unroll=1 pragma.
        return source
    if source.count(ORIGINAL_INNER)!=1:
        raise RuntimeError("Locked inner-loop source changed")
    return source.replace(ORIGINAL_INNER,inner_source(factor))


class UnrolledOpenCL(mapping.ColumnFastOpenCL):
    def __init__(self,factor):
        super().__init__()
        try:
            source=ct.c_char_p(kernel_source(factor).encode())
            err=base.I()
            program=self.call("clCreateProgramWithSource",
                [base.P,base.U,base.P,base.P,ct.POINTER(base.I)],base.P)(
                self.context,1,ct.byref(source),None,ct.byref(err))
            base.check(err.value,"create unrolled program")
            self.resources.append(("clReleaseProgram",program))
            device=base.P(self.device)
            rc=self.call("clBuildProgram",[base.P,base.U,base.P,ct.c_char_p,base.P,base.P])(
                program,1,ct.byref(device),b"-cl-std=CL1.2",None,None)
            if rc:
                log=ct.create_string_buffer(65536)
                self.call("clGetProgramBuildInfo",[base.P,base.P,base.U,base.S,base.P,base.P])(
                    program,device,0x1183,len(log),log,None)
                raise RuntimeError("Unrolled build failed: "+log.value.decode(errors="replace"))
            self.kernel=self.call("clCreateKernel",[base.P,ct.c_char_p,ct.POINTER(base.I)],base.P)(
                program,b"gemm_tiled",ct.byref(err))
            base.check(err.value,"create unrolled kernel")
            self.resources.append(("clReleaseKernel",self.kernel))
            self.kernel_limits=tiled.kernel_capabilities(self,self.kernel)
            reasons=tiled.tile_legality(TILE,self.device_limits,self.kernel_limits)
            if reasons: raise ValueError("; ".join(reasons))
            for index,buffer in enumerate(self.buffers):
                handle=base.P(buffer)
                base.check(self.call("clSetKernelArg",[base.P,base.U,base.S,base.P])(
                    self.kernel,index,ct.sizeof(handle),ct.byref(handle)),"unrolled argument")
        except Exception:
            self.close()
            raise


def measure(factor):
    if factor not in (1,*FACTORS): raise ValueError("Unknown factor")
    factory=mapping.ColumnFastOpenCL if factor==1 else lambda:UnrolledOpenCL(factor)
    # Reuse immutable Experiment 007's exact correctness/warmup/measurement routine.
    run=mapping.measure_candidate(factory)
    run.update(variant="compiler_default" if factor==1 else f"unroll_{factor}",
               unroll_factor=factor,
               kernel_sha256=hashlib.sha256(kernel_source(factor).encode()).hexdigest())
    return run


def run_child(factor,prefix,index):
    proc=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),
                         "--single-run",str(factor)],capture_output=True,text=True)
    try: run=json.loads(proc.stdout)
    except ValueError:
        run=dict(result="FAILED",unroll_factor=factor,reason=proc.stderr or proc.stdout or "No child result",
                 process_id=None,samples=[])
    if proc.returncode and run.get("result")=="PASS":
        run.update(result="FAILED",reason="Nonzero child exit code")
    run["round"]=index+1
    base.save_record(Path(str(prefix)+f"_factor_{factor}_run_{index+1:03d}.json"),run)
    return run


def summarize(factor,runs,locked,control=None,control_gate=None):
    result=tiled.summarize(TILE,runs,locked)
    result["unroll_factor"]=factor
    result["variant"]="compiler_default" if factor==1 else f"unroll_{factor}"
    if factor!=1:
        # Keep the locked-reference decision separately from the final dual gate.
        result["locked_reference_decision"]=result["decision"]
        adoption=mapping.final_gate(result,control,control_gate)
        result["adoption_gate"]=adoption
        result["decision"]=adoption["decision"]
    return result


def best_candidate(candidates):
    eligible=[c for c in candidates if c.get("result")=="PASS" and c.get("decision")=="ADOPT"]
    return min(eligible,key=lambda c:c["aggregate_timing_ms"]["kernel_ms"]["median"]) if eligible else None


def report(record):
    lines=["# Experiment 008 — explicit inner-loop unrolling","",
           "## Immutable reference and hypothesis","",
           "Experiment 007: AMD Radeon gfx90c; native Windows OpenCL; 512x512x512 FP32 GEMM; 8x8 tiles; column-first mapping.",
           "Locked kernel median 3.71454 ms, GFLOPS 72.2661, run-median CV approximately 0.22%.",
           "Hypothesis: explicit unrolling of the eight accumulation steps may reduce loop/index overhead or improve compiler scheduling.",
           "Only the inner-loop unroll factor varies: 2, 4, 8. Arithmetic order and operations, precision, tile size, mapping, local-memory layout, barriers, input bytes and build flags remain fixed.",
           "The original compiler may already unroll the loop. No machine-code or counter evidence is collected, so compiler behavior and hardware causes are not inferred from timings alone.","",
           "## Device capabilities and compiled legality","", "```json",json.dumps(record["discovery"],indent=2),"```","",
           "## Methodology and predeclared gates","",
           "Exact Experiment 007 measurement routine: fresh process/context per run, correctness preflight, 10 warmups, 50 measurements, 3 runs per configuration.",
           "Every output is checked against CPU math.fsum of the same FP32 input values, rounded to FP32 output.",
           "Finite outputs must satisfy abs(GPU-CPU) <= 0.001 + 0.001*abs(CPU); relative diagnostic denominator is max(abs(CPU),1e-12).",
           "Primary latency uses completed OpenCL kernel event START/END timestamps. The final read event is waited on before profiling.",
           "H2D, D2H and synchronized end-to-end timings are retained separately. No end-to-end speedup claim is made.",
           "All raw samples, including outliers, are retained. No package, driver or system-setting changes.",
           "Control is measured first; legal candidates rotate order across rounds. Thermal/background state is not controlled.",
           "```json",json.dumps(record["decision_rule"],indent=2),"```","",
           "Variability gates are conservative heuristics, not formal significance tests.","",
           "## Control reproducibility","", "```json",json.dumps(record["control_gate"],indent=2),"```","",
           "## Kernel performance","",
           "| Configuration | Result | Median ms | Mean ms | P95 ms | Min ms | Max ms | GFLOPS | Run CV % | Speedup vs E7 | Improvement % | Decision |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    all_results=[record["control"],*record["candidates"]]
    for c in all_results:
        label=c["variant"]
        if c.get("result")!="PASS":
            lines.append(f"| {label} | {c.get('result')} | — | — | — | — | — | — | — | — | — | REJECT |")
            continue
        t=c["aggregate_timing_ms"]["kernel_ms"]
        decision="control only" if c["unroll_factor"]==1 else c["decision"]
        lines.append(f"| {label} | PASS | {t['median']:.6f} | {t['mean']:.6f} | {t['p95']:.6f} | {t['min']:.6f} | {t['max']:.6f} | {c['gflops']:.6f} | {c['run_median_cv_percent']:.4f} | {c['speedup_vs_baseline']:.6f} | {c['improvement_percent']:.4f} | {decision} |")
    for c in all_results:
        lines += ["",f"## {c['variant']} evidence",""]
        if c.get("result")!="PASS":
            lines += [c.get("reason","Not measured"),""]
            continue
        lines += ["Run medians (ms): "+json.dumps(c["run_medians_ms"]),
                  "Worst correctness diagnostics across all measured outputs:",
                  "```json",json.dumps(c["worst_correctness_metrics"],indent=2),"```",
                  "All timing components (ms):","```json",json.dumps(c["aggregate_timing_ms"],indent=2),"```"]
        if "adoption_gate" in c:
            lines += ["Final gates:","```json",json.dumps(c["adoption_gate"],indent=2),"```"]
    lines += ["","## Final decision","",
              "**"+record["decision"]+"**: "+record["explanation"],
              "Best adopted factor: "+str(record["recommended_factor"]),
              "The locked Experiment 007 reference is never replaced or modified by this script.",
              "No end-to-end causal claim, ROCm/HIP/CUDA execution, or further optimization.","",
              "## Reproduction","", "```powershell",
              "python -B experiments/opencl_gemm_unrolling.py","```","",
              "Requires the existing native AMD OpenCL runtime. New timestamped files use exclusive creation.",
              "Stop after Experiment 008; do not start Experiment 009.",""]
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-run",type=int,choices=(1,*FACTORS),help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    verify_preserved()
    if args.single_run is not None:
        run=measure(args.single_run)
        verify_preserved()
        print(json.dumps(run,allow_nan=False))
        return 0 if run["result"]=="PASS" else 1
    locked=locked_reference()
    discovery={}
    for factor in (1,*FACTORS):
        factory=mapping.ColumnFastOpenCL if factor==1 else lambda f=factor:UnrolledOpenCL(f)
        try:
            with factory() as b:
                discovery[str(factor)]=dict(legal=True,device=b.metadata,
                    device_capabilities=b.device_limits,kernel_capabilities=b.kernel_limits,
                    kernel_sha256=hashlib.sha256(kernel_source(factor).encode()).hexdigest())
        except Exception as exc:
            discovery[str(factor)]=dict(legal=False,reason=str(exc))
    token=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix=OUTPUT/f"experiment_008_{token}"
    record=dict(schema_version="1.0",experiment_id="experiment-008-opencl-inner-loop-unrolling",
        timestamp=base.timestamp(),hypothesis="Reduce inner-loop control/index overhead through explicit unrolling",
        baseline=dict(path=str(LOCKED.relative_to(ROOT)),sha256=LOCKED_SHA256,median_ms=3.71454),
        workload=dict(M=512,N=512,K=512,precision="FP32"),tile=[8,8],mapping="column_first",
        allowed_atol=base.ATOL,allowed_rtol=base.RTOL,
        protocol=dict(warmups=base.WARMUPS,measured_iterations=base.ITERATIONS,independent_runs=base.RUNS,
                      primary_metric="completed OpenCL kernel event latency",
                      build_options="-cl-std=CL1.2; FP_CONTRACT OFF; no fast-math"),
        discovery=discovery,decision_rule=RULE,environment=base.common_record()["environment"],
        provenance=dict(script_sha256=base.digest(__file__),reused_experiment7_sha256=base.digest(mapping.__file__)))
    base.save_record(Path(str(prefix)+"_plan.json"),record)
    runs=[]
    if discovery["1"]["legal"]:
        for index in range(base.RUNS):
            print(f"Exact Experiment 007 control: run {index+1}/3",flush=True)
            run=run_child(1,prefix,index);runs.append(run)
            if run["result"]!="PASS": break
    control=summarize(1,runs,locked)
    gate=mapping.control_gate(control,locked)
    base.save_record(Path(str(prefix)+"_control_summary.json"),control)
    print("Control reproducibility:",json.dumps(gate),flush=True)
    results={f:[] for f in FACTORS}
    blocked=set()
    if gate["passed"]:
        for round_index in range(base.RUNS):
            order=FACTORS[round_index:]+FACTORS[:round_index]
            for factor in order:
                if not discovery[str(factor)]["legal"] or factor in blocked: continue
                print(f"Unroll {factor}: independent run {round_index+1}/3",flush=True)
                run=run_child(factor,prefix,round_index);results[factor].append(run)
                if run["result"]!="PASS":blocked.add(factor)
    candidates=[]
    for factor in FACTORS:
        if not results[factor]:
            candidate=dict(result="NOT_RUN",decision="REJECT",unroll_factor=factor,
                           variant=f"unroll_{factor}",reason="Control reproducibility or compiled legality gate failed",runs=[])
        else:
            candidate=summarize(factor,results[factor],locked,control,gate)
        candidates.append(candidate)
        base.save_record(Path(str(prefix)+f"_factor_{factor}_summary.json"),candidate)
    best=best_candidate(candidates)
    record.update(control=control,control_gate=gate,candidates=candidates,
                  decision="ADOPT" if best else "REJECT",
                  recommended_factor=best["unroll_factor"] if best else None)
    record["explanation"]=(
        f"Factor {best['unroll_factor']} passes all gates; {best['improvement_percent']:.4f}% lower kernel median versus locked Experiment 007."
        if best else "No candidate passed all correctness, reproducibility and improvement gates. Retain locked Experiment 007.")
    verify_preserved()
    base.save_record(Path(str(prefix)+"_summary.json"),record)
    with Path(str(prefix)+"_report.md").open("x",encoding="utf-8") as stream:stream.write(report(record))
    print(record["decision"],record["explanation"])
    print("Summary:",str(prefix)+"_summary.json")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
