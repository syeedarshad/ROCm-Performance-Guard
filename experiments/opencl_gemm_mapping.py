"""Experiment 007: change only row/column work-item axis mapping of locked 8x8 GEMM."""
from __future__ import annotations
import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from experiments import opencl_gemm_baseline as base
from experiments import opencl_gemm_tiling as tiled

OUTPUT=ROOT/"experiments/results/experiment_007"
MANIFEST=OUTPUT/"experiment_007_preservation_manifest.json"
LOCKED=ROOT/"experiments/results/experiment_006/experiment_006_20261003_161254_144008_summary.json"
LOCKED_SHA256="67637ec667dcc5bee35e984344a9edfa74871c3fb147831a2a85d355a44ff9da"
TILE=8
VARIANTS=("row_fast","column_fast")
RULE=dict(control_max_absolute_drift_percent=5.0,control_max_run_cv_percent=5.0,
          candidate_gate=dict(tiled.RULE),
          additional_gate="Must pass the same improvement gates against fresh control AND locked Experiment 006",
          failure_action="REJECT; no adoption if control is unreproduced or any correctness/protocol check fails")


def verify_preserved():
    for item in json.loads(MANIFEST.read_text())["files"]:
        if base.digest(ROOT/item["path"])!=item["sha256"]:
            raise RuntimeError("Immutable file changed: "+item["path"])
    if base.digest(LOCKED)!=LOCKED_SHA256:
        raise RuntimeError("Experiment 006 checksum mismatch")
    base.verify_locked_baseline()


def locked_reference():
    document=json.loads(LOCKED.read_text())
    record=next(c for c in document["candidates"] if c["tile"]==8)
    if record["result"]!="PASS" or record["decision"]!="ADOPT":
        raise RuntimeError("Locked 8x8 configuration is not adopted and correct")
    # Adapter for Experiment 006's established summary/decision code.
    return dict(record,kernel_median_between_runs_cv_percent=record["run_median_cv_percent"])


def kernel_source(variant):
    source=tiled.kernel_source(TILE)
    if variant=="row_fast":
        return source
    if variant!="column_fast":
        raise ValueError("Unknown mapping")
    changes={
        "row = get_global_id(0)":"row = get_global_id(1)",
        "col = get_global_id(1)":"col = get_global_id(0)",
        "lr = get_local_id(0)":"lr = get_local_id(1)",
        "lc = get_local_id(1)":"lc = get_local_id(0)",
    }
    for before,after in changes.items():
        if source.count(before)!=1:
            raise RuntimeError("Locked source layout changed")
        source=source.replace(before,after)
    return source


class ColumnFastOpenCL(tiled.TiledOpenCL):
    def __init__(self):
        super().__init__(TILE)
        try:
            source=ct.c_char_p(kernel_source("column_fast").encode())
            err=base.I()
            program=self.call("clCreateProgramWithSource",
                [base.P,base.U,base.P,base.P,ct.POINTER(base.I)],base.P)(
                self.context,1,ct.byref(source),None,ct.byref(err))
            base.check(err.value,"create column-fast program")
            self.resources.append(("clReleaseProgram",program))
            device=base.P(self.device)
            rc=self.call("clBuildProgram",[base.P,base.U,base.P,ct.c_char_p,base.P,base.P])(
                program,1,ct.byref(device),b"-cl-std=CL1.2",None,None)
            if rc:
                log=ct.create_string_buffer(65536)
                self.call("clGetProgramBuildInfo",[base.P,base.P,base.U,base.S,base.P,base.P])(
                    program,device,0x1183,len(log),log,None)
                raise RuntimeError("Column-fast build failed: "+log.value.decode(errors="replace"))
            self.kernel=self.call("clCreateKernel",[base.P,ct.c_char_p,ct.POINTER(base.I)],base.P)(
                program,b"gemm_tiled",ct.byref(err))
            base.check(err.value,"create column-fast kernel")
            self.resources.append(("clReleaseKernel",self.kernel))
            self.kernel_limits=tiled.kernel_capabilities(self,self.kernel)
            reasons=tiled.tile_legality(TILE,self.device_limits,self.kernel_limits)
            if reasons:
                raise ValueError("; ".join(reasons))
            for index,buffer in enumerate(self.buffers):
                handle=base.P(buffer)
                base.check(self.call("clSetKernelArg",[base.P,base.U,base.S,base.P])(
                    self.kernel,index,ct.sizeof(handle),ct.byref(handle)),"column-fast argument")
        except Exception:
            self.close()
            raise


def measure_candidate(backend_factory=ColumnFastOpenCL):
    # Identical order and operations to Experiment 006.one_run; no module monkeypatching.
    run=dict(tile=TILE,variant="column_fast",timestamp=base.timestamp(),process_id=os.getpid(),
             result="FAILED",samples=[],warmup_checks=[],warmups_completed=0)
    try:
        with backend_factory() as backend:
            run.update(device=backend.metadata,device_capabilities=backend.device_limits,
                       kernel_capabilities=backend.kernel_limits,
                       kernel_sha256=hashlib.sha256(kernel_source("column_fast").encode()).hexdigest())
            a,b=base.deterministic_inputs()
            reference=base.cpu_reference(a,b)
            run["input_sha256"]=dict(A=hashlib.sha256(bytes(a)).hexdigest(),B=hashlib.sha256(bytes(b)).hexdigest())
            run["preflight_timing"]=backend.sample(a,b)
            run["correctness"]=base.correctness(reference,backend.output)
            if run["correctness"]["result"]!="PASS":
                run["reason"]="Preflight correctness failed; no benchmark"
                return run
            for _ in range(base.WARMUPS):
                backend.sample(a,b)
                check=base.correctness(reference,backend.output)
                run["warmup_checks"].append(check)
                if check["result"]!="PASS":
                    run["reason"]="Warmup correctness failed; no adoption"
                    return run
                run["warmups_completed"]+=1
            for index in range(base.ITERATIONS):
                sample=backend.sample(a,b)
                sample.update(iteration=index+1,correctness=base.correctness(reference,backend.output))
                run["samples"].append(sample)
                if sample["correctness"]["result"]!="PASS":
                    run["reason"]="Measured correctness failed; no adoption"
                    return run
            run["timing_ms"]={key:base.timing_summary([s[key] for s in run["samples"]])
                for key in ("kernel_ms","h2d_ms","d2h_ms","end_to_end_ms")}
            run["gflops_from_median_kernel"]=base.gflops(run["timing_ms"]["kernel_ms"]["median"])
            run["result"]="PASS"
    except Exception as exc:
        run.update(result="FAILED",reason=f"{type(exc).__name__}: {exc}")
    return run


def control_gate(control,locked):
    if control.get("result")!="PASS":
        return dict(passed=False,reason="Control correctness/protocol failed")
    median=control["aggregate_timing_ms"]["kernel_ms"]["median"]
    reference=locked["aggregate_timing_ms"]["kernel_ms"]["median"]
    drift=(median-reference)/reference*100
    passed=abs(drift)<=RULE["control_max_absolute_drift_percent"] and control["run_median_cv_percent"]<=RULE["control_max_run_cv_percent"]
    return dict(passed=passed,signed_drift_percent=drift,absolute_drift_percent=abs(drift),
                run_cv_percent=control["run_median_cv_percent"],
                reason="Control reproduced within predefined bounds" if passed else
                       "Control drift or variability exceeds predefined bounds; do not adopt")


def final_gate(candidate,control,gate):
    if candidate.get("result")!="PASS" or not gate["passed"]:
        return dict(decision="REJECT",reason="Correctness/protocol or baseline reproducibility gate failed")
    timing=candidate["aggregate_timing_ms"]["kernel_ms"]
    fresh=control["aggregate_timing_ms"]["kernel_ms"]
    current=tiled.decide(fresh["median"],fresh["p95"],control["run_median_cv_percent"],
                        timing["median"],timing["p95"],candidate["run_median_cv_percent"],
                        candidate["run_medians_ms"])
    adopted=candidate["decision"]=="ADOPT" and current["decision"]=="ADOPT"
    return dict(decision="ADOPT" if adopted else "REJECT",fresh_control_comparison=current,
                reason="Passes gates against both locked and fresh control" if adopted else
                       "Does not pass all improvement gates against both locked and fresh control")


def run_child(variant,prefix,index):
    proc=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),
                         "--single-run",variant],capture_output=True,text=True)
    try:
        run=json.loads(proc.stdout)
    except (ValueError,json.JSONDecodeError):
        run=dict(result="FAILED",variant=variant,reason=proc.stderr or proc.stdout or "No child output",
                 process_id=None,samples=[])
    if proc.returncode and run.get("result")=="PASS":
        run.update(result="FAILED",reason="Nonzero child exit code")
    run["round"]=index+1
    base.save_record(Path(str(prefix)+f"_{variant}_run_{index+1:03d}.json"),run)
    return run


def report(record):
    lines=["# Experiment 007 — 8x8 GEMM work-item axis mapping","",
           "## Locked reference and hypothesis","",
           "Experiment 006: AMD Radeon gfx90c, native Windows OpenCL, FP32 512x512x512, 8x8 tiles.",
           "Locked kernel median: 6.13649 ms. This result and all prior artifacts remain immutable.",
           "Independent variable: mapping of work-item dimensions to rows/columns.",
           "Control is the exact Experiment 006 kernel. Candidate maps dimension 0 to columns at both global and local scope.",
           "Hypothesis: adjacent column accesses may improve global-memory access efficiency for row-major A/B/C.",
           "Only four axis assignments change. Tile size, local-memory arrays, barriers, arithmetic order, FP32, compiler options and inputs are unchanged.",
           "No profiling counters are collected; an observed gain does not isolate a hardware-level causal mechanism.","",
           "## Device and compiled legality","", "```json",json.dumps(record["discovery"],indent=2),"```","",
           "## Protocol and predeclared gates","",
           "Each configuration: 3 fresh processes, 1 correctness preflight, 10 warmups, 50 measured iterations per process.",
           "Every output is checked against the same CPU math.fsum reference rounded to FP32.",
           "atol=0.001, rtol=0.001; finite outputs and abs(GPU-CPU) <= atol + rtol*abs(CPU).",
           "Relative-error denominator: max(abs(CPU), 1e-12).",
           "Primary latency comes from completed kernel OpenCL event START/END timestamps on the same in-order profiling queue.",
           "Transfers and synchronized end-to-end timing remain separate, with identical Experiment 006 timing boundaries.",
           "The control is remeasured first. Candidate measurements are skipped if the control fails reproducibility.",
           "```json",json.dumps(record["decision_rule"],indent=2),"```","",
           "Variability margins are heuristics, not formal statistical-significance tests. Runs are sequential; thermal and background state are not controlled.",
           "Fresh control supplements, and never replaces, the locked comparison.","",
           "## Baseline reproducibility","", "```json",json.dumps(record["control_gate"],indent=2),"```","",
           "## Kernel results","",
           "| Configuration | Status | Median ms | Mean ms | P95 ms | Min ms | Max ms | GFLOPS | Run CV % | Speedup vs locked | Reduction % |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("control","candidate"):
        c=record[label]
        if c.get("result")!="PASS":
            lines.append(f"| {label} | {c.get('result')} | — | — | — | — | — | — | — | — | — |")
            continue
        t=c["aggregate_timing_ms"]["kernel_ms"]
        lines.append(f"| {label} | PASS | {t['median']:.6f} | {t['mean']:.6f} | {t['p95']:.6f} | {t['min']:.6f} | {t['max']:.6f} | {c['gflops']:.6f} | {c['run_median_cv_percent']:.4f} | {c['speedup_vs_baseline']:.6f} | {c['improvement_percent']:.4f} |")
        lines += []
    for label in ("control","candidate"):
        c=record[label]
        lines += ["",f"## {label.title()} detailed evidence",""]
        if c.get("result")!="PASS":
            lines += [c.get("reason","Not measured"),""]
            continue
        lines += ["Run medians (ms): "+json.dumps(c["run_medians_ms"]),
                  "Worst correctness diagnostics across all measured outputs:",
                  "```json",json.dumps(c["worst_correctness_metrics"],indent=2),"```",
                  "All timing components (ms):","```json",
                  json.dumps(c["aggregate_timing_ms"],indent=2),"```",""]
    lines += ["## Final decision","",
              "**"+record["decision"]+"**: "+record["explanation"],
              "```json",json.dumps(record["adoption_gate"],indent=2),"```","",
              "An ADOPT recommendation applies only to this measured configuration; old baselines stay locked.",
              "Raw events, all correctness checks, source/input hashes, device limits and independent-run records are saved.",
              "## Reproduction","", "```powershell",
              "python -B experiments/opencl_gemm_mapping.py","```","",
              "Requires the existing native Windows AMD OpenCL runtime. No packages, drivers or system settings are changed.",
              "Timestamped artifacts use exclusive creation. No CUDA/HIP/ROCm execution. Stop after Experiment 007.",""]
    return "\n".join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single-run",choices=VARIANTS,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    verify_preserved()
    if args.single_run:
        run=tiled.one_run(TILE) if args.single_run=="row_fast" else measure_candidate()
        run["variant"]=args.single_run
        verify_preserved()
        print(json.dumps(run,allow_nan=False))
        return 0 if run["result"]=="PASS" else 1
    locked=locked_reference()
    discovery={}
    for variant,factory in (("row_fast",lambda:tiled.TiledOpenCL(TILE)),("column_fast",ColumnFastOpenCL)):
        try:
            with factory() as b:
                discovery[variant]=dict(legal=True,device=b.metadata,
                    device_capabilities=b.device_limits,kernel_capabilities=b.kernel_limits,
                    kernel_sha256=hashlib.sha256(kernel_source(variant).encode()).hexdigest())
        except Exception as exc:
            discovery[variant]=dict(legal=False,reason=str(exc))
    token=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix=OUTPUT/f"experiment_007_{token}"
    record=dict(schema_version="1.0",experiment_id="experiment-007-opencl-work-item-mapping",
                timestamp=base.timestamp(),hypothesis="Improve row-major memory access by mapping dimension 0 to columns",
                baseline=dict(path=str(LOCKED.relative_to(ROOT)),sha256=LOCKED_SHA256,
                              tile=8,median_ms=locked["aggregate_timing_ms"]["kernel_ms"]["median"]),
                workload=dict(M=512,N=512,K=512,precision="FP32"),allowed_atol=base.ATOL,allowed_rtol=base.RTOL,
                protocol=dict(warmups=base.WARMUPS,measured_iterations=base.ITERATIONS,independent_runs=base.RUNS,
                              primary_metric="completed OpenCL kernel event latency",order=["row_fast x3","column_fast x3"]),
                discovery=discovery,decision_rule=RULE,
                provenance=dict(script_sha256=base.digest(__file__),experiment6_script_sha256=base.digest(tiled.__file__),
                                experiment5_script_sha256=base.digest(base.__file__)),
                environment=base.common_record()["environment"])
    base.save_record(Path(str(prefix)+"_plan.json"),record)
    runs=[]
    if discovery["row_fast"]["legal"]:
        for i in range(base.RUNS):
            print(f"Exact Experiment 006 control: independent run {i+1}/3",flush=True)
            run=run_child("row_fast",prefix,i)
            runs.append(run)
            if run["result"]!="PASS":
                break
    control=tiled.summarize(TILE,runs,locked)
    control["variant"]="row_fast"
    # summarize reuses Experiment 006 decision code; control's comparison is descriptive, not a candidate decision.
    control["role"]="reproducibility_control"
    gate=control_gate(control,locked)
    base.save_record(Path(str(prefix)+"_control_summary.json"),control)
    print("Control reproducibility:",json.dumps(gate),flush=True)
    runs=[]
    if gate["passed"] and discovery["column_fast"]["legal"]:
        for i in range(base.RUNS):
            print(f"Column-fast candidate: independent run {i+1}/3",flush=True)
            run=run_child("column_fast",prefix,i)
            runs.append(run)
            if run["result"]!="PASS":
                break
        candidate=tiled.summarize(TILE,runs,locked)
    else:
        candidate=dict(result="NOT_RUN",decision="REJECT",
                       reason="Control gate failed or candidate failed compiled legality check",runs=[])
    candidate["variant"]="column_fast"
    base.save_record(Path(str(prefix)+"_candidate_summary.json"),candidate)
    adoption=final_gate(candidate,control,gate)
    record.update(control=control,control_gate=gate,candidate=candidate,
                  adoption_gate=adoption,decision=adoption["decision"],explanation=adoption["reason"])
    verify_preserved()
    base.save_record(Path(str(prefix)+"_summary.json"),record)
    with Path(str(prefix)+"_report.md").open("x",encoding="utf-8") as f:
        f.write(report(record))
    print(record["decision"],record["explanation"])
    print("Summary:",str(prefix)+"_summary.json")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
