"""Experiment 006: isolated AMD OpenCL local-memory tiling versus locked Experiment 005."""
from __future__ import annotations
import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import opencl_gemm_baseline as base

GPU_BASELINE = ROOT / "experiments/results/experiment_005/experiment_005_20261003_145248_372949_summary.json"
GPU_BASELINE_SHA256 = "f207e61543d17cbd3aaefc82d489ca55f47b00e8b9e3cc3403f27760c76dfb58"
CANDIDATES = (8, 16, 32)
M, N, K = base.M, base.N, base.K
OUTPUT = ROOT / "experiments/results/experiment_006"
RULE = {
    "minimum_improvement_percent": 5.0,
    "maximum_p95_ratio": 1.10,
    "variability_margin": "improvement > 2 * max(baseline run-median CV%, candidate run-median CV%)",
    "repeatability": "every candidate run median improves at least 5% versus locked pooled baseline median",
    "correctness": "all preflight, warmup and measured outputs PASS at atol=rtol=0.001",
    "limitation": "Variability margin is a conservative heuristic, not a statistical significance test.",
}


def verify_locked():
    base.verify_locked_baseline()
    if base.digest(GPU_BASELINE) != GPU_BASELINE_SHA256:
        raise RuntimeError("Locked Experiment 005 summary checksum differs")
    return json.loads(GPU_BASELINE.read_text(encoding="utf-8"))


def kernel_source(tile):
    if tile not in CANDIDATES:
        raise ValueError("Tile must be one of the preregistered candidates")
    return f"""
#pragma OPENCL FP_CONTRACT OFF
#define TILE {tile}
__attribute__((reqd_work_group_size(TILE, TILE, 1)))
__kernel void gemm_tiled(__global const float *A,
                         __global const float *B,
                         __global float *C) {{
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int lr = get_local_id(0);
    const int lc = get_local_id(1);
    __local float tile_a[TILE][TILE];
    __local float tile_b[TILE][TILE];
    float value = 0.0f;
    for (int block = 0; block < 512; block += TILE) {{
        tile_a[lr][lc] = A[row * 512 + block + lc];
        tile_b[lr][lc] = B[(block + lr) * 512 + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; ++k)
            value += tile_a[lr][k] * tile_b[k][lc];
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    C[row * 512 + col] = value;
}}
"""


def device_capabilities(backend):
    dims = backend.scalar_info(backend.device, 0x1003, base.U)
    sizes = (base.S * dims)()
    base.check(backend.call("clGetDeviceInfo", [base.P,base.U,base.S,base.P,base.P])(
        backend.device, 0x1005, ct.sizeof(sizes), sizes, None), "work-item limits")
    return dict(max_work_group_size=backend.scalar_info(backend.device, 0x1004, base.S),
                max_work_item_sizes=list(sizes),
                local_memory_bytes=backend.scalar_info(backend.device, 0x1023, base.Q))


def kernel_capabilities(backend, kernel):
    f = backend.call("clGetKernelWorkGroupInfo", [base.P,base.P,base.U,base.S,base.P,base.P])
    result = {}
    for key, param, typ in (
        ("max_work_group_size", 0x11B0, base.S),
        ("preferred_work_group_multiple", 0x11B3, base.S),
        ("local_memory_bytes", 0x11B2, base.Q),
        ("private_memory_bytes", 0x11B4, base.Q),
    ):
        value = typ()
        base.check(f(kernel, backend.device, param, ct.sizeof(value), ct.byref(value), None),
                   "kernel work-group limits")
        result[key] = value.value
    sizes = (base.S * 3)()
    base.check(f(kernel, backend.device, 0x11B1, ct.sizeof(sizes), sizes, None),
               "compiled work-group size")
    result["compile_work_group_size"] = list(sizes)
    return result


def tile_legality(tile, device, kernel=None):
    reasons = []
    if tile not in CANDIDATES or tile <= 0:
        return ["Tile is not a preregistered candidate"]
    if any(size % tile for size in (M,N,K)):
        reasons.append("Tile must divide fixed dimensions")
    if tile * tile > device["max_work_group_size"]:
        reasons.append("Work-group exceeds device work-item limit")
    if len(device["max_work_item_sizes"]) < 2 or any(
        tile > limit for limit in device["max_work_item_sizes"][:2]):
        reasons.append("Per-dimension work-item limit exceeded")
    if 2 * tile * tile * 4 > device["local_memory_bytes"]:
        reasons.append("Two FP32 local tiles exceed device local memory")
    if kernel is not None:
        if tile * tile > kernel["max_work_group_size"]:
            reasons.append("Work-group exceeds compiled kernel limit")
        if kernel["local_memory_bytes"] > device["local_memory_bytes"]:
            reasons.append("Compiled kernel local memory exceeds device limit")
        if kernel["compile_work_group_size"] not in ([0,0,0], [tile,tile,1]):
            reasons.append("Compiled required work-group size does not match tile")
    return reasons


class TiledOpenCL(base.WindowsOpenCL):
    """Reuse frozen baseline allocation, transfers and event profiling; replace only kernel/local size."""
    def __init__(self, tile):
        super().__init__()
        try:
            self.tile = tile
            self.device_limits = device_capabilities(self)
            reasons = tile_legality(tile, self.device_limits)
            if reasons:
                raise ValueError("; ".join(reasons))
            source = ct.c_char_p(kernel_source(tile).encode())
            err = base.I()
            program = self.call("clCreateProgramWithSource",
                               [base.P,base.U,base.P,base.P,ct.POINTER(base.I)],base.P)(
                self.context,1,ct.byref(source),None,ct.byref(err))
            base.check(err.value, "create tiled program")
            self.resources.append(("clReleaseProgram",program))
            device = base.P(self.device)
            rc = self.call("clBuildProgram",
                           [base.P,base.U,base.P,ct.c_char_p,base.P,base.P])(
                program,1,ct.byref(device),b"-cl-std=CL1.2",None,None)
            if rc:
                log = ct.create_string_buffer(65536)
                self.call("clGetProgramBuildInfo",
                          [base.P,base.P,base.U,base.S,base.P,base.P])(
                    program,device,0x1183,len(log),log,None)
                raise RuntimeError("Tiled build failed: "+log.value.decode(errors="replace"))
            self.kernel = self.call("clCreateKernel",[base.P,ct.c_char_p,ct.POINTER(base.I)],base.P)(
                program,b"gemm_tiled",ct.byref(err))
            base.check(err.value,"create tiled kernel")
            self.resources.append(("clReleaseKernel",self.kernel))
            self.kernel_limits = kernel_capabilities(self,self.kernel)
            reasons = tile_legality(tile,self.device_limits,self.kernel_limits)
            if reasons:
                raise ValueError("; ".join(reasons))
            for index,buf in enumerate(self.buffers):
                handle = base.P(buf)
                base.check(self.call("clSetKernelArg",[base.P,base.U,base.S,base.P])(
                    self.kernel,index,ct.sizeof(handle),ct.byref(handle)),"tiled kernel arg")
            self.local_size = (base.S * 2)(tile,tile)
            self.raw_enqueue = self.enqueue
            self.enqueue = self.enqueue_tiled
        except Exception:
            self.close()
            raise

    def enqueue_tiled(self,queue,kernel,dimensions,offset,global_size,local_size,nwait,waits,event):
        return self.raw_enqueue(queue,kernel,dimensions,offset,global_size,
                                self.local_size,nwait,waits,event)


def one_run(tile):
    verify_locked()
    run = dict(tile=tile,timestamp=base.timestamp(),process_id=os.getpid(),
               result="FAILED",samples=[],warmup_checks=[],warmups_completed=0)
    try:
        with TiledOpenCL(tile) as backend:
            run.update(device=backend.metadata,device_capabilities=backend.device_limits,
                       kernel_capabilities=backend.kernel_limits,
                       kernel_sha256=hashlib.sha256(kernel_source(tile).encode()).hexdigest())
            a,b = base.deterministic_inputs()
            reference = base.cpu_reference(a,b)
            run["input_sha256"] = dict(A=hashlib.sha256(bytes(a)).hexdigest(),
                                       B=hashlib.sha256(bytes(b)).hexdigest())
            # Same preflight + 10 warmups + 50 measurements as locked Experiment 005.
            run["preflight_timing"] = backend.sample(a,b)
            run["correctness"] = base.correctness(reference,backend.output)
            if run["correctness"]["result"] != "PASS":
                run["reason"] = "Preflight correctness failed; performance not eligible"
                return run
            for _ in range(base.WARMUPS):
                backend.sample(a,b)
                verified = base.correctness(reference,backend.output)
                run["warmup_checks"].append(verified)
                if verified["result"] != "PASS":
                    run["reason"] = "Warmup correctness failed; performance not eligible"
                    return run
                run["warmups_completed"] += 1
            for index in range(base.ITERATIONS):
                sample = backend.sample(a,b)
                sample.update(iteration=index+1,correctness=base.correctness(reference,backend.output))
                run["samples"].append(sample)
                if sample["correctness"]["result"] != "PASS":
                    run["reason"] = "Measured correctness failed; performance not eligible"
                    return run
            run["timing_ms"] = {key:base.timing_summary([s[key] for s in run["samples"]])
                                for key in ("kernel_ms","h2d_ms","d2h_ms","end_to_end_ms")}
            run["gflops_from_median_kernel"] = base.gflops(run["timing_ms"]["kernel_ms"]["median"])
            run["result"] = "PASS"
    except Exception as exc:
        run.update(result="FAILED",reason=f"{type(exc).__name__}: {exc}")
    finally:
        verify_locked()
    return run


def decide(baseline_ms,baseline_p95,baseline_cv,candidate_ms,candidate_p95,candidate_cv,run_medians,
           correct=True):
    if not correct:
        return dict(decision="REJECT",verification_status="FAILED",
                    reason="Correctness or measurement protocol failed")
    values = [baseline_ms,baseline_p95,candidate_ms,candidate_p95,*run_medians]
    if not run_medians or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Invalid timing")
    if any(not math.isfinite(x) or x < 0 for x in (baseline_cv,candidate_cv)):
        raise ValueError("Invalid run variability")
    improvement = (baseline_ms-candidate_ms)/baseline_ms*100
    margin = 2*max(baseline_cv,candidate_cv)
    checks = dict(minimum_gain=improvement>=5,
                  exceeds_variability=improvement>margin,
                  p95_guard=candidate_p95<=baseline_p95*1.10,
                  repeated_gain=all((baseline_ms-x)/baseline_ms*100>=5 for x in run_medians))
    adopted = all(checks.values())
    status = "IMPROVED" if adopted else "REGRESSED" if improvement<0 else "NO_MEANINGFUL_CHANGE"
    return dict(decision="ADOPT" if adopted else "REJECT",verification_status=status,
                speedup_vs_baseline=baseline_ms/candidate_ms,improvement_percent=improvement,
                variability_margin_percent=margin,checks=checks,
                reason="All correctness, gain, variability and P95 gates pass" if adopted else
                "Failed gates: "+", ".join(k for k,v in checks.items() if not v))


def summarize(tile,runs,locked):
    record = dict(tile=tile,local_work_size=[tile,tile],work_items=tile*tile,
                  theoretical_local_memory_bytes=2*tile*tile*4,runs=runs,
                  result="FAILED",decision="REJECT")
    expected_inputs = locked["runs"][0]["input_sha256"]
    expected_device = locked["runs"][0]["device"]
    valid = len(runs)==base.RUNS and len({r.get("process_id") for r in runs})==base.RUNS
    for run in runs:
        valid = valid and (
            run.get("result")=="PASS" and run.get("warmups_completed")==base.WARMUPS and
            len(run.get("samples",[]))==base.ITERATIONS and
            run.get("input_sha256")==expected_inputs and run.get("device")==expected_device and
            run.get("correctness",{}).get("result")=="PASS" and
            len(run.get("warmup_checks",[]))==base.WARMUPS and
            all(s["result"]=="PASS" for s in run.get("warmup_checks",[])) and
            all(s["correctness"]["result"]=="PASS" for s in run.get("samples",[])))
    if not valid:
        record["reason"] = "Incomplete/failed protocol, input/device mismatch, or correctness failure"
        return record
    samples = [s for run in runs for s in run["samples"]]
    timing = {key:base.timing_summary([s[key] for s in samples])
              for key in ("kernel_ms","h2d_ms","d2h_ms","end_to_end_ms")}
    medians = [statistics.median(s["kernel_ms"] for s in run["samples"]) for run in runs]
    cv = statistics.stdev(medians)/statistics.mean(medians)*100
    baseline = locked["aggregate_timing_ms"]["kernel_ms"]
    decision = decide(baseline["median"],baseline["p95"],locked["kernel_median_between_runs_cv_percent"],
                      timing["kernel_ms"]["median"],timing["kernel_ms"]["p95"],cv,medians)
    record.update(result="PASS",aggregate_timing_ms=timing,run_medians_ms=medians,
                  run_median_cv_percent=cv,kernel_capabilities=runs[0]["kernel_capabilities"],
                  gflops=base.gflops(timing["kernel_ms"]["median"]),
                  worst_correctness_metrics={key:max(s["correctness"][key] for s in samples)
                  for key in ("max_absolute_error","mean_absolute_error","max_relative_error","mean_relative_error")},
                  **decision)
    return record


def inspect():
    with base.WindowsOpenCL() as backend:
        return dict(device=backend.metadata,device_capabilities=device_capabilities(backend),
                    baseline_kernel_capabilities=kernel_capabilities(backend,backend.kernel))


def report(record):
    locked = record["baseline"]
    lines = ["# Experiment 006 — OpenCL GEMM local-memory tiling","",
        "## Baseline","",
        f"Locked Experiment 005: AMD Radeon gfx90c, native Windows OpenCL, 512 x 512 x 512 FP32.",
        f"Kernel median {locked['median_ms']:.6f} ms; GFLOPS {locked['gflops']:.6f}; run-median CV {locked['cv_percent']:.4f}%.",
        f"Summary SHA256: {GPU_BASELINE_SHA256}","",
        "## Optimization hypothesis","",
        "Cooperatively load A and B tiles into __local memory and reuse them across work-items, reducing repeated global loads.",
        "Only the kernel and its required local work size change. Inputs, reference, arithmetic precision, accumulation order, tolerances,",
        "compiler flags, transfers, event profiling and measurement counts are held fixed. No transposed B, vectorization or fast math.",
        "Observed latency can support this configuration, but without hardware counters does not prove a cache/bandwidth cause.","",
        "## Device capabilities","", "```json",json.dumps(record["discovery"],indent=2),"```","",
        "## Candidate configurations and timing results","",
        "| Tile | Status | Median ms | Mean ms | P95 ms | Min ms | Max ms | GFLOPS | Speedup | Improvement % | Run CV % | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in record["candidates"]:
        if c["result"]!="PASS":
            lines.append(f"| {c['tile']}x{c['tile']} | {c['result']} | — | — | — | — | — | — | — | — | — | REJECT |")
            continue
        s=c["aggregate_timing_ms"]["kernel_ms"]
        lines.append(f"| {c['tile']}x{c['tile']} | PASS | {s['median']:.6f} | {s['mean']:.6f} | {s['p95']:.6f} | {s['min']:.6f} | {s['max']:.6f} | {c['gflops']:.6f} | {c['speedup_vs_baseline']:.6f} | {c['improvement_percent']:.4f} | {c['run_median_cv_percent']:.4f} | {c['decision']} |")
    lines += ["","## Correctness results","",
        "Same deterministic generator and byte-identical FP32 A/B inputs as Experiment 005.",
        "CPU reference: math.fsum over the FP32 inputs, rounded to FP32 output.",
        "All elements must be finite and satisfy abs(GPU-CPU) <= 0.001 + 0.001*abs(CPU).",
        "Relative error denominator: max(abs(CPU), 1e-12). Statistics below are worst diagnostics across measured outputs.","",
        "| Tile | Max absolute | Mean absolute | Max relative | Mean relative |",
        "|---|---:|---:|---:|---:|"]
    for c in record["candidates"]:
        if c["result"]=="PASS":
            e=c["worst_correctness_metrics"]
            lines.append(f"| {c['tile']} | {e['max_absolute_error']:.10e} | {e['mean_absolute_error']:.10e} | {e['max_relative_error']:.10e} | {e['mean_relative_error']:.10e} |")
    lines += ["","## Per-candidate evidence",""]
    for c in record["candidates"]:
        lines += [f"### Tile {c['tile']}x{c['tile']}","",c.get("reason",""),""]
        if c["result"]=="PASS":
            lines += ["Compiled kernel limits: "+json.dumps(c["kernel_capabilities"]),
                      "Run medians (ms): "+json.dumps(c["run_medians_ms"]),
                      "Secondary transfer and synchronized end-to-end timings (ms):",
                      "```json",json.dumps(c["aggregate_timing_ms"],indent=2),"```",""]
    lines += ["## Measurement and decision rules","",
        "Per candidate: three fresh child processes/contexts; each performs one correctness preflight, 10 warmups and 50 measurements.",
        "Every warmup and measured output is checked. Incorrect candidates are FAILED and cannot be adopted.",
        "In-order profiling queue; kernel START/END event timestamps queried only after waiting for final read completion.",
        "Primary metric: pooled median kernel latency. H2D, D2H, synchronized end-to-end remain secondary.",
        "Identical Experiment 005 timing boundaries exclude CPU reference, compilation, allocation and validation.",
        "Speedup = baseline median / candidate median. Improvement = (baseline median - candidate median) / baseline median * 100.",
        "```json",json.dumps(record["decision_rule"],indent=2),"```","",
        "The 5% threshold follows EXPERIMENT_PROTOCOL.md. Added variability, per-run and P95 gates are declared before measurements.",
        "Preferred work-group multiples are performance hints, not proof of optimality. Each compiled kernel's hard limits are checked.",
        "Candidates run in rotated order across rounds to reduce order bias. Separate processes do not control thermals or background load.",
        "Comparison is against a historical locked baseline, not a contemporaneous randomized control; no causal or statistical-significance claim is made.","",
        "## Best configuration and final decision","",
        f"Best measured correct tile: {record['best_configuration']}.",
        f"Final decision: **{record['decision']}**. Recommended tile: {record['recommended_tile']}.",
        record["explanation"],
        "ADOPT is a recommendation for this local branch; Experiment 005 remains locked and unchanged.","",
        "## Reproduction","", "```powershell",
        "python -B experiments/opencl_gemm_tiling.py --inspect",
        "python -B experiments/opencl_gemm_tiling.py","```","",
        "Requires this Acer's existing native AMD OpenCL runtime. No installation, CUDA, HIP or ROCm.",
        "Timestamped files use exclusive creation. Raw run results, capability checks, event timestamps and input hashes are preserved.",
        "CPU locked baseline SHA256: "+base.BASELINE_SHA256,
        "Stop after Experiment 006; no follow-on optimization is executed.",""]
    return "\n".join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect",action="store_true")
    p.add_argument("--single-run",type=int,choices=CANDIDATES,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    locked=verify_locked()
    if args.single_run is not None:
        run=one_run(args.single_run)
        print(json.dumps(run,allow_nan=False))
        return 0 if run["result"]=="PASS" else 1
    discovery=inspect()
    if args.inspect:
        print(json.dumps(discovery,indent=2))
        return 0
    token=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix=OUTPUT/f"experiment_006_{token}"
    record=dict(schema_version="1.0",experiment_id="experiment-006-opencl-local-memory-tiling",
        timestamp=base.timestamp(),hypothesis="Reuse global A/B loads through __local tiles",
        baseline=dict(path=str(GPU_BASELINE.relative_to(ROOT)),sha256=GPU_BASELINE_SHA256,
                      median_ms=locked["aggregate_timing_ms"]["kernel_ms"]["median"],
                      p95_ms=locked["aggregate_timing_ms"]["kernel_ms"]["p95"],
                      gflops=locked["gflops_from_median_kernel"],
                      cv_percent=locked["kernel_median_between_runs_cv_percent"]),
        workload=locked["workload"],protocol=dict(locked["protocol"]),
        discovery=discovery,decision_rule=RULE,allowed_atol=base.ATOL,allowed_rtol=base.RTOL,
        environment=base.common_record()["environment"],
        provenance=dict(script_sha256=base.digest(__file__),
                        reused_baseline_script_sha256=base.digest(base.__file__)),
        candidates=[])
    record["protocol"]["local_work_size"]="[tile,tile]; all other timing/protocol settings unchanged"
    record["protocol"]["candidate_order"]="rotate legal candidate order by one per round"
    # Save the hypothesis, limits and decision gates before measuring any candidate.
    base.save_record(Path(str(prefix)+"_plan.json"),record)
    eligible=[]; results={}
    for tile in CANDIDATES:
        reasons=tile_legality(tile,discovery["device_capabilities"])
        if reasons:
            results[tile]=dict(tile=tile,result="UNSUPPORTED",decision="REJECT",
                               reason="; ".join(reasons),work_items=tile*tile,
                               theoretical_local_memory_bytes=2*tile*tile*4,runs=[])
        else:
            eligible.append(tile)
            results[tile]=[]
    blocked=set()
    for round_index in range(base.RUNS):
        order=eligible[round_index%len(eligible):]+eligible[:round_index%len(eligible)] if eligible else []
        for tile in order:
            if tile in blocked:
                continue
            print(f"Tile {tile}x{tile}, independent run {round_index+1}/{base.RUNS}",flush=True)
            proc=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),
                                 "--single-run",str(tile)],capture_output=True,text=True)
            try:
                run=json.loads(proc.stdout)
            except (json.JSONDecodeError,ValueError):
                run=dict(tile=tile,result="FAILED",reason=proc.stderr or proc.stdout or "No child result",
                         process_id=None,samples=[])
            if proc.returncode and run.get("result")=="PASS":
                run.update(result="FAILED",reason="Nonzero child exit code")
            run["round"]=round_index+1
            base.save_record(Path(str(prefix)+f"_tile_{tile}_run_{round_index+1:03d}.json"),run)
            results[tile].append(run)
            if run["result"]!="PASS":
                blocked.add(tile)
    for tile in CANDIDATES:
        candidate=results[tile] if isinstance(results[tile],dict) else summarize(tile,results[tile],locked)
        record["candidates"].append(candidate)
        base.save_record(Path(str(prefix)+f"_tile_{tile}_summary.json"),candidate)
    correct=[c for c in record["candidates"] if c["result"]=="PASS"]
    adopted=[c for c in correct if c["decision"]=="ADOPT"]
    best=min(correct,key=lambda c:c["aggregate_timing_ms"]["kernel_ms"]["median"]) if correct else None
    recommended=min(adopted,key=lambda c:c["aggregate_timing_ms"]["kernel_ms"]["median"]) if adopted else None
    record.update(best_configuration=best["tile"] if best else None,
                  recommended_tile=recommended["tile"] if recommended else None,
                  decision="ADOPT" if recommended else "REJECT")
    record["explanation"]=(
        f"Tile {recommended['tile']} passes correctness and all predeclared gates; "
        f"{recommended['improvement_percent']:.4f}% lower kernel median, "
        f"{recommended['speedup_vs_baseline']:.4f}x speedup versus locked Experiment 005."
        if recommended else "No candidate met all correctness, protocol and meaningful-improvement gates.")
    verify_locked()
    base.save_record(Path(str(prefix)+"_summary.json"),record)
    with Path(str(prefix)+"_report.md").open("x",encoding="utf-8") as f:
        f.write(report(record))
    print(record["decision"],record["explanation"])
    print("Summary:",str(prefix)+"_summary.json")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
