"""Experiment 005: native Windows AMD OpenCL FP32 GEMM baseline, no optimization."""
from __future__ import annotations

import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.benchmark import calculate_percentile, generate_deterministic_matrix
from experiments.gpu_correctness import compare_results, save_record

M = N = K = 512
WARMUPS, ITERATIONS, RUNS = 10, 50, 3
ATOL = RTOL = 0.001
FLOPS = 2 * M * N * K
BASELINE = ROOT / "experiments/results/gemm_cpu_512x512x512_fp32_20261003_043010.json"
BASELINE_SHA256 = "ded6b85ff3ec29349018c25ae7e0d7aedcb3ea719380dc5dec57d1f31d2376ad"
P, U, I, Q, S = ct.c_void_p, ct.c_uint, ct.c_int, ct.c_ulonglong, ct.c_size_t
QUEUE_PROFILING_ENABLE = 2

KERNEL = """
#pragma OPENCL FP_CONTRACT OFF
__kernel void gemm(__global const float *A,
                   __global const float *B,
                   __global float *C) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    float value = 0.0f;
    for (int k = 0; k < 512; ++k)
        value += A[row * 512 + k] * B[k * 512 + col];
    C[row * 512 + col] = value;
}
"""


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_locked_baseline():
    if digest(BASELINE) != BASELINE_SHA256:
        raise RuntimeError("Locked CPU baseline checksum does not match")


def deterministic_inputs():
    a = (ct.c_float * (M * K))(*[
        x for row in generate_deterministic_matrix(M, K, 1) for x in row])
    b = (ct.c_float * (K * N))(*[
        x for row in generate_deterministic_matrix(K, N, 2) for x in row])
    return a, b


def cpu_reference(a, b):
    # Same rounded FP32 input values; accurate CPU sum, then FP32 output.
    aa, bb = list(a), list(b)
    return [ct.c_float(math.fsum(
        aa[row * K + k] * bb[k * N + col] for k in range(K))).value
        for row in range(M) for col in range(N)]


def event_latency_ms(start, end):
    if start < 0 or end <= start:
        raise ValueError("Invalid or zero OpenCL event duration")
    return (end - start) / 1e6


def timing_summary(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timing samples must be finite and positive")
    return dict(count=len(values), median=statistics.median(values),
                mean=statistics.mean(values), p95=calculate_percentile(sorted(values), 95),
                min=min(values), max=max(values))


def gflops(latency_ms):
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("Latency must be finite and positive")
    return FLOPS / (latency_ms / 1000) / 1e9


def check(rc, operation):
    if rc:
        raise RuntimeError(f"{operation}: OpenCL error {rc}")


class WindowsOpenCL:
    """Small ctypes backend. Only AMD's native gfx90c GPU is accepted."""
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("This experiment requires native Windows")
        self.lib = ct.WinDLL(str(Path(os.environ["SystemRoot"]) / "System32/OpenCL.dll"))
        self.resources = []
        self.functions = {}
        try:
            self.device, self.metadata = self.find_device()
            err = I()
            dev = P(self.device)
            self.context = self.call("clCreateContext", [P, U, P, P, P, ct.POINTER(I)], P)(
                None, 1, ct.byref(dev), None, None, ct.byref(err))
            check(err.value, "create context")
            self.resources.append(("clReleaseContext", self.context))
            self.queue = self.call("clCreateCommandQueue", [P, P, Q, ct.POINTER(I)], P)(
                self.context, dev, QUEUE_PROFILING_ENABLE, ct.byref(err))
            check(err.value, "create profiling queue")
            self.resources.append(("clReleaseCommandQueue", self.queue))
            source = ct.c_char_p(KERNEL.encode())
            program = self.call("clCreateProgramWithSource", [P, U, P, P, ct.POINTER(I)], P)(
                self.context, 1, ct.byref(source), None, ct.byref(err))
            check(err.value, "create program")
            self.resources.append(("clReleaseProgram", program))
            rc = self.call("clBuildProgram", [P, U, P, ct.c_char_p, P, P])(
                program, 1, ct.byref(dev), b"-cl-std=CL1.2", None, None)
            if rc:
                log = ct.create_string_buffer(65536)
                self.call("clGetProgramBuildInfo", [P, P, U, S, P, P])(
                    program, dev, 0x1183, len(log), log, None)
                raise RuntimeError("Build failed: " + log.value.decode(errors="replace"))
            self.kernel = self.call("clCreateKernel", [P, ct.c_char_p, ct.POINTER(I)], P)(
                program, b"gemm", ct.byref(err))
            check(err.value, "create kernel")
            self.resources.append(("clReleaseKernel", self.kernel))
            self.buffers = []
            for index, flags in enumerate((4, 4, 2)):  # read-only A/B; write-only C
                buf = self.call("clCreateBuffer", [P, Q, S, P, ct.POINTER(I)], P)(
                    self.context, flags, M * N * 4, None, ct.byref(err))
                check(err.value, "create buffer")
                self.resources.append(("clReleaseMemObject", buf))
                self.buffers.append(buf)
                handle = P(buf)
                check(self.call("clSetKernelArg", [P, U, S, P])(
                    self.kernel, index, ct.sizeof(handle), ct.byref(handle)), "set argument")
            self.output = (ct.c_float * (M * N))()
            # Bind dispatch functions before measuring any host interval.
            self.write = self.call("clEnqueueWriteBuffer", [P,P,U,S,S,P,U,P,P])
            self.read = self.call("clEnqueueReadBuffer", [P,P,U,S,S,P,U,P,P])
            self.enqueue = self.call("clEnqueueNDRangeKernel", [P,P,U,P,P,P,U,P,P])
            self.wait = self.call("clWaitForEvents", [U,P])
            self.profile = self.call("clGetEventProfilingInfo", [P,U,S,P,P])
            self.release_event = self.call("clReleaseEvent", [P])
            self.global_size = (S * 2)(M, N)
        except Exception:
            self.close()
            raise

    def call(self, name, args, ret=I):
        if name not in self.functions:
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, ret
            self.functions[name] = f
        return self.functions[name]

    def info_text(self, obj, key, platform_info=False):
        name = "clGetPlatformInfo" if platform_info else "clGetDeviceInfo"
        f = self.call(name, [P,U,S,P,P])
        length = S()
        check(f(obj, key, 0, None, ct.byref(length)), name)
        buf = ct.create_string_buffer(length.value)
        check(f(obj, key, len(buf), buf, None), name)
        return buf.value.decode(errors="replace")

    def scalar_info(self, device, key, scalar_type):
        value = scalar_type()
        check(self.call("clGetDeviceInfo", [P,U,S,P,P])(
            device, key, ct.sizeof(value), ct.byref(value), None), "device info")
        return value.value

    def find_device(self):
        f = self.call("clGetPlatformIDs", [U,P,P])
        count = U()
        check(f(0, None, ct.byref(count)), "platform count")
        platforms = (P * count.value)()
        check(f(count, platforms, None), "platforms")
        found = []
        for plat in platforms:
            n = U()
            enum = self.call("clGetDeviceIDs", [P,Q,U,P,P])
            rc = enum(plat, 4, 0, None, ct.byref(n))  # GPU only
            if rc == -1:
                continue
            check(rc, "GPU enumeration")
            devices = (P * n.value)()
            check(enum(plat, 4, n, devices, None), "GPU devices")
            for device in devices:
                vendor = self.info_text(device, 0x102C)
                name = self.info_text(device, 0x102B)
                if ("Advanced Micro Devices" not in vendor or name != "gfx90c"):
                    continue
                found.append((device, dict(
                    gpu_name="AMD Radeon integrated graphics (Ryzen 7 7730U)",
                    gpu_vendor=vendor, opencl_device=name,
                    platform=self.info_text(plat, 0x0902, True),
                    platform_vendor=self.info_text(plat, 0x0903, True),
                    platform_version=self.info_text(plat, 0x0901, True),
                    opencl_version=self.info_text(device, 0x102F),
                    opencl_c_version=self.info_text(device, 0x103D),
                    driver_version=self.info_text(device, 0x102D),
                    global_memory_bytes=self.scalar_info(device, 0x101F, Q),
                    global_memory_note="Includes shared memory; not dedicated VRAM",
                    profiling_timer_resolution_ns=self.scalar_info(device, 0x1025, S),
                    device_type="CL_DEVICE_TYPE_GPU")))
        if len(found) != 1:
            raise RuntimeError(f"Expected exactly one native AMD gfx90c GPU, found {len(found)}")
        return found[0]

    def sample(self, a, b):
        events = [P() for _ in range(4)]
        try:
            started = time.perf_counter_ns()
            for i, host in enumerate((a, b)):
                check(self.write(self.queue, self.buffers[i], 0, 0, ct.sizeof(host),
                                 host, 0, None, ct.byref(events[i])), "enqueue H2D")
            check(self.enqueue(self.queue, self.kernel, 2, None, self.global_size,
                               None, 0, None, ct.byref(events[2])), "enqueue GEMM")
            check(self.read(self.queue, self.buffers[2], 0, 0, ct.sizeof(self.output),
                            self.output, 0, None, ct.byref(events[3])), "enqueue D2H")
            # In-order queue: completing the read also completes both writes and GEMM.
            check(self.wait(1, ct.byref(events[3])), "wait for read completion")
            finished = time.perf_counter_ns()
            stamps = {}
            for name, event in zip(("write_a", "write_b", "kernel", "read_c"), events):
                start, end = Q(), Q()
                check(self.profile(event, 0x1282, 8, ct.byref(start), None), "event START")
                check(self.profile(event, 0x1283, 8, ct.byref(end), None), "event END")
                stamps[name] = dict(start_ns=start.value, end_ns=end.value,
                                   latency_ms=event_latency_ms(start.value, end.value))
            return dict(kernel_ms=stamps["kernel"]["latency_ms"],
                        h2d_ms=stamps["write_a"]["latency_ms"] + stamps["write_b"]["latency_ms"],
                        d2h_ms=stamps["read_c"]["latency_ms"],
                        end_to_end_ms=(finished - started) / 1e6,
                        events=stamps)
        finally:
            for event in events:
                if event.value:
                    self.release_event(event)

    def close(self):
        if getattr(self, "queue", None):
            self.call("clFinish", [P])(self.queue)
        for name, obj in reversed(self.resources):
            self.call(name, [P])(obj)
        self.resources.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def correctness(reference, output):
    record = compare_results(reference, output, ATOL, RTOL)
    record.update(allowed_atol=ATOL, allowed_rtol=RTOL,
                  criterion="all finite and abs(GPU-CPU) <= atol + rtol*abs(CPU)",
                  relative_error_definition="abs(GPU-CPU)/max(abs(CPU), 1e-12)")
    return record


def one_run(correctness_only=False):
    verify_locked_baseline()
    a, b = deterministic_inputs()
    reference = cpu_reference(a, b)
    record = dict(timestamp=timestamp(), process_id=os.getpid(),
                  input_sha256=dict(A=hashlib.sha256(bytes(a)).hexdigest(),
                                    B=hashlib.sha256(bytes(b)).hexdigest()),
                  reference="CPU math.fsum of FP32 inputs, rounded to FP32 output")
    with WindowsOpenCL() as backend:
        record["device"] = backend.metadata
        record["initial_timing"] = backend.sample(a, b)
        record["correctness"] = correctness(reference, backend.output)
        record["result"] = record["correctness"]["result"]
        record["samples"] = []
        if record["result"] == "PASS" and not correctness_only:
            for _ in range(WARMUPS):
                backend.sample(a, b)
                if correctness(reference, backend.output)["result"] != "PASS":
                    raise RuntimeError("Warmup correctness failed; baseline rejected")
            for iteration in range(ITERATIONS):
                sample = backend.sample(a, b)
                sample["iteration"] = iteration + 1
                sample["correctness"] = correctness(reference, backend.output)
                record["samples"].append(sample)
                if sample["correctness"]["result"] != "PASS":
                    record["result"] = "FAIL"
                    break
            record["timing_ms"] = {
                key: timing_summary([s[key] for s in record["samples"]])
                for key in ("kernel_ms", "h2d_ms", "d2h_ms", "end_to_end_ms")}
            record["gflops_from_median_kernel"] = gflops(record["timing_ms"]["kernel_ms"]["median"])
    verify_locked_baseline()
    return record


def common_record():
    return dict(schema_version="1.0", experiment_id="experiment-005-opencl-gemm-baseline",
                experiment_name="Acer AMD Radeon native Windows OpenCL GEMM baseline",
                timestamp=timestamp(), workload=dict(M=M,N=N,K=K,precision="FP32",
                operation="C = A x B", flops=FLOPS),
                protocol=dict(warmup_iterations=WARMUPS, measured_iterations=ITERATIONS,
                              independent_runs=RUNS, independence="fresh child process and OpenCL context per run",
                              correctness_preflight_iterations=1,
                              primary_metric="median OpenCL kernel event latency",
                              queue_profiling_enabled=True, in_order_queue=True,
                              global_work_size=[M,N], local_work_size="runtime selected",
                              build_options="-cl-std=CL1.2; FP_CONTRACT OFF; no fast-math",
                              end_to_end_scope="host before enqueue A/B writes through completion of C read",
                              exclusions="input/reference generation, context/build/allocation, validation, reporting",
                              transfer_scope="H2D=sum of A/B write events; D2H=C read event"),
                environment=dict(os=platform.platform(), python=sys.version,
                                 execution="native Windows; AMD OpenCL; no ROCm or CUDA"),
                provenance=dict(script_sha256=digest(__file__),
                                kernel_sha256=hashlib.sha256(KERNEL.encode()).hexdigest(),
                                input_generator_sha256=digest(ROOT / "src/benchmark.py")),
                locked_cpu_baseline_sha256=BASELINE_SHA256)


def markdown(record):
    lines = ["# Experiment 005 — AMD OpenCL GEMM baseline", "",
             f"Result: **{record['result']}**", "",
             "Native Windows, Acer Ryzen 7 7730U integrated AMD Radeon (gfx90c).",
             "512 × 512 × 512, FP32, C = A × B. No optimization, ROCm, or CUDA.",
             "", "CPU reference: math.fsum over the same FP32 inputs, rounded to FP32 output.",
             "Tolerance: atol=0.001, rtol=0.001; every element must be finite and satisfy",
             "abs(GPU-CPU) <= atol + rtol*abs(CPU). Relative error denominator is max(abs(CPU), 1e-12).",
             "", "Each fresh process/context uses 1 correctness preflight, 10 warmups and 50 measured iterations.",
             "Every measured output is checked. JSON preserves raw samples and device event timestamps.",
             "", "| Run | Kernel median ms | Mean ms | P95 ms | Min ms | Max ms | GFLOPS |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for i, run in enumerate(record["runs"], 1):
        s = run["timing_ms"]["kernel_ms"]
        lines.append(f"| {i} | {s['median']:.6f} | {s['mean']:.6f} | {s['p95']:.6f} | {s['min']:.6f} | {s['max']:.6f} | {run['gflops_from_median_kernel']:.6f} |")
    lines += ["", "## Aggregate timings (all 150 measured samples)", "",
              "| Component | Median ms | Mean ms | P95 ms | Min ms | Max ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, s in record["aggregate_timing_ms"].items():
        lines.append(f"| {name} | {s['median']:.6f} | {s['mean']:.6f} | {s['p95']:.6f} | {s['min']:.6f} | {s['max']:.6f} |")
    lines += ["", f"Primary GFLOPS from pooled median kernel latency: {record['gflops_from_median_kernel']:.6f}.",
              "GFLOPS = (2*M*N*K) / latency_seconds / 1e9.",
              "", "## Correctness: worst observed diagnostic across measured outputs", ""]
    for name, value in record["worst_correctness_metrics"].items():
        lines.append(f"- {name}: {value:.10e}")
    lines += ["", "## Device and runtime", "", "```json",
              json.dumps(record["runs"][0]["device"], indent=2), "```", "",
              "## Timing interpretation", "",
              "Kernel and transfer durations come from completed OpenCL event START/END timestamps.",
              "The profiling-enabled queue is in order. The host waits for the final read event.",
              "End-to-end latency uses a synchronized host clock interval; it includes enqueue and wait overhead.",
              "It excludes initialization, compilation, allocation, CPU reference, correctness comparison and reporting.",
              "H2D/D2H are OpenCL write/read commands on an integrated GPU with shared memory; these are not PCIe bandwidth measurements.",
              "Component medians need not add to the end-to-end median.",
              "Fresh processes provide separate runs, not control over thermals, power policy or background activity.",
              "This unoptimized, runtime-selected workgroup kernel is a starting point, not a peak-throughput claim.",
              "No direct speedup claim is made against the historical CPU baseline with different arithmetic and timing scope.",
              "", "## Reproduction", "", "```powershell",
              "python -B experiments/opencl_gemm_baseline.py --correctness-only",
              "python -B experiments/opencl_gemm_baseline.py", "```", "",
              "Uses the existing Windows OpenCL runtime and Python standard library; installs nothing.",
              "New timestamped filenames are exclusively created. Existing evidence cannot be overwritten.",
              f"Locked CPU baseline SHA256: {BASELINE_SHA256}", "",
              "Profiling reference: https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetEventProfilingInfo.html",
              ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.single_run:
        record = one_run(args.correctness_only)
        print(json.dumps(record, allow_nan=False))
        return 0 if record["result"] == "PASS" else 1
    verify_locked_baseline()
    output = ROOT / "experiments/results/experiment_005"
    token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix = output / f"experiment_005_{token}"
    record = common_record()
    if args.correctness_only:
        run = one_run(True)
        record.update(mode="correctness_only", result=run["result"], run=run,
                      executed_protocol=dict(preflight_iterations=1, warmup_iterations=0,
                                             measured_iterations=0, independent_runs=1))
        path = Path(str(prefix) + "_correctness.json")
        save_record(path, record)
        print(json.dumps(run["correctness"], indent=2))
        print("Saved:", path)
        return 0 if run["result"] == "PASS" else 1
    runs = []
    for index in range(RUNS):
        print(f"Running independent OpenCL baseline {index+1}/{RUNS}...", flush=True)
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                               "--single-run"], capture_output=True, text=True)
        if proc.returncode not in (0, 1) or not proc.stdout.strip():
            raise RuntimeError(proc.stderr or "Child run failed without output")
        run = json.loads(proc.stdout)
        save_record(Path(str(prefix) + f"_run_{index+1:03d}.json"), run)
        runs.append(run)
        if proc.returncode or run["result"] != "PASS":
            raise RuntimeError("Correctness failed; run saved, baseline rejected")
    samples = [sample for run in runs for sample in run["samples"]]
    if len(samples) != RUNS * ITERATIONS:
        raise RuntimeError("Incomplete measurement protocol")
    if any(run["device"] != runs[0]["device"] or run["input_sha256"] != runs[0]["input_sha256"]
           for run in runs):
        raise RuntimeError("Device or inputs changed between runs")
    record.update(result="PASS", runs=runs,
                  aggregate_timing_ms={key: timing_summary([s[key] for s in samples])
                                       for key in ("kernel_ms","h2d_ms","d2h_ms","end_to_end_ms")},
                  worst_correctness_metrics={key: max(s["correctness"][key] for s in samples)
                                             for key in ("max_absolute_error","mean_absolute_error",
                                                         "max_relative_error","mean_relative_error")})
    record["gflops_from_median_kernel"] = gflops(record["aggregate_timing_ms"]["kernel_ms"]["median"])
    record["kernel_median_between_runs_cv_percent"] = (
        statistics.stdev(r["timing_ms"]["kernel_ms"]["median"] for r in runs) /
        statistics.mean(r["timing_ms"]["kernel_ms"]["median"] for r in runs) * 100)
    verify_locked_baseline()
    save_record(Path(str(prefix) + "_summary.json"), record)
    with Path(str(prefix) + "_report.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown(record))
    print("PASS. Summary:", str(prefix) + "_summary.json")
    print("Kernel median ms:", record["aggregate_timing_ms"]["kernel_ms"]["median"])
    print("GFLOPS:", record["gflops_from_median_kernel"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
