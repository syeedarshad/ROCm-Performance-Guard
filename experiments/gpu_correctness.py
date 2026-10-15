"""Experiment 004: NVIDIA RTX 2050 FP32 GEMM correctness gate (no timing)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark import generate_deterministic_matrix

M = N = K = 512
ATOL = RTOL = 0.001
EXPECTED_GPU = "NVIDIA GeForce RTX 2050"
RELATIVE_ERROR_FLOOR = 1e-12


def compare_results(reference, actual, atol=ATOL, rtol=RTOL):
    """Compare flat outputs using abs(actual-ref) <= atol + rtol*abs(ref).

    Relative-error diagnostics use max(abs(ref), 1e-12) as denominator.
    Nonfinite values always fail; their metrics are null for strict JSON.
    """
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    reference, actual = list(reference), list(actual)
    if not reference or len(reference) != len(actual):
        raise ValueError("Outputs must be nonempty and have identical lengths")
    finite = all(math.isfinite(x) for x in reference + actual)
    if not finite:
        return dict(result="FAIL", finite=False, failed_elements=len(reference),
                    max_absolute_error=None, mean_absolute_error=None,
                    max_relative_error=None, mean_relative_error=None)
    absolute = [abs(a - r) for r, a in zip(reference, actual)]
    relative = [e / max(abs(r), RELATIVE_ERROR_FLOOR)
                for r, e in zip(reference, absolute)]
    failed = sum(e > atol + rtol * abs(r) for r, e in zip(reference, absolute))
    return dict(result="PASS" if failed == 0 else "FAIL", finite=True,
                failed_elements=failed, max_absolute_error=max(absolute),
                mean_absolute_error=math.fsum(absolute) / len(absolute),
                max_relative_error=max(relative),
                mean_relative_error=math.fsum(relative) / len(relative))


def save_record(path, record):
    """Exclusive creation: existing experiment evidence is never overwritten."""
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def run_check(device_index=0):
    record = dict(
        schema_version="1.0", experiment_id="experiment-004-gpu-correctness",
        experiment_name="RTX 2050 GPU correctness validation",
        status="UNAVAILABLE", result="NOT_RUN", gpu_name=None,
        expected_gpu_name=EXPECTED_GPU, workload="GEMM: C = A @ B",
        M=M, N=N, K=K, precision="FP32", allowed_atol=ATOL, allowed_rtol=RTOL,
        max_absolute_error=None, mean_absolute_error=None,
        max_relative_error=None, mean_relative_error=None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        timestamp_kind="execution_attempt", provenance="script_execution",
        tensor_device=None, reference_device="cpu",
        input_generation="src.benchmark.generate_deterministic_matrix; A offset=1, B offset=2",
        relative_error_definition="abs(actual-reference)/max(abs(reference), 1e-12)",
        environment=dict(python_version=platform.python_version(),
                         platform=platform.platform()),
    )
    try:
        import torch
    except ImportError as exc:
        record["reason"] = "PyTorch unavailable: " + str(exc)
        return record
    record["environment"].update(
        pytorch_version=torch.__version__, cuda_version=torch.version.cuda,
        hip_version=torch.version.hip)
    if torch.version.hip or not torch.version.cuda:
        record["reason"] = "An existing NVIDIA CUDA build of PyTorch is required."
        return record
    if not torch.cuda.is_available():
        record["reason"] = "PyTorch cannot access an NVIDIA CUDA GPU."
        return record
    if not 0 <= device_index < torch.cuda.device_count():
        record["reason"] = "Requested GPU index is unavailable."
        return record
    device = torch.device(f"cuda:{device_index}")
    record["gpu_name"] = torch.cuda.get_device_name(device)
    record["tensor_device"] = str(device)
    if record["gpu_name"] != EXPECTED_GPU:
        record["reason"] = "Selected GPU is not the required NVIDIA GeForce RTX 2050."
        return record
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # Enforce FP32 for this correctness gate, restoring the process setting afterward.
        torch.backends.cuda.matmul.allow_tf32 = False
        record["environment"]["allow_tf32"] = False
        with torch.no_grad():
            a = torch.tensor(generate_deterministic_matrix(M, K, 1),
                             dtype=torch.float32, device="cpu")
            b = torch.tensor(generate_deterministic_matrix(K, N, 2),
                             dtype=torch.float32, device="cpu")
            reference = torch.mm(a, b)
            gpu_a, gpu_b = a.to(device), b.to(device)
            actual = torch.mm(gpu_a, gpu_b)
            torch.cuda.synchronize(device)
            record["tensor_device"] = str(actual.device)
            record["tensor_dtype"] = str(actual.dtype)
            if actual.shape != reference.shape or actual.dtype != torch.float32:
                raise RuntimeError("GPU output shape or precision differs from reference")
            record.update(compare_results(reference.flatten().tolist(),
                                          actual.cpu().flatten().tolist()))
        record["status"] = "COMPLETED"
        record["timestamp_kind"] = "execution_completed"
    except Exception as exc:
        record.update(status="ERROR", result="NOT_RUN",
                      reason=f"{type(exc).__name__}: {exc}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        help="Optional NEW JSON path; refuses to overwrite existing files")
    args = parser.parse_args(argv)
    record = run_check(args.device_index)
    print("=== GPU CORRECTNESS TEST ===")
    print("GPU:", record["gpu_name"] or "unavailable")
    print("Tensor device:", record["tensor_device"] or "unavailable")
    print("CPU reference device: cpu")
    print("Workload: 512 x 512 x 512\nPrecision: FP32")
    for key in ("max_absolute_error", "mean_absolute_error",
                "max_relative_error", "mean_relative_error", "allowed_atol", "allowed_rtol"):
        value = record[key]
        print(f"{key}: {value:.10e}" if value is not None else f"{key}: unavailable")
    print("STATUS:", record["status"])
    print("RESULT:", record["result"])
    if "reason" in record:
        print("Reason:", record["reason"])
    if args.output:
        try:
            save_record(args.output, record)
        except (OSError, ValueError) as exc:
            print(f"Cannot save result: {exc}", file=sys.stderr)
            return 2
    return 0 if record["result"] == "PASS" else 1 if record["result"] == "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
