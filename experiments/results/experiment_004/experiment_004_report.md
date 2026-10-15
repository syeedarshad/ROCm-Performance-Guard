# Experiment 004 — RTX 2050 GPU correctness validation

## Purpose and scope

Establish a numerical correctness gate before any GPU performance optimization.
Compare the CPU reference matrix product with the GPU matrix product for
M = N = K = 512 using FP32. The observed GPU is **NVIDIA GeForce RTX 2050**.
This is NVIDIA correctness evidence, not an AMD/ROCm result or a performance benchmark.

## Supplied observation

| Measurement | Value |
| --- | ---: |
| Max absolute error | 5.3405761719e-05 |
| Mean absolute error | 4.8291999519e-06 |
| Max relative error | 2.0715631547e-02 |
| Mean relative error | 1.8085577267e-06 |
| Allowed absolute tolerance | 0.001 |
| Allowed relative tolerance | 0.001 |
| Final reported decision | **PASS** |

These values and the PASS decision were supplied by the user. The original
CPU reference implementation, input-generation method, relative-error denominator,
tensor device, software versions, and execution timestamp were not supplied.
The JSON timestamp records artifact creation, not the original GPU run.
The recording host's environment must not be mistaken for the RTX 2050 run environment.

The reported maximum absolute error is below the absolute tolerance. A maximum
relative error above rtol does not by itself imply failure: the new gate checks
each element using abs(GPU - CPU) <= atol + rtol * abs(CPU). Raw output matrices
were not supplied, so the original comparison has not been independently repeated.

## Reproducible correctness gate

Run from the project root in the RTX 2050 environment with an existing
NVIDIA CUDA-enabled PyTorch installation:

```bash
python -B experiments/gpu_correctness.py
```

To preserve a new run, choose a new filename:

```bash
python -B experiments/gpu_correctness.py --output experiments/results/experiment_004/rtx2050_correctness_rerun_001.json
```

Existing output paths are refused. Exit codes: 0 = PASS, 1 = numerical FAIL,
2 = unavailable environment, runtime error, or output-save error.
For multiple GPUs, use --device-index to select the RTX 2050; other models are refused.

The script reuses src.benchmark.generate_deterministic_matrix with seed offsets
1 for A and 2 for B, converts both to torch.float32 on CPU, computes the CPU
reference using torch.mm, copies the same tensors to the selected GPU, and
computes torch.mm on that GPU. It does not transpose B, change dimensions,
change worker counts, or alter the existing benchmark. TF32 is disabled for
the check and the process setting is restored afterward. There is no timing.

The script reports the actual GPU name, tensor device, tensor dtype in its JSON,
and all four error metrics. Relative error is defined as
abs(GPU - CPU) / max(abs(CPU), 1e-12), including zero-reference values.
Nonfinite results fail. The FP32 CPU result is a numerical reference, not an
exact-arithmetic oracle.

This defines a repeatable check with the project's known inputs; the original
observation's inputs are unknown, so identical error values are not promised.

## Local verification and limitations

The current project virtual environment does not contain PyTorch. A local
execution attempt is saved separately in local_validation_attempt.json and
must not be interpreted as a fresh GPU PASS. No packages or drivers were installed.
Focused standard-library tests exercise metrics, tolerance decisions, nonfinite
values, invalid inputs, and overwrite protection without requiring a GPU.

A passing correctness gate establishes agreement for this workload and tolerance.
It is required before optimizing, but establishes neither speedup nor correctness
for every possible matrix. No next optimization experiment has been started.
The locked CPU baseline and existing experiment evidence are preserved.


## Verification results for this addition

- Focused correctness logic: 9 tests passed.
- Full suite: 269 tests passed (260 existing + 9 new); no failures.
- Both Experiment 004 JSON artifacts parsed successfully; Markdown report exists.
- All 36 pre-existing project files matched their pre-change SHA256 checksums.
- Locked baseline SHA256: ded6b85ff3ec29349018c25ae7e0d7aedcb3ea719380dc5dec57d1f31d2376ad.
- Actual RTX 2050 re-execution remains unverified here: the script recorded UNAVAILABLE / NOT_RUN because PyTorch is absent.
