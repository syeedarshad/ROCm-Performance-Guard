# ROCm Performance Guard ? Phase 4: Production Benchmarks

This directory contains benchmark specifications, experimental runs, and persisted results for the **ROCm Performance Guard** suite.

---

## 1. Overview & Workload: GEMM (General Matrix Multiplication)

The primary benchmark workload for Phase 4 is **General Matrix Multiplication (GEMM)**:

$$C = A \times B$$

Where:
- Matrix $A$ has dimensions $M \times K$
- Matrix $B$ has dimensions $K \times N$
- Result Matrix $C$ has dimensions $M \times N$

### Deterministic Input Generation
To ensure rigorous reproducibility across runs and environments, matrices $A$ and $B$ are generated deterministically using bounded arithmetic:
- $A_{i, k} = \frac{(17i + 31k + 1) \pmod{1000}}{1000.0}$
- $B_{k, j} = \frac{(23k + 41j + 2) \pmod{1000}}{1000.0}$

Random number generators with uncontrolled seeds are strictly forbidden.

---

## 2. Benchmark Parameters

| Parameter | Default | Description |
|---|---|---|
| **M** | `512` | Number of rows in matrix $A$ and $C$ |
| **N** | `512` | Number of columns in matrix $B$ and $C$ |
| **K** | `512` | Inner dimension (columns of $A$, rows of $B$) |
| **dtype** | `fp32` | 32-bit floating point precision |
| **Warmup Runs** | `5` | Untimed initial iterations to eliminate cold-start effects |
| **Measurement Runs** | `30` | High-resolution timed measurement iterations |
| **Backend** | `cpu` | Target execution backend (`cpu`, `rocm`) |

---

## 3. Timing Methodology & Metric Calculations

- **Timer:** High-resolution monotonic timer via `time.perf_counter_ns()`.
- **Warmup Discard:** Warmup runs are untimed and discarded from statistical metrics.
- **Latency Metrics:**
  - `min`, `max`, `mean`, `median`
  - `stddev` (sample standard deviation)
  - `p95`, `p99` percentiles calculated using the standard NIST linear interpolation method.
- **Operation Count:**
  $$\text{FLOPs} = 2 \times M \times N \times K$$
  For $M=512, N=512, K=512$, each run executes **268,435,456 FLOPs**.
- **Throughput (GFLOPS):**
  Calculated using the **median latency** to ensure robustness against system noise:
  $$\text{GFLOPS} = \frac{2 \times M \times N \times K}{\text{median\_latency\_seconds} \times 10^9}$$

---

## 4. Backend Architecture

The benchmark engine implements a clean backend abstraction (`BaseGemmBackend`):
1. **CPU Backend (`cpu`):**
   - Pure Python standard library implementation.
   - Cache-friendly loop ordering with matrix transposition ($B^T$).
   - Parallelized chunk execution across CPU cores.
   - Works deterministically without requiring external packages (NumPy, PyTorch, etc.).
2. **ROCm/HIP Backend (`rocm`):**
   - Automatically probes for the AMD ROCm compute runtime, `rocminfo`, `hipcc`, and `/dev/kfd`.
   - When ROCm is unavailable, gracefully returns a structured `UNAVAILABLE` capability status instead of crashing.
   - Prepares the system for future hardware GPU acceleration when supported devices are attached.

---

## 5. Result Persistence

All benchmark results are automatically saved as JSON in:
```
experiments/results/
```
Filename format:
```
gemm_{backend}_{M}x{N}x{K}_{dtype}_{YYYYMMDD_HHMMSS}.json
```
Existing files are preserved; new runs append unique sequence numbers if timestamps coincide.

---

## 6. Execution Instructions

### Run the Default GEMM Benchmark
```bash
python -m src.benchmark
```

### Configurable Matrix Sizes & Iterations
```bash
python -m src.benchmark --m 512 --n 512 --k 512 --warmup 5 --iterations 30 --backend cpu
```

### Display CLI Help & All Options
```bash
python -m src.benchmark --help
```
