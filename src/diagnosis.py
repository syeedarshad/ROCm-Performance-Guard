"""Root-cause and evidence diagnosis engine for ROCm Performance Guard (V1).

Analyzes performance differences between baseline and current benchmark runs by
comparing hardware snapshots, software environments, and benchmark configurations.
Produces a transparent, conservative, evidence-based ranked list of potential
contributing factors without claiming unproven causality.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from src.benchmark import BenchmarkResult
except ImportError:
    BenchmarkResult = Any  # type: ignore[misc,assignment]

try:
    from src.regression import RegressionResult, RegressionStatus, compare_benchmarks
except ImportError:
    RegressionResult = Any  # type: ignore[misc,assignment]
    RegressionStatus = Any  # type: ignore[misc,assignment]
    compare_benchmarks = None  # type: ignore[assignment]


class DiagnosisStatus(str, Enum):
    """Status classification for performance diagnosis."""

    NO_REGRESSION_TO_DIAGNOSE = "NO_REGRESSION_TO_DIAGNOSE"
    NO_CHANGES_DETECTED = "NO_CHANGES_DETECTED"
    POTENTIAL_CAUSES_FOUND = "POTENTIAL_CAUSES_FOUND"
    CONFIGURATION_CHANGED = "CONFIGURATION_CHANGED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ChangePriority(str, Enum):
    """Investigation priority for observed differences."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ChangeRole(str, Enum):
    """Conservative role classification for observed differences."""

    POTENTIAL_CONTRIBUTOR = "POTENTIAL_CONTRIBUTOR"
    OBSERVED_CHANGE = "OBSERVED_CHANGE"


TARGET_AI_PACKAGES: Set[str] = {
    "torch",
    "torchvision",
    "transformers",
    "onnxruntime",
    "numpy",
    "scipy",
}

COMPUTE_ENV_VARS: Set[str] = {
    "ROCM_PATH",
    "HIP_PATH",
    "HSA_PATH",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "CUDA_VISIBLE_DEVICES",
    "MIOPEN_LOG_LEVEL",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM",
    "PYTORCH_ROCM_ARCH",
    "TORCH_CUDA_ARCH_LIST",
}

THREAD_ENV_VARS: Set[str] = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
}

DEFAULT_LIMITATIONS: List[str] = [
    "Observed changes indicate possible contributors but do not prove causality.",
    "Temporal correlation between configuration changes and performance differences does not prove direct causation.",
    "Unmeasured background system load, thermal throttling, or OS scheduling variance may also contribute.",
]


@dataclass
class DetectedChange:
    """A single detected difference between baseline and current runs."""

    category: str
    field: str
    baseline: Any
    current: Any
    priority: str
    investigation_priority_score: int
    role: str
    explanation: str
    reason: Optional[str] = None
    change_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.reason is None:
            self.reason = self.explanation
        elif not self.explanation:
            self.explanation = self.reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean, JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "category": self.category,
            "field": self.field,
            "baseline": self.baseline,
            "current": self.current,
            "priority": self.priority,
            "investigation_priority_score": self.investigation_priority_score,
            "role": self.role,
            "explanation": self.explanation,
            "reason": self.reason or self.explanation,
        }
        if self.change_type:
            d["change_type"] = self.change_type
        return d


@dataclass
class DiagnosisResult:
    """Comprehensive root-cause and evidence diagnosis outcome."""

    schema_version: str
    status: DiagnosisStatus
    regression_status: str
    summary: str
    changes: List[DetectedChange] = field(default_factory=list)
    limitations: List[str] = field(default_factory=lambda: list(DEFAULT_LIMITATIONS))

    def to_dict(self) -> Dict[str, Any]:
        """Convert diagnosis result to dictionary representation."""
        return {
            "schema_version": self.schema_version,
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "regression_status": self.regression_status,
            "summary": self.summary,
            "changes": [c.to_dict() for c in self.changes],
            "limitations": list(self.limitations),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert diagnosis result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def diff_dicts(
    base: Dict[str, Any],
    curr: Dict[str, Any],
    prefix: str = "",
    ignore_keys: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Recursively compare two dictionaries and return leaf differences.

    Safe for nested dictionaries and primitive JSON values.
    Ignores keys such as timestamps or fingerprints that do not constitute causes.
    """
    if ignore_keys is None:
        ignore_keys = {"timestamp", "created_at", "fingerprint", "time", "date"}

    differences: List[Dict[str, Any]] = []
    all_keys = sorted(set(base.keys()) | set(curr.keys()))

    for key in all_keys:
        if key in ignore_keys:
            continue
        field_path = f"{prefix}.{key}" if prefix else key

        if key not in base:
            differences.append({
                "field": field_path,
                "baseline": None,
                "current": curr[key],
                "change_type": "added",
            })
        elif key not in curr:
            differences.append({
                "field": field_path,
                "baseline": base[key],
                "current": None,
                "change_type": "removed",
            })
        else:
            val_base = base[key]
            val_curr = curr[key]

            if isinstance(val_base, dict) and isinstance(val_curr, dict):
                sub_diffs = diff_dicts(val_base, val_curr, prefix=field_path, ignore_keys=ignore_keys)
                differences.extend(sub_diffs)
            elif val_base != val_curr:
                differences.append({
                    "field": field_path,
                    "baseline": val_base,
                    "current": val_curr,
                    "change_type": "modified",
                })

    return differences


def _classify_version_change(v_base: Any, v_curr: Any) -> str:
    """Classify version change as upgraded, downgraded, added, removed, or modified."""
    if v_base is None and v_curr is not None:
        return "added"
    if v_base is not None and v_curr is None:
        return "removed"
    if not isinstance(v_base, str) or not isinstance(v_curr, str):
        return "modified"

    def parse_ver(v_str: str) -> List[Union[int, str]]:
        tokens = re.split(r"[-.+_]", v_str)
        parts: List[Union[int, str]] = []
        for t in tokens:
            if t.isdigit():
                parts.append(int(t))
            elif t:
                parts.append(t)
        return parts

    try:
        p_base = parse_ver(v_base)
        p_curr = parse_ver(v_curr)
        for b, c in zip(p_base, p_curr):
            if type(b) is type(c):
                if c > b:
                    return "upgraded"
                elif c < b:
                    return "downgraded"
        if len(p_curr) > len(p_base):
            return "upgraded"
        elif len(p_curr) < len(p_base):
            return "downgraded"
    except Exception:
        pass
    return "modified"


def _get_subdomain_rank(category: str, field_name: str) -> int:
    """Assign a deterministic sub-domain rank (lower integer indicates higher priority).

    Ordering hierarchy:
      0: Workload / benchmark configuration
      1: Compute hardware (GPU, ROCm suite version, HIP, KFD)
      2: Compute environment variables (ROCM_PATH, HIP_PATH, etc.)
      3: AI / ML framework packages (torch, transformers, etc.)
      4: CPU, threading, and Python runtime
      5: Other environment variables
      6: Other packages, build tools, OS, generic metadata
    """
    f_lower = field_name.lower()
    c_lower = category.lower()

    if c_lower in ("benchmark", "configuration"):
        return 0
    if ("gpu" in f_lower or "rocm" in f_lower or "hip" in f_lower or "kfd" in f_lower) and "environment_variables" not in f_lower:
        return 1
    if "environment_variables" in f_lower and any(var.lower() in f_lower for var in COMPUTE_ENV_VARS):
        return 2
    if any(pkg in f_lower for pkg in TARGET_AI_PACKAGES):
        return 3
    if "python" in f_lower or "cpu" in f_lower or "thread" in f_lower:
        return 4
    if "environment_variables" in f_lower or "env" in c_lower:
        return 5
    return 6


def _sort_changes(changes: List[DetectedChange]) -> List[DetectedChange]:
    """Sort detected changes deterministically by investigation priority.

    Ordering rules:
      1. Priority tier DESC (HIGH=3, MEDIUM=2, LOW=1)
      2. Sub-domain rank ASC (Benchmark -> GPU/ROCm -> Compute Env -> ML Frameworks -> CPU/Runtime -> Env -> Other)
      3. Field name ASC (lexicographical tie-breaker)
    """
    def key_fn(c: DetectedChange) -> Tuple[int, int, str]:
        p_score = c.investigation_priority_score
        sub_rank = _get_subdomain_rank(c.category, c.field)
        return (-p_score, sub_rank, c.field)

    return sorted(changes, key=key_fn)


def _to_dict_safe(obj: Any) -> Optional[Dict[str, Any]]:
    """Safely extract dictionary representation from object or dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return None


def _check_benchmark_configuration_changes(
    base_bm: Dict[str, Any],
    curr_bm: Dict[str, Any],
) -> List[DetectedChange]:
    """Detect differences in benchmark name, version, and execution configuration."""
    changes: List[DetectedChange] = []

    # Name
    b_name = base_bm.get("benchmark", {}).get("name")
    c_name = curr_bm.get("benchmark", {}).get("name")
    if b_name != c_name and (b_name is not None or c_name is not None):
        changes.append(DetectedChange(
            category="configuration",
            field="benchmark.name",
            baseline=b_name,
            current=c_name,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"Benchmark name differs between baseline ('{b_name}') and current ('{c_name}'). Workloads are distinct; causality cannot be attributed to environment.",
        ))

    # Version
    b_ver = base_bm.get("benchmark", {}).get("version")
    c_ver = curr_bm.get("benchmark", {}).get("version")
    if b_ver != c_ver and (b_ver is not None or c_ver is not None):
        ch_type = _classify_version_change(b_ver, c_ver)
        changes.append(DetectedChange(
            category="configuration",
            field="benchmark.version",
            baseline=b_ver,
            current=c_ver,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type=ch_type,
            explanation=f"Benchmark version changed from '{b_ver}' to '{c_ver}'. Workload implementation differences directly affect results; causality cannot be attributed to environment.",
        ))

    # Configuration parameters
    b_cfg = base_bm.get("configuration", {})
    c_cfg = curr_bm.get("configuration", {})
    cfg_diffs = diff_dicts(b_cfg, c_cfg, prefix="configuration", ignore_keys={"timestamp", "fingerprint"})
    for diff in cfg_diffs:
        changes.append(DetectedChange(
            category="configuration",
            field=diff["field"],
            baseline=diff["baseline"],
            current=diff["current"],
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type=diff.get("change_type", "modified"),
            explanation=f"Benchmark configuration parameter '{diff['field']}' changed from {diff['baseline']} to {diff['current']}. Execution settings differ; results may not be directly attributable to a software regression.",
        ))

    return changes


def _analyze_hardware_changes(
    base_hw: Dict[str, Any],
    curr_hw: Dict[str, Any],
) -> List[DetectedChange]:
    """Analyze hardware differences and assign heuristic priority."""
    changes: List[DetectedChange] = []

    # 1. GPU Identity and availability (HIGH)
    base_gpu = base_hw.get("gpu", {})
    curr_gpu = curr_hw.get("gpu", {})

    # Linux-visible GPU presence
    b_vis = base_gpu.get("linux_has_gpu")
    c_vis = curr_gpu.get("linux_has_gpu")
    if b_vis != c_vis and (b_vis is not None or c_vis is not None):
        changes.append(DetectedChange(
            category="hardware",
            field="gpu.linux_has_gpu",
            baseline=b_vis,
            current=c_vis,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"Linux GPU availability changed from {b_vis} to {c_vis}. Hardware compute path directly altered; evidence does not prove causality.",
        ))

    # Host GPU presence
    b_host_avail = base_gpu.get("wsl_has_host_gpu")
    c_host_avail = curr_gpu.get("wsl_has_host_gpu")
    if b_host_avail != c_host_avail and (b_host_avail is not None or c_host_avail is not None):
        changes.append(DetectedChange(
            category="hardware",
            field="gpu.wsl_has_host_gpu",
            baseline=b_host_avail,
            current=c_host_avail,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"Host physical GPU availability changed from {b_host_avail} to {c_host_avail}; evidence does not prove causality.",
        ))

    # Host GPU devices list / model
    b_devices = [d.get("name") for d in base_gpu.get("host_devices", []) if isinstance(d, dict)]
    c_devices = [d.get("name") for d in curr_gpu.get("host_devices", []) if isinstance(d, dict)]
    if b_devices != c_devices and (b_devices or c_devices):
        changes.append(DetectedChange(
            category="hardware",
            field="gpu.host_devices",
            baseline=b_devices,
            current=c_devices,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"Host GPU device identity changed from {b_devices} to {c_devices}. This directly affects compute hardware capabilities; evidence does not prove causality.",
        ))

    # AMD hardware presence
    b_amd = base_hw.get("amd", {}).get("is_amd_hardware")
    c_amd = curr_hw.get("amd", {}).get("is_amd_hardware")
    if b_amd != c_amd and (b_amd is not None or c_amd is not None):
        changes.append(DetectedChange(
            category="hardware",
            field="amd.is_amd_hardware",
            baseline=b_amd,
            current=c_amd,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"AMD hardware presence indicator changed from {b_amd} to {c_amd}; evidence does not prove causality.",
        ))

    # ROCm availability in hardware
    b_rocm_avail = base_hw.get("rocm", {}).get("is_available")
    c_rocm_avail = curr_hw.get("rocm", {}).get("is_available")
    if b_rocm_avail != c_rocm_avail and (b_rocm_avail is not None or c_rocm_avail is not None):
        changes.append(DetectedChange(
            category="hardware",
            field="rocm.is_available",
            baseline=b_rocm_avail,
            current=c_rocm_avail,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"ROCm hardware tooling availability changed from {b_rocm_avail} to {c_rocm_avail}; evidence does not prove causality.",
        ))

    # ROCm driver / KFD presence
    b_kfd = base_hw.get("rocm", {}).get("kfd_available")
    c_kfd = curr_hw.get("rocm", {}).get("kfd_available")
    if b_kfd != c_kfd and (b_kfd is not None or c_kfd is not None):
        changes.append(DetectedChange(
            category="hardware",
            field="rocm.kfd_available",
            baseline=b_kfd,
            current=c_kfd,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"ROCm KFD kernel driver interface availability changed from {b_kfd} to {c_kfd}; evidence does not prove causality.",
        ))

    # ROCm version in hardware
    b_rocm_ver = base_hw.get("rocm", {}).get("version")
    c_rocm_ver = curr_hw.get("rocm", {}).get("version")
    if b_rocm_ver != c_rocm_ver and (b_rocm_ver is not None or c_rocm_ver is not None):
        ch_type = _classify_version_change(b_rocm_ver, c_rocm_ver)
        changes.append(DetectedChange(
            category="hardware",
            field="rocm.version",
            baseline=b_rocm_ver,
            current=c_rocm_ver,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type=ch_type,
            explanation=f"ROCm runtime version in hardware snapshot changed from '{b_rocm_ver}' to '{c_rocm_ver}'; evidence does not prove causality.",
        ))

    # 2. CPU changes (MEDIUM)
    b_cpu = base_hw.get("cpu", {})
    c_cpu = curr_hw.get("cpu", {})
    b_model = b_cpu.get("model_name")
    c_model = c_cpu.get("model_name")
    if b_model != c_model and (b_model is not None or c_model is not None):
        changes.append(DetectedChange(
            category="hardware",
            field="cpu.model_name",
            baseline=b_model,
            current=c_model,
            priority=ChangePriority.MEDIUM.value,
            investigation_priority_score=2,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"CPU model changed from '{b_model}' to '{c_model}'; evidence does not prove causality.",
        ))

    b_cores = b_cpu.get("logical_cores")
    c_cores = c_cpu.get("logical_cores")
    if b_cores != c_cores and (b_cores is not None or c_cores is not None):
        changes.append(DetectedChange(
            category="hardware",
            field="cpu.logical_cores",
            baseline=b_cores,
            current=c_cores,
            priority=ChangePriority.MEDIUM.value,
            investigation_priority_score=2,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"Logical CPU core count changed from {b_cores} to {c_cores}; evidence does not prove causality.",
        ))

    # 3. Memory changes (MEDIUM)
    b_mem = base_hw.get("memory", {}).get("linux_total_mb")
    c_mem = curr_hw.get("memory", {}).get("linux_total_mb")
    if b_mem is not None and c_mem is not None and abs(b_mem - c_mem) > 256:
        changes.append(DetectedChange(
            category="hardware",
            field="memory.linux_total_mb",
            baseline=b_mem,
            current=c_mem,
            priority=ChangePriority.MEDIUM.value,
            investigation_priority_score=2,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"Total visible Linux memory changed from {b_mem} MB to {c_mem} MB; evidence does not prove causality.",
        ))

    # 4. WSL virtualization (MEDIUM / OBSERVED_CHANGE)
    b_wsl = base_hw.get("wsl", {}).get("is_wsl")
    c_wsl = curr_hw.get("wsl", {}).get("is_wsl")
    if b_wsl != c_wsl and (b_wsl is not None or c_wsl is not None):
        changes.append(DetectedChange(
            category="hardware",
            field="wsl.is_wsl",
            baseline=b_wsl,
            current=c_wsl,
            priority=ChangePriority.MEDIUM.value,
            investigation_priority_score=2,
            role=ChangeRole.OBSERVED_CHANGE.value,
            change_type="modified",
            explanation=f"WSL virtualization state changed from {b_wsl} to {c_wsl}; evidence does not prove causality.",
        ))

    return changes


def _analyze_environment_changes(
    base_env: Dict[str, Any],
    curr_env: Dict[str, Any],
) -> List[DetectedChange]:
    """Analyze software and runtime environment differences."""
    changes: List[DetectedChange] = []

    # 1. ROCm Software Suite (HIGH)
    b_rocm = base_env.get("rocm", {})
    c_rocm = curr_env.get("rocm", {})

    b_rocm_ver = b_rocm.get("version")
    c_rocm_ver = c_rocm.get("version")
    if b_rocm_ver != c_rocm_ver and (b_rocm_ver is not None or c_rocm_ver is not None):
        ch_type = _classify_version_change(b_rocm_ver, c_rocm_ver)
        changes.append(DetectedChange(
            category="software",
            field="rocm.version",
            baseline=b_rocm_ver,
            current=c_rocm_ver,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type=ch_type,
            explanation=f"ROCm version changed from '{b_rocm_ver}' to '{c_rocm_ver}' between benchmark runs. This is a potential contributor because it directly affects the compute runtime; the supplied evidence does not prove that the ROCm change caused the regression.",
        ))

    b_rocm_avail = b_rocm.get("available")
    c_rocm_avail = c_rocm.get("available")
    if b_rocm_avail != c_rocm_avail and (b_rocm_avail is not None or c_rocm_avail is not None):
        changes.append(DetectedChange(
            category="software",
            field="rocm.available",
            baseline=b_rocm_avail,
            current=c_rocm_avail,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"ROCm software availability changed from {b_rocm_avail} to {c_rocm_avail}; evidence does not prove causality.",
        ))

    # HIP version
    b_hip = b_rocm.get("hipconfig", {}).get("version")
    c_hip = c_rocm.get("hipconfig", {}).get("version")
    if b_hip != c_hip and (b_hip is not None or c_hip is not None):
        ch_type = _classify_version_change(b_hip, c_hip)
        changes.append(DetectedChange(
            category="software",
            field="rocm.hipconfig.version",
            baseline=b_hip,
            current=c_hip,
            priority=ChangePriority.HIGH.value,
            investigation_priority_score=3,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type=ch_type,
            explanation=f"HIP compiler/runtime version changed from '{b_hip}' to '{c_hip}'. Directly relates to kernel generation; evidence does not prove causality.",
        ))

    # 2. Python Runtime (MEDIUM)
    b_py = base_env.get("python", {}).get("version")
    c_py = curr_env.get("python", {}).get("version")
    if b_py != c_py and (b_py is not None or c_py is not None):
        ch_type = _classify_version_change(b_py, c_py)
        changes.append(DetectedChange(
            category="runtime",
            field="python.version",
            baseline=b_py,
            current=c_py,
            priority=ChangePriority.MEDIUM.value,
            investigation_priority_score=2,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type=ch_type,
            explanation=f"Python runtime version changed from '{b_py}' to '{c_py}' between benchmark runs. This is a potential contributor; evidence does not prove causality.",
        ))

    # 3. Python Packages
    b_pkgs = base_env.get("python_packages", {})
    c_pkgs = curr_env.get("python_packages", {})
    all_pkg_names = sorted(set(b_pkgs.keys()) | set(c_pkgs.keys()))

    for pkg in all_pkg_names:
        p_base = b_pkgs.get(pkg, {})
        p_curr = c_pkgs.get(pkg, {})
        base_inst = p_base.get("installed", False)
        curr_inst = p_curr.get("installed", False)
        base_v = p_base.get("version")
        curr_v = p_curr.get("version")

        is_target = pkg.lower() in TARGET_AI_PACKAGES
        priority = ChangePriority.MEDIUM.value if is_target else ChangePriority.LOW.value
        score = 2 if is_target else 1
        role = ChangeRole.POTENTIAL_CONTRIBUTOR.value if is_target else ChangeRole.OBSERVED_CHANGE.value

        if not base_inst and curr_inst:
            changes.append(DetectedChange(
                category="software",
                field=f"python_packages.{pkg}",
                baseline=None,
                current=curr_v,
                priority=priority,
                investigation_priority_score=score,
                role=role,
                change_type="added",
                explanation=f"Package '{pkg}' was added in current environment (version: '{curr_v}'). Observed difference; evidence does not prove causality.",
            ))
        elif base_inst and not curr_inst:
            changes.append(DetectedChange(
                category="software",
                field=f"python_packages.{pkg}",
                baseline=base_v,
                current=None,
                priority=priority,
                investigation_priority_score=score,
                role=role,
                change_type="removed",
                explanation=f"Package '{pkg}' was removed from current environment (previous version: '{base_v}'). Observed difference; evidence does not prove causality.",
            ))
        elif base_inst and curr_inst and base_v != curr_v:
            ch_type = _classify_version_change(base_v, curr_v)
            changes.append(DetectedChange(
                category="software",
                field=f"python_packages.{pkg}",
                baseline=base_v,
                current=curr_v,
                priority=priority,
                investigation_priority_score=score,
                role=role,
                change_type=ch_type,
                explanation=f"Package '{pkg}' was {ch_type} from '{base_v}' to '{curr_v}'. This is a potential contributor for AI workload performance; evidence does not prove causality.",
            ))

    # 4. Environment Variables
    b_vars = base_env.get("environment_variables", {})
    c_vars = curr_env.get("environment_variables", {})
    all_var_names = sorted(set(b_vars.keys()) | set(c_vars.keys()))

    for var in all_var_names:
        v_base = b_vars.get(var)
        v_curr = c_vars.get(var)
        if v_base != v_curr:
            ch_type = "added" if v_base is None else ("removed" if v_curr is None else "modified")

            if var in COMPUTE_ENV_VARS:
                prio = ChangePriority.HIGH.value
                sc = 3
                ro = ChangeRole.POTENTIAL_CONTRIBUTOR.value
            elif var in THREAD_ENV_VARS:
                prio = ChangePriority.MEDIUM.value
                sc = 2
                ro = ChangeRole.POTENTIAL_CONTRIBUTOR.value
            else:
                prio = ChangePriority.LOW.value
                sc = 1
                ro = ChangeRole.OBSERVED_CHANGE.value

            changes.append(DetectedChange(
                category="environment",
                field=f"environment_variables.{var}",
                baseline=v_base,
                current=v_curr,
                priority=prio,
                investigation_priority_score=sc,
                role=ro,
                change_type=ch_type,
                explanation=f"Environment variable '{var}' changed from '{v_base}' to '{v_curr}'. Potential runtime contributor; evidence does not prove causality.",
            ))

    # 5. CPU Performance / Governor (MEDIUM)
    b_cpu_perf = base_env.get("cpu_performance", {})
    c_cpu_perf = curr_env.get("cpu_performance", {})
    b_gov = b_cpu_perf.get("scaling_governor")
    c_gov = c_cpu_perf.get("scaling_governor")
    if b_gov != c_gov and (b_gov is not None or c_gov is not None):
        changes.append(DetectedChange(
            category="runtime",
            field="cpu_performance.scaling_governor",
            baseline=b_gov,
            current=c_gov,
            priority=ChangePriority.MEDIUM.value,
            investigation_priority_score=2,
            role=ChangeRole.POTENTIAL_CONTRIBUTOR.value,
            change_type="modified",
            explanation=f"CPU scaling governor changed from '{b_gov}' to '{c_gov}'. Directly influences CPU clock scaling; evidence does not prove causality.",
        ))

    # 6. Build tools (LOW / OBSERVED_CHANGE)
    b_tools = base_env.get("build_tools", {})
    c_tools = curr_env.get("build_tools", {})
    for tool_name in sorted(set(b_tools.keys()) | set(c_tools.keys())):
        t_base = b_tools.get(tool_name, {}).get("version")
        t_curr = c_tools.get(tool_name, {}).get("version")
        if t_base != t_curr and (t_base is not None or t_curr is not None):
            ch_type = _classify_version_change(t_base, t_curr)
            changes.append(DetectedChange(
                category="software",
                field=f"build_tools.{tool_name}",
                baseline=t_base,
                current=t_curr,
                priority=ChangePriority.LOW.value,
                investigation_priority_score=1,
                role=ChangeRole.OBSERVED_CHANGE.value,
                change_type=ch_type,
                explanation=f"Build tool '{tool_name}' version changed from '{t_base}' to '{t_curr}'. Observed difference; evidence does not prove causality.",
            ))

    # 7. Operating system / Kernel (LOW / OBSERVED_CHANGE)
    b_os = base_env.get("os", {}).get("kernel_release")
    c_os = curr_env.get("os", {}).get("kernel_release")
    if b_os != c_os and (b_os is not None or c_os is not None):
        changes.append(DetectedChange(
            category="environment",
            field="os.kernel_release",
            baseline=b_os,
            current=c_os,
            priority=ChangePriority.LOW.value,
            investigation_priority_score=1,
            role=ChangeRole.OBSERVED_CHANGE.value,
            change_type="modified",
            explanation=f"OS kernel release changed from '{b_os}' to '{c_os}'. Observed difference; evidence does not prove causality.",
        ))

    return changes


def diagnose_regression(
    baseline_benchmark: Union[BenchmarkResult, Dict[str, Any]],
    current_benchmark: Union[BenchmarkResult, Dict[str, Any]],
    baseline_hardware: Optional[Union[Dict[str, Any], Any]] = None,
    current_hardware: Optional[Union[Dict[str, Any], Any]] = None,
    baseline_environment: Optional[Union[Dict[str, Any], Any]] = None,
    current_environment: Optional[Union[Dict[str, Any], Any]] = None,
    regression_result: Optional[Union[RegressionResult, Dict[str, Any]]] = None,
) -> DiagnosisResult:
    """Perform root-cause and evidence diagnosis between baseline and current benchmark runs.

    Args:
        baseline_benchmark: Baseline benchmark result object or dictionary.
        current_benchmark: Current benchmark result object or dictionary.
        baseline_hardware: Optional baseline hardware snapshot.
        current_hardware: Optional current hardware snapshot.
        baseline_environment: Optional baseline environment snapshot.
        current_environment: Optional current environment snapshot.
        regression_result: Optional pre-computed regression analysis result.

    Returns:
        DiagnosisResult: Transparent, ranked, evidence-based diagnosis.
    """
    base_bm_dict = _to_dict_safe(baseline_benchmark) or {}
    curr_bm_dict = _to_dict_safe(current_benchmark) or {}
    base_hw_dict = _to_dict_safe(baseline_hardware)
    curr_hw_dict = _to_dict_safe(current_hardware)
    base_env_dict = _to_dict_safe(baseline_environment)
    curr_env_dict = _to_dict_safe(current_environment)
    reg_dict = _to_dict_safe(regression_result)

    # If regression result is not provided, compute it using compare_benchmarks
    if reg_dict is None:
        if compare_benchmarks is not None:
            computed_reg = compare_benchmarks(base_bm_dict, curr_bm_dict)
            reg_dict = computed_reg.to_dict()
        else:
            reg_dict = {"status": "UNKNOWN", "reason": "Regression engine unavailable"}

    reg_status_str = str(reg_dict.get("status", "UNKNOWN"))

    # 1. Check for benchmark configuration / workload comparability differences
    bm_changes = _check_benchmark_configuration_changes(base_bm_dict, curr_bm_dict)
    if bm_changes or reg_status_str == "INCOMPARABLE":
        sorted_bm_changes = _sort_changes(bm_changes)
        return DiagnosisResult(
            schema_version="1.0",
            status=DiagnosisStatus.CONFIGURATION_CHANGED,
            regression_status=reg_status_str,
            summary="Benchmark configuration changed; results may not be directly attributable to a software/environment regression.",
            changes=sorted_bm_changes,
            limitations=[
                "Workload configuration differences directly alter execution characteristics and prevent reliable attribution to software or environment changes.",
                *DEFAULT_LIMITATIONS,
            ],
        )

    # 2. Check if there was no regression to diagnose
    if reg_status_str == "NO_REGRESSION":
        return DiagnosisResult(
            schema_version="1.0",
            status=DiagnosisStatus.NO_REGRESSION_TO_DIAGNOSE,
            regression_status=reg_status_str,
            summary="No performance regression detected (change is within acceptable variance); root-cause diagnosis is not required.",
            changes=[],
            limitations=list(DEFAULT_LIMITATIONS),
        )

    # 3. Check for missing snapshots
    has_hw = bool(base_hw_dict) or bool(curr_hw_dict)
    has_env = bool(base_env_dict) or bool(curr_env_dict)

    if not has_hw and not has_env:
        return DiagnosisResult(
            schema_version="1.0",
            status=DiagnosisStatus.INSUFFICIENT_EVIDENCE,
            regression_status=reg_status_str,
            summary="Performance change was evaluated, but environment and hardware snapshots were not supplied to identify potential contributors.",
            changes=[],
            limitations=list(DEFAULT_LIMITATIONS),
        )

    # 4. Collect and analyze changes across hardware and environment
    all_changes: List[DetectedChange] = []
    if base_hw_dict and curr_hw_dict:
        all_changes.extend(_analyze_hardware_changes(base_hw_dict, curr_hw_dict))
    if base_env_dict and curr_env_dict:
        all_changes.extend(_analyze_environment_changes(base_env_dict, curr_env_dict))

    sorted_changes = _sort_changes(all_changes)

    # 5. Evaluate outcomes when no changes were found
    if not sorted_changes:
        return DiagnosisResult(
            schema_version="1.0",
            status=DiagnosisStatus.INSUFFICIENT_EVIDENCE,
            regression_status=reg_status_str,
            summary="Performance regression was detected, but no relevant configuration or environment changes were observed in the supplied evidence.",
            changes=[],
            limitations=list(DEFAULT_LIMITATIONS),
        )

    # 6. Formulate summary based on regression status
    metric_info = reg_dict.get("metric", {})
    pct_change = metric_info.get("percentage_change")
    metric_name = metric_info.get("name", "metric")

    if "IMPROVEMENT" in reg_status_str:
        summary = f"Performance improved; {len(sorted_changes)} relevant changes were observed."
    elif pct_change is not None:
        summary = f"{metric_name} changed by {pct_change:+.1f}%. {len(sorted_changes)} potential contributing changes were detected."
    else:
        summary = f"Performance change detected with status {reg_status_str}. {len(sorted_changes)} potential contributing changes were detected."

    return DiagnosisResult(
        schema_version="1.0",
        status=DiagnosisStatus.POTENTIAL_CAUSES_FOUND,
        regression_status=reg_status_str,
        summary=summary,
        changes=sorted_changes,
        limitations=list(DEFAULT_LIMITATIONS),
    )


# Convenient alias
diagnose = diagnose_regression


if __name__ == "__main__":
    # Synthetic demonstration of diagnosis engine V1
    demo_baseline_bm = {
        "schema_version": "1.0",
        "fingerprint": "a1b2c3d4e5f60000",
        "benchmark": {"name": "gemm_fp16", "version": "1.0.0"},
        "configuration": {"warmup_runs": 5, "measurement_runs": 30, "operation_count": 1000},
        "latency_ms": {"count": 30, "median": 50.0, "stddev": 1.2},
    }

    demo_current_bm = {
        "schema_version": "1.0",
        "fingerprint": "a1b2c3d4e5f60000",
        "benchmark": {"name": "gemm_fp16", "version": "1.0.0"},
        "configuration": {"warmup_runs": 5, "measurement_runs": 30, "operation_count": 1000},
        "latency_ms": {"count": 30, "median": 60.0, "stddev": 1.9},
    }

    demo_baseline_env = {
        "python": {"version": "3.13.1"},
        "rocm": {"version": "6.2.0", "available": True, "hipconfig": {"version": "6.2.0"}},
        "python_packages": {
            "torch": {"installed": True, "version": "2.4.0+rocm6.2"},
        },
        "environment_variables": {"ROCM_PATH": "/opt/rocm-6.2.0"},
    }

    demo_current_env = {
        "python": {"version": "3.13.1"},
        "rocm": {"version": "7.0.0", "available": True, "hipconfig": {"version": "7.0.0"}},
        "python_packages": {
            "torch": {"installed": True, "version": "2.5.0+rocm7.0"},
        },
        "environment_variables": {"ROCM_PATH": "/opt/rocm-7.0.0"},
    }

    demo_baseline_hw = {
        "cpu": {"model_name": "AMD Ryzen 7 7730U", "logical_cores": 16},
        "gpu": {
            "linux_has_gpu": True,
            "wsl_has_host_gpu": True,
            "host_devices": [{"name": "AMD Radeon (TM) Graphics"}],
        },
        "amd": {"is_amd_hardware": True, "gpu_is_amd": True},
        "rocm": {"is_available": True, "version": "6.2.0", "kfd_available": True},
        "memory": {"linux_total_mb": 7620},
    }

    demo_current_hw = {
        "cpu": {"model_name": "AMD Ryzen 7 7730U", "logical_cores": 16},
        "gpu": {
            "linux_has_gpu": True,
            "wsl_has_host_gpu": True,
            "host_devices": [{"name": "AMD Radeon (TM) Graphics"}],
        },
        "amd": {"is_amd_hardware": True, "gpu_is_amd": True},
        "rocm": {"is_available": True, "version": "7.0.0", "kfd_available": True},
        "memory": {"linux_total_mb": 7620},
    }

    print("=== ROCm Performance Guard — Evidence Diagnosis Engine V1 Demo ===")
    res = diagnose_regression(
        baseline_benchmark=demo_baseline_bm,
        current_benchmark=demo_current_bm,
        baseline_hardware=demo_baseline_hw,
        current_hardware=demo_current_hw,
        baseline_environment=demo_baseline_env,
        current_environment=demo_current_env,
    )
    print(res.to_json(indent=2))
