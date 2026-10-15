"""System and GPU Capability Detection Engine V1 for ROCm Performance Guard.

Detects host and virtualized hardware capabilities, runtime libraries (DXCore, D3D12),
ROCm tools, and AI framework availability honestly from empirical evidence.
Never mutates system configuration, driver state, or environment variables.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CapabilityStatus(str, Enum):
    """Explicit status classification for individual capabilities."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"
    NOT_INSTALLED = "NOT_INSTALLED"


class ComputePath(str, Enum):
    """Recommended compute execution path based on verified system capabilities."""

    ROCM_HIP = "ROCm/HIP"
    DIRECT3D12_WINDOWS = "Direct3D12/Windows acceleration"
    CPU = "CPU"
    UNKNOWN = "UNKNOWN"


DEFAULT_CAPABILITY_LIMITATIONS: List[str] = [
    "Capability detection inspects existing software, drivers, and runtime states without altering them.",
    "Presence of WSL /dev/dxg and Direct3D12 libraries indicates Windows graphics paravirtualization, NOT native ROCm/HIP compute.",
    "ROCm compute requires the ROCm runtime, hipcc/rocminfo tooling, and compatible compute devices.",
    "No system drivers, environment variables, or Python packages are modified or installed.",
]


def _safe_run_cmd(args: List[str], timeout: float = 3.0) -> Optional[str]:
    """Safely execute a command without shell=True, returning stripped stdout if successful."""
    cmd_name = args[0]
    if not shutil.which(cmd_name) and not (Path(cmd_name).is_file() and os.access(cmd_name, os.X_OK)):
        return None
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
        return None
    return None


def _read_file(path_str: str) -> Optional[str]:
    """Safely read the content of a text file, returning None on any error."""
    path = Path(path_str)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, PermissionError):
        return None
    return None


def detect_os_info() -> Dict[str, Any]:
    """Collect operating system details from /etc/os-release and platform facilities."""
    distro_name: Optional[str] = None
    distro_version: Optional[str] = None

    try:
        os_release = platform.freedesktop_os_release()
        distro_name = os_release.get("PRETTY_NAME") or os_release.get("NAME")
        distro_version = os_release.get("VERSION_ID") or os_release.get("VERSION")
    except (AttributeError, OSError):
        os_release_text = _read_file("/etc/os-release") or _read_file("/usr/lib/os-release")
        if os_release_text:
            parsed: Dict[str, str] = {}
            for line in os_release_text.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    parsed[k.strip()] = v.strip().strip("\"'")
            distro_name = parsed.get("PRETTY_NAME") or parsed.get("NAME")
            distro_version = parsed.get("VERSION_ID") or parsed.get("VERSION")

    if not distro_name:
        distro_name = platform.system()

    return {
        "distribution": distro_name,
        "distribution_version": distro_version,
        "kernel_version": platform.release(),
        "architecture": platform.machine(),
    }


def detect_wsl_info() -> Dict[str, Any]:
    """Detect whether running inside Windows Subsystem for Linux (WSL) and collect details."""
    version_text = _read_file("/proc/version") or ""
    release_text = platform.release()
    env_distro = os.environ.get("WSL_DISTRO_NAME")
    env_interop = os.environ.get("WSL_INTEROP")
    dxg_present = os.path.exists("/dev/dxg")

    is_wsl = (
        "microsoft" in version_text.lower()
        or "wsl" in version_text.lower()
        or "microsoft" in release_text.lower()
        or "wsl" in release_text.lower()
        or bool(env_distro)
        or bool(env_interop)
        or dxg_present
    )

    if not is_wsl:
        return {
            "is_wsl": False,
            "wsl_version": None,
            "wsl_package_version": None,
            "distro_name": None,
        }

    wsl_version: Optional[str] = None
    if "wsl2" in version_text.lower() or "wsl2" in release_text.lower() or dxg_present:
        wsl_version = "2"
    elif "wsl" in version_text.lower() or "wsl" in release_text.lower():
        wsl_version = "1"

    pkg_version = _safe_run_cmd(["wslinfo", "--version"], timeout=2.0)

    return {
        "is_wsl": True,
        "wsl_version": wsl_version,
        "wsl_package_version": pkg_version,
        "distro_name": env_distro,
    }


def detect_device_nodes() -> Dict[str, Any]:
    """Detect availability of graphics and compute device nodes in /dev."""
    dxg_exists = os.path.exists("/dev/dxg")
    dri_exists = os.path.exists("/dev/dri")
    kfd_exists = os.path.exists("/dev/kfd")

    return {
        "dxg": {
            "path": "/dev/dxg",
            "available": dxg_exists,
            "status": CapabilityStatus.AVAILABLE.value if dxg_exists else CapabilityStatus.UNAVAILABLE.value,
            "description": "DirectX Graphics Kernel driver device node (WSL2 GPU passthrough)",
        },
        "dri": {
            "path": "/dev/dri",
            "available": dri_exists,
            "status": CapabilityStatus.AVAILABLE.value if dri_exists else CapabilityStatus.UNAVAILABLE.value,
            "description": "Direct Rendering Infrastructure device directory (Linux native GPU access)",
        },
        "kfd": {
            "path": "/dev/kfd",
            "available": kfd_exists,
            "status": CapabilityStatus.AVAILABLE.value if kfd_exists else CapabilityStatus.UNAVAILABLE.value,
            "description": "Kernel Fusion Driver device node (AMD ROCm compute interface)",
        },
    }


def detect_shared_library(names: List[str], candidate_paths: List[str]) -> Dict[str, Any]:
    """Safely test whether a shared library can be located and loaded into process memory.

    Does not modify system files or library paths.
    """
    loaded = False
    resolved_path: Optional[str] = None
    error_msg: Optional[str] = None

    # First attempt explicit candidate paths
    for path in candidate_paths:
        if os.path.isfile(path):
            try:
                ctypes.CDLL(path)
                loaded = True
                resolved_path = path
                break
            except (OSError, Exception) as exc:
                error_msg = str(exc)

    # Next attempt dynamic discovery via ctypes find_library and standard load
    if not loaded:
        for name in names:
            lib_found = ctypes.util.find_library(name)
            target = lib_found or (name if name.endswith(".so") else f"lib{name}.so")
            try:
                ctypes.CDLL(target)
                loaded = True
                resolved_path = target
                break
            except (OSError, Exception) as exc:
                if error_msg is None:
                    error_msg = str(exc)

    status = CapabilityStatus.AVAILABLE if loaded else CapabilityStatus.UNAVAILABLE
    return {
        "available": loaded,
        "path": resolved_path,
        "status": status.value,
        "error": error_msg if not loaded else None,
    }


def detect_graphics_libraries() -> Dict[str, Any]:
    """Detect availability of DXCore and Direct3D 12 acceleration libraries in WSL."""
    dxcore = detect_shared_library(
        names=["dxcore", "libdxcore.so"],
        candidate_paths=[
            "/usr/lib/wsl/lib/libdxcore.so",
            "/usr/lib/wsl/drivers/libdxcore.so",
            "libdxcore.so",
        ],
    )
    d3d12 = detect_shared_library(
        names=["d3d12", "libd3d12.so"],
        candidate_paths=[
            "/usr/lib/wsl/lib/libd3d12.so",
            "/usr/lib/wsl/drivers/libd3d12.so",
            "libd3d12.so",
        ],
    )

    return {
        "dxcore": dxcore,
        "d3d12": d3d12,
    }


def _detect_tool(name: str, candidate_paths: List[str], version_flag: str = "--version") -> Dict[str, Any]:
    """Detect availability, filesystem path, and reported version for an executable tool."""
    path = shutil.which(name)
    if not path:
        for candidate in candidate_paths:
            cand_p = Path(candidate)
            if cand_p.is_file() and os.access(cand_p, os.X_OK):
                path = str(cand_p)
                break

    if not path:
        return {
            "available": False,
            "path": None,
            "version": None,
            "status": CapabilityStatus.UNAVAILABLE.value,
        }

    out = _safe_run_cmd([path, version_flag], timeout=3.0)
    version: Optional[str] = None
    if out:
        first_line = out.splitlines()[0].strip()
        m = re.search(r"(\d+(?:\.\d+)+(?:-[a-zA-Z0-9.+]+)?)", out)
        version = m.group(1) if m else first_line

    return {
        "available": True,
        "path": path,
        "version": version,
        "status": CapabilityStatus.AVAILABLE.value,
    }


def detect_rocm_tools() -> Dict[str, Any]:
    """Detect presence and version info for ROCm tools (rocminfo, rocm-smi, amd-smi, hipcc)."""
    return {
        "rocminfo": _detect_tool("rocminfo", ["/opt/rocm/bin/rocminfo"]),
        "rocm_smi": _detect_tool("rocm-smi", ["/opt/rocm/bin/rocm-smi"]),
        "amd_smi": _detect_tool("amd-smi", ["/opt/rocm/bin/amd-smi"], version_flag="version"),
        "hipcc": _detect_tool("hipcc", ["/opt/rocm/bin/hipcc"]),
    }


def detect_python_frameworks() -> Dict[str, Any]:
    """Detect presence, version, and GPU integration status of AI/ML frameworks."""
    frameworks = ["torch", "onnxruntime", "tensorflow"]
    results: Dict[str, Any] = {}

    for fw in frameworks:
        try:
            ver = importlib.metadata.version(fw)
            gpu_support = "standard"
            if fw == "torch":
                if "+rocm" in ver:
                    gpu_support = "rocm"
                elif "+cu" in ver:
                    gpu_support = "cuda"
                elif "+cpu" in ver:
                    gpu_support = "cpu"
            results[fw] = {
                "installed": True,
                "version": ver,
                "status": CapabilityStatus.AVAILABLE.value,
                "gpu_support": gpu_support,
            }
        except importlib.metadata.PackageNotFoundError:
            results[fw] = {
                "installed": False,
                "version": None,
                "status": CapabilityStatus.NOT_INSTALLED.value,
                "gpu_support": None,
            }

    return results


@dataclass
class SystemCapabilities:
    """Comprehensive, structured, and JSON-serializable capability profile."""

    schema_version: str
    operating_system: Dict[str, Any]
    wsl: Dict[str, Any]
    devices: Dict[str, Any]
    libraries: Dict[str, Any]
    rocm_tools: Dict[str, Any]
    rocm_status: CapabilityStatus
    direct3d12_status: CapabilityStatus
    environment_variables: Dict[str, Optional[str]]
    python: Dict[str, Any]
    frameworks: Dict[str, Any]
    recommended_compute_path: ComputePath
    summary: str
    limitations: List[str] = field(default_factory=lambda: list(DEFAULT_CAPABILITY_LIMITATIONS))

    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities profile to a clean, serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "operating_system": self.operating_system,
            "wsl": self.wsl,
            "devices": self.devices,
            "libraries": self.libraries,
            "rocm_tools": self.rocm_tools,
            "rocm_status": self.rocm_status.value if isinstance(self.rocm_status, CapabilityStatus) else str(self.rocm_status),
            "direct3d12_status": self.direct3d12_status.value if isinstance(self.direct3d12_status, CapabilityStatus) else str(self.direct3d12_status),
            "environment_variables": self.environment_variables,
            "python": self.python,
            "frameworks": self.frameworks,
            "recommended_compute_path": self.recommended_compute_path.value if isinstance(self.recommended_compute_path, ComputePath) else str(self.recommended_compute_path),
            "summary": self.summary,
            "limitations": list(self.limitations),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize capability profile to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def detect_capabilities() -> SystemCapabilities:
    """Evaluate and produce the complete, empirical system capability profile.

    Does not modify system files, environment variables, or drivers.
    """
    os_info = detect_os_info()
    wsl_info = detect_wsl_info()
    devices = detect_device_nodes()
    libraries = detect_graphics_libraries()
    rocm_tools = detect_rocm_tools()
    frameworks = detect_python_frameworks()

    # Track relevant environment variables
    env_vars = {
        "ROCM_PATH": os.environ.get("ROCM_PATH"),
        "HIP_PATH": os.environ.get("HIP_PATH"),
        "HSA_PATH": os.environ.get("HSA_PATH"),
    }

    # Python runtime info
    python_info = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "architecture": platform.architecture()[0],
    }

    # 1. Independent evaluation of ROCm compute capability
    # Do NOT infer ROCm availability merely from /dev/dxg or libd3d12/dxcore
    any_rocm_tool = any(t.get("available") for t in rocm_tools.values())
    rocm_path_present = bool(env_vars.get("ROCM_PATH"))
    kfd_present = devices.get("kfd", {}).get("available", False)

    if any_rocm_tool and (kfd_present or rocm_path_present):
        rocm_status = CapabilityStatus.AVAILABLE
    elif any_rocm_tool:
        # Tools present without KFD compute driver
        rocm_status = CapabilityStatus.AVAILABLE
    else:
        rocm_status = CapabilityStatus.UNAVAILABLE

    # 2. Independent evaluation of Direct3D 12 acceleration capability
    dxg_avail = devices.get("dxg", {}).get("available", False)
    dxcore_avail = libraries.get("dxcore", {}).get("available", False)
    d3d12_avail = libraries.get("d3d12", {}).get("available", False)

    if dxg_avail and (dxcore_avail or d3d12_avail):
        direct3d12_status = CapabilityStatus.AVAILABLE
    else:
        direct3d12_status = CapabilityStatus.UNAVAILABLE

    # 3. Determine recommended compute path
    if rocm_status == CapabilityStatus.AVAILABLE:
        recommended_path = ComputePath.ROCM_HIP
        summary = "ROCm/HIP tooling detected and available for compute workloads."
    elif direct3d12_status == CapabilityStatus.AVAILABLE:
        recommended_path = ComputePath.DIRECT3D12_WINDOWS
        summary = (
            "Direct3D 12 and /dev/dxg detected in WSL; Windows host graphics acceleration is available "
            "(e.g., DirectML). Native ROCm GPU compute is UNAVAILABLE."
        )
    else:
        recommended_path = ComputePath.CPU
        summary = "No GPU compute runtime or graphics acceleration detected. CPU execution is recommended."

    # Construct active limitations
    limitations = list(DEFAULT_CAPABILITY_LIMITATIONS)
    if rocm_status == CapabilityStatus.UNAVAILABLE:
        limitations.insert(0, "ROCm GPU compute support is UNAVAILABLE on this host/WSL environment.")

    return SystemCapabilities(
        schema_version="1.0",
        operating_system=os_info,
        wsl=wsl_info,
        devices=devices,
        libraries=libraries,
        rocm_tools=rocm_tools,
        rocm_status=rocm_status,
        direct3d12_status=direct3d12_status,
        environment_variables=env_vars,
        python=python_info,
        frameworks=frameworks,
        recommended_compute_path=recommended_path,
        summary=summary,
        limitations=limitations,
    )


def print_capability_report(caps: SystemCapabilities) -> None:
    """Print human-readable capability summary followed by complete JSON."""
    os_dist = caps.operating_system.get("distribution") or "Unknown"
    os_arch = caps.operating_system.get("architecture") or "Unknown"
    kernel = caps.operating_system.get("kernel_version") or "Unknown"

    wsl_status = (
        f"WSL{caps.wsl.get('wsl_version') or ''} (Distro: {caps.wsl.get('distro_name') or 'unknown'})"
        if caps.wsl.get("is_wsl")
        else "Native Linux (Non-WSL)"
    )

    dxg_st = caps.devices.get("dxg", {}).get("status", "UNAVAILABLE")
    dri_st = caps.devices.get("dri", {}).get("status", "UNAVAILABLE")
    kfd_st = caps.devices.get("kfd", {}).get("status", "UNAVAILABLE")

    dxcore_st = caps.libraries.get("dxcore", {}).get("status", "UNAVAILABLE")
    dxcore_path = caps.libraries.get("dxcore", {}).get("path") or "not found"
    d3d12_st = caps.libraries.get("d3d12", {}).get("status", "UNAVAILABLE")
    d3d12_path = caps.libraries.get("d3d12", {}).get("path") or "not found"

    print("=" * 60)
    print("ROCm Performance Guard - System Capability Profile V1")
    print("=" * 60)
    print(f"Operating System   : {os_dist} ({os_arch})")
    print(f"Kernel Version     : {kernel}")
    print(f"Environment        : {wsl_status}")
    print(f"Python Runtime     : {caps.python.get('version')} ({caps.python.get('executable')})")
    print("-" * 60)
    print("Device Interfaces:")
    print(f"  /dev/dxg (WSL)   : {dxg_st}")
    print(f"  /dev/dri (Linux) : {dri_st}")
    print(f"  /dev/kfd (ROCm)  : {kfd_st}")
    print("Graphics Libraries:")
    print(f"  DXCore           : {dxcore_st} ({dxcore_path})")
    print(f"  Direct3D 12      : {d3d12_st} ({d3d12_path})")
    print("ROCm Tooling:")
    for tool_name, tool_info in caps.rocm_tools.items():
        t_st = tool_info.get("status", "UNAVAILABLE")
        t_ver = f" (v{tool_info.get('version')})" if tool_info.get("version") else ""
        print(f"  {tool_name:<16} : {t_st}{t_ver}")
    print("Python Frameworks:")
    for fw_name, fw_info in caps.frameworks.items():
        f_st = fw_info.get("status", "NOT_INSTALLED")
        f_ver = f" ({fw_info.get('version')})" if fw_info.get("version") else ""
        print(f"  {fw_name:<16} : {f_st}{f_ver}")
    print("-" * 60)
    print(f"ROCm Compute       : {caps.rocm_status.value}")
    print(f"Direct3D 12 Accel  : {caps.direct3d12_status.value}")
    print(f"Recommended Path   : {caps.recommended_compute_path.value}")
    print(f"Summary            : {caps.summary}")
    print("=" * 60)
    print("Capability JSON:")
    print(caps.to_json(indent=2))


if __name__ == "__main__":
    capabilities = detect_capabilities()
    print_capability_report(capabilities)
