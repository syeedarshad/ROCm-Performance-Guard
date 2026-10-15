"""Environment detection module for ROCm Performance Guard.

Collects a reproducible snapshot of software, compiler, ROCm, Python packages,
and runtime conditions that may affect AI workload performance.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RELEVANT_ENV_VARS: List[str] = [
    "PATH",
    "LD_LIBRARY_PATH",
    "ROCM_PATH",
    "HIP_PATH",
    "HSA_PATH",
    "HSA_OVERRIDE_GFX_VERSION",
    "AMD_LOG_LEVEL",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "CUDA_VISIBLE_DEVICES",
    "MIOPEN_LOG_LEVEL",
    "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM",
    "PYTORCH_ROCM_ARCH",
    "TORCH_CUDA_ARCH_LIST",
]

TARGET_PACKAGES: List[str] = [
    "torch",
    "torchvision",
    "transformers",
    "onnxruntime",
    "numpy",
    "scipy",
]


def _safe_run_cmd(args: List[str], timeout: float = 3.0) -> Optional[str]:
    """Safely execute a command without shell=True, returning stripped stdout if successful."""
    cmd_name = args[0]
    if not shutil.which(cmd_name):
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


def get_python_info() -> Dict[str, Any]:
    """Collect Python runtime information.

    Returns:
        Dict[str, Any]: Python version, implementation, executable path, and architecture.
    """
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "architecture": platform.architecture()[0],
    }


def get_os_info() -> Dict[str, Any]:
    """Collect operating system details from /etc/os-release and platform facilities.

    Returns:
        Dict[str, Any]: OS distribution, version, kernel release, and architecture.
    """
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


def get_wsl_info() -> Dict[str, Any]:
    """Detect whether running inside Windows Subsystem for Linux (WSL) and collect details.

    Returns:
        Dict[str, Any]: WSL detection status, WSL version, and distro name.
    """
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


def _detect_tool_version(cmd_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Detect tool availability, version string, and path."""
    path = shutil.which(cmd_name)
    if not path:
        return False, None, None

    out = _safe_run_cmd([path, "--version"], timeout=3.0)
    if not out:
        return True, None, path

    first_line = out.splitlines()[0].strip()
    m = re.search(r"(\d+(?:\.\d+)+(?:-[a-zA-Z0-9.+]+)?)", first_line)
    version = m.group(1) if m else first_line
    return True, version, path


def get_build_tools_info() -> Dict[str, Any]:
    """Detect availability and versions of common compilers and build tools.

    Returns:
        Dict[str, Any]: Status for gcc, g++, clang, cmake, and make.
    """
    tools = ["gcc", "g++", "clang", "cmake", "make"]
    result: Dict[str, Any] = {}

    for tool in tools:
        avail, ver, path = _detect_tool_version(tool)
        result[tool] = {
            "available": avail,
            "version": ver,
            "path": path,
        }

    return result


def get_rocm_software_info() -> Dict[str, Any]:
    """Detect whether AMD ROCm tools are available and collect version information.

    Does not install anything and does not assume ROCm is present.

    Returns:
        Dict[str, Any]: Status of rocminfo, amd-smi, rocm-smi, and hipconfig.
    """
    tools = ["rocminfo", "amd-smi", "rocm-smi", "hipconfig"]
    tool_status: Dict[str, Any] = {}
    any_available = False

    for tool in tools:
        path = shutil.which(tool)
        if not path:
            candidate = Path(f"/opt/rocm/bin/{tool}")
            if candidate.is_file() and os.access(candidate, os.X_OK):
                path = str(candidate)

        key = tool.replace("-", "_")
        if not path:
            tool_status[key] = {
                "available": False,
                "version": None,
                "summary": None,
                "path": None,
            }
        else:
            any_available = True
            version = None
            summary = None
            ver_arg = "version" if tool == "amd-smi" else "--version"
            out = _safe_run_cmd([path, ver_arg], timeout=3.0)
            if out:
                lines = out.splitlines()
                summary = lines[0].strip() if lines else None
                m = re.search(r"(\d+(?:\.\d+)+(?:-[a-zA-Z0-9.+]+)?)", out)
                version = m.group(1) if m else summary

            tool_status[key] = {
                "available": True,
                "version": version,
                "summary": summary,
                "path": path,
            }

    rocm_version: Optional[str] = None
    for vfile in ("/opt/rocm/.info/version", "/opt/rocm/version"):
        content = _read_file(vfile)
        if content:
            rocm_version = content.splitlines()[0].strip()
            break

    if not rocm_version and tool_status.get("hipconfig", {}).get("version"):
        rocm_version = tool_status["hipconfig"]["version"]

    rocm_available = any_available or bool(rocm_version) or Path("/opt/rocm").is_dir()

    return {
        "available": rocm_available,
        "version": rocm_version,
        "rocminfo": tool_status["rocminfo"],
        "amd_smi": tool_status["amd_smi"],
        "rocm_smi": tool_status["rocm_smi"],
        "hipconfig": tool_status["hipconfig"],
    }


def get_python_packages_info() -> Dict[str, Any]:
    """Inspect installation status and versions of key AI/ML Python packages.

    Uses importlib.metadata to avoid importing heavy libraries.

    Returns:
        Dict[str, Any]: Mapping of package name to installation status and version.
    """
    result: Dict[str, Any] = {}
    for pkg in TARGET_PACKAGES:
        try:
            ver = importlib.metadata.version(pkg)
            result[pkg] = {
                "installed": True,
                "version": ver,
            }
        except importlib.metadata.PackageNotFoundError:
            result[pkg] = {
                "installed": False,
                "version": None,
            }
    return result


def get_environment_variables() -> Dict[str, Optional[str]]:
    """Collect only environment variables relevant to performance diagnostics.

    Guarantees secrets, API keys, passwords, and tokens are never collected.

    Returns:
        Dict[str, Optional[str]]: Whitelisted performance environment variables.
    """
    result: Dict[str, Optional[str]] = {}
    for var in RELEVANT_ENV_VARS:
        val = os.environ.get(var)
        result[var] = val
    return result


def get_cpu_performance_info() -> Dict[str, Any]:
    """Inspect CPU scaling governor and frequency configuration if exposed by Linux kernel.

    Returns:
        Dict[str, Any]: CPU governor, frequency in MHz, and availability status.
    """
    scaling_dir = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    if not scaling_dir.is_dir():
        return {
            "available": False,
            "governor": None,
            "frequency_mhz": None,
            "details": "CPU frequency scaling information not available in this environment",
        }

    governor = _read_file(str(scaling_dir / "scaling_governor"))
    cur_freq_khz = _read_file(str(scaling_dir / "scaling_cur_freq"))
    freq_mhz = None
    if cur_freq_khz and cur_freq_khz.isdigit():
        freq_mhz = round(int(cur_freq_khz) / 1000.0, 2)

    return {
        "available": True,
        "governor": governor,
        "frequency_mhz": freq_mhz,
        "details": None,
    }


def get_tool_paths() -> Dict[str, Optional[str]]:
    """Record executable locations for key compilers, interpreters, and ROCm tools.

    Uses runtime PATH resolution without assuming standard directories.

    Returns:
        Dict[str, Optional[str]]: Mapping of tool name to executable path.
    """
    tools = ["python", "gcc", "clang", "rocminfo", "amd-smi", "hipconfig"]
    paths: Dict[str, Optional[str]] = {}

    for tool in tools:
        if tool == "python":
            paths[tool] = sys.executable
        else:
            p = shutil.which(tool)
            if not p and tool in ("rocminfo", "amd-smi", "hipconfig"):
                candidate = Path(f"/opt/rocm/bin/{tool}")
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    p = str(candidate)
            paths[tool] = p

    return paths


def get_environment_info() -> Dict[str, Any]:
    """Collect comprehensive environment and software runtime configuration.

    Returns:
        Dict[str, Any]: A JSON-serializable dictionary containing:
            - python: Python runtime details.
            - os: Operating system details.
            - wsl: WSL environment detection.
            - build_tools: Available build tools and versions.
            - rocm: AMD ROCm software tools and version.
            - python_packages: Status and versions of key AI/ML libraries.
            - environment_variables: Diagnostic environment variables.
            - cpu_performance: CPU frequency and governor configuration.
            - tool_paths: Paths to key executables.
    """
    return {
        "python": get_python_info(),
        "os": get_os_info(),
        "wsl": get_wsl_info(),
        "build_tools": get_build_tools_info(),
        "rocm": get_rocm_software_info(),
        "python_packages": get_python_packages_info(),
        "environment_variables": get_environment_variables(),
        "cpu_performance": get_cpu_performance_info(),
        "tool_paths": get_tool_paths(),
    }


if __name__ == "__main__":
    info = get_environment_info()
    print(json.dumps(info, indent=2))