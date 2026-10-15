"""Hardware and environment detection module for ROCm Performance Guard.

Provides runtime detection of CPU, memory, operating system, WSL environment,
GPU devices (distinguishing Linux-visible, virtualized, and physical host hardware),
AMD hardware presence, and ROCm tooling status.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HOST_HARDWARE_CACHE: Optional[Tuple[Optional[int], List[Dict[str, Any]]]] = None


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


def is_running_in_wsl() -> bool:
    """Determine whether running inside Windows Subsystem for Linux (WSL)."""
    version_text = _read_file("/proc/version") or ""
    release_text = platform.release()
    env_distro = os.environ.get("WSL_DISTRO_NAME")
    env_interop = os.environ.get("WSL_INTEROP")
    dxg_present = os.path.exists("/dev/dxg")

    return (
        "microsoft" in version_text.lower()
        or "wsl" in version_text.lower()
        or "microsoft" in release_text.lower()
        or "wsl" in release_text.lower()
        or bool(env_distro)
        or bool(env_interop)
        or dxg_present
    )


def _get_wsl_host_hardware(force_refresh: bool = False) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """Query Windows host memory and GPU hardware via WSL interop if available.

    Returns:
        Tuple of (host_physical_memory_bytes, list_of_host_gpu_devices).
    """
    global _HOST_HARDWARE_CACHE
    if _HOST_HARDWARE_CACHE is not None and not force_refresh:
        return _HOST_HARDWARE_CACHE

    if not is_running_in_wsl():
        _HOST_HARDWARE_CACHE = (None, [])
        return _HOST_HARDWARE_CACHE

    powershell_path = shutil.which("powershell.exe")
    if not powershell_path:
        _HOST_HARDWARE_CACHE = (None, [])
        return _HOST_HARDWARE_CACHE

    ps_command = (
        "@{Memory = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
        "GPUs = @(Get-CimInstance Win32_VideoController | "
        "Select-Object Name, DeviceID, PNPDeviceID, DriverVersion, AdapterRAM)} | "
        "ConvertTo-Json -Depth 3"
    )

    try:
        res = subprocess.run(
            [powershell_path, "-NoProfile", "-NonInteractive", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=8.0,
            check=False,
        )
        if res.returncode != 0 or not res.stdout.strip():
            _HOST_HARDWARE_CACHE = (None, [])
            return _HOST_HARDWARE_CACHE

        data = json.loads(res.stdout)
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        _HOST_HARDWARE_CACHE = (None, [])
        return _HOST_HARDWARE_CACHE

    # Parse physical host memory
    host_bytes: Optional[int] = None
    raw_mem = data.get("Memory")
    if isinstance(raw_mem, (int, float)) and raw_mem > 0:
        host_bytes = int(raw_mem)

    # Parse physical host GPUs
    host_gpus: List[Dict[str, Any]] = []
    raw_gpus = data.get("GPUs")
    if isinstance(raw_gpus, dict):
        raw_gpus = [raw_gpus]
    elif not isinstance(raw_gpus, list):
        raw_gpus = []

    for gpu_entry in raw_gpus:
        if not isinstance(gpu_entry, dict):
            continue
        name = gpu_entry.get("Name")
        pnp_id = gpu_entry.get("PNPDeviceID") or ""
        driver_version = gpu_entry.get("DriverVersion")

        ven_match = re.search(r"VEN_([0-9a-fA-F]{4})", pnp_id)
        dev_match = re.search(r"DEV_([0-9a-fA-F]{4})", pnp_id)
        pci_vendor_id = ven_match.group(1).upper() if ven_match else None
        pci_device_id = dev_match.group(1).upper() if dev_match else None

        vendor: Optional[str] = None
        if pci_vendor_id == "1002":
            vendor = "Advanced Micro Devices, Inc."
        elif pci_vendor_id == "10DE":
            vendor = "NVIDIA Corporation"
        elif pci_vendor_id == "8086":
            vendor = "Intel Corporation"
        elif name:
            if "AMD" in name.upper() or "RADEON" in name.upper():
                vendor = "Advanced Micro Devices, Inc."
            elif "NVIDIA" in name.upper() or "GEFORCE" in name.upper():
                vendor = "NVIDIA Corporation"
            elif "INTEL" in name.upper():
                vendor = "Intel Corporation"

        host_gpus.append({
            "vendor": vendor,
            "name": name,
            "pci_vendor_id": pci_vendor_id,
            "pci_device_id": pci_device_id,
            "driver_version": driver_version,
        })

    _HOST_HARDWARE_CACHE = (host_bytes, host_gpus)
    return _HOST_HARDWARE_CACHE


def get_cpu_info() -> Dict[str, Any]:
    """Collect CPU information including vendor, model name, physical cores, and logical cores.

    Returns:
        Dict[str, Any]: CPU details.
    """
    vendor: Optional[str] = None
    model_name: Optional[str] = None
    physical_cores: Optional[int] = None
    logical_cores: Optional[int] = os.cpu_count()

    cpuinfo_text = _read_file("/proc/cpuinfo")
    if cpuinfo_text:
        vendor_match = re.search(r"^vendor_id\s*:\s*(.+)$", cpuinfo_text, re.MULTILINE)
        if vendor_match:
            vendor = vendor_match.group(1).strip()

        model_match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo_text, re.MULTILINE)
        if model_match:
            model_name = model_match.group(1).strip()

        core_pairs = set()
        current_phys_id: Optional[str] = None
        current_core_id: Optional[str] = None

        for line in cpuinfo_text.splitlines():
            line = line.strip()
            if not line:
                if current_phys_id is not None and current_core_id is not None:
                    core_pairs.add((current_phys_id, current_core_id))
                current_phys_id = None
                current_core_id = None
                continue

            if line.startswith("physical id"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    current_phys_id = parts[1].strip()
            elif line.startswith("core id"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    current_core_id = parts[1].strip()

        if current_phys_id is not None and current_core_id is not None:
            core_pairs.add((current_phys_id, current_core_id))

        if core_pairs:
            physical_cores = len(core_pairs)
        else:
            cores_match = re.search(r"^cpu cores\s*:\s*(\d+)$", cpuinfo_text, re.MULTILINE)
            if cores_match:
                try:
                    physical_cores = int(cores_match.group(1).strip())
                except ValueError:
                    pass

    if not vendor:
        proc = platform.processor()
        if proc:
            vendor = proc
    if not model_name:
        proc = platform.processor()
        if proc:
            model_name = proc

    if logical_cores is None and cpuinfo_text:
        processors = re.findall(r"^processor\s*:\s*\d+", cpuinfo_text, re.MULTILINE)
        if processors:
            logical_cores = len(processors)

    return {
        "vendor": vendor,
        "model_name": model_name,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
    }


def get_memory_info() -> Dict[str, Any]:
    """Collect total system RAM information, distinguishing visible Linux memory and host memory.

    Returns:
        Dict[str, Any]: Memory details containing visible and host physical memory metrics.
    """
    visible_bytes: Optional[int] = None

    meminfo_text = _read_file("/proc/meminfo")
    if meminfo_text:
        match = re.search(r"^MemTotal:\s*(\d+)\s*kB", meminfo_text, re.MULTILINE)
        if match:
            try:
                visible_bytes = int(match.group(1)) * 1024
            except ValueError:
                pass

    if visible_bytes is None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if page_size > 0 and phys_pages > 0:
                visible_bytes = page_size * phys_pages
        except (ValueError, OSError, AttributeError):
            pass

    visible_gb = round(visible_bytes / (1024 * 1024 * 1024), 2) if visible_bytes is not None else None

    # Host memory query via WSL interop
    host_bytes, _ = _get_wsl_host_hardware()
    host_gb = round(host_bytes / (1024 * 1024 * 1024), 2) if host_bytes is not None else None

    source = "proc_meminfo_and_wsl_host_interop" if host_bytes is not None else "proc_meminfo"

    return {
        "visible_to_linux_bytes": visible_bytes,
        "visible_to_linux_gb": visible_gb,
        "host_physical_bytes": host_bytes,
        "host_physical_gb": host_gb,
        "source": source,
        # Backward-compatible fields
        "total_bytes": visible_bytes,
        "total_mb": round(visible_bytes / (1024 * 1024), 2) if visible_bytes is not None else None,
        "total_gb": visible_gb,
    }


def get_os_info() -> Dict[str, Any]:
    """Collect operating system information including distribution, kernel, and architecture.

    Returns:
        Dict[str, Any]: OS details.
    """
    distro_name: Optional[str] = None
    distro_id: Optional[str] = None
    distro_version: Optional[str] = None

    try:
        os_release = platform.freedesktop_os_release()
        distro_name = os_release.get("PRETTY_NAME") or os_release.get("NAME")
        distro_id = os_release.get("ID")
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
            distro_id = parsed.get("ID")
            distro_version = parsed.get("VERSION_ID") or parsed.get("VERSION")

    if not distro_name:
        distro_name = platform.system()

    return {
        "distribution": distro_name,
        "distribution_id": distro_id,
        "distribution_version": distro_version,
        "kernel_version": platform.release(),
        "architecture": platform.machine(),
    }


def get_wsl_info() -> Dict[str, Any]:
    """Detect whether running inside Windows Subsystem for Linux (WSL) and collect details.

    Returns:
        Dict[str, Any]: WSL detection status, WSL version, and distro name.
    """
    is_wsl = is_running_in_wsl()
    if not is_wsl:
        return {
            "is_wsl": False,
            "wsl_version": None,
            "wsl_package_version": None,
            "distro_name": None,
        }

    version_text = _read_file("/proc/version") or ""
    release_text = platform.release()
    env_distro = os.environ.get("WSL_DISTRO_NAME")
    dxg_present = os.path.exists("/dev/dxg")

    wsl_version: Optional[str] = None
    if "wsl2" in version_text.lower() or "wsl2" in release_text.lower() or dxg_present:
        wsl_version = "2"
    elif "wsl" in version_text.lower() or "wsl" in release_text.lower():
        wsl_version = "1"

    wsl_pkg_version = _safe_run_cmd(["wslinfo", "--version"], timeout=2.0)

    return {
        "is_wsl": True,
        "wsl_version": wsl_version,
        "wsl_package_version": wsl_pkg_version,
        "distro_name": env_distro,
    }


def _parse_lspci_block(block: str) -> Optional[Dict[str, Optional[str]]]:
    """Parse a single lspci -Dvmmnn device block if it is a display/GPU class controller."""
    lines = block.strip().splitlines()
    data: Dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip()

    class_val = data.get("Class", "")
    class_match = re.search(r"^(.*?)\s*\[([0-9a-fA-F]{4})\]", class_val)
    class_name = class_match.group(1).strip() if class_match else class_val
    class_code = class_match.group(2).lower() if class_match else ""

    if not class_code.startswith("03"):
        return None

    def _extract_name_and_id(raw_val: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.search(r"^(.*?)\s*\[([0-9a-fA-F]{4})\]", raw_val)
        if m:
            return m.group(1).strip(), m.group(2).lower()
        return (raw_val if raw_val else None), None

    vendor_name, vendor_id = _extract_name_and_id(data.get("Vendor", ""))
    device_name, device_id = _extract_name_and_id(data.get("Device", ""))

    return {
        "pci_slot": data.get("Slot"),
        "vendor": vendor_name,
        "vendor_id": vendor_id,
        "device": device_name,
        "device_id": device_id,
        "class_name": class_name or None,
        "class_code": class_code or None,
    }


def get_gpu_info() -> Dict[str, Any]:
    """Detect GPU information, distinguishing Linux-visible, virtualized, and host devices.

    Returns:
        Dict[str, Any]: Detected GPU device list and virtualization status.
    """
    linux_visible_devices: List[Dict[str, Optional[str]]] = []

    # Method 1: Try lspci -Dvmmnn for detailed machine-readable output
    lspci_out = _safe_run_cmd(["lspci", "-Dvmmnn"], timeout=3.0)
    if lspci_out:
        blocks = re.split(r"\n\s*\n", lspci_out.strip())
        for block in blocks:
            parsed = _parse_lspci_block(block)
            if parsed:
                linux_visible_devices.append(parsed)

    # Method 2: Fallback to reading /sys/bus/pci/devices if lspci gave nothing or was missing
    if not linux_visible_devices and Path("/sys/bus/pci/devices").is_dir():
        try:
            for pci_path in Path("/sys/bus/pci/devices").iterdir():
                class_file = pci_path / "class"
                if not class_file.is_file():
                    continue
                class_raw = _read_file(str(class_file)) or ""
                class_code = class_raw.strip().replace("0x", "")[:4].lower()
                if class_code.startswith("03"):
                    vendor_raw = _read_file(str(pci_path / "vendor")) or ""
                    device_raw = _read_file(str(pci_path / "device")) or ""
                    vendor_id = vendor_raw.strip().replace("0x", "").lower() or None
                    device_id = device_raw.strip().replace("0x", "").lower() or None
                    linux_visible_devices.append({
                        "pci_slot": pci_path.name,
                        "vendor": None,
                        "vendor_id": vendor_id,
                        "device": None,
                        "device_id": device_id,
                        "class_name": "Display controller",
                        "class_code": class_code,
                    })
        except (OSError, PermissionError):
            pass

    is_wsl = is_running_in_wsl()
    _, host_devices = _get_wsl_host_hardware()

    detected = bool(linux_visible_devices or host_devices)
    count = len(linux_visible_devices)

    return {
        "virtualized": is_wsl,
        "linux_visible_devices": linux_visible_devices,
        "host_devices": host_devices,
        # Backward-compatible fields
        "detected": detected,
        "count": count,
        "devices": linux_visible_devices,
    }


def get_amd_info(cpu_info: Dict[str, Any], gpu_info: Dict[str, Any]) -> Dict[str, Any]:
    """Determine whether AMD hardware (CPU or GPU) is present in the system.

    Distinguishes physical AMD GPUs from virtualized render drivers and considers
    both host physical GPUs and Linux-visible GPUs.

    Args:
        cpu_info: The dictionary returned by get_cpu_info().
        gpu_info: The dictionary returned by get_gpu_info().

    Returns:
        Dict[str, Any]: AMD detection status and identified hardware details.
    """
    cpu_vendor = (cpu_info.get("vendor") or "").lower()
    cpu_model = (cpu_info.get("model_name") or "").lower()
    cpu_is_amd = (
        "authenticamd" in cpu_vendor
        or bool(re.search(r"\bamd\b", cpu_vendor))
        or bool(re.search(r"\bamd\b", cpu_model))
    )

    gpu_is_amd = False
    details: List[str] = []

    if cpu_is_amd:
        details.append(f"AMD CPU: {cpu_info.get('model_name') or cpu_info.get('vendor')}")

    # Check physical host GPUs (e.g. detected via WSL Windows interop)
    for host_dev in gpu_info.get("host_devices", []):
        vendor_id = (host_dev.get("pci_vendor_id") or "").upper()
        vendor_str = (host_dev.get("vendor") or "").lower()
        name_str = (host_dev.get("name") or "").lower()

        is_amd = (
            vendor_id == "1002"
            or "advanced micro devices" in vendor_str
            or bool(re.search(r"\bamd\b", vendor_str))
            or bool(re.search(r"\bradeon\b", name_str))
        )
        if is_amd:
            gpu_is_amd = True
            name = host_dev.get("name") or "Unknown AMD Host GPU"
            ven_id = host_dev.get("pci_vendor_id") or "1002"
            dev_id = host_dev.get("pci_device_id") or "Unknown"
            details.append(f"AMD Host GPU: {name} (PCI {ven_id}:{dev_id})")

    # Check Linux-visible GPUs (e.g. discrete GPUs passed through or native Linux)
    # Note: Microsoft Basic Render Driver (VEN_1414, DEV_008E) is NOT treated as AMD.
    for dev in gpu_info.get("linux_visible_devices", []):
        vendor_str = (dev.get("vendor") or "").lower()
        vendor_id = (dev.get("vendor_id") or "").lower().replace("0x", "")

        is_amd = (
            vendor_id == "1002"
            or "advanced micro devices" in vendor_str
            or bool(re.search(r"\bamd\b", vendor_str))
            or bool(re.search(r"\bati\b", vendor_str))
        )
        if is_amd:
            gpu_is_amd = True
            dev_desc = dev.get("device") or dev.get("device_id") or "Unknown AMD GPU"
            slot = dev.get("pci_slot")
            details.append(f"AMD Linux-visible GPU [{slot}]: {dev_desc}")

    present = cpu_is_amd or gpu_is_amd

    return {
        "present": present,
        "cpu_is_amd": cpu_is_amd,
        "gpu_is_amd": gpu_is_amd,
        "details": details,
    }


def get_rocm_info() -> Dict[str, Any]:
    """Detect whether AMD ROCm tools and runtime environment are available.

    Does not assume ROCm is present even if AMD hardware exists.

    Returns:
        Dict[str, Any]: ROCm presence, detected tool statuses, and version.
    """
    def find_tool(name: str) -> Optional[str]:
        path = shutil.which(name)
        if path:
            return path
        candidate = Path(f"/opt/rocm/bin/{name}")
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None

    rocminfo_path = find_tool("rocminfo")
    amd_smi_path = find_tool("amd-smi")
    rocm_smi_path = find_tool("rocm-smi")
    hipconfig_path = find_tool("hipconfig")

    rocm_version: Optional[str] = None

    # 1. Check /opt/rocm/.info/version
    version_file = Path("/opt/rocm/.info/version")
    if version_file.is_file():
        text = _read_file(str(version_file))
        if text:
            rocm_version = text.splitlines()[0].strip()

    # 2. Check /opt/rocm/version
    if not rocm_version:
        legacy_version_file = Path("/opt/rocm/version")
        if legacy_version_file.is_file():
            text = _read_file(str(legacy_version_file))
            if text:
                rocm_version = text.splitlines()[0].strip()

    # 3. Check hipconfig --version
    if not rocm_version and hipconfig_path:
        out = _safe_run_cmd([hipconfig_path, "--version"], timeout=3.0)
        if out:
            rocm_version = out.splitlines()[0].strip()

    # 4. Check amd-smi version
    if not rocm_version and amd_smi_path:
        out = _safe_run_cmd([amd_smi_path, "version"], timeout=3.0)
        if out:
            m = re.search(r"ROCm version:\s*([^\s,]+)", out, re.IGNORECASE)
            if m:
                rocm_version = m.group(1).strip()

    # 5. Check rocm-smi --version
    if not rocm_version and rocm_smi_path:
        out = _safe_run_cmd([rocm_smi_path, "--version"], timeout=3.0)
        if out:
            rocm_version = out.splitlines()[0].strip()

    kfd_available = os.path.exists("/dev/kfd")
    opt_rocm_exists = os.path.isdir("/opt/rocm")

    is_available = bool(
        rocminfo_path or amd_smi_path or rocm_smi_path or hipconfig_path or rocm_version
    )

    return {
        "is_available": is_available,
        "version": rocm_version,
        "tools": {
            "rocminfo": bool(rocminfo_path),
            "amd_smi": bool(amd_smi_path),
            "rocm_smi": bool(rocm_smi_path),
            "hipconfig": bool(hipconfig_path),
        },
        "kfd_available": kfd_available,
        "install_path": "/opt/rocm" if opt_rocm_exists else None,
    }


def get_hardware_info() -> Dict[str, Any]:
    """Collect comprehensive hardware and environment information.

    Returns:
        Dict[str, Any]: A JSON-serializable dictionary containing:
            - cpu: CPU vendor, model, core counts.
            - memory: System RAM details (visible Linux and physical host RAM).
            - os: Linux distribution, kernel release, architecture.
            - wsl: WSL environment detection and details.
            - gpu: Detected GPU devices (Linux-visible, host devices, virtualized).
            - amd: AMD hardware presence indicator and details.
            - rocm: AMD ROCm tool suite presence and version.
    """
    cpu = get_cpu_info()
    memory = get_memory_info()
    os_info = get_os_info()
    wsl = get_wsl_info()
    gpu = get_gpu_info()
    amd = get_amd_info(cpu, gpu)
    rocm = get_rocm_info()

    return {
        "cpu": cpu,
        "memory": memory,
        "os": os_info,
        "wsl": wsl,
        "gpu": gpu,
        "amd": amd,
        "rocm": rocm,
    }


if __name__ == "__main__":
    info = get_hardware_info()
    print(json.dumps(info, indent=2))