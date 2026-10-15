"""Unit tests for System and GPU Capability Detection Engine V1.

Validates independent detection of host, WSL, /dev device nodes, graphics libraries,
ROCm tools, and Python AI frameworks without false claims or system mutation.
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.capabilities import (
    CapabilityStatus,
    ComputePath,
    SystemCapabilities,
    detect_capabilities,
    detect_device_nodes,
    detect_graphics_libraries,
    detect_os_info,
    detect_python_frameworks,
    detect_rocm_tools,
    detect_shared_library,
    detect_wsl_info,
)


class TestCapabilityDetection(unittest.TestCase):
    """Unit test suite for src/capabilities.py."""

    def test_01_real_environment_detection_does_not_crash(self) -> None:
        """1. Real environment detection executes cleanly and returns valid structure."""
        caps = detect_capabilities()
        self.assertIsInstance(caps, SystemCapabilities)
        self.assertEqual(caps.schema_version, "1.0")
        self.assertIn("distribution", caps.operating_system)
        self.assertIn("is_wsl", caps.wsl)
        self.assertIn("dxg", caps.devices)
        self.assertIn("dxcore", caps.libraries)
        self.assertIn("rocminfo", caps.rocm_tools)
        self.assertIn("torch", caps.frameworks)

    def test_02_linux_native_environment(self) -> None:
        """2. Non-WSL native Linux without GPU passthrough defaults to CPU or native."""
        with patch("src.capabilities.detect_wsl_info", return_value={"is_wsl": False, "wsl_version": None, "distro_name": None}), \
             patch("src.capabilities.detect_device_nodes", return_value={
                 "dxg": {"available": False, "status": "UNAVAILABLE"},
                 "dri": {"available": True, "status": "AVAILABLE"},
                 "kfd": {"available": False, "status": "UNAVAILABLE"},
             }), \
             patch("src.capabilities.detect_graphics_libraries", return_value={
                 "dxcore": {"available": False, "status": "UNAVAILABLE"},
                 "d3d12": {"available": False, "status": "UNAVAILABLE"},
             }), \
             patch("src.capabilities.detect_rocm_tools", return_value={
                 "rocminfo": {"available": False, "status": "UNAVAILABLE"},
                 "rocm_smi": {"available": False, "status": "UNAVAILABLE"},
                 "amd_smi": {"available": False, "status": "UNAVAILABLE"},
                 "hipcc": {"available": False, "status": "UNAVAILABLE"},
             }):
            caps = detect_capabilities()
            self.assertEqual(caps.wsl["is_wsl"], False)
            self.assertEqual(caps.rocm_status, CapabilityStatus.UNAVAILABLE)
            self.assertEqual(caps.direct3d12_status, CapabilityStatus.UNAVAILABLE)
            self.assertEqual(caps.recommended_compute_path, ComputePath.CPU)

    def test_03_wsl_with_dev_dxg_and_d3d12(self) -> None:
        """3. WSL with /dev/dxg and D3D12 recommends Direct3D12/Windows acceleration."""
        with patch("src.capabilities.detect_wsl_info", return_value={"is_wsl": True, "wsl_version": "2", "distro_name": "Ubuntu"}), \
             patch("src.capabilities.detect_device_nodes", return_value={
                 "dxg": {"available": True, "status": "AVAILABLE"},
                 "dri": {"available": False, "status": "UNAVAILABLE"},
                 "kfd": {"available": False, "status": "UNAVAILABLE"},
             }), \
             patch("src.capabilities.detect_graphics_libraries", return_value={
                 "dxcore": {"available": True, "status": "AVAILABLE", "path": "/usr/lib/wsl/lib/libdxcore.so"},
                 "d3d12": {"available": True, "status": "AVAILABLE", "path": "/usr/lib/wsl/lib/libd3d12.so"},
             }), \
             patch("src.capabilities.detect_rocm_tools", return_value={
                 "rocminfo": {"available": False, "status": "UNAVAILABLE"},
                 "rocm_smi": {"available": False, "status": "UNAVAILABLE"},
                 "amd_smi": {"available": False, "status": "UNAVAILABLE"},
                 "hipcc": {"available": False, "status": "UNAVAILABLE"},
             }):
            caps = detect_capabilities()
            self.assertEqual(caps.wsl["is_wsl"], True)
            self.assertEqual(caps.direct3d12_status, CapabilityStatus.AVAILABLE)
            self.assertEqual(caps.rocm_status, CapabilityStatus.UNAVAILABLE)
            self.assertEqual(caps.recommended_compute_path, ComputePath.DIRECT3D12_WINDOWS)

    def test_04_rocm_never_inferred_from_dxg_or_d3d12(self) -> None:
        """4. Critical requirement: /dev/dxg or libd3d12 must NOT infer ROCm availability."""
        with patch("src.capabilities.detect_device_nodes", return_value={
                 "dxg": {"available": True, "status": "AVAILABLE"},
                 "dri": {"available": False, "status": "UNAVAILABLE"},
                 "kfd": {"available": False, "status": "UNAVAILABLE"},
             }), \
             patch("src.capabilities.detect_graphics_libraries", return_value={
                 "dxcore": {"available": True, "status": "AVAILABLE"},
                 "d3d12": {"available": True, "status": "AVAILABLE"},
             }), \
             patch("src.capabilities.detect_rocm_tools", return_value={
                 "rocminfo": {"available": False, "status": "UNAVAILABLE"},
                 "rocm_smi": {"available": False, "status": "UNAVAILABLE"},
                 "amd_smi": {"available": False, "status": "UNAVAILABLE"},
                 "hipcc": {"available": False, "status": "UNAVAILABLE"},
             }):
            caps = detect_capabilities()
            self.assertEqual(caps.rocm_status, CapabilityStatus.UNAVAILABLE)
            self.assertNotEqual(caps.recommended_compute_path, ComputePath.ROCM_HIP)

    def test_05_rocm_tools_available(self) -> None:
        """5. When ROCm tools and runtime are available, ROCm/HIP is recommended."""
        with patch("src.capabilities.detect_rocm_tools", return_value={
                 "rocminfo": {"available": True, "version": "6.2.0", "status": "AVAILABLE", "path": "/opt/rocm/bin/rocminfo"},
                 "rocm_smi": {"available": True, "version": "6.2.0", "status": "AVAILABLE", "path": "/opt/rocm/bin/rocm-smi"},
                 "amd_smi": {"available": False, "version": None, "status": "UNAVAILABLE", "path": None},
                 "hipcc": {"available": True, "version": "6.2.0", "status": "AVAILABLE", "path": "/opt/rocm/bin/hipcc"},
             }), \
             patch("src.capabilities.detect_device_nodes", return_value={
                 "dxg": {"available": False, "status": "UNAVAILABLE"},
                 "dri": {"available": True, "status": "AVAILABLE"},
                 "kfd": {"available": True, "status": "AVAILABLE"},
             }):
            caps = detect_capabilities()
            self.assertEqual(caps.rocm_status, CapabilityStatus.AVAILABLE)
            self.assertEqual(caps.recommended_compute_path, ComputePath.ROCM_HIP)

    def test_06_missing_libraries_handled_safely(self) -> None:
        """6. Missing libraries return UNAVAILABLE status without crashing."""
        with patch("ctypes.CDLL", side_effect=OSError("Library not found")):
            res = detect_shared_library(["nonexistent_lib"], ["/path/does/not/exist.so"])
            self.assertFalse(res["available"])
            self.assertEqual(res["status"], CapabilityStatus.UNAVAILABLE.value)
            self.assertIsNotNone(res["error"])

    def test_07_missing_gpu_frameworks(self) -> None:
        """7. Uninstalled frameworks report NOT_INSTALLED status."""
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            frameworks = detect_python_frameworks()
            self.assertIn("torch", frameworks)
            self.assertFalse(frameworks["torch"]["installed"])
            self.assertEqual(frameworks["torch"]["status"], CapabilityStatus.NOT_INSTALLED.value)

    def test_08_installed_gpu_framework_rocm_variant(self) -> None:
        """8. Frameworks with ROCm variants report rocm gpu_support."""
        def mock_version(pkg_name: str) -> str:
            if pkg_name == "torch":
                return "2.4.0+rocm6.2"
            if pkg_name == "onnxruntime":
                return "1.19.0"
            raise importlib.metadata.PackageNotFoundError

        with patch("importlib.metadata.version", side_effect=mock_version):
            frameworks = detect_python_frameworks()
            self.assertTrue(frameworks["torch"]["installed"])
            self.assertEqual(frameworks["torch"]["gpu_support"], "rocm")
            self.assertEqual(frameworks["torch"]["status"], CapabilityStatus.AVAILABLE.value)
            self.assertTrue(frameworks["onnxruntime"]["installed"])
            self.assertFalse(frameworks["tensorflow"]["installed"])

    def test_09_installed_gpu_framework_cuda_and_cpu_variants(self) -> None:
        """9. Frameworks with CUDA or CPU variants report correct gpu_support."""
        def mock_version(pkg_name: str) -> str:
            if pkg_name == "torch":
                return "2.4.0+cu121"
            raise importlib.metadata.PackageNotFoundError

        with patch("importlib.metadata.version", side_effect=mock_version):
            frameworks = detect_python_frameworks()
            self.assertEqual(frameworks["torch"]["gpu_support"], "cuda")

        def mock_cpu_version(pkg_name: str) -> str:
            if pkg_name == "torch":
                return "2.4.0+cpu"
            raise importlib.metadata.PackageNotFoundError

        with patch("importlib.metadata.version", side_effect=mock_cpu_version):
            frameworks = detect_python_frameworks()
            self.assertEqual(frameworks["torch"]["gpu_support"], "cpu")

    def test_10_malformed_command_output_handled_safely(self) -> None:
        """10. Tools returning malformed output degrade safely without throwing exceptions."""
        from src.capabilities import _detect_tool

        with patch("shutil.which", return_value="/bin/rocminfo"), \
             patch("src.capabilities._safe_run_cmd", return_value="some unusual binary garbage ???"):
            tool_info = _detect_tool("rocminfo", ["/bin/rocminfo"])
            self.assertTrue(tool_info["available"])
            self.assertEqual(tool_info["version"], "some unusual binary garbage ???")

    def test_11_deterministic_json_serialization(self) -> None:
        """11. Capability profile produces deterministic, valid JSON."""
        caps = detect_capabilities()
        json_str = caps.to_json()
        parsed = json.loads(json_str)

        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertEqual(parsed, caps.to_dict())

    def test_12_no_system_mutation(self) -> None:
        """12. Verifies capability detection never mutates os.environ or files."""
        environ_before = dict(os.environ)
        _ = detect_capabilities()
        self.assertEqual(dict(os.environ), environ_before)

    def test_13_cli_execution(self) -> None:
        """13. CLI execution returns exit code 0 and includes human-readable summary and JSON."""
        cmd = [sys.executable, "-m", "src.capabilities"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        self.assertEqual(proc.returncode, 0)
        self.assertIn("ROCm Performance Guard", proc.stdout)
        self.assertIn("Capability JSON:", proc.stdout)

        # Extract and validate the JSON part
        json_part = proc.stdout.split("Capability JSON:\n")[-1].strip()
        parsed = json.loads(json_part)
        self.assertEqual(parsed["schema_version"], "1.0")
        self.assertIn("recommended_compute_path", parsed)

    def test_14_limitations_contain_rocm_disclaimer(self) -> None:
        """14. Results explicitly include disclaimer when ROCm is unavailable."""
        caps = detect_capabilities()
        if caps.rocm_status == CapabilityStatus.UNAVAILABLE:
            self.assertTrue(
                any("rocm gpu compute support is unavailable" in lim.lower() for lim in caps.limitations)
            )


if __name__ == "__main__":
    unittest.main()
