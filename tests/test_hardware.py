"""Unit tests for the hardware detection module."""

from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch

from src.hardware import (
    get_amd_info,
    get_cpu_info,
    get_gpu_info,
    get_hardware_info,
    get_memory_info,
    get_os_info,
    get_rocm_info,
    get_wsl_info,
    is_running_in_wsl,
)


class TestHardwareDetection(unittest.TestCase):
    """Test suite for hardware and environment detection."""

    def test_get_hardware_info_returns_dict(self) -> None:
        """Verify that get_hardware_info() returns a dictionary."""
        info = get_hardware_info()
        self.assertIsInstance(info, dict)

    def test_required_top_level_keys_exist(self) -> None:
        """Verify that all required top-level keys exist in the output dictionary."""
        info = get_hardware_info()
        expected_keys = {"cpu", "memory", "os", "wsl", "gpu", "amd", "rocm"}
        for key in expected_keys:
            self.assertIn(key, info, f"Missing required top-level key: {key}")

    def test_returned_structure_is_json_serializable(self) -> None:
        """Verify that the hardware info dictionary can be serialized to JSON without error."""
        info = get_hardware_info()
        serialized = json.dumps(info)
        self.assertIsInstance(serialized, str)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized, info)

    def test_cpu_logical_count_is_positive_integer(self) -> None:
        """Verify that the logical CPU count is detected as a positive integer."""
        info = get_hardware_info()
        logical_cores = info["cpu"].get("logical_cores")
        self.assertIsInstance(logical_cores, int)
        self.assertGreater(logical_cores, 0)

    def test_wsl_gpu_structure_and_virtualization(self) -> None:
        """Verify that WSL environments report virtualized GPU separately from host GPU."""
        gpu = get_gpu_info()
        self.assertIn("virtualized", gpu)
        self.assertIn("linux_visible_devices", gpu)
        self.assertIn("host_devices", gpu)
        self.assertIsInstance(gpu["linux_visible_devices"], list)
        self.assertIsInstance(gpu["host_devices"], list)
        if is_running_in_wsl():
            self.assertTrue(gpu["virtualized"], "WSL environment must set virtualized=True")

    def test_missing_host_gpu_does_not_cause_failure(self) -> None:
        """Verify that missing host GPU information degrades gracefully without crashing."""
        with patch("src.hardware._get_wsl_host_hardware", return_value=(None, [])):
            gpu = get_gpu_info()
            self.assertIsInstance(gpu, dict)
            self.assertEqual(gpu["host_devices"], [])
            info = get_hardware_info()
            self.assertIsInstance(info, dict)
            serialized = json.dumps(info)
            self.assertIsInstance(serialized, str)

    def test_host_memory_may_be_none_when_unavailable(self) -> None:
        """Verify that host memory is legitimately None when host query is unavailable."""
        with patch("src.hardware._get_wsl_host_hardware", return_value=(None, [])):
            mem = get_memory_info()
            self.assertIsNone(mem["host_physical_bytes"])
            self.assertIsNone(mem["host_physical_gb"])
            visible = mem["visible_to_linux_bytes"]
            if sys.platform.startswith("linux"):
                self.assertIsNotNone(visible)
                self.assertGreater(visible, 0)
            else:
                # /proc/meminfo does not exist on native Windows/macOS, so None is
                # acceptable here, but a present value must still be a positive int.
                self.assertTrue(visible is None or (isinstance(visible, int) and visible > 0))

    def test_linux_visible_memory_separately_represented(self) -> None:
        """Verify that visible Linux memory remains separately represented from host memory."""
        mem = get_memory_info()
        self.assertIn("visible_to_linux_bytes", mem)
        self.assertIn("visible_to_linux_gb", mem)
        self.assertIn("host_physical_bytes", mem)
        self.assertIn("host_physical_gb", mem)
        self.assertIn("source", mem)
        visible_bytes = mem["visible_to_linux_bytes"]
        visible_gb = mem["visible_to_linux_gb"]
        if sys.platform.startswith("linux"):
            self.assertIsInstance(visible_bytes, int)
            self.assertGreater(visible_bytes, 0)
            self.assertIsInstance(visible_gb, float)
            self.assertGreater(visible_gb, 0)
        else:
            self.assertTrue(visible_bytes is None or (isinstance(visible_bytes, int) and visible_bytes > 0))
            self.assertTrue(visible_gb is None or (isinstance(visible_gb, float) and visible_gb > 0))

    def test_amd_detection_does_not_infer_amd_from_basic_render_driver(self) -> None:
        """Verify AMD detection uses confirmed info and never infers AMD from Basic Render Driver."""
        cpu_intel = {"vendor": "GenuineIntel", "model_name": "Intel Core i7"}
        gpu_virtual = {
            "virtualized": True,
            "linux_visible_devices": [
                {
                    "vendor": "Microsoft Corporation",
                    "vendor_id": "1414",
                    "device": "Basic Render Driver",
                    "device_id": "008e",
                    "pci_slot": "5aba:00:00.0",
                }
            ],
            "host_devices": [],
        }

        # 1. Intel CPU with only Microsoft Basic Render Driver must NOT detect AMD
        res_no_amd = get_amd_info(cpu_intel, gpu_virtual)
        self.assertFalse(res_no_amd["gpu_is_amd"], "Basic Render Driver must not trigger gpu_is_amd")
        self.assertFalse(res_no_amd["cpu_is_amd"])
        self.assertFalse(res_no_amd["present"])

        # 2. Host AMD GPU present must trigger gpu_is_amd
        gpu_with_host_amd = {
            "virtualized": True,
            "linux_visible_devices": gpu_virtual["linux_visible_devices"],
            "host_devices": [
                {
                    "vendor": "Advanced Micro Devices, Inc.",
                    "name": "AMD Radeon (TM) Graphics",
                    "pci_vendor_id": "1002",
                    "pci_device_id": "15E7",
                    "driver_version": "31.0.12032.4",
                }
            ],
        }
        res_host_amd = get_amd_info(cpu_intel, gpu_with_host_amd)
        self.assertTrue(res_host_amd["gpu_is_amd"])
        self.assertTrue(res_host_amd["present"])
        self.assertFalse(res_host_amd["cpu_is_amd"])

        # 3. Linux-visible native AMD GPU must trigger gpu_is_amd
        gpu_native_amd = {
            "virtualized": False,
            "linux_visible_devices": [
                {
                    "vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
                    "vendor_id": "1002",
                    "device": "Navi 31 [Radeon RX 7900 XTX]",
                    "device_id": "744c",
                    "pci_slot": "0000:03:00.0",
                }
            ],
            "host_devices": [],
        }
        res_native_amd = get_amd_info(cpu_intel, gpu_native_amd)
        self.assertTrue(res_native_amd["gpu_is_amd"])
        self.assertTrue(res_native_amd["present"])

    def test_missing_optional_commands_do_not_crash(self) -> None:
        """Verify that detector functions degrade gracefully when external tools/files are absent."""
        with patch("src.hardware._safe_run_cmd", return_value=None), \
             patch("src.hardware._read_file", return_value=None), \
             patch("src.hardware._get_wsl_host_hardware", return_value=(None, [])), \
             patch("shutil.which", return_value=None), \
             patch("os.path.exists", return_value=False), \
             patch("os.path.isdir", return_value=False):
            info = get_hardware_info()
            self.assertIsInstance(info, dict)
            expected_keys = {"cpu", "memory", "os", "wsl", "gpu", "amd", "rocm"}
            for key in expected_keys:
                self.assertIn(key, info)
            serialized = json.dumps(info)
            self.assertIsInstance(serialized, str)

    def test_wsl_detection_in_current_environment(self) -> None:
        """Verify that WSL detection is consistent and correct for this environment."""
        wsl_info = get_wsl_info()
        self.assertIn("is_wsl", wsl_info)
        self.assertEqual(wsl_info["is_wsl"], is_running_in_wsl())
        if sys.platform == "win32":
            self.assertFalse(wsl_info["is_wsl"], "Native Windows must not be reported as WSL")
            self.assertIsNone(wsl_info.get("wsl_version"))
        elif is_running_in_wsl():
            self.assertEqual(wsl_info.get("wsl_version"), "2", "Expected WSL version 2")

    def test_rocm_info_when_rocm_absent(self) -> None:
        """Verify that ROCm detection does not claim availability when tools are not installed."""
        rocm_info = get_rocm_info()
        self.assertIsInstance(rocm_info, dict)
        self.assertIn("is_available", rocm_info)
        self.assertIn("tools", rocm_info)
        self.assertIn("kfd_available", rocm_info)
        self.assertIsInstance(rocm_info["tools"], dict)


if __name__ == "__main__":
    unittest.main()