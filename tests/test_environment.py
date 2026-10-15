"""Unit tests for the environment detection module."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.environment import (
    TARGET_PACKAGES,
    get_build_tools_info,
    get_cpu_performance_info,
    get_environment_info,
    get_environment_variables,
    get_os_info,
    get_python_info,
    get_python_packages_info,
    get_rocm_software_info,
    get_tool_paths,
    get_wsl_info,
)


class TestEnvironmentDetection(unittest.TestCase):
    """Test suite for environment detection."""

    def test_get_environment_info_returns_dict(self) -> None:
        """Verify get_environment_info() returns a dictionary."""
        info = get_environment_info()
        self.assertIsInstance(info, dict)

    def test_required_top_level_sections_exist(self) -> None:
        """Verify all required top-level sections exist in the environment info dictionary."""
        info = get_environment_info()
        expected_sections = {
            "python",
            "os",
            "wsl",
            "build_tools",
            "rocm",
            "python_packages",
            "environment_variables",
            "cpu_performance",
            "tool_paths",
        }
        for section in expected_sections:
            self.assertIn(section, info, f"Missing required top-level section: {section}")

    def test_result_is_json_serializable(self) -> None:
        """Verify the returned dictionary can be serialized to JSON without error."""
        info = get_environment_info()
        serialized = json.dumps(info)
        self.assertIsInstance(serialized, str)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized, info)

    def test_python_information_is_valid(self) -> None:
        """Verify Python runtime information fields are populated and valid."""
        py_info = get_python_info()
        self.assertIsInstance(py_info.get("version"), str)
        self.assertTrue(py_info["version"].startswith("3."))
        self.assertIsInstance(py_info.get("implementation"), str)
        self.assertIsInstance(py_info.get("executable"), str)
        self.assertIsInstance(py_info.get("architecture"), str)

    def test_os_information_exists(self) -> None:
        """Verify operating system information is populated."""
        os_info = get_os_info()
        self.assertIsInstance(os_info.get("distribution"), str)
        self.assertGreater(len(os_info["distribution"]), 0)
        self.assertIsInstance(os_info.get("kernel_version"), str)
        self.assertIsInstance(os_info.get("architecture"), str)

    def test_wsl_detection_does_not_crash(self) -> None:
        """Verify WSL detection returns valid status and does not crash."""
        wsl_info = get_wsl_info()
        self.assertIsInstance(wsl_info, dict)
        self.assertIn("is_wsl", wsl_info)
        self.assertIsInstance(wsl_info["is_wsl"], bool)
        self.assertIn("wsl_version", wsl_info)

    def test_missing_external_commands_handled_safely(self) -> None:
        """Verify graceful degradation when optional commands/files are unavailable."""
        with patch("src.environment._safe_run_cmd", return_value=None), \
             patch("src.environment._read_file", return_value=None), \
             patch("shutil.which", return_value=None):
            info = get_environment_info()
            self.assertIsInstance(info, dict)
            # Ensure JSON serialization remains valid
            serialized = json.dumps(info)
            self.assertIsInstance(serialized, str)
            # Verify build tools report available=False
            for tool_data in info["build_tools"].values():
                self.assertFalse(tool_data["available"])
            # Verify rocm reports available=False
            self.assertFalse(info["rocm"]["available"])

    def test_package_detection_does_not_import_packages(self) -> None:
        """Verify AI/ML packages are inspected via metadata without importing them."""
        # Ensure none of the target packages are imported into sys.modules
        for pkg in TARGET_PACKAGES:
            self.assertNotIn(pkg, sys.modules, f"Package {pkg} was unexpectedly imported before test")

        packages_info = get_python_packages_info()
        self.assertIsInstance(packages_info, dict)

        for pkg in TARGET_PACKAGES:
            self.assertIn(pkg, packages_info)
            self.assertIn("installed", packages_info[pkg])
            self.assertIn("version", packages_info[pkg])
            # Ensure calling package detection did not import the package
            self.assertNotIn(pkg, sys.modules, f"Package {pkg} was imported during detection")

        # Test metadata mock behavior
        with patch("importlib.metadata.version", return_value="2.5.0"):
            mock_info = get_python_packages_info()
            self.assertTrue(mock_info["torch"]["installed"])
            self.assertEqual(mock_info["torch"]["version"], "2.5.0")

    def test_environment_collection_does_not_expose_secrets(self) -> None:
        """Verify secrets, passwords, and arbitrary environment variables are never collected."""
        sensitive_vars = {
            "AWS_SECRET_ACCESS_KEY": "super_secret_value",
            "GITHUB_TOKEN": "ghp_secret_token",
            "DATABASE_PASSWORD": "secret_db_pass",
            "API_KEY": "secret_api_key",
            "PRIVATE_KEY": "secret_private_key",
        }
        with patch.dict(os.environ, sensitive_vars):
            collected = get_environment_variables()
            for secret_key in sensitive_vars:
                self.assertNotIn(secret_key, collected, f"Secret variable {secret_key} was collected!")
                for collected_val in collected.values():
                    if collected_val:
                        self.assertNotIn(sensitive_vars[secret_key], collected_val)

    def test_cli_runs_successfully(self) -> None:
        """Verify python -m src.environment executes and returns valid JSON."""
        res = subprocess.run(
            [sys.executable, "-m", "src.environment"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"CLI exited with error: {res.stderr}")
        data = json.loads(res.stdout)
        self.assertIsInstance(data, dict)
        self.assertIn("python", data)
        self.assertIn("os", data)
        self.assertIn("rocm", data)


if __name__ == "__main__":
    unittest.main()