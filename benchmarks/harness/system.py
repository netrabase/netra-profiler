"""
Hardware and Environment Auto-Fingerprinting for Netra Profiler.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

import psutil


def _get_package_version(package_name: str) -> str:
    """Safely extracts a package version, falling back to 'unknown' if missing."""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def get_system_fingerprint() -> dict[str, Any]:
    """
    Collects a comprehensive snapshot of the host hardware and runtime environment.
    """

    cpu_cores_physical = psutil.cpu_count(logical=False) or 0
    cpu_cores_logical = psutil.cpu_count(logical=True) or 0

    try:
        if platform.system() == "Linux":
            output = subprocess.check_output("lscpu", text=True)
            cpu_model = next(
                (line.split(":")[1].strip() for line in output.split("\n") if "Model name" in line),
                platform.processor(),
            )
        elif platform.system() == "Darwin":
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        else:
            cpu_model = platform.processor()
    except Exception:
        cpu_model = platform.processor()

    ram_total_bytes = psutil.virtual_memory().total
    ram_total_gb = round(ram_total_bytes / (1024**3), 2)

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu_model": cpu_model,
        "cpu_cores": f"{cpu_cores_physical} Physical / {cpu_cores_logical} Logical",
        "ram_total_gb": ram_total_gb,
        "python_version": sys.version.split()[0],
        "versions": {
            "netra_profiler": _get_package_version("netra-profiler"),
            "polars": _get_package_version("polars"),
            "numpy": _get_package_version("numpy"),
            "pyarrow": _get_package_version("pyarrow"),
            "pyyaml": _get_package_version("PyYAML"),
        },
    }
