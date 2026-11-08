"""
Isolated Subprocess Telemetry Collector for Netra Profiler.

Guarantees pristine memory tracking by executing every benchmark iteration
inside a freshly spawned Python subprocess, preventing allocator retention
and cross-iteration page-cache pollution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))


def get_peak_rss_mb() -> float:
    """
    Extracts the true OS-level Peak Resident Set Size (RSS) for the current process.
    Handles unit differences between Linux (Kilobytes), macOS (Bytes), and Windows.
    """

    if sys.platform == "win32":
        import psutil  # type: ignore # noqa: PLC0415

        return psutil.Process(os.getpid()).memory_info().peak_wset / (1024 * 1024)

    import resource  # noqa: PLC0415

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return rusage.ru_maxrss / (1024 * 1024)
    return rusage.ru_maxrss / 1024


def run_target_isolated(
    target_name: str,
    dataset_path: Path,
    config_path: Path,
    dump_profile_path: Path | None = None,
) -> dict[str, Any]:
    """
    Spawns a fresh Python subprocess to execute the benchmark target.
    """

    cmd = [
        sys.executable,
        __file__,
        "--worker-mode",
        "--target",
        target_name,
        "--dataset",
        str(dataset_path),
        "--config",
        str(config_path),
    ]
    if dump_profile_path is not None:
        cmd.extend(["--dump-profile", str(dump_profile_path)])

    worker_env = os.environ.copy()
    worker_env["POLARS_FORCE_OOC"] = "1"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PROJECT_ROOT),
            env=worker_env,
        )

        output_lines = result.stdout.strip().split("\n")
        json_payload = output_lines[-1]
        return json.loads(json_payload)

    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL] Subprocess Worker Failed (Exit {e.returncode})")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise RuntimeError(f"Benchmark worker for '{target_name}' crashed.") from e


def _execute_worker_mode(
    target_name: str,
    dataset_path: Path,
    config_path: Path,
    dump_profile_path: Path | None = None,
) -> None:
    """Executes the target adapter, extracts OS telemetry, and dumps JSON."""

    from benchmarks.targets.netra import NetraProfilerBenchmarkTarget # noqa: PLC0415

    if target_name == "netra":
        target_adapter = NetraProfilerBenchmarkTarget(low_memory=False)
    elif target_name == "netra-low-memory":
        target_adapter = NetraProfilerBenchmarkTarget(low_memory=True)
    else:
        raise ValueError(f"Unknown benchmark target adapter: '{target_name}'")

    target_adapter.setup(dataset_path, config_path)

    try:
        result_obj, profile_dict = target_adapter.run_profiling()

        result_dict = result_obj.__dict__.copy()
        result_dict["peak_rss_mb"] = round(get_peak_rss_mb(), 2)

        if dump_profile_path is not None:
            dump_profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dump_profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_dict, f, indent=2)

        print(json.dumps(result_dict))
    finally:
        target_adapter.teardown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated Profiler Benchmark Worker")
    parser.add_argument("--worker-mode", action="store_true", required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dump-profile", type=Path, default=None)
    args = parser.parse_args()

    _execute_worker_mode(args.target, args.dataset, args.config, args.dump_profile)
