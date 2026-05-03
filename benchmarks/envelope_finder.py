import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl


def get_dir_metrics(data_dir: Path) -> tuple[float, int]:
    """Returns (Size_in_MB, Row_Count) of all Parquet files in a directory."""
    files = list(data_dir.glob("*.parquet"))
    if not files:
        return 0.0, 0

    size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

    total_rows = 0
    try:
        # The safe loop: counts rows per file without merging schemas
        for file_path in files:
            total_rows += pl.scan_parquet(file_path).select(pl.len()).collect().item()
    except Exception as e:
        print(f"  [Warning: Row count failed - {e}]")

    return size_mb, total_rows


def run_worker(tool: str, data_dir: str, mode: str) -> bool:
    """Runs the worker. Returns True if successful, False if it crashed/OOM'd."""
    cmd = [sys.executable, "benchmarks/worker.py", tool, data_dir, mode]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0 or result.stdout.startswith("ERROR"):
        return False

    try:
        # Check if it successfully printed the time,ram string
        _ = result.stdout.strip().split(",")
        return True
    except ValueError:
        return False


def find_envelope(tool: str, mode: str, source_files: list[Path], temp_dir: Path) -> None:
    print(f"\nFinding Data Envelope for {tool.upper()} ({mode})")
    print("-" * 50)

    last_success_mb = 0.0
    last_success_rows = 0

    # We test in increments of months
    # months_progression = [1, 2, 4, 8, 12, 18, 24, 36, 48, 60, 84]
    months_progression = [53]
    max_file_count = len(source_files)

    for months in months_progression:
        file_count = min(months, max_file_count)

        # 1. Clear temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        # 2. Copy 'count' files to temp dir
        test_files = source_files[:file_count]
        for f in test_files:
            shutil.copy(f, temp_dir / f.name)

        # 3. Calculate current payload
        mb, rows = get_dir_metrics(temp_dir)
        print(f"Testing {file_count} months (~{mb:.1f} MB | {rows:,} rows)... ", end="", flush=True)

        # 4. Run the worker
        success = run_worker(tool, str(temp_dir), mode)

        if success:
            print("Survived")
            last_success_mb = mb
            last_success_rows = rows
        else:
            print("CRASHED / OOM)")
            print("-" * 50)
            print(f"MAXIMUM SAFE ENVELOPE: {last_success_mb:.1f} MB ({last_success_rows:,} rows)")
            return

        if file_count == len(source_files):
            break

    print("-" * 50)
    print(f"SURVIVED ENTIRE DATASET: {last_success_mb:.1f} MB ({last_success_rows:,} rows)")


def main() -> None:
    source_dir = Path("./benchmarks/data/nyc_taxi/massive")
    temp_dir = Path("./benchmarks/data/temp_envelope")

    if not source_dir.exists():
        print("ERROR: Run fetch_data.py on massive scale first to get source files.")
        sys.exit(1)

    # Sort files so we add them chronologically
    source_files = sorted(list(source_dir.glob("*.parquet")))

    try:
        # find_envelope("ydata", "standard", source_files, temp_dir)
        # find_envelope("ydata", "minimal", source_files, temp_dir)
        find_envelope("netra", "standard", source_files, temp_dir)
        find_envelope("netra", "low_memory", source_files, temp_dir)

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
