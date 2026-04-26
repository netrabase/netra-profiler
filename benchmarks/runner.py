import argparse
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path

import polars as pl

MILLION = 1_000_000
THOUSAND = 1_000
MASSIVE_MULTIPLIER = 1


def get_row_count(data_dir: Path, scale: str) -> str:
    """Reads Parquet metadata to instantly get the total row count."""

    try:
        count = 0

        for file_path in data_dir.glob("*.parquet"):
            # This only reads the metadata footer
            count += pl.scan_parquet(file_path).select(pl.len()).collect().item()

        # Account for our duplication strategy on the massive run
        if scale == "massive":
            count *= MASSIVE_MULTIPLIER

        if count >= MILLION:
            return f"~{count / 1_000_000:.1f} Million Rows"
        elif count >= THOUSAND:
            return f"~{count / 1_000:.1f}K Rows"
        return f"{count} Rows"
    except Exception:
        return "Unknown Rows"


def run_worker(tool: str, data_dir: str, mode: str) -> tuple[float | None, float | None]:
    """Spawns an isolated worker process and returns time (s) and peak RAM usage (MB)."""

    cmd = [sys.executable, "benchmarks/worker.py", tool, data_dir, mode]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or result.stdout.startswith("ERROR"):
        # This usually means the tool crashed via OOM
        return None, None

    try:
        time_str, ram_str = result.stdout.strip().split(",")
        return float(time_str), float(ram_str)
    except ValueError:
        return None, None


def execute_matrix(tool: str, data_dir: str, mode: str, runs: int) -> str:
    """Runs the warmup and execution loop, returning a formatted result string."""

    print(f"    Running {tool} ({mode}) - Warmup...", end="", flush=True)
    warmup_time, warmup_ram = run_worker(tool, data_dir, mode)

    if warmup_time is None or warmup_ram is None:
        print(" [CRASHED/OOM]")
        return "CRASH / OOM"

    print(" [DONE]")

    times, rams = [], []
    for i in range(1, runs + 1):
        print(f"    Running {tool} ({mode}) - Run {i}/{runs}...", end="", flush=True)
        time_s, ram_mb = run_worker(tool, data_dir, mode)

        if time_s is None or ram_mb is None:
            print(" [CRASHED/OOM]")
            return "CRASH / OOM"

        times.append(time_s)
        rams.append(ram_mb)
        print(f" [{time_s:.2f}s, {ram_mb / 1024:.2f} GB]")

    avg_time = statistics.mean(times)
    avg_ram_gb = statistics.mean(rams) / 1024

    return f"**{avg_time:.2f}s** ({avg_ram_gb:.1f} GB RAM)"


def main():
    parser = argparse.ArgumentParser(description="Netra Benchmark Execution Orchestrator")
    parser.add_argument("--dataset", type=str, default="nyc_taxi")
    parser.add_argument(
        "--scale", type=str, choices=["small", "medium", "massive"], default="small"
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs to average")
    parser.add_argument("--data-root", type=str, default="./benchmarks/data")
    parser.add_argument("--output", type=str, default="./benchmarks/benchmark_results.md")
    parser.add_argument(
        "--tool",
        type=str,
        choices=["both", "netra", "ydata"],
        default="both",
        help="Specify a single tool to run, or 'both' (default).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset / args.scale
    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir} does not exist. Run fetch_data.py first.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("NETRA BENCHMARK SUITE")
    print("=" * 50)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Scale: {args.scale.upper()}")

    netra_std = "Skipped"
    netra_low = "Skipped"
    ydata_std = "Skipped"
    ydata_min = "Skipped"

    print("\n" + "-" * 50)
    print("Running benchmarks...\n")

    if args.tool in ["both", "netra"]:
        print("[*] Netra Profiler (Standard)")
        netra_std = execute_matrix("netra", str(data_dir), "standard", args.runs)

        print("\n[*] Netra Profiler (Low Memory)")
        netra_low = execute_matrix("netra", str(data_dir), "low_memory", args.runs)

    if args.tool in ["both", "ydata"] and args.scale != "massive":
        print("\n[*] YData-Profiling (Standard)")
        ydata_std = execute_matrix("ydata", str(data_dir), "standard", args.runs)

        print("\n[*] YData-Profiling (Minimal)")
        ydata_min = execute_matrix("ydata", str(data_dir), "minimal", args.runs)

    print("\nBenchmarks completed.")
    print("-" * 50 + "\n")

    # Calculate dataset size
    dir_size_bytes = sum(f.stat().st_size for f in data_dir.glob("**/*") if f.is_file())
    dir_size_gb = dir_size_bytes / (1024**3)

    # Calculate massive multiplier for disk size
    if args.scale == "massive":
        dir_size_gb *= MASSIVE_MULTIPLIER

    formatted_rows = get_row_count(data_dir, args.scale)

    # Generate the Human Readable Table
    result = textwrap.dedent(f"""\
        Dataset: {args.dataset.upper()}
        Size: {dir_size_gb:.2f}GB ({formatted_rows})

        Results (Averaged over {args.runs} runs):

        | Execution Mode | netra-profiler (Parquet) | ydata-profiling (Parquet) |
        | :--- | :--- | :--- |
        | Standard | {netra_std} | {ydata_std} |
        | Low Memory / Minimal | {netra_low} | {ydata_min} |
    """)

    print(result)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"Results saved to {args.output}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
