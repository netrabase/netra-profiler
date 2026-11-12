"""
Unified Benchmark Orchestration CLI for Netra Profiler.

Manages dataset generation, environment verification, isolated subprocess
benchmark execution, statistical compilation, golden accuracy verification,
and result persistence.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# isort: split
from benchmarks.generators.base import GeneratorConfig, parse_scale  # noqa: E402
from benchmarks.generators.lofar_telemetry import LofarTelemetryGenerator  # noqa: E402
from benchmarks.harness.collector import run_target_isolated  # noqa: E402
from benchmarks.harness.golden import compare_profiles  # noqa: E402
from benchmarks.harness.system import get_system_fingerprint  # noqa: E402

BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
DATA_DIR = BENCHMARKS_DIR / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
CONFIGS_DIR = BENCHMARKS_DIR / "configs"
RESULTS_DIR = BENCHMARKS_DIR / "results"
HISTORY_DIR = RESULTS_DIR / "history"
GOLDEN_DIR = RESULTS_DIR / "golden"
LATEST_FILE = RESULTS_DIR / "latest.json"


def get_git_commit() -> str:
    """Safely extracts the current short git commit hash."""

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def resolve_synthetic_dataset_path(suite: str, scale_str: str) -> Path:
    """Computes standard deterministic file paths for generated datasets."""
    return SYNTHETIC_DATA_DIR / f"{suite}_{scale_str.upper()}.parquet"


def resolve_config_path(suite: str) -> Path:
    """Locates the Netra YAML config for the requested suite."""

    config_path = CONFIGS_DIR / f"{suite}_netra_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Netra config not found for suite '{suite}' at: {config_path}")
    return config_path


def resolve_golden_path(suite: str, scale_str: str, target: str) -> Path:
    """Computes the path for the golden reference profile JSON."""
    return GOLDEN_DIR / f"golden_{suite}_{scale_str.upper()}_{target}.json"


def handle_generation(suite: str, scale_str: str, chunk_size: int, seed: int, force: bool) -> Path:
    """Coordinates dataset generation or verifies pre-existing files."""

    rows = parse_scale(scale_str)
    dataset_path = resolve_synthetic_dataset_path(suite, scale_str)

    if dataset_path.exists() and not force:
        size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"[i] Dataset exists: {dataset_path} ({size_mb:.2f} MB)")
        print("    (Use --force-generate to overwrite)\n")
        return dataset_path

    if suite == "lofar":
        config = GeneratorConfig(
            total_rows=rows,
            chunk_size=chunk_size,
            seed=seed,
            output_path=dataset_path,
        )
        generator = LofarTelemetryGenerator(config)
        return generator.generate(dataset_path)
    else:
        raise ValueError(f"Unknown benchmark suite: '{suite}'. Available: ['lofar']")


def compute_metric_stats(values: list[float]) -> dict[str, float]:
    """Computes median, mean, standard deviation, min, and max for metric samples."""

    if not values:
        return {"median": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def print_summary_table(
    target: str,
    suite: str,
    scale: str,
    stats: dict[str, dict[str, float]],
    sample_iteration: dict[str, Any],
) -> None:
    """Renders a clean, fixed-width ASCII comparison table of benchmark statistics."""

    title = f"BENCHMARK SUMMARY :: [{target.upper()}] on [{suite.upper()}_{scale.upper()}]"
    print("\n" + "=" * 78)
    print(f" {title.center(76)}")
    print("=" * 78)
    print(f" {'Metric':<25} {'Median':>10} {'Mean':>10} {'StdDev':>10} {'Min / Max':>18}")
    print("-" * 78)

    labels = [
        ("Wall Time (s)", "wall_time_seconds", ".2f"),
        ("Engine Time (s)", "engine_time_seconds", ".2f"),
        ("Peak RSS (MB)", "peak_rss_mb", ".1f"),
        ("Throughput (MB/s)", "throughput_mb_s", ".1f"),
        ("Velocity (M rows/s)", "row_velocity_m_rows_s", ".2f"),
    ]

    for label, key, fmt in labels:
        s = stats[key]
        min_max = f"{s['min']:{fmt}} / {s['max']:{fmt}}"
        std_str = f"±{s['std']:{fmt}}"
        print(
            f" {label:<25} {s['median']:>10{fmt}} {s['mean']:>10{fmt}} {std_str:>10} {min_max:>18}"
        )

    print("-" * 78)
    rows_eval = sample_iteration.get("rows_evaluated", 0)
    cols_prof = sample_iteration.get("columns_profiled", 0)
    anomalies = sample_iteration.get("anomalies_detected", 0)
    crit = sample_iteration.get("critical_alerts", 0)
    warn = sample_iteration.get("warning_alerts", 0)
    info = sample_iteration.get("info_alerts", 0)
    status_str = "PASSED" if sample_iteration.get("passed") else "FAILED"
    exit_code = sample_iteration.get("extra_meta", {}).get("exit_code", 0)

    print(
        f" Rows: {rows_eval:,} | Columns (Flattened): {cols_prof} | "
        f"Alerts: {anomalies} (C:{crit} W:{warn} I:{info})"
    )
    print(f" Quality Gate Status: {status_str} (Exit Code: {exit_code})")
    print("=" * 78 + "\n")


def run_benchmark_for_target(
    target: str,
    suite: str,
    scale: str,
    dataset_file: Path,
    config_file: Path,
    iterations: int,
    save_golden: bool,
    verify_golden: bool,
) -> dict[str, Any]:
    """Executes warmup, measurement iterations, and golden verification for a single target."""

    golden_path = resolve_golden_path(suite, scale, target)
    candidate_profile_path = RESULTS_DIR / f".candidate_{target}.json"

    print(f"\n[*] Running Warmup Pass [{target}]...", end="", flush=True)
    t_warmup = time.perf_counter()
    warmup_res = run_target_isolated(
        target,
        dataset_file,
        config_file,
        dump_profile_path=candidate_profile_path if (save_golden or verify_golden) else None,
    )
    print(
        f" done in {time.perf_counter() - t_warmup:.2f}s "
        f"(Peak RSS: {warmup_res['peak_rss_mb']:.1f} MB)"
    )

    # Handle Golden Profile Save / Verification
    if save_golden and candidate_profile_path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        candidate_profile_path.replace(golden_path)
        print(f"[✔] Saved Golden Reference Profile to: {golden_path.relative_to(PROJECT_ROOT)}")
    elif verify_golden and candidate_profile_path.exists():
        if not golden_path.exists():
            print(f"[!] Golden profile not found at {golden_path}. Run with --save-golden first.")
        else:
            with (
                open(golden_path, encoding="utf-8") as fg,
                open(candidate_profile_path, encoding="utf-8") as fc,
            ):
                diffs = compare_profiles(json.load(fg), json.load(fc))
            if not diffs:
                print("[✔] GOLDEN BASELINE VERIFIED: 100% mathematical accuracy match!")
            else:
                print(f"[✘] GOLDEN BASELINE FAILED ({len(diffs)} discrepancies detected):")
                for d in diffs[:15]:
                    print(f"    - {d}")
        candidate_profile_path.unlink(missing_ok=True)

    # Measurement Iterations
    results: list[dict[str, Any]] = []
    print(f"\n[*] Commencing {iterations} isolated measurement iterations [{target}]:")

    for iteration in range(1, iterations + 1):
        print(f"    -> Iteration {iteration:02d}/{iterations:02d}: ", end="", flush=True)
        res = run_target_isolated(target, dataset_file, config_file)
        results.append(res)
        print(
            f"Wall: {res['wall_time_seconds']:.2f}s | "
            f"Engine: {res['engine_time_seconds']:.2f}s | "
            f"RSS: {res['peak_rss_mb']:.1f} MB | "
            f"Velocity: {res['row_velocity_m_rows_s']:.2f}M rows/s | "
            f"Alerts: {res['anomalies_detected']}"
        )

    metrics_to_aggregate = [
        "wall_time_seconds",
        "engine_time_seconds",
        "peak_rss_mb",
        "throughput_mb_s",
        "row_velocity_m_rows_s",
    ]
    aggregated_stats = {
        metric: compute_metric_stats([float(r[metric]) for r in results if r[metric] is not None])
        for metric in metrics_to_aggregate
    }

    print_summary_table(
        target=target,
        suite=suite,
        scale=scale,
        stats=aggregated_stats,
        sample_iteration=results[0],
    )

    return {
        "statistics": aggregated_stats,
        "raw_iterations": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="netra-profiler-benchmark",
        description="Netra Profiler Benchmark Suite Orchestrator.",
    )
    parser.add_argument("--suite", "-s", type=str, default="lofar", choices=["lofar"])
    parser.add_argument("--scale", type=str, default="10M")
    parser.add_argument("--chunk-size", type=str, default="2M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="netra",
        choices=["netra", "netra-low-memory", "all"],
        help="Target profiler mode to evaluate (default: netra).",
    )
    parser.add_argument("--iterations", "-i", type=int, default=5)
    parser.add_argument(
        "--save-golden",
        action="store_true",
        help="Save the profile output as the golden mathematical baseline.",
    )
    parser.add_argument(
        "--verify-golden",
        action="store_true",
        help="Verify that the profile output matches the saved golden baseline within 1e-6.",
    )

    args = parser.parse_args()
    chunk_rows = parse_scale(args.chunk_size)

    print("=" * 78)
    print(" NETRA PROFILER BENCHMARK HARNESS")
    print(f" Suite: [{args.suite.upper()}]  |  Scale: [{args.scale.upper()}]")
    print("=" * 78)

    t0 = time.perf_counter()
    dataset_file = handle_generation(
        suite=args.suite,
        scale_str=args.scale,
        chunk_size=chunk_rows,
        seed=args.seed,
        force=args.force_generate,
    )

    if args.generate_only:
        print(f"[✔] Dataset prepared at: {dataset_file}")
        print(f"[✔] Total prep duration: {time.perf_counter() - t0:.2f}s")
        return

    config_file = resolve_config_path(args.suite)
    system_fingerprint = get_system_fingerprint()

    print(
        f"[+] Dataset:  {dataset_file.name} ({dataset_file.stat().st_size / (1024 * 1024):.2f} MB)"
    )
    print(f"[+] Config:   {config_file.name}")
    print(f"[+] CPU:      {system_fingerprint['cpu_model']} ({system_fingerprint['cpu_cores']})")
    print(f"[+] Memory:   {system_fingerprint['ram_total_gb']} GB RAM")

    targets = ["netra", "netra-low-memory"] if args.target == "all" else [args.target]
    target_payloads: dict[str, Any] = {}

    for target in targets:
        target_payloads[target] = run_benchmark_for_target(
            target=target,
            suite=args.suite,
            scale=args.scale,
            dataset_file=dataset_file,
            config_file=config_file,
            iterations=args.iterations,
            save_golden=args.save_golden,
            verify_golden=args.verify_golden,
        )

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_utc = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_commit()

    first_target_results = target_payloads[targets[0]]["raw_iterations"]
    run_payload = {
        "metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "git_commit": git_hash,
            "suite": args.suite,
            "scale": args.scale,
            "targets": targets,
            "iterations": args.iterations,
        },
        "system": system_fingerprint,
        "dataset": {
            "path": str(dataset_file),
            "size_bytes": dataset_file.stat().st_size,
            "rows": first_target_results[0].get("rows_evaluated", 0),
            "columns_profiled": first_target_results[0].get("columns_profiled", 0),
        },
        "config": str(config_file),
        "results": target_payloads,
    }

    history_file = HISTORY_DIR / f"run_{timestamp_utc}_{git_hash}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)

    with open(LATEST_FILE, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)

    print(f"[✔] Benchmark run saved to:    {history_file.relative_to(PROJECT_ROOT)}")
    print(f"[✔] Latest metrics linked to: {LATEST_FILE.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
