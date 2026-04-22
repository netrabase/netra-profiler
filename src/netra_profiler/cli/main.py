"""
Netra Profiler CLI Entry Point.

This module orchestrates the user-facing Command Line Interface (CLI).
It enforces a strict separation of concerns by handling all Operating System interactions
(File I/O, RAM usage polling), User Interface rendering (Rich console updates), and
argument parsing (Typer), while keeping the core `Profiler` and `engine` purely mathematical.
"""

import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import typer
import yaml

from netra_profiler import Profiler, __version__
from netra_profiler.cli.console import NetraCLIRenderer, console
from netra_profiler.types import NetraProfile, PipelineContext


def _get_peak_ram_usage_in_mb() -> float:
    """
    OS TELEMETRY: High-Water Mark Memory Polling.

    Retrieves the maximum physical RAM consumed by the Netra process during its lifetime.
    Because Polars executes in a Rust multi-threaded backend, standard Python memory
    profilers (`tracemalloc`) fail to capture the true memory footprint. This function drops
    down to the OS level to read the C-level memory high-water mark.

    Returns:
        The peak RAM usage in Megabytes (MB) as a float.

    Notes:
        - Windows uses `psutil` to read the Peak Working Set.
        - Unix (Linux/macOS) uses the native `resource` module to read `ru_maxrss`.
    """
    if sys.platform == "win32":
        import psutil  # Deferred Windows-only import # noqa: PLC0415

        process = psutil.Process(os.getpid())
        # Windows "Peak Working Set" in bytes
        return process.memory_info().peak_wset / (1024 * 1024)
    else:
        import resource  # Deferred Unix-only import # noqa: PLC0415

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            # macOS ru_maxrss is in bytes
            return rusage.ru_maxrss / (1024 * 1024)
        else:
            # Linux ru_maxrss is in kilobytes
            return rusage.ru_maxrss / 1024


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"netra-profiler v{__version__}")
        raise typer.Exit()


def _format_bytes(size: int) -> str:
    """Converts raw bytes to human readable string (e.g. 6.9 GB)."""
    power = 1024.0
    n: float = float(size)

    labels = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    max_index = len(labels) - 1
    count = 0

    while n >= power and count < max_index:
        n /= power
        count += 1

    return f"{n:.2f} {labels[count]}"


def _detect_csv_separator(path: Path) -> str:
    """
    Peeks at the first 5 lines of a CSV file to automatically detect the delimiter.
    This prevents Polars from panicking or misparsing datasets that use semicolons or tabs.

    Args:
        path: The file path to the CSV.

    Returns:
        The detected string delimiter (e.g., `,`, `;`, `\t`). Defaults to `,` if detection fails.
    """
    try:
        with open(path, encoding="utf-8") as file:
            sample = "".join([file.readline() for _ in range(5)])
            # We restrict the sniffer to common data delimiters to prevent false positives
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
    except Exception:
        # If sniffing fails (e.g., weird encoding or 1-column file), fallback to standard comma
        return ","


def _scan_file(path: Path, full_inference: bool = False) -> tuple[pl.LazyFrame, str]:
    """
    DATA INGESTION: Lazy Execution Graph Initialization.

    Maps a file on disk to a Polars `LazyFrame` based on its extension.
    Crucially, this step does NOT load the data into RAM. It only creates the initial
    blueprint for the execution graph.

    Args:
        path: The file path to the dataset.
        full_inference: If True, forces Polars to scan the entire CSV to determine data
            types, preventing schema drift panics at the cost of slower initialization.

    Returns:
        A tuple containing the initialized `pl.LazyFrame` and a formatted string label
        representing the detected file format.
    """
    extension = path.suffix.lower()

    if extension == ".csv":
        # We pass the detected separator explicitly to Polars
        separator = _detect_csv_separator(path)

        infer_schema_length = None if full_inference else 10000
        return pl.scan_csv(
            path, separator=separator, infer_schema_length=infer_schema_length
        ), f"CSV (separator: '{separator}')"
    elif extension == ".parquet":
        return pl.scan_parquet(path), "Parquet"
    elif extension in {".ipc", ".arrow"}:
        return pl.scan_ipc(path), "IPC/Arrow"
    elif extension == ".json":
        try:
            return pl.scan_ndjson(path), "JSON (Newline)"
        except Exception:
            # Fallback for standard JSON array
            return pl.read_json(path).lazy(), "JSON (Standard)"
    else:
        raise ValueError(f"Unsupported extension: {extension}")


def _evaluate_pipeline_context(
    profile: NetraProfile, fail_on_critical: bool, fail_on_warnings: bool
) -> PipelineContext:
    """Generates the Quality Gate execution metadata."""

    quality_gate_active = fail_on_critical or fail_on_warnings
    alerts = profile.get("alerts", [])

    critical_alerts_count = sum(1 for alert in alerts if alert.get("level") == "CRITICAL")
    warning_alerts_count = sum(1 for alert in alerts if alert.get("level") == "WARNING")

    if not quality_gate_active:
        return {
            "quality_gate_active": False,
            "fail_on_critical": False,
            "fail_on_warnings": False,
            "status": "PASSIVE",
            "exit_code": 0,
            "reason": "No strict halting conditions were configured.",
        }

    should_fail = (
        fail_on_warnings and (warning_alerts_count > 0 or critical_alerts_count > 0)
    ) or (fail_on_critical and critical_alerts_count > 0)

    # Generate Unified Reason String
    if should_fail:
        parts = []
        if warning_alerts_count > 0:
            s = "S" if warning_alerts_count > 1 else ""
            parts.append(f"{warning_alerts_count} WARNING{s}")
        if critical_alerts_count > 0:
            parts.append(f"{critical_alerts_count} CRITICAL")

        anomaly_str = " and ".join(parts)
        reason = f"Halted due to {anomaly_str} quality issues."
    else:
        reason = "No data quality issues found."

    return {
        "quality_gate_active": True,
        "fail_on_critical": fail_on_critical,
        "fail_on_warnings": fail_on_warnings,
        "status": "FAILED" if should_fail else "PASSED",
        "exit_code": 1 if should_fail else 0,
        "reason": reason,
    }


app = typer.Typer(
    name="netra",
    help="Netra Profiler: High-performance profiling and data quality tool built with Polars.",
    add_completion=False,
)


def _run_json_mode(  # noqa: PLR0913
    path: Path,
    bins: int,
    top_k: int,
    full_inference: bool,
    ignore_columns: list[str],
    fail_on_critical: bool,
    fail_on_warnings: bool,
    config: dict[str, Any] | None,
    config_source: str,
) -> None:
    """
    HEADLESS MODE: Raw JSON output execution.

    Silences the Rich UI entirely and prints the raw `NetraProfile` contract to stdout.
    Used for piping output directly into downstream systems or files (e.g., `> report.json`).
    """
    try:
        df, _ = _scan_file(path, full_inference=full_inference)
        profiler = Profiler(
            df, ignore_columns=ignore_columns, config=config, config_source=config_source
        )
        profile = profiler.run(bins=bins, top_k=top_k)

        # 1. Generate & Inject Pipeline Context
        pipeline_context = _evaluate_pipeline_context(profile, fail_on_critical, fail_on_warnings)
        profile["_meta"]["pipeline_context"] = pipeline_context

        # 2. Output the JSON Payload
        print(json.dumps(profile, default=str))

        # 3. Exit
        if pipeline_context["exit_code"] == 1:
            raise typer.Exit(code=1)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        raise typer.Exit(code=1) from None


def _connect_data_source(
    ui: NetraCLIRenderer, path: Path, full_inference: bool
) -> tuple[pl.LazyFrame, str, int]:
    """
    PHASE 1: I/O Connection & Schema Resolution.

    Handles file scanning and resolves the schema of the dataset. This updates the UI
    with the table topology (column count, size, types) before the heavy mathematical
    engine starts its run.
    """
    try:
        ui.render_data_source_spinner(path.name)

        load_start = time.time()
        df, file_type = _scan_file(path, full_inference=full_inference)
        file_size = os.path.getsize(path)

        schema = df.collect_schema()
        load_time = time.time() - load_start

        datatypes = [str(t) for t in schema.values()]
        datatype_counts = Counter(datatypes)
        schema_info = ", ".join([f"{v} {k}" for k, v in datatype_counts.items()])

        file_info = {"path": str(path), "size": _format_bytes(file_size), "type": file_type}

        ui.render_data_source_panel(
            file_info=file_info,
            schema_info=schema_info,
            columns=len(schema),
            time_taken=load_time,
        )

        return df, file_type, file_size

    except Exception as e:
        raw_error = str(e)
        clean_error = raw_error.split("You might want to try:")[0].strip()

        ui.render_fatal_error(
            step="data_source",
            message=f"Connection failed to data source: {path}",
            hint=clean_error,
        )
        raise typer.Exit(code=1) from None


def _execute_profiling(
    ui: NetraCLIRenderer,
    profiler: Profiler,
    file_size: int,
    bins: int,
    top_k: int,
) -> NetraProfile:
    """
    PHASE 2: Engine Execution & Telemetry Calculation.

    Wraps the core mathematical profiling loop. Manages the animated terminal progress
    and calculates hardware telemetry (Throughput and Peak RAM usage) immediately after
    the engine releases its memory buffers.
    """
    try:
        progress = ui.render_engine_status_panel()
        engine_messages = [
            "Resolving lazy execution graph...",
            "Allocating zero-copy memory buffers...",
            "Vectorizing data streams...",
            "Threading execution pipelines...",
            "Collapsing data dimensions...",
            "Calibrating Apache Arrow matrices...",
            "Synthesizing profile topology...",
        ]
        active_message = random.choice(engine_messages)
        progress.add_task(active_message, total=None)

        profile = profiler.run(bins=bins, top_k=top_k)

        engine_time = profile["_meta"]["engine_time_seconds"]
        peak_ram_usage = _get_peak_ram_usage_in_mb()

        throughput = (file_size / engine_time) / (1024**3) if engine_time > 0 else 0.0

        ui.render_engine_telemetry_panel(
            engine_time=engine_time,
            throughput_gb_s=throughput,
            peak_ram_usage=peak_ram_usage,
        )

        return profile

    except Exception as e:
        raw_error = str(e)
        clean_error = raw_error.split("You might want to try:")[0].strip()

        if "parse" in raw_error or "primitive" in raw_error:
            message = "Dataset contains mixed-typed columns."
            hint = (
                f"{clean_error}\n\n[brand]Tip:[/] Run with [bold]--full-inference[/bold] "
                "to force full-file schema inference."
            )
        else:
            message = "Engine panicked during profiling."
            hint = clean_error

        ui.render_fatal_error(
            step="profiling",
            message=message,
            hint=hint,
        )
        raise typer.Exit(code=1) from None


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    pass


@app.command()
def profile(  # noqa: PLR0913
    file_path: str = typer.Argument(..., help="Path to the dataset (CSV, Parquet, JSON, IPC)."),
    json_output: bool = typer.Option(
        False, "--json", help="Output raw JSON to stdout (silences UI)."
    ),
    bins: int = typer.Option(20, "--bins", min=1, help="Number of histogram bins."),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Number of frequent items to show."),
    full_inference: bool = typer.Option(
        False, "--full-inference", help="Force full-file schema inference for messy CSVs."
    ),
    ignore_columns: list[str] = typer.Option(  # noqa: B008
        [],
        "--ignore",
        "-i",
        help="Columns to ignore (e.g. primary keys). Can be used multiple times.",
    ),
    low_memory: bool = typer.Option(
        False,
        "--low-memory",
        help=(
            "Enables bounded-memory profiling. Uses approximate unique counts and disables exact"
            " quantiles to prevent OOM on massive datasets."
        ),
    ),
    fail_on_critical: bool = typer.Option(
        False,
        "--fail-on-critical",
        help="Halt the pipeline (exit 1) if CRITICAL data quality alerts are found.",
    ),
    fail_on_warnings: bool = typer.Option(
        False,
        "--fail-on-warnings",
        help="Halt the pipeline (exit 1) if Warnings or Critical data quality alerts are found.",
    ),
    config_file_path: str | None = typer.Option(
        None, "--config", "-c", help="Path to netra_config.yaml. Overrides NETRA_CONFIG env var."
    ),
) -> None:
    """
    Profile the connected data source and generate the CLI report.
    """
    path = Path(file_path)

    # --- CONFIGURATION RESOLUTION ---
    config = None
    config_source = "Default"
    resolved_config_path = config_file_path or os.environ.get("NETRA_CONFIG") or "netra_config.yaml"
    config_path_object = Path(resolved_config_path)

    if config_path_object.exists():
        try:
            with open(config_path_object, encoding="utf-8") as f:
                config = yaml.safe_load(f)
                config_source = str(config_path_object)
        except Exception as e:
            error_message = f"Failed to parse YAML config file: {e}"
            if json_output:
                print(json.dumps({"error": error_message}))
            else:
                console.print(f"[bold red]Configuration Error:[/] {error_message}")
            raise typer.Exit(code=1) from None

    elif config_file_path or os.environ.get("NETRA_CONFIG"):
        # Fail hard ONLY if an explicit path was provided and not found
        error_message = f"Config file not found at '{resolved_config_path}'"
        if json_output:
            print(json.dumps({"error": error_message}))
        else:
            console.print(f"[bold red]Configuration Error:[/] {error_message}")
        raise typer.Exit(code=1)

    # --- MODE 1: JSON OUTPUT ---

    if json_output:
        _run_json_mode(
            path,
            bins,
            top_k,
            full_inference,
            ignore_columns,
            fail_on_critical,
            fail_on_warnings,
            config,
            config_source,
        )
        return

    # --- MODE 2: CLI OUTPUT ---

    exit_code = 0  # Default to success

    with NetraCLIRenderer() as ui:
        if not path.exists():
            ui.render_data_source_spinner(path.name)
            ui.render_fatal_error(
                step="data_source",
                message="File not found on disk.",
                hint=f"Verify the path: {path}",
            )
            raise typer.Exit(code=1)

        # Phase 1: Data Source Connection
        df, file_type, file_size = _connect_data_source(ui, path, full_inference)

        # Phase 2: Data Profiling & Engine Telemetry
        profiler = Profiler(
            df,
            dataset_name=path.name,
            dataset_format=file_type,
            ignore_columns=ignore_columns,
            low_memory=low_memory,
            config=config,
            config_source=config_source,
        )
        profile = _execute_profiling(ui, profiler, file_size, bins, top_k)

        # Phase 3: Evaluate Pipeline Gatekeeper and Inject JSON Meta
        pipeline_context = _evaluate_pipeline_context(profile, fail_on_critical, fail_on_warnings)
        profile["_meta"]["pipeline_context"] = pipeline_context
        exit_code = pipeline_context["exit_code"]

        # Phase 4: Render Dashboards
        ui.render_profiling_results(profile)

        profiler_warnings = profile.get("_meta", {}).get("profiler_warnings", [])
        ui.render_pipeline_info(pipeline_context, profiler_warnings)

    # Clean Exit: Executed only after the 'with ui:' context safely closes the Rich terminal state
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command()
def info() -> None:
    """Prints environment info for debugging."""
    console.print(f"Netra Version: {__version__}")
    console.print(f"Polars Version: {pl.__version__}")
    console.print(f"Python Version: {sys.version.split()[0]}")


if __name__ == "__main__":
    app()
