"""
Netra Profiler Benchmark Adapter (Supports Standard and Low-Memory modes).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from benchmarks.targets.base import BenchmarkResult, BenchmarkTarget
from netra_profiler.config import PipelineConfig
from netra_profiler.profiler import Profiler


class NetraProfilerBenchmarkTarget(BenchmarkTarget):
    """Benchmark target adapter for the Netra Profiler engine."""

    def __init__(self, low_memory: bool = False) -> None:
        target_name = "netra-low-memory" if low_memory else "netra"
        super().__init__(name=target_name)
        self.low_memory = low_memory
        self.dataset_path: Path | None = None
        self.config_path: Path | None = None
        self.scratch_dir: Path | None = None
        self.config_dict: dict[str, Any] = {}
        self.pipeline_config: PipelineConfig | None = None

    def setup(
        self,
        dataset_path: Path,
        config_path: Path,
        **kwargs: Any,
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        self.config_path = Path(config_path).resolve()

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        # Provision an isolated temporary workspace for out-of-core spills
        self.scratch_dir = self.dataset_path.parent / f".netra_prof_bench_{os.getpid()}"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, encoding="utf-8") as f:
            self.config_dict = yaml.safe_load(f) or {}

        self.pipeline_config = PipelineConfig(self.config_dict)

    def run_profiling(self) -> tuple[BenchmarkResult, dict[str, Any]]:
        if self.dataset_path is None or self.config_path is None:
            raise RuntimeError(
                "NetraProfilerBenchmarkTarget.setup() must be called prior to run_profiling()."
            )

        t0 = time.perf_counter()

        # 1. Construct LazyFrame scan
        lf = pl.scan_parquet(self.dataset_path)

        # 2. Initialize & Execute Profiler
        profiler = Profiler(
            df=lf,
            dataset_name=self.dataset_path.stem,
            dataset_format="Parquet",
            low_memory=self.low_memory,
            config=self.config_dict,
            config_source=self.config_path.name,
        )
        profile = profiler.run(bins=20, top_k=10)

        wall_time = time.perf_counter() - t0

        # 3. Extract Metrics & Evaluate Quality Gate
        engine_time = profile["_meta"].get("engine_time_seconds")
        total_rows = profile["dataset"].get("row_count", 0)
        columns_profiled = len(profile.get("columns", {}))

        alerts = profile.get("alerts", [])
        critical_count = sum(1 for a in alerts if a["level"] == "CRITICAL")
        warning_count = sum(1 for a in alerts if a["level"] == "WARNING")
        info_count = sum(1 for a in alerts if a["level"] == "INFO")

        passed = True
        exit_code = 0
        if self.pipeline_config and (
            (self.pipeline_config.fail_on_critical and critical_count > 0)
            or (self.pipeline_config.fail_on_warnings and (critical_count + warning_count) > 0)
        ):
            passed = False
            exit_code = 1

        # 4. Compute Throughput & Row Velocity
        file_size_bytes = self.dataset_path.stat().st_size
        effective_time = engine_time if (engine_time and engine_time > 0) else wall_time

        throughput_mb_s = (
            (file_size_bytes / (1024 * 1024)) / effective_time if effective_time > 0 else 0.0
        )
        row_velocity = (total_rows / effective_time) / 1_000_000 if effective_time > 0 else 0.0

        pearson_edges = len(profile.get("correlations", {}).get("pearson", []))
        spearman_edges = len(profile.get("correlations", {}).get("spearman", []))

        result = BenchmarkResult(
            target_name=self.name,
            wall_time_seconds=round(wall_time, 4),
            engine_time_seconds=round(engine_time, 4) if engine_time else None,
            peak_rss_mb=0.0,  # Stamped by external subprocess collector
            rows_evaluated=total_rows,
            columns_profiled=columns_profiled,
            throughput_mb_s=round(throughput_mb_s, 2),
            row_velocity_m_rows_s=round(row_velocity, 2),
            passed=passed,
            anomalies_detected=len(alerts),
            critical_alerts=critical_count,
            warning_alerts=warning_count,
            info_alerts=info_count,
            extra_meta={
                "polars_threads": pl.thread_pool_size(),
                "low_memory": self.low_memory,
                "pearson_pairs": pearson_edges,
                "spearman_pairs": spearman_edges,
                "sampling_method": profile.get("correlations", {}).get("sampling_method"),
                "config_warnings": len(profile["_meta"].get("config_warnings", [])),
                "profiler_warnings": len(profile["_meta"].get("profiler_warnings", [])),
                "exit_code": exit_code,
            },
        )
        return result, dict(profile)

    def teardown(self) -> None:
        """Purges temporary spill workspace."""
        if self.scratch_dir and self.scratch_dir.exists():
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
