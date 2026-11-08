"""
Abstract Base Target and standardized result schemas for Netra Profiler benchmark targets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkResult:
    """Standardized performance and diagnostic metrics emitted by a profiler adapter run."""

    target_name: str
    wall_time_seconds: float
    engine_time_seconds: float | None
    peak_rss_mb: float
    rows_evaluated: int
    columns_profiled: int
    throughput_mb_s: float
    row_velocity_m_rows_s: float
    passed: bool
    anomalies_detected: int
    critical_alerts: int
    warning_alerts: int
    info_alerts: int
    extra_meta: dict[str, Any] = field(default_factory=dict)


class BenchmarkTarget(ABC):
    """Abstract interface governing the lifecycle of a benchmark profiler adapter."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def setup(
        self,
        dataset_path: Path,
        config_path: Path,
        **kwargs: Any,
    ) -> None:
        """
        Prepares the adapter for execution (config loading, path resolution).
        Excluded from the timed phase.
        """
        ...

    @abstractmethod
    def run_profiling(self) -> tuple[BenchmarkResult, dict[str, Any]]:
        """
        Executes strictly the profiling pass over the target dataset.

        Returns:
            A tuple of (BenchmarkResult, finalized NetraProfile dictionary).
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Performs post-run resource reclamation."""
        ...
