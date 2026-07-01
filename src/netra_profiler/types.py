"""
Type definitions, schemas, and inference utilities for the NetraBase Ecosystem.

This module serves as the structural foundation for Netra Profiler. It defines:
1. The strict hierarchical output structure for `NetraProfile`.
2. The schema definitions for the Diff Engine (`DiffReport`, `DiffAlert`).
3. Pipeline execution contexts and OS-level telemetry metadata.
4. Type inference utilities to map native Polars data types to Netra's logical types.
"""

from typing import Any, TypedDict


class PipelineContext(TypedDict):
    """
    Context injected by the CLI orchestrator (`main.py`) when running from the command line.
    This does not exist in the Python API context.
    """

    quality_gate_active: bool
    fail_on_critical: bool
    fail_on_warnings: bool
    status: str  # "PASSED", "FAILED", or "PASSIVE"
    exit_code: int
    reason: str | None


class BaseExecutionMeta(TypedDict):
    """Shared telemetry across all Netra Profiler operations."""

    created_at: str
    execution_start_epoch: float
    execution_end_epoch: float
    engine_time_seconds: float
    profiler_version: str
    config_source: str
    config_warnings: list[str]
    pipeline_context: PipelineContext | None


class ProfileExecutionMeta(BaseExecutionMeta):
    """Telemetry specific to the Profiling engine."""

    is_low_memory_run: bool
    profiler_warnings: list[str]


class DiffExecutionMeta(BaseExecutionMeta):
    """Telemetry specific to the Diff engine."""

    diff_warnings: list[str]


# --- NetraProfile Object Definition ---


class DatasetMeta(TypedDict):
    """Metadata describing the ingested dataset."""

    name: str
    format: str
    row_count: int


class ExecutionMeta(TypedDict):
    """Telemetry and metadata regarding the profiler's execution environment and performance."""

    created_at: str
    execution_start_epoch: float
    execution_end_epoch: float
    engine_time_seconds: float
    profiler_version: str
    is_low_memory_run: bool
    config_source: str
    profiler_warnings: list[str]
    config_warnings: list[str]
    pipeline_context: PipelineContext | None


class ColumnMetrics(TypedDict, total=False):
    """
    Comprehensive statistical profile for a single column.

    Note: total=False is used because numeric and string columns have
    mutually exclusive fields (e.g., 'mean' vs 'mean_length').
    """

    data_type: str
    null_count: int
    n_unique: int
    histogram: list[dict[str, Any]]
    top_k: list[dict[str, Any]]

    # Numeric & String Shared Metrics
    min: float | int | str | None
    max: float | int | str | None

    # Numeric Specific
    mean: float | None
    zero_count: int | None
    std: float | None
    skew: float | None
    kurtosis: float | None
    p25: float | None
    p50: float | None
    p75: float | None

    # String Specific
    min_length: int | None
    max_length: int | None
    mean_length: float | None
    blank_count: int | None


class CorrelationPair(TypedDict):
    """Represents a single edge in the de-duplicated correlation matrix."""

    column_a: str
    column_b: str
    score: float


class CorrelationData(TypedDict):
    """Container for the correlation matrices and the sampling methodology."""

    pearson: list[CorrelationPair]
    spearman: list[CorrelationPair]
    sampling_method: str | None


class DiagnosticAlert(TypedDict):
    """A data quality anomaly detected by the DiagnosticEngine."""

    column_name: str
    type: str
    level: str  # "CRITICAL", "WARNING", "INFO"
    message: str
    value: float | int | None


class NetraProfile(TypedDict):
    """The root schema for the Netra Profiler Data Contract."""

    dataset: DatasetMeta
    columns: dict[str, ColumnMetrics]
    correlations: CorrelationData
    alerts: list[DiagnosticAlert]
    _meta: ProfileExecutionMeta


# --- DiffReport Object Definition ---


class DiffAlert(TypedDict):
    """A data drift or schema anomaly detected by the DiffEngine."""

    column_name: str
    diff_type: str  # "SCHEMA_CHANGE", "MEAN_SHIFT", "NULL_SHIFT"
    level: str  # "CRITICAL", "WARNING", "INFO"
    metric: str  # "mean", "null_count", "data_type"
    reference_value: Any
    target_value: Any
    delta_percentage: float | str | None
    message: str


class DiffReport(TypedDict):
    """The root schema for the Netra Profiler Diff output."""

    reference_dataset: DatasetMeta
    target_dataset: DatasetMeta
    alerts: list[DiffAlert]
    _meta: DiffExecutionMeta


# --- Type Inference Utilities ---


def is_numeric_type(data_type: str) -> bool:
    """
    Determines if a Polars stringified data type belongs to a numeric family.

    Args:
        data_type: The string representation of a Polars datatype (e.g., "Int64", "Float32").
    """
    # Logic: if "Int" in "Int64": return True
    return any(base_type in data_type for base_type in ["Int", "Float", "Decimal"])


def is_string_type(data_type: str) -> bool:
    """
    Determines if a Polars stringified data type belongs to a text/categorical family.

    Args:
        data_type: The string representation of a Polars datatype (e.g., "Utf8", "Enum").
    """
    if not data_type:
        return False
    return any(base_type in data_type for base_type in ["String", "Utf8", "Categorical", "Enum"])
