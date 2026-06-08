"""
The NetraBase Data and Schema Diff Engine.

This module performs O(1) threshold-based comparisons between two NetraProfile JSON contracts.
It detects structural schema changes, volume shifts, and statistical drift without requiring
access to the underlying raw data.
"""

import datetime
from typing import Any

from netra_profiler import __version__
from netra_profiler.config import DiffConfig
from netra_profiler.types import (
    ColumnMetrics,
    DiffAlert,
    DiffReport,
    NetraProfile,
)


class DiffEngine:
    """
    Executes rule-based threshold diffing between a Reference Profile and a Target Profile.
    """

    def __init__(
        self,
        reference_profile: NetraProfile,
        target_profile: NetraProfile,
        config: DiffConfig | None = None,
    ):
        """
        Initializes the DiffEngine with two finalized profiles.

        Args:
            reference_profile: The baseline Profile (e.g., Yesterday's data).
            target_profile: The new Profile to be evaluated (e.g., Today's data).
            config: The threshold configurations.
        """

        self.reference = reference_profile
        self.target = target_profile
        self.config = config or DiffConfig()

        self.reference_columns: dict[str, ColumnMetrics] = self.reference.get("columns", {})
        self.target_columns: dict[str, ColumnMetrics] = self.target.get("columns", {})

        self.alerts: list[DiffAlert] = []

    def run(self) -> DiffReport:
        """
        Executes the three-layer diffing architecture.

        Returns:
            A strictly typed `DiffReport` containing all detected anomalies and metadata.
        """

        start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

        # Layer 1: Structural Integrity
        self._check_schema_changes()

        # Layer 2 & 3: Volume, Completeness, and Magnitude
        self._check_dataset_volume()
        self._check_column_metrics()

        end_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

        # Construct the final DiffReport
        report: DiffReport = {
            "reference_dataset": self.reference.get("dataset", {}),  # type: ignore
            "target_dataset": self.target.get("dataset", {}),  # type: ignore
            "alerts": self.alerts,
            "_meta": {
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "execution_start_epoch": start_time,
                "execution_end_epoch": end_time,
                "engine_time_seconds": round(end_time - start_time, 4),
                "profiler_version": __version__,
                "is_low_memory_run": False,  # Not applicable to diff engine
                "config_source": "Loaded via CLI/API",
                "profiler_warnings": [],
                "config_warnings": [],
                "pipeline_context": None,  # Will be injected by the CLI orchestrator
            },
        }

        return report

    def _calculate_delta_percentage(self, reference_value: Any, target_value: Any) -> float | None:
        """
        Safely calculates the absolute percentage difference between two scalars.
        """

        if reference_value is None or target_value is None:
            return None

        if not isinstance(reference_value, (int, float)) or not isinstance(
            target_value, (int, float)
        ):
            return None

        # If both are exactly 0, there is 0% shift
        if reference_value == 0 and target_value == 0:
            return 0.0

        if reference_value == 0:
            return float("inf")

        return abs(target_value - reference_value) / abs(reference_value)

    def _check_schema_changes(self) -> None:
        """
        Detects dropped columns, added columns, and data type mutations.
        """

        schema_level = self.config.get_rule("schema_change_level")
        type_level = self.config.get_rule("type_change_level")

        reference_keys = set(self.reference_columns.keys())
        target_keys = set(self.target_columns.keys())

        # 1. Dropped Columns
        if schema_level:
            dropped_columns = reference_keys - target_keys
            for column_name in dropped_columns:
                self.alerts.append(
                    {
                        "column_name": column_name,
                        "diff_type": "COLUMN_DROPPED",
                        "level": schema_level,
                        "metric": "schema",
                        "reference_value": "PRESENT",
                        "target_value": "MISSING",
                        "delta_percentage": None,
                        "message": f"Column '{column_name}' is missing in the target dataset.",
                    }
                )

        # 2. Added Columns
        if schema_level:
            added_columns = target_keys - reference_keys
            for column_name in added_columns:
                self.alerts.append(
                    {
                        "column_name": column_name,
                        "diff_type": "COLUMN_ADDED",
                        "level": schema_level,
                        "metric": "schema",
                        "reference_value": "MISSING",
                        "target_value": "PRESENT",
                        "delta_percentage": None,
                        "message": f"New column '{column_name}' was found in the target dataset.",
                    }
                )

        # 3. Data Type Mutations
        if type_level:
            common_columns = reference_keys.intersection(target_keys)
            for column_name in common_columns:
                reference_type = self.reference_columns[column_name].get("data_type")
                target_type = self.target_columns[column_name].get("data_type")

                if reference_type != target_type:
                    self.alerts.append(
                        {
                            "column_name": column_name,
                            "diff_type": "TYPE_CHANGED",
                            "level": type_level,
                            "metric": "data_type",
                            "reference_value": reference_type,
                            "target_value": target_type,
                            "delta_percentage": None,
                            "message": f"Data type changed from {reference_type} to {target_type}.",
                        }
                    )

    def _check_dataset_volume(self) -> None:
        """Evaluates global row count shifts."""
        threshold = self.config.get_rule("row_count_shift_pct")
        if threshold in (False, None):
            return

        reference_rows = self.reference.get("dataset", {}).get("row_count", 0)
        target_rows = self.target.get("dataset", {}).get("row_count", 0)

        delta = self._calculate_delta_percentage(reference_rows, target_rows)

        if delta is not None and delta > threshold:
            if delta == float("inf"):
                message = f"Total row count shifted from 0 to {target_rows}."
            elif delta >= 5.0:  # If shift is 500% or greater, switch to multiplier
                multiplier = target_rows / reference_rows
                direction = "spiked" if target_rows > reference_rows else "plummeted"
                message = (
                    f"Total row count {direction} by {multiplier:.1f}x "
                    f"(from {reference_rows} to {target_rows})."
                )
            else:
                direction = "increased" if target_rows > reference_rows else "decreased"
                message = (
                    f"Total row count {direction} by {delta:.1%} "
                    f"(from {reference_rows} to {target_rows})."
                )
            self.alerts.append(
                {
                    "column_name": "TABLE",
                    "diff_type": "VOLUME_SHIFT",
                    "level": "WARNING",
                    "metric": "row_count",
                    "reference_value": reference_rows,
                    "target_value": target_rows,
                    "delta_percentage": "Infinity" if delta == float("inf") else delta,
                    "message": message,
                }
            )

    def _check_column_metrics(self) -> None:
        """
        Evaluates completeness and scalar magnitude shifts for all columns
        that exist in both datasets.
        """
        common_columns = set(self.reference_columns.keys()).intersection(self.target_columns.keys())

        # We define the exact dictionary keys we want to test, and their corresponding config rules
        metric_checks = [
            ("null_count", "null_count_shift_pct", "COMPLETENESS_SHIFT"),
            ("n_unique", "n_unique_shift_pct", "CARDINALITY_SHIFT"),
            ("mean", "mean_shift_pct", "MEAN_SHIFT"),
            ("min", "min_shift_pct", "MIN_SHIFT"),
            ("max", "max_shift_pct", "MAX_SHIFT"),
            ("p50", "p50_shift_pct", "MEDIAN_SHIFT"),
            ("std", "std_shift_pct", "VARIANCE_SHIFT"),
            ("mean_length", "mean_length_shift_pct", "STRING_LENGTH_SHIFT"),
        ]

        for column_name in common_columns:
            reference_metrics = self.reference_columns[column_name]
            target_metrics = self.target_columns[column_name]

            for metric_key, rule_name, diff_type in metric_checks:
                threshold = self.config.get_rule(rule_name, column_name)
                if threshold in (False, None):
                    continue

                reference_value = reference_metrics.get(metric_key)
                target_value = target_metrics.get(metric_key)

                # Explicitly guarantee these are numbers before executing math
                if not isinstance(reference_value, (int, float)) or not isinstance(
                    target_value, (int, float)
                ):
                    continue

                delta = self._calculate_delta_percentage(reference_value, target_value)

                if delta is not None and delta > threshold:
                    metric_name = metric_key.replace("_", " ").title()

                    if delta == float("inf"):
                        message = f"{metric_name} shifted from 0 to {target_value}."
                    elif delta >= 5.0:  # If shift is 500% or greater, switch to multiplier
                        multiplier = target_value / reference_value
                        direction = "spiked" if target_value > reference_value else "plummeted"
                        message = (
                            f"{metric_name} {direction} by {multiplier:.1f}x "
                            f"(from {reference_value} to {target_value})."
                        )
                    else:
                        direction = "increased" if target_value > reference_value else "decreased"
                        message = (
                            f"{metric_name} {direction} by {delta:.1%} "
                            f"(from {reference_value} to {target_value})."
                        )
                    self.alerts.append(
                        {
                            "column_name": column_name,
                            "diff_type": diff_type,
                            "level": "WARNING",
                            "metric": metric_key,
                            "reference_value": reference_value,
                            "target_value": target_value,
                            "delta_percentage": "Infinity" if delta == float("inf") else delta,
                            "message": message,
                        }
                    )
