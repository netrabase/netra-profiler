"""
Open Data Contract Standard (ODCS) Generator.

This module transforms a strictly typed NetraProfile
dictionary into an ODCS v3.1.0 YAML Data Contract.
"""

import re
from typing import Any

import yaml

from netra_profiler.config import DiagnosticConfig
from netra_profiler.types import ColumnMetrics, NetraProfile, is_numeric_type, is_string_type

VALID_VALUES_THRESHOLD = 5


class ODCSBuilder:
    """
    Translates Netra profiling data and diagnostic thresholds
    into an Open Data Contract Standard (ODCS) document.
    """

    def __init__(self, profile: NetraProfile, config: DiagnosticConfig):
        """
        Args:
            profile: The NetraProfile dictionary.
            config: The DiagnosticConfig for the data quality thresholds.
        """
        self.profile = profile
        self.config = config
        self.row_count = self.profile["dataset"]["row_count"]

        # Dictionaries to hold validation rule hints before string replacement
        self._constant_hints: dict[str, str] = {}
        self._enum_hints: dict[str, list[str]] = {}
        self._temporal_hints: dict[str, tuple[str, str]] = {}
        self._distinct_count_hints: dict[str, int] = {}

        self.odcs_document: dict[str, Any] = {
            "apiVersion": "v3.1.0",
            "kind": "DataContract",
            "version": "0.0.1",
            "status": "draft",
            "dataset": {},
            "schema": [],
        }

    def build(self) -> str:
        """
        Executes the mapping, injects hint comments,
        and returns the serialized YAML string.
        """
        self._build_dataset_block()
        self._build_schema_block()

        # We use sort_keys=False to preserve our visual hierarchy (dataset -> schema -> quality)
        yaml_string = yaml.safe_dump(
            self.odcs_document, sort_keys=False, default_flow_style=False, allow_unicode=True
        )

        # Post-Processing: Injecting Contextual Comments for Suggested Rules
        # PyYAML natively strips out the comments, so we inject placeholder keys into the
        # dictionary, and use RegEx with a callable replace them with comments for
        # rules suggested derived from data

        schema_comment = (
            "\n# --- SCHEMA BLOCK ---\n"
            "# Data Quality constraints are drafted from empirical data.\n"
            "# Review and adjust these bounds to match your intended business logic.\n"
            "schema:"
        )
        yaml_string = yaml_string.replace("schema:", schema_comment)

        # Insert Dataset-Level Row Count Hint
        def insert_row_count_hint(match: re.Match[str]) -> str:
            indent = match.group(1)
            return (
                f"{indent}# Candidate Volume (Observed Row Count: {self.row_count})."
                f"{indent}# Uncomment the lines below to enforce:"
                f"{indent}# quality:"
                f"{indent}# - metric: rowCount"
                f"{indent}#   mustBeBetween: [{self.row_count}, {self.row_count}]"
            )

        yaml_string = re.sub(
            r"(\s*)__netra_row_count_hint: true\n", insert_row_count_hint, yaml_string
        )

        # Insert Primary Key Hint
        def insert_pk_hint(match: re.Match[str]) -> str:
            indent = match.group(1)
            return (
                f"{indent}# Candidate Primary Key (Highly Unique)."
                f"{indent}# Uncomment the line below to enforce:"
                f"{indent}# primaryKey: true\n"
            )

        yaml_string = re.sub(r"(\s*)__netra_pk_hint: true\n", insert_pk_hint, yaml_string)

        # Insert Constant Value Hint
        for column_id, value in self._constant_hints.items():

            def insert_constant_hint(match: re.Match[str], v: str = value) -> str:
                indent = match.group(1)
                return (
                    f"{indent}# Candidate Constant Value (100% Single Value)."
                    f"{indent}# Uncomment the lines below to enforce:"
                    f"{indent}# logicalTypeOptions:"
                    f"{indent}#   pattern: '^{v}$'\n"
                )

            yaml_string = re.sub(
                rf"(\s*)__netra_constant_hint: {re.escape(column_id)}\n",
                insert_constant_hint,
                yaml_string,
            )

        # Insert Valid Values Hint
        for column_id, vals in self._enum_hints.items():

            def insert_valid_values_hint(match: re.Match[str], v_list: list[str] = vals) -> str:
                indent = match.group(1)
                lines = [
                    f"{indent}# Candidate Valid Values (Low Cardinality).",
                    f"{indent}# Uncomment the lines below to enforce:",
                    f"{indent}# quality:",
                    f"{indent}# - metric: invalidValues",
                    f"{indent}#   mustBe: 0",
                    f"{indent}#   arguments:",
                    f"{indent}#     validValues:",
                ]
                for v in v_list:
                    lines.append(f"{indent}#       - {v}")
                return "".join(lines) + "\n"

            yaml_string = re.sub(
                rf"(\s*)__netra_enum_hint: {re.escape(column_id)}\n",
                insert_valid_values_hint,
                yaml_string,
            )

        # Insert Temporal Bounds Hint
        for column_id, (min_value, max_value) in self._temporal_hints.items():

            def insert_temporal_hint(
                match: re.Match[str], mn: str = min_value, mx: str = max_value
            ) -> str:
                indent = match.group(1)
                return (
                    f"{indent}# Candidate Temporal Bounds (Observed in data)."
                    f"{indent}# Uncomment the lines below to enforce:"
                    f"{indent}# logicalTypeOptions:"
                    f"{indent}#   minimum: '{mn}'"
                    f"{indent}#   maximum: '{mx}'\n"
                )

            yaml_string = re.sub(
                rf"(\s*)__netra_temporal_hint: {re.escape(column_id)}\n",
                insert_temporal_hint,
                yaml_string,
            )

        # Insert Distinct Count Hint
        for column_id, distinct_count in self._distinct_count_hints.items():

            def insert_distinct_hint(match: re.Match[str], count: int = distinct_count) -> str:
                indent = match.group(1)
                return (
                    f"{indent}# Candidate Distinct Count (Observed Cardinality)."
                    f"{indent}# Uncomment the lines below to enforce:"
                    f"{indent}# quality:"
                    f"{indent}# - type: sql"
                    f"{indent}#   query: 'SELECT COUNT(DISTINCT {{property}}) FROM {{object}}'"
                    f"{indent}#   mustBeBetween: [{count}, {count}]\n"
                )

            yaml_string = re.sub(
                rf"(\s*)__netra_distinct_hint: {re.escape(column_id)}\n",
                insert_distinct_hint,
                yaml_string,
            )

        return yaml_string

    def _build_dataset_block(self) -> None:
        """Maps dataset metadata into the ODCS dataset block."""
        dataset_name = self.profile["dataset"]["name"]
        self.odcs_document["dataset"] = {"name": dataset_name}

    def _map_logical_type(self, polars_type: str) -> str:
        """Maps Polars native data types to ODCS logical types."""
        if not polars_type:  # REVIEW FOR DEFENSIVENESS
            return "string"

        if "Int" in polars_type:
            return "integer"
        if "Float" in polars_type or "Decimal" in polars_type:
            return "number"
        if "Date" in polars_type or "Time" in polars_type:
            return "timestamp"
        if "Boolean" in polars_type:
            return "boolean"

        return "string"

    def _build_logical_options(self, column_type: str, metrics: ColumnMetrics) -> dict[str, Any]:
        """Extracts value bounds based on column data type."""

        options: dict[str, Any] = {}

        if is_string_type(column_type):
            if (max_length := metrics.get("max_length")) is not None:
                options["maxLength"] = max_length
            if (min_length := metrics.get("min_length")) is not None:
                options["minLength"] = min_length

        elif is_numeric_type(column_type):
            if (min_value := metrics.get("min")) is not None:
                options["minimum"] = min_value
            if (max_value := metrics.get("max")) is not None:
                options["maximum"] = max_value

        return options

    def _build_property_quality_rules(
        self, column_name: str, is_required: bool
    ) -> list[dict[str, Any]]:
        """Evaluates column configs to generate nested v3.1.0 quality rules."""

        rules = []

        # Null Values (Only if column config allows nulls)
        if not is_required:
            null_threshold = self.config.get_rule("null_critical_threshold", column_name)
            if null_threshold not in (False, None):
                rules.append(
                    {
                        "metric": "nullValues",
                        "mustBeLessThan": null_threshold * 100,
                        "unit": "percent",
                    }
                )

        # Duplicate Values
        max_duplicate_percent = self.config.get_rule("max_duplicate_percent", column_name)
        if max_duplicate_percent not in (False, None):
            rules.append(
                {
                    "metric": "duplicateValues",
                    "mustBeLessThan": max_duplicate_percent * 100,
                    "unit": "percent",
                }
            )

        return rules

    def _apply_primary_key_hint(
        self, property_entry: dict[str, Any], column_name: str, metrics: ColumnMetrics
    ) -> None:
        """Injects a primary key hint if uniqueness meets the threshold."""
        uniqueness_threshold = self.config.get_rule("id_uniqueness_threshold", column_name)
        if uniqueness_threshold is not False and uniqueness_threshold is not None:
            empirical_uniqueness = metrics.get("n_unique", 0) / max(self.row_count, 1)
            if empirical_uniqueness >= uniqueness_threshold:
                property_entry["__netra_pk_hint"] = True

    def _apply_string_heuristics(
        self,
        property_entry: dict[str, Any],
        column_id: str,
        column_type: str,
        metrics: ColumnMetrics,
    ) -> None:
        """Injects hints for Constants, Enums, and Distinct Counts for string types."""
        if not is_string_type(column_type):
            return

        n_unique = metrics.get("n_unique")
        top_k = metrics.get("top_k", [])

        # Constant (n_unique == 1)
        if n_unique == 1 and top_k:
            constant_value = top_k[0].get("value")
            if constant_value is not None:
                self._constant_hints[column_id] = str(constant_value)
                property_entry["__netra_constant_hint"] = column_id

        # Set Membership (1 < n_unique <= 5)
        elif n_unique is not None and 1 < n_unique <= VALID_VALUES_THRESHOLD and top_k:
            valid_values = [str(item["value"]) for item in top_k if item.get("value") is not None]
            if len(valid_values) == n_unique:
                self._enum_hints[column_id] = valid_values
                property_entry["__netra_enum_hint"] = column_id

        # Distinct Value Bounds (> 5)
        elif n_unique is not None and n_unique > VALID_VALUES_THRESHOLD:
            if "__netra_pk_hint" not in property_entry:
                self._distinct_count_hints[column_id] = n_unique
                property_entry["__netra_distinct_hint"] = column_id

    def _apply_temporal_heuristics(
        self,
        property_entry: dict[str, Any],
        column_id: str,
        column_type: str,
        metrics: ColumnMetrics,
    ) -> None:
        """Injects hints for temporal min/max bounds."""
        if "Date" in column_type or "Time" in column_type:
            min_value = metrics.get("min")
            max_value = metrics.get("max")

            if min_value is not None and max_value is not None:
                self._temporal_hints[column_id] = (str(min_value), str(max_value))
                property_entry["__netra_temporal_hint"] = column_id

    def _build_schema_block(self) -> None:
        """Iterates over columns to build the ODCS v3.1.0 schema and nested quality blocks."""

        dataset_name = self.profile["dataset"]["name"]

        # Root Object
        root_object: dict[str, Any] = {
            "name": dataset_name,
            "logicalType": "object",
            "properties": [],
            "__netra_row_count_hint": True,
        }

        for column_name, metrics in self.profile["columns"].items():
            column_type = metrics.get("data_type", "")

            # Generate a slugified ID (e.g., "Vendor ID" -> "vendor_id_prop")
            column_id = str(column_name).replace(" ", "_").lower()

            # Property Element
            property_entry: dict[str, Any] = {
                "id": f"{column_id}_prop",
                "name": column_name,
                "logicalType": self._map_logical_type(column_type),
                "physicalType": column_type,
            }

            self._apply_primary_key_hint(property_entry, column_name, metrics)
            self._apply_string_heuristics(property_entry, column_id, column_type, metrics)
            self._apply_temporal_heuristics(property_entry, column_id, column_type, metrics)

            # --- Completeness ---
            is_required = metrics.get("null_count", 0) == 0
            if is_required:
                property_entry["required"] = True

            # --- Value Bounds ---
            logical_options = self._build_logical_options(column_type, metrics)
            if logical_options:
                property_entry["logicalTypeOptions"] = logical_options

            # --- Data Quality ---
            quality_rules = self._build_property_quality_rules(column_name, is_required)
            if quality_rules:
                property_entry["quality"] = quality_rules

            root_object["properties"].append(property_entry)

        self.odcs_document["schema"].append(root_object)
