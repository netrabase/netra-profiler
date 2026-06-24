"""
Open Data Contract Standard (ODCS) Generator.

This module transforms a strictly typed NetraProfile
dictionary into an ODCS v3.1.0 YAML Data Contract.
"""

from typing import Any

import yaml

from netra_profiler.config import DiagnosticConfig
from netra_profiler.types import ColumnMetrics, NetraProfile, is_numeric_type, is_string_type


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
        Executes the mapping, injects contextual YAML comments,
        and returns the serialized YAML string.
        """
        self._build_dataset_block()
        self._build_schema_block()

        # We use sort_keys=False to preserve our visual hierarchy (dataset -> schema -> quality)
        yaml_string = yaml.safe_dump(
            self.odcs_document, sort_keys=False, default_flow_style=False, allow_unicode=True
        )

        # Post-Processing: Injecting Contextual Comments
        # PyYAML strips out the comments, so we inject them via string replacement to ensure
        # the user receives clear instructions on how to build on top of this draft contract.

        schema_comment = (
            "\n# --- SCHEMA BLOCK ---\n"
            "# Data Quality constraints are drafted from empirical data.\n"
            "# Review and adjust these bounds to match your intended business logic.\n"
            "schema:"
        )
        yaml_string = yaml_string.replace("schema:", schema_comment)

        pk_hint_replacement = (
            "  # Candidate Primary Key (Highly Unique). Uncomment the line below to enforce:\n"
            "    # primaryKey: true\n"
        )
        yaml_string = yaml_string.replace("  __netra_pk_hint: true\n", pk_hint_replacement)

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

        # 1. Null Values (Only if column config allows nulls)
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

        # 2. Duplicate Values
        max_duplicate_percent = self.config.get_rule("max_duplicate_percent", column_name)
        if max_duplicate_percent not in (False, None):
            rules.append(
                {
                    "metric": "duplicateValues",
                    "mustBeLessThan": max_duplicate_percent * 100,
                    "unit": "percent",
                }
            )

        # 3. Variance (Constant Check)
        column_constant_check_override = (
            column_name in self.config.column_overrides
            and "constant_check_enabled" in self.config.column_overrides[column_name]
        )
        if column_constant_check_override:
            constant_check = self.config.get_rule("constant_check_enabled", column_name)
            if constant_check is True:
                rules.append(
                    {
                        "type": "sql",
                        "query": "SELECT COUNT(DISTINCT {property}) FROM {object}",
                        "mustBeGreaterThan": 1,
                    }
                )

        return rules

    def _build_schema_block(self) -> None:
        """Iterates over columns to build the ODCS v3.1.0 schema and nested quality blocks."""

        dataset_name = self.profile["dataset"]["name"]

        # Root Element: The Object
        root_object: dict[str, Any] = {
            "name": dataset_name,
            "logicalType": "object",
            "properties": [],
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

            # --- Primary Key (Heuristic) ---
            uniqueness_threshold = self.config.get_rule("id_uniqueness_threshold", column_name)
            if uniqueness_threshold is not False and uniqueness_threshold is not None:
                # This is the actual uniqueness in the data for IDs or ID-like columns
                empirical_uniqueness = metrics.get("n_unique", 0) / max(self.row_count, 1)
                if empirical_uniqueness >= uniqueness_threshold:
                    # We inject a hidden key that we will string-replace with a YAML comment later
                    property_entry["__netra_pk_hint"] = True

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
