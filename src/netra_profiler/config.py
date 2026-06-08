"""
Universal Configuration Manager for Netra Profiler.

This module parses the `netra_config.yaml` file, manages global pipeline context,
and provides strictly typed rule resolution for the Diagnostic and Diff engines.
"""

from typing import Any


class BaseConfig:
    """Base class providing cascading resolution for global vs column-level rules."""

    def __init__(
        self, config_dict: dict[str, Any] | None, section_key: str, default_globals: dict[str, Any]
    ) -> None:
        self.global_thresholds = default_globals.copy()
        self.column_overrides: dict[str, dict[str, Any]] = {}
        self.unused_overrides: set[str] = set()

        if config_dict and section_key in config_dict:
            section_config = config_dict[section_key]

            # 1. Load and Validate Globals
            if "global_thresholds" in section_config:
                for key, value in section_config["global_thresholds"].items():
                    if key in self.global_thresholds:
                        self.global_thresholds[key] = self._validate_type(
                            key, value, default_globals
                        )

            # 2. Load Overrides
            if "column_overrides" in section_config:
                raw_overrides = section_config["column_overrides"]
                for column_name, rules in raw_overrides.items():
                    validated_rules = {}
                    for rule_key, rule_value in rules.items():
                        if rule_key in self.global_thresholds:
                            validated_rules[rule_key] = self._validate_type(
                                rule_key, rule_value, default_globals
                            )
                    self.column_overrides[column_name] = validated_rules

                self.unused_overrides = set(self.column_overrides.keys())

    def _validate_type(self, key: str, value: Any, defaults: dict[str, Any]) -> Any:
        """Ensures YAML config inputs match expected types and whitelisted values."""

        if value is None or value is False:
            return value  # Allow null/false to explicitly disable a check

        # Strict validation for Alert Levels (severity strings)
        if key.endswith("_level"):
            valid_levels = {"CRITICAL", "WARNING", "INFO"}
            if str(value).upper() not in valid_levels:
                raise ValueError(
                    f"Config Error: '{key}' must be one of {valid_levels}. Received: '{value}'"
                )
            return str(value).upper()

        expected_type = type(defaults[key])
        if not isinstance(value, expected_type):
            try:
                # Soft cast (e.g., if they pass int '1' for a float threshold)
                return expected_type(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Config Error: '{key}' must be of type {expected_type.__name__}. "
                    f"Received: '{value}'"
                ) from None
        return value

    def get_rule(self, rule_name: str, column_name: str | None = None) -> Any:
        """
        Returns quality metric threshold by checking column overrides first,
        and then falling back to global default. Pass empty column_name to
        get the global default.
        """

        if column_name and column_name in self.column_overrides:
            # We mark this override as 'used'
            if column_name in self.unused_overrides:
                self.unused_overrides.remove(column_name)

            column_rules = self.column_overrides[column_name]
            if rule_name in column_rules:
                return column_rules[rule_name]

        return self.global_thresholds[rule_name]

    def get_unused_override_warnings(self, dataset_columns: list[str]) -> list[str]:
        """Returns warnings for any column overrides that didn't match dataset columns."""

        warnings_list = []
        for column in self.unused_overrides:
            if column not in dataset_columns:
                warnings_list.append(
                    f"ConfigWarning: Override for '{column}' was ignored (column not found)."
                )
        return warnings_list


class DiagnosticConfig(BaseConfig):
    """Dynamic configuration manager for diagnostic thresholds."""

    DEFAULT_GLOBALS = {
        "null_critical_threshold": 0.95,
        "null_warning_threshold": 0.50,
        "skew_threshold": 2.0,
        "zero_inflated_threshold": 0.10,
        "high_cardinality_threshold": 10000,
        "high_correlation_threshold": 0.95,
        "id_uniqueness_threshold": 0.99,
        "min_rows_for_pk_check": 100,
        "possible_numeric_sample_size": 5,
        "outlier_iqr_multiplier": 3.0,
        "string_length_anomaly_multiplier": 50.0,
        "constant_check_enabled": True,
    }

    def __init__(self, config_dict: dict[str, Any] | None = None):
        super().__init__(config_dict, "diagnostics", self.DEFAULT_GLOBALS)


class DiffConfig(BaseConfig):
    """Dynamic configuration manager for data drift and schema thresholds."""

    DEFAULT_GLOBALS = {
        # Structural & Schema Changes
        "schema_change_level": "CRITICAL",
        "type_change_level": "CRITICAL",
        # Volume & Completeness
        "row_count_shift_pct": 0.20,
        "null_count_shift_pct": 0.10,
        "n_unique_shift_pct": 0.15,
        # Magnitude & Scalar Shifts (Numeric)
        "mean_shift_pct": 0.10,
        "min_shift_pct": 0.10,
        "max_shift_pct": 0.10,
        "p50_shift_pct": 0.10,
        "std_shift_pct": 0.10,
        # Magnitude Shifts (Strings)
        "mean_length_shift_pct": 0.20,
    }

    def __init__(self, config_dict: dict[str, Any] | None = None):
        super().__init__(config_dict, "diff", self.DEFAULT_GLOBALS)


class PipelineConfig:
    """Manages the global execution state (CI/CD Quality Gates) from the YAML file."""

    def __init__(self, config_dict: dict[str, Any] | None = None):
        self.fail_on_critical: bool = False
        self.fail_on_warnings: bool = False
        self.quality_gate_active: bool = False

        if config_dict and "pipeline" in config_dict:
            pipeline_block = config_dict["pipeline"]

            # YAML definitions act as the baseline fallback if CLI flags aren't used
            self.fail_on_critical = pipeline_block.get("fail_on_critical", False)
            self.fail_on_warnings = pipeline_block.get("fail_on_warnings", False)
            self.quality_gate_active = pipeline_block.get(
                "quality_gate_active", self.fail_on_critical or self.fail_on_warnings
            )
