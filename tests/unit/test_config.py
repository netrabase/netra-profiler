import pytest

from netra_profiler.config import DiagnosticConfig, DiffConfig, PipelineConfig

MEAN_SHIFT_PCT_THRESHOLD_10_PCT = 0.10
MEAN_SHIFT_PCT_THRESHOLD_50_PCT = 0.50
NULL_COUNT_SHIFT_PCT_THRESHOLD_10_PCT = 0.10
NULL_COUNT_SHIFT_PCT_THRESHOLD_01_PCT = 0.01


def test_diff_config_defaults():
    """Ensure the engine loads with sane defaults if no config is provided."""
    config = DiffConfig(None)
    assert config.get_rule("schema_change_level") == "CRITICAL"
    assert config.get_rule("mean_shift_pct") == MEAN_SHIFT_PCT_THRESHOLD_10_PCT


def test_diff_config_global_overrides():
    """Ensure global rules in YAML overwrite the engine defaults."""
    yaml_dict = {
        "diff": {"global_thresholds": {"mean_shift_pct": 0.50, "schema_change_level": "WARNING"}}
    }
    config = DiffConfig(yaml_dict)
    assert config.get_rule("mean_shift_pct") == MEAN_SHIFT_PCT_THRESHOLD_50_PCT
    assert config.get_rule("schema_change_level") == "WARNING"


def test_diff_config_column_overrides():
    """Ensure column-specific rules override global rules."""
    yaml_dict = {
        "diff": {
            "global_thresholds": {"null_count_shift_pct": 0.10},
            "column_overrides": {
                "optional_field": {
                    "null_count_shift_pct": False  # Disable check
                },
                "revenue": {
                    "null_count_shift_pct": 0.01  # Hyper-strict check
                },
            },
        }
    }
    config = DiffConfig(yaml_dict)

    # Baseline global
    assert (
        config.get_rule("null_count_shift_pct", "other_column")
        == NULL_COUNT_SHIFT_PCT_THRESHOLD_10_PCT
    )

    # Overrides
    assert config.get_rule("null_count_shift_pct", "optional_field") is False
    assert (
        config.get_rule("null_count_shift_pct", "revenue") == NULL_COUNT_SHIFT_PCT_THRESHOLD_01_PCT
    )


def test_diff_config_type_validation_error():
    """Ensure the config parser catches invalid types and raises ValueErrors."""
    yaml_dict = {
        "diff": {
            "global_thresholds": {
                # This should be a float, not an arbitrary string
                "mean_shift_pct": "not_a_number"
            }
        }
    }
    # We use pytest to assert that the specific ValueError is raised
    with pytest.raises(ValueError, match="must be of type float"):
        DiffConfig(yaml_dict)


def test_pipeline_config_parsing():
    """Ensure the Pipeline context extracts correctly from the YAML."""
    yaml_dict = {"pipeline": {"fail_on_critical": True, "fail_on_warnings": False}}
    config = PipelineConfig(yaml_dict)
    assert config.fail_on_critical is True
    assert config.fail_on_warnings is False
    assert config.quality_gate_active is True


def test_diagnostic_config_defaults():
    """Ensure DiagnosticConfig loads correct defaults via BaseConfig."""
    config = DiagnosticConfig(None)
    assert config.get_rule("null_critical_threshold") == 0.95
    assert config.get_rule("constant_check_enabled") is True


def test_config_type_validation_soft_cast():
    """Ensure BaseConfig safely casts compatible types (e.g., int to float)."""
    yaml_dict = {
        "diagnostics": {
            "global_thresholds": {
                # skew_threshold expects a float, but YAML might parse '2' as an int
                "skew_threshold": 2
            }
        }
    }
    config = DiagnosticConfig(yaml_dict)
    # It should be safely cast to 2.0 (float)
    assert config.get_rule("skew_threshold") == 2.0
    assert isinstance(config.get_rule("skew_threshold"), float)


def test_config_type_validation_failure():
    """Ensure BaseConfig raises a loud ValueError on completely invalid types."""
    yaml_dict = {
        "diagnostics": {
            "global_thresholds": {
                # Expects a float, but we pass a string that can't be cast
                "null_critical_threshold": "ninety_five_percent"
            }
        }
    }
    with pytest.raises(ValueError) as exc_info:
        DiagnosticConfig(yaml_dict)

    assert "Config Error" in str(exc_info.value)
    assert "null_critical_threshold" in str(exc_info.value)


def test_unused_override_warnings():
    """Ensure the config system correctly tracks which overrides were actually applied."""
    yaml_dict = {
        "diagnostics": {
            "column_overrides": {
                "age": {"null_critical_threshold": 0.50},
                "ghost_column": {"null_critical_threshold": 0.10},  # Not in dataset
            }
        }
    }
    config = DiagnosticConfig(yaml_dict)

    # Simulate the engine checking rules for the 'age' column
    _ = config.get_rule("null_critical_threshold", "age")

    # Validate warnings
    dataset_columns = ["id", "age", "name"]
    warnings = config.get_unused_override_warnings(dataset_columns)

    assert len(warnings) == 1
    assert "ghost_column" in warnings[0]
    assert "age" not in warnings[0]
