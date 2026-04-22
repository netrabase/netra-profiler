import polars as pl

from netra_profiler import Profiler


def test_dirty_data_alerts() -> None:
    """
    Verifies that the diagnostic engine correctly flags scalar anomalies:
    - Constant columns (Single value)
    - Empty columns (100% null)
    - High Null columns (> 95%)
    - Skewed columns
    - Zero-inflated columns
    - Outliers (IQR method)
    - String Length Anomalies
    - ID-like columns (Negative test: Ignored if row count < 100)
    """

    # 1. Setup: Create data with enough rows (20) to generate statistical significance
    rows = 20
    data = {
        "constant_column": [1] * rows,  # CONSTANT
        "empty_column": [None] * rows,  # 100% Null triggers EMPTY_COLUMN
        "high_nulls_column": [None] * 12 + [1] * 8,  # 60% Null triggers HIGH_NULLS
        "id_column": [str(i) for i in range(rows)],  # Negative test for min_rows_for_pk_check
        "skewed_column": [1] * (rows - 1) + [1000],  # Heavy Skew)
        "zero_column": [0] * 15 + [1] * 5,  # ZERO_INFLATED (75% zeros)
        # Outlier needs non-zero IQR (10 to 13) with a massive spike
        "outlier_column": [10, 11, 12, 13] * 4 + [10, 11, 12, 9999],
        # String anomaly needs a max length much larger than the mean
        "string_column": ["a"] * 19 + ["a" * 100],
    }

    df = pl.DataFrame(data)

    test_config = {"diagnostics": {"global_thresholds": {"string_length_anomaly_multiplier": 5.0}}}

    # 2. Execution
    profiler = Profiler(df, config=test_config)
    profile = profiler.run()

    # 3. Validation
    alerts = profile["alerts"]
    alert_types = {alert["type"] for alert in alerts}  # We use a set for O(1) lookup

    assert "CONSTANT" in alert_types, "Failed to detect CONSTANT column."
    assert "EMPTY_COLUMN" in alert_types, "Failed to detect 100% EMPTY_COLUMN."
    assert "HIGH_NULLS" in alert_types, "Failed to detect HIGH_NULLS warning."
    assert "SKEWED" in alert_types, f"Failed to detect skew. Found: {alert_types}"
    assert "ZERO_INFLATED" in alert_types, "Failed to detect zero inflation."
    assert "OUTLIERS_DETECTED" in alert_types, "Failed to detect IQR outliers."
    assert "INCONSISTENT_STRING_LENGTH" in alert_types, "Failed to detect string length anomaly."
    assert "ALL_DISTINCT" not in alert_types, "ALL_DISTINCT triggered on small dataset."


def test_diagnostic_config() -> None:
    """
    Verifies that the config Dict loads and overrides apply correctly.
    """

    # A column that is 100% constant and would trigger a CRITICAL alert
    df = pl.DataFrame({"constant_column": [42] * 50})

    # 1. Default Config (Should raise CRITICAL alert)
    default_profile = Profiler(df).run()
    default_alert_types = {a["type"] for a in default_profile["alerts"]}
    assert "CONSTANT" in default_alert_types, "Baseline override test failed."

    # We explicitly disable the check via the column_overrides config
    override_config = {
        "diagnostics": {
            "column_overrides": {
                "constant_column": {
                    "constant_check_enabled": False,
                }
            }
        }
    }

    # 2. Rule Overriden (CRITICAL alert should be disabled)
    overriden_profile = Profiler(df, config=override_config).run()
    overriden_alert_types = {a["type"] for a in overriden_profile["alerts"]}
    assert "CONSTANT" not in overriden_alert_types, "Rule override test failed."


def test_diagnostic_typo_warnings() -> None:
    """
    Verifies that unused column overrides correctly generate warnings in the meta block.
    """
    df = pl.DataFrame({"real_column": [1, 2, 3, 4, 5]})

    typo_config = {
        "diagnostics": {
            "column_overrides": {
                "real_column": {"skew_threshold": 1.0},
                "fake_column_or_typo": {"skew_threshold": 1.0},
            }
        }
    }

    profiler = Profiler(df, config=typo_config)
    profile = profiler.run()

    warnings = profile["_meta"]["config_warnings"]

    # We expect exactly 1 warning about the orphaned 'fake_column_or_typo'
    assert len(warnings) == 1, f"Expected 1 warning, got {len(warnings)}"
    assert "fake_column_or_typo" in warnings[0], "Warning string did not identify the typo column!"
