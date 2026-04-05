import polars as pl

from netra_profiler import Profiler


def test_dirty_data_alerts() -> None:
    """
    Verifies that the diagnostic engine correctly flags:
    - Constant columns (Single value)
    - High Null columns (> 95%)
    - Skewed columns (> 2.0 skew)
    - ID-like columns (Ignored if row count < 100)
    """
    # 1. Setup: Create data with enough rows (20) to generate statistical significance
    rows = 20
    data = {
        "constant_column": [1] * rows,  # CONSTANT
        "empty_column": [None] * (rows - 1) + [1],  # HIGH_NULLS (>95%)
        "id_column": [str(i) for i in range(rows)],  # ALL_DISTINCT (20 unique)
        # Should NOT trigger if threshold is 100 rows
        "skewed_column": [1] * (rows - 1) + [1_000_000],  # SKEWED (Skew > 2.0)
        "zero_column": [0] * 15 + [1] * 5,  # ZERO_INFLATED (75% zeros)
    }

    df = pl.DataFrame(data)

    # 2. Execution
    profiler = Profiler(df)
    profile = profiler.run()

    # 3. Validation
    alerts = profile["alerts"]
    alert_types = {alert["type"] for alert in alerts}  # We use a set for O(1) lookup

    # Critical Integrity Checks
    assert "CONSTANT" in alert_types, "Failed to detect CONSTANT column."

    # Null Check (Accepts either Critical or Warning level)
    assert "EMPTY_COLUMN" in alert_types or "HIGH_NULLS" in alert_types, (
        "Failed to detect high null density."
    )

    # Statistical Checks
    assert "SKEWED" in alert_types, f"Failed to detect skew. Found: {alert_types}"
    assert "ZERO_INFLATED" in alert_types, "Failed to detect zero inflation."

    # Logic Checks (Config Validation)
    # Ensure strict config adherence: 20 rows is too small to confidently flag an ID
    assert "ALL_DISTINCT" not in alert_types, (
        "ALL_DISTINCT triggered on small dataset! 'MIN_ROWS' config ignored."
    )
