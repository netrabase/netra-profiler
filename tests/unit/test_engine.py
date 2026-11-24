"""
Unit tests for Netra v0.7.0 streaming query-plan builders, cost-based memory
batching, deterministic hash partitioning, and 10,240-bin quantile refinement.
"""

import polars as pl
import pytest

from netra_profiler import Profiler, engine


def test_dtype_byte_width_and_memory_budget_packing() -> None:
    """
    Verifies that `get_datatype_byte_width` maps Polars types to physical byte widths
    and `batch_columns_by_memory` respects both byte budgets and column caps.
    """

    assert engine.get_datatype_byte_width(None) == 8
    assert engine.get_datatype_byte_width(pl.Boolean()) == 1
    assert engine.get_datatype_byte_width(pl.Int8()) == 1
    assert engine.get_datatype_byte_width(pl.UInt16()) == 2
    assert engine.get_datatype_byte_width(pl.Float32()) == 4
    assert engine.get_datatype_byte_width(pl.Float64()) == 8
    assert engine.get_datatype_byte_width(pl.Int128()) == 16
    assert engine.batch_columns_by_memory([]) == []

    costs = [
        ("col_heavy", 450),
        ("col_a", 100),
        ("col_b", 150),
        ("col_c", 200),
        ("col_oversized", 1000),
    ]
    batches = engine.batch_columns_by_memory(costs, budget_bytes=500, max_columns_per_batch=3)

    # Sorted ascending: col_a (100) + col_b (150) + col_c (200) = 450 <= 500 (batch 1)
    # col_heavy (450) -> batch 2; col_oversized (1000 > 500) -> isolated in batch 3
    assert batches == [["col_a", "col_b", "col_c"], ["col_heavy"], ["col_oversized"]]


def test_even_and_odd_quantile_refinement_branches() -> None:
    """
    Verifies exact quantile (`p25`, `p50`, `p75`) accuracy across all three median
    extraction paths in `extract_quantiles_from_bins`:
    1. Odd valid row count (`r50_lo == r50_hi`).
    2. Even valid row count where `r50_lo` and `r50_hi` fall inside the same fine bin.
    3. Even valid row count where `r50_lo` and `r50_hi` straddle two distinct fine bins.
    """

    # 1. Even valid count (6 values) where the two middle values (10.0 and 90.0)
    # are far apart and guaranteed to land in different fine bins (`b50_lo != b50_hi`).
    # 2. Even valid count (6 values) where the two middle values (50.00001 and 50.00002)
    # on a [0, 1000] range land in the exact same fine bin (`b50_lo == b50_hi`).
    # 3. Odd valid count (5 values + 1 null) where `r50_lo == r50_hi`.
    df = pl.DataFrame(
        {
            "cross_bin_even": [0.0, 5.0, 10.0, 90.0, 95.0, 100.0],
            "same_bin_even": [0.0, 10.0, 50.00001, 50.00002, 900.0, 1000.0],
            "odd_with_null": [10.0, 20.0, 30.0, 40.0, 100.0, None],
        }
    )

    profile = Profiler(df).run(bins=20)

    for col_name in ["cross_bin_even", "same_bin_even", "odd_with_null"]:
        col_metrics = profile["columns"][col_name]
        series = df[col_name].drop_nulls()

        assert col_metrics.get("p25") == pytest.approx(
            series.quantile(0.25, interpolation="nearest")
        )
        assert col_metrics.get("p50") == pytest.approx(series.median())
        assert col_metrics.get("p75") == pytest.approx(
            series.quantile(0.75, interpolation="nearest")
        )


def test_quantile_fallback_on_pathological_spike(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that when a column's target fine bins exceed the row-fraction threshold,
    `_build_quantile_bin_spec` returns `None` and `build_quantile_plans` seamlessly
    falls back to native Polars quickselect with identical mathematical results.
    """

    df = pl.DataFrame({"spiked": [float(i % 5) for i in range(200)]})

    # Force `_build_quantile_bin_spec` to reject fine-bin bracketing so fallback_plans runs
    monkeypatch.setattr(
        engine,
        "_build_quantile_bin_spec",
        lambda *_args, **_kwargs: None,
    )

    profile = Profiler(df).run(bins=10)
    col_metrics = profile["columns"]["spiked"]

    assert col_metrics.get("p25") == pytest.approx(df["spiked"].quantile(0.25))
    assert col_metrics.get("p50") == pytest.approx(df["spiked"].median())
    assert col_metrics.get("p75") == pytest.approx(df["spiked"].quantile(0.75))


def test_intra_column_hash_partitioned_nunique(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that decomposing a column into `P` deterministic hash partitions via
    `build_partitioned_unique_count_plan` and `_execute_partitioned_unique_counts` sums to the
    exact global `n_unique` (including IEEE-754 `NaN`, `-0.0`, `+0.0`, and `None`).
    """

    values = [float(i) for i in range(150)] + [-0.0, 0.0, float("nan"), None, 42.0, 99.0]
    df = pl.DataFrame({"telemetry_val": values})

    # Force the profiler into the extreme-cardinality hash-partitioned branch
    monkeypatch.setattr(engine, "BASE_PLAN_OVERHEAD_BYTES", 0)
    monkeypatch.setattr(engine, "LOW_CARDINALITY_THRESHOLD", 10)
    monkeypatch.setattr(engine, "SINGLE_COLUMN_PARTITION_THRESHOLD_BYTES", 1_000)
    monkeypatch.setattr(engine, "TARGET_PARTITION_MEMORY_BYTES", 800)

    profile = Profiler(df).run()
    assert profile["columns"]["telemetry_val"].get("n_unique") == 150


def test_analytical_bypasses_and_boolean_cardinality() -> None:
    """
    Verifies analytical shortcuts for constant columns (`min == max`), all-null
    columns, single-valid-row columns, and Boolean columns.
    """

    df = pl.DataFrame(
        {
            "all_null_num": pl.Series([None, None, None], dtype=pl.Float64),
            "constant_num": [7.5, 7.5, 7.5],
            "single_valid_num": [None, 42.0, None],
            "bool_const": [True, True, True],
            "bool_mixed": [True, False, None],
            "all_null_str": pl.Series([None, None, None], dtype=pl.String),
        }
    )

    profile = Profiler(df).run()
    cols = profile["columns"]

    assert cols["all_null_num"].get("n_unique") == 0
    assert cols["all_null_num"].get("p50") is None
    assert cols["all_null_num"].get("histogram") == []

    assert cols["constant_num"].get("n_unique") == 1
    assert cols["constant_num"].get("p25") == 7.5
    assert cols["constant_num"].get("p50") == 7.5
    assert cols["constant_num"].get("p75") == 7.5
    assert len(cols["constant_num"].get("histogram", [])) == 1

    assert cols["single_valid_num"].get("n_unique") == 1
    assert cols["single_valid_num"].get("p50") == 42.0

    assert cols["bool_const"].get("n_unique") == 1
    assert cols["bool_mixed"].get("n_unique") == 2
    assert cols["all_null_str"].get("n_unique") == 0


def test_low_memory_mode_retains_exact_skew_and_kurtosis() -> None:
    """
    Verifies that `low_memory=True` computes exact O(1) streaming `skew` and
    `kurtosis` while omitting exact quantiles (`p25`, `p50`, `p75`).
    """

    df = pl.DataFrame({"x": [1.0, 2.0, 2.0, 3.0, 4.0, 100.0]})

    standard_profile = Profiler(df, low_memory=False).run()
    low_mem_profile = Profiler(df, low_memory=True).run()

    std_col = standard_profile["columns"]["x"]
    low_col = low_mem_profile["columns"]["x"]

    assert low_col.get("skew") is not None
    assert low_col.get("kurtosis") is not None
    assert low_col.get("skew") == pytest.approx(std_col.get("skew"))
    assert low_col.get("kurtosis") == pytest.approx(std_col.get("kurtosis"))

    assert low_col.get("p25") is None
    assert low_col.get("p50") is None
    assert low_col.get("p75") is None
