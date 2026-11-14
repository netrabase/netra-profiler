"""
Polars lazy query-plan builders for Netra Profiler.

This module constructs stateless `pl.LazyFrame` execution graphs for schema
normalization, scalar aggregations, histograms, and categorical Top-K frequencies.
All plans are collected by the `Profiler` object in `profiler.py`.
"""

from typing import Any

import polars as pl


def build_scalar_plan(lf: pl.LazyFrame, low_memory: bool = False) -> pl.LazyFrame:
    """
    Constructs a single-row `pl.LazyFrame` plan computing scalar metrics across all columns.

    All column aggregations are bundled into a single `lf.select()` so the Polars query optimizer
    can apply Common Subexpression Elimination (CSE) and evaluate all metrics in one pass.

    Design Notes:
        - Inline Subexpressions vs. `with_columns()`: Helper expressions such as float
            `fill_nan(None)` and string `str.len_chars()` are kept inline inside `select()`
            rather than pre-projected via `with_columns()`. When `low_memory=False`, order
            statistics (`quantile`, `median`) and higher moments (`skew`, `kurtosis`) require
            full-column materialization. Keeping subexpressions inline allows Polars' CSE to
            evaluate and free temporary arrays.
        - Non-Null Cardinality: Polars' `n_unique()` and `approx_n_unique()` count `null` as
            a distinct value. Subtracting `(null_count() > 0).cast(pl.UInt32)` recovers the
            exact non-null cardinality using O(1) validity bitmap metadata.
        - Blank String Detection: Uses `str.len_bytes() == 0` instead of `str.len_chars() == 0`
            after `str.strip_chars()`. An empty string always has zero bytes, and checking byte
            length is a O(1) metadata lookup.

    Args:
        lf: Input Polars LazyFrame with flattened primitive columns.
        low_memory: If True, uses approximate unique counts (`approx_n_unique()`) and skips
            quantiles, skewness, and kurtosis (`p25`, `p50`, `p75`, `skew`, `kurtosis`) to
            keep memory usage low.

    Returns:
        A `pl.LazyFrame` yielding a 1-row DataFrame with aliased metric columns
        (e.g., `table_row_count`, `{column_name}_null_count`, `{column_name}_mean`).
    """

    expressions: list[pl.Expr] = []

    # 1. Global Computations (Table level)
    # We prefix these with 'table_' to keep the namespace clean.
    expressions.append(pl.len().alias("table_row_count"))

    # 2. Column-Level Computations
    # We iterate over the schema to compute the stats for each type.
    schema = lf.collect_schema()
    for column_name, data_type in schema.items():
        # Add column data type to the profile
        expressions.append(pl.lit(str(data_type)).alias(f"{column_name}_data_type"))

        # Convert IEEE 754 NaN to Null on float columns so missingness and aggregations
        # treat NaN and Null identically.
        column = pl.col(column_name).fill_nan(None) if data_type.is_float() else pl.col(column_name)

        # Polars' `.n_unique()` and `.approx_n_unique()` treat `null` as a
        # distinct value (e.g., `[42.0, 42.0, null]` yields `n_unique == 2`, and an
        # all-null column `[null, null]` yields `n_unique == 1`).
        # Subtracting `int(null_count > 0)` recovers the correct non-null cardinality with
        # zero memory allocation. Null count is a O(1) metadata lookup operation, so it's cheap.
        raw_n_unique = column.approx_n_unique() if low_memory else column.n_unique()
        has_nulls = (column.null_count() > 0).cast(pl.UInt32)
        n_unique_expression = (raw_n_unique - has_nulls).alias(f"{column_name}_n_unique")

        # Universal Stats (All Columns)
        expressions.extend(
            [
                column.null_count().alias(f"{column_name}_null_count"),
                n_unique_expression,
            ]
        )

        # Numeric Stats (Integers & Floats)
        if data_type.is_numeric():
            expressions.extend(
                [
                    # Basic Stats
                    column.mean().alias(f"{column_name}_mean"),
                    column.min().alias(f"{column_name}_min"),
                    column.max().alias(f"{column_name}_max"),
                    # Native zero counting
                    (column == 0).sum().alias(f"{column_name}_zero_count"),
                    # Distribution Stats (1/2)
                    column.std().alias(f"{column_name}_std"),
                ]
            )

            if not low_memory:
                expressions.extend(
                    [
                        # Distribution Stats (2/2)
                        column.skew().alias(f"{column_name}_skew"),
                        column.kurtosis().alias(f"{column_name}_kurtosis"),
                        # Quantiles (Percentiles)
                        column.quantile(0.25).alias(f"{column_name}_p25"),
                        column.median().alias(f"{column_name}_p50"),
                        column.quantile(0.75).alias(f"{column_name}_p75"),
                    ]
                )

        # Temporal Stats (Date/Datetime/Time)
        elif data_type.is_temporal():
            expressions.extend(
                [
                    # We cast to String at the Polars level to guarantee native
                    # ISO 8601 string formatting for JSON serialization.
                    column.min().cast(pl.String).alias(f"{column_name}_min"),
                    column.max().cast(pl.String).alias(f"{column_name}_max"),
                ]
            )

        # String/Categorical Stats
        elif data_type in (pl.String, pl.Categorical, pl.Enum):
            # Avoid redundant cast if already pl.String
            string_column = (
                pl.col(column_name)
                if data_type == pl.String
                else pl.col(column_name).cast(pl.String)
            )
            char_lengths = string_column.str.len_chars()

            expressions.extend(
                [
                    # 1. Lexicographical stats (First/Last alphabetical value)
                    string_column.min().alias(f"{column_name}_min"),
                    string_column.max().alias(f"{column_name}_max"),
                    # 2. Length stats (Polars CSE deduplicates char_lengths within select)
                    char_lengths.mean().alias(f"{column_name}_mean_length"),
                    char_lengths.min().alias(f"{column_name}_min_length"),
                    char_lengths.max().alias(f"{column_name}_max_length"),
                    # 3. Blank String Check
                    # Strips whitespace; if byte length is 0 (O(1) check), it's blank.
                    # Nulls naturally propagate as Null and are ignored by .sum()
                    (string_column.str.strip_chars().str.len_bytes() == 0)
                    .sum()
                    .alias(f"{column_name}_blank_count"),
                ]
            )

    # 3. Construct the Query Plan
    return lf.select(expressions)


def build_histogram_plans(
    lf: pl.LazyFrame, numeric_columns: list[str], profile_data: dict[str, Any], bins: int
) -> dict[str, pl.LazyFrame]:
    """
    Constructs streaming histogram plans for numeric columns.

    Because binning requires each column's `[min, max]` bounds, these plans are
    constructed after the scalar scan completes.

    Design Notes:
        - Deferred Null Filtering: Rather than calling `.filter()` on the raw column
            (which allocates more memory per numeric column), float `NaN`s are converted
            to `Null` using `fill_nan(None)`. Nulls are grouped into a single `null` group
            during `group_by`, which is dropped afterwards using `.drop_nulls("bin_index")`.
        - Boundary Clamping: `.clip(0, bins - 1)` ensures the maximum value (`val == max`)
            maps to the final bin `bins - 1` rather than spilling into index `bins`.

    Args:
        lf: Input Polars LazyFrame.
        numeric_columns: Names of numeric columns in `lf`.
        profile_data: Flat metric dictionary containing `{col}_min` and `{col}_max` bounds.
        bins: Target number of equi-width intervals.

    Returns:
        A dictionary mapping each valid numeric column name to a `pl.LazyFrame` yielding
        (`bin_index`, `len`). Constant (`min == max`) and all-null columns are omitted
        and handled by the orchestrator (`Profile` object).
    """

    plans: dict[str, pl.LazyFrame] = {}
    schema = lf.collect_schema()

    for column in numeric_columns:
        min_value = profile_data.get(f"{column}_min")
        max_value = profile_data.get(f"{column}_max")

        if min_value is not None and max_value is not None and min_value < max_value:
            bin_size = (max_value - min_value) / bins
            data_type = schema.get(column)

            # 1. float NaNs to Null so they propagate as Null without pre-filtering
            base_col = (
                pl.col(column).fill_nan(None)
                if data_type is not None and data_type.is_float()
                else pl.col(column)
            )

            # 2. Map values to bin index [0, bins - 1]
            bin_index = (
                ((base_col.cast(pl.Float64) - min_value) / bin_size)
                .floor()
                .cast(pl.Int64)
                .clip(0, bins - 1)
                .alias("bin_index")
            )

            # 3. Group without pre-filtering the raw column; drop the single Null group after
            plan = lf.select(bin_index).group_by("bin_index").len().drop_nulls("bin_index")
            plans[column] = plan

    return plans


def build_top_k_plan(lf: pl.LazyFrame, k: int = 10) -> list[pl.LazyFrame]:
    """
    Constructs streaming Top-K frequency plans for all string, categorical, and enum columns.

    Each plan groups non-null string values via `group_by("value").len()` and sorts by
    `["len", "value"]` (`descending=[True, False]`) so frequency ties resolve
    deterministically in alphabetical order.

    Args:
        lf: Input Polars LazyFrame.
        k: Maximum number of frequent values to retain per column.

    Returns:
        A list of `pl.LazyFrame` plans, each yielding a DataFrame with columns
        (`column_name`, `value`, `count`).
    """

    plans: list[pl.LazyFrame] = []
    schema = lf.collect_schema()

    for column_name, data_type in schema.items():
        if data_type in (pl.String, pl.Categorical, pl.Enum):
            # Logic: GroupBy -> Count -> Sort -> Head(k)
            plan = (
                lf.select(pl.col(column_name).cast(pl.String).alias("value"))
                .drop_nulls()  # Prevent null from consuming a Top-K slot
                .group_by("value")
                .len()  # counts the group size
                .sort(["len", "value"], descending=[True, False])
                .head(k)
                .select(
                    pl.lit(column_name).alias("column_name"),
                    pl.col("value"),
                    pl.col("len").alias("count"),
                )
            )
            plans.append(plan)

    return plans


def build_correlation_plan(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Constructs a `pl.LazyFrame` projection isolating numeric columns for correlation analysis.

    Args:
        lf: Input Polars LazyFrame.

    Returns:
        A `pl.LazyFrame` selecting only numeric columns, or an empty `pl.LazyFrame({})`
        if no numeric columns exist in the schema.
    """

    schema = lf.collect_schema()
    numeric_columns = []

    for column_name, data_type in schema.items():
        if data_type.is_numeric():
            numeric_columns.append(column_name)

    # Return an empty plan if no numeric columns exist
    if not numeric_columns:
        return pl.LazyFrame({})

    return lf.select(numeric_columns)


def preprocess_complex_types(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Normalizes nested Polars types (`Struct`, `List`, `Array`) into flat 1D scalar columns.

    Transformation Rules:
        - `pl.Struct`: Unnested into individual fields aliased as `{struct_column}_{field}`
          to prevent column name collisions.
        - `pl.List` / `pl.Array`: Projected to their integer element counts (`{column}_length`)
          so downstream passes profile array length distributions.
        - Primitive Scalars: Passed through unchanged.

    Args:
        lf: Input Polars LazyFrame potentially containing nested columns.

    Returns:
        A `pl.LazyFrame` containing strictly 1D scalar columns.
    """

    schema = lf.collect_schema()

    expressions = []

    for column_name, data_type in schema.items():
        # 1. Handle Structs (Flattening)
        if isinstance(data_type, pl.Struct):
            # We explicitly alias fields to prevent naming collisions
            # e.g. "user" -> "user_name", "user_age"
            struct_fields = data_type.fields
            for field in struct_fields:
                expressions.append(
                    pl.col(column_name)
                    .struct.field(field.name)
                    .alias(f"{column_name}_{field.name}")
                )

        # 2. Handle Lists & Arrays (Length Stats)
        elif isinstance(data_type, pl.List):
            expressions.append(pl.col(column_name).list.len().alias(f"{column_name}_length"))

        elif isinstance(data_type, pl.Array):
            expressions.append(pl.col(column_name).arr.len().alias(f"{column_name}_length"))

        # 3. Pass through everything else (Scalars)
        else:
            expressions.append(pl.col(column_name))

    return lf.select(expressions)
