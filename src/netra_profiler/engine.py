from typing import Any

import polars as pl


def build_scalar_plan(lf: pl.LazyFrame, low_memory: bool = False) -> pl.LazyFrame:
    """
    PASS 1: Scalar Statistics Plan Generation.

    Constructs a unified lazy execution graph to compute all single-value statistical
    metrics (e.g., Min, Max, Mean, Nulls, Quantiles) concurrently across all columns.

    By bundling all column-level aggregations into a single `pl.select()` context,
    this architecture allows the Polars query optimizer to evaluate the entire dataset
    in a single, highly-threaded streaming pass, minimizing disk I/O and memory overhead.

    Args:
        lf: The input Polars LazyFrame containing the dataset.
        low_memory: A boolean flag that toggles bounded-memory execution to safely
            process massive datasets (e.g., 500M+ rows).
            - If False (Default): Computes exact `n_unique()`, exact quantiles, and exact
                moments (skew/kurtosis). This generates global hash-sets and sorting buffers,
                which will trigger OOM crashes on highly cardinal or large-scale data.
            - If True: Swaps to the HyperLogLog algorithm (`approx_n_unique()`) for
                cardinality, and entirely excludes global sorting/multi-pass operations
                (`median`, `quantile`, `skew`, `kurtosis`). This guarantees a strict,
                lower bounded RAM footprint.

    Returns:
        A `pl.LazyFrame` that, when collected, yields a 1-row wide DataFrame containing
        all computed scalar statistics. Column names are strictly aliased to prevent
        collisions (e.g., `age_mean`, `age_null_count`).

    Notes:
        - Floating-Point NaN Coercion: Polars treats `NaN` and `Null` differently. To ensure
            accurate missing-value counts, `NaN` values in float columns are explicitly
            coerced to `Null` prior to aggregation.
        - Eager Fallback Prevention: When `low_memory` is False, the exact `.median()`,
            `.quantile()`, `.skew()`, and `.kurtosis()` aggregations inherently require multi-pass
            evaluation or global data sorting. This forces the Polars streaming engine to silently
            fall back to eager memory materialization for those specific nodes.
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

        # We handle NaN for Float columns before doing Null counts
        column = pl.col(column_name).fill_nan(None) if data_type.is_float() else pl.col(column_name)

        n_unique_expression = (
            column.approx_n_unique().alias(f"{column_name}_n_unique")
            if low_memory
            else column.n_unique().alias(f"{column_name}_n_unique")
        )

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

        # String/Categorical Stats
        elif data_type in (pl.String, pl.Categorical, pl.Enum):
            # We cast column to String for .str operations
            column = pl.col(column_name).cast(pl.String)

            expressions.extend(
                [
                    # 1. Lexicographical stats (First/Last alphabetical value)
                    column.min().alias(f"{column_name}_min"),
                    column.max().alias(f"{column_name}_max"),
                    # 2. Length stats
                    column.str.len_chars().mean().alias(f"{column_name}_mean_length"),
                    column.str.len_chars().min().alias(f"{column_name}_min_length"),
                    column.str.len_chars().max().alias(f"{column_name}_max_length"),
                ]
            )

    # 3. Construct the Query Plan
    return lf.select(expressions)


def build_histogram_plans(
    lf: pl.LazyFrame, numeric_columns: list[str], profile_data: dict[str, Any], bins: int
) -> dict[str, pl.LazyFrame]:
    """
    PASS 2: Histogram Map-Reduce Plan Generation.

    Constructs a dictionary of lazy execution plans to compute frequency distributions
    for numeric columns using a streaming-compatible Map-Reduce architecture.

    This approach maps continuous values to discrete bin indices and aggregates them using
    `group_by().len()`. This guarantees bounded memory usage and prevents Out-Of-Memory (OOM)
    crashes on massive datasets by keeping the evaluation entirely within the lazy execution graph.

    Args:
        lf: The input Polars LazyFrame containing the dataset.
        numeric_columns: A list of column names identified as numeric (integers/floats).
        profile_data: The dictionary containing Pass 1 scalar statistics. Used to
            retrieve the pre-calculated `min` and `max` boundaries for each column.
        bins: The target number of intervals to divide the data range into.

    Returns:
        A dictionary mapping column names to their respective `pl.LazyFrame` execution
        plans. Each plan yields a small 2-column DataFrame (`bin_index`, `len`) when collected.

    Notes:
        - Right-Edge Clamping: Values exactly equal to the `max_value` are clamped to the
            final bin (`bins - 1`) to prevent floating-point division from creating an
            accidental out-of-bounds index.
        - Null/NaN Safety: Missing and undefined values are explicitly filtered out
            prior to mathematical mapping to maintain structural integrity.
    """
    plans: dict[str, pl.LazyFrame] = {}

    for column in numeric_columns:
        min_value = profile_data.get(f"{column}_min")
        max_value = profile_data.get(f"{column}_max")

        if min_value is not None and max_value is not None and min_value < max_value:
            bin_size = (max_value - min_value) / bins

            # 1. Cast to Float64 so we can safely evaluate the math and check for NaNs
            expression = pl.col(column).cast(pl.Float64)

            # 2. Map values to a bin index: floor((val - min) / bin_size)
            bin_index = ((expression - min_value) / bin_size).floor().cast(pl.Int64)

            # 3. We clamp the absolute max value so it doesn't spill into an out-of-bounds bin
            bin_index = (
                pl.when(bin_index >= bins).then(bins - 1).otherwise(bin_index).alias("bin_index")
            )

            # 4. The Streaming Plan: Filter -> Math -> Group
            plan = (
                lf.select(expression)
                .filter(pl.col(column).is_not_null() & pl.col(column).is_not_nan())
                .select(bin_index)
                .group_by("bin_index")
                .len()
            )
            plans[column] = plan

    return plans


def build_top_k_plan(lf: pl.LazyFrame, k: int = 10) -> list[pl.LazyFrame]:
    """
    PASS 3: Top-K Frequent Items Plan Generation.

    Constructs a list of lazy execution plans to identify the `k` most frequently
    occurring values for all text-based columns (String, Categorical, Enum).

    This plan leverages a streaming Map-Reduce pattern (`group_by().len()`) followed
    by an optimized Top-K reduction (`sort().head()`). This allows the engine to compute
    exact string frequencies without pulling massive, high-cardinality text arrays into
    Python's memory, ensuring large datasets remain safely bounded.

    Args:
        lf: The input Polars LazyFrame containing the dataset.
        k: The maximum number of frequent items to return per column (default: 10).

    Returns:
        A list of `pl.LazyFrame` execution plans. When collected, each plan yields
        a DataFrame containing exactly three columns: `column_name`, the string `value`,
        and its occurrence `count`.

    Notes:
        - Type Normalization: Categoricals and Enums are explicitly cast to standard
            Strings early in the execution graph to guarantee safe and consistent JSON
            serialization in the final Data Contract.
        - Null Exclusion: Missing values (`null`) are explicitly dropped prior to
            aggregation.
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
                .sort("len", descending=True)
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
    PASS 4: Correlation Data Fetcher Plan Generation.

    Constructs a lazy execution plan that isolates and selects strictly numeric
    columns from the dataset. This acts as the foundational type-safety filter
    before Pearson and Spearman correlation matrices are computed.

    Unlike the scalar and histogram passes, the actual mathematical computation
    of the correlation matrix (and rank calculations for Spearman) is deliberately
    deferred to the orchestrator (`profiler.py`). Computing a global correlation
    matrix requires eager materialization and cannot currently be fully streamed
    natively in Polars. By returning just the isolated numeric blueprint here,
    the orchestrator can safely apply systematic sampling before pulling the data
    into physical memory, preventing OOM crashes on massive datasets.

    Args:
        lf: The input Polars LazyFrame containing the full dataset.

    Returns:
        A `pl.LazyFrame` execution plan that selects only columns with numeric
        data types (Integers, Floats). If the dataset contains no numeric
        columns, it safely returns an empty `pl.LazyFrame({})` to gracefully
        bypass the correlation pass without triggering downstream panics.
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
    PRE-PROCESSING: Complex Type Normalization.

    Transforms nested and multi-dimensional data types (Structs, Lists, Arrays) into
    flat, 1D scalar columns prior to execution.

    The core statistical and correlation steps (Passes 1-4) operate
    exclusively on primitive scalar types (Integers, Floats, Strings). Nested data cannot
    be passed directly into mathematical aggregations would cause the lazy execution graph.
    This function intercepts the data source and projects it into a flat schema, ensuring
    engine stability regardless of the source file's complexity.

    Args:
        lf: The raw input Polars LazyFrame containing potentially nested data.

    Returns:
        A transformed `pl.LazyFrame` where all columns are guaranteed to be 1D primitives.

    Transformation Rules:
        - Structs: Flattened into individual columns. A Struct named 'user' with fields
          'name' and 'age' is unnested into two columns: 'user_name' and 'user_age'.
          This strict namespace aliasing prevents downstream naming collisions.
        - Lists & Arrays: Replaced by their integer lengths. A list column named
          'tags' becomes an integer column 'tags_length', allowing the engine to
          profile the statistical distribution of the array sizes rather than failing
          on the nested arrays.
        - Scalars: All primitive types are passed through the graph unmodified.
    """
    schema = lf.collect_schema()

    # We will build a list of expressions to select/transform
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
