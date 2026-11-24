"""
Query plan builders and memory-budgeted batching for Netra Profiler.

This module builds the `pl.LazyFrame` query plans used to compute column
statistics, histograms, quantiles, and top-K frequencies, along with the
batching logic that keeps memory usage bounded on large datasets. All plans
are executed by the `Profiler` orchestrator in `profiler.py`.
"""

import math
import os
from typing import Any, cast

import polars as pl


def _get_system_ram_bytes() -> int:
    """
    Auto-detects total physical system RAM in bytes using standard library primitives.

    Design Notes:
        To avoid hardcoding magic numbers that fail on 16 GB laptops or 256 GB enterprise
        servers, Netra dynamically inspects available system memory. It defaults to
        using 25% of total physical RAM as the cumulative memory ceiling for a single
        concurrent execution batch, falling back to 4.0 GiB if OS inspection fails.
    """
    try:
        # Standard Unix/Linux/macOS RAM inspection via os.sysconf
        if (
            hasattr(os, "sysconf")
            and "SC_PHYS_PAGES" in os.sysconf_names
            and "SC_PAGE_SIZE" in os.sysconf_names
        ):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass

    # Safe fallback default (16 GB total RAM -> 4 GB batch budget)
    return 16 * 1024 * 1024 * 1024


# Default Batch Budget: 25% of total system RAM (capped between 2 GiB and 32 GiB)
DEFAULT_BATCH_MEMORY_BYTES = min(
    max(int(_get_system_ram_bytes() * 0.25), 2 * 1024 * 1024 * 1024),
    32 * 1024 * 1024 * 1024,
)

# Split a single column into hash partitions if its estimated hash table exceeds
# 50% of the batch budget, sizing each partition to ~40% of the batch budget.
SINGLE_COLUMN_PARTITION_THRESHOLD_BYTES = int(DEFAULT_BATCH_MEMORY_BYTES * 0.50)
TARGET_PARTITION_MEMORY_BYTES = int(DEFAULT_BATCH_MEMORY_BYTES * 0.40)

BASE_PLAN_OVERHEAD_BYTES = 32 * 1024 * 1024
LOW_CARDINALITY_THRESHOLD = 1_000_000

# Subdivision factor (2^9 = 512) used to expand a 20-bin display histogram
# into 10,240 fine bins for exact quantile pinpointing without rounding error.
FINE_HISTOGRAM_MULTIPLIER = 512
HASH_PARTITION_SEED = 42

# Maximum fraction of valid rows permitted inside the target fine bins before
# falling back from filtered bin sorting to native O(N) quickselect.
MAX_QUANTILE_BIN_ROW_FRACTION = 0.25


def get_datatype_byte_width(data_type: pl.DataType | type[pl.DataType] | None) -> int:
    """
    Returns the in-memory byte width of a Polars primitive data type.

    Defaults to 8 bytes (the width of 64-bit types such as `Int64`, `UInt64`,
    `Float64`, `Datetime`, `Duration`, and `Time`) when `data_type` is `None`.
    """

    if data_type in (pl.Int8, pl.UInt8, pl.Boolean):
        return 1
    if data_type in (pl.Int16, pl.UInt16):
        return 2
    if data_type in (pl.Int32, pl.UInt32, pl.Float32, pl.Date):
        return 4
    if data_type in (pl.Int128, pl.UInt128, pl.Decimal):
        return 16
    return 8


def batch_columns_by_memory(
    column_costs: list[tuple[str, int]],
    budget_bytes: int = DEFAULT_BATCH_MEMORY_BYTES,
    max_columns_per_batch: int = 4,
) -> list[list[str]]:
    """
    Groups columns into execution batches that fit within a target memory budget.

    To profile wide tables efficiently without running out of RAM or thrashing
    disk I/O, this function sorts columns by estimated memory cost (ascending)
    and packs them into batches. A new batch is started whenever adding a column
    would exceed the `budget_bytes` ceiling or reach the `max_columns_per_batch` limit.

    Args:
        column_costs: List of `(column_name, estimated_bytes)` tuples.
        budget_bytes: Maximum total memory allowed for a single batch.
        max_columns_per_batch: Maximum number of columns processed concurrently.

    Returns:
        A list of column name batches (`list[list[str]]`).
    """

    if not column_costs:
        return []

    sorted_costs = sorted(column_costs, key=lambda item: item[1])
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_bytes = 0

    for column_name, cost_bytes in sorted_costs:
        if current_batch and (
            current_bytes + cost_bytes > budget_bytes or len(current_batch) >= max_columns_per_batch
        ):
            batches.append(current_batch)
            current_batch = [column_name]
            current_bytes = cost_bytes
        else:
            current_batch.append(column_name)
            current_bytes += cost_bytes

    if current_batch:
        batches.append(current_batch)

    return batches


def build_scalar_plan(lf: pl.LazyFrame, low_memory: bool = False) -> pl.LazyFrame:
    """
    Builds a single-row streaming `pl.LazyFrame` plan for all scalar column metrics.

    Design Notes:
        - Streaming-Only Calculations: Only constant-memory operations (`len`,
          `null_count`, `approx_n_unique`, `min`, `max`, `mean`, `std`, `skew`,
          `kurtosis`, and string length/blank counts) are included here. Exact
          `n_unique` and quantiles are handled in separate passes so Polars
          never materializes the full table in RAM.
        - Cardinality Pre-Estimation: `approx_n_unique()` runs on every column
          in constant memory using HyperLogLog++. In `low_memory` mode it serves
          as the final cardinality estimate; in standard mode it tells the batch
          scheduler and hash partitioner how much RAM each column's exact
          `n_unique` pass will need.
    """

    expressions: list[pl.Expr] = [pl.len().alias("table_row_count")]

    schema = lf.collect_schema()
    for column_name, data_type in schema.items():
        expressions.append(pl.lit(str(data_type)).alias(f"{column_name}_data_type"))

        column = pl.col(column_name).fill_nan(None) if data_type.is_float() else pl.col(column_name)

        expressions.append(column.null_count().alias(f"{column_name}_null_count"))

        has_nulls = (column.null_count() > 0).cast(pl.UInt32)
        approx_n_unique_expression = (column.approx_n_unique() - has_nulls).alias(
            f"{column_name}_n_unique" if low_memory else f"{column_name}_approx_n_unique"
        )
        expressions.append(approx_n_unique_expression)

        if not low_memory and data_type == pl.Boolean:
            bool_n_unique = (
                pl.when(column.null_count() >= pl.len())
                .then(pl.lit(0, dtype=pl.UInt32))
                .when(column.min() == column.max())
                .then(pl.lit(1, dtype=pl.UInt32))
                .otherwise(pl.lit(2, dtype=pl.UInt32))
                .alias(f"{column_name}_n_unique")
            )
            expressions.append(bool_n_unique)

        if data_type.is_numeric():
            expressions.extend(
                [
                    column.mean().alias(f"{column_name}_mean"),
                    column.min().alias(f"{column_name}_min"),
                    column.max().alias(f"{column_name}_max"),
                    (column == 0).sum().alias(f"{column_name}_zero_count"),
                    column.std().alias(f"{column_name}_std"),
                    column.skew().alias(f"{column_name}_skew"),
                    column.kurtosis().alias(f"{column_name}_kurtosis"),
                ]
            )

        elif data_type.is_temporal():
            expressions.extend(
                [
                    column.min().cast(pl.String).alias(f"{column_name}_min"),
                    column.max().cast(pl.String).alias(f"{column_name}_max"),
                ]
            )

        elif data_type in (pl.String, pl.Categorical, pl.Enum):
            string_column = (
                pl.col(column_name)
                if data_type == pl.String
                else pl.col(column_name).cast(pl.String)
            )
            char_lengths = string_column.str.len_chars()

            expressions.extend(
                [
                    string_column.min().alias(f"{column_name}_min"),
                    string_column.max().alias(f"{column_name}_max"),
                    char_lengths.mean().alias(f"{column_name}_mean_length"),
                    char_lengths.min().alias(f"{column_name}_min_length"),
                    char_lengths.max().alias(f"{column_name}_max_length"),
                    (string_column.str.strip_chars().str.len_bytes() == 0)
                    .sum()
                    .alias(f"{column_name}_blank_count"),
                ]
            )

    return lf.select(expressions)


def build_unique_count_plans(lf: pl.LazyFrame, columns: list[str]) -> list[pl.LazyFrame]:
    """
    Builds one streaming lazy plan per column to count exact non-null unique values.

    Each plan returns a single-row result aliased as `{column_name}_n_unique`.
    Float columns have `NaN` values normalized to `None` before counting.
    """

    plans: list[pl.LazyFrame] = []
    schema = lf.collect_schema()

    for column_name in columns:
        data_type = schema.get(column_name)
        if data_type is None:
            continue
        column_expression = (
            pl.col(column_name).fill_nan(None) if data_type.is_float() else pl.col(column_name)
        )
        plan = (
            lf.select(column_expression.alias("value"))
            .drop_nulls()
            .group_by("value")
            .len()
            .select(pl.len().alias(f"{column_name}_n_unique"))
        )
        plans.append(plan)

    return plans


def build_partitioned_unique_count_plan(
    lf: pl.LazyFrame,
    column_name: str,
    partition_index: int,
    number_of_partitions: int,
    output_alias: str,
) -> pl.LazyFrame:
    """
    Builds a streaming plan to count exact unique values for one hash partition of a column.

    Why Partitioning Works:
        Counting exact unique values on a column with hundreds of millions of distinct
        entries requires a hash table that can easily exceed available RAM. Instead of
        hashing the whole column at once, this function splits the data into
        `number_of_partitions` independent buckets using `hash(value) % number_of_partitions`.

        Because hashing is deterministic, identical values always land in the same bucket,
        and values in different buckets can never be equal. Counting unique values one
        bucket at a time cuts peak hash-table memory by `1 / number_of_partitions`, and
        summing all partition counts produces the 100% exact global unique count with no
        disk spills or approximation error.

    Args:
        lf: Input Polars `LazyFrame`.
        column_name: Column to evaluate.
        partition_index: Index of the bucket to count (`0 <= index < number_of_partitions`).
        number_of_partitions: Total number of hash buckets splitting the column.
        output_alias: Column name for the single-row partition count result.

    Returns:
        A single-row `pl.LazyFrame` containing the exact unique count for this partition.
    """

    schema = lf.collect_schema()
    data_type = schema.get(column_name)
    column_expression = (
        pl.col(column_name).fill_nan(None)
        if data_type is not None and data_type.is_float()
        else pl.col(column_name)
    )

    return (
        lf.select(column_expression.alias("value"))
        .drop_nulls()
        .filter(
            (pl.col("value").hash(seed=HASH_PARTITION_SEED) % number_of_partitions)
            == partition_index
        )
        .group_by("value")
        .len()
        .select(pl.len().alias(output_alias))
    )


def _build_bin_index_expression(
    column_name: str,
    data_type: pl.DataType | None,
    min_value: float | int,
    bin_size: float,
    total_bins: int,
) -> pl.Expr:
    """
    Builds a Polars expression that maps each numeric value to a bin index.

    Values near bin edges are always assigned to the exact same bin, avoiding
    floating-point rounding mismatches. Used by both `build_histogram_plans`
    and `build_quantile_plans`.

    Args:
        column_name: Name of the numeric column to bin.
        data_type: Polars data type of the column (used to normalize float `NaN`s to `None`).
        min_value: Lower bound of the first bin.
        bin_size: Width of each bin.
        total_bins: Total number of bins; indices are clipped to `[0, total_bins - 1]`.

    Returns:
        A `pl.Int64` Polars expression yielding each row's bin index.
    """

    column_expression = (
        pl.col(column_name).fill_nan(None)
        if data_type is not None and data_type.is_float()
        else pl.col(column_name)
    )

    return (
        ((column_expression.cast(pl.Float64) - min_value) / bin_size)
        .floor()
        .cast(pl.Int64)
        .clip(0, total_bins - 1)
    )


def _find_bin_for_rank(
    target_rank: int, sorted_bin_counts: list[tuple[int, int]]
) -> tuple[int, int, int]:
    """
    Finds which histogram bin contains a given position in the sorted column.

    Walks through the non-empty bins in ascending order, accumulating row counts
    until the target sorted position (`target_rank`) is reached.

    Args:
        target_rank: Position in the sorted non-null values (e.g., `n // 2` for a median).
        sorted_bin_counts: Non-empty `(bin_index, bin_count)` pairs sorted ascending by `bin_index`.

    Returns:
        A tuple of `(bin_index, rows_before_bin, bin_count)` identifying the target bin,
        how many values fall in earlier bins, and how many values are in the target bin itself.
    """

    rows_before_bin = 0
    for bin_index, bin_count in sorted_bin_counts:
        if rows_before_bin + bin_count > target_rank:
            return bin_index, rows_before_bin, bin_count
        rows_before_bin += bin_count

    last_bin, last_count = sorted_bin_counts[-1]
    return last_bin, rows_before_bin - last_count, last_count


def _extract_value_at_bin_rank(
    filtered_bins: pl.DataFrame, bin_index: int, rank_in_bin: int
) -> float:
    """
    Returns the value at a given sorted position (`rank_in_bin`) inside one histogram bin.

    When the target position is the first or last element in the bin, `.min()` or
    `.max()` is used directly to avoid an unnecessary sort.
    """

    bin_values = filtered_bins.filter(pl.col("bin_index") == bin_index)["value"]
    if rank_in_bin <= 0:
        return float(cast(float, bin_values.min()))
    if rank_in_bin >= bin_values.len() - 1:
        return float(cast(float, bin_values.max()))
    return float(bin_values.sort()[rank_in_bin])


def extract_quantiles_from_bins(
    filtered_bins: pl.DataFrame, quantile_spec: dict[str, Any]
) -> dict[str, float]:
    """
    Computes exact `p25`, `p50`, and `p75` from the rows that landed in the target bins.

    In the preceding step, `build_quantile_plans` used the fine-grained histogram to
    identify which specific bins contain `p25`, `p50`, and `p75`, then filtered the
    dataset down to only the rows inside those bins (`filtered_bins`, with columns
    `"value"` and `"bin_index"`).

    This function isolates each target bin, sorts its small slice of values, and picks
    the exact element at the precomputed local rank. When the column has an even number
    of non-null rows, the median (`p50`) averages the two middle values whether they
    fall in the same bin or across two different bins.

    Args:
        filtered_bins: Collected `pl.DataFrame` containing only the target quantile bins.
        quantile_spec: Bin and rank coordinates produced by `build_quantile_plans`.

    Returns:
        A dictionary mapping `{column_name}_p25`, `{column_name}_p50`, and
        `{column_name}_p75` to their exact float values.
    """

    column_name: str = quantile_spec["column_name"]
    p25 = _extract_value_at_bin_rank(
        filtered_bins,
        quantile_spec["p25_bin"],
        quantile_spec["p25_rank_in_bin"],
    )

    if quantile_spec["p50_low_rank"] == quantile_spec["p50_high_rank"]:
        p50 = _extract_value_at_bin_rank(
            filtered_bins,
            quantile_spec["p50_low_bin"],
            quantile_spec["p50_low_rank_in_bin"],
        )
    elif quantile_spec["p50_low_bin"] == quantile_spec["p50_high_bin"]:
        sorted_bin_values = filtered_bins.filter(
            pl.col("bin_index") == quantile_spec["p50_low_bin"]
        )["value"].sort()
        low_rank_in_bin = quantile_spec["p50_low_rank_in_bin"]
        p50 = 0.5 * (
            float(sorted_bin_values[low_rank_in_bin])
            + float(sorted_bin_values[low_rank_in_bin + 1])
        )
    else:
        low_value = _extract_value_at_bin_rank(
            filtered_bins,
            quantile_spec["p50_low_bin"],
            quantile_spec["p50_low_rank_in_bin"],
        )
        high_value = _extract_value_at_bin_rank(
            filtered_bins,
            quantile_spec["p50_high_bin"],
            quantile_spec["p50_high_rank_in_bin"],
        )
        p50 = 0.5 * (low_value + high_value)

    p75 = _extract_value_at_bin_rank(
        filtered_bins,
        quantile_spec["p75_bin"],
        quantile_spec["p75_rank_in_bin"],
    )

    return {
        f"{column_name}_p25": p25,
        f"{column_name}_p50": p50,
        f"{column_name}_p75": p75,
    }


def _build_quantile_bin_spec(
    lf: pl.LazyFrame,
    column_name: str,
    data_type: pl.DataType | None,
    profile_data: dict[str, Any],
    bins: int,
) -> dict[str, Any] | None:
    """
    Builds a streaming bin-filter plan and rank lookup spec for one numeric column.

    Uses the fine-grained histogram counts from Stage 2 to pinpoint the exact bins
    containing `p25`, `p50`, and `p75`, then builds a lightweight filter plan that
    retains only rows inside those bins.

    Returns `None` when fine-bin counts are missing or when a heavy spike places
    more than `MAX_QUANTILE_BIN_ROW_FRACTION` (25%) of valid rows into the target bins,
    signaling `build_quantile_plans` to fall back to native Polars quantiles.
    """

    row_count = profile_data.get("table_row_count", 0)
    null_count = profile_data.get(f"{column_name}_null_count", 0)
    valid_row_count = max(1, row_count - null_count)
    fine_bin_counts: dict[int, int] | None = profile_data.get(f"{column_name}_fine_counts")

    if not fine_bin_counts or sum(fine_bin_counts.values()) != valid_row_count:
        return None

    sorted_bin_counts = sorted(fine_bin_counts.items(), key=lambda item: item[0])

    # Match Polars' defaults: quantile(interpolation="nearest")
    # for p25/p75, linear midpoint for median.
    p25_rank = int(math.floor((valid_row_count - 1) * 0.25 + 0.5))
    p50_low_rank = (valid_row_count - 1) // 2
    p50_high_rank = valid_row_count // 2
    p75_rank = int(math.floor((valid_row_count - 1) * 0.75 + 0.5))

    p25_bin, p25_rows_before, p25_bin_count = _find_bin_for_rank(p25_rank, sorted_bin_counts)
    p50_low_bin, p50_low_rows_before, p50_low_bin_count = _find_bin_for_rank(
        p50_low_rank, sorted_bin_counts
    )
    p50_high_bin, p50_high_rows_before, p50_high_bin_count = _find_bin_for_rank(
        p50_high_rank, sorted_bin_counts
    )
    p75_bin, p75_rows_before, p75_bin_count = _find_bin_for_rank(p75_rank, sorted_bin_counts)

    target_bin_counts = {
        p25_bin: p25_bin_count,
        p50_low_bin: p50_low_bin_count,
        p50_high_bin: p50_high_bin_count,
        p75_bin: p75_bin_count,
    }
    selected_row_count = sum(target_bin_counts.values())

    if selected_row_count > max(100_000, int(MAX_QUANTILE_BIN_ROW_FRACTION * valid_row_count)):
        return None

    min_value = profile_data[f"{column_name}_min"]
    max_value = profile_data[f"{column_name}_max"]
    fine_bin_size = ((max_value - min_value) / bins) / FINE_HISTOGRAM_MULTIPLIER
    total_fine_bins = bins * FINE_HISTOGRAM_MULTIPLIER

    column_expression = (
        pl.col(column_name).fill_nan(None)
        if data_type is not None and data_type.is_float()
        else pl.col(column_name)
    )
    bin_index_expression = _build_bin_index_expression(
        column_name, data_type, min_value, fine_bin_size, total_fine_bins
    )

    bin_index_column = pl.col("bin_index")
    target_bins = sorted(target_bin_counts.keys())
    filter_condition = bin_index_column == target_bins[0]
    for target_bin in target_bins[1:]:
        filter_condition = filter_condition | (bin_index_column == target_bin)

    plan = lf.select(
        column_expression.alias("value"),
        bin_index_expression.alias("bin_index"),
    ).filter(filter_condition)

    return {
        "column_name": column_name,
        "plan": plan,
        "p25_bin": p25_bin,
        "p25_rank_in_bin": p25_rank - p25_rows_before,
        "p50_low_rank": p50_low_rank,
        "p50_high_rank": p50_high_rank,
        "p50_low_bin": p50_low_bin,
        "p50_low_rank_in_bin": p50_low_rank - p50_low_rows_before,
        "p50_high_bin": p50_high_bin,
        "p50_high_rank_in_bin": p50_high_rank - p50_high_rows_before,
        "p75_bin": p75_bin,
        "p75_rank_in_bin": p75_rank - p75_rows_before,
    }


def build_quantile_plans(
    lf: pl.LazyFrame,
    numeric_columns: list[str],
    profile_data: dict[str, Any],
    budget_bytes: int = DEFAULT_BATCH_MEMORY_BYTES,
    bins: int = 20,
) -> tuple[list[dict[str, Any]], list[pl.LazyFrame]]:
    """
    Builds streaming quantile filter specs and fallback quickselect plans for numeric columns.

    Design Notes:
        - Why `.sort()` stays out of the lazy plan: Including `.sort()` inside `.select()`
          forces Polars to materialize the entire column in RAM instead of streaming it.
          By generating a pure `select` + `filter` plan, Polars streams the column in
          constant memory, discards over 99.9% of rows on the fly, and returns only the
          small slice of rows inside the target bins for `extract_quantiles_from_bins`.
        - Spike Fallback: Any column where a discrete spike concentrates too many rows into
          the target bins falls back to `.quantile()` / `.median()` batches.

    Returns:
        A tuple of `(quantile_bin_specs, fallback_plans)`:
        - `quantile_bin_specs`: Per-column plan and rank dictionaries for fine-bin extraction.
        - `fallback_plans`: Memory-budgeted native quantile plans for spiked columns.
    """

    schema = lf.collect_schema()
    row_count = profile_data.get("table_row_count", 0)
    quantile_bin_specs: list[dict[str, Any]] = []
    fallback_costs: list[tuple[str, int]] = []

    for column_name in numeric_columns:
        min_value = profile_data.get(f"{column_name}_min")
        max_value = profile_data.get(f"{column_name}_max")
        if min_value is None or max_value is None or min_value >= max_value:
            continue

        data_type = schema.get(column_name)
        spec = _build_quantile_bin_spec(lf, column_name, data_type, profile_data, bins)
        if spec is not None:
            quantile_bin_specs.append(spec)
        else:
            null_count = profile_data.get(f"{column_name}_null_count", 0)
            valid_row_count = max(1, row_count - null_count)
            byte_width = get_datatype_byte_width(data_type)
            estimated_bytes = BASE_PLAN_OVERHEAD_BYTES + int(valid_row_count * byte_width * 1.6)
            fallback_costs.append((column_name, estimated_bytes))

    fallback_batches = batch_columns_by_memory(
        fallback_costs, budget_bytes=budget_bytes, max_columns_per_batch=4
    )
    fallback_plans: list[pl.LazyFrame] = []

    for batch_columns in fallback_batches:
        expressions: list[pl.Expr] = []
        for column_name in batch_columns:
            data_type = schema.get(column_name)
            column_expression = (
                pl.col(column_name).fill_nan(None)
                if data_type is not None and data_type.is_float()
                else pl.col(column_name)
            )
            expressions.extend(
                [
                    column_expression.quantile(0.25).alias(f"{column_name}_p25"),
                    column_expression.median().alias(f"{column_name}_p50"),
                    column_expression.quantile(0.75).alias(f"{column_name}_p75"),
                ]
            )
        if expressions:
            fallback_plans.append(lf.select(expressions))

    return quantile_bin_specs, fallback_plans


def build_histogram_plans(
    lf: pl.LazyFrame,
    numeric_columns: list[str],
    profile_data: dict[str, Any],
    bins: int,
    fine_multiplier: int = 1,
) -> dict[str, pl.LazyFrame]:
    """
    Builds one streaming histogram plan per non-constant numeric column.

    Each plan maps values into equi-width bin indices and counts non-null rows per bin.
    In standard mode, `fine_multiplier` subdivides each display bin (e.g., 20 x 512 = 10,240
    fine bins) so a single scan produces both the display histogram and the fine bin counts
    needed for the calculation of quantiles.

    Args:
        lf: Input Polars `LazyFrame`.
        numeric_columns: Numeric columns to consider.
        profile_data: Stage 1 metrics containing `{column}_min` and `{column}_max`.
        bins: Number of display histogram bins.
        fine_multiplier: Number of fine sub-bins per display bin (default: 1).

    Returns:
        A mapping of `column_name -> LazyFrame` with columns `"bin_index"` and `"len"`.
    """

    plans: dict[str, pl.LazyFrame] = {}
    schema = lf.collect_schema()
    total_bins = bins * fine_multiplier

    for column_name in numeric_columns:
        min_value = profile_data.get(f"{column_name}_min")
        max_value = profile_data.get(f"{column_name}_max")

        if min_value is not None and max_value is not None and min_value < max_value:
            coarse_bin_size = (max_value - min_value) / bins
            bin_size = coarse_bin_size / fine_multiplier if fine_multiplier > 1 else coarse_bin_size
            data_type = schema.get(column_name)

            bin_index = _build_bin_index_expression(
                column_name, data_type, min_value, bin_size, total_bins
            ).alias("bin_index")

            plan = lf.select(bin_index).drop_nulls().group_by("bin_index").len()
            plans[column_name] = plan

    return plans


def build_top_k_plans(lf: pl.LazyFrame, k: int = 10) -> dict[str, pl.LazyFrame]:
    """
    Builds streaming Top-K frequency and exact unique-count plans for text columns.

    Design Notes:
        - Dictionary Preservation: `Categorical` and `Enum` columns are grouped on their
          integer codes first and cast to `pl.String` only after aggregation.
        - Piggybacked Exact `n_unique`: The total number of distinct groups (`pl.len()`)
          is recorded right after `group_by`, giving exact string cardinality for free
          before taking the top `k` rows.
        - Slice Pushdown: Computing `n_unique` before `.sort()` lets Polars push `.head(k)`
          into the sort, avoiding a full sort across all unique values.

    Returns:
        A mapping of `column_name -> LazyFrame` with columns `"value"`, `"count"`,
        and `"n_unique"`.
    """

    plans: dict[str, pl.LazyFrame] = {}
    schema = lf.collect_schema()

    for column_name, data_type in schema.items():
        if data_type in (pl.String, pl.Categorical, pl.Enum):
            plan = (
                lf.select(pl.col(column_name).alias("value"))
                .drop_nulls()
                .group_by("value")
                .len()
                .with_columns(
                    pl.col("value").cast(pl.String),
                    pl.len().alias("n_unique"),
                )
                .sort(["len", "value"], descending=[True, False])
                .head(k)
                .select(
                    pl.col("value"),
                    pl.col("len").alias("count"),
                    pl.col("n_unique"),
                )
            )
            plans[column_name] = plan

    return plans


def build_correlation_sample_plan(
    lf: pl.LazyFrame,
    columns: list[str],
    row_count: int,
    sample_size: int = 100_000,
) -> tuple[pl.LazyFrame, str]:
    """
    Builds a lazy plan that extracts a bounded numeric sample for correlation analysis.

    Datasets at or below `sample_size` rows are kept in full (`"exact"`); larger
    datasets use deterministic systematic sampling (`gather_every`) to cap memory
    usage. All selected columns are cast to `Float64` with `NaN` values normalized
    to `None`. Pairwise Pearson and Spearman correlations are computed on the
    collected sample in `profiler.py`.

    Args:
        lf: Input Polars `LazyFrame`.
        columns: Non-constant numeric columns to include.
        row_count: Total row count of the dataset from Stage 1.
        sample_size: Target maximum number of rows to materialize (default: 100,000).

    Returns:
        A tuple of `(plan, sampling_method)` where `sampling_method` is `"exact"` or
        `"systematic_sample (~{sample_size})"`.
    """

    plan = lf.select(columns)
    if row_count > sample_size:
        step_size = max(1, row_count // sample_size)
        plan = plan.gather_every(step_size)
        sampling_method = f"systematic_sample (~{sample_size})"
    else:
        sampling_method = "exact"

    plan = plan.select(pl.all().cast(pl.Float64).fill_nan(None))
    return plan, sampling_method


def preprocess_complex_types(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Flattens nested Polars columns into 1D scalar columns that the profiler can analyze.

    Transformations:
        - `Struct` columns are unpacked into one column per field (`{column}_{field}`).
        - `List` and fixed-size `Array` columns are replaced by their element counts
          (`{column}_length`).
        - Flat datasets with no nested columns are returned unchanged.
    """

    schema = lf.collect_schema()
    has_complex_types = any(
        isinstance(data_type, (pl.Struct, pl.List, pl.Array)) for data_type in schema.values()
    )
    if not has_complex_types:
        return lf

    expressions: list[pl.Expr] = []
    for column_name, data_type in schema.items():
        if isinstance(data_type, pl.Struct):
            for field in data_type.fields:
                expressions.append(
                    pl.col(column_name)
                    .struct.field(field.name)
                    .alias(f"{column_name}_{field.name}")
                )
        elif isinstance(data_type, pl.List):
            expressions.append(pl.col(column_name).list.len().alias(f"{column_name}_length"))
        elif isinstance(data_type, pl.Array):
            expressions.append(pl.col(column_name).arr.len().alias(f"{column_name}_length"))
        else:
            expressions.append(pl.col(column_name))

    return lf.select(expressions)
