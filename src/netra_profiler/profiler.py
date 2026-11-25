import datetime
import math
import time
import warnings
from typing import Any, cast

import polars as pl

from netra_profiler import __version__, engine
from netra_profiler.config import DiagnosticConfig
from netra_profiler.diagnostics import DiagnosticEngine
from netra_profiler.types import NetraProfile, is_numeric_type, is_string_type

CORRELATION_SAMPLE_SIZE = 100_000

# Disable common subplan elimination so Polars does not insert an in-memory
# CACHE node across sibling streaming plans evaluated via `pl.collect_all()`.
NO_CSE_OPTIMIZATIONS = pl.QueryOptFlags(comm_subplan_elim=False)


class Profiler:
    """
    Orchestrates dataset profiling from a Polars `DataFrame` or `LazyFrame`.

    Execution Pipeline:
        1. Schema Normalization (`__init__`): Converts eager inputs to a `LazyFrame`,
           drops ignored columns, and flattens nested `Struct`, `List`, and `Array` columns.
        2. Stage 1 — Scalars, Top-K & Exact Unique Counts (`_collect_scalars_and_top_k`):
           Runs the streaming scalar plan alongside string Top-K plans (which also return
           exact string `n_unique`), then dispatches memory-budgeted and hash-partitioned
           exact unique counts for numeric and temporal columns.
        3. Stage 2 — Histograms, Correlations & Exact Quantiles
           (`_collect_histograms_and_correlations`): Collects fine-grained histograms
           (downsampled to `bins` display intervals) alongside a bounded correlation
           sample, then filters only the target histogram bins to compute exact `p25`,
           `p50`, and `p75` in streaming memory.
        4. Assembly & Diagnostics (`_build_profile`, `_run_diagnostics`): Builds the typed
           `NetraProfile` report and evaluates data quality rules.
    """

    def __init__(  # noqa: PLR0913
        self,
        df: pl.DataFrame | pl.LazyFrame,
        dataset_name: str = "Unknown",
        dataset_format: str = "Unknown",
        ignore_columns: list[str] | None = None,
        low_memory: bool = False,
        config: dict[str, Any] | None = None,
        config_source: str = "Default",
    ):
        """
        Initializes the profiling session and normalizes the input schema.

        Args:
            df: Input Polars `DataFrame` or `LazyFrame`.
            dataset_name: Identifier or filename for the dataset.
            dataset_format: Source format label (e.g., `'Parquet'`, `'CSV'`).
            ignore_columns: Column names to exclude from profiling.
            low_memory: If True, uses approximate HyperLogLog cardinality (`approx_n_unique`)
                and skips exact quantiles to keep memory usage constant.
            config: Optional dictionary of threshold overrides parsed from `netra_config.yaml`.
            config_source: Label indicating where the configuration was loaded from.
        """

        if isinstance(df, pl.DataFrame):
            self._df = df.lazy()
        elif isinstance(df, pl.LazyFrame):
            self._df = df
        else:
            raise TypeError(f"Unsupported type: {type(df)}. Must be pl.DataFrame or pl.LazyFrame")

        if ignore_columns:
            self._df = self._df.drop(ignore_columns)

        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.low_memory = low_memory
        self.config_source = config_source

        self.diagnostic_config = DiagnosticConfig(config)
        self._df = engine.preprocess_complex_types(self._df)

    def run(self, bins: int = 20, top_k: int = 10) -> NetraProfile:
        """
        Executes the profiling pipeline and returns the finalized `NetraProfile` report.

        Args:
            bins: Number of equi-width intervals for numeric histograms (default: 20).
            top_k: Maximum number of frequent values to retain for text columns (default: 10).

        Returns:
            A typed, JSON-serializable `NetraProfile` dictionary containing dataset metadata,
            column statistics, correlation edges, diagnostic alerts, and execution telemetry.
        """

        profiling_start_epoch = time.time()
        profiling_start_perf = time.perf_counter()

        profiler_warnings: list[str] = []

        # Stage 1: Scalar metrics, string Top-K tables, and exact unique counts
        profile_data = self._collect_scalars_and_top_k(top_k)

        # Stage 2: Histograms, correlation sample, and bin-filtered exact quantiles
        self._collect_histograms_and_correlations(profile_data, bins, profiler_warnings)

        # Assemble NetraProfile object
        profile = self._build_profile(
            profile_data,
            profiling_start_epoch,
            profiling_start_perf,
            profiler_warnings,
        )

        # Evaluate quality diagnostics
        self._run_diagnostics(profile)

        return profile

    def _collect_scalars_and_top_k(self, top_k: int) -> dict[str, Any]:
        """
        Collects streaming scalar metrics, string Top-K tables, and exact unique counts.

        In standard mode, numeric and temporal columns are scheduled in three tiers using
        Stage 1's `approx_n_unique` estimate:
            1. Analytical bypass for all-null or constant (`min == max`) columns.
            2. Memory-budgeted batches for columns whose hash tables fit within
               `engine.SINGLE_COLUMN_PARTITION_THRESHOLD_BYTES`.
            3. Deterministic hash partitioning (`hash(value) % P == p`) for extreme-cardinality
               columns whose single-column hash tables exceed the partition threshold.
        """

        scalar_plan = engine.build_scalar_plan(self._df, low_memory=self.low_memory)
        top_k_plans = engine.build_top_k_plans(self._df, k=top_k)
        top_k_columns = list(top_k_plans.keys())

        initial_results = pl.collect_all(
            [scalar_plan, *top_k_plans.values()],
            engine="streaming",
            optimizations=NO_CSE_OPTIMIZATIONS,
        )

        scalar_df = initial_results[0]
        profile_data: dict[str, Any] = scalar_df.rows(named=True)[0]
        row_count = profile_data.get("table_row_count", 0)

        for column_name, top_k_df in zip(top_k_columns, initial_results[1:], strict=True):
            if top_k_df.height > 0:
                profile_data[f"{column_name}_top_k"] = top_k_df.select("value", "count").to_dicts()
                if not self.low_memory:
                    profile_data[f"{column_name}_n_unique"] = int(top_k_df["n_unique"][0])
            elif not self.low_memory:
                profile_data[f"{column_name}_n_unique"] = 0

        if not self.low_memory:
            schema = self._df.collect_schema()
            self._collect_exact_unique_counts(profile_data, schema, row_count)

        return profile_data

    def _collect_exact_unique_counts(
        self,
        profile_data: dict[str, Any],
        schema: pl.Schema,
        row_count: int,
    ) -> None:
        """
        Classifies numeric and temporal columns by estimated hash-table memory and
        dispatches memory-budgeted or hash-partitioned exact `n_unique` passes.
        """

        low_cardinality_columns: list[str] = []
        high_cardinality_costs: list[tuple[str, int]] = []
        partitioned_tasks: dict[str, pl.LazyFrame] = {}
        partitioned_costs: list[tuple[str, int]] = []

        for column_name, data_type in schema.items():
            if not (data_type.is_numeric() or data_type.is_temporal()):
                continue

            null_count = profile_data.get(f"{column_name}_null_count", 0)
            min_value = profile_data.get(f"{column_name}_min")
            max_value = profile_data.get(f"{column_name}_max")
            approx_unique_count = int(profile_data.get(f"{column_name}_approx_n_unique", 0))
            byte_width = engine.get_datatype_byte_width(data_type)
            hash_table_bytes = approx_unique_count * (byte_width + 8)
            estimated_bytes = engine.BASE_PLAN_OVERHEAD_BYTES + hash_table_bytes

            if null_count >= row_count or min_value is None or max_value is None:
                profile_data[f"{column_name}_n_unique"] = 0
            elif min_value == max_value or (row_count - null_count) == 1:
                profile_data[f"{column_name}_n_unique"] = 1
            elif approx_unique_count <= engine.LOW_CARDINALITY_THRESHOLD:
                low_cardinality_columns.append(column_name)
            elif estimated_bytes <= engine.SINGLE_COLUMN_PARTITION_THRESHOLD_BYTES:
                high_cardinality_costs.append((column_name, estimated_bytes))
            else:
                number_of_partitions = max(
                    2, math.ceil(hash_table_bytes / engine.TARGET_PARTITION_MEMORY_BYTES)
                )
                partition_cost = engine.BASE_PLAN_OVERHEAD_BYTES + (
                    (approx_unique_count // number_of_partitions) * (byte_width + 8)
                )
                profile_data[f"{column_name}_n_unique"] = 0

                for partition_index in range(number_of_partitions):
                    task_key = f"{column_name}__part_{partition_index}"
                    partitioned_tasks[task_key] = engine.build_partitioned_unique_count_plan(
                        self._df, column_name, partition_index, number_of_partitions, task_key
                    )
                    partitioned_costs.append((task_key, partition_cost))

        high_cardinality_batches = engine.batch_columns_by_memory(
            high_cardinality_costs, max_columns_per_batch=3
        )
        standard_batches = (
            [low_cardinality_columns] if low_cardinality_columns else []
        ) + high_cardinality_batches
        self._execute_unique_count_batches(standard_batches, profile_data)

        if partitioned_costs:
            self._execute_partitioned_unique_counts(
                partitioned_tasks, partitioned_costs, profile_data
            )

    def _execute_unique_count_batches(
        self,
        column_batches: list[list[str]],
        profile_data: dict[str, Any],
    ) -> None:
        """
        Executes unpartitioned exact `n_unique` plans for each memory-budgeted batch.
        """

        for batch_columns in column_batches:
            batch_plans = engine.build_unique_count_plans(self._df, batch_columns)
            for result_df in pl.collect_all(
                batch_plans, engine="streaming", optimizations=NO_CSE_OPTIMIZATIONS
            ):
                if result_df.height > 0:
                    profile_data.update(result_df.row(0, named=True))

    def _execute_partitioned_unique_counts(
        self,
        partitioned_tasks: dict[str, pl.LazyFrame],
        partitioned_costs: list[tuple[str, int]],
        profile_data: dict[str, Any],
    ) -> None:
        """
        Executes hash-partitioned exact `n_unique` sub-tasks and sums partition counts per column.
        """

        for batch_keys in engine.batch_columns_by_memory(
            partitioned_costs, max_columns_per_batch=3
        ):
            partition_plans = [partitioned_tasks[key] for key in batch_keys]
            for result_df in pl.collect_all(
                partition_plans, engine="streaming", optimizations=NO_CSE_OPTIMIZATIONS
            ):
                if result_df.height > 0:
                    for task_key, partition_count in result_df.row(0, named=True).items():
                        column_name = task_key.rsplit("__part_", maxsplit=1)[0]
                        profile_data[f"{column_name}_n_unique"] += int(partition_count)

    def _collect_histograms_and_correlations(
        self, profile_data: dict[str, Any], bins: int, profiler_warnings: list[str]
    ) -> None:
        """
        Collects numeric histograms, correlation samples, and exact quantiles in Stage 2.
        """

        schema = self._df.collect_schema()
        numeric_columns = [
            column_name for column_name, data_type in schema.items() if data_type.is_numeric()
        ]

        fine_multiplier = 1 if self.low_memory else engine.FINE_HISTOGRAM_MULTIPLIER
        histogram_plans = engine.build_histogram_plans(
            self._df, numeric_columns, profile_data, bins, fine_multiplier=fine_multiplier
        )
        histogram_columns = list(histogram_plans.keys())
        stage_2_plans: list[pl.LazyFrame] = list(histogram_plans.values())

        row_count = profile_data.get("table_row_count", 0)
        valid_correlation_columns = [
            column_name
            for column_name in numeric_columns
            if profile_data.get(f"{column_name}_n_unique", 0) > 1
        ]
        has_correlation_plan = row_count > 1 and len(valid_correlation_columns) > 1
        correlations_sampling_method: str | None = None

        if has_correlation_plan:
            correlation_plan, correlations_sampling_method = engine.build_correlation_sample_plan(
                self._df,
                valid_correlation_columns,
                row_count,
                sample_size=CORRELATION_SAMPLE_SIZE,
            )
            stage_2_plans.append(correlation_plan)

        if stage_2_plans:
            stage_2_results = pl.collect_all(
                stage_2_plans, engine="streaming", optimizations=NO_CSE_OPTIMIZATIONS
            )

            histogram_results = stage_2_results[: len(histogram_columns)]
            self._build_histograms(
                histogram_columns,
                histogram_results,
                numeric_columns,
                histogram_plans,
                profile_data,
                bins,
                profiler_warnings,
                fine_multiplier=fine_multiplier,
            )

            if has_correlation_plan:
                correlation_df = stage_2_results[-1]
                self._compute_correlations(
                    correlation_df,
                    valid_correlation_columns,
                    correlations_sampling_method,
                    profile_data,
                    profiler_warnings,
                )
        else:
            self._fill_constant_histograms(numeric_columns, histogram_plans, profile_data)

        if not self.low_memory and numeric_columns:
            self._collect_exact_quantiles(numeric_columns, profile_data, bins)

    def _collect_exact_quantiles(
        self,
        numeric_columns: list[str],
        profile_data: dict[str, Any],
        bins: int,
    ) -> None:
        """
        Resolves constant/empty quantiles analytically and evaluates fine-bin filtered
        or fallback quickselect plans for remaining numeric columns.
        """

        for column_name in numeric_columns:
            min_value = profile_data.get(f"{column_name}_min")
            max_value = profile_data.get(f"{column_name}_max")
            if min_value is None or max_value is None:
                profile_data[f"{column_name}_p25"] = None
                profile_data[f"{column_name}_p50"] = None
                profile_data[f"{column_name}_p75"] = None
            elif min_value == max_value:
                constant_quantile = float(min_value)
                profile_data[f"{column_name}_p25"] = constant_quantile
                profile_data[f"{column_name}_p50"] = constant_quantile
                profile_data[f"{column_name}_p75"] = constant_quantile

        quantile_bin_specs, fallback_plans = engine.build_quantile_plans(
            self._df, numeric_columns, profile_data, bins=bins
        )
        if quantile_bin_specs:
            quantile_bin_plans = [spec["plan"] for spec in quantile_bin_specs]
            quantile_bin_dfs = pl.collect_all(
                quantile_bin_plans, engine="streaming", optimizations=NO_CSE_OPTIMIZATIONS
            )
            for quantile_spec, filtered_bins in zip(
                quantile_bin_specs, quantile_bin_dfs, strict=True
            ):
                if filtered_bins.height > 0:
                    profile_data.update(
                        engine.extract_quantiles_from_bins(filtered_bins, quantile_spec)
                    )

        for quantile_plan in fallback_plans:
            quantile_df = quantile_plan.collect(engine="streaming")
            if quantile_df.height > 0:
                profile_data.update(quantile_df.row(0, named=True))

        # Release temporary fine-bin counts now that exact quantiles are computed
        for column_name in numeric_columns:
            profile_data.pop(f"{column_name}_fine_counts", None)

    def _build_histograms(  # noqa: PLR0913
        self,
        histogram_columns: list[str],
        histogram_results: list[pl.DataFrame],
        numeric_columns: list[str],
        histogram_plans: dict[str, pl.LazyFrame],
        profile_data: dict[str, Any],
        bins: int,
        profiler_warnings: list[str],
        fine_multiplier: int = 1,
    ) -> None:
        """
        Builds contiguous display histogram intervals from sparse `group_by` bin counts,
        and retains fine-bin counts when `fine_multiplier > 1` for exact quantile lookups.
        """

        for column_name, histogram_df in zip(histogram_columns, histogram_results, strict=True):
            try:
                min_value = profile_data.get(f"{column_name}_min")
                max_value = profile_data.get(f"{column_name}_max")

                if min_value is None or max_value is None:
                    continue
                step = (max_value - min_value) / bins

                coarse_bin_counts: dict[int, int] = {}
                fine_bin_counts: dict[int, int] = {}

                if histogram_df.height > 0:
                    for raw_bin_index, raw_count in zip(
                        histogram_df["bin_index"], histogram_df["len"], strict=True
                    ):
                        bin_index = int(raw_bin_index)
                        count = int(raw_count)
                        fine_bin_counts[bin_index] = count
                        coarse_bin_index = bin_index // fine_multiplier
                        coarse_bin_counts[coarse_bin_index] = (
                            coarse_bin_counts.get(coarse_bin_index, 0) + count
                        )

                if not self.low_memory and fine_multiplier > 1:
                    profile_data[f"{column_name}_fine_counts"] = fine_bin_counts

                histogram = []
                for bin_index in range(bins):
                    left_edge = min_value + (step * bin_index)
                    right_edge = (
                        min_value + (step * (bin_index + 1)) if bin_index < bins - 1 else max_value
                    )
                    count = coarse_bin_counts.get(bin_index, 0)

                    left_bracket = "(" if bin_index > 0 else "["
                    bin_label = f"{left_bracket}{left_edge:g}, {right_edge:g}]"

                    histogram.append({"breakpoint": right_edge, "bin": bin_label, "count": count})

                profile_data[f"{column_name}_histogram"] = histogram

            except Exception as e:
                profiler_warnings.append(
                    f"Histogram generation failed for column '{column_name}': {e}"
                )

        self._fill_constant_histograms(numeric_columns, histogram_plans, profile_data)

    def _fill_constant_histograms(
        self,
        numeric_columns: list[str],
        histogram_plans: dict[str, pl.LazyFrame],
        profile_data: dict[str, Any],
    ) -> None:
        """Populates a single-bin histogram for constant numeric columns (`min == max`)."""

        for column_name in numeric_columns:
            if column_name not in histogram_plans:
                min_value = profile_data.get(f"{column_name}_min")
                max_value = profile_data.get(f"{column_name}_max")

                if min_value is not None and min_value == max_value:
                    row_count = profile_data.get("table_row_count", 0)
                    null_count = profile_data.get(f"{column_name}_null_count", 0)
                    count = row_count - null_count

                    profile_data[f"{column_name}_histogram"] = [
                        {
                            "breakpoint": max_value,
                            "bin": f"[{min_value:g}, {max_value:g}]",
                            "count": count,
                        }
                    ]

    def _compute_correlations(  # noqa: PLR0913
        self,
        correlation_df: pl.DataFrame,
        valid_columns: list[str],
        correlations_sampling_method: str | None,
        profile_data: dict[str, Any],
        profiler_warnings: list[str],
    ) -> None:
        """
        Computes pairwise-complete Pearson and Spearman correlations on the upper triangle.
        """

        if correlation_df.height <= 1 or len(valid_columns) <= 1:
            return

        profile_data["correlations"] = {
            "pearson": [],
            "spearman": [],
            "sampling_method": correlations_sampling_method,
        }

        pearson_expressions: list[pl.Expr] = []
        spearman_expressions: list[pl.Expr] = []
        pair_keys: list[tuple[str, str]] = []

        for i, column_a in enumerate(valid_columns):
            for column_b in valid_columns[i + 1 :]:
                pair_key = f"{column_a}::{column_b}"
                pair_keys.append((column_a, column_b))
                valid_mask = pl.col(column_a).is_not_null() & pl.col(column_b).is_not_null()

                a_valid = pl.col(column_a).filter(valid_mask)
                b_valid = pl.col(column_b).filter(valid_mask)

                pearson_expressions.append(pl.corr(a_valid, b_valid).alias(pair_key))
                spearman_expressions.append(pl.corr(a_valid.rank(), b_valid.rank()).alias(pair_key))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            for method, expressions in [
                ("pearson", pearson_expressions),
                ("spearman", spearman_expressions),
            ]:
                try:
                    correlation_row = (
                        correlation_df.select(expressions).fill_nan(None).row(0, named=True)
                    )
                    pairs = [
                        {"column_a": column_a, "column_b": column_b, "score": float(score)}
                        for (column_a, column_b), score in zip(
                            pair_keys, correlation_row.values(), strict=True
                        )
                        if score is not None
                    ]
                    pairs.sort(key=lambda x: abs(float(cast(float, x["score"]))), reverse=True)
                    profile_data["correlations"][method] = pairs
                except Exception as e:
                    profiler_warnings.append(f"Correlations ({method}) calculation failed: {e}")

    def _build_profile(
        self,
        profile_data: dict[str, Any],
        profiling_start_epoch: float,
        profiling_start_perf: float,
        profiler_warnings: list[str],
    ) -> NetraProfile:
        """
        Assembles the flat state dictionary into the `NetraProfile` schema.
        """

        row_count = profile_data.get("table_row_count", 0)

        profiling_end_epoch = time.time()
        engine_time_seconds = round(time.perf_counter() - profiling_start_perf, 4)

        profile = {
            "dataset": {
                "name": self.dataset_name,
                "format": self.dataset_format,
                "row_count": row_count,
            },
            "columns": {},
            "correlations": profile_data.get(
                "correlations", {"pearson": [], "spearman": [], "sampling_method": None}
            ),
            "alerts": [],
            "_meta": {
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "execution_start_epoch": profiling_start_epoch,
                "execution_end_epoch": profiling_end_epoch,
                "engine_time_seconds": engine_time_seconds,
                "profiler_version": __version__,
                "is_low_memory_run": self.low_memory,
                "config_source": self.config_source,
                "profiler_warnings": profiler_warnings,
                "pipeline_context": None,
            },
        }

        column_names = sorted(
            {key.removesuffix("_null_count") for key in profile_data if key.endswith("_null_count")}
        )

        for column_name in column_names:
            n_unique_value = profile_data.get(f"{column_name}_n_unique", 0)

            column_profile = {
                "data_type": profile_data.get(f"{column_name}_data_type"),
                "null_count": profile_data.get(f"{column_name}_null_count", 0),
                "n_unique": min(n_unique_value, row_count),
                "histogram": profile_data.get(f"{column_name}_histogram", []),
                "top_k": profile_data.get(f"{column_name}_top_k", []),
            }

            data_type_string = column_profile.get("data_type", "")

            if is_numeric_type(data_type_string):
                numeric_metrics = [
                    "min",
                    "max",
                    "mean",
                    "zero_count",
                    "std",
                    "skew",
                    "kurtosis",
                    "p25",
                    "p50",
                    "p75",
                ]
                for metric in numeric_metrics:
                    column_profile[metric] = profile_data.get(f"{column_name}_{metric}")

            elif "Date" in data_type_string or "Time" in data_type_string:
                temporal_metrics = ["min", "max"]
                for metric in temporal_metrics:
                    column_profile[metric] = profile_data.get(f"{column_name}_{metric}")

            elif is_string_type(data_type_string):
                string_metrics = [
                    "min_length",
                    "max_length",
                    "mean_length",
                    "min",
                    "max",
                    "blank_count",
                ]
                for metric in string_metrics:
                    column_profile[metric] = profile_data.get(f"{column_name}_{metric}")

            profile["columns"][column_name] = column_profile

        return cast(NetraProfile, profile)

    def _run_diagnostics(self, profile: NetraProfile) -> None:
        """
        Evaluates data quality diagnostic rules against `profile`.
        """

        diagnostic_engine = DiagnosticEngine(profile, config=self.diagnostic_config)
        alerts = diagnostic_engine.run()

        profile["alerts"] = [
            {
                "column_name": alert.column_name,
                "type": alert.type,
                "level": alert.level.value,
                "message": alert.message,
                "value": alert.value,
            }
            for alert in alerts
        ]

        dataset_columns = list(profile.get("columns", {}).keys())
        config_warnings = self.diagnostic_config.get_unused_override_warnings(dataset_columns)
        if config_warnings:
            profile["_meta"]["config_warnings"] = config_warnings
