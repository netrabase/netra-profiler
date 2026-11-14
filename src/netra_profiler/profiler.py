import datetime
import time
import warnings
from typing import Any, cast

import polars as pl

from netra_profiler import __version__, engine
from netra_profiler.config import DiagnosticConfig
from netra_profiler.diagnostics import DiagnosticEngine
from netra_profiler.types import NetraProfile, is_numeric_type, is_string_type

CORRELATION_SAMPLE_SIZE = 100_000


class Profiler:
    """
    Orchestrates dataset profiling from a Polars `DataFrame` or `LazyFrame`.

    Execution Architecture:
        1.  Schema Normalization (`__init__`): Converts eager inputs to `LazyFrame`, drops
            user-ignored columns, and flattens nested `Struct`, `List`, and `Array` columns.
        2.  Stage 1 — Summary Collection (`_collect_scalars_and_top_k`): Evaluates the global
            scalar aggregation plan and per-column string Top-K plans concurrently in a single
            `pl.collect_all()` (`SinkMultiple`) pass.
        3.  Stage 2 — Distribution & Correlation Scan (`_collect_histograms_and_correlations`):
            Uses the `[min, max]` bounds, `n_unique`, and `table_row_count` from Stage 1 to
            build the histogram plans and a sampled correlation plan, evaluating them together
            in a second `pl.collect_all()` pass.
        4.  Profile Assembly & Diagnostics (`_build_profile`, `_run_diagnostics`): Compiles
            collected metrics into a typed `NetraProfile` dictionary and runs `DiagnosticEngine`
            in memory to generate data quality alerts.
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
            ignore_columns: Column names to exclude from profiling prior to plan construction.
            low_memory: If True, uses HyperLogLog cardinality (`approx_n_unique`) and omits
                exact quantiles, skewness, and kurtosis to bound memory consumption.
            config: Optional dictionary of threshold overrides parsed from `netra_config.yaml`.
            config_source: Label indicating where the configuration was loaded from.
        """

        # If a DataFrame is passed, it is converted to LazyFrame to
        # ensure all downstream operations use the query optimizer.
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

        # Preprocess Complex Types (Structs/Lists)
        # This "flattens" the data view for the engine, enabling
        # support for nested JSON/Parquet without engine changes.
        self._df = engine.preprocess_complex_types(self._df)

    def run(self, bins: int = 20, top_k: int = 10) -> NetraProfile:
        """
        Executes the profiling pipeline and returns the finalized `NetraProfile` report.

        Args:
            bins: Number of equi-width intervals for numeric histograms (default: 20).
            top_k: Maximum number of frequent values to retain for string columns (default: 10).

        Returns:
            A typed, JSON-serializable `NetraProfile` dictionary containing dataset metadata,
            column statistics, correlation edges, diagnostic alerts, and execution telemetry.
        """

        profiling_start_time = time.time()
        profiler_warnings: list[str] = []

        # Scalar Statistics + Top-K Values (Shared Scan 1)
        profile_data = self._collect_scalars_and_top_k(top_k)

        # Histograms + Correlations (Shared Scan 2)
        self._collect_histograms_and_correlations(profile_data, bins, profiler_warnings)

        # Build the structured Profile object based on our Profile Object Schema
        profile = self._build_profile(profile_data, profiling_start_time, profiler_warnings)

        # Quality Diagnostics
        self._run_diagnostics(profile)

        return profile

    def _collect_scalars_and_top_k(self, top_k: int) -> dict[str, Any]:
        """
        Collects scalar statistics and string Top-K frequency tables in a single `collect_all` pass.

        Because neither plan depends on prior summary statistics, submitting them together to
        `pl.collect_all()` allows the Polars streaming engine to schedule all plans concurrently
        within a single execution graph rather than on sequential Python-level `.collect()`.
        """

        scalar_plan = engine.build_scalar_plan(self._df, low_memory=self.low_memory)
        top_k_plans = engine.build_top_k_plan(self._df, k=top_k)

        wave_1_results = pl.collect_all([scalar_plan, *top_k_plans], engine="streaming")

        scalar_df = wave_1_results[0]
        profile_data: dict[str, Any] = scalar_df.rows(named=True)[0]

        for top_k_df in wave_1_results[1:]:
            if top_k_df.height > 0:
                column_name = top_k_df["column_name"][0]
                top_k_values = top_k_df.select("value", "count").to_dicts()
                profile_data[f"{column_name}_top_k"] = top_k_values

        return profile_data

    def _collect_histograms_and_correlations(
        self, profile_data: dict[str, Any], bins: int, profiler_warnings: list[str]
    ) -> None:
        """
        Collects numeric histograms and the correlation sample in a second `collect_all` pass.

        Both operations depend on metrics computed in `_collect_scalars_and_top_k`:
            - Histograms require each column's `[min, max]` bounds to compute bin widths.
            - Correlation sampling requires `table_row_count` to compute the `gather_every`
                stride and `{col}_n_unique` to exclude constant or empty numeric columns upfront.
        """

        schema = self._df.collect_schema()
        numeric_columns = [column for column, dtype in schema.items() if dtype.is_numeric()]

        histogram_plans = engine.build_histogram_plans(
            self._df, numeric_columns, profile_data, bins
        )
        hist_columns = list(histogram_plans.keys())
        wave_2_plans: list[pl.LazyFrame] = list(histogram_plans.values())

        # Prepare Correlation Sampling Plan (only for numeric columns with variance > 0)
        row_count = profile_data.get("table_row_count", 0)
        valid_corr_columns = [
            col for col in numeric_columns if profile_data.get(f"{col}_n_unique", 0) > 1
        ]
        has_correlation_plan = row_count > 1 and len(valid_corr_columns) > 1
        correlations_sampling_method: str | None = None

        if has_correlation_plan:
            corr_plan = self._df.select(valid_corr_columns)
            if row_count > CORRELATION_SAMPLE_SIZE:
                step_size = max(1, row_count // CORRELATION_SAMPLE_SIZE)
                corr_plan = corr_plan.gather_every(step_size)
                correlations_sampling_method = f"systematic_sample (~{CORRELATION_SAMPLE_SIZE})"
            else:
                correlations_sampling_method = "exact"

            # Cast and coerce NaNs AFTER gather_every so we only transform the sampled rows
            corr_plan = corr_plan.select(pl.all().cast(pl.Float64).fill_nan(None))
            wave_2_plans.append(corr_plan)

        if not wave_2_plans:
            self._fill_constant_histograms(numeric_columns, histogram_plans, profile_data)
            return

        wave_2_results = pl.collect_all(wave_2_plans, engine="streaming")

        hist_results = wave_2_results[: len(hist_columns)]
        self._build_histograms(
            hist_columns,
            hist_results,
            numeric_columns,
            histogram_plans,
            profile_data,
            bins,
            profiler_warnings,
        )

        if has_correlation_plan:
            correlation_df = wave_2_results[-1]
            self._compute_correlations(
                correlation_df,
                valid_corr_columns,
                correlations_sampling_method,
                profile_data,
                profiler_warnings,
            )

    def _build_histograms(  # noqa: PLR0913
        self,
        hist_columns: list[str],
        hist_results: list[pl.DataFrame],
        numeric_columns: list[str],
        histogram_plans: dict[str, pl.LazyFrame],
        profile_data: dict[str, Any],
        bins: int,
        profiler_warnings: list[str],
    ) -> None:
        """
        Builds contiguous histogram intervals from sparse `group_by` bin counts.

        Because `group_by("bin_index")` omits empty bins, this method iterates over
        `range(bins)` and fills missing indices with `count: 0` to preserve interval
        continuity.
        """

        for column_name, hist_df in zip(hist_columns, hist_results, strict=True):
            try:
                min_value = profile_data.get(f"{column_name}_min")
                max_value = profile_data.get(f"{column_name}_max")

                if min_value is None or max_value is None:
                    continue
                step = (max_value - min_value) / bins

                if hist_df.height > 0:
                    counts_dict = dict(zip(hist_df["bin_index"], hist_df["len"], strict=True))
                else:
                    counts_dict = {}

                histogram = []
                for bin_idx in range(bins):
                    left_edge = min_value + (step * bin_idx)
                    right_edge = (
                        min_value + (step * (bin_idx + 1)) if bin_idx < bins - 1 else max_value
                    )
                    count = counts_dict.get(bin_idx, 0)

                    left_bracket = "(" if bin_idx > 0 else "["
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

        for column in numeric_columns:
            if column not in histogram_plans:
                min_value = profile_data.get(f"{column}_min")
                max_value = profile_data.get(f"{column}_max")

                if min_value is not None and min_value == max_value:
                    row_count = profile_data.get("table_row_count", 0)
                    null_count = profile_data.get(f"{column}_null_count", 0)
                    count = row_count - null_count

                    profile_data[f"{column}_histogram"] = [
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

        Design Notes:
            -   Pairwise Null Filtering: Each `(column_a, column_b)` pair filters nulls using only
                    those two columns rather than calling `drop_nulls()` on the whole table. This
                    prevents a single mostly-null column from wiping out valid rows for other pairs.
            -   Unique Pairs Only: Iterating over `valid_columns[i + 1 :]` evaluates only the upper
                    triangle of the matrix, skipping self-correlations (`column_a == column_b`)
                    and duplicate mirror pairs (`(B, A)`, `(A, B)`).
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

            for method, exprs in [
                ("pearson", pearson_expressions),
                ("spearman", spearman_expressions),
            ]:
                try:
                    corr_row = correlation_df.select(exprs).fill_nan(None).row(0, named=True)
                    pairs = [
                        {"column_a": col_a, "column_b": col_b, "score": float(score)}
                        for (col_a, col_b), score in zip(pair_keys, corr_row.values(), strict=True)
                        if score is not None
                    ]
                    pairs.sort(key=lambda x: abs(float(cast(float, x["score"]))), reverse=True)
                    profile_data["correlations"][method] = pairs
                except Exception as e:
                    profiler_warnings.append(f"Correlations ({method}) calculation failed: {e}")

    def _build_profile(
        self,
        profile_data: dict[str, Any],
        profiling_start_time: float,
        profiler_warnings: list[str],
    ) -> NetraProfile:
        """
        Assembles the flat state dictionary into the `NetraProfile` schema.

        Design Notes:
            -   When `low_memory=True`, HyperLogLog's standard error (~2%) can estimate `n_unique`
                slightly above `row_count` on unique key columns, which is handled by using
                `min(n_unique_value, row_count)` which clamps cardinality to the physical row count.
        """

        row_count = profile_data.get("table_row_count", 0)
        profiling_end_time = time.time()

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
            "alerts": [],  # Will be populated by Pass 5
            "_meta": {
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "execution_start_epoch": profiling_start_time,
                "execution_end_epoch": profiling_end_time,
                "engine_time_seconds": round(profiling_end_time - profiling_start_time, 4),
                "profiler_version": __version__,
                "is_low_memory_run": self.low_memory,
                "config_source": self.config_source,
                "profiler_warnings": profiler_warnings,
                "pipeline_context": None,
            },
        }

        # Extract and alphabetize column names
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

        # Serialize alerts to dicts for JSON output
        profile["alerts"] = [
            {
                "column_name": alert.column_name,
                "type": alert.type,
                "level": alert.level.value,  # Convert Enum to string
                "message": alert.message,
                "value": alert.value,
            }
            for alert in alerts
        ]

        # Validate config typos and inject warnings into the _meta block
        dataset_columns = list(profile.get("columns", {}).keys())
        config_warnings = self.diagnostic_config.get_unused_override_warnings(dataset_columns)
        if config_warnings:
            profile["_meta"]["config_warnings"] = config_warnings
