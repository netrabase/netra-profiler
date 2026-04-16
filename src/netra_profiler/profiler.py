import datetime
import time
import warnings
from typing import Any, cast

import polars as pl

from netra_profiler import __version__, engine
from netra_profiler.diagnostics import DiagnosticEngine
from netra_profiler.types import NetraProfile, is_numeric_type, is_string_type

CORRELATION_SAMPLE_SIZE = 100_000


class Profiler:
    """
    The core orchestrator class for Netra Profiler.

    This class manages the complete lifecycle of a profiling session, decoupling the
    execution flow from the mathematical plan generation (which is delegated to `engine.py`).
    It is responsible for transforming raw Polars data into a strictly typed `NetraProfile`
    Enterprise Data Contract.

    Architecture Notes:
    1. Optimization First: Forces all data into Polars' `LazyFrame` to utilize
       the Rust-based query optimizer and streaming engine.
    2. Schema Flattening: Pre-processes complex nested types (Structs/Lists) into
       flat scalar columns to prevent engine panics and simplify downstream math.
    3. Multi-Pass Execution: Evaluates the execution graph in distinct, sequential
       stages. This manages memory state and allows downstream passes (e.g., Histograms)
       to rely on the outputs of upstream computations (e.g., global Min/Max).
    """

    def __init__(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        dataset_name: str = "Unknown",
        dataset_format: str = "Unknown",
        ignore_columns: list[str] | None = None,
        low_memory: bool = False,
    ):
        """
        Initializes the profiling session.

        During initialization, the Profiler prepares the dataset for optimized execution:
        1. Eager-to-Lazy Conversion: If a raw DataFrame is passed, it is immediately
           converted to a LazyFrame. This forces all downstream operations to pass through
           the Polars query optimizer.
        2. Cardinality Bypassing: Processes the `ignore_columns` list, preemptively dropping
           primary keys or other listed columns from the graph to prevent memory-crushing hash
           set allocations during Pass 1.
        3. Type Flattening: Unpacks nested structs and lists so the core engine can process
           deeply nested JSON/Parquet files as standard scalar columns.

        Args:
            df: The input Polars DataFrame (eager) or LazyFrame (streaming).
            dataset_name: The human-readable name of the dataset (e.g., filename).
            dataset_format: The file format or source type (e.g., 'CSV', 'Parquet').
            ignore_columns: A list of column names to explicitly exclude from profiling.
                Highly recommended for unique Primary Keys to preserve RAM.
            memory_safe: If True, invokes bounded-memory execution. Replaces exact unique
                counts with HyperLogLog approximations and strips global sorts (quantiles)
                from the execution graph to prevent OOM panics on massive datasets.
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

        # Preprocess Complex Types (Structs/Lists)
        # This "flattens" the data view for the engine, enabling
        # support for nested JSON/Parquet without engine changes.
        self._df = engine.preprocess_complex_types(self._df)

    def run(self, bins: int = 20, top_k: int = 10) -> NetraProfile:
        """
        Executes the 5-pass profiling architecture and generates the final data contract.

        This method orchestrates the sequential execution of the profiling graph. Because certain
        operations (like histogram bin sizing) mathematically require the global bounds (min/max)
        of the data, the execution is split into sequential passes, passing the `profile_data`
        state dictionary between them.

        The Execution Graph:
        - PASS 1 (Scalars): Streaming pass for row count, data type, min, max, mean, skew,
            kurtosis, percentiles, and nulls, zeros and unique counts.
        - PASS 2 (Histograms): Streaming Map-Reduce pass for continuous distribution grouping.
        - PASS 3 (Top-K): Streaming pass for frequent categorical item extraction.
        - PASS 4 (Correlations): Eager, sampled pass to compute Pearson & Spearman matrices.
        - PASS 5 (Diagnostics): Rule-based alerting engine run against the compiled profile.

        Args:
            bins: The target number of intervals to generate for numeric histograms. Defaults to 20.
            top_k: The maximum number of frequent items to compute for categorical/string columns.
                Defaults to 10.

        Returns:
            A `NetraProfile` typed dictionary containing the rigidly structured, JSON-serializable
            Data Contract, including all column statistics, correlation matrices, data quality
            alerts, and engine telemetry metadata.
        """

        profiling_start_time = time.time()
        profiler_warnings: list[str] = []

        # PASS 1: Scalar Statistics (Foundation)
        profile_data = self._run_scalar_pass()

        # PASS 2: Histograms
        self._run_histogram_pass(profile_data, bins, profiler_warnings)

        # PASS 3: Top-K Values
        self._run_top_k_pass(profile_data, top_k)

        # PASS 4: Correlations
        self._run_correlation_pass(profile_data, profiler_warnings)

        # Build the structured Profile object based on our Profile Object Schema
        profile = self._build_profile_object(profile_data, profiling_start_time, profiler_warnings)

        # PASS 5: Diagnostics
        self._run_diagnostics_pass(profile)

        return profile

    def _run_scalar_pass(self) -> dict[str, Any]:
        """
        Executes PASS 1: Scalar Statistics.

        Triggers the evaluation of the scalar blueprint generated by the engine.
        Executed entirely within Polars' streaming engine (unless `memory_safe=False` forces
        eager fallback for quantiles).

        Returns:
            A flat dictionary containing all computed scalar statistics with strictly
            aliased keys (e.g., `{'age_mean': 34.2, 'status_null_count': 0}`).
        """

        scalar_plan = engine.build_scalar_plan(self._df, low_memory=self.low_memory)
        scalar_df = scalar_plan.collect(engine="streaming")
        return scalar_df.rows(named=True)[0]

    def _run_histogram_pass(
        self, profile_data: dict[str, Any], bins: int, profiler_warnings: list[str]
    ) -> None:
        """
        Executes PASS 2: Histograms.

        Evaluates the Map-Reduce histogram execution plans in parallel across all numeric
        columns. Once Polars returns the discrete bin counts, this method performs
        post-processing to construct contiguous, formatted distribution intervals.

        Args:
            profile_data: The state dictionary containing Pass 1 boundaries (Min/Max).
                The constructed histograms are appended directly into this dictionary.
            bins: The target number of intervals.
            profiler_warnings: A mutable list to capture runtime execution errors.

        Notes:
            - Sparsity Handling: The Map-Reduce engine drops bins with a count of zero.
              The post-processing loop here reconstructs the full geometric space,
              re-injecting `count: 0` for empty intervals to ensure chart integrity.
            - Constant Fallback: If a column has zero variance (`min == max`), a single
              interval encompassing all non-null rows is generated.
        """

        # 1. Select numeric columns
        schema = self._df.collect_schema()
        numeric_columns = [column for column, dtype in schema.items() if dtype.is_numeric()]

        # 2. Build plans
        histogram_plans = engine.build_histogram_plans(
            self._df, numeric_columns, profile_data, bins
        )

        if histogram_plans:
            columns = list(histogram_plans.keys())
            plans = list(histogram_plans.values())

            # 3. Parallel streaming execution
            results = pl.collect_all(plans, engine="streaming")

            # 4. Post-processing (Handling sparsity and formatting)
            for column_name, hist_df in zip(columns, results, strict=True):
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
                    for bin in range(bins):
                        left_edge = min_value + (step * bin)
                        right_edge = min_value + (step * (bin + 1)) if bin < bins - 1 else max_value
                        count = counts_dict.get(bin, 0)  # 0 if the bin was dropped by group_by

                        # Format left/right bracket notation
                        left_bracket = "(" if bin > 0 else "["
                        bin_label = f"{left_bracket}{left_edge:g}, {right_edge:g}]"

                        histogram.append(
                            {"breakpoint": right_edge, "bin": bin_label, "count": count}
                        )

                    profile_data[f"{column_name}_histogram"] = histogram

                except Exception as e:
                    profiler_warnings.append(
                        f"Histogram generation failed for column '{column_name}': {e}"
                    )

        # 5. Fallback for constant columns (min == max)
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
                            "break_point": max_value,
                            "bin": f"[{min_value:g}, {max_value:g}]",
                            "count": count,
                        }
                    ]

    def _run_top_k_pass(self, profile_data: dict[str, Any], top_k: int) -> None:
        """
        Executes PASS 3: Top-K Frequent Items.

        Evaluates the categorical frequency plans in parallel using the streaming engine.
        The resulting DataFrames are serialized into JSON-friendly lists of dictionaries
        and injected into the state payload.

        Args:
            profile_data: The state dictionary. Mutated in-place to add `[column]_top_k`.
            top_k: The maximum number of frequent items requested.
        """

        top_k_plans = engine.build_top_k_plan(self._df, k=top_k)

        if top_k_plans:
            # We execute all columns in parallel using the streaming engine
            top_k_dfs = pl.collect_all(top_k_plans, engine="streaming")

            for top_k_df in top_k_dfs:
                if top_k_df.height > 0:
                    column_name = top_k_df["column_name"][0]
                    top_k_values = top_k_df.select("value", "count").to_dicts()
                    profile_data[f"{column_name}_top_k"] = top_k_values

    def _run_correlation_pass(
        self, profile_data: dict[str, Any], profiler_warnings: list[str]
    ) -> None:
        """
        Executes PASS 4: Correlations.

        Computes the Pearson and Spearman correlation matrices for all numeric columns.
        Due to the global state requirements of matrix computation and row indexing,
        this pass fundamentally triggers eager memory materialization.

        To prevent OOM crashes on massive datasets, this method implements an Adaptive
        Systematic Sampling architecture. If the dataset exceeds `CORRELATION_SAMPLE_SIZE`,
        it uses `gather_every()` to slice a representative subset before executing the math.

        Args:
            profile_data: The state dictionary. Mutated in-place to add the `correlations` matrix.
            profiler_warnings: A mutable list to capture runtime execution errors.

        Notes:
            - Divide-by-Zero Protection: Constant columns mathematically yield `NaN` in
              correlation matrices and throw native `RuntimeWarnings`. These warnings are
              explicitly suppressed here to prevent terminal spam on dirty data.
        """

        correlation_plan = engine.build_correlation_plan(self._df)

        # Check if we actually have columns to correlate
        # We use collect_schema() to check cheaply
        if len(correlation_plan.collect_schema()) > 1:
            # Adaptive sampling logic

            # We reuse the row count from Pass 1 (cost = 0)
            row_count = profile_data.get("table_row_count", 0)

            # Fetch Data (Sampled or Full)
            if row_count > CORRELATION_SAMPLE_SIZE:
                # Case 1: Big Data -> Sample
                # We use Systematic Sampling
                step_size = max(1, row_count // CORRELATION_SAMPLE_SIZE)

                correlation_df = (
                    # We cast to Float64 to handle potential integer overflow
                    correlation_plan.select(pl.all().cast(pl.Float64))
                    # gather_every() only uses every step_size row in the dataset
                    # which is more robust than using head or tail and closest to
                    # random sampling in streaming mode. High disk I/O, low RAM usage
                    .gather_every(step_size)
                    .collect()
                )
                correlations_sampling_method = f"systematic_sample (~{CORRELATION_SAMPLE_SIZE})"
            else:
                # Case 2: Small Data -> Exact
                correlation_df = correlation_plan.select(pl.all().cast(pl.Float64)).collect()
                correlations_sampling_method = "exact"

            # We drop Nulls to prevent NaN propagation in the matrix
            correlation_df = correlation_df.drop_nulls()

            # Compute Matrices
            if correlation_df.height > 0 and correlation_df.width > 1:
                profile_data["correlations"] = {
                    "pearson": [],
                    "spearman": [],
                    "sampling_method": correlations_sampling_method,
                }

                # We expect RuntimeWarnings (Divide by Zero) when correlating constant columns.
                # This is normal behavior for dirty data, so we suppress the log noise locally.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)

                    # Compute PEARSON (Standard .corr())
                    try:
                        # Format: Add a 'column' column so we know which row is which
                        # Output structure: [{'column': 'age', 'age': 1.0, 'income': 0.8}, ...]
                        pearson_matrix = correlation_df.corr().with_columns(
                            pl.Series(name="column", values=correlation_df.columns)
                        )
                        # Reorder so that "column" appears first and
                        # Clean NaNs (for JSON safety)
                        pearson_matrix = pearson_matrix.select(
                            pl.col("column"), pl.exclude("column").fill_nan(None)
                        )

                        profile_data["correlations"]["pearson"] = self._extract_correlation_pairs(
                            pearson_matrix
                        )
                    except Exception as e:
                        profiler_warnings.append(f"Correlations (pearson) calculation failed: {e}")

                    # Compute SPEARMAN (Ranks -> .corr())
                    try:
                        # Spearman is just Pearson on the ranks
                        spearman_matrix = (
                            correlation_df.select(pl.all().rank())
                            .corr()
                            .with_columns(pl.Series(name="column", values=correlation_df.columns))
                        )

                        spearman_matrix = spearman_matrix.select(
                            pl.col("column"), pl.exclude("column").fill_nan(None)
                        )

                        profile_data["correlations"]["spearman"] = self._extract_correlation_pairs(
                            spearman_matrix
                        )
                    except Exception as e:
                        profiler_warnings.append(f"Correlations (spearman) calculation failed: {e}")

                # Metadata Injection
                profile_data["correlations"]["sampling_method"] = correlations_sampling_method

    def _extract_correlation_pairs(self, matrix_df: pl.DataFrame) -> list[dict[str, Any]]:
        """
        Transforms a dense symmetric correlation matrix into a sparse Edge-List format.

        Standard matrices compute $N^2$ relationships, resulting in heavy redundancy.
        This method parses the DataFrame and drops:
        1. Self-correlations (the diagonal, where A correlates with A at 1.0).
        2. Mirrored duplicates (if A-B is recorded, B-A is skipped via tuple signatures).

        Args:
            matrix_df: The eager Polars DataFrame containing the computed `.corr()` matrix.

        Returns:
            A list of dictionary objects (e.g., `{'column_a': 'X', 'column_b': 'Y', 'score': 0.8}`),
            sorted by absolute correlation strength descending.
        """

        matrix_dicts = matrix_df.to_dicts()
        pairs = []
        seen = set()

        for row in matrix_dicts:
            column_a = row["column"]
            for column_b, score in row.items():
                if column_b == "column" or score is None:
                    continue
                if column_a == column_b:  # Drop diagonal (self-correlation = 1.0)
                    continue

                # Tuple sorting ensures A-B and B-A generate the same signature
                pair_signature = tuple(sorted([column_a, column_b]))

                if pair_signature not in seen:
                    seen.add(pair_signature)
                    pairs.append({"column_a": column_a, "column_b": column_b, "score": score})

        # Sort by correlation strength (absolute value) descending
        pairs.sort(key=lambda x: abs(x["score"]), reverse=True)
        return pairs

    def _build_profile_object(
        self,
        profile_data: dict[str, Any],
        profiling_start_time: float,
        profiler_warnings: list[str],
    ) -> NetraProfile:
        """
        Compiles the flat state dictionary into the strict Enterprise Data Contract.

        Acts as the final schema mapping layer. It extracts telemetry metadata, normalizes
        missing values, and guarantees structural type stability (e.g., ensuring arrays are
        empty `[]` rather than omitted from the JSON if an operation fails).

        Args:
            profile_data: The fully populated state dictionary from Passes 1-4.
            profiling_start_time: Epoch timestamp to calculate final engine throughput.
            profiler_warnings: The accumulated list of non-fatal execution errors.

        Returns:
            A strictly typed `NetraProfile` dictionary ready for JSON serialization.

        Notes:
            - HyperLogLog Clamping: If `memory_safe` mode is active, the HLL algorithm's
              ±2% standard error margin can cause exact primary keys to report a unique count
              higher than the total row count. This function explicitly clamps `n_unique` to
              `row_count` to prevent mathematical impossibilities in downstream alerts.
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
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "execution_start_epoch": profiling_start_time,
                "execution_end_epoch": profiling_end_time,
                "engine_time_seconds": round(profiling_end_time - profiling_start_time, 4),
                "profiler_version": __version__,
                "is_low_memory_run": self.low_memory,
                "warnings": profiler_warnings,
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

            elif is_string_type(data_type_string):
                string_metrics = ["min_length", "max_length", "mean_length", "min", "max"]
                for metric in string_metrics:
                    column_profile[metric] = profile_data.get(f"{column_name}_{metric}")

            profile["columns"][column_name] = column_profile

        return cast(NetraProfile, profile)

    def _run_diagnostics_pass(self, profile: NetraProfile) -> None:
        """
        Executes PASS 5: Diagnostics.

        Initializes and runs the standalone `DiagnosticEngine` against the finalized Data Contract.
        The resulting alert objects are serialized into dictionaries and injected into the
        `profile["alerts"]` list.

        Args:
            profile: The finalized `NetraProfile` contract. Mutated in-place.
        """

        diagnostic_engine = DiagnosticEngine(profile)
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
