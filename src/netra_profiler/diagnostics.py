"""
Rule-Based Diagnostics and Alert Generation.

This module evaluates a finalized `NetraProfile` against the thresholds defined
in the `DiagnosticConfig` to generate data quality alerts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from netra_profiler.config import DiagnosticConfig
from netra_profiler.types import ColumnMetrics, NetraProfile, is_numeric_type


class AlertLevel(str, Enum):
    """
    Defines the severity hierarchy of diagnostic alerts.
    Strings are used as mixins for seamless JSON serialization in the final output.
    """

    CRITICAL = "CRITICAL"  # Data is broken/unusable
    WARNING = "WARNING"  # Data is suspicious/requires attention
    INFO = "INFO"  # Optimization tip


@dataclass
class Alert:
    """
    Standardized payload for a single diagnostic finding.
    Designed to map 1:1 with the expected JSON schema of the Data Contract.
    """

    column_name: str
    type: str  # e.g. "HIGH_NULLS"
    level: AlertLevel
    message: str
    value: float | None = None


class DiagnosticEngine:
    """
    PASS 5: Rule-Based Diagnostics and Alert Generation.

    The Diagnostic Engine acts as a static analysis layer over the computed statistics.
    Unlike Passes 1-4, this engine is purely CPU-bound logic; it never touches the Polars
    LazyFrame, queries raw data, or incurs Disk/RAM I/O. It solely evaluates the finalized
    `NetraProfile` dictionary against the `DiagnosticConfig` data quality checks.
    """

    def __init__(self, profile: NetraProfile, config: DiagnosticConfig | None = None):
        """
        Initializes the Diagnostic Engine with the compiled profile state.

        Args:
            profile: The fully computed `NetraProfile` dictionary resulting from Passes 1-4.
        """

        self.profile = profile
        self.config = config or DiagnosticConfig()
        self.dataset = self.profile.get("dataset", {})
        self.columns = self.profile.get("columns", {})
        self.correlations = self.profile.get("correlations", {})

        meta = self.profile.get("_meta", {})
        self.is_memory_safe_run = meta.get("is_memory_safe_run", False)

        self.row_count = self.dataset.get("row_count", 0)
        self.alerts: list[Alert] = []

    def run(self) -> list[Alert]:
        """
        Executes the full suite of diagnostic checks against the profile.

        Returns:
            A list of `Alert` dataclasses detailing every anomaly found in the data.
            Returns an empty list if the dataset has 0 rows or no checks fail.
        """

        if self.row_count == 0:
            return []

        for column_name, column_profile in self.columns.items():
            data_type = column_profile.get("data_type", "")
            is_numeric = is_numeric_type(data_type)

            # 1. Universal Checks
            null_count = column_profile.get("null_count", 0)
            self._check_nulls(column_name, null_count)

            n_unique = column_profile.get("n_unique")
            if n_unique is not None:
                self._check_constant(column_name, n_unique)

                # Primary Key Checks (Only for Ints, Strings, Categoricals)
                is_id_candidate = any(
                    t in data_type for t in ["Int", "String", "Utf8", "Categorical"]
                )
                if is_id_candidate:
                    self._check_primary_key(column_name, n_unique)

                self._check_duplicates(column_name, n_unique)

            # 2. Type-Specific Checks
            if is_numeric:
                self._run_numeric_checks(column_name, column_profile)
            else:
                self._run_string_checks(column_name, column_profile, n_unique)

        # 4. Correlation Check
        self._check_correlations()

        return self.alerts

    def _run_numeric_checks(self, column_name: str, column_profile: ColumnMetrics) -> None:
        """Evaluates all diagnostic rules specific to numeric columns."""
        skew = column_profile.get("skew")
        if skew is not None:
            self._check_skew(column_name, skew)

        zero_count = column_profile.get("zero_count")
        if zero_count is not None:
            self._check_zeros(column_name, zero_count)

        p25 = column_profile.get("p25")
        p75 = column_profile.get("p75")
        min_value = column_profile.get("min")
        max_value = column_profile.get("max")

        if (
            isinstance(p25, (int, float))
            and isinstance(p75, (int, float))
            and isinstance(min_value, (int, float))
            and isinstance(max_value, (int, float))
        ):
            self._check_outliers(column_name, p25, p75, min_value, max_value)

    def _run_string_checks(
        self, column_name: str, column_profile: ColumnMetrics, n_unique: int | None
    ) -> None:
        """Evaluates all diagnostic rules specific to string and categorical columns."""
        if n_unique is not None:
            self._check_cardinality(column_name, n_unique)

        top_k = column_profile.get("top_k")
        if top_k:
            self._check_possible_numeric(column_name, top_k)

        mean_length = column_profile.get("mean_length")
        max_length = column_profile.get("max_length")
        min_length = column_profile.get("min_length")

        if (
            isinstance(mean_length, (int, float))
            and isinstance(max_length, (int, float))
            and isinstance(min_length, (int, float))
        ):
            self._check_string_length(column_name, mean_length, max_length, min_length)

        blank_count = column_profile.get("blank_count")
        if blank_count is not None:
            self._check_blanks(column_name, blank_count)

    def _check_nulls(self, column_name: str, null_count: int) -> None:
        """
        Evaluates the column for critical data loss via null/missing values.

        Alerts:
            - EMPTY_COLUMN (CRITICAL): Nulls > 95%. Indicates the column is structurally empty.
            - HIGH_NULLS (WARNING): Nulls > 50%. Indicates standard imputation methods
              (mean/median) will likely introduce severe statistical bias.
        """
        critical_threshold = self.config.get_rule("null_critical_threshold", column_name)
        warning_threshold = self.config.get_rule("null_warning_threshold", column_name)

        # We skip the check if both rules are set to False or None
        if critical_threshold in (False, None) and warning_threshold in (False, None):
            return

        null_percentage = null_count / self.row_count

        # 1. Check Critical First
        if critical_threshold not in (False, None) and null_percentage > critical_threshold:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="EMPTY_COLUMN",
                    level=AlertLevel.CRITICAL,
                    message=(
                        f"Column is {null_percentage:.1%} empty. "
                        "It likely contains no useful information."
                    ),
                    value=null_percentage,
                )
            )
        # 2. Fallback to Warning if not Critical (or if Critical is disabled)
        elif warning_threshold not in (False, None) and null_percentage > warning_threshold:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="HIGH_NULLS",
                    level=AlertLevel.WARNING,
                    message=f"Column is {null_percentage:.1%} empty. Imputation may be difficult.",
                    value=null_percentage,
                )
            )

    def _check_blanks(self, column_name: str, blank_count: int) -> None:
        """
        Evaluates potential blank/whitespace strings.

        Alerts:
            - BLANK_STRINGS (WARNING): Triggers when the percentage of purely blank
              strings exceeds the configured threshold.
        """

        blank_warning_threshold = self.config.get_rule("blank_warning_threshold", column_name)
        if blank_warning_threshold in (False, None):
            return

        if blank_count > 0 and self.row_count > 0:
            blank_percentage = blank_count / self.row_count

            if blank_percentage > blank_warning_threshold:
                self.alerts.append(
                    Alert(
                        column_name=column_name,
                        type="BLANK_STRINGS",
                        level=AlertLevel.WARNING,
                        message=(
                            f"Column contains {blank_percentage:.1%} blank/whitespace strings. "
                            "These may bypass standard completeness checks."
                        ),
                        value=blank_percentage,
                    )
                )

    def _check_constant(self, column_name: str, n_unique: int) -> None:
        """
        Alerts:
            - CONSTANT (CRITICAL): Only 1 unique value. The column provides no entropy.
        """
        constant_check_enabled = self.config.get_rule("constant_check_enabled", column_name)
        if not constant_check_enabled:
            return

        if n_unique == 1:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="CONSTANT",
                    level=AlertLevel.CRITICAL,
                    message="Column has only 1 unique value. It adds no variance to the dataset.",
                    value=1.0,
                )
            )

    def _check_primary_key(self, column_name: str, n_unique: int) -> None:
        """
        Analyzes distinct counts to detect Primary Keys.

        Alerts:
            - ALL_DISTINCT (INFO): Exact 100% uniqueness. Confirms a clean Primary Key.
            - ALL_DISTINCT_APPROX (INFO): Softened variant of ALL_DISTINCT for memory-safe runs.
            - DUPLICATE_KEYS (WARNING): Triggers when a presumed ID column falls slightly short.
        """

        min_rows_for_pk_check = self.config.get_rule("min_rows_for_pk_check", column_name)
        id_uniqueness_threshold = self.config.get_rule("id_uniqueness_threshold", column_name)

        if min_rows_for_pk_check in (False, None) or id_uniqueness_threshold in (False, None):
            return

        if self.row_count > min_rows_for_pk_check and n_unique > (
            self.row_count * id_uniqueness_threshold
        ):
            if self.is_memory_safe_run:
                self.alerts.append(
                    Alert(
                        column_name=column_name,
                        type="ALL_DISTINCT_APPROX",
                        level=AlertLevel.INFO,
                        message=(
                            "Column is ~100% distinct (Approximate). Likely a Primary Key or ID."
                        ),
                        value=n_unique,
                    )
                )
            elif n_unique == self.row_count:
                # Perfect Primary Key
                self.alerts.append(
                    Alert(
                        column_name=column_name,
                        type="ALL_DISTINCT",
                        level=AlertLevel.INFO,
                        message="Column is 100% distinct. Likely a Primary Key or ID.",
                        value=n_unique,
                    )
                )
            else:
                # Corrupted Primary Key
                duplicates = self.row_count - n_unique
                self.alerts.append(
                    Alert(
                        column_name=column_name,
                        type="DUPLICATE_KEYS",
                        level=AlertLevel.WARNING,
                        message=(
                            f"Column is almost unique, but contains {duplicates} duplicate values."
                        ),
                        value=n_unique,
                    )
                )

    def _check_duplicates(self, column_name: str, n_unique: int) -> None:
        """
        Validates the duplication ratio against the user limit.

        Alerts:
            - EXCESSIVE_DUPLICATES (WARNING/CRITICAL): Triggers when the percentage of
              duplicate rows crosses the allowed threshold.
        """

        max_duplicate_percent = self.config.get_rule("max_duplicate_percent", column_name)
        if max_duplicate_percent in (False, None):
            return

        duplicate_ratio = (self.row_count - n_unique) / max(self.row_count, 1)

        if duplicate_ratio > max_duplicate_percent:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="EXCESSIVE_DUPLICATES",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Column contains {duplicate_ratio:.1%} duplicate rows, "
                        f"exceeding the configured limit of {max_duplicate_percent:.1%}."
                    ),
                    value=duplicate_ratio,
                )
            )

    def _check_cardinality(self, column_name: str, n_unique: int) -> None:
        """
        Analyzes categorical density and unique value counts.

        Alerts:
            - HIGH_CARDINALITY (WARNING): Flags string columns with > 10,000 unique values.
              High cardinality strings will geometrically explode machine learning pipelines
              using One-Hot Encoding, requiring dimensionality reduction (e.g., Target Encoding).
        """
        high_cardinality_threshold = self.config.get_rule("high_cardinality_threshold", column_name)
        if high_cardinality_threshold in (False, None):
            return

        if n_unique > high_cardinality_threshold and n_unique < self.row_count:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="HIGH_CARDINALITY",
                    level=AlertLevel.WARNING,
                    message=f"High cardinality ({n_unique} unique values). Avoid One-Hot Encoding.",
                    value=n_unique,
                )
            )

    def _check_skew(self, column_name: str, skew: float) -> None:
        """
        Analyzes the third moment (asymmetry) of a statistical distribution.

        Alerts:
            - SKEWED (WARNING): |Skew| > 2.0. Extreme tails will heavily penalize linear models
              (e.g., OLS regression). Usually requires a log/box-cox transformation.
        """
        skew_threshold = self.config.get_rule("skew_threshold", column_name)
        if skew_threshold in (False, None):
            return

        if abs(skew) > skew_threshold:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="SKEWED",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Distribution is highly skewed ({skew:.2f}). "
                        "Linear models may require transformation."
                    ),
                    value=skew,
                )
            )

    def _check_zeros(self, column_name: str, zero_count: int) -> None:
        """
        Evaluates potential zero-inflation across numeric features.

        Alerts:
            - ZERO_INFLATED (WARNING): > 10% zeros. Often flags a scenario where the
              upstream data source used integer '0' to represent a missing/null state
              rather than an actual mathematical zero.
        """
        zero_inflated_threshold = self.config.get_rule("zero_inflated_threshold", column_name)
        if zero_inflated_threshold in (False, None):
            return

        if zero_count > 0 and self.row_count > 0:
            zero_percentage = zero_count / self.row_count

            if zero_percentage > zero_inflated_threshold:
                self.alerts.append(
                    Alert(
                        column_name=column_name,
                        type="ZERO_INFLATED",
                        level=AlertLevel.WARNING,
                        message=(
                            f"Column is {zero_percentage:.1%} zeros. "
                            "Check if '0' represents missing data."
                        ),
                        value=zero_percentage,
                    )
                )

    def _check_possible_numeric(self, column_name: str, top_k: list[dict[str, Any]]) -> None:
        """
        Heuristic schema-drift detector for misclassified columns.

        Uses the computationally cheap `Top-K` list generated in Pass 3 to determine if a
        string column is actually a hidden numeric column. By limiting the float-casting
        attempt to only the top 5 most frequent items, this check achieves O(1) complexity
        rather than an expensive O(N) full-column scan.

        Alerts:
            - POSSIBLE_NUMERIC (INFO): Recommendation to cast the pipeline schema to Int/Float
              to enable statistical profiling metrics.
        """
        possible_numeric_sample_size = self.config.get_rule(
            "possible_numeric_sample_size", column_name
        )
        if possible_numeric_sample_size in (False, None):
            return

        # We check the Top-K most frequent values. If the top 5 values
        # can all be cast to float, it is highly likely the column is numeric.
        # We use Top-K because checking every row is expensive (O(N)),
        # while Top-K is O(1) here.

        # 2. Check the sample (Top 5)
        # We only look at values that are NOT None
        sample_values = [
            top_k_item["value"]
            for top_k_item in top_k[:possible_numeric_sample_size]
            if top_k_item["value"] is not None
        ]

        if not sample_values:
            return

        # 3. Try to convert all sampled values
        try:
            # If ANY value fails conversion, the whole column is treated as String
            # This is strict but safe.
            for value in sample_values:
                float(value)

            # If we survived the loop, they are all numbers
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="POSSIBLE_NUMERIC",
                    level=AlertLevel.INFO,
                    message=("Top values look like numbers. Consider casting to Integer/Float."),
                    value=None,
                )
            )
        except ValueError:
            # None of the values in the sample are "numeric"
            pass

    def _check_correlations(self) -> None:
        """
        Identifies mathematically redundant columns via the correlation matrices.

        Iterates over the de-duplicated edge lists (A <-> B) produced in Pass 4.

        Alerts:
            - HIGH_CORRELATION (WARNING): |Score| > 0.99. Flags extreme collinearity,
              which destabilizes model weights and indicates data redundancy.
        """
        high_correlation_threshold = self.config.get_rule("high_correlation_threshold")
        if high_correlation_threshold in (False, None):
            return

        # We explicitly extract the lists using string literals so mypy
        # statically knows these are list[CorrelationPair] and not the sampling_method string.
        pearson_matrix = self.correlations.get("pearson", [])
        spearman_matrix = self.correlations.get("spearman", [])

        for method, matrix in [("pearson", pearson_matrix), ("spearman", spearman_matrix)]:
            for edge in matrix:
                column_a = edge["column_a"]
                column_b = edge["column_b"]
                score = edge["score"]

                if abs(score) > high_correlation_threshold:
                    self.alerts.append(
                        Alert(
                            column_name=f"{column_a} <-> {column_b}",
                            type="HIGH_CORRELATION",
                            level=AlertLevel.WARNING,
                            message=(
                                f"Columns are highly correlated ({score:.4f}) via {method}. "
                                "They contain redundant information."
                            ),
                            value=score,
                        )
                    )

    def _check_outliers(
        self,
        column_name: str,
        p25: float,
        p75: float,
        min_value: float | int,
        max_value: float | int,
    ) -> None:
        """
        Static heuristic for Extreme Outlier Detection using the Tukey IQR method.
        Alerts:
            - OUTLIERS_DETECTED (WARNING): Max or Min value severely exceeds mathematical bounds.
        """
        iqr_multiplier = self.config.get_rule("outlier_iqr_multiplier", column_name)
        if iqr_multiplier in (False, None):
            return

        iqr = p75 - p25
        if iqr == 0:
            return  # Prevents dividing by zero or overly aggressive bounds on tight distributions

        upper_bound = p75 + (iqr_multiplier * iqr)
        lower_bound = p25 - (iqr_multiplier * iqr)

        if max_value > upper_bound or min_value < lower_bound:
            extreme_value = max_value if max_value > upper_bound else min_value
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="OUTLIERS_DETECTED",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Extreme outliers present. The value {extreme_value:.2f} "
                        f"significantly exceeds the normal expected bounds."
                    ),
                    value=extreme_value,
                )
            )

    def _check_string_length(
        self, column_name: str, mean_length: float, max_length: int, min_length: int
    ) -> None:
        """
        Detects text fields where a single entry is drastically longer or shorter than the norm.
        Alerts:
            - INCONSISTENT_STRING_LENGTH (WARNING): Max or Min length vastly deviates from the mean.
        """
        length_multiplier = self.config.get_rule("string_length_anomaly_multiplier", column_name)
        if length_multiplier in (False, None) or mean_length == 0:
            return

        if max_length > (mean_length * length_multiplier):
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="INCONSISTENT_STRING_LENGTH",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Max string length ({max_length}) is highly disproportionate "
                        f"to the mean length ({mean_length:.1f}). Possible data corruption."
                    ),
                    value=max_length,
                )
            )

        # We strictly ensure min_length > 0 to prevent ZeroDivisionError.
        # (Length 0 is an empty string, which is a completeness issue, not a length anomaly).
        if min_length > 0 and (mean_length / min_length) > length_multiplier:
            self.alerts.append(
                Alert(
                    column_name=column_name,
                    type="INCONSISTENT_STRING_LENGTH",
                    level=AlertLevel.WARNING,
                    message=(
                        f"Min string length ({min_length}) is highly disproportionate "
                        f"to the mean length ({mean_length:.1f}). Possible severe truncation."
                    ),
                    value=min_length,
                )
            )
