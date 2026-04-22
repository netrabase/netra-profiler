from dataclasses import dataclass
from enum import Enum
from typing import Any

from netra_profiler.types import NetraProfile, is_numeric_type


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


@dataclass
class DiagnosticConfig:
    """
    Dynamic configuration manager for diagnostic thresholds.
    Handles global defaults, column overrides, and type validation.
    """

    DEFAULT_GLOBALS = {
        "null_critical_threshold": 0.95,
        "null_warning_threshold": 0.50,
        "skew_threshold": 2.0,
        "zero_inflated_threshold": 0.10,
        "high_cardinality_threshold": 10000,
        "high_correlation_threshold": 0.95,
        "id_uniqueness_threshold": 0.99,
        "min_rows_for_pk_check": 100,
        "possible_numeric_sample_size": 5,
        "outlier_iqr_multiplier": 3.0,
        "string_length_anomaly_multiplier": 50.0,
        "constant_check_enabled": True,
    }

    def __init__(self, config_dict: dict[str, Any] | None = None):
        self.global_thresholds = self.DEFAULT_GLOBALS.copy()
        self.column_overrides: dict[str, dict[str, Any]] = {}
        self.unused_overrides: set[str] = set()

        if config_dict and "diagnostics" in config_dict:
            diagnostics_config = config_dict["diagnostics"]

            # 1. Load and Validate Globals
            if "global_thresholds" in diagnostics_config:
                for key, value in diagnostics_config["global_thresholds"].items():
                    if key in self.global_thresholds:
                        self.global_thresholds[key] = self._validate_type(key, value)

            # 2. Load Overrides
            if "column_overrides" in diagnostics_config:
                raw_overrides = diagnostics_config["column_overrides"]
                for column_name, rules in raw_overrides.items():
                    validated_rules = {}
                    for rule_key, rule_value in rules.items():
                        if rule_key in self.global_thresholds:
                            validated_rules[rule_key] = self._validate_type(rule_key, rule_value)
                    self.column_overrides[column_name] = validated_rules

                self.unused_overrides = set(self.column_overrides.keys())

    def _validate_type(self, key: str, value: Any) -> Any:
        """Ensures YAML config inputs match the expected column data types."""
        if value is None or value is False:
            return value  # Allow null/false to explicitly disable a check

        expected_type = type(self.DEFAULT_GLOBALS[key])
        if not isinstance(value, expected_type):
            try:
                # Soft cast (e.g., if they pass int '1' for a float threshold)
                return expected_type(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Config Error: '{key}' must be of type {expected_type.__name__}. "
                    f"Received: '{value}'"
                ) from None
        return value

    def get_rule(self, rule_name: str, column_name: str | None = None) -> Any:
        """
        Returns quality metric threshold by checking column overrides first,
        and then falling back to global default. Pass empty column_name to
        get the global default.
        """
        if column_name and column_name in self.column_overrides:
            # We mark this override as 'used'
            if column_name in self.unused_overrides:
                self.unused_overrides.remove(column_name)

            column_rules = self.column_overrides[column_name]
            if rule_name in column_rules:
                return column_rules[rule_name]

        return self.global_thresholds[rule_name]

    def get_unused_override_warnings(self, dataset_columns: list[str]) -> list[str]:
        """Returns warnings for any column overrides that didn't match dataset columns."""
        warnings_list = []
        for column in self.unused_overrides:
            if column not in dataset_columns:
                warnings_list.append(
                    f"ConfigWarning: Override for '{column}' was ignored (column not found)."
                )
        return warnings_list


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

            # 2. Numeric Checks
            if is_numeric:
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

            # 3. String / Categorical Checks
            if not is_numeric:
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

        # 4. Correlation Check
        self._check_correlations()

        return self.alerts

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
