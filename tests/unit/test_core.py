import polars as pl
import pytest

from netra_profiler import Profiler
from netra_profiler.types import NetraProfile


@pytest.fixture(scope="module")
def sample_df() -> pl.DataFrame:
    """
    Creates a synthetic DataFrame with mixed types
    and edge cases(nulls, duplicates, negative numbers).
    """
    return pl.DataFrame(
        {
            "age": [25, 30, 35, None, 25],  # Numeric with nulls & duplicates
            "salary": [50000.0, 0.0, 75000.0, 50000.0, None],  # Float with nulls
            "city": ["Groningen", "Thrissur", "Delhi", None, "Groningen"],  # String
        }
    )


@pytest.fixture(scope="module")
def profile(sample_df: pl.DataFrame) -> NetraProfile:
    """
    Runs the profiler ONCE and returns the result.
    This fixture is shared by all tests below.
    """
    profiler = Profiler(sample_df)
    profile = profiler.run()

    # DEBUG: Print warnings if they exist
    if profile["_meta"]["warnings"]:
        print("\n!!! WARNINGS FOUND !!!")
        for warning in profile["_meta"]["warnings"]:
            print(f"- {warning}")

    return profile


def test_global_stats(profile: NetraProfile, sample_df: pl.DataFrame) -> None:
    """Verifies table-level statistics."""
    assert profile["dataset"]["row_count"] == sample_df.height


def test_numeric_stats(profile: NetraProfile, sample_df: pl.DataFrame) -> None:
    """Verifies numeric column statistics (Age)."""
    age_column = profile["columns"]["age"]

    # We use .get() to access the keys because some keys are optional
    # in ColumnMetrics and mypy doesn't like it when you try to directly
    # access the keys by index (age_column["min"]) for optional keys.
    assert age_column.get("min") == sample_df["age"].min()
    assert age_column.get("max") == sample_df["age"].max()
    assert age_column.get("null_count") == sample_df["age"].null_count()

    assert age_column.get("zero_count") == 0  # Age has no zeros
    salary_column = profile["columns"]["salary"]
    assert salary_column.get("zero_count") == 1  # Salary has exactly one 0.0

    # For approx, we ensure it's not None to prevent pytest TypeErrors
    age_mean = age_column.get("mean")
    assert age_mean is not None
    assert age_mean == pytest.approx(sample_df["age"].mean())

    age_std = age_column.get("std")
    assert age_std is not None
    assert age_std == pytest.approx(sample_df["age"].std())

    assert age_column.get("p25") == sample_df["age"].quantile(0.25)
    assert age_column.get("p50") == sample_df["age"].median()
    assert age_column.get("p75") == sample_df["age"].quantile(0.75)


def test_string_stats(profile: NetraProfile, sample_df: pl.DataFrame) -> None:
    """Verifies string column statistics (City)."""
    city_column = profile["columns"]["city"]

    assert city_column.get("null_count") == sample_df["city"].null_count()
    assert city_column.get("n_unique") is not None

    # Lexicographical min/max
    # "Delhi" should be min alphabetical
    # "Thrissur" should be max alphabetical
    assert city_column.get("min") == "Delhi"
    assert city_column.get("max") == "Thrissur"

    # String Lengths
    # Groningen (9), Thrissur (8), Delhi (5)
    city_lengths = sample_df["city"].str.len_chars()
    assert city_column.get("min_length") == city_lengths.min()  # 5
    assert city_column.get("max_length") == city_lengths.max()  # 9

    city_mean_length = city_column.get("mean_length")
    assert city_mean_length is not None
    assert city_mean_length == pytest.approx(city_lengths.mean())


def test_top_k_stats(profile: NetraProfile, sample_df: pl.DataFrame) -> None:
    """Verifies Top-K frequent item generation."""
    city_top_k = profile["columns"]["city"].get("top_k", [])

    # Check Top-K Structure (City)
    # Expected: A list of dicts like [{'value': 'Groningen', 'count': 2}, ...]
    assert isinstance(city_top_k, list)
    assert len(city_top_k) > 0

    # Verify counts for the most frequent item
    most_frequent_city = city_top_k[0]["value"]
    expected_count = sample_df.filter(pl.col("city") == most_frequent_city).height
    assert city_top_k[0]["count"] == expected_count


def test_histogram_stats(profile: NetraProfile) -> None:
    """Verifies Histogram generation."""
    age_histogram = profile["columns"]["age"].get("histogram", [])

    assert isinstance(age_histogram, list)
    assert len(age_histogram) > 0

    # Verify Struct keys
    first_bin = age_histogram[0]
    # # Polars Histogram Struct keys: 'breakpoint', 'category', 'count'
    # The 'category' key holds the range string e.g. "(25, 30]"
    assert "breakpoint" in first_bin
    assert "count" in first_bin


def test_correlation_stats(profile: NetraProfile) -> None:
    """Verifies Correlation Matrix generation."""
    correlations = profile["correlations"]

    # 1. Pearson
    pearson_data = correlations.get("pearson", [])
    assert isinstance(pearson_data, list)
    assert len(pearson_data) > 0

    first_pair = pearson_data[0]
    assert "column_a" in first_pair
    assert "column_b" in first_pair
    assert "score" in first_pair

    # Verify self-correlation is 1.0
    for pair in pearson_data:
        assert pair["column_a"] != pair["column_b"], "Self-correlation detected!"

    # 2. Spearman
    spearman_data = correlations.get("spearman", [])
    assert len(spearman_data) > 0

    # 3. Method Metadata
    assert correlations["sampling_method"] == "exact"


def test_metadata(profile: NetraProfile) -> None:
    """Verifies execution metadata."""
    meta = profile["_meta"]

    assert "engine_time_seconds" in meta
    assert meta["engine_time_seconds"] > 0
    assert isinstance(meta["warnings"], list)
    assert len(meta["warnings"]) == 0


def test_alerts(profile: NetraProfile) -> None:
    """Verifies that the diagnostic engine generates alerts."""
    alerts = profile["alerts"]
    assert isinstance(alerts, list)
