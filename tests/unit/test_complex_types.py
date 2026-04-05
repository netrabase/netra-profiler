import polars as pl
import pytest

from netra_profiler import Profiler


def test_complex_types_flattening() -> None:
    """
    Verifies that:
    1. Structs are flattened into separate columns.
    2. Lists are converted to length integers.
    3. The profiler generates stats for these new columns.
    """
    # 1. Setup: Create nested data (Simulating a JSON log)
    df = pl.DataFrame(
        {
            "user": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": None, "age": None},  # Null inside struct
            ],
            "tags": [
                ["pro", "admin"],
                ["newbie"],
                [],  # Empty list
            ],
        }
    )

    # 2. Execution
    # Initialize the profiler (This triggers preprocess_complex_types)
    profiler = Profiler(df)
    # Run Profile
    profile = profiler.run()

    # 3. Validation
    # We first inspect the transformed schema to ensure flattening happened
    # Note: We access the internal lazyframe's schema to see the transformation
    schema = profiler._df.collect_schema()

    # Check Struct Flattening (user -> user_name, user_age)
    assert "user" not in schema, "Original struct column was not removed."
    assert "user_name" in schema, "Struct field 'name' was not flattened."
    assert "user_age" in schema, "Struct field 'age' was not flattened."

    # Check List Transformation (tags -> tags_len)
    assert "tags" not in schema, "Original list column was not removed."
    assert "tags_length" in schema, "List column was not converted to length."

    # 4. Dynamic Assertion
    # Expected Age Mean
    # Extract the 'age' field from the struct and calculate mean
    expected_age_mean = df.select(pl.col("user").struct.field("age").mean()).item()
    user_age_mean = profile["columns"]["user_age"].get("mean")
    assert user_age_mean is not None
    assert user_age_mean == pytest.approx(expected_age_mean)

    # Expected Name Null Count
    expected_null_count = df.select(pl.col("user").struct.field("name").null_count()).item()
    assert profile["columns"]["user_name"].get("null_count") == expected_null_count

    # Expected List Length Stats
    # Calculate lengths of the lists
    lengths_df = df.select(pl.col("tags").list.len().alias("length"))

    expected_min_length = lengths_df.select(pl.min("length")).item()
    expected_max_length = lengths_df.select(pl.max("length")).item()
    expected_mean_length = lengths_df.select(pl.mean("length")).item()

    assert profile["columns"]["tags_length"].get("min") == expected_min_length
    assert profile["columns"]["tags_length"].get("max") == expected_max_length

    tags_length_mean = profile["columns"]["tags_length"].get("mean")
    assert tags_length_mean is not None
    assert tags_length_mean == pytest.approx(expected_mean_length)
