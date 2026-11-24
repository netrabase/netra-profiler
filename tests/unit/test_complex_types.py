import polars as pl
import pytest

from netra_profiler import Profiler


def test_complex_types_flattening() -> None:
    """
    Verifies that:
    1. Structs are flattened into separate columns.
    2. Lists and fixed-size Arrays are converted to length integers.
    3. The profiler generates stats for these new columns.
    """
    df = pl.DataFrame(
        {
            "user": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": None, "age": None},
            ],
            "tags": [
                ["pro", "admin"],
                ["newbie"],
                [],
            ],
            "coords": pl.Series(
                "coords",
                [[10, 20], [30, 40], [50, 60]],
                dtype=pl.Array(pl.Int64, 2),
            ),
        }
    )

    profiler = Profiler(df)
    profile = profiler.run()
    schema = profiler._df.collect_schema()

    # Check Struct Flattening (user -> user_name, user_age)
    assert "user" not in schema, "Original struct column was not removed."
    assert "user_name" in schema, "Struct field 'name' was not flattened."
    assert "user_age" in schema, "Struct field 'age' was not flattened."

    # Check List & Array Transformation (tags -> tags_length, coords -> coords_length)
    assert "tags" not in schema, "Original list column was not removed."
    assert "tags_length" in schema, "List column was not converted to length."
    assert "coords" not in schema, "Original array column was not removed."
    assert "coords_length" in schema, "Array column was not converted to length."
    assert profile["columns"]["coords_length"].get("min") == 2
    assert profile["columns"]["coords_length"].get("max") == 2
    assert profile["columns"]["coords_length"].get("n_unique") == 1

    # Expected Age Mean
    expected_age_mean = df.select(pl.col("user").struct.field("age").mean()).item()
    user_age_mean = profile["columns"]["user_age"].get("mean")
    assert user_age_mean is not None
    assert user_age_mean == pytest.approx(expected_age_mean)

    # Expected Name Null Count
    expected_null_count = df.select(pl.col("user").struct.field("name").null_count()).item()
    assert profile["columns"]["user_name"].get("null_count") == expected_null_count

    # Expected List Length Stats
    lengths_df = df.select(pl.col("tags").list.len().alias("length"))
    expected_min_length = lengths_df.select(pl.min("length")).item()
    expected_max_length = lengths_df.select(pl.max("length")).item()
    expected_mean_length = lengths_df.select(pl.mean("length")).item()

    assert profile["columns"]["tags_length"].get("min") == expected_min_length
    assert profile["columns"]["tags_length"].get("max") == expected_max_length

    tags_length_mean = profile["columns"]["tags_length"].get("mean")
    assert tags_length_mean is not None
    assert tags_length_mean == pytest.approx(expected_mean_length)
