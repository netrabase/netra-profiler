import copy

import pytest

from netra_profiler.diff import DiffEngine
from netra_profiler.types import NetraProfile

DELTA_PERCENTAGE_THRESHOLD = 0.50
ALERTS_COUNT = 3


@pytest.fixture
def baseline_profile() -> NetraProfile:
    """A highly controlled baseline profile for deterministic testing."""
    return {
        "dataset": {"name": "test_data", "format": "CSV", "row_count": 1000},
        "columns": {
            "id": {"data_type": "Int64", "null_count": 0, "n_unique": 1000},
            "age": {"data_type": "Int64", "null_count": 50, "mean": 30.0, "max": 100.0},
            "name": {"data_type": "Utf8", "null_count": 0, "mean_length": 5.0},
            "balance": {"data_type": "Float64", "mean": 0.0},  # Used for the Infinity zero-test
        },
        "correlations": {"pearson": [], "spearman": [], "sampling_method": None},
        "alerts": [],
        "_meta": {},  # type: ignore
    }


def test_diff_identical_profiles(baseline_profile):
    """If profiles are identical, zero alerts should be generated."""
    target_profile = copy.deepcopy(baseline_profile)

    engine = DiffEngine(baseline_profile, target_profile)
    report = engine.run()

    assert len(report["alerts"]) == 0


def test_schema_changes(baseline_profile):
    """Test dropping a column, adding a column, and changing a type."""
    target_profile = copy.deepcopy(baseline_profile)

    # 1. Drop 'name'
    del target_profile["columns"]["name"]
    # 2. Add 'income'
    target_profile["columns"]["income"] = {"data_type": "Float64", "null_count": 0}
    # 3. Mutate 'id' type
    target_profile["columns"]["id"]["data_type"] = "Utf8"

    engine = DiffEngine(baseline_profile, target_profile)
    report = engine.run()

    alerts = report["alerts"]
    assert len(alerts) == ALERTS_COUNT

    diff_types = [a["diff_type"] for a in alerts]
    assert "COLUMN_DROPPED" in diff_types
    assert "COLUMN_ADDED" in diff_types
    assert "TYPE_CHANGED" in diff_types


def test_volume_shift(baseline_profile):
    """Test that a >20% row count shift triggers an alert."""
    target_profile = copy.deepcopy(baseline_profile)
    target_profile["dataset"]["row_count"] = 1500  # 50% increase

    engine = DiffEngine(baseline_profile, target_profile)
    report = engine.run()

    alerts = report["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["diff_type"] == "VOLUME_SHIFT"
    assert alerts[0]["delta_percentage"] == DELTA_PERCENTAGE_THRESHOLD


def test_scalar_shift(baseline_profile):
    """Test that metric shifts (mean, nulls) are calculated correctly."""
    target_profile = copy.deepcopy(baseline_profile)

    # Baseline age mean is 30.0. A 10% shift would be 33.0. Let's push it to 45.0 (50% shift).
    target_profile["columns"]["age"]["mean"] = 45.0

    engine = DiffEngine(baseline_profile, target_profile)
    report = engine.run()

    alerts = report["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["column_name"] == "age"
    assert alerts[0]["metric"] == "mean"
    assert alerts[0]["delta_percentage"] == DELTA_PERCENTAGE_THRESHOLD


def test_zero_to_value_infinity_shift(baseline_profile):
    """Ensure shifting from exactly 0.0 to a value returns 'Infinity' and doesn't crash."""
    target_profile = copy.deepcopy(baseline_profile)

    # Baseline balance mean is 0.0. Let's shift it to 500.0.
    target_profile["columns"]["balance"]["mean"] = 500.0

    engine = DiffEngine(baseline_profile, target_profile)
    report = engine.run()

    alerts = report["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["column_name"] == "balance"
    assert alerts[0]["delta_percentage"] == "Infinity"
    assert "shifted from 0" in alerts[0]["message"]
