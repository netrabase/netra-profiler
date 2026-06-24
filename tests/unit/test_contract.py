"""
Unit tests for the ODCS contract generation functionality.
"""

from typing import cast
from unittest.mock import MagicMock

import pytest
import yaml
from typing_extensions import Any

from netra_profiler.config import DiagnosticConfig
from netra_profiler.odcs import ODCSBuilder
from netra_profiler.types import NetraProfile

EMAIL_MIN_LENGTH = 5
EMAIL_MAX_LENGTH = 254
MINIMUM_ORDERS = 1.99
MAXIMUM_ORDERS = 1499.99
DUPLICATE_THRESHOLD = 25.0  # 0.25 * 100
NULL_THRESHOLD = 8.0  # 0.08 * 100


@pytest.fixture
def sample_netra_profile() -> NetraProfile:
    """Returns a synthetic NetraProfile dictionary across multiple types."""

    profile_data = {
        "dataset": {
            "name": "ecommerce_orders",
            "row_count": 1000,
        },
        "columns": {
            "order_id": {
                "data_type": "Int64",
                "null_count": 0,
                "n_unique": 1000,
            },
            "customer_email": {
                "data_type": "String",
                "null_count": 50,
                "n_unique": 800,
                "min_length": 5,
                "max_length": 254,
            },
            "order_total": {
                "data_type": "Float64",
                "null_count": 0,
                "n_unique": 950,
                "min": 1.99,
                "max": 1499.99,
            },
            "platform_id": {
                "data_type": "Int32",
                "null_count": 0,
                "n_unique": 1,
            },
        },
    }

    return cast(NetraProfile, profile_data)


@pytest.fixture
def mock_diagnostic_config() -> DiagnosticConfig:
    """Returns a mock DiagnosticConfig instance to simulate configuration thresholds."""

    config = MagicMock(spec=DiagnosticConfig)

    def get_rule_side_effect(rule_name: str, column_name: str) -> Any:
        if rule_name == "id_uniqueness_threshold" and column_name == "order_id":
            return 0.99
        if rule_name == "null_critical_threshold" and column_name == "customer_email":
            return 0.08  # 8%
        if rule_name == "max_duplicate_percent" and column_name == "customer_email":
            return 0.25  # 25%
        if rule_name == "constant_check_enabled" and column_name == "platform_id":
            return True
        return None

    config.get_rule.side_effect = get_rule_side_effect
    config.column_overrides = {"platform_id": {"constant_check_enabled": True}}
    return config


def test_odcs_builder_metadata_and_structure(
    sample_netra_profile: NetraProfile, mock_diagnostic_config: DiagnosticConfig
) -> None:
    """Verifies baseline ODCS document metadata structure and dataset block mapping."""

    builder = ODCSBuilder(sample_netra_profile, mock_diagnostic_config)
    yaml_output = builder.build()
    parsed_yaml = yaml.safe_load(yaml_output)

    assert parsed_yaml["apiVersion"] == "v3.1.0"
    assert parsed_yaml["kind"] == "DataContract"
    assert parsed_yaml["version"] == "0.0.1"
    assert parsed_yaml["status"] == "draft"
    assert parsed_yaml["dataset"]["name"] == "ecommerce_orders"
    assert isinstance(parsed_yaml["schema"], list)


def test_logical_type_and_bounds_mapping(
    sample_netra_profile: NetraProfile, mock_diagnostic_config: DiagnosticConfig
) -> None:
    """Asserts native Polars types map properly to logical ODCS types and values."""

    builder = ODCSBuilder(sample_netra_profile, mock_diagnostic_config)
    parsed_yaml = yaml.safe_load(builder.build())

    properties = parsed_yaml["schema"][0]["properties"]
    prop_dict = {p["name"]: p for p in properties}

    # Verify Integer Mapping
    assert prop_dict["order_id"]["logicalType"] == "integer"
    assert prop_dict["order_id"]["physicalType"] == "Int64"

    # Verify String Mapping & Bounds Options
    assert prop_dict["customer_email"]["logicalType"] == "string"
    assert prop_dict["customer_email"]["logicalTypeOptions"]["minLength"] == EMAIL_MIN_LENGTH
    assert prop_dict["customer_email"]["logicalTypeOptions"]["maxLength"] == EMAIL_MAX_LENGTH

    # Verify Number/Float Mapping & Bounds Options
    assert prop_dict["order_total"]["logicalType"] == "number"
    assert prop_dict["order_total"]["logicalTypeOptions"]["minimum"] == MINIMUM_ORDERS
    assert prop_dict["order_total"]["logicalTypeOptions"]["maximum"] == MAXIMUM_ORDERS


def test_completeness_inversion_and_quality_rules(
    sample_netra_profile: NetraProfile, mock_diagnostic_config: DiagnosticConfig
) -> None:
    """Validates that zero nulls sets 'required' while non-zero nulls creates percentage checks."""

    builder = ODCSBuilder(sample_netra_profile, mock_diagnostic_config)
    parsed_yaml = yaml.safe_load(builder.build())

    properties = parsed_yaml["schema"][0]["properties"]
    properties_dict = {p["name"]: p for p in properties}

    # 0 Nulls: Must be marked required, no null values quality check metric injected
    assert properties_dict["order_id"].get("required") is True
    if "quality" in properties_dict["order_id"]:
        metrics = [q["metric"] for q in properties_dict["order_id"]["quality"] if "metric" in q]
        assert "nullValues" not in metrics

    # >0 Nulls: Must not be marked required, quality metric generated and converted to percentage
    assert "required" not in properties_dict["customer_email"]
    quality_rules = properties_dict["customer_email"]["quality"]

    null_rule = next(r for r in quality_rules if r.get("metric") == "nullValues")
    assert null_rule["mustBeLessThan"] == NULL_THRESHOLD
    assert null_rule["unit"] == "percent"


def test_duplicate_percentage_conversion(
    sample_netra_profile: NetraProfile, mock_diagnostic_config: DiagnosticConfig
) -> None:
    """Ensures duplicate percentage configurations are accurately scaled to values out of 100."""

    builder = ODCSBuilder(sample_netra_profile, mock_diagnostic_config)
    parsed_yaml = yaml.safe_load(builder.build())

    properties = parsed_yaml["schema"][0]["properties"]
    customer_email_prop = next(p for p in properties if p["name"] == "customer_email")

    duplicate_rule = next(
        r for r in customer_email_prop["quality"] if r.get("metric") == "duplicateValues"
    )
    assert duplicate_rule["mustBeLessThan"] == DUPLICATE_THRESHOLD
    assert duplicate_rule["unit"] == "percent"


def test_variance_constant_sql_fallback(
    sample_netra_profile: NetraProfile, mock_diagnostic_config: DiagnosticConfig
) -> None:
    """Verifies custom SQL check injection for single-value column variance boundaries."""

    builder = ODCSBuilder(sample_netra_profile, mock_diagnostic_config)
    parsed_yaml = yaml.safe_load(builder.build())

    properties = parsed_yaml["schema"][0]["properties"]
    platform_prop = next(p for p in properties if p["name"] == "platform_id")

    sql_rule = next(r for r in platform_prop["quality"] if r.get("type") == "sql")
    assert "SELECT COUNT(DISTINCT {property}) FROM {object}" in sql_rule["query"]
    assert sql_rule["mustBeGreaterThan"] == 1


def test_string_replacement_pipeline_and_comments(
    sample_netra_profile: NetraProfile, mock_diagnostic_config: DiagnosticConfig
) -> None:
    """Validates the raw YAML post-processing string replacement inserts comments cleanly."""

    builder = ODCSBuilder(sample_netra_profile, mock_diagnostic_config)
    yaml_output = builder.build()

    # 1. Verify schema comment substitution happened
    assert "# --- SCHEMA BLOCK ---" in yaml_output
    assert "# Data Quality constraints are drafted from empirical data." in yaml_output

    # 2. Verify primary key comment translation replaced the hidden hint tracking key
    assert "__netra_pk_hint" not in yaml_output
    assert (
        "# Candidate Primary Key (Highly Unique). Uncomment the line below to enforce:"
        in yaml_output
    )
    assert "# primaryKey: true" in yaml_output

    # 3. Verify that string operations didn't affect parsing of the YAML contract
    parsed_yaml = yaml.safe_load(yaml_output)
    assert parsed_yaml["apiVersion"] == "v3.1.0"
