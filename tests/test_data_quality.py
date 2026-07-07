"""Tests for data_quality — CB metrics sanity checks."""

import pandas as pd
import pytest
from core.data_quality import validate_cb_data, DATA_QUALITY_RULES, VALID_RATINGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_cb_df() -> pd.DataFrame:
    """Return a small DataFrame with all values in-range."""
    return pd.DataFrame({
        "cb_price": [120.0, 105.5, 98.0],
        "premium_rate": [15.0, 25.0, 10.0],
        "ytm": [0.02, 0.01, 0.03],
        "conversion_value": [80.0, 90.0, 110.0],
        "outstanding_balance": [5000.0, 8000.0, 3000.0],
        "double_low": [135.0, 130.5, 108.0],
        "redemption_ratio": [0.5, 0.0, 1.2],
        "putback_ratio": [0.0, 0.5, 0.0],
        "conversion_price": [10.0, 15.0, 8.5],
        "remaining_years_raw": [3.0, 2.5, 4.0],
        "credit_rating": ["AAA", "AA", "A+"],
        "maturity_date": ["20291231", "20280615", "20270310"],
    })


# ---------------------------------------------------------------------------
# Valid data — no warnings
# ---------------------------------------------------------------------------

class TestValidData:
    def test_all_valid_no_warnings(self):
        df = _valid_cb_df()
        result, warnings = validate_cb_data(df)
        assert len(warnings) == 0
        assert not result["_data_quality_flag"].any()


# ---------------------------------------------------------------------------
# Individual field range checks
# ---------------------------------------------------------------------------

class TestCbPrice:
    def test_price_below_min(self):
        df = _valid_cb_df()
        df.loc[0, "cb_price"] = 30.0
        result, warnings = validate_cb_data(df)
        assert any("cb_price" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_price_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "cb_price"] = 600.0
        result, warnings = validate_cb_data(df)
        assert any("cb_price" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_price_at_boundary_ok(self):
        df = _valid_cb_df()
        df.loc[0, "cb_price"] = 50.0  # min boundary
        df.loc[1, "cb_price"] = 500.0  # max boundary
        result, warnings = validate_cb_data(df)
        assert not any("cb_price" in w for w in warnings)
        assert not result.loc[0, "_data_quality_flag"]
        assert not result.loc[1, "_data_quality_flag"]


class TestPremiumRate:
    def test_premium_below_min(self):
        df = _valid_cb_df()
        df.loc[0, "premium_rate"] = -60.0
        result, warnings = validate_cb_data(df)
        assert any("premium_rate" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_premium_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "premium_rate"] = 600.0
        result, warnings = validate_cb_data(df)
        assert any("premium_rate" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


class TestYtm:
    def test_ytm_below_min(self):
        df = _valid_cb_df()
        df.loc[0, "ytm"] = -0.20
        result, warnings = validate_cb_data(df)
        assert any("ytm" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_ytm_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "ytm"] = 0.50
        result, warnings = validate_cb_data(df)
        assert any("ytm" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


class TestConversionValue:
    def test_cv_below_min(self):
        df = _valid_cb_df()
        df.loc[0, "conversion_value"] = -1.0
        result, warnings = validate_cb_data(df)
        assert any("conversion_value" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_cv_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "conversion_value"] = 6000.0
        result, warnings = validate_cb_data(df)
        assert any("conversion_value" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


class TestOutstandingBalance:
    def test_balance_negative(self):
        df = _valid_cb_df()
        df.loc[0, "outstanding_balance"] = -100.0
        result, warnings = validate_cb_data(df)
        assert any("outstanding_balance" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_balance_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "outstanding_balance"] = 2e7
        result, warnings = validate_cb_data(df)
        assert any("outstanding_balance" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


class TestDoubleLow:
    def test_double_low_below_min(self):
        df = _valid_cb_df()
        df.loc[0, "double_low"] = 30.0
        result, warnings = validate_cb_data(df)
        assert any("double_low" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_double_low_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "double_low"] = 600.0
        result, warnings = validate_cb_data(df)
        assert any("double_low" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


class TestRedemptionRatio:
    def test_redemption_ratio_negative(self):
        df = _valid_cb_df()
        df.loc[0, "redemption_ratio"] = -0.5
        result, warnings = validate_cb_data(df)
        assert any("redemption_ratio" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_redemption_ratio_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "redemption_ratio"] = 15.0
        result, warnings = validate_cb_data(df)
        assert any("redemption_ratio" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


class TestPutbackRatio:
    def test_putback_ratio_negative(self):
        df = _valid_cb_df()
        df.loc[0, "putback_ratio"] = -0.1
        result, warnings = validate_cb_data(df)
        assert any("putback_ratio" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_putback_ratio_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "putback_ratio"] = 20.0
        result, warnings = validate_cb_data(df)
        assert any("putback_ratio" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


class TestConversionPrice:
    def test_conversion_price_too_low(self):
        df = _valid_cb_df()
        df.loc[0, "conversion_price"] = 0.05
        result, warnings = validate_cb_data(df)
        assert any("conversion_price" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_conversion_price_too_high(self):
        df = _valid_cb_df()
        df.loc[0, "conversion_price"] = 1000.0
        result, warnings = validate_cb_data(df)
        assert any("conversion_price" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


class TestRemainingYears:
    def test_remaining_years_negative(self):
        df = _valid_cb_df()
        df.loc[0, "remaining_years_raw"] = -0.5
        result, warnings = validate_cb_data(df)
        assert any("remaining_years_raw" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_remaining_years_above_max(self):
        df = _valid_cb_df()
        df.loc[0, "remaining_years_raw"] = 10.0
        result, warnings = validate_cb_data(df)
        assert any("remaining_years_raw" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]


# ---------------------------------------------------------------------------
# Credit rating validation
# ---------------------------------------------------------------------------

class TestCreditRating:
    def test_invalid_rating(self):
        df = _valid_cb_df()
        df.loc[0, "credit_rating"] = "ZZZ"
        result, warnings = validate_cb_data(df)
        assert any("信用评级" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_all_valid_ratings_ok(self):
        df = _valid_cb_df()
        df["credit_rating"] = ["AAA", "AA+", "BBB+"]
        result, warnings = validate_cb_data(df)
        assert not any("信用评级" in w for w in warnings)

    def test_na_rating_not_flagged(self):
        df = _valid_cb_df()
        df.loc[0, "credit_rating"] = None
        result, warnings = validate_cb_data(df)
        assert not any("信用评级" in w for w in warnings)
        assert not result.loc[0, "_data_quality_flag"]

    def test_rating_column_cn_name(self):
        df = _valid_cb_df()
        df = df.drop(columns=["credit_rating"])
        df["债券评级"] = ["ZZZ", "AA", "BBB"]
        result, warnings = validate_cb_data(df)
        assert any("信用评级" in w for w in warnings)


# ---------------------------------------------------------------------------
# Maturity date format
# ---------------------------------------------------------------------------

class TestMaturityDate:
    def test_bad_date_format(self):
        df = _valid_cb_df()
        df.loc[0, "maturity_date"] = "2026"
        result, warnings = validate_cb_data(df)
        assert any("maturity_date" in w for w in warnings)
        assert result.loc[0, "_data_quality_flag"]

    def test_good_date_format(self):
        df = _valid_cb_df()
        df["maturity_date"] = ["20251231", "20260615", "20270310"]
        result, warnings = validate_cb_data(df)
        assert not any("maturity_date" in w for w in warnings)

    def test_na_date_not_flagged(self):
        df = _valid_cb_df()
        df.loc[0, "maturity_date"] = None
        result, warnings = validate_cb_data(df)
        assert not any("maturity_date" in w for w in warnings)
        assert not result.loc[0, "_data_quality_flag"]


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------

class TestMissingColumns:
    def test_missing_optional_columns_no_crash(self):
        """validate_cb_data should skip fields not present in the DataFrame."""
        df = pd.DataFrame({
            "cb_price": [120.0, 105.5],
            "premium_rate": [15.0, 25.0],
        })
        result, warnings = validate_cb_data(df)
        # Only cb_price and premium_rate are present, both valid → no warnings
        assert len(warnings) == 0

    def test_missing_all_columns_no_crash(self):
        df = pd.DataFrame({"some_other_col": [1, 2, 3]})
        result, warnings = validate_cb_data(df)
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Multiple issues
# ---------------------------------------------------------------------------

class TestMultipleIssues:
    def test_multiple_outliers_multiple_warnings(self):
        df = _valid_cb_df()
        df.loc[0, "cb_price"] = 10.0
        df.loc[0, "premium_rate"] = 600.0
        df.loc[0, "credit_rating"] = "INVALID"
        result, warnings = validate_cb_data(df)
        assert len(warnings) >= 3
        assert result.loc[0, "_data_quality_flag"]

    def test_all_rows_flagged(self):
        df = _valid_cb_df()
        df["cb_price"] = [1.0, 2.0, 3.0]  # all below min
        result, warnings = validate_cb_data(df)
        assert result["_data_quality_flag"].sum() == 3


# ---------------------------------------------------------------------------
# DATA_QUALITY_RULES constant
# ---------------------------------------------------------------------------

class TestDataQualityRulesConstant:
    def test_rules_have_expected_fields(self):
        fields = {r[0] for r in DATA_QUALITY_RULES}
        assert "cb_price" in fields
        assert "premium_rate" in fields
        assert "ytm" in fields
        assert "conversion_value" in fields
        assert "double_low" in fields

    def test_each_rule_has_four_elements(self):
        for rule in DATA_QUALITY_RULES:
            assert len(rule) == 4


# ---------------------------------------------------------------------------
# VALID_RATINGS constant
# ---------------------------------------------------------------------------

class TestValidRatingsConstant:
    def test_ratings_frozenset(self):
        assert isinstance(VALID_RATINGS, frozenset)

    def test_contains_common_ratings(self):
        for rating in ["AAA", "AA+", "AA", "AA-", "A+", "A", "BBB+"]:
            assert rating in VALID_RATINGS
