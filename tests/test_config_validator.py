"""Tests for config_validator — config.json value sanity checks."""

import pytest
from core.config_validator import validate_config, RATING_RANKS


# ---------------------------------------------------------------------------
# Helper: build a minimal valid config
# ---------------------------------------------------------------------------

def _base_config() -> dict:
    return {
        "scoring": {
            "valuation_weight": 0.40,
            "clause_weight": 0.30,
            "linkage_weight": 0.20,
            "structure_weight": 0.10,
            "dimension_floor": 0.30,
            "credit_penalty": -20,
            "liquidity_penalty": -10,
        },
        "detector_weights": {
            "double_low": 4,
            "ytm_defense": 3,
            "bond_floor": 3,
        },
        "backtest": {
            "forward_days": 5,
            "n_quintiles": 5,
            "ic_horizons": [1, 3, 5, 10, 20],
            "cost_model": {
                "stamp_duty": 0.0005,
                "commission": 0.0001,
                "slippage": 0.0001,
            },
        },
        "scan": {"lookback_days": 120},
        "valuation": {
            "double_low_price_max": 120,
            "double_low_premium_max": 20,
            "ytm_threshold": 3.0,
        },
        "risk": {"credit_exclude_rated_below": "A"},
        "clause": {
            "redemption_warn_ratio": 1.20,
            "redemption_danger_ratio": 1.28,
            "putback_consecutive_days": 30,
        },
        "options": {
            "risk_free_rate": 0.025,
            "iv_low_percentile": 25,
            "iv_high_percentile": 75,
            "hv_window": 60,
        },
        "stock_linkage": {
            "momentum_bullish_threshold": 3.0,
            "momentum_bearish_threshold": -3.0,
        },
        "llm": {
            "max_tokens": 2048,
            "timeout": 120.0,
            "top_n": 5,
        },
    }


# ---------------------------------------------------------------------------
# Valid config
# ---------------------------------------------------------------------------

class TestValidConfig:
    def test_valid_config_no_warnings(self):
        cfg = _base_config()
        assert validate_config(cfg) == []


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

class TestScoringWeights:
    def test_weight_sum_not_one(self):
        cfg = _base_config()
        cfg["scoring"]["valuation_weight"] = 0.90
        warnings = validate_config(cfg)
        assert any("权重之和" in w for w in warnings)

    def test_negative_weight(self):
        cfg = _base_config()
        cfg["scoring"]["valuation_weight"] = -0.40
        warnings = validate_config(cfg)
        assert any("valuation_weight" in w and "负数" in w for w in warnings)

    def test_weight_sum_within_tolerance(self):
        """Sum of 0.97 (within 5% tolerance) should not trigger warning."""
        cfg = _base_config()
        cfg["scoring"]["valuation_weight"] = 0.37
        warnings = validate_config(cfg)
        assert not any("权重之和" in w for w in warnings)


# ---------------------------------------------------------------------------
# Detector weights
# ---------------------------------------------------------------------------

class TestDetectorWeights:
    def test_negative_detector_weight(self):
        cfg = _base_config()
        cfg["detector_weights"]["double_low"] = -1
        warnings = validate_config(cfg)
        assert any("double_low" in w for w in warnings)

    def test_zero_detector_weight_ok(self):
        cfg = _base_config()
        cfg["detector_weights"]["double_low"] = 0
        warnings = validate_config(cfg)
        assert not any("double_low" in w for w in warnings)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_forward_days_lt_one(self):
        cfg = _base_config()
        cfg["backtest"]["forward_days"] = 0
        warnings = validate_config(cfg)
        assert any("forward_days" in w for w in warnings)

    def test_n_quintiles_lt_two(self):
        cfg = _base_config()
        cfg["backtest"]["n_quintiles"] = 1
        warnings = validate_config(cfg)
        assert any("n_quintiles" in w for w in warnings)

    def test_ic_horizons_non_positive(self):
        cfg = _base_config()
        cfg["backtest"]["ic_horizons"] = [1, 0, 5]
        warnings = validate_config(cfg)
        assert any("ic_horizons" in w for w in warnings)

    def test_cost_model_negative(self):
        cfg = _base_config()
        cfg["backtest"]["cost_model"]["stamp_duty"] = -0.01
        warnings = validate_config(cfg)
        assert any("stamp_duty" in w for w in warnings)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

class TestScan:
    def test_lookback_days_too_small(self):
        cfg = _base_config()
        cfg["scan"]["lookback_days"] = 5
        warnings = validate_config(cfg)
        assert any("lookback_days" in w for w in warnings)


# ---------------------------------------------------------------------------
# Valuation thresholds
# ---------------------------------------------------------------------------

class TestValuation:
    def test_double_low_price_max_zero(self):
        cfg = _base_config()
        cfg["valuation"]["double_low_price_max"] = 0
        warnings = validate_config(cfg)
        assert any("double_low_price_max" in w for w in warnings)

    def test_double_low_price_max_negative(self):
        cfg = _base_config()
        cfg["valuation"]["double_low_price_max"] = -10
        warnings = validate_config(cfg)
        assert any("double_low_price_max" in w for w in warnings)

    def test_double_low_premium_max_zero(self):
        cfg = _base_config()
        cfg["valuation"]["double_low_premium_max"] = 0
        warnings = validate_config(cfg)
        assert any("double_low_premium_max" in w for w in warnings)

    def test_ytm_threshold_too_high(self):
        cfg = _base_config()
        cfg["valuation"]["ytm_threshold"] = 25.0
        warnings = validate_config(cfg)
        assert any("ytm_threshold" in w for w in warnings)

    def test_ytm_threshold_negative(self):
        cfg = _base_config()
        cfg["valuation"]["ytm_threshold"] = -1.0
        warnings = validate_config(cfg)
        assert any("ytm_threshold" in w for w in warnings)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

class TestRisk:
    def test_invalid_credit_rating(self):
        cfg = _base_config()
        cfg["risk"]["credit_exclude_rated_below"] = "ZZZ"
        warnings = validate_config(cfg)
        assert any("credit_exclude_rated_below" in w for w in warnings)

    def test_valid_credit_rating(self):
        cfg = _base_config()
        cfg["risk"]["credit_exclude_rated_below"] = "AA+"
        warnings = validate_config(cfg)
        assert not any("credit_exclude_rated_below" in w for w in warnings)


# ---------------------------------------------------------------------------
# Clause
# ---------------------------------------------------------------------------

class TestClause:
    def test_redemption_warn_ge_danger(self):
        cfg = _base_config()
        cfg["clause"]["redemption_warn_ratio"] = 1.30
        cfg["clause"]["redemption_danger_ratio"] = 1.28
        warnings = validate_config(cfg)
        assert any("redemption_warn_ratio" in w for w in warnings)

    def test_putback_consecutive_days_zero(self):
        cfg = _base_config()
        cfg["clause"]["putback_consecutive_days"] = 0
        warnings = validate_config(cfg)
        assert any("putback_consecutive_days" in w for w in warnings)

    def test_putback_consecutive_days_negative(self):
        cfg = _base_config()
        cfg["clause"]["putback_consecutive_days"] = -5
        warnings = validate_config(cfg)
        assert any("putback_consecutive_days" in w for w in warnings)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_risk_free_rate_negative(self):
        cfg = _base_config()
        cfg["options"]["risk_free_rate"] = -0.01
        warnings = validate_config(cfg)
        assert any("risk_free_rate" in w for w in warnings)

    def test_risk_free_rate_too_high(self):
        cfg = _base_config()
        cfg["options"]["risk_free_rate"] = 0.30
        warnings = validate_config(cfg)
        assert any("risk_free_rate" in w for w in warnings)

    def test_iv_low_ge_iv_high(self):
        cfg = _base_config()
        cfg["options"]["iv_low_percentile"] = 80
        cfg["options"]["iv_high_percentile"] = 20
        warnings = validate_config(cfg)
        assert any("iv_low_percentile" in w for w in warnings)

    def test_hv_window_zero(self):
        cfg = _base_config()
        cfg["options"]["hv_window"] = 0
        warnings = validate_config(cfg)
        assert any("hv_window" in w for w in warnings)


# ---------------------------------------------------------------------------
# Stock linkage
# ---------------------------------------------------------------------------

class TestStockLinkage:
    def test_momentum_thresholds_swapped(self):
        cfg = _base_config()
        cfg["stock_linkage"]["momentum_bullish_threshold"] = -3.0
        cfg["stock_linkage"]["momentum_bearish_threshold"] = 3.0
        warnings = validate_config(cfg)
        assert any("momentum_bullish_threshold" in w for w in warnings)


# ---------------------------------------------------------------------------
# Scoring penalties
# ---------------------------------------------------------------------------

class TestScoringPenalties:
    def test_credit_penalty_positive(self):
        cfg = _base_config()
        cfg["scoring"]["credit_penalty"] = 20
        warnings = validate_config(cfg)
        assert any("credit_penalty" in w for w in warnings)

    def test_liquidity_penalty_positive(self):
        cfg = _base_config()
        cfg["scoring"]["liquidity_penalty"] = 10
        warnings = validate_config(cfg)
        assert any("liquidity_penalty" in w for w in warnings)

    def test_dimension_floor_negative(self):
        cfg = _base_config()
        cfg["scoring"]["dimension_floor"] = -0.1
        warnings = validate_config(cfg)
        assert any("dimension_floor" in w for w in warnings)

    def test_dimension_floor_above_one(self):
        cfg = _base_config()
        cfg["scoring"]["dimension_floor"] = 1.5
        warnings = validate_config(cfg)
        assert any("dimension_floor" in w for w in warnings)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class TestLLM:
    def test_max_tokens_zero(self):
        cfg = _base_config()
        cfg["llm"]["max_tokens"] = 0
        warnings = validate_config(cfg)
        assert any("max_tokens" in w for w in warnings)

    def test_timeout_negative(self):
        cfg = _base_config()
        cfg["llm"]["timeout"] = -10
        warnings = validate_config(cfg)
        assert any("timeout" in w for w in warnings)

    def test_top_n_zero(self):
        cfg = _base_config()
        cfg["llm"]["top_n"] = 0
        warnings = validate_config(cfg)
        assert any("top_n" in w for w in warnings)


# ---------------------------------------------------------------------------
# Multiple issues
# ---------------------------------------------------------------------------

class TestMultipleIssues:
    def test_multiple_warnings_returned(self):
        cfg = _base_config()
        cfg["backtest"]["forward_days"] = 0
        cfg["scan"]["lookback_days"] = 5
        cfg["llm"]["max_tokens"] = 0
        warnings = validate_config(cfg)
        assert len(warnings) >= 3


# ---------------------------------------------------------------------------
# RATING_RANKS constant
# ---------------------------------------------------------------------------

class TestRatingRanks:
    def test_rating_ranks_contains_expected(self):
        assert "AAA" in RATING_RANKS
        assert "AA+" in RATING_RANKS
        assert "A" in RATING_RANKS
        assert "C" in RATING_RANKS
        assert len(RATING_RANKS) >= 13
