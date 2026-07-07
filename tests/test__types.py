"""Tests for core._types — safe_float and SignalResult helpers."""

import math

import pytest

from core._types import bearish_signal, bullish_signal, neutral_signal, safe_float


# ---------------------------------------------------------------------------
# safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_normal(self):
        assert safe_float(3.14) == 3.14
        assert safe_float("3.14") == 3.14
        assert safe_float(42) == 42.0

    def test_nan_fallback(self):
        assert safe_float(float("nan")) == 0.0
        assert safe_float(float("nan"), fallback=999.0) == 999.0

    def test_inf_fallback(self):
        assert safe_float(float("inf")) == 0.0
        assert safe_float(float("-inf")) == 0.0
        assert safe_float(float("inf"), fallback=-1.0) == -1.0

    def test_negative(self):
        assert safe_float(-5) == -5.0
        assert safe_float("-3.14") == -3.14

    def test_zero(self):
        assert safe_float(0) == 0.0
        assert safe_float(0.0) == 0.0
        assert safe_float("0") == 0.0

    def test_none_fallback(self):
        assert safe_float(None) == 0.0
        assert safe_float(None, fallback=100.0) == 100.0

    def test_string_non_numeric(self):
        assert safe_float("hello") == 0.0
        assert safe_float("") == 0.0
        assert safe_float("hello", fallback=50.0) == 50.0

    def test_bool(self):
        assert safe_float(True) == 1.0
        assert safe_float(False) == 0.0


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

class TestNeutralSignal:
    def test_fields(self):
        sig = neutral_signal("test_key", "Test Label")
        assert sig.key == "test_key"
        assert sig.label == "Test Label"
        assert sig.triggered is False
        assert sig.strength == 0.0
        assert sig.direction == "neutral"
        assert sig.summary == ""
        assert sig.detail == {}

    def test_with_summary_and_detail(self):
        sig = neutral_signal("k", "L", summary="OK", detail={"a": 1})
        assert sig.summary == "OK"
        assert sig.detail == {"a": 1}

    def test_detail_defaults_to_empty_dict(self):
        sig = neutral_signal("k", "L")
        assert sig.detail == {}


class TestBullishSignal:
    def test_normal(self):
        sig = bullish_signal("k", "L", 0.7, summary="bull")
        assert sig.key == "k"
        assert sig.label == "L"
        assert sig.triggered is True
        assert sig.strength == 0.7
        assert sig.direction == "bullish"
        assert sig.summary == "bull"

    def test_capped_at_one(self):
        sig = bullish_signal("k", "L", 5.0)
        assert sig.strength == 1.0

    def test_negative_capped_to_zero(self):
        sig = bullish_signal("k", "L", -0.5)
        assert sig.strength == 0.0

    def test_zero_strength(self):
        sig = bullish_signal("k", "L", 0.0)
        assert sig.strength == 0.0

    def test_detail(self):
        sig = bullish_signal("k", "L", 0.5, detail={"x": 1})
        assert sig.detail == {"x": 1}


class TestBearishSignal:
    def test_normal(self):
        sig = bearish_signal("k", "L", 0.7, summary="bear")
        assert sig.key == "k"
        assert sig.label == "L"
        assert sig.triggered is True
        assert sig.strength == -0.7
        assert sig.direction == "bearish"
        assert sig.summary == "bear"

    def test_capped_at_negative_one(self):
        sig = bearish_signal("k", "L", 5.0)
        assert sig.strength == -1.0

    def test_negative_input_capped(self):
        sig = bearish_signal("k", "L", -0.5)
        assert sig.strength == -0.5

    def test_zero_strength(self):
        sig = bearish_signal("k", "L", 0.0)
        assert sig.strength == 0.0

    def test_detail(self):
        sig = bearish_signal("k", "L", 0.3, detail={"y": 2})
        assert sig.detail == {"y": 2}
