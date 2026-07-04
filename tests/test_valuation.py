"""Tests for valuation detectors."""

import pytest
from core.valuation import ValuationDetector


class TestDoubleLow:
    def test_triggered(self, sample_config):
        import pandas as pd
        d = ValuationDetector(sample_config)
        row = pd.Series({"cb_price": 110, "premium_rate": 10})
        sig = d.detect_double_low(row)
        assert sig.triggered
        assert sig.direction == "bullish"
        assert sig.strength > 0.25

    def test_not_triggered_high_price(self, sample_config):
        import pandas as pd
        d = ValuationDetector(sample_config)
        row = pd.Series({"cb_price": 130, "premium_rate": 5})
        sig = d.detect_double_low(row)
        assert not sig.triggered

    def test_not_triggered_high_premium(self, sample_config):
        import pandas as pd
        d = ValuationDetector(sample_config)
        row = pd.Series({"cb_price": 110, "premium_rate": 30})
        sig = d.detect_double_low(row)
        assert not sig.triggered

    def test_perfect_double_low(self, sample_config):
        import pandas as pd
        d = ValuationDetector(sample_config)
        row = pd.Series({"cb_price": 95, "premium_rate": 2})
        sig = d.detect_double_low(row)
        assert sig.triggered
        assert sig.strength > 0.5


class TestYTMDefense:
    def test_triggered(self, sample_config):
        import pandas as pd
        d = ValuationDetector(sample_config)
        row = pd.Series({"ytm": 0.05, "remaining_years_raw": 3})
        sig = d.detect_ytm_defense(row)
        assert sig.triggered

    def test_not_triggered(self, sample_config):
        import pandas as pd
        d = ValuationDetector(sample_config)
        row = pd.Series({"ytm": 0.01, "remaining_years_raw": 1})
        sig = d.detect_ytm_defense(row)
        assert not sig.triggered


class TestBondFloor:
    def test_triggered(self, sample_config):
        import pandas as pd
        d = ValuationDetector(sample_config)
        row = pd.Series({"cb_price": 101, "bond_floor_value": 100})
        sig = d.detect_bond_floor(row)
        assert sig.triggered
        assert sig.direction == "bullish"

    def test_not_triggered(self, sample_config):
        import pandas as pd
        d = ValuationDetector(sample_config)
        row = pd.Series({"cb_price": 130, "bond_floor_value": 100})
        sig = d.detect_bond_floor(row)
        assert not sig.triggered
