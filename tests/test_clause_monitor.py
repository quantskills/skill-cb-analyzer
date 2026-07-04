"""Tests for clause monitor detectors."""

import pytest
from core.clause_monitor import ClauseMonitor


class TestRedemption:
    def test_danger(self, sample_config):
        import pandas as pd
        m = ClauseMonitor(sample_config)
        row = pd.Series({"redemption_ratio": 1.35, "cb_price": 145})
        sig = m.detect_redemption(row)
        assert sig.triggered
        assert sig.direction == "bearish"

    def test_warning(self, sample_config):
        import pandas as pd
        m = ClauseMonitor(sample_config)
        row = pd.Series({"redemption_ratio": 1.22, "cb_price": 125})
        sig = m.detect_redemption(row)
        assert sig.triggered
        assert "预警" in sig.label

    def test_safe(self, sample_config):
        import pandas as pd
        m = ClauseMonitor(sample_config)
        row = pd.Series({"redemption_ratio": 1.05, "cb_price": 108})
        sig = m.detect_redemption(row)
        assert not sig.triggered


class TestDownwardRevision:
    def test_high_probability(self, sample_config):
        import pandas as pd
        m = ClauseMonitor(sample_config)
        row = pd.Series({
            "stock_price": 5.0, "conversion_price": 10.0,
            "remaining_years_raw": 1.5, "cb_price": 90,
        })
        sig = m.detect_downward_revision(row)
        assert sig.triggered
        assert sig.direction == "bullish"

    def test_low_probability(self, sample_config):
        import pandas as pd
        m = ClauseMonitor(sample_config)
        row = pd.Series({
            "stock_price": 12.0, "conversion_price": 10.0,
            "remaining_years_raw": 5.0, "cb_price": 130,
        })
        sig = m.detect_downward_revision(row)
        assert not sig.triggered


class TestMaturity:
    def test_near_maturity_discount(self, sample_config):
        import pandas as pd
        m = ClauseMonitor(sample_config)
        row = pd.Series({
            "maturity_date": "2026-08-15", "premium_rate": -3,
            "cb_price": 102,
        })
        sig = m.detect_maturity_alert(row, "20260703")
        assert sig.triggered

    def test_no_maturity_data(self, sample_config):
        import pandas as pd
        m = ClauseMonitor(sample_config)
        row = pd.Series({"maturity_date": ""})
        sig = m.detect_maturity_alert(row)
        assert not sig.triggered
