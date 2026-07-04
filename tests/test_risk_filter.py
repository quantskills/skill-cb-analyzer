"""Tests for risk filter detectors."""

import pytest
from core.risk_filter import RiskFilter


class TestVolume:
    def test_active(self, sample_config):
        import pandas as pd
        f = RiskFilter(sample_config)
        row = pd.Series({"turnover": 10000})
        sig = f.detect_volume(row)
        assert sig.triggered
        assert sig.direction == "bullish"

    def test_low(self, sample_config):
        import pandas as pd
        f = RiskFilter(sample_config)
        row = pd.Series({"turnover": 50})
        sig = f.detect_volume(row)
        assert sig.triggered
        assert sig.direction == "bearish"


class TestCreditRisk:
    def test_normal(self, sample_config):
        import pandas as pd
        f = RiskFilter(sample_config)
        row = pd.Series({"credit_rating": "AA", "cb_price": 110, "stock_code": "600001.SH"})
        sig = f.detect_credit_risk(row)
        assert not sig.triggered

    def test_low_rating(self, sample_config):
        import pandas as pd
        f = RiskFilter(sample_config)
        row = pd.Series({"credit_rating": "BBB", "cb_price": 90, "stock_code": "000001.SZ"})
        sig = f.detect_credit_risk(row)
        assert sig.triggered
        assert sig.direction == "bearish"


class TestRedemptionExclusion:
    def test_not_announced(self, sample_config):
        import pandas as pd
        f = RiskFilter(sample_config)
        row = pd.Series({"status": "", "redemption_ratio": 1.15, "cb_price": 118})
        assert not f.is_redemption_announced(row)

    def test_announced(self, sample_config):
        import pandas as pd
        f = RiskFilter(sample_config)
        row = pd.Series({"status": "已公告赎回", "redemption_ratio": 1.35, "cb_price": 145})
        assert f.is_redemption_announced(row)
