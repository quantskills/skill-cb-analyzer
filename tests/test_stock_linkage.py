"""Tests for stock linkage detectors."""

import pandas as pd
import pytest
from core.stock_linkage import StockLinkageDetector


def _make_kline(prices: list[float], symbol: str = "600001.SH") -> pd.DataFrame:
    """Build deterministic K-line DataFrame from a list of closing prices."""
    import numpy as np
    records = []
    for i, close in enumerate(prices):
        records.append({
            "symbol": symbol,
            "date": f"202607{str(i+1).zfill(2)}",
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 5000000,
        })
    return pd.DataFrame(records)


class TestStockMomentum:
    def test_bullish_ma_aligned(self, sample_config):
        """+10% return + MA5 > MA10 > MA20 → bullish."""
        d = StockLinkageDetector(sample_config)
        # Build 25 days: steady uptrend so MAs align bullish
        prices = [10.0 + i * 0.05 for i in range(25)]
        kline = _make_kline(prices)
        sig = d.detect_stock_momentum(kline, "600001.SH")
        assert sig.triggered
        assert sig.direction == "bullish"

    def test_bullish_no_ma(self, sample_config):
        """+5% return without MA alignment → weak bullish."""
        d = StockLinkageDetector(sample_config)
        # Sharp jump at end: MA5 > MA10 but maybe not > MA20
        prices = [10.0] * 20 + [10.5]
        kline = _make_kline(prices)
        sig = d.detect_stock_momentum(kline, "600001.SH")
        # 20d return = (10.5-10.0)/10.0*100 = 5%, above 3% threshold
        # MA alignment may or may not hold (mostly flat prices)
        assert sig.triggered
        assert sig.direction == "bullish"

    def test_bearish_ma_aligned(self, sample_config):
        """-10% return + MA5 < MA10 < MA20 → bearish."""
        d = StockLinkageDetector(sample_config)
        prices = [10.0 - i * 0.05 for i in range(25)]
        kline = _make_kline(prices)
        sig = d.detect_stock_momentum(kline, "600001.SH")
        assert sig.triggered
        assert sig.direction == "bearish"

    def test_bearish_no_ma(self, sample_config):
        """-5% return without MA alignment → weak bearish."""
        d = StockLinkageDetector(sample_config)
        prices = [10.0] * 20 + [9.5]
        kline = _make_kline(prices)
        sig = d.detect_stock_momentum(kline, "600001.SH")
        assert sig.triggered
        assert sig.direction == "bearish"

    def test_neutral(self, sample_config):
        """Flat return → neutral."""
        d = StockLinkageDetector(sample_config)
        prices = [10.0] * 25
        kline = _make_kline(prices)
        sig = d.detect_stock_momentum(kline, "600001.SH")
        assert not sig.triggered

    def test_insufficient_data(self, sample_config):
        """Less than lookback days → neutral."""
        d = StockLinkageDetector(sample_config)
        prices = [10.0] * 10
        kline = _make_kline(prices)
        sig = d.detect_stock_momentum(kline, "600001.SH")
        assert not sig.triggered

    def test_empty_kline(self, sample_config):
        """Empty K-line → neutral."""
        d = StockLinkageDetector(sample_config)
        sig = d.detect_stock_momentum(pd.DataFrame(), "600001.SH")
        assert not sig.triggered


class TestCBStockDeviation:
    def test_bullish_lag(self, sample_config):
        """Stock +5% but CB +0.5% → CB lagging, bullish catch-up."""
        import pandas as pd
        d = StockLinkageDetector(sample_config)
        row = pd.Series({"cb_change": 0.5, "pct_change": 0.5, "premium_rate": 15})
        sig = d.detect_cb_stock_deviation(row, stock_chg=5.0)
        assert sig.triggered
        assert sig.direction == "bullish"

    def test_overreaction(self, sample_config):
        """Stock +3%, CB +10% → CB overreacted, neutral."""
        import pandas as pd
        d = StockLinkageDetector(sample_config)
        row = pd.Series({"cb_change": 10.0, "pct_change": 10.0, "premium_rate": 30})
        sig = d.detect_cb_stock_deviation(row, stock_chg=3.0)
        # Overreaction returns neutral, not triggered
        assert sig.direction == "neutral"

    def test_low_volatility(self, sample_config):
        """Stock barely moved → neutral."""
        import pandas as pd
        d = StockLinkageDetector(sample_config)
        row = pd.Series({"cb_change": 0.1, "pct_change": 0.1, "premium_rate": 10})
        sig = d.detect_cb_stock_deviation(row, stock_chg=0.3)
        assert not sig.triggered

    def test_normal_linkage(self, sample_config):
        """Stock +2%, CB +2% → normal, not triggered."""
        import pandas as pd
        d = StockLinkageDetector(sample_config)
        row = pd.Series({"cb_change": 2.0, "pct_change": 2.0, "premium_rate": 10})
        sig = d.detect_cb_stock_deviation(row, stock_chg=2.0)
        assert not sig.triggered


class TestDelta:
    def test_high_delta(self, sample_config):
        """CV=120, price=100 → delta=1.2 → high delta, bullish."""
        import pandas as pd
        d = StockLinkageDetector(sample_config)
        row = pd.Series({"cb_price": 100, "conversion_value": 120})
        sig = d.detect_delta(row)
        assert sig.triggered
        assert sig.direction == "bullish"

    def test_low_delta(self, sample_config):
        """CV=20, price=100 → delta=0.2 → bond-like, neutral."""
        import pandas as pd
        d = StockLinkageDetector(sample_config)
        row = pd.Series({"cb_price": 100, "conversion_value": 20})
        sig = d.detect_delta(row)
        assert not sig.triggered

    def test_invalid_price(self, sample_config):
        """Zero price → neutral."""
        import pandas as pd
        d = StockLinkageDetector(sample_config)
        row = pd.Series({"cb_price": 0, "conversion_value": 100})
        sig = d.detect_delta(row)
        assert not sig.triggered


class TestStockPattern:
    def test_golden_cross(self, sample_config):
        """MA5 crosses above MA20 → golden cross, bullish."""
        d = StockLinkageDetector(sample_config)
        # 23 days at 10.0, dip to 8.0 then spike to 15.0
        # MA5[-2]=9.6 <= MA20[-2]=9.9, MA5[-1]=10.6 > MA20[-1]=10.15 → cross!
        prices = [10.0] * 23 + [8.0, 15.0]
        kline = _make_kline(prices)
        sig = d.detect_stock_pattern(kline, "600001.SH")
        assert sig.triggered
        assert sig.direction == "bullish"

    def test_death_cross(self, sample_config):
        """MA5 crosses below MA20 → death cross, bearish."""
        d = StockLinkageDetector(sample_config)
        # 20 days at 10.0, 2 days at 11.0, then 3-day crash to 9.0, 9.0, 8.0
        # MA5[-2]=10.0 >= MA20[-2]=10.0, MA5[-1]=9.6 < MA20[-1]=9.9 → cross!
        prices = [10.0] * 20 + [11.0] * 2 + [9.0, 9.0, 8.0]
        kline = _make_kline(prices)
        sig = d.detect_stock_pattern(kline, "600001.SH")
        assert sig.triggered
        assert sig.direction == "bearish"

    def test_no_pattern(self, sample_config):
        """No crossover → neutral."""
        d = StockLinkageDetector(sample_config)
        prices = [10.0] * 30
        kline = _make_kline(prices)
        sig = d.detect_stock_pattern(kline, "600001.SH")
        assert not sig.triggered

    def test_insufficient_data(self, sample_config):
        """Less than 25 days → neutral."""
        d = StockLinkageDetector(sample_config)
        prices = [10.0] * 10
        kline = _make_kline(prices)
        sig = d.detect_stock_pattern(kline, "600001.SH")
        assert not sig.triggered


class TestCompositeScore:
    def test_all_bullish(self, sample_config, sample_cb_df):
        """All 4 signals triggered bullish → composite > 0."""
        d = StockLinkageDetector(sample_config)
        signals = {
            "stock_momentum": type("Sig", (), {"triggered": True, "strength": 0.8})(),
            "cb_stock_deviation": type("Sig", (), {"triggered": True, "strength": 0.6})(),
            "delta": type("Sig", (), {"triggered": True, "strength": 0.9})(),
            "stock_pattern": type("Sig", (), {"triggered": True, "strength": 0.6})(),
        }
        score = d.composite_score(signals)
        # weights: momentum=3, deviation=2, delta=2, pattern=1, total=8
        # sum = 3*0.8 + 2*0.6 + 2*0.9 + 1*0.6 = 2.4+1.2+1.8+0.6 = 6.0
        # score = 6.0/8 = 0.75
        assert score > 0.5

    def test_all_neutral(self, sample_config):
        """No signals triggered → composite == 0."""
        d = StockLinkageDetector(sample_config)
        signals = {
            "stock_momentum": type("Sig", (), {"triggered": False, "strength": 0.0})(),
            "cb_stock_deviation": type("Sig", (), {"triggered": False, "strength": 0.0})(),
            "delta": type("Sig", (), {"triggered": False, "strength": 0.0})(),
            "stock_pattern": type("Sig", (), {"triggered": False, "strength": 0.0})(),
        }
        score = d.composite_score(signals)
        assert score == 0.0
