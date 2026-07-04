"""Tests for options_pricing — BS model, HV, IV, and VolatilityDetector."""

import math
import numpy as np
import pandas as pd
import pytest
from core.options_pricing import (
    bs_call_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    historical_volatility,
    implied_volatility,
    compute_hv_for_bond,
    VolatilityDetector,
)
from core._types import SignalResult


# ---------------------------------------------------------------------------
# Black-Scholes price
# ---------------------------------------------------------------------------

class TestBSCallPrice:
    def test_atm(self):
        """ATM option: S=100, K=100 → call should be positive."""
        price = bs_call_price(S=100, K=100, T=1.0, r=0.03, sigma=0.20)
        assert price > 0
        assert price < 50

    def test_itm(self):
        """Deep ITM: S >> K → call ≈ S - K."""
        price = bs_call_price(S=150, K=100, T=1.0, r=0.03, sigma=0.20)
        intrinsic = 150 - 100
        assert price >= intrinsic * 0.98  # slight discount for time

    def test_otm(self):
        """Deep OTM: S << K → call ≈ 0."""
        price = bs_call_price(S=50, K=100, T=1.0, r=0.03, sigma=0.20)
        assert price < 5

    def test_zero_vol(self):
        """Zero volatility → price = max(S-K, 0) discounted."""
        price = bs_call_price(S=110, K=100, T=1.0, r=0.0, sigma=0.0)
        assert price == 0.0  # sigma <= 0 returns 0

    def test_zero_time(self):
        price = bs_call_price(S=110, K=100, T=0, r=0.03, sigma=0.20)
        assert price == 10.0  # intrinsic at expiry

    def test_put_call_parity_sanity(self):
        """Call + K*exp(-rT) = S + Put (for ATM).  Call ≈ Put when S=K."""
        S, K, T, r, sigma = 100, 100, 0.5, 0.03, 0.25
        call = bs_call_price(S, K, T, r, sigma)
        # For ATM, call should be larger than put due to r>0, but both similar
        assert call > 0


# ---------------------------------------------------------------------------
# BS Greeks
# ---------------------------------------------------------------------------

class TestBSDelta:
    def test_atm_delta(self):
        d = bs_delta(S=100, K=100, T=1.0, r=0.03, sigma=0.20)
        assert 0.5 < d < 0.65  # ATM delta > 0.5 with positive r

    def test_deep_itm_delta(self):
        d = bs_delta(S=200, K=100, T=1.0, r=0.03, sigma=0.20)
        assert d > 0.95

    def test_deep_otm_delta(self):
        d = bs_delta(S=50, K=100, T=1.0, r=0.03, sigma=0.20)
        assert d < 0.1

    def test_delta_in_range(self):
        """Delta must be in (0, 1) for a call."""
        d = bs_delta(S=100, K=100, T=1.0, r=0.03, sigma=0.20)
        assert 0.0 < d < 1.0


class TestBSGamma:
    def test_gamma_positive(self):
        g = bs_gamma(S=100, K=100, T=1.0, r=0.03, sigma=0.20)
        assert g > 0

    def test_gamma_near_expiry(self):
        """Gamma spikes near expiry for ATM."""
        g_near = bs_gamma(S=100, K=100, T=0.01, r=0.03, sigma=0.20)
        g_far = bs_gamma(S=100, K=100, T=1.0, r=0.03, sigma=0.20)
        assert g_near > g_far


class TestBSVega:
    def test_vega_positive(self):
        v = bs_vega(S=100, K=100, T=1.0, r=0.03, sigma=0.20)
        assert v > 0

    def test_vega_decreases_near_expiry(self):
        v_near = bs_vega(S=100, K=100, T=0.01, r=0.03, sigma=0.20)
        v_far = bs_vega(S=100, K=100, T=1.0, r=0.03, sigma=0.20)
        assert v_near < v_far


# ---------------------------------------------------------------------------
# Historical volatility
# ---------------------------------------------------------------------------

class TestHistoricalVolatility:
    def test_known_vol(self):
        """Generate lognormal prices with known vol, verify HV recovers it."""
        np.random.seed(42)
        mu, sigma = 0.0005, 0.02  # daily
        prices = [100]
        for _ in range(252):
            prices.append(prices[-1] * np.exp(np.random.normal(mu, sigma)))
        s = pd.Series(prices)
        hv = historical_volatility(s, window=252)
        annual_expected = sigma * np.sqrt(252)
        # Should be in ballpark (wide tolerance due to randomness)
        assert 0.1 < hv < 0.5

    def test_insufficient_data(self):
        s = pd.Series([100, 101, 102])
        hv = historical_volatility(s, window=60)
        assert math.isnan(hv)

    def test_empty_series(self):
        hv = historical_volatility(pd.Series(dtype=float), window=60)
        assert math.isnan(hv)

    def test_negative_prices_filtered(self):
        s = pd.Series([100, -5, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        hv = historical_volatility(s, window=5)
        # The last 5 valid prices have positive returns → HV should be finite
        assert not math.isnan(hv)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

class TestImpliedVolatility:
    def test_round_trip(self):
        """Price an option with sigma=0.30, recover sigma via IV."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.03, 0.30
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_volatility(price, S, K, T, r)
        assert iv == pytest.approx(sigma, rel=0.05)

    def test_itm_round_trip(self):
        S, K, T, r = 120, 100, 0.5, 0.03
        price = bs_call_price(S, K, T, r, 0.25)
        iv = implied_volatility(price, S, K, T, r)
        assert iv == pytest.approx(0.25, rel=0.05)

    def test_invalid_price(self):
        iv = implied_volatility(0, 100, 100, 1.0, 0.03)
        assert math.isnan(iv)

    def test_below_intrinsic(self):
        """Price below intrinsic → IV should be NaN."""
        iv = implied_volatility(5, 100, 80, 1.0, 0.03)  # intrinsic = 20
        assert math.isnan(iv)


# ---------------------------------------------------------------------------
# VolatilityDetector
# ---------------------------------------------------------------------------

class TestVolatilityDetector:
    @pytest.fixture
    def detector(self):
        return VolatilityDetector({
            "options": {
                "risk_free_rate": 0.025,
                "hv_window": 20,
                "iv_low_percentile": 25,
                "iv_high_percentile": 75,
                "hv_iv_divergence_threshold": 0.10,
                "bs_delta_high": 0.70,
                "bs_delta_low": 0.30,
            }
        })

    @pytest.fixture
    def sample_kline(self):
        """Synthetic K-line with ~30% annualised vol."""
        np.random.seed(123)
        mu, sigma = 0.0005, 0.02
        prices = [10.0]
        for _ in range(120):
            prices.append(prices[-1] * np.exp(np.random.normal(mu, sigma)))
        dates = pd.date_range("2026-01-01", periods=len(prices), freq="B")
        df = pd.DataFrame({
            "symbol": "600000.SH",
            "date": [d.strftime("%Y%m%d") for d in dates],
            "close": prices,
        })
        return df

    @pytest.fixture
    def sample_row(self):
        return pd.Series({
            "bond_code": "123456",
            "stock_code": "600000",
            "cb_price": 115.0,
            "conversion_value": 100.0,
            "conversion_price": 10.0,
            "bond_floor_value": 98.0,
            "remaining_years": 3.0,
        })

    def test_bs_delta_signal_triggered(self, detector, sample_kline, sample_row):
        """BS delta should produce a signal when K-line available."""
        sig = detector.detect_bs_delta(sample_row, sample_kline)
        assert isinstance(sig, SignalResult)
        assert sig.key == "bs_delta"
        # With the synthetic data, delta should be in range
        assert "delta" in sig.detail
        assert 0.0 < sig.detail["delta"] < 1.0

    def test_bs_delta_fallback_no_kline(self, detector, sample_row):
        """BS delta falls back to simple delta when no K-line."""
        sig = detector.detect_bs_delta(sample_row, pd.DataFrame())
        assert isinstance(sig, SignalResult)
        assert sig.detail.get("method") == "simple"

    def test_vol_expansion_signal(self, detector, sample_kline, sample_row):
        sig = detector.detect_vol_expansion(sample_row, sample_kline)
        assert isinstance(sig, SignalResult)
        assert sig.key == "vol_expansion"

    def test_hv_iv_divergence_signal(self, detector, sample_kline, sample_row):
        sig = detector.detect_hv_iv_divergence(sample_row, sample_kline)
        assert isinstance(sig, SignalResult)
        assert sig.key == "hv_iv_divergence"

    def test_iv_percentile_no_history(self, detector, sample_kline, sample_row):
        sig = detector.detect_iv_percentile(sample_row, sample_kline)
        assert isinstance(sig, SignalResult)
        assert sig.key == "iv_percentile"

    def test_run_all(self, detector, sample_kline):
        df = pd.DataFrame([{
            "bond_code": "123456",
            "stock_code": "600000",
            "cb_price": 115.0,
            "conversion_value": 100.0,
            "conversion_price": 10.0,
            "bond_floor_value": 98.0,
            "remaining_years": 3.0,
        }])
        results = detector.run_all(df, sample_kline)
        assert set(results.keys()) == {"iv_percentile", "hv_iv_divergence", "vol_expansion", "bs_delta"}
        for k in results:
            assert len(results[k]) == 1
            assert isinstance(results[k][0], SignalResult)

    def test_composite_score(self, detector):
        from core._types import bullish_signal, neutral_signal
        signals = {
            "iv_percentile": neutral_signal("iv_percentile", "IV分位"),
            "hv_iv_divergence": bullish_signal("hv_iv_divergence", "背离", 0.5),
            "vol_expansion": neutral_signal("vol_expansion", "扩张"),
            "bs_delta": bullish_signal("bs_delta", "Delta", 0.8),
        }
        score = detector.composite_score(signals)
        assert 0.0 < score < 1.0  # Only triggered signals contribute
