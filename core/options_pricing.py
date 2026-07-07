"""Options pricing model for convertible bond analysis.

Provides Black-Scholes European call pricing, historical / implied volatility
estimation, and a :class:`VolatilityDetector` that feeds into the stock-linkage
(C) scoring dimension.

Core formulas
-------------
* d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
* d2 = d1 - sigma * sqrt(T)
* Call = S * N(d1) - K * exp(-rT) * N(d2)
* Delta = N(d1)
* Gamma = N'(d1) / (S * sigma * sqrt(T))
* Vega = S * N'(d1) * sqrt(T)   (in price units per 1% vol change, so / 100)

Uses ``scipy.stats.norm.cdf`` / ``.pdf`` — already available as a pandas
dependency.  No additional installation required.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from core._types import (SignalResult, bullish_signal, bearish_signal,
                          neutral_signal, safe_float)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-Scholes core
# ---------------------------------------------------------------------------

def bs_call_price(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """Black-Scholes European call option price.

    Args:
        S: Spot price of the underlying.
        K: Strike price (conversion price).
        T: Time to expiry in years.
        r: Risk-free rate (continuous, e.g. 0.025 for 2.5%).
        sigma: Annualised volatility (e.g. 0.30 for 30%).

    Returns:
        Call option theoretical price.  Returns 0.0 when inputs are degenerate.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if T <= 0 else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """Black-Scholes delta (N(d1)) — sensitivity to underlying price.

    For a European call, delta is always in (0, 1).
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))


def bs_gamma(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """Black-Scholes gamma — rate of change of delta per unit spot move."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_vega(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """Black-Scholes vega — sensitivity to 1% (0.01) change in volatility."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(S * norm.pdf(d1) * np.sqrt(T) / 100.0)


# ---------------------------------------------------------------------------
# Volatility estimation
# ---------------------------------------------------------------------------

def historical_volatility(
    close_prices: pd.Series,
    window: int = 60,
    trading_days: int = 252,
) -> float:
    """Compute annualised historical volatility from daily log returns.

    Args:
        close_prices: Series of daily closing prices (sorted chronologically).
        window: Look-back window in trading days.
        trading_days: Annualisation factor (252 for equities).

    Returns:
        Annualised volatility (e.g. 0.30 = 30%).  Returns NaN if insufficient data.
    """
    prices = close_prices.dropna()
    if len(prices) < max(window, 5) or prices.iloc[-1] <= 0:
        return float("nan")
    recent = prices.tail(window)
    log_returns = np.log(recent / recent.shift(1)).dropna()
    if len(log_returns) < 2:
        return float("nan")
    daily_std = log_returns.std(ddof=1)
    return float(daily_std * np.sqrt(trading_days))


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """Solve for implied volatility via bisection.

    Finds *sigma* such that ``bs_call_price(S, K, T, r, sigma) == market_price``.

    Args:
        market_price: Observed call option price (CB option value).
        S: Underlying spot price.
        K: Strike.
        T: Time to expiry.
        r: Risk-free rate.
        max_iter: Maximum bisection iterations.
        tol: Convergence tolerance on price difference.

    Returns:
        Implied volatility (annualised).  Returns NaN if the solver fails.
    """
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic:
        return float("nan")

    lo, hi = 0.001, 5.0  # 0.1% to 500% vol
    # Make sure the root is bracketed
    for _ in range(50):
        p_hi = bs_call_price(S, K, T, r, hi)
        if p_hi >= market_price:
            break
        hi *= 1.5
        if hi > 50.0:
            return float("nan")

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p_mid = bs_call_price(S, K, T, r, mid)
        diff = p_mid - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Batch computation helpers
# ---------------------------------------------------------------------------

def _resolve_stock_close(stock_code: str, stock_kline: pd.DataFrame) -> float:
    """Get the latest close price for *stock_code* from K-line data."""
    if stock_kline.empty:
        return float("nan")
    date_col = next((c for c in ["date", "trade_date"] if c in stock_kline.columns), None)
    if date_col is None:
        return float("nan")
    code = str(stock_code).strip()
    if code.endswith((".SH", ".SZ", ".BJ")):
        candidates = (code,)
    else:
        candidates = (code, code + ".SH", code + ".SZ", code + ".BJ")
    for candidate in candidates:
        subset = stock_kline[stock_kline["symbol"].astype(str) == candidate]
        if not subset.empty:
            latest = subset.sort_values(date_col).iloc[-1]
            return safe_float(latest.get("close", 0), float("nan"))
    return float("nan")


def compute_hv_for_bond(
    stock_code: str, stock_kline: pd.DataFrame, window: int = 60,
) -> float:
    """Compute HV for a single bond's underlying stock."""
    if stock_kline.empty:
        return float("nan")
    date_col = next((c for c in ["date", "trade_date"] if c in stock_kline.columns), None)
    if date_col is None:
        return float("nan")
    code = str(stock_code).strip()
    if code.endswith((".SH", ".SZ", ".BJ")):
        candidates = (code,)
    else:
        candidates = (code, code + ".SH", code + ".SZ", code + ".BJ")
    for candidate in candidates:
        subset = stock_kline[stock_kline["symbol"].astype(str) == candidate].sort_values(date_col)
        if not subset.empty:
            return historical_volatility(subset["close"], window=window)
    return float("nan")


# ---------------------------------------------------------------------------
# Volatility Detector (Group C extension — stock-linkage dimension)
# ---------------------------------------------------------------------------

class VolatilityDetector:
    """4 volatility / options-based detectors for the stock-linkage dimension.

    Detectors
    ---------
    * iv_percentile     — Implied volatility percentile vs history
    * hv_iv_divergence  — HV vs IV spread
    * vol_expansion     — Recent HV change (expansion / contraction)
    * bs_delta          — True Black-Scholes delta (replaces simplified cv/cb_price)

    All detectors produce :class:`SignalResult` instances following the
    same protocol as ``ValuationDetector``, ``ClauseMonitor``, etc.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        opt = cfg.get("options", {})
        self._risk_free_rate = float(opt.get("risk_free_rate", 0.025))
        self._hv_window = int(opt.get("hv_window", 60))
        self._iv_pct_low = float(opt.get("iv_low_percentile", 25))
        self._iv_pct_high = float(opt.get("iv_high_percentile", 75))
        self._divergence_threshold = float(opt.get("hv_iv_divergence_threshold", 0.10))
        self._expansion_lookback = int(opt.get("vol_expansion_lookback", 20))
        self._bs_delta_high = float(opt.get("bs_delta_high", 0.70))
        self._bs_delta_low = float(opt.get("bs_delta_low", 0.30))

    # -- individual detectors ---------------------------------------------

    def detect_iv_percentile(
        self,
        row: pd.Series,
        stock_kline: pd.DataFrame,
        iv_history: pd.Series | None = None,
    ) -> SignalResult:
        """A5: Implied volatility percentile vs history.

        Low IV percentile → options are cheap → bullish.
        High IV percentile → options are expensive → bearish.
        """
        key = "iv_percentile"
        cv = safe_float(row.get("conversion_value", 0), 0.0)
        cb_p = safe_float(row.get("cb_price", 0), 0.0)
        stock_code = str(row.get("stock_code", ""))

        # IV needs the option value embedded in the CB price
        remaining = safe_float(row.get("remaining_years", 0), 0.0)
        bond_floor_val = safe_float(row.get("bond_floor_value", 100.0), 100.0)

        # Approximate option value = CB price - bond floor
        option_value = cb_p - bond_floor_val
        K = safe_float(row.get("conversion_price", 0), 0.0)
        if K <= 0 or cv <= 0:
            return neutral_signal(key, "IV分位",
                                  summary="IV不可计算（缺少转股价/转股价值）")
        S = cv * K / 100.0

        iv = implied_volatility(option_value, S, K, remaining, self._risk_free_rate)

        if not np.isfinite(iv) or iv <= 0:
            return neutral_signal(key, "IV分位",
                                  summary="IV不可计算，使用历史波动率替代")

        # Percentile check against history
        if iv_history is not None and len(iv_history.dropna()) >= 10:
            clean = iv_history.dropna()
            percentile = (clean < iv).mean() * 100.0
            if percentile <= self._iv_pct_low:
                return bullish_signal(key, "IV分位", min((self._iv_pct_low - percentile) / self._iv_pct_low, 1.0),
                                      summary=f"IV处于{percentile:.0f}%低分位(IV={iv*100:.0f}%)，期权价格便宜",
                                      detail={"iv": round(iv, 4), "percentile": round(percentile, 1)})
            if percentile >= self._iv_pct_high:
                return bearish_signal(key, "IV分位", min((percentile - self._iv_pct_high) / (100 - self._iv_pct_high), 1.0),
                                      summary=f"IV处于{percentile:.0f}%高分位(IV={iv*100:.0f}%)，期权偏贵",
                                      detail={"iv": round(iv, 4), "percentile": round(percentile, 1)})
            return neutral_signal(key, "IV分位",
                                  summary=f"IV处于{percentile:.0f}%分位(IV={iv*100:.0f}%)，正常",
                                  detail={"iv": round(iv, 4), "percentile": round(percentile, 1)})

        # No history — report IV but no directional signal
        return neutral_signal(key, "IV分位",
                              summary=f"IV={iv*100:.0f}%（历史数据不足，无分位信号）",
                              detail={"iv": round(iv, 4)})

    def detect_hv_iv_divergence(
        self, row: pd.Series, stock_kline: pd.DataFrame,
    ) -> SignalResult:
        """A6: HV vs IV divergence.

        When IV >> HV, the market is pricing in much more future volatility
        than has been realised — the option may be overpriced (bearish for
        CB buyers).  When IV << HV, options are cheap relative to realised vol
        (bullish).
        """
        key = "hv_iv_divergence"
        stock_code = str(row.get("stock_code", ""))
        hv = compute_hv_for_bond(stock_code, stock_kline, window=self._hv_window)
        if not np.isfinite(hv) or hv <= 0:
            return neutral_signal(key, "波动率背离", summary="HV数据不足，跳过")

        cv = safe_float(row.get("conversion_value", 0), 0.0)
        cb_p = safe_float(row.get("cb_price", 0), 0.0)
        remaining = safe_float(row.get("remaining_years", 0), 0.0)
        bond_floor_val = safe_float(row.get("bond_floor_value", 100.0), 100.0)
        K = safe_float(row.get("conversion_price", 0), 0.0)
        if K <= 0 or cv <= 0:
            return neutral_signal(key, "波动率背离",
                                  summary=f"HV={hv*100:.1f}%，缺少转股价/转股价值，无法计算IV",
                                  detail={"hv": round(hv, 4)})
        option_value = cb_p - bond_floor_val
        S = cv * K / 100.0
        iv = implied_volatility(option_value, S, K, remaining, self._risk_free_rate)

        if not np.isfinite(iv) or iv <= 0:
            return neutral_signal(key, "波动率背离",
                                  summary=f"HV={hv*100:.1f}%，IV不可计算",
                                  detail={"hv": round(hv, 4)})

        spread = iv - hv
        detail = {"hv": round(hv, 4), "iv": round(iv, 4), "spread": round(spread, 4)}

        if spread > self._divergence_threshold:
            strength = min((spread - self._divergence_threshold) / self._divergence_threshold, 1.0)
            return bearish_signal(key, "波动率背离", strength,
                                  summary=f"IV({iv*100:.0f}%) > HV({hv*100:.0f}%)，偏离{spread*100:.1f}%，期权偏贵",
                                  detail=detail)
        if spread < -self._divergence_threshold:
            strength = min((-spread - self._divergence_threshold) / self._divergence_threshold, 1.0)
            return bullish_signal(key, "波动率背离", strength,
                                  summary=f"HV({hv*100:.0f}%) > IV({iv*100:.0f}%)，偏离{-spread*100:.1f}%，期权便宜",
                                  detail=detail)
        return neutral_signal(key, "波动率背离",
                              summary=f"IV({iv*100:.0f}%)≈HV({hv*100:.0f}%)，无显著背离",
                              detail=detail)

    def detect_vol_expansion(
        self, row: pd.Series, stock_kline: pd.DataFrame,
    ) -> SignalResult:
        """A7: Volatility expansion / contraction.

        Compares current HV to HV from *expansion_lookback* days ago.
        Rising vol → more opportunity for CB option value (bullish in trend).
        """
        key = "vol_expansion"
        stock_code = str(row.get("stock_code", ""))
        if stock_kline.empty:
            return neutral_signal(key, "波动率扩张", summary="K线数据缺失")

        date_col = next((c for c in ["date", "trade_date"] if c in stock_kline.columns), None)
        if date_col is None:
            return neutral_signal(key, "波动率扩张", summary="K线无日期列")

        code = str(stock_code).strip()
        if code.endswith((".SH", ".SZ", ".BJ")):
            candidates = (code,)
        else:
            candidates = (code, code + ".SH", code + ".SZ", code + ".BJ")
        subset = pd.DataFrame()
        for candidate in candidates:
            sub = stock_kline[stock_kline["symbol"].astype(str) == candidate].sort_values(date_col)
            if not sub.empty:
                subset = sub
                break

        if subset.empty:
            return neutral_signal(key, "波动率扩张", summary=f"无法匹配正股{code}K线")

        # Current HV
        current_hv = historical_volatility(subset["close"], window=self._hv_window)
        if not np.isfinite(current_hv):
            return neutral_signal(key, "波动率扩张", summary="HV数据不足")

        # HV from N days ago
        lookback_pos = max(0, len(subset) - 1 - self._expansion_lookback)
        past_subset = subset.iloc[:lookback_pos + 1]
        past_hv = historical_volatility(past_subset["close"], window=self._hv_window)

        if not np.isfinite(past_hv) or past_hv <= 0:
            return neutral_signal(key, "波动率扩张",
                                  summary=f"HV={current_hv*100:.1f}%（历史对比数据不足）",
                                  detail={"current_hv": round(current_hv, 4)})

        change_pct = (current_hv - past_hv) / past_hv
        detail = {"current_hv": round(current_hv, 4), "past_hv": round(past_hv, 4),
                   "change_pct": round(change_pct, 4)}

        if change_pct > 0.20:
            return bullish_signal(key, "波动率扩张", min(change_pct, 1.0),
                                  summary=f"HV从{past_hv*100:.0f}%升至{current_hv*100:.0f}%(+{change_pct*100:.0f}%)，波动率扩张",
                                  detail=detail)
        if change_pct < -0.20:
            return bearish_signal(key, "波动率扩张", min(-change_pct, 1.0),
                                  summary=f"HV从{past_hv*100:.0f}%降至{current_hv*100:.0f}%({change_pct*100:.0f}%)，波动率收缩",
                                  detail=detail)
        return neutral_signal(key, "波动率扩张",
                              summary=f"HV={current_hv*100:.0f}%，变化{change_pct*100:+.0f}%，稳定",
                              detail=detail)

    def detect_bs_delta(
        self, row: pd.Series, stock_kline: pd.DataFrame,
    ) -> SignalResult:
        """C3-replacement: Delta quality via Gamma (v1.7 — deduplicated).

        **v1.7 change:** This detector now outputs a **Gamma-quality signal**
        instead of duplicating the Delta signal from ``StockLinkageDetector``.
        ``delta`` (C3, stock_linkage.py) is the canonical Delta source for
        linkage dimension; ``bs_delta`` (this detector) assesses Gamma/Vega
        to judge the *quality* of the Delta estimate:

        - High Gamma (> 0.05): CB is near ATM, delta changes rapidly →
          useful for active trading, strong signal quality.
        - Low Gamma (< 0.01): CB is deep ITM/OTM, delta is stable →
          less actionable for short-term trading.

        BS Delta is still computed (for detail output) but the *signal*
        is driven by Gamma magnitude.  This eliminates the double-counting
        described in SKILL.md §波动率与期权.
        """
        key = "bs_delta"
        cv = safe_float(row.get("conversion_value", 0), 0.0)
        cb_p = safe_float(row.get("cb_price", 0), 0.0)

        K = safe_float(row.get("conversion_price", 0), 0.0)
        if K <= 0 or cv <= 0:
            return neutral_signal(key, "Delta质量", summary="缺少转股价/转股价值数据")

        S = cv * K / 100.0
        remaining = safe_float(row.get("remaining_years", 0), 0.0)
        stock_code = str(row.get("stock_code", ""))

        sigma = compute_hv_for_bond(stock_code, stock_kline, window=self._hv_window)
        if not np.isfinite(sigma) or sigma <= 0:
            delta_approx = cv / cb_p if cb_p > 0 else 0.0
            return neutral_signal(key, "Delta质量",
                                  summary=f"近似Delta={delta_approx:.2f}(HV不可用)，Gamma不可估计",
                                  detail={"delta_approx": round(delta_approx, 4), "method": "simple"})

        T = max(remaining, 0.01)
        delta = bs_delta(S, K, T, self._risk_free_rate, sigma)
        gamma = bs_gamma(S, K, T, self._risk_free_rate, sigma)
        vega = bs_vega(S, K, T, self._risk_free_rate, sigma)
        detail = {"delta": round(delta, 4), "gamma": round(gamma, 4),
                   "vega": round(vega, 4), "sigma": round(sigma, 4),
                   "method": "black_scholes"}

        # ── v1.7: Signal driven by Gamma (not Delta) ──
        # Gamma thresholds: > 0.05 = high sensitivity, < 0.01 = low
        gamma_high = 0.05
        gamma_low = 0.01

        if gamma >= gamma_high:
            return bullish_signal(key, "Delta质量",
                                  min((gamma - gamma_high) / gamma_high, 1.0),
                                  summary=f"Gamma={gamma:.4f}(高)，Delta={delta:.3f}，期权敏感度高，适合主动波段交易",
                                  detail=detail)
        if gamma < gamma_low and gamma > 0:
            return neutral_signal(key, "Delta质量",
                                  summary=f"Gamma={gamma:.4f}(低)，Delta={delta:.3f}，期权反应迟钝，短期交易价值有限",
                                  detail=detail)
        if gamma <= 0:
            return neutral_signal(key, "Delta质量",
                                  summary=f"Gamma不可用，Delta={delta:.3f}",
                                  detail=detail)
        return neutral_signal(key, "Delta质量",
                              summary=f"Gamma={gamma:.4f}，Delta={delta:.3f}，期权敏感度适中",
                              detail=detail)

    # -- batch runner ----------------------------------------------------

    def run_all(
        self,
        cb_df: pd.DataFrame,
        stock_kline: pd.DataFrame,
        iv_history: dict[str, pd.Series] | None = None,
    ) -> dict[str, list[SignalResult]]:
        """Run all 4 volatility detectors across the CB universe.

        Args:
            cb_df: CB quote DataFrame (must have bond_code, conversion_value,
                   cb_price, conversion_price, bond_floor_value, stock_code).
            stock_kline: Stock daily K-line DataFrame.
            iv_history: Optional per-bond IV history for percentile calculation.

        Returns:
            Dict keyed by detector key, each value is a list of SignalResult
            aligned with cb_df rows.
        """
        results: dict[str, list[SignalResult]] = {
            "iv_percentile": [],
            "hv_iv_divergence": [],
            "vol_expansion": [],
            "bs_delta": [],
        }
        iv_hist = iv_history or {}

        for _, row in cb_df.iterrows():
            bond = str(row.get("bond_code", ""))
            iv_series = iv_hist.get(bond)

            results["iv_percentile"].append(
                self.detect_iv_percentile(row, stock_kline, iv_series))
            results["hv_iv_divergence"].append(
                self.detect_hv_iv_divergence(row, stock_kline))
            results["vol_expansion"].append(
                self.detect_vol_expansion(row, stock_kline))
            results["bs_delta"].append(
                self.detect_bs_delta(row, stock_kline))

        return results

    def composite_score(self, signals: dict[str, SignalResult],
                        weights: dict[str, int] | None = None) -> float:
        """Compute weighted composite for volatility-group signals.

        Default weights match config.json detector_weights.
        """
        if weights is None:
            weights = {
                "iv_percentile": 2, "hv_iv_divergence": 2,
                "vol_expansion": 2, "bs_delta": 1,   # v1.7: 2→1 (Gamma-quality signal)
            }
        total_w = sum(weights.get(k, 0) for k in signals)
        if total_w == 0:
            return 0.0
        score = 0.0
        for key, sig in signals.items():
            w = weights.get(key, 0)
            if sig.triggered:
                score += w * sig.strength
        return score / total_w
