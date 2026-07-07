"""Stock linkage signal detectors (Group C): 4 detectors.

C1 - Stock Momentum (正股动量): underlying stock trend strength
C2 - CB-Stock Deviation (转债-正股偏离): CB price divergence from fair value
C3 - Delta Elasticity (Delta弹性): sensitivity of CB price to stock movement
C4 - Stock Pattern (正股技术形态): optional technical pattern overlay
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core._types import SignalResult, bearish_signal, bullish_signal, neutral_signal, safe_float

logger = logging.getLogger(__name__)


class StockLinkageDetector:
    """Detects CB-stock linkage and convergence signals."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        link_cfg = cfg.get("stock_linkage", {})
        self._mom_lookback = int(link_cfg.get("momentum_lookback", 20))
        self._mom_bullish = float(link_cfg.get("momentum_bullish_threshold", 3.0))
        self._mom_bearish = float(link_cfg.get("momentum_bearish_threshold", -3.0))
        self._deviation_threshold = float(link_cfg.get("deviation_threshold", 3.0))
        self._delta_high = float(link_cfg.get("delta_high_threshold", 0.8))

        weights = cfg.get("detector_weights", {})
        self._w_momentum = float(weights.get("stock_momentum", 3))
        self._w_deviation = float(weights.get("cb_stock_deviation", 2))
        self._w_delta = float(weights.get("delta_elasticity", 2))
        self._w_pattern = float(weights.get("stock_pattern", 1))

    # -- C1: Stock Momentum (正股动量) --------------------------------

    def detect_stock_momentum(self, stock_kline: pd.DataFrame,
                               stock_code: str) -> SignalResult:
        """Detect underlying stock momentum from K-line data.

        Computes 20-day return and MA alignment for a single stock.
        """
        if stock_kline.empty:
            return neutral_signal("stock_momentum", "正股动量", "无K线数据")

        # Filter to this stock
        sym_col = "symbol" if "symbol" in stock_kline.columns else None
        if sym_col:
            df = stock_kline[stock_kline[sym_col] == stock_code].copy()
        else:
            df = stock_kline.copy()

        if len(df) < self._mom_lookback:
            return neutral_signal("stock_momentum", "正股动量",
                                  f"K线数据不足（{len(df)}<{self._mom_lookback}天）")

        date_col = next((c for c in ["date", "trade_date"] if c in df.columns), None)
        if date_col:
            df = df.sort_values(date_col)

        close = df["close"].astype(float)

        # 20-day return
        denom = close.iloc[-self._mom_lookback]
        if abs(denom) < 1e-9:
            return neutral_signal("stock_momentum", "正股动量", "K线价格异常（除零）")
        ret_20d = (close.iloc[-1] - denom) / denom * 100
        if not np.isfinite(ret_20d):
            return neutral_signal("stock_momentum", "正股动量", "K线数据异常")

        # MA alignment: MA5 > MA10 > MA20 (bullish) / MA5 < MA10 < MA20 (bearish)
        if len(close) >= 20:
            ma5 = close.iloc[-5:].mean()
            ma10 = close.iloc[-10:].mean()
            ma20 = close.iloc[-20:].mean()
            ma_bullish = ma5 > ma10 > ma20
            ma_bearish = ma5 < ma10 < ma20
        else:
            ma_bullish = False
            ma_bearish = False

        if ret_20d > self._mom_bullish and ma_bullish:
            strength = min(1.0, ret_20d / 20.0)
            return bullish_signal(
                "stock_momentum", "正股动量",
                strength=strength,
                summary=f"正股强势：20日涨幅{ret_20d:.1f}%，均线多头排列",
                detail={"ret_20d": ret_20d, "ma_alignment": "bullish"},
            )

        if ret_20d > self._mom_bullish:
            strength = min(1.0, ret_20d / 20.0) * 0.7
            return bullish_signal(
                "stock_momentum", "正股动量",
                strength=strength,
                summary=f"正股走强：20日涨幅{ret_20d:.1f}%",
                detail={"ret_20d": ret_20d},
            )

        if ret_20d < self._mom_bearish and ma_bearish:
            strength = min(1.0, abs(ret_20d) / 20.0)
            return bearish_signal(
                "stock_momentum", "正股动量",
                strength=strength,
                summary=f"正股弱势：20日跌幅{abs(ret_20d):.1f}%，均线空头排列",
                detail={"ret_20d": ret_20d, "ma_alignment": "bearish"},
            )

        if ret_20d < self._mom_bearish:
            strength = min(1.0, abs(ret_20d) / 20.0) * 0.7
            return bearish_signal(
                "stock_momentum", "正股动量",
                strength=strength,
                summary=f"正股走弱：20日跌幅{abs(ret_20d):.1f}%",
                detail={"ret_20d": ret_20d},
            )

        return neutral_signal(
            "stock_momentum", "正股动量",
            f"正股横盘：20日涨跌{ret_20d:.1f}%",
            detail={"ret_20d": ret_20d},
        )

    # -- C2: CB-Stock Deviation (转债-正股偏离) -----------------------

    def detect_cb_stock_deviation(self, row: pd.Series,
                                   stock_chg: float = 0.0) -> SignalResult:
        """Detect CB price deviation from stock-implied fair value.

        If stock rose but CB barely moved → CB undervalued (bullish catch-up).
        If stock fell but CB held steady → CB overvalued (bearish catch-down).

        Args:
            row: CB data row.
            stock_chg: Stock daily change in %.
        """
        cb_chg = safe_float(row.get("cb_change", row.get("pct_change", 0)), 0.0)
        premium = safe_float(row.get("premium_rate", 999), 999.0)

        if abs(stock_chg) < 0.5:
            return neutral_signal("cb_stock_deviation", "转债-正股偏离",
                                  "正股波动较小", detail={"stock_chg": stock_chg, "cb_chg": cb_chg})

        deviation = cb_chg - stock_chg  # negative = CB lagging stock

        if stock_chg > 1.0 and deviation < -self._deviation_threshold:
            # Stock rose but CB lagged → bullish catch-up opportunity
            strength = min(1.0, abs(deviation) / 10.0)
            return bullish_signal(
                "cb_stock_deviation", "转债偏离（补涨）",
                strength=strength,
                summary=f"正股涨{stock_chg:.1f}%但转债仅涨{cb_chg:.1f}%（滞后{abs(deviation):.1f}%），可能补涨",
                detail={
                    "stock_chg": stock_chg, "cb_chg": cb_chg,
                    "deviation": deviation, "signal": "bullish_lag",
                },
            )

        if stock_chg > 1.0 and deviation > self._deviation_threshold:
            # CB overreacted relative to stock
            return neutral_signal(
                "cb_stock_deviation", "转债偏离（超涨）",
                f"转债涨幅{cb_chg:.1f}%远超正股{stock_chg:.1f}%，溢价扩大",
                detail={"stock_chg": stock_chg, "cb_chg": cb_chg, "deviation": deviation},
            )

        return neutral_signal(
            "cb_stock_deviation", "转债-正股偏离",
            f"转债与正股联动正常（偏离{deviation:.1f}%）",
            detail={"stock_chg": stock_chg, "cb_chg": cb_chg, "deviation": deviation},
        )

    # -- C3: Delta Elasticity (Delta弹性) -----------------------------

    def detect_delta(self, row: pd.Series,
                      stock_kline: pd.DataFrame | None = None) -> SignalResult:
        """Detect CB delta (price sensitivity to stock movement).

        Uses Black-Scholes delta = N(d1) when stock K-line data is available
        for historical volatility estimation.  Falls back to the simplified
        approximation (conversion_value / cb_price) when HV is unavailable.

        High delta (>0.8): CB behaves like equity, strong upside participation.
        Low delta (<0.3): CB behaves like bond, limited upside.
        """
        cb_price = safe_float(row.get("cb_price", 0), 0.0)
        cv = safe_float(row.get("conversion_value", 0), 0.0)

        if cb_price <= 0:
            return neutral_signal("delta", "Delta弹性", "转债价格无效")

        # Try Black-Scholes delta when stock K-line is available
        if stock_kline is not None and not stock_kline.empty:
            try:
                from core.options_pricing import bs_delta, compute_hv_for_bond
                K = safe_float(row.get("conversion_price", 0), 0.0)
                if K > 0 and cv > 0:
                    S = cv * K / 100.0
                    stock_code = str(row.get("stock_code", ""))
                    hv = compute_hv_for_bond(stock_code, stock_kline, window=60)
                    if np.isfinite(hv) and hv > 0:
                        T = max(safe_float(row.get("remaining_years", 0), 0.01), 0.01)
                        delta = bs_delta(S, K, T, 0.025, hv)
                        if delta >= self._delta_high:
                            return bullish_signal(
                                "delta", "Delta弹性",
                                strength=min(1.0, (delta - self._delta_high) / (1 - self._delta_high)),
                                summary=f"BS Delta={delta:.3f}，高股性，跟涨能力强",
                                detail={"delta": round(delta, 4), "method": "black_scholes",
                                        "hv": round(hv, 4)},
                            )
                        if delta < 0.3:
                            return neutral_signal(
                                "delta", "Delta弹性",
                                f"BS Delta={delta:.3f}，偏债性，正股上涨传导有限",
                                detail={"delta": round(delta, 4), "method": "black_scholes",
                                        "hv": round(hv, 4)},
                            )
                        return neutral_signal(
                            "delta", "Delta弹性",
                            f"BS Delta={delta:.3f}，适度股性",
                            detail={"delta": round(delta, 4), "method": "black_scholes",
                                    "hv": round(hv, 4)},
                        )
            except Exception:
                logger.debug("BS delta computation failed for %s, falling back to simplified", stock_code)

        # Simplified fallback
        delta_approx = cv / cb_price

        if delta_approx >= self._delta_high:
            return bullish_signal(
                "delta", "Delta弹性",
                strength=min(1.0, delta_approx),
                summary=f"近似Delta={delta_approx:.2f}：转债跟涨能力强，股性活跃",
                detail={"delta": delta_approx, "conversion_value": cv, "cb_price": cb_price,
                        "method": "simple"},
            )

        if delta_approx < 0.3:
            return neutral_signal(
                "delta", "Delta弹性",
                f"低Delta({delta_approx:.2f})：转债偏债性，正股上涨传导有限",
                detail={"delta": delta_approx, "conversion_value": cv, "cb_price": cb_price,
                        "method": "simple"},
            )

        return neutral_signal(
            "delta", "Delta弹性",
            f"中等Delta({delta_approx:.2f})",
            detail={"delta": delta_approx, "method": "simple"},
        )

    # -- C4: Stock Pattern (正股技术形态) ------------------------------

    def detect_stock_pattern(self, stock_kline: pd.DataFrame,
                              stock_code: str) -> SignalResult:
        """Detect a simplified bullish technical pattern on the underlying stock.

        Checks for: MA5 golden cross MA20 (simplified from Skill 01's 8 patterns).
        This is a lightweight wrapper — full 8-pattern import is optional.
        """
        if stock_kline.empty:
            return neutral_signal("stock_pattern", "正股形态", "无K线数据")

        sym_col = "symbol" if "symbol" in stock_kline.columns else None
        if sym_col:
            df = stock_kline[stock_kline[sym_col] == stock_code].copy()
        else:
            df = stock_kline.copy()

        if len(df) < 25:
            return neutral_signal("stock_pattern", "正股形态",
                                  f"K线不足（{len(df)}<25天）")

        date_col = next((c for c in ["date", "trade_date"] if c in df.columns), None)
        if date_col:
            df = df.sort_values(date_col)

        close = df["close"].astype(float)
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()

        # MA golden cross: MA5 crosses above MA20
        if len(ma5) >= 3 and len(ma20) >= 3:
            cross_up = ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]
            cross_down = ma5.iloc[-1] < ma20.iloc[-1] and ma5.iloc[-2] >= ma20.iloc[-2]

            if cross_up:
                return bullish_signal(
                    "stock_pattern", "正股形态（金叉）",
                    strength=0.6,
                    summary="正股触发均线金叉（MA5↑MA20），技术面偏多",
                    detail={"pattern": "ma_golden_cross", "ma5": ma5.iloc[-1], "ma20": ma20.iloc[-1]},
                )

            if cross_down:
                return bearish_signal(
                    "stock_pattern", "正股形态（死叉）",
                    strength=0.6,
                    summary="正股触发均线死叉（MA5↓MA20），技术面偏空",
                    detail={"pattern": "ma_death_cross", "ma5": ma5.iloc[-1], "ma20": ma20.iloc[-1]},
                )

        # Volume breakout / breakdown
        if len(close) >= 20 and "volume" in df.columns:
            vol = df["volume"].astype(float)
            high_20d = close.iloc[-20:].max()
            low_20d = close.iloc[-20:].min()
            vol_5d_avg = vol.iloc[-6:-1].mean()
            if close.iloc[-1] >= high_20d * 0.98 and vol.iloc[-1] > vol_5d_avg * 1.5:
                return bullish_signal(
                    "stock_pattern", "正股形态（放量突破）",
                    strength=0.5,
                    summary="正股放量创20日新高，量价配合",
                    detail={"pattern": "volume_breakout"},
                )
            if close.iloc[-1] <= low_20d * 1.02 and vol.iloc[-1] > vol_5d_avg * 1.5:
                return bearish_signal(
                    "stock_pattern", "正股形态（放量破位）",
                    strength=0.5,
                    summary="正股放量创20日新低，量价背离",
                    detail={"pattern": "volume_breakdown"},
                )

        return neutral_signal("stock_pattern", "正股形态", "无明显技术形态信号")

    # -- Batch -------------------------------------------------------

    def run_all(
        self,
        cb_df: pd.DataFrame,
        stock_kline: pd.DataFrame,
        cb_stock_map: dict[str, str],
        stock_changes: dict[str, float] | None = None,
    ) -> dict[str, list[SignalResult]]:
        """Run all 4 stock linkage detectors."""
        results = {
            "stock_momentum": [],
            "cb_stock_deviation": [],
            "delta": [],
            "stock_pattern": [],
        }

        for _, row in cb_df.iterrows():
            bond_code = str(row.get("bond_code", row.get("转债代码", "")))
            stock_code = cb_stock_map.get(bond_code, "")

            # C1: Stock momentum
            if stock_code and not stock_kline.empty:
                results["stock_momentum"].append(
                    self.detect_stock_momentum(stock_kline, stock_code)
                )
            else:
                results["stock_momentum"].append(
                    neutral_signal("stock_momentum", "正股动量", "无法匹配正股数据")
                )

            # C2: CB-Stock deviation
            stock_chg = stock_changes.get(stock_code, 0.0) if stock_changes else 0.0
            results["cb_stock_deviation"].append(
                self.detect_cb_stock_deviation(row, stock_chg)
            )

            # C3: Delta (uses BS delta when K-line is available)
            results["delta"].append(self.detect_delta(row, stock_kline))

            # C4: Stock pattern
            if stock_code and not stock_kline.empty:
                results["stock_pattern"].append(
                    self.detect_stock_pattern(stock_kline, stock_code)
                )
            else:
                results["stock_pattern"].append(
                    neutral_signal("stock_pattern", "正股形态", "无法匹配正股数据")
                )

        return results

    def composite_score(
        self, signals: dict[str, SignalResult],
        weight_overrides: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted composite score for stock linkage + volatility group.

        Only counts weights for signals actually present in the dict, so the
        score is backward-compatible when volatility signals are absent.

        Args:
            signals: Dict of signal_name → SignalResult.
            weight_overrides: Optional per-call weight overrides for dynamic
                              IC weighting (config-level keys, e.g.
                              stock_momentum, delta_elasticity, iv_percentile).
        """
        if weight_overrides:
            weights = {
                "stock_momentum": float(weight_overrides.get("stock_momentum", self._w_momentum)),
                "cb_stock_deviation": float(weight_overrides.get("cb_stock_deviation", self._w_deviation)),
                "delta": float(weight_overrides.get("delta_elasticity", self._w_delta)),
                "stock_pattern": float(weight_overrides.get("stock_pattern", self._w_pattern)),
                "iv_percentile": float(weight_overrides.get("iv_percentile", 2)),
                "hv_iv_divergence": float(weight_overrides.get("hv_iv_divergence", 2)),
                "vol_expansion": float(weight_overrides.get("vol_expansion", 2)),
                "bs_delta": float(weight_overrides.get("bs_delta", 2)),
            }
        else:
            weights = {
                "stock_momentum": self._w_momentum,
                "cb_stock_deviation": self._w_deviation,
                "delta": self._w_delta,
                "stock_pattern": self._w_pattern,
                # Volatility signals (from VolatilityDetector) — same weight system
                "iv_percentile": 2,
                "hv_iv_divergence": 2,
                "vol_expansion": 2,
                "bs_delta": 1,             # v1.7: reduced from 2→1 (now Gamma-quality, not Delta duplicate)
            }

        # Only count weights for keys that are present in the signals dict
        present_weights = {k: w for k, w in weights.items() if k in signals}
        total_weight = sum(present_weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            present_weights.get(key, 0) * sig.strength
            for key, sig in signals.items()
            if sig.triggered
        )

        return max(0.0, weighted_sum / total_weight)
