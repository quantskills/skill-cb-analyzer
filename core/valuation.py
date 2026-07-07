"""Valuation signal detectors (Group A): 4 detectors.

A1 - Double Low (双低信号): price < 120 AND premium < 20%
A2 - YTM Defense (YTM防御): YTM > treasury + threshold
A3 - Bond Floor (纯债溢价率): CB price near bond floor value
A4 - Premium Percentile (溢价率分位): low percentile of historical premium
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from core._types import (SignalResult, bearish_signal, bullish_signal,
                          neutral_signal, safe_float)

logger = logging.getLogger(__name__)

class ValuationDetector:
    """Detects convertible bond valuation signals."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        val_cfg = cfg.get("valuation", {})
        self._price_max = float(val_cfg.get("double_low_price_max", 120))
        self._premium_max = float(val_cfg.get("double_low_premium_max", 20))
        self._ytm_spread = float(val_cfg.get("ytm_treasury_spread", 2.0))
        self._floor_ratio = float(val_cfg.get("bond_floor_ratio", 1.05))
        self._percentile_window = int(val_cfg.get("premium_percentile_window", 60))
        # Treasury benchmarks (configurable, fallback to reasonable defaults)
        self._treasury_1y = float(val_cfg.get("treasury_1y", 1.5))
        self._treasury_3y = float(val_cfg.get("treasury_3y", 2.2))

        # Backtest-calibrated weights (from config.json detector_weights)
        weights = cfg.get("detector_weights", {})
        self._w_double_low = float(weights.get("double_low", 4))
        self._w_ytm = float(weights.get("ytm_defense", 3))
        self._w_floor = float(weights.get("bond_floor", 3))
        self._w_percentile = float(weights.get("premium_percentile", 2))

    # -- A1: Double Low (双低信号) -----------------------------------

    def detect_double_low(self, row: pd.Series) -> SignalResult:
        """Detect classic double-low signal.

        Trigger: cb_price < 120 AND premium_rate < 20%
        Lower price + lower premium = better.
        """
        cb_price = safe_float(row.get("cb_price", 0), 0.0)
        premium = safe_float(row.get("premium_rate", 999), 999.0)

        if cb_price <= 0 or premium >= 999:
            return neutral_signal("double_low", "双低信号", "数据不足，无法判断")

        price_ok = cb_price < self._price_max
        premium_ok = premium < self._premium_max

        if price_ok and premium_ok:
            # Strength: the lower the price&premium, the stronger the signal
            price_score = max(0.0, (self._price_max - cb_price) / self._price_max)
            premium_score = max(0.0, (self._premium_max - premium) / self._premium_max)
            strength = (price_score + premium_score) / 2.0
            return bullish_signal(
                "double_low", "双低信号",
                strength=strength,
                summary=f"双低达标：价格{cb_price:.2f}元，溢价率{premium:.1f}%",
                detail={
                    "cb_price": cb_price,
                    "premium_rate": premium,
                    "double_low_value": cb_price + max(premium, 0.0),
                },
            )

        return neutral_signal(
            "double_low", "双低信号",
            f"不满足双低条件（价格{cb_price:.2f}/{self._price_max}，溢价率{premium:.1f}/{self._premium_max}%）",
            detail={"cb_price": cb_price, "premium_rate": premium},
        )

    # -- A2: YTM Defense (YTM防御) ------------------------------------

    def detect_ytm_defense(self, row: pd.Series) -> SignalResult:
        """Detect YTM-based defensive signal.

        Trigger: YTM > (treasury_rate + spread_threshold)
        High YTM means the bond offers attractive income even without conversion.
        """
        ytm = safe_float(row.get("ytm", 0), 0.0)
        remaining = safe_float(row.get("remaining_years_raw", 3), 3.0)

        ytm_pct = ytm * 100

        # Choose treasury benchmark by remaining term (in percentage)
        treasury = self._treasury_1y if remaining <= 2 else self._treasury_3y
        threshold = treasury + self._ytm_spread

        if ytm_pct <= 0:
            return neutral_signal("ytm_defense", "YTM防御",
                                  "YTM数据不可用", detail={"ytm": ytm_pct})

        if ytm_pct > threshold:
            strength = min(1.0, (ytm_pct - threshold) / 5.0)  # extra 5% = full strength
            return bullish_signal(
                "ytm_defense", "YTM防御信号",
                strength=strength,
                summary=f"高YTM：{ytm_pct:.1f}%（阈值{threshold:.1f}%），债性保护强",
                detail={"ytm": ytm_pct, "threshold": threshold, "remaining_years": remaining},
            )

        return neutral_signal(
            "ytm_defense", "YTM防御信号",
            f"YTM={ytm_pct:.1f}%，未达到防御阈值{threshold:.1f}%",
            detail={"ytm": ytm_pct, "threshold": threshold},
        )

    # -- A3: Bond Floor (纯债溢价率) ---------------------------------

    def detect_bond_floor(self, row: pd.Series) -> SignalResult:
        """Detect bond-floor proximity signal.

        Trigger: cb_price / bond_floor_value < floor_ratio (default 1.05)
        When the CB trades near its bond floor, downside is limited.
        """
        cb_price = safe_float(row.get("cb_price", 0), 0.0)
        floor = safe_float(row.get("bond_floor_value", 100), 100.0)

        if cb_price <= 0 or floor <= 0:
            return neutral_signal("bond_floor", "纯债保护",
                                  "数据不足", detail={})

        ratio = cb_price / floor

        if ratio < self._floor_ratio:
            # Guard: if floor_ratio is 1.0 (degenerate), clamp to avoid div-by-zero
            denom = self._floor_ratio - 1.0
            if denom <= 0:
                strength = 1.0
            else:
                strength = 1.0 - (ratio - 1.0) / denom
            strength = max(0.0, min(1.0, strength))
            return bullish_signal(
                "bond_floor", "纯债保护信号",
                strength=strength,
                summary=f"接近债底：价格/纯债价值={ratio:.3f}（<{self._floor_ratio}），下行保护强",
                detail={"cb_price": cb_price, "bond_floor": floor, "ratio": ratio},
            )

        return neutral_signal(
            "bond_floor", "纯债保护信号",
            f"价格/纯债价值={ratio:.3f}，距债底较远",
            detail={"cb_price": cb_price, "bond_floor": floor, "ratio": ratio},
        )

    # -- A4: Premium Percentile (溢价率分位) --------------------------

    def detect_premium_percentile(
        self, row: pd.Series, premium_history: pd.Series | None = None,
    ) -> SignalResult:
        """Detect premium rate at low percentile vs historical.

        Trigger: current premium rate is in low percentile of historical window.
        Low percentile means the CB is relatively undervalued vs its own history.

        Args:
            row: CB data row.
            premium_history: Historical premium rates for this CB. If None or too
                             short, returns neutral.
        """
        current_premium = safe_float(row.get("premium_rate", 999), 999.0)

        if current_premium >= 999:
            return neutral_signal("premium_percentile", "溢价率分位",
                                  "溢价率数据不可用")

        if premium_history is None or len(premium_history) < 10:
            return neutral_signal("premium_percentile", "溢价率分位",
                                  "历史数据不足（< 10天）",
                                  detail={"current_premium": current_premium})

        clean = premium_history.dropna()
        if len(clean) < 10:
            return neutral_signal("premium_percentile", "溢价率分位",
                                  "有效历史数据不足",
                                  detail={"current_premium": current_premium})

        percentile = (clean < current_premium).mean() * 100.0
        median = clean.median()

        if percentile < 30:
            strength = 1.0 - (percentile / 30.0)
            return bullish_signal(
                "premium_percentile", "溢价率分位信号",
                strength=strength,
                summary=f"溢价率处于历史{percentile:.0f}%分位（中位数{median:.1f}%），相对低估",
                detail={
                    "current_premium": current_premium,
                    "percentile": percentile,
                    "median": median,
                    "window": len(clean),
                },
            )

        if percentile > 80:
            return bearish_signal(
                "premium_percentile", "溢价率分位信号",
                strength=(percentile - 80) / 20.0,
                summary=f"溢价率处于历史{percentile:.0f}%分位（中位数{median:.1f}%），相对高估",
                detail={
                    "current_premium": current_premium,
                    "percentile": percentile,
                    "median": median,
                },
            )

        return neutral_signal(
            "premium_percentile", "溢价率分位信号",
            f"溢价率处于历史{percentile:.0f}%分位，中性",
            detail={"current_premium": current_premium, "percentile": percentile, "median": median},
        )

    # -- Batch -------------------------------------------------------

    def run_all(
        self,
        cb_df: pd.DataFrame,
        premium_history: dict[str, pd.Series] | None = None,
    ) -> dict[str, list[SignalResult]]:
        """Run all 4 valuation detectors against the CB universe.

        Args:
            cb_df: CB quote DataFrame with computed metrics.
            premium_history: Optional dict of bond_code → historical premium series.

        Returns:
            dict with keys 'double_low', 'ytm_defense', 'bond_floor',
            'premium_percentile' → list of SignalResult per bond.
        """
        results = {
            "double_low": [],
            "ytm_defense": [],
            "bond_floor": [],
            "premium_percentile": [],
        }

        for _, row in cb_df.iterrows():
            results["double_low"].append(self.detect_double_low(row))
            results["ytm_defense"].append(self.detect_ytm_defense(row))
            results["bond_floor"].append(self.detect_bond_floor(row))

            bond_code = row.get("bond_code", row.get("转债代码", ""))
            hist = premium_history.get(str(bond_code)) if premium_history else None
            results["premium_percentile"].append(
                self.detect_premium_percentile(row, hist)
            )

        return results

    def composite_score(
        self, signals: dict[str, SignalResult],
        weight_overrides: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted composite score for valuation group.

        Includes bearish signals (e.g. premium_percentile above 80th
        percentile) as negative contributions.  Result is clamped to [0, 1].

        Args:
            signals: Dict of signal_name → SignalResult.
            weight_overrides: Optional per-call weight overrides for dynamic
                              IC weighting (keys: double_low, ytm_defense,
                              bond_floor, premium_percentile).
        """
        if weight_overrides:
            weights = {
                "double_low": float(weight_overrides.get("double_low", self._w_double_low)),
                "ytm_defense": float(weight_overrides.get("ytm_defense", self._w_ytm)),
                "bond_floor": float(weight_overrides.get("bond_floor", self._w_floor)),
                "premium_percentile": float(weight_overrides.get("premium_percentile", self._w_percentile)),
            }
        else:
            weights = {
                "double_low": self._w_double_low,
                "ytm_defense": self._w_ytm,
                "bond_floor": self._w_floor,
                "premium_percentile": self._w_percentile,
            }

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for key, sig in signals.items():
            w = weights.get(key, 0)
            if sig.triggered:
                weighted_sum += w * sig.strength  # preserves sign

        return max(0.0, weighted_sum / total_weight)
