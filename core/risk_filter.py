"""Market structure & risk detectors (Group D): 5 detectors.

D1 - Volume Activity (成交量活跃度): daily turnover check
D2 - Balance Trend (余额趋势): CB outstanding balance change
D3 - Credit Risk (信用风险): ST stock / rating downgrade / delisting risk
D4 - Redemption Excluded (强赎已公告): announced redemption → exclude
D5 - Liquidity Risk (流动性风险): insufficient daily turnover
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from core._types import SignalResult, bearish_signal, bullish_signal, neutral_signal, safe_float

logger = logging.getLogger(__name__)


class RiskFilter:
    """Detects market structure and risk signals for convertible bonds."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        risk_cfg = cfg.get("risk", {})
        self._min_turnover = float(risk_cfg.get("min_daily_turnover", 100))
        self._exclude_st = bool(risk_cfg.get("credit_exclude_st", True))
        self._exclude_rated_below = str(risk_cfg.get("credit_exclude_rated_below", "A"))
        self._exclude_redemption = bool(risk_cfg.get("redemption_announced_exclude", True))

        weights = cfg.get("detector_weights", {})
        self._w_volume = float(weights.get("volume_active", 2))
        self._w_balance = float(weights.get("balance_trend", 1))

        # Penalties (from scoring config)
        scoring = cfg.get("scoring", {})
        self._credit_penalty = float(scoring.get("credit_penalty", -20))
        self._liquidity_penalty = float(scoring.get("liquidity_penalty", -10))

    @property
    def exclude_st(self) -> bool:
        """Whether ST/delisted stocks are excluded."""
        return self._exclude_st

    # -- D1: Volume Activity (成交量活跃度) ---------------------------

    def detect_volume(self, row: pd.Series) -> SignalResult:
        """Check if CB has sufficient daily trading volume.

        Args:
            row: CB data row. Expected to have 'turnover' or '成交额' in 万元.
        """
        turnover = safe_float(row.get("turnover", row.get("成交额", 0)), 0.0)

        if turnover <= 0:
            return neutral_signal("volume", "成交量", "成交额数据不可用",
                                  detail={"turnover_wanyuan": 0})

        if turnover >= 5000:
            return bullish_signal(
                "volume", "成交量活跃",
                strength=min(1.0, turnover / 20000),
                summary=f"成交活跃：日成交额{turnover:.0f}万元",
                detail={"turnover_wanyuan": turnover},
            )

        if turnover >= self._min_turnover:
            return neutral_signal(
                "volume", "成交量",
                f"成交一般：日成交额{turnover:.0f}万元",
                detail={"turnover_wanyuan": turnover},
            )

        return bearish_signal(
            "volume", "成交量不足",
            strength=0.5,
            summary=f"成交清淡：日成交额{turnover:.0f}万元（<{self._min_turnover}万）",
            detail={"turnover_wanyuan": turnover, "threshold": self._min_turnover},
        )

    # -- D2: Balance Trend (余额趋势) ---------------------------------

    def detect_balance_trend(self, row: pd.Series,
                              previous_balance: float | None = None) -> SignalResult:
        """Detect CB outstanding balance trend.

        When CB balance declines (due to conversion), the overhang on the
        underlying stock decreases — bullish for the stock and CB.

        Args:
            row: Current CB data.
            previous_balance: Outstanding balance from a previous period (万元).
        """
        current_bal = float(row.get("outstanding_balance",
                            row.get("issue_scale",
                            row.get("余额", 0)))) or 0.0

        if previous_balance is None or previous_balance <= 0 or current_bal <= 0:
            return neutral_signal("balance_trend", "余额趋势", "余额数据不足")

        change_pct = (current_bal - previous_balance) / previous_balance * 100

        if change_pct < -5:
            return bullish_signal(
                "balance_trend", "余额下降",
                strength=min(1.0, abs(change_pct) / 20.0),
                summary=f"转债余额减少{abs(change_pct):.1f}%，转股持续推进",
                detail={"current_balance": current_bal, "previous_balance": previous_balance,
                        "change_pct": change_pct},
            )

        if change_pct > 10:
            return neutral_signal(
                "balance_trend", "余额上升",
                f"转债余额增加{change_pct:.1f}%（可能新发或回售少）",
                detail={"change_pct": change_pct},
            )

        return neutral_signal(
            "balance_trend", "余额趋势",
            f"余额变化{change_pct:.1f}%",
            detail={"change_pct": change_pct},
        )

    # -- D3: Credit Risk (信用风险) -----------------------------------

    def detect_credit_risk(self, row: pd.Series,
                            stock_info: pd.DataFrame | None = None) -> SignalResult:
        """Check for credit risk: ST stock / low rating / delisting risk.

        Returns a penalty signal (bearish/neutral) — does NOT add positive score.
        """
        credit_rating = str(row.get("credit_rating", row.get("债券评级", row.get("信用评级", "AA")))).strip().upper()
        stock_code = str(row.get("stock_code", row.get("正股代码", ""))).strip()
        cb_price = safe_float(row.get("cb_price", 0), 0.0)

        risks = []
        severity = 0.0

        # Check 1: Credit rating below threshold
        if credit_rating and credit_rating != "NAN":
            rating_rank = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BB", "B", "CCC", "CC", "C"]
            try:
                rating_idx = rating_rank.index(credit_rating)
                threshold_idx = rating_rank.index(self._exclude_rated_below) if self._exclude_rated_below in rating_rank else 5
                if rating_idx > threshold_idx:
                    risks.append(f"评级{credit_rating}偏低")
                    severity += 0.3
            except ValueError:
                pass

        # Check 2: ST / delisted stock (use list_status field, fallback to name)
        if stock_code and self._exclude_st:
            if stock_info is not None and not stock_info.empty:
                sym_col = "symbol" if "symbol" in stock_info.columns else None
                name_col = "name" if "name" in stock_info.columns else None
                status_col = "list_status" if "list_status" in stock_info.columns else None
                if sym_col:
                    # Normalize: strip .SH/.SZ/.BJ suffix for matching (CB data may omit it)
                    stock_code_clean = stock_code.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
                    si_syms = stock_info[sym_col].astype(str)
                    si_clean = si_syms.str.replace(".SH", "", regex=False).str.replace(".SZ", "", regex=False).str.replace(".BJ", "", regex=False)
                    matches = stock_info[si_clean == stock_code_clean]
                    if not matches.empty:
                        si_row = matches.iloc[0]
                        # Check list_status first (more reliable than name matching)
                        if status_col:
                            status = str(si_row.get(status_col, "")).upper()
                            if status in ("ST", "*ST", "PT", "DELISTED", "退市"):
                                risks.append(f"正股状态异常({status})")
                                severity += 0.6
                        # Fallback: check stock name for ST prefix
                        elif name_col:
                            name = str(si_row.get(name_col, ""))
                            if "ST" in name.upper() or "*ST" in name.upper():
                                risks.append("正股为ST股")
                                severity += 0.6

        # Check 3: Price near delisting zone (< 2 CNY)
        if cb_price < 95:
            risks.append(f"转债价格偏低({cb_price:.2f})，可能存在信用担忧")
            severity += 0.1

        if not risks:
            return neutral_signal("credit_risk", "信用风险",
                                  "无明显信用风险信号",
                                  detail={"rating": credit_rating, "risks": []})

        return bearish_signal(
            "credit_risk", "信用风险",
            strength=min(severity, 1.0),
            summary="信用风险告警：" + "；".join(risks),
            detail={"rating": credit_rating, "risks": risks, "severity": severity},
        )

    # -- D4: Redemption Announced (强赎已公告排除) --------------------

    def is_redemption_announced(self, row: pd.Series) -> bool:
        """Check if the CB has an active redemption announcement.

        Bonds with announced redemption should be EXCLUDED from the selection
        because:
        - They will be called at ~103 CNY within days
        - Any premium over the redemption price will be lost
        - Trading typically halts soon after announcement

        Returns:
            True if the bond should be excluded.
        """
        # Check for explicit redemption status keywords
        status = str(row.get("status", row.get("状态", "")))
        redemption_tag = str(row.get("redemption_tag", row.get("强赎标志", "")))

        exclude_keywords = ["强赎", "已公告赎回", "赎回登记日", "最后交易日", "redeemed", "called"]

        for kw in exclude_keywords:
            if kw in status or kw in redemption_tag:
                return True

        return False

    # -- D5: Liquidity Risk (流动性风险) ------------------------------

    def detect_liquidity_risk(self, row: pd.Series) -> SignalResult:
        """Check for severe liquidity risk.

        CBs with extremely low turnover may be hard to exit.
        """
        turnover = safe_float(row.get("turnover", row.get("成交额", 0)), 0.0)

        if turnover <= 0:
            return neutral_signal("liquidity", "流动性", "成交额数据不可用")

        if turnover < self._min_turnover:
            severity = max(0.3, (self._min_turnover - turnover) / self._min_turnover)
            return bearish_signal(
                "liquidity", "流动性风险",
                strength=severity,
                summary=f"流动性不足：日成交额仅{turnover:.0f}万元，可能难以出清",
                detail={"turnover_wanyuan": turnover, "threshold": self._min_turnover},
            )

        return neutral_signal(
            "liquidity", "流动性",
            f"流动性正常（日成交{turnover:.0f}万元）",
            detail={"turnover_wanyuan": turnover},
        )

    # -- Batch -------------------------------------------------------

    def run_all(
        self,
        cb_df: pd.DataFrame,
        stock_info: pd.DataFrame | None = None,
        previous_balances: dict[str, float] | None = None,
    ) -> dict[str, list[SignalResult]]:
        """Run all 5 risk/structure detectors.

        Returns:
            dict with 'volume', 'balance_trend', 'credit_risk', 'liquidity' keys.
        """
        results = {
            "volume": [],
            "balance_trend": [],
            "credit_risk": [],
            "liquidity": [],
        }

        for _, row in cb_df.iterrows():
            results["volume"].append(self.detect_volume(row))

            bond_code = str(row.get("bond_code", row.get("转债代码", "")))
            prev_bal = previous_balances.get(bond_code) if previous_balances else None
            results["balance_trend"].append(self.detect_balance_trend(row, prev_bal))

            results["credit_risk"].append(self.detect_credit_risk(row, stock_info))
            results["liquidity"].append(self.detect_liquidity_risk(row))

        return results

    def composite_score(
        self, signals: dict[str, SignalResult],
        weight_overrides: dict[str, float] | None = None,
    ) -> float:
        """Compute composite for structure group.

        Includes bearish signals (e.g. low volume) as negative contributions.
        credit_risk and liquidity are handled separately via risk_penalty().
        Result is clamped to [0, 1].

        Args:
            signals: Dict of signal_name → SignalResult.
            weight_overrides: Optional per-call weight overrides for dynamic
                              IC weighting (keys: volume_active, balance_trend).
        """
        if weight_overrides:
            weights = {
                "volume": float(weight_overrides.get("volume_active", self._w_volume)),
                "balance_trend": float(weight_overrides.get("balance_trend", self._w_balance)),
            }
        else:
            weights = {
                "volume": self._w_volume,
                "balance_trend": self._w_balance,
            }

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for key, w in weights.items():
            sig = signals.get(key)
            if sig is not None and sig.triggered:
                weighted_sum += w * sig.strength  # preserves sign

        return max(0.0, weighted_sum / total_weight)

    def risk_penalty(self, signals: dict[str, SignalResult]) -> float:
        """Compute total risk penalty (negative value).

        Returns:
            Negative float representing total penalty points.
        """
        penalty = 0.0

        if "credit_risk" in signals:
            sig = signals["credit_risk"]
            if sig.triggered and sig.strength < 0:
                penalty += self._credit_penalty * abs(sig.strength)

        if "liquidity" in signals:
            sig = signals["liquidity"]
            if sig.triggered and sig.strength < 0:
                penalty += self._liquidity_penalty * abs(sig.strength)

        return penalty
