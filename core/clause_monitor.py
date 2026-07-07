"""Clause/event signal detectors (Group B): 4 detectors.

B1 - Redemption Progress (强赎触发进度): stock price vs redemption trigger
B2 - Downward Revision Probability (下修概率): estimate likelihood of conversion price revision
B3 - Putback Progress (回售触发进度): stock price vs putback trigger
B4 - Maturity Alert (临近到期): remaining term < 1 year
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd

from core._types import SignalResult, bearish_signal, bullish_signal, neutral_signal, safe_float

logger = logging.getLogger(__name__)


class ClauseMonitor:
    """Detects convertible bond clause/event signals."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        clause_cfg = cfg.get("clause", {})
        self._redemption_warn = float(clause_cfg.get("redemption_warn_ratio", 1.20))
        self._redemption_danger = float(clause_cfg.get("redemption_danger_ratio", 1.28))
        self._redemption_consec = int(clause_cfg.get("redemption_consecutive_days", 15))
        self._redemption_total = int(clause_cfg.get("redemption_total_days", 30))
        self._downward_prob_threshold = float(clause_cfg.get("downward_revision_prob_threshold", 0.4))
        self._putback_consec = int(clause_cfg.get("putback_consecutive_days", 30))
        self._maturity_warn_days = int(clause_cfg.get("maturity_warn_days", 365))

        weights = cfg.get("detector_weights", {})
        self._w_redemption = float(weights.get("redemption_progress", 4))
        self._w_downward = float(weights.get("downward_revision", 3))
        self._w_putback = float(weights.get("putback_progress", 2))
        self._w_maturity = float(weights.get("maturity_alert", 1))

    # -- B1: Redemption Progress (强赎触发进度) ---------------------

    def detect_redemption(self, row: pd.Series,
                           bond_code: str = "",
                           history_store=None) -> SignalResult:
        """Detect redemption (强赎) trigger progress.

        Now uses consecutive-day tracking from HistoryStore: checks how many
        of the last N trading days the ratio was above the danger threshold.

        Args:
            row: CB data row with redemption_ratio.
            bond_code: Bond code for history lookup.
            history_store: Optional HistoryStore instance for consecutive-day check.
        """
        ratio = safe_float(row.get("redemption_ratio", 0), 0.0)
        cb_price = safe_float(row.get("cb_price", 0), 0.0)

        if ratio <= 0:
            return neutral_signal("redemption", "强赎进度",
                                  "转股比数据不可用", detail={"ratio": ratio})

        # Consecutive-day check if history is available
        consec_days = 0
        total_window = 0
        if history_store and bond_code:
            consec_days, total_window = history_store.get_consecutive_days(
                bond_code, "redemption_ratio",
                threshold=self._redemption_danger, direction="above",
                window=self._redemption_total,
            )

        consec_note = ""
        if consec_days > 0:
            consec_note = f"（近{total_window}日中{consec_days}日触发）"

        if ratio >= self._redemption_danger:
            strength = min(1.0, (ratio - 1.0) / 0.5)
            # Boost strength if many consecutive days above threshold
            if consec_days >= self._redemption_consec:
                strength = min(1.0, strength + 0.2)
            return bearish_signal(
                "redemption", "强赎高危",
                strength=strength,
                summary=f"强赎高危：正股价/转股价={ratio:.2f}≥{self._redemption_danger}{consec_note}，可能随时公告强赎",
                detail={
                    "ratio": ratio, "level": "danger", "cb_price": cb_price,
                    "consecutive_days": consec_days,
                    "note": "已触发强赎条款，关注公司公告；转债价格>130面临赎回风险",
                },
            )

        if ratio >= self._redemption_warn:
            denom = self._redemption_danger - self._redemption_warn
            if denom <= 0:
                progress = 1.0  # warn == danger → treat as danger level
            else:
                progress = (ratio - self._redemption_warn) / denom
            return bearish_signal(
                "redemption", "强赎预警",
                strength=0.3 + progress * 0.3,
                summary=f"强赎预警：正股价/转股价={ratio:.2f}≥{self._redemption_warn}，正在接近强赎价",
                detail={
                    "ratio": ratio, "level": "warn", "cb_price": cb_price,
                    "note": f"正在接近强赎触发价，已满足{ratio*100:.0f}%/130%",
                },
            )

        return neutral_signal(
            "redemption", "强赎进度",
            f"距强赎较远（正股价/转股价={ratio:.2f}）",
            detail={"ratio": ratio, "level": "safe"},
        )

    # -- B2: Downward Revision Probability (下修概率) ----------------

    def detect_downward_revision(self, row: pd.Series) -> SignalResult:
        """Estimate probability of conversion price downward revision (下修).

        Conditions that increase downward revision probability:
        1. Stock price is well below conversion price (ratio < 0.85)
        2. CB approaching putback period
        3. Major shareholder holds CB (incentive alignment)

        Since we cannot observe major shareholder holdings in real-time,
        we score based on observable conditions.

        Bullish: successful downward revision → conversion value rises → CB price rises.
        """
        stock_price = safe_float(row.get("stock_price_raw", row.get("stock_price", row.get("close", 0))), 0.0)
        conversion_price = safe_float(row.get("conversion_price", 0), 0.0)
        remaining = safe_float(row.get("remaining_years_raw", 5), 5.0)
        cb_price = safe_float(row.get("cb_price", 0), 0.0)

        if stock_price <= 0 or conversion_price <= 0:
            return neutral_signal("downward_revision", "下修概率",
                                  "数据不足", detail={})

        price_ratio = stock_price / conversion_price

        # Scoring factors
        score = 0.0
        reasons = []

        # Factor 1: Price well below conversion price (< 0.80)
        if price_ratio < 0.70:
            score += 0.50
            reasons.append(f"正股价仅为转股价{price_ratio*100:.0f}%，下修压力极大")
        elif price_ratio < 0.80:
            score += 0.35
            reasons.append(f"正股价为转股价{price_ratio*100:.0f}%，有较强下修动机")
        elif price_ratio < 0.85:
            score += 0.15
            reasons.append(f"正股价为转股价{price_ratio*100:.0f}%，有一定下修可能")

        # Factor 2: Approaching putback (within 1 year)
        if remaining < 2 and price_ratio < 0.80:
            score += 0.25
            reasons.append("临近回售期，公司有动力下修以避免回售")

        # Factor 3: CB deeply below par (100)
        if cb_price < 95:
            score += 0.15
            reasons.append("转债价格深度破发，大股东持有转债有下修动机")

        score = min(score, 0.80)  # Cap at 80%

        if score >= self._downward_prob_threshold:
            return bullish_signal(
                "downward_revision", "下修概率",
                strength=score,
                summary=f"下修概率较高（{score*100:.0f}%）：{'；'.join(reasons)}",
                detail={
                    "price_ratio": price_ratio,
                    "probability": score,
                    "reasons": reasons,
                    "remaining_years": remaining,
                },
            )

        if score > 0.15:
            return bullish_signal(
                "downward_revision", "下修概率",
                strength=score * 0.5,
                summary=f"存在一定下修可能（{score*100:.0f}%）：{'；'.join(reasons)}",
                detail={"price_ratio": price_ratio, "probability": score, "reasons": reasons},
            )

        return neutral_signal(
            "downward_revision", "下修概率",
            f"下修概率较低（{score*100:.0f}%），正股价/转股价={price_ratio:.2f}",
            detail={"price_ratio": price_ratio, "probability": score},
        )

    # -- B3: Putback Progress (回售触发进度) -------------------------

    def detect_putback(self, row: pd.Series,
                        bond_code: str = "",
                        history_store=None) -> SignalResult:
        """Detect putback (回售) trigger progress.

        Now uses consecutive-day tracking: checks how many of the last N trading
        days the stock has been below the putback trigger.

        Args:
            row: CB data row with putback_ratio.
            bond_code: Bond code for history lookup.
            history_store: Optional HistoryStore instance for consecutive-day check.
        """
        ratio = safe_float(row.get("putback_ratio", 999), 999.0)
        cb_price = safe_float(row.get("cb_price", 0), 0.0)
        remaining = safe_float(row.get("remaining_years_raw", 5), 5.0)

        if ratio >= 999:
            return neutral_signal("putback", "回售进度",
                                  "回售触发比数据不可用", detail={})

        # Consecutive-day check
        consec_days = 0
        total_window = 0
        if history_store and bond_code:
            consec_days, total_window = history_store.get_consecutive_days(
                bond_code, "putback_ratio",
                threshold=1.0, direction="below",
                window=self._putback_consec,
            )

        consec_note = ""
        if consec_days > 0:
            consec_note = f"（近{total_window}日中{consec_days}日触发）"

        # putback_ratio = stock_price / (conversion_price * 0.70)
        # ratio < 1.0 means stock_price < conversion_price * 0.70, triggering putback
        if ratio < 1.0:
            strength = min(1.0, (1.0 - ratio) / 0.3)
            if consec_days >= self._putback_consec:
                strength = min(1.0, strength + 0.2)
            return bullish_signal(
                "putback", "回售触发",
                strength=strength,
                summary=f"已触发回售条款：正股价仅为转股价的{ratio*0.7*100:.0f}%（< 70%）{consec_note}",
                detail={
                    "ratio": ratio, "cb_price": cb_price,
                    "remaining_years": remaining,
                    "consecutive_days": consec_days,
                    "note": "进入回售期后投资者可要求公司回售",
                },
            )

        if ratio < 1.2:
            progress = (1.2 - ratio) / 0.2
            return bullish_signal(
                "putback", "回售预警",
                strength=progress * 0.5,
                summary=f"正在接近回售触发价（正股价/触发价比={ratio:.2f}）{consec_note}",
                detail={
                    "ratio": ratio, "cb_price": cb_price,
                    "remaining_years": remaining,
                },
            )

        return neutral_signal(
            "putback", "回售进度",
            f"距回售触发较远（比={ratio:.2f}）",
            detail={"ratio": ratio},
        )

    # -- B4: Maturity Alert (临近到期) --------------------------------

    def detect_maturity_alert(self, row: pd.Series, trade_date: str = "") -> SignalResult:
        """Detect bonds nearing maturity.

        Bonds within 1 year of maturity with negative premium rate (discount)
        present a conversion arbitrage opportunity: buy CB → convert to stock →
        sell stock for profit.

        Bonds near maturity with positive premium are risky: time decay
        accelerates.
        """
        maturity_str = str(row.get("maturity_date", ""))
        premium = safe_float(row.get("premium_rate", 999), 999.0)
        cb_price = safe_float(row.get("cb_price", 0), 0.0)

        if not maturity_str or maturity_str in ("nan", "None", ""):
            return neutral_signal("maturity", "临近到期", "到期日数据不可用")

        try:
            maturity_str = maturity_str.strip().replace("-", "").replace("/", "")
            maturity_dt = date(int(maturity_str[:4]), int(maturity_str[4:6]), int(maturity_str[6:8]))
        except (ValueError, IndexError):
            return neutral_signal("maturity", "临近到期", f"到期日格式异常: {maturity_str}")

        if trade_date:
            try:
                today = date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:8]))
            except (ValueError, IndexError):
                today = date.today()
        else:
            today = date.today()

        days_remaining = (maturity_dt - today).days

        if days_remaining <= 0:
            return neutral_signal("maturity", "临近到期", "已到期或今日到期")

        if days_remaining <= 90:
            # Very close: check for discount arbitrage
            if premium < 0:
                return bullish_signal(
                    "maturity", "到期套利",
                    strength=0.9,
                    summary=f"仅剩{days_remaining}天到期且折价{abs(premium):.1f}%，存在转股套利机会",
                    detail={"days_remaining": days_remaining, "premium": premium, "cb_price": cb_price},
                )
            return bearish_signal(
                "maturity", "临近到期风险",
                strength=0.7,
                summary=f"仅剩{days_remaining}天到期且溢价{premium:.1f}%，时间价值加速衰减",
                detail={"days_remaining": days_remaining, "premium": premium, "cb_price": cb_price},
            )

        if days_remaining <= self._maturity_warn_days:
            if premium < -3:
                return bullish_signal(
                    "maturity", "到期折价",
                    strength=0.5,
                    summary=f"距到期{days_remaining}天，折价{abs(premium):.1f}%，关注转股套利",
                    detail={"days_remaining": days_remaining, "premium": premium},
                )
            return neutral_signal(
                "maturity", "临近到期",
                f"距到期{days_remaining}天，溢价率{premium:.1f}%",
                detail={"days_remaining": days_remaining, "premium": premium},
            )

        return neutral_signal(
            "maturity", "临近到期",
            f"距到期{days_remaining}天",
            detail={"days_remaining": days_remaining},
        )

    # -- Batch -------------------------------------------------------

    def run_all(
        self, cb_df: pd.DataFrame, trade_date: str = "",
        history_store=None,
    ) -> dict[str, list[SignalResult]]:
        """Run all 4 clause detectors.

        Args:
            cb_df: CB DataFrame with computed metrics.
            trade_date: Target trade date.
            history_store: Optional HistoryStore for consecutive-day tracking (B1/B3).

        Returns:
            dict with 'redemption', 'downward_revision', 'putback', 'maturity' keys.
        """
        results = {
            "redemption": [],
            "downward_revision": [],
            "putback": [],
            "maturity": [],
        }

        for _, row in cb_df.iterrows():
            bond_code = str(row.get("bond_code", row.get("转债代码", "")))
            results["redemption"].append(
                self.detect_redemption(row, bond_code, history_store)
            )
            results["downward_revision"].append(self.detect_downward_revision(row))
            results["putback"].append(
                self.detect_putback(row, bond_code, history_store)
            )
            results["maturity"].append(self.detect_maturity_alert(row, trade_date))

        return results

    def composite_score(
        self, signals: dict[str, SignalResult],
        weight_overrides: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted composite score for clause group.

        Note: redemption is bearish (negative strength), downward revision and
        putback are bullish (positive strength). The composite accounts for
        the sign of each signal.

        Args:
            signals: Dict of signal_name → SignalResult.
            weight_overrides: Optional per-call weight overrides for dynamic
                              IC weighting (keys: redemption_progress,
                              downward_revision, putback_progress, maturity_alert).
        """
        if weight_overrides:
            weights = {
                "redemption": float(weight_overrides.get("redemption_progress", self._w_redemption)),
                "downward_revision": float(weight_overrides.get("downward_revision", self._w_downward)),
                "putback": float(weight_overrides.get("putback_progress", self._w_putback)),
                "maturity": float(weight_overrides.get("maturity_alert", self._w_maturity)),
            }
        else:
            weights = {
                "redemption": self._w_redemption,
                "downward_revision": self._w_downward,
                "putback": self._w_putback,
                "maturity": self._w_maturity,
            }

        total_weight = 0.0
        weighted_sum = 0.0

        for key, sig in signals.items():
            w = weights.get(key, 0)
            if sig.triggered:
                weighted_sum += w * sig.strength  # preserves sign
            total_weight += w

        if total_weight == 0:
            return 0.0

        # Normalize: [-1, 1] → [0, 1]
        raw = weighted_sum / total_weight
        return max(0.0, raw)
