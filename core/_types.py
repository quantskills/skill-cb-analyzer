"""Shared types used across core modules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def safe_float(value, fallback: float = 0.0) -> float:
    """Convert *value* to float, returning *fallback* if NaN, Inf, or non-numeric.

    Unlike the common ``float(v) or fallback`` idiom, this handles NaN correctly:
    ``float("nan")`` is truthy, so ``float("nan") or 999.0`` returns nan instead
    of 999.0.  Use this helper anywhere a DataFrame cell might be NaN.
    """
    try:
        v = float(value)
    except (ValueError, TypeError):
        return fallback
    if math.isnan(v) or math.isinf(v):
        return fallback
    return v

# Mapping from Chinese/raw column names → English, shared between
# data_fetcher.py and bond_calculator.py for column normalization.
CB_COLUMN_MAP = {
    "代码": "bond_code", "转债代码": "bond_code", "债券代码": "bond_code",
    "转债名称": "bond_name", "债券简称": "bond_name",
    "转债最新价": "cb_price", "现价": "cb_price", "债现价": "cb_price",
    "正股代码": "stock_code",
    "正股名称": "stock_name", "正股简称": "stock_name",
    "转股价": "conversion_price",
    "转股价值": "conversion_value_raw",
    "转股溢价率": "premium_raw",
    "到期日": "maturity_date", "到期时间": "maturity_date",
    "上市时间": "list_date",
    "债券评级": "credit_rating", "信用评级": "credit_rating",
    "剩余年限": "remaining_years_raw",
    "回售触发价": "putback_trigger_price",
    "强赎触发价": "redemption_trigger_price",
    "到期赎回价": "maturity_redemption_price",
    "正股价": "stock_price_raw",
    "成交额": "turnover", "amount": "turnover",
    "volume": "volume",
    "到期税前收益": "ytm_raw",
    "发行规模": "issue_scale",
}


@dataclass
class SignalResult:
    """Output from a single signal detector."""
    key: str
    label: str
    triggered: bool
    strength: float       # 0.0 (none) to 1.0 (strong); can be negative for bearish
    direction: str        # "bullish", "bearish", "neutral"
    summary: str
    detail: dict = field(default_factory=dict)


def neutral_signal(key: str, label: str, summary: str = "",
                   detail: dict | None = None) -> SignalResult:
    """Create a neutral (non-triggered) SignalResult."""
    return SignalResult(
        key=key, label=label, triggered=False,
        strength=0.0, direction="neutral", summary=summary,
        detail=detail or {},
    )


def bullish_signal(key: str, label: str, strength: float, summary: str = "",
                   detail: dict | None = None) -> SignalResult:
    """Create a bullish SignalResult."""
    return SignalResult(
        key=key, label=label, triggered=True,
        strength=min(max(strength, 0.0), 1.0),
        direction="bullish", summary=summary,
        detail=detail or {},
    )


def bearish_signal(key: str, label: str, strength: float, summary: str = "",
                   detail: dict | None = None) -> SignalResult:
    """Create a bearish SignalResult (negative strength)."""
    return SignalResult(
        key=key, label=label, triggered=True,
        strength=-min(max(abs(strength), 0.0), 1.0),
        direction="bearish", summary=summary,
        detail=detail or {},
    )
