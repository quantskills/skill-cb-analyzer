"""Convertible bond core calculation engine.

Key metrics computed from raw market data:

- Conversion value (转股价值): (100 / conversion_price) × stock_price
- Premium rate (转股溢价率): (cb_price - conversion_value) / conversion_value × 100%
- Bond floor (纯债价值): discounted present value of future cash flows
- YTM (到期收益率): IRR of remaining cash flows to maturity
- Double-low value (双低值): cb_price + premium_rate × 100
- Redemption trigger ratio (强赎触发比): stock_price / conversion_price
- Putback trigger ratio (回售触发比): stock_price / conversion_price
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core._types import safe_float

logger = logging.getLogger(__name__)


# ── Core conversion metrics ──────────────────────────────────────


def conversion_value(conversion_price: float, stock_price: float) -> float:
    """Compute conversion value (转股价值).

    CV = (100 / conversion_price) × stock_price

    Args:
        conversion_price: 转股价
        stock_price: 正股收盘价

    Returns:
        转股价值; returns NaN if conversion_price <= 0.
    """
    if conversion_price <= 0:
        return float("nan")
    return (100.0 / conversion_price) * stock_price


def premium_rate(cb_price: float, conversion_value: float) -> float:
    """Compute conversion premium rate (转股溢价率) as percentage.

    premium = (cb_price - conversion_value) / conversion_value × 100%

    Args:
        cb_price: 可转债价格
        conversion_value: 转股价值

    Returns:
        溢价率 (%); negative means the CB trades at a discount to conversion value.
    """
    if conversion_value <= 0:
        return float("nan")
    return (cb_price - conversion_value) / conversion_value * 100.0


def double_low_value(cb_price: float, premium: float) -> float:
    """Classic double-low score (双低值).

    double_low = cb_price + premium_rate (premium already in percentage, e.g. 6.5 = 6.5%)

    Lower is better. < 120 is traditionally considered attractive.
    """
    return cb_price + max(premium, 0.0)


# ── Bond floor / YTM calculations ───────────────────────────────


def bond_floor(
    coupon_rates: list[float],
    remaining_years: list[float],
    principal: float = 100.0,
    discount_rate: float = 0.035,
    redemption_price: float | None = None,
) -> float:
    """Compute bond floor (纯债价值) by discounting future cash flows.

    The bond floor is the present value of all future coupon payments plus
    the principal repayment at maturity, discounted at a rate reflecting
    the issuer's credit risk.

    Args:
        coupon_rates: Annual coupon rates for each remaining payment (e.g. [0.005, 0.008, ...]).
        remaining_years: Time to each payment in years (e.g. [0.5, 1.5, 2.5, ...]).
        principal: Face value, typically 100 CNY.
        discount_rate: Annual discount rate (default 3.5% for AA-rated approximation).
        redemption_price: If the bond has a maturity redemption premium (e.g. 108),
                          this overrides principal for the final cash flow.

    Returns:
        Present value (bond floor).
    """
    if not coupon_rates or not remaining_years:
        return principal

    pv = 0.0
    for coupon, t in zip(coupon_rates, remaining_years):
        cf = principal * coupon  # coupon payment
        pv += cf / ((1.0 + discount_rate) ** t)

    # Principal repayment at maturity
    final_t = remaining_years[-1]
    final_principal = redemption_price if redemption_price is not None else principal
    pv += final_principal / ((1.0 + discount_rate) ** final_t)

    return pv


def ytm(
    cb_price: float,
    coupon_rates: list[float],
    remaining_years: list[float],
    principal: float = 100.0,
    redemption_price: float | None = None,
) -> float:
    """Compute yield-to-maturity (到期收益率) by solving for IRR.

    Uses Newton's method to find the discount rate that equates the present
    value of future cash flows with the current CB price.

    Args:
        cb_price: Current CB market price.
        coupon_rates: Annual coupon rates.
        remaining_years: Time to each payment in years.
        principal: Face value.
        redemption_price: Maturity redemption price override.

    Returns:
        YTM as annual rate (e.g. 0.03 = 3%). Returns 0.0 if calculation fails.
    """
    if cb_price <= 0 or not coupon_rates or not remaining_years:
        return 0.0

    final_principal = redemption_price if redemption_price is not None else principal

    def npv(r: float) -> float:
        total = -cb_price
        for coupon, t in zip(coupon_rates, remaining_years):
            total += principal * coupon / ((1.0 + r) ** t)
        total += final_principal / ((1.0 + r) ** remaining_years[-1])
        return total

    def npv_derivative(r: float) -> float:
        total = 0.0
        for coupon, t in zip(coupon_rates, remaining_years):
            total += -t * principal * coupon / ((1.0 + r) ** (t + 1))
        total += -remaining_years[-1] * final_principal / ((1.0 + r) ** (remaining_years[-1] + 1))
        return total

    # Newton-Raphson iteration
    r = 0.03  # initial guess
    for _ in range(100):
        f = npv(r)
        if abs(f) < 1e-8:
            return r
        df = npv_derivative(r)
        if abs(df) < 1e-12:
            break
        r_new = r - f / df
        if r_new < -1.0 or r_new > 5.0:
            break
        if abs(r_new - r) < 1e-8:
            return r_new
        r = r_new

    # Fallback: simple yield approximation
    avg_coupon = np.mean(coupon_rates) if coupon_rates else 0.0
    annual_income = principal * avg_coupon
    return annual_income / cb_price


def estimate_remaining_years(maturity_date_str: str, trade_date_str: str = "") -> float:
    """Estimate remaining years to maturity from date strings.

    Args:
        maturity_date_str: Maturity date (e.g. '2028-06-15' or '20280615').
        trade_date_str: Trade date (default: today).

    Returns:
        Remaining years as float.
    """
    from datetime import datetime, date

    def _parse(d: str) -> date:
        d = d.strip().replace("-", "").replace("/", "")
        if len(d) == 8:
            return date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        return date.today()

    maturity = _parse(maturity_date_str)
    if trade_date_str:
        today = _parse(trade_date_str)
    else:
        today = date.today()

    days = (maturity - today).days
    return max(days, 1) / 365.25


def _parse_coupon_field(row) -> list[float] | None:
    """Try to extract actual coupon rates from a DataFrame row.

    Checks multiple candidate columns that may contain coupon data:
    - ``票面利率`` (face rate) — may be a single float or comma-separated string
    - ``利率说明`` (rate description)

    Returns a list of annual coupon rates (as decimals) or None if no data.
    """
    for col in ["票面利率", "利率说明", "coupon_rate", "actual_coupon_rate"]:
        val = row.get(col)
        if val is None or (isinstance(val, float) and (val != val or val <= 0)):
            continue
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ("nan", "none", ""):
            continue
        # Try comma-separated list: "0.3,0.5,0.8,1.2,1.5,1.8"
        if "," in val_str:
            try:
                rates = [float(x.strip()) / 100.0 if float(x.strip()) > 1
                         else float(x.strip())
                         for x in val_str.split(",")]
                if rates and all(r >= 0 for r in rates):
                    return rates
            except (ValueError, TypeError):
                pass
        # Try single percentage or decimal
        try:
            rate = float(val_str)
            if rate > 1:
                rate = rate / 100.0
            if 0 < rate < 0.5:
                return [rate]
        except (ValueError, TypeError):
            pass
    return None


def _fit_coupon_to_remaining(coupon_schedule: list[float],
                              n_remaining: int) -> list[float]:
    """Fit a coupon schedule to the remaining number of payments.

    Takes the last *n_remaining* entries from the schedule. If the schedule
    is shorter, repeats the last value.
    """
    if len(coupon_schedule) >= n_remaining:
        return coupon_schedule[-n_remaining:]
    # Too few entries: pad with last value
    return coupon_schedule + [coupon_schedule[-1]] * (n_remaining - len(coupon_schedule))


def estimate_coupon_schedule(
    credit_rating: str = "AA",
    total_years: int = 6,
    remaining_years: float | None = None,
) -> list[float]:
    """Estimate annual coupon rates based on credit rating and typical CB structure.

    Chinese CBs use stepped coupons (逐年递增票息). Higher-rated issuers pay lower
    coupons. This provides a rating-based fallback when actual coupon data is
    unavailable from the API.

    Typical 6-year structures:
        AAA:  0.2% → 0.4% → 0.6% → 0.8% → 1.5% → 2.0%
        AA:   0.3% → 0.5% → 0.8% → 1.2% → 1.5% → 1.8%
        A:    0.5% → 0.8% → 1.0% → 1.5% → 2.0% → 2.5%
        Below:0.8% → 1.0% → 1.5% → 2.0% → 2.5% → 3.0%

    Args:
        credit_rating: Bond credit rating (e.g. 'AA', 'A+').
        total_years: Total tenor of the CB (typically 6).
        remaining_years: If set, only return coupons for remaining payments.
                         Each coupon represents one annual payment.

    Returns:
        List of annual coupon rates (as decimals, e.g. 0.005 = 0.5%).
    """
    rating = credit_rating.strip().upper()

    # Tiered coupon schedules for different rating bands
    if rating in ("AAA", "AA+"):
        base_schedule = [0.002, 0.004, 0.006, 0.008, 0.015, 0.020]
    elif rating in ("AA", "AA-"):
        base_schedule = [0.003, 0.005, 0.008, 0.012, 0.015, 0.018]
    elif rating in ("A+", "A"):
        base_schedule = [0.005, 0.008, 0.010, 0.015, 0.020, 0.025]
    else:  # A- and below
        base_schedule = [0.008, 0.010, 0.015, 0.020, 0.025, 0.030]

    # Extend or truncate to match total_years
    if total_years > len(base_schedule):
        # Extend with last coupon value
        schedule = base_schedule + [base_schedule[-1]] * (total_years - len(base_schedule))
    else:
        schedule = base_schedule[:total_years]

    # If remaining_years specified, take the last N coupons
    if remaining_years is not None and remaining_years > 0:
        n_remaining = max(1, int(remaining_years))
        # Take from the end of the schedule (later years = higher coupons)
        start_idx = max(0, len(schedule) - n_remaining)
        schedule = schedule[start_idx:]

    return schedule


def estimate_discount_rate(credit_rating: str = "AA") -> float:
    """Estimate discount rate based on credit rating.

    Approximate yields for Chinese corporate bonds:
      AAA  → 3.0%
      AA+  → 3.5%
      AA   → 4.0%
      AA-  → 5.0%
      A+   → 6.0%
      lower → 8.0%
    """
    mapping = {
        "AAA": 0.030, "AA+": 0.035, "AA": 0.040, "AA-": 0.050,
        "A+": 0.060, "A": 0.070, "A-": 0.080,
    }
    return mapping.get(credit_rating.strip().upper(), 0.080)


# ── Clause / event calculations ─────────────────────────────────


def redemption_trigger_ratio(stock_price: float, conversion_price: float) -> float:
    """Compute redemption trigger ratio (强赎触发比).

    ratio = stock_price / conversion_price

    > 1.30 means the 130% redemption threshold is crossed.
    """
    if conversion_price <= 0:
        return 0.0
    return stock_price / conversion_price


def putback_trigger_ratio(stock_price: float, conversion_price: float) -> float:
    """Compute putback trigger ratio (回售触发比).

    ratio = stock_price / (conversion_price × 0.70)

    < 1.0 means the stock is below 70% of conversion price,
    potentially triggering the putback clause.
    """
    if conversion_price <= 0:
        return float("inf")
    return stock_price / (conversion_price * 0.70)


def theoretical_cb_price(
    conversion_value: float,
    bond_floor: float,
    call_value: float = 0.0,
) -> float:
    """Theoretical CB price: max(conversion_value, bond_floor) + call_option_value.

    The CB price floor is max(equity value, bond value). The call option
    represents the time value of the embedded option.
    """
    return max(conversion_value, bond_floor) + call_value


# ── Batch computation ────────────────────────────────────────────


def compute_cb_metrics(
    cb_df: pd.DataFrame,
    stock_kline: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    """Compute all CB metrics for a universe of convertible bonds.

    Args:
        cb_df: DataFrame from AKShare bond_cb_jsl with columns:
               - 转债代码 / bond_code
               - 转债最新价 / cb_price
               - 转股价 / conversion_price
               - 回售触发价 / putback_price (optional)
               - 强赎触发价 / redemption_price (optional)
               - 到期日 / maturity_date
               - 评级 / credit_rating
        stock_kline: Stock daily K-line from Pandadata.
        trade_date: Target trade date (YYYYMMDD).

    Returns:
        DataFrame with all computed metrics added.
    """
    df = cb_df.copy().reset_index(drop=True)

    # Normalize column names (supports 东方财富 bond_zh_cov + 同花顺 + 集思录)
    from core._types import CB_COLUMN_MAP as col_map

    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Ensure numeric columns
    for col in ["cb_price", "conversion_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Get latest stock close prices from K-line (on or before trade_date) ---
    stock_close_map = {}
    if not stock_kline.empty:
        date_col = next((c for c in ["date", "trade_date"] if c in stock_kline.columns), None)
        if date_col:
            # Filter to trade_date or earlier to avoid look-ahead bias
            kline_filtered = stock_kline
            if trade_date:
                kline_filtered = stock_kline[stock_kline[date_col] <= trade_date]
            if kline_filtered.empty:
                kline_filtered = stock_kline  # fallback if all data is after trade_date
            latest = kline_filtered.sort_values(date_col).groupby("symbol").last().reset_index()
            for _, row in latest.iterrows():
                sym = str(row["symbol"])
                close = float(row["close"])
                stock_close_map[sym] = close
                # Also add bare code (strip .SH/.SZ suffix)
                if "." in sym:
                    bare = sym.split(".")[0]
                    stock_close_map[bare] = close

    # --- Compute conversion value ---
    if "conversion_price" in df.columns and "cb_price" in df.columns:
        # Build lookup: bond index → stock close price
        def _get_stock_price(idx: int, stock_code: str) -> float:
            """Resolve stock close price; try bare code, .SH, .SZ suffixes."""
            if not stock_code:
                return float("nan")
            sc = str(stock_code).strip().zfill(6)
            # Try cached map first
            sp = stock_close_map.get(sc, float("nan"))
            if not pd.isna(sp):
                return float(sp)
            # Try with exchange suffix
            for sfx in (".SH", ".SZ", ".BJ"):
                sp = stock_close_map.get(sc + sfx, float("nan"))
                if not pd.isna(sp):
                    return float(sp)
            return float("nan")

        cv_values = []
        for idx, row in df.iterrows():
            cp = float(row.get("conversion_price", 0)) or float("nan")
            sc = str(row.get("stock_code", ""))
            sp = _get_stock_price(idx, sc)
            cv_values.append(conversion_value(cp, sp))
        df["conversion_value"] = cv_values

        # Premium rate
        df["premium_rate"] = df.apply(
            lambda r: premium_rate(
                safe_float(r.get("cb_price", 0), 0.0),
                safe_float(r.get("conversion_value", 0), 0.0),
            ),
            axis=1,
        )

        # Double-low value
        df["double_low"] = df.apply(
            lambda r: double_low_value(
                safe_float(r.get("cb_price", 0), 0.0),
                safe_float(r.get("premium_rate", 0), 0.0),
            ),
            axis=1,
        )

        # Redemption & putback ratios (using same stock price lookup)
        ratios = []
        putback_ratios = []
        for idx, row in df.iterrows():
            cp = safe_float(row.get("conversion_price", 0), 0.0)
            if cp <= 0 or pd.isna(row.get("conversion_price", 0)):
                cp = float("nan")
            sc = str(row.get("stock_code", ""))
            sp = _get_stock_price(idx, sc)
            sp = sp if not pd.isna(sp) else 0.0
            ratios.append(redemption_trigger_ratio(sp, cp))
            putback_ratios.append(putback_trigger_ratio(sp, cp))
        df["redemption_ratio"] = ratios
        df["putback_ratio"] = putback_ratios

    # --- YTM estimation ---
    df["ytm"] = 0.0
    df["bond_floor_value"] = 100.0

    if "maturity_date" in df.columns and "cb_price" in df.columns:
        for idx, row in df.iterrows():
            try:
                maturity_str = str(row.get("maturity_date", ""))
                if not maturity_str or maturity_str in ("nan", "None", ""):
                    continue

                remaining = estimate_remaining_years(maturity_str, trade_date)
                if remaining <= 0:
                    continue

                cb_p = safe_float(row.get("cb_price", 0), 0.0)
                rating = str(row.get("credit_rating", "AA"))
                disc_rate = estimate_discount_rate(rating)

                # Use actual coupon rates when available, otherwise estimate
                actual_coupon = _parse_coupon_field(row)
                n_payments = max(int(remaining), 1)
                years = [(i + 1) for i in range(n_payments)]
                if actual_coupon is not None:
                    coupons = _fit_coupon_to_remaining(actual_coupon, n_payments)
                else:
                    coupons = estimate_coupon_schedule(rating, total_years=6,
                                                       remaining_years=remaining)

                # Maturity redemption price: try 到期赎回价 field first
                redemption_price = safe_float(
                    row.get("maturity_redemption_price",
                    row.get("到期赎回价",
                    row.get("赎回价", 0))), 0.0)
                if redemption_price <= 0:
                    redemption_price = None

                df.at[idx, "bond_floor_value"] = bond_floor(
                    coupons, [float(y) for y in years], 100.0, disc_rate, redemption_price,
                )
                df.at[idx, "ytm"] = ytm(
                    cb_p, coupons, [float(y) for y in years], 100.0, redemption_price,
                )
            except Exception:
                continue

    return df
