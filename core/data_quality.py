"""Data quality validation for convertible bond daily metrics.

Non-blocking: flags outliers with warnings but does not halt the pipeline.
Extreme values are logged so the user can investigate upstream data issues.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# (field, min, max, label)
DATA_QUALITY_RULES: list[tuple[str, float, float, str]] = [
    ("cb_price", 50, 500, "转债价格"),
    ("premium_rate", -50, 500, "转股溢价率(%)"),
    ("ytm", -0.10, 0.30, "到期收益率(decimal)"),
    ("conversion_value", 0, 5000, "转股价值"),
    ("outstanding_balance", 0, 1e7, "余额(万元)"),
    ("double_low", 50, 500, "双低值"),
    ("redemption_ratio", 0, 10, "强赎触发比"),
    ("putback_ratio", 0, 10, "回售触发比"),
    ("conversion_price", 0.1, 500, "转股价"),
    ("remaining_years_raw", 0, 6, "剩余年限"),
]

# Known valid credit rating values
VALID_RATINGS = frozenset([
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BB", "B", "CCC", "CC", "C",
])


def validate_cb_data(cb_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run sanity checks on computed CB metrics.

    Args:
        cb_df: DataFrame after ``compute_cb_metrics()`` with columns like
               cb_price, premium_rate, ytm, conversion_value, double_low.

    Returns:
        (cb_df, warnings) — cb_df is the original DataFrame with an added
        ``_data_quality_flag`` column (bool). warnings is a list of
        human-readable warning strings.
    """
    df = cb_df.copy()
    df["_data_quality_flag"] = False
    warnings: list[str] = []

    for field, low, high, label in DATA_QUALITY_RULES:
        if field not in df.columns:
            continue
        col = pd.to_numeric(df[field], errors="coerce")
        mask_low = col < low
        mask_high = col > high
        outliers = mask_low | mask_high
        n_out = int(outliers.sum())
        if n_out > 0:
            df.loc[outliers, "_data_quality_flag"] = True
            msg = (
                f"数据质量告警：{label}({field}) 有 {n_out} 条超出合理范围 "
                f"[{low}, {high}]"
            )
            warnings.append(msg)
            logger.warning(msg)

            # Show first few outlier values for diagnosis
            outlier_vals = col[outliers].head(3).tolist()
            logger.info("  outliers sample (%s): %s", field, outlier_vals)

    # Check credit rating validity
    rating_col = next((c for c in ["credit_rating", "债券评级"] if c in df.columns), None)
    if rating_col is not None:
        invalid_mask = ~df[rating_col].isin(VALID_RATINGS) & df[rating_col].notna()
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            df.loc[invalid_mask, "_data_quality_flag"] = True
            msg = f"数据质量告警：信用评级 有 {n_invalid} 条不在已知评级列表中"
            warnings.append(msg)
            logger.warning(msg)

    # Check maturity_date format (YYYYMMDD or similar)
    if "maturity_date" in df.columns:
        date_col = df["maturity_date"].astype(str)
        bad_date_mask = (date_col.str.len() < 6) & date_col.notna()
        n_bad = int(bad_date_mask.sum())
        if n_bad > 0:
            df.loc[bad_date_mask, "_data_quality_flag"] = True
            msg = f"数据质量告警：到期日(maturity_date) 有 {n_bad} 条格式异常"
            warnings.append(msg)
            logger.warning(msg)

    if warnings:
        total_flagged = int(df["_data_quality_flag"].sum())
        logger.warning("数据质量：%d 只转债被标记（共 %d 条告警）",
                       total_flagged, len(warnings))

    return df, warnings
