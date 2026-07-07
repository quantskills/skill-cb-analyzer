"""Configuration validation for skill-cb-analyzer.

Validates config.json values are in reasonable ranges. Non-blocking:
returns warnings rather than raising exceptions, so the pipeline can still
run with imperfect configs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

RATING_RANKS = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BB", "B", "CCC", "CC", "C",
]


def validate_config(config: dict) -> list[str]:
    """Validate config values are in reasonable ranges.

    Args:
        config: Full config dict (from config.json).

    Returns:
        List of warning strings. Empty list means no issues found.
    """
    warnings: list[str] = []

    # --- scoring weights ---
    sc = config.get("scoring", {})
    w_val = float(sc.get("valuation_weight", 0.40))
    w_clause = float(sc.get("clause_weight", 0.30))
    w_link = float(sc.get("linkage_weight", 0.20))
    w_struct = float(sc.get("structure_weight", 0.10))
    w_sum = w_val + w_clause + w_link + w_struct
    if abs(w_sum - 1.0) > 0.05:
        warnings.append(
            f"评分权重之和={w_sum:.2f}，偏离 1.0 超过 5%（当前: "
            f"估值{w_val:.2f} 条款{w_clause:.2f} 联动{w_link:.2f} 结构{w_struct:.2f}）"
        )
    for name, w in [("valuation", w_val), ("clause", w_clause),
                    ("linkage", w_link), ("structure", w_struct)]:
        if w < 0:
            warnings.append(f"评分权重 {name}_weight={w} 为负数")

    # --- detector weights ---
    dw = config.get("detector_weights", {})
    for key, w in dw.items():
        if float(w) < 0:
            warnings.append(f"检测器权重 {key}={w} 为负数")

    # --- backtest ---
    bt = config.get("backtest", {})
    forward_days = int(bt.get("forward_days", 5))
    if forward_days < 1:
        warnings.append(f"backtest.forward_days={forward_days} 必须 >= 1")
    n_quintiles = int(bt.get("n_quintiles", 5))
    if n_quintiles < 2:
        warnings.append(f"backtest.n_quintiles={n_quintiles} 必须 >= 2")

    # --- scan ---
    lookback = int(config.get("scan", {}).get("lookback_days", 120))
    if lookback < 10:
        warnings.append(f"scan.lookback_days={lookback} 过小，建议 >= 10")

    # --- valuation thresholds ---
    val_cfg = config.get("valuation", {})
    dl_price = float(val_cfg.get("double_low_price_max", 120))
    if dl_price <= 0:
        warnings.append(f"valuation.double_low_price_max={dl_price} 必须 > 0")
    dl_premium = float(val_cfg.get("double_low_premium_max", 20))
    if dl_premium <= 0:
        warnings.append(f"valuation.double_low_premium_max={dl_premium} 必须 > 0")
    ytm_th = float(val_cfg.get("ytm_threshold", 3.0))
    if not (0 <= ytm_th / 100 <= 0.20):
        warnings.append(f"valuation.ytm_threshold={ytm_th} 应在 [0, 20] 范围")

    # --- risk ---
    risk_cfg = config.get("risk", {})
    credit_below = str(risk_cfg.get("credit_exclude_rated_below", "A"))
    if credit_below not in RATING_RANKS:
        warnings.append(
            f"risk.credit_exclude_rated_below='{credit_below}' 不在已知评级列表中"
        )

    # --- clause thresholds ---
    clause_cfg = config.get("clause", {})
    warn_r = float(clause_cfg.get("redemption_warn_ratio", 1.20))
    danger_r = float(clause_cfg.get("redemption_danger_ratio", 1.28))
    if warn_r >= danger_r:
        warnings.append(f"clause.redemption_warn_ratio={warn_r} 必须 < redemption_danger_ratio={danger_r}")
    pcb_days = int(clause_cfg.get("putback_consecutive_days", 30))
    if pcb_days <= 0:
        warnings.append(f"clause.putback_consecutive_days={pcb_days} 必须 > 0")

    # --- options thresholds ---
    opt_cfg = config.get("options", {})
    rf = float(opt_cfg.get("risk_free_rate", 0.025))
    if not (0 <= rf <= 0.2):
        warnings.append(f"options.risk_free_rate={rf} 应在 [0, 0.20] 范围")
    iv_low = float(opt_cfg.get("iv_low_percentile", 25))
    iv_high = float(opt_cfg.get("iv_high_percentile", 75))
    if iv_low >= iv_high:
        warnings.append(f"options.iv_low_percentile={iv_low} 必须 < iv_high_percentile={iv_high}")
    hv_win = int(opt_cfg.get("hv_window", 60))
    if hv_win <= 0:
        warnings.append(f"options.hv_window={hv_win} 必须 > 0")

    # --- stock_linkage thresholds ---
    sl_cfg = config.get("stock_linkage", {})
    sl_bull = float(sl_cfg.get("momentum_bullish_threshold", 3.0))
    sl_bear = float(sl_cfg.get("momentum_bearish_threshold", -3.0))
    if sl_bull <= sl_bear:
        warnings.append(
            f"stock_linkage.momentum_bullish_threshold={sl_bull} 必须 > "
            f"momentum_bearish_threshold={sl_bear}"
        )

    # --- scoring penalties ---
    sc_cfg = config.get("scoring", {})
    credit_pen = float(sc_cfg.get("credit_penalty", -20))
    if credit_pen > 0:
        warnings.append(f"scoring.credit_penalty={credit_pen} 应为负值（惩罚项）")
    liq_pen = float(sc_cfg.get("liquidity_penalty", -10))
    if liq_pen > 0:
        warnings.append(f"scoring.liquidity_penalty={liq_pen} 应为负值（惩罚项）")
    dim_floor = float(sc_cfg.get("dimension_floor", 0.30))
    if not (0 <= dim_floor <= 1):
        warnings.append(f"scoring.dimension_floor={dim_floor} 应在 [0, 1] 范围")

    # --- backtest.cost_model ---
    bt = config.get("backtest", {})
    cm = bt.get("cost_model", {})
    for key in ("stamp_duty", "commission", "slippage"):
        val = float(cm.get(key, 0))
        if val < 0:
            warnings.append(f"backtest.cost_model.{key}={val} 必须 >= 0")

    # --- backtest.ic_horizons ---
    ic_horizons = bt.get("ic_horizons", [1, 3, 5, 10, 20])
    if any(h <= 0 for h in ic_horizons):
        warnings.append(f"backtest.ic_horizons 包含非正整数")

    # --- llm ---
    llm_cfg = config.get("llm", {})
    if int(llm_cfg.get("max_tokens", 2048)) <= 0:
        warnings.append("llm.max_tokens 必须 > 0")
    if float(llm_cfg.get("timeout", 120)) <= 0:
        warnings.append("llm.timeout 必须 > 0")
    if int(llm_cfg.get("top_n", 5)) < 1:
        warnings.append("llm.top_n 必须 >= 1")

    if warnings:
        logger.warning("配置校验发现 %d 个问题", len(warnings))
        for w in warnings:
            logger.warning("  - %s", w)

    return warnings
