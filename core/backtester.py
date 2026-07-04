"""Backtesting framework for convertible bond scoring strategy.

Validates the 4-dimension scoring model by computing:

1. **IC Analysis** — Spearman rank correlation between composite_score and
   forward N-day returns, aggregated to mean IC / IC IR / win rate.
2. **Stratified Backtest** — Group bonds into quintiles by score each day,
   equal-weight forward return per quintile, cumulative return curves.

Data sources (tried in order):
* ``data/cb_history.parquet`` — fastest, contains scores when available
* ``output/YYYY-MM-DD/cb_daily_YYYYMMDD.json`` — full rankings per date
* ``cache/YYYYMMDD/cb_quote.parquet`` — historical CB prices

Graceful degradation: returns partial results when data is insufficient.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Result of a single backtest run."""
    start_date: str = ""
    end_date: str = ""
    num_periods: int = 0
    forward_days: int = 5
    ic_summary: dict = field(default_factory=dict)
    ic_decay: dict = field(default_factory=dict)
    quintile_returns: dict[int, float] = field(default_factory=dict)
    quintile_cumulative: pd.DataFrame | None = None
    quintile_risk_metrics: dict = field(default_factory=dict)
    factor_attribution: dict = field(default_factory=dict)
    summary: str = ""
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores_from_output(
    output_dir: str | Path,
    date_range: list[str] | None = None,
) -> pd.DataFrame:
    """Load historical composite scores from output JSON files.

    Args:
        output_dir: Root output directory (contains YYYY-MM-DD subdirs).
        date_range: Optional list of YYYYMMDD dates to include.

    Returns:
        DataFrame with columns [trade_date, bond_code, composite_score].
    """
    root = Path(output_dir)
    records = []

    if not root.exists():
        return pd.DataFrame(columns=["trade_date", "bond_code", "composite_score"])

    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name.replace("-", "")
        if date_range and date_str not in date_range:
            continue
        for json_file in date_dir.glob("cb_daily_*.json"):
            try:
                jdata = json.loads(json_file.read_text(encoding="utf-8"))
                for r in jdata.get("rankings", []):
                    rec = {
                        "trade_date": date_str,
                        "bond_code": str(r.get("bond_code", "")),
                        "composite_score": float(r.get("composite_score", 0)),
                    }
                    # Also extract dimension scores for factor attribution
                    for dim_col in ["valuation_score", "clause_score",
                                    "linkage_score", "structure_score"]:
                        if dim_col in r:
                            rec[dim_col] = float(r[dim_col])
                    records.append(rec)
            except Exception as e:
                logger.warning("Failed to read %s: %s", json_file, e)

    if not records:
        return pd.DataFrame(columns=["trade_date", "bond_code", "composite_score"])
    return pd.DataFrame(records)


def load_scores_from_history(
    history_store,
    date_range: list[str] | None = None,
) -> pd.DataFrame:
    """Load historical scores from HistoryStore parquet file.

    Faster than reading JSON files — used when score columns have been
    populated in the history parquet.

    Args:
        history_store: HistoryStore instance with loaded data.
        date_range: Optional list of YYYYMMDD dates to filter.

    Returns:
        DataFrame with [trade_date, bond_code, composite_score] plus any
        additional dimension score columns present.
    """
    df = history_store.load_history()
    if df.empty:
        return pd.DataFrame(columns=["trade_date", "bond_code", "composite_score"])

    # Filter to dates that actually have composite_score
    if "composite_score" not in df.columns:
        return pd.DataFrame(columns=["trade_date", "bond_code", "composite_score"])

    scored = df.dropna(subset=["composite_score"]).copy()
    if date_range:
        scored = scored[scored["trade_date"].astype(str).isin(date_range)]

    # Keep relevant columns
    keep = ["trade_date", "bond_code", "composite_score"]
    for col in ["valuation_score", "clause_score", "linkage_score", "structure_score"]:
        if col in scored.columns:
            keep.append(col)
    return scored[keep].reset_index(drop=True)


def load_cb_prices_from_cache(
    cache_root: str | Path,
    date_range: list[str],
) -> pd.DataFrame:
    """Load historical CB prices from cached parquet files.

    Args:
        cache_root: Cache root directory.
        date_range: List of YYYYMMDD dates to load.

    Returns:
        DataFrame with [trade_date, bond_code, cb_price].
    """
    root = Path(cache_root)
    records = []

    for date_str in date_range:
        date_dir = root / date_str
        quote_file = date_dir / "cb_quote.parquet"
        if not quote_file.exists():
            continue
        try:
            df = pd.read_parquet(quote_file)
            bond_col = next((c for c in ["bond_code", "转债代码", "债券代码"] if c in df.columns), None)
            price_col = next((c for c in ["cb_price", "转债最新价", "close"] if c in df.columns), None)
            if bond_col and price_col:
                for _, row in df.iterrows():
                    records.append({
                        "trade_date": date_str,
                        "bond_code": str(row[bond_col]),
                        "cb_price": float(row[price_col]),
                    })
        except Exception as e:
            logger.warning("Failed to read cache %s: %s", date_str, e)

    if not records:
        return pd.DataFrame(columns=["trade_date", "bond_code", "cb_price"])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_forward_returns(
    prices_df: pd.DataFrame,
    forward_days: int = 5,
) -> pd.DataFrame:
    """Compute forward N-day returns for each bond on each date.

    Forward return = (price[t + N] - price[t]) / price[t].

    Args:
        prices_df: DataFrame with [trade_date, bond_code, cb_price].
        forward_days: Number of trading days forward.

    Returns:
        DataFrame with [trade_date, bond_code, fwd_return].
    """
    if prices_df.empty:
        return pd.DataFrame(columns=["trade_date", "bond_code", "fwd_return"])

    df = prices_df.copy()
    df["trade_date"] = df["trade_date"].astype(str)

    # Sort by bond and date
    df = df.sort_values(["bond_code", "trade_date"])

    results = []
    for bond, grp in df.groupby("bond_code"):
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        prices = grp["cb_price"].values
        dates = grp["trade_date"].values
        for i in range(len(grp) - forward_days):
            p_now = prices[i]
            p_fwd = prices[i + forward_days]
            if p_now > 0 and p_fwd >= 0:
                results.append({
                    "trade_date": dates[i],
                    "bond_code": bond,
                    "fwd_return": (p_fwd - p_now) / p_now,
                })

    if not results:
        return pd.DataFrame(columns=["trade_date", "bond_code", "fwd_return"])
    return pd.DataFrame(results)


def compute_ic(
    scores_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> dict:
    """Compute Spearman rank IC between composite_score and forward returns.

    Args:
        scores_df: [trade_date, bond_code, composite_score].
        returns_df: [trade_date, bond_code, fwd_return].

    Returns:
        Dict with mean_ic, ic_std, ic_ir, ic_win_rate, daily_ics, num_periods.
    """
    merged = scores_df.merge(returns_df, on=["trade_date", "bond_code"], how="inner")
    if merged.empty:
        return {
            "mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0,
            "ic_win_rate": 0.0, "daily_ics": [], "num_periods": 0,
        }

    daily_ics = []
    for date, grp in merged.groupby("trade_date"):
        if len(grp) < 10:
            continue
        try:
            ic, _ = spearmanr(grp["composite_score"], grp["fwd_return"])
            if not np.isnan(ic):
                daily_ics.append({"trade_date": date, "ic": ic, "n_bonds": len(grp)})
        except Exception:
            pass

    if not daily_ics:
        return {
            "mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0,
            "ic_win_rate": 0.0, "daily_ics": [], "num_periods": 0,
        }

    ic_values = [d["ic"] for d in daily_ics]
    mean_ic = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
    ic_win_rate = sum(1 for v in ic_values if v > 0) / len(ic_values)

    return {
        "mean_ic": round(mean_ic, 6),
        "ic_std": round(ic_std, 6),
        "ic_ir": round(ic_ir, 4),
        "ic_win_rate": round(ic_win_rate, 4),
        "daily_ics": daily_ics,
        "num_periods": len(daily_ics),
    }


def compute_ic_decay(
    scores_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: list[int],
    min_bonds: int = 10,
) -> dict:
    """Compute IC across multiple forward horizons.

    Args:
        scores_df: [trade_date, bond_code, composite_score].
        prices_df: [trade_date, bond_code, cb_price].
        horizons: List of forward-day horizons (e.g. [1, 3, 5, 10, 20]).
        min_bonds: Minimum bonds per date for IC computation.

    Returns:
        Dict mapping horizon → ic_summary dict. Horizons with insufficient
        forward data are skipped (absent from the result dict).
    """
    result = {}
    for horizon in sorted(horizons):
        fwd_df = compute_forward_returns(prices_df, horizon)
        if fwd_df.empty:
            continue
        ic = compute_ic(scores_df, fwd_df)
        if ic.get("num_periods", 0) > 0:
            result[horizon] = ic
    return result


def compute_risk_metrics(
    daily_returns: list[float],
    risk_free_rate: float = 0.025,
    trading_days: int = 252,
) -> dict:
    """Compute risk-adjusted performance metrics from daily returns.

    Args:
        daily_returns: List of daily returns (as decimals, e.g. 0.01 = 1%).
        risk_free_rate: Annual risk-free rate (default 2.5%).
        trading_days: Trading days per year (default 252).

    Returns:
        Dict with cumulative_return, annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, n_days.
    """
    n = len(daily_returns)
    if n < 2:
        return {
            "cumulative_return": float(np.prod([1 + r for r in daily_returns]) - 1) if n > 0 else 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "n_days": n,
        }

    rets = np.array(daily_returns, dtype=float)
    cumulative = float(np.prod(1 + rets) - 1)
    annualized_ret = float((1 + cumulative) ** (trading_days / n) - 1)
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(trading_days))
    sharpe = (annualized_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown from cumulative curve
    cum_curve = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum_curve)
    drawdowns = (cum_curve - peak) / peak
    max_dd = float(drawdowns.min())

    return {
        "cumulative_return": round(cumulative, 6),
        "annualized_return": round(annualized_ret, 6),
        "annualized_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "n_days": n,
    }


def fama_macbeth(
    scores_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    factor_columns: list[str] | None = None,
    min_dates: int = 10,
) -> dict:
    """Fama-MacBeth cross-sectional regression for factor return attribution.

    Step 1: For each date, regress fwd_return on factor scores (OLS).
    Step 2: Time-series average of coefficients → factor risk premiums.

    Uses ``np.linalg.lstsq`` — no additional dependency required.

    Args:
        scores_df: [trade_date, bond_code, composite_score, ...factor columns].
        returns_df: [trade_date, bond_code, fwd_return].
        factor_columns: Column names for factor exposures. Defaults to
                        [valuation_score, clause_score, linkage_score, structure_score].
        min_dates: Minimum number of dates with valid regressions.

    Returns:
        Dict with factor_premiums, t_stats, n_dates, r_squared_avg, or error key.
    """
    if factor_columns is None:
        factor_columns = [
            "valuation_score", "clause_score", "linkage_score", "structure_score",
        ]

    # Check which factor columns are actually available
    available_factors = [c for c in factor_columns if c in scores_df.columns]
    if len(available_factors) < 2:
        return {"error": "insufficient_factor_columns",
                "available": available_factors, "required": factor_columns}

    merged = scores_df.merge(returns_df, on=["trade_date", "bond_code"], how="inner")
    if merged.empty:
        return {"error": "no_merged_data", "factor_premiums": {}, "t_stats": {}}

    factor_columns = available_factors
    coefs: dict[str, list[float]] = {f: [] for f in factor_columns}
    r_squared_list: list[float] = []

    for date, grp in merged.groupby("trade_date"):
        # Drop rows with NaN in any factor column
        valid = grp.dropna(subset=factor_columns + ["fwd_return"])
        if len(valid) < max(len(factor_columns) + 1, 10):
            continue
        y = valid["fwd_return"].values.astype(float)
        X = valid[factor_columns].values.astype(float)
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        try:
            coeffs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        for i, f in enumerate(["intercept"] + factor_columns):
            if f not in coefs:
                coefs[f] = []
            coefs[f].append(float(coeffs[i]))
        # R-squared
        if len(residuals) > 0:
            ss_res = float(residuals[0]) if residuals.size > 0 else 0.0
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r_squared_list.append(r2)

    if all(len(v) < min_dates for v in coefs.values()):
        return {"error": "insufficient_dates", "n_dates": max((len(v) for v in coefs.values()), default=0),
                "min_required": min_dates, "factor_premiums": {}, "t_stats": {}}

    factor_premiums = {}
    t_stats = {}
    n_dates = 0
    for f, vals in coefs.items():
        n = len(vals)
        if n < min_dates:
            continue
        n_dates = max(n_dates, n)
        mean_coef = float(np.mean(vals))
        std_coef = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        se_coef = std_coef / np.sqrt(n) if std_coef > 0 else float("inf")
        factor_premiums[f] = round(mean_coef, 8)
        t_stats[f] = round(mean_coef / se_coef, 4) if se_coef > 0 else 0.0

    return {
        "factor_premiums": factor_premiums,
        "t_stats": t_stats,
        "n_dates": n_dates,
        "r_squared_avg": round(float(np.mean(r_squared_list)), 4) if r_squared_list else 0.0,
        "factor_columns": factor_columns,
    }


def stratified_backtest(
    scores_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_quintiles: int = 5,
    risk_free_rate: float = 0.025,
) -> dict | None:
    """Stratified backtest: group bonds into quintiles by score each day.

    Args:
        scores_df: [trade_date, bond_code, composite_score].
        returns_df: [trade_date, bond_code, fwd_return].
        n_quintiles: Number of groups (default 5).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dict with keys:
        - "cumulative": {quintile: cumulative_return}
        - "daily": {quintile: [daily_mean_returns]}
        - "metrics": {quintile: risk_metrics_dict}
        Returns None if insufficient data.
    """
    merged = scores_df.merge(returns_df, on=["trade_date", "bond_code"], how="inner")
    if merged.empty:
        return None

    quintile_daily: dict[int, list[float]] = {q: [] for q in range(1, n_quintiles + 1)}

    for date, grp in merged.groupby("trade_date"):
        if len(grp) < n_quintiles * 3:
            continue
        grp = grp.copy()
        grp["quintile"] = pd.qcut(
            grp["composite_score"], q=n_quintiles, labels=False, duplicates="drop",
        ) + 1
        for q in range(1, n_quintiles + 1):
            q_grp = grp[grp["quintile"] == q]
            if not q_grp.empty:
                quintile_daily[q].append(q_grp["fwd_return"].mean())

    # Check we have enough data
    if all(len(v) < 2 for v in quintile_daily.values()):
        return None

    # Cumulative returns and risk metrics per quintile
    cumulative: dict[int, float] = {}
    metrics: dict[int, dict] = {}
    for q, rets in quintile_daily.items():
        if rets:
            cumulative[q] = round(float(np.prod([1 + r for r in rets]) - 1), 6)
        else:
            cumulative[q] = 0.0
        metrics[q] = compute_risk_metrics(rets, risk_free_rate=risk_free_rate)

    return {
        "cumulative": cumulative,
        "daily": quintile_daily,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Backtester:
    """Backtesting orchestrator for the CB scoring strategy."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        cfg = self._config
        bt = cfg.get("backtest", {})
        self._forward_days = int(bt.get("forward_days", 5))
        self._n_quintiles = int(bt.get("n_quintiles", 5))
        self._min_periods = int(bt.get("min_periods", 2))
        self._ic_horizons = [int(h) for h in bt.get("ic_horizons", [1, 3, 5, 10, 20])]
        self._output_dir = Path(cfg.get("output", {}).get("dir", "output"))
        self._cache_dir = Path("cache")
        opt = cfg.get("options", {})
        self._risk_free_rate = float(opt.get("risk_free_rate", 0.025))

    def run(
        self,
        date_range: list[str] | None = None,
    ) -> BacktestResult:
        """Run the full backtest analysis.

        Args:
            date_range: Optional list of YYYYMMDD dates to include.
                        If None, uses all available dates from output dir.

        Returns:
            BacktestResult with IC analysis and stratified returns.
        """
        errors: list[str] = []

        # 1. Load scores
        scores_df = load_scores_from_output(self._output_dir, date_range)
        if scores_df.empty:
            errors.append("No score data found in output directory")
            return BacktestResult(errors=errors)

        available_dates = sorted(scores_df["trade_date"].unique())
        if len(available_dates) < self._min_periods:
            errors.append(
                f"Insufficient data: {len(available_dates)} periods, "
                f"need at least {self._min_periods}"
            )
            return BacktestResult(
                start_date=available_dates[0] if available_dates else "",
                end_date=available_dates[-1] if available_dates else "",
                num_periods=len(available_dates),
                errors=errors,
            )

        # 2. Load CB prices
        prices_df = load_cb_prices_from_cache(self._cache_dir, available_dates)

        # 3. Compute forward returns (primary horizon)
        fwd_df = compute_forward_returns(prices_df, self._forward_days)

        # 4. IC decay analysis (multi-horizon)
        ic_decay = compute_ic_decay(scores_df, prices_df, self._ic_horizons)

        # 5. IC analysis (primary horizon, backward compat)
        ic_summary = ic_decay.get(self._forward_days, {})
        if not ic_summary:
            ic_summary = compute_ic(scores_df, fwd_df)

        # 6. Stratified backtest (primary horizon)
        quintile_data = stratified_backtest(
            scores_df, fwd_df, self._n_quintiles,
            risk_free_rate=self._risk_free_rate,
        )

        # Extract backward-compatible quintile_returns
        quintile_rets = quintile_data.get("cumulative", {}) if quintile_data else {}
        quintile_metrics = quintile_data.get("metrics", {}) if quintile_data else {}

        # 7. Factor attribution (if enabled)
        factor_attr = {}
        fa_cfg = self._config.get("factor_analysis", {})
        if fa_cfg.get("enabled", False):
            try:
                factor_attr = fama_macbeth(scores_df, fwd_df)
            except Exception as e:
                errors.append(f"Factor attribution failed: {e}")

        # 8. Summary
        summary_parts = []
        if ic_summary.get("num_periods", 0) > 0:
            mean_ic = ic_summary["mean_ic"]
            ic_ir = ic_summary["ic_ir"]
            summary_parts.append(
                f"Mean Rank IC: {mean_ic:.4f} | IC IR: {ic_ir:.2f} | "
                f"Win Rate: {ic_summary['ic_win_rate']:.1%}"
            )
            if abs(mean_ic) >= 0.03 and ic_ir >= 0.5:
                summary_parts.append("评分与未来收益显著正相关，策略有效。")
            elif abs(mean_ic) < 0.01:
                summary_parts.append("评分与未来收益相关性较弱，需关注权重校准。")
            else:
                summary_parts.append("评分有一定预测能力，建议持续跟踪。")

        if quintile_rets and len(quintile_rets) >= 2:
            q1 = quintile_rets.get(1, 0)
            q5 = quintile_rets.get(self._n_quintiles, 0)
            spread = q1 - q5
            summary_parts.append(
                f"Q1-Q{self._n_quintiles} 多空收益: {spread:.2%}"
            )
            if spread > 0.02:
                summary_parts.append("高分组合显著跑赢低分组合，评分区分度良好。")
            elif spread > 0:
                summary_parts.append("高分组略优于低分组，区分度尚可。")
            else:
                summary_parts.append("高分组未跑赢低分组，评分区分度不足。")

        return BacktestResult(
            start_date=available_dates[0],
            end_date=available_dates[-1],
            num_periods=len(available_dates),
            forward_days=self._forward_days,
            ic_summary=ic_summary,
            ic_decay=ic_decay,
            quintile_returns=quintile_rets or {},
            quintile_risk_metrics=quintile_metrics,
            factor_attribution=factor_attr,
            summary=" | ".join(summary_parts) if summary_parts else "",
            errors=errors,
        )
