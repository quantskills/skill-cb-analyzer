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
    benchmark_returns: dict = field(default_factory=dict)
    benchmark_comparison: dict = field(default_factory=dict)
    weight_calibration: dict = field(default_factory=dict)
    dynamic_weights: dict = field(default_factory=dict)
    cost_model: dict = field(default_factory=dict)
    delisting_analysis: dict = field(default_factory=dict)  # v1.7 delisting tracking
    summary: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class DelistingRecord:
    """Record of a bond that exited the universe during backtest period."""
    bond_code: str = ""
    bond_name: str = ""
    last_date: str = ""             # last date bond appeared in data
    delisting_reason: str = ""      # "redemption"(强赎), "maturity"(到期), "stock_delist"(正股退市), "other"
    final_price: float = 0.0
    cumulative_return: float = 0.0  # return from first observation to last
    note: str = ""


# Known delisting reason patterns
_DELISTING_KEYWORDS = {
    "redemption": ["强赎", "赎回", "提前赎回", "已公告强赎", "redeem"],
    "maturity": ["到期", "摘牌", "到期赎回", "maturity", "expir"],
    "stock_delist": ["正股退市", "终止上市", "面值退市", "delist"],
}


def _classify_delisting_reason(bond_code: str, histories: dict) -> str:
    """Heuristic classification of why a bond disappeared from the universe.

    Checks (in priority order):
    1. Price > 130 near exit → likely redemption
    2. Price near 100 near exit + remaining_years → maturity
    3. Stock status = ST/delisted → stock delisting
    4. Otherwise → "other"
    """
    if bond_code not in histories:
        return "other"
    hist = histories[bond_code]
    if hist.empty:
        return "other"

    last = hist.iloc[-1]
    price = float(last.get("cb_price", last.get("close", 0)))
    cv = float(last.get("conversion_value", 0))
    remaining = float(last.get("remaining_years", 99))

    # Strong price + high conversion value → likely redemption
    if price >= 130 or (cv > 0 and cv / max(price, 1) > 1.2):
        return "redemption"
    # Near par + short remaining life → maturity
    if 95 <= price <= 110 and remaining < 0.5:
        return "maturity"
    # Very low price → possible stock delisting
    if price < 70:
        return "stock_delist"

    return "other"


def analyze_delisting_survivorship(
    scores_df: pd.DataFrame,
    prices_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Analyze delisted bonds and their impact on backtest results.

    During the backtest period, bonds that exit the universe (via redemption,
    maturity, or stock delisting) are NOT included in subsequent periods.
    This creates a survivorship bias:
    - Redemption-delisted bonds typically performed WELL (stock rose → forced conversion)
    - Excluding them UNDERSTATES the strategy's actual returns
    - Maturity-delisted bonds are neutral
    - Stock-delisted bonds typically performed POORLY
    - Excluding them OVERSTATES the strategy's actual returns

    This function quantifies the magnitude and direction of the bias.

    Args:
        scores_df: DataFrame with trade_date, bond_code, composite_score, cb_price, ...
        prices_df: Optional historical prices for return calculation.
        output_dir: Optional directory to scan for delisted bond data.

    Returns:
        Dict with delisting_stats: total bonds, delisted count by reason,
        estimated bias direction and magnitude.
    """
    if scores_df.empty:
        return {"analyzed": False, "note": "无评分数据，跳过退市分析"}

    # Build per-bond history
    all_dates = sorted(scores_df["trade_date"].unique())
    if len(all_dates) < 10:
        return {"analyzed": False, "note": f"数据周期过短（{len(all_dates)}天），跳过退市分析"}

    first_date = all_dates[0]
    last_date = all_dates[-1]
    first_bonds = set(scores_df[scores_df["trade_date"] == first_date]["bond_code"].unique())
    last_bonds = set(scores_df[scores_df["trade_date"] == last_date]["bond_code"].unique())

    # Bonds that existed at start but not at end
    disappeared = first_bonds - last_bonds
    # Bonds that appeared after start
    appeared = last_bonds - first_bonds

    # Classify each disappeared bond
    delisting_records: list[DelistingRecord] = []
    histories: dict[str, pd.DataFrame] = {}
    for bond in disappeared:
        bond_hist = scores_df[scores_df["bond_code"] == bond].sort_values("trade_date")
        if bond_hist.empty:
            continue
        histories[bond] = bond_hist
        last_row = bond_hist.iloc[-1]
        reason = _classify_delisting_reason(bond, histories)

        # Get bond name
        name_col = next((c for c in ["bond_name", "转债名称", "name"] if c in bond_hist.columns), None)
        bond_name = str(last_row.get(name_col, bond)) if name_col else bond

        # Estimate cumulative return
        if len(bond_hist) >= 2:
            first_price = float(bond_hist.iloc[0].get("cb_price", 0))
            last_price = float(last_row.get("cb_price", 0))
            if first_price > 0:
                cum_ret = (last_price - first_price) / first_price
            else:
                cum_ret = 0.0
        else:
            cum_ret = 0.0

        delisting_records.append(DelistingRecord(
            bond_code=bond,
            bond_name=bond_name,
            last_date=str(last_row.get("trade_date", "")),
            delisting_reason=reason,
            final_price=float(last_row.get("cb_price", 0)),
            cumulative_return=round(cum_ret, 4),
        ))

    # Aggregate by reason
    reasons: dict[str, list[DelistingRecord]] = {}
    for r in delisting_records:
        reasons.setdefault(r.delisting_reason, []).append(r)

    reason_stats = {}
    for reason, records in reasons.items():
        avg_ret = np.mean([r.cumulative_return for r in records]) if records else 0.0
        reason_stats[reason] = {
            "count": len(records),
            "avg_cumulative_return": round(float(avg_ret), 4),
            "direction": "positive" if avg_ret > 0.02 else ("negative" if avg_ret < -0.02 else "neutral"),
            "examples": [{"code": r.bond_code, "name": r.bond_name, "last_date": r.last_date,
                          "reason": r.delisting_reason, "final_price": r.final_price,
                          "cum_return": r.cumulative_return}
                         for r in records[:5]],
        }

    # Estimate survivorship bias
    total_delisted = len(disappeared)
    redemption_count = reason_stats.get("redemption", {}).get("count", 0)
    stock_delist_count = reason_stats.get("stock_delist", {}).get("count", 0)
    maturity_count = reason_stats.get("maturity", {}).get("count", 0)
    other_count = reason_stats.get("other", {}).get("count", 0)

    # Redemption-delisted bonds were winners → excluding them UNDERSTATES returns
    # Stock-delisted bonds were losers → excluding them OVERSTATES returns
    net_bias = 0.0
    if total_delisted > 0:
        redemption_avg = reason_stats.get("redemption", {}).get("avg_cumulative_return", 0.0)
        stock_avg = reason_stats.get("stock_delist", {}).get("avg_cumulative_return", 0.0)
        # Net effect: negative = backtest understates true returns
        net_bias = -(redemption_count * redemption_avg + stock_delist_count * stock_avg) / max(total_delisted, 1)

    bias_direction = ""
    if net_bias < -0.03:
        bias_direction = "低估（强赎退市券多为正收益，排除后策略收益被系统性低估）"
    elif net_bias > 0.03:
        bias_direction = "高估（正股退市券多为负收益，排除后策略收益被系统性高估）"
    else:
        bias_direction = f"大致平衡（净偏倚{net_bias:+.2%}，在可接受范围内）"

    note_parts = [
        f"退市转债分析（{first_date} → {last_date}，共{len(all_dates)}个交易日）：",
        f"期初 {len(first_bonds)} 只 → 期末 {len(last_bonds)} 只",
        f"退市 {total_delisted} 只：强赎 {redemption_count} | 到期 {maturity_count} | 正股退市 {stock_delist_count} | 其他 {other_count}",
        f"新上市 {len(appeared)} 只",
        f"生存偏差方向：{bias_direction}",
    ]

    return {
        "analyzed": True,
        "first_date": first_date,
        "last_date": last_date,
        "n_periods": len(all_dates),
        "bonds_start": len(first_bonds),
        "bonds_end": len(last_bonds),
        "delisted_total": total_delisted,
        "newly_listed": len(appeared),
        "by_reason": reason_stats,
        "net_bias": round(float(net_bias), 4),
        "bias_direction": bias_direction,
        "note": "\n".join(note_parts),
    }
    """Transaction cost configuration for realistic backtesting."""
    enabled: bool = True
    stamp_duty: float = 0.0005       # 0.05% 单边卖出印花税
    commission: float = 0.0001       # 0.01% 单边佣金
    slippage: float = 0.0001         # 0.01% 滑点
    min_daily_turnover: float = 100  # 万元，最低日成交额
    filter_limit_hit: bool = True    # 过滤涨跌停

    @property
    def round_trip_cost(self) -> float:
        """Total round-trip cost as decimal fraction."""
        if not self.enabled:
            return 0.0
        return self.stamp_duty + 2 * self.commission + 2 * self.slippage


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
        DataFrame with [trade_date, bond_code, cb_price, turnover, pre_close].
        turnover and pre_close columns are included when available.
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
            turnover_col = next((c for c in ["turnover", "amount", "成交额"] if c in df.columns), None)
            pre_close_col = next((c for c in ["pre_close"] if c in df.columns), None)
            if bond_col and price_col:
                for _, row in df.iterrows():
                    rec = {
                        "trade_date": date_str,
                        "bond_code": str(row[bond_col]),
                        "cb_price": float(row[price_col]),
                    }
                    if turnover_col is not None:
                        rec["turnover"] = float(row.get(turnover_col, 0) or 0)
                    if pre_close_col is not None:
                        rec["pre_close"] = float(row.get(pre_close_col, 0) or 0)
                    records.append(rec)
        except Exception as e:
            logger.warning("Failed to read cache %s: %s", date_str, e)

    if not records:
        return pd.DataFrame(columns=["trade_date", "bond_code", "cb_price"])
    return pd.DataFrame(records)


def load_index_prices_from_cache(
    cache_root: str | Path,
    date_range: list[str],
    index_code: str = "000832",
) -> pd.DataFrame:
    """Load index daily prices from cached parquet file.

    Args:
        cache_root: Cache root directory.
        date_range: List of YYYYMMDD dates to include.
        index_code: Index code (default: 000832 中证转债指数).

    Returns:
        DataFrame with [trade_date, close].
    """
    cache_file = Path(cache_root) / f"index_{index_code}.parquet"
    if not cache_file.exists():
        logger.warning("Index cache not found: %s", cache_file)
        return pd.DataFrame(columns=["trade_date", "close"])

    try:
        df = pd.read_parquet(cache_file)
        df["trade_date"] = df["trade_date"].astype(str)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["trade_date"].isin(set(date_range))]
        return df.sort_values("trade_date").reset_index(drop=True)
    except Exception as e:
        logger.warning("Failed to read index cache: %s", e)
        return pd.DataFrame(columns=["trade_date", "close"])


def compute_benchmark_returns(
    index_df: pd.DataFrame,
    date_range: list[str],
    forward_days: int = 5,
) -> dict:
    """Compute benchmark (index) returns for the same forward horizons.

    For each trade_date, index_return = close[t+N] / close[t] - 1.

    Args:
        index_df: [trade_date, close] DataFrame.
        date_range: Sorted list of trade dates.
        forward_days: Forward holding period in trading days.

    Returns:
        dict with daily_returns, cumulative_return, annualized_return,
        annualized_volatility, sharpe_ratio, max_drawdown.
    """
    if index_df.empty or len(date_range) < 2:
        return {"daily_returns": [], "cumulative_return": 0.0, "error": "insufficient_data"}

    close_map = dict(zip(index_df["trade_date"], index_df["close"]))
    daily_returns: list[float] = []

    for i, t_date in enumerate(date_range):
        cur_price = close_map.get(t_date)
        if cur_price is None:
            continue
        # Find close after forward_days trading days
        fwd_idx = i + forward_days
        if fwd_idx < len(date_range):
            fwd_date = date_range[fwd_idx]
            fwd_price = close_map.get(fwd_date)
            if fwd_price is not None and cur_price > 0:
                daily_returns.append(float(fwd_price / cur_price - 1))

    if len(daily_returns) < 2:
        return {"daily_returns": daily_returns, "cumulative_return": 0.0, "error": "insufficient_data"}

    rets = np.array(daily_returns)
    cumulative = float(np.prod(1 + rets) - 1)
    ann_ret = float((1 + cumulative) ** (252 / len(rets)) - 1)
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(252))
    sharpe = (ann_ret - 0.025) / ann_vol if ann_vol > 0 else 0.0

    cum_curve = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum_curve)
    drawdowns = (cum_curve - peak) / peak
    max_dd = float(drawdowns.min())

    return {
        "daily_returns": [float(r) for r in daily_returns],
        "cumulative_return": round(cumulative, 6),
        "annualized_return": round(ann_ret, 6),
        "annualized_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "n_periods": len(daily_returns),
    }


def _compute_benchmark_comparison(
    q1_daily: list[float],
    bm_daily: list[float],
) -> dict:
    """Compare strategy Q1 vs benchmark.

    Aligns the two daily return series by index position.
    Computes excess return, information ratio, tracking error, win rate.

    Args:
        q1_daily: Q1 (top quintile) daily returns.
        bm_daily: Benchmark daily returns.

    Returns:
        dict with comparison metrics.
    """
    n = min(len(q1_daily), len(bm_daily))
    if n < 2:
        return {"error": "insufficient_overlap", "n_periods": n}

    q1_arr = np.array(q1_daily[:n])
    bm_arr = np.array(bm_daily[:n])
    excess = q1_arr - bm_arr

    tracking_error = float(np.std(excess, ddof=1) * np.sqrt(252))
    mean_excess = float(np.mean(excess))
    info_ratio = (mean_excess * 252) / tracking_error if tracking_error > 0 else 0.0

    q1_cum = float(np.prod(1 + q1_arr) - 1)
    bm_cum = float(np.prod(1 + bm_arr) - 1)

    return {
        "strategy_cumulative": round(q1_cum, 6),
        "benchmark_cumulative": round(bm_cum, 6),
        "excess_return": round(q1_cum - bm_cum, 6),
        "information_ratio": round(info_ratio, 4),
        "tracking_error": round(tracking_error, 6),
        "win_rate": round(float(np.mean(excess > 0)), 4),
        "n_periods": n,
    }


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_forward_returns(
    prices_df: pd.DataFrame,
    forward_days: int = 5,
    cost_model: CostModel | None = None,
) -> pd.DataFrame:
    """Compute forward N-day returns for each bond on each date.

    Forward return = (price[t + N] - price[t]) / price[t].

    When cost_model is provided:
      - Bonds that hit limit-up/down on date t are skipped
      - Bonds with turnover < min_daily_turnover on date t are skipped

    Args:
        prices_df: DataFrame with [trade_date, bond_code, cb_price]
                   plus optional [turnover, pre_close] columns.
        forward_days: Number of trading days forward.
        cost_model: Optional CostModel for filtering.

    Returns:
        DataFrame with [trade_date, bond_code, fwd_return].
    """
    if prices_df.empty:
        return pd.DataFrame(columns=["trade_date", "bond_code", "fwd_return"])

    df = prices_df.copy()
    df["trade_date"] = df["trade_date"].astype(str)

    has_turnover = cost_model is not None and "turnover" in df.columns
    has_pre_close = cost_model is not None and "pre_close" in df.columns

    # Sort by bond and date
    df = df.sort_values(["bond_code", "trade_date"])

    results = []
    for bond, grp in df.groupby("bond_code"):
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        prices = grp["cb_price"].values
        dates = grp["trade_date"].values
        turnovers = grp["turnover"].values if has_turnover else None
        pre_closes = grp["pre_close"].values if has_pre_close else None

        for i in range(len(grp) - forward_days):
            p_now = prices[i]
            p_fwd = prices[i + forward_days]
            if p_now <= 0 or p_fwd <= 0 or np.isnan(p_now) or np.isnan(p_fwd):
                continue

            # Cost model filtering on entry date
            if cost_model is not None and cost_model.enabled:
                # Limit-hit filter
                if cost_model.filter_limit_hit and pre_closes is not None:
                    pre = pre_closes[i]
                    if pre > 0:
                        day_ret = abs(p_now / pre - 1)
                        if day_ret >= 0.099:  # ~10% limit
                            continue
                # Turnover filter
                if cost_model.min_daily_turnover > 0 and turnovers is not None:
                    if float(turnovers[i]) < cost_model.min_daily_turnover:
                        continue

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
            logger.debug("IC computation skipped for one date", exc_info=True)

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


def _newey_west_se(series: np.ndarray, max_lags: int | None = None) -> float:
    """Newey-West HAC standard error of the mean.

    Uses Bartlett kernel weighting with automatic lag selection
    (Newey-West 1994): ``max_lags = int(4 * (n/100)^(2/9))``.

    Args:
        series: 1-D array of time-series observations (e.g. coefficient estimates).
        max_lags: Maximum lag for autocorrelation. Auto-selected if None.

    Returns:
        HAC standard error of the mean. Returns NaN if series has < 2 obs.
    """
    n = len(series)
    if n < 2:
        return float("nan")

    if max_lags is None:
        max_lags = int(4 * (n / 100) ** (2 / 9))
    max_lags = min(max_lags, n - 1)

    mean = np.mean(series)
    resid = series - mean

    # Variance of the mean under i.i.d. → γ̂_0 / n
    gamma0 = np.sum(resid ** 2) / n

    # Autocovariance terms with Bartlett kernel
    autocov_sum = 0.0
    for j in range(1, max_lags + 1):
        w = 1.0 - j / (max_lags + 1)  # Bartlett kernel
        gamma_j = np.sum(resid[j:] * resid[:-j]) / n
        autocov_sum += w * gamma_j

    nw_var = (gamma0 + 2 * autocov_sum) / n
    if nw_var < 0:
        nw_var = gamma0 / n  # Fallback to i.i.d. if negative

    return float(np.sqrt(nw_var))


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
    nw_t_stats = {}
    n_dates = 0
    for f, vals in coefs.items():
        n = len(vals)
        if n < min_dates:
            continue
        n_dates = max(n_dates, n)
        mean_coef = float(np.mean(vals))
        std_coef = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        se_coef = std_coef / np.sqrt(n) if std_coef > 0 else float("inf")

        # Newey-West HAC standard error
        nw_se = _newey_west_se(np.array(vals))
        nw_t = round(mean_coef / nw_se, 4) if nw_se > 0 and not np.isnan(nw_se) else 0.0

        factor_premiums[f] = round(mean_coef, 8)
        t_stats[f] = round(mean_coef / se_coef, 4) if se_coef > 0 else 0.0
        nw_t_stats[f] = nw_t

    return {
        "factor_premiums": factor_premiums,
        "t_stats": t_stats,
        "nw_t_stats": nw_t_stats,
        "n_dates": n_dates,
        "r_squared_avg": round(float(np.mean(r_squared_list)), 4) if r_squared_list else 0.0,
        "factor_columns": factor_columns,
    }


def stratified_backtest(
    scores_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    n_quintiles: int = 5,
    risk_free_rate: float = 0.025,
    cost_pct: float = 0.0,
) -> dict | None:
    """Stratified backtest: group bonds into quintiles by score each day.

    Args:
        scores_df: [trade_date, bond_code, composite_score].
        returns_df: [trade_date, bond_code, fwd_return].
        n_quintiles: Number of groups (default 5).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        cost_pct: Transaction cost deducted from each period's return.

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
                mean_ret = q_grp["fwd_return"].mean()
                if cost_pct > 0:
                    mean_ret -= cost_pct
                quintile_daily[q].append(mean_ret)

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
# Weight calibration
# ---------------------------------------------------------------------------

def _generate_simplex_grid(n_vars: int, step: float) -> list[tuple[float, ...]]:
    """Generate all weight tuples on the simplex sum-to-1 constraint.

    Uses stars-and-bars combinatorial enumeration: for n_vars dimensions
    and 1/step total units, generates all combinations of n_vars non-negative
    integers summing to total, then converts to float weights.

    Example: n_vars=4, step=0.05 → 1771 grid points.
    """
    from itertools import combinations_with_replacement
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    total = round(1.0 / step)
    step = 1.0 / total  # Recompute step from rounded total so weights sum to 1.0
    # Generate unique combos of size n_vars from [0, ..., total + n_vars - 1]
    grid = []
    for combo in combinations_with_replacement(range(total + 1), n_vars - 1):
        # Convert to weights via stars-and-bars
        weights = []
        prev = 0
        for c in sorted(combo):
            weights.append((c - prev) * step)
            prev = c
        weights.append((total - prev) * step)
        grid.append(tuple(weights))
    return grid


def calibrate_dimension_weights(
    scores_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """Grid search over 4-dimension weight space to maximize mean Rank IC.

    Walk-forward split:
      1. Sort dates chronologically
      2. Train on first train_ratio (default 60%) of dates
      3. Validate on remaining 40%

    For each weight combo on the simplex grid:
      composite = w1*val + w2*clause + w3*link + w4*struct
      Compute Spearman Rank IC vs forward returns on train set.

    Args:
        scores_df: [trade_date, bond_code, composite_score, valuation_score,
                    clause_score, linkage_score, structure_score].
        returns_df: [trade_date, bond_code, fwd_return].
        config: Optional config dict for calibration parameters.

    Returns:
        dict with optimal_weights, train_ic, valid_ic, total_combinations.
    """
    cfg = config or {}
    cal_cfg = cfg.get("backtest", {}).get("calibration", {})
    step = float(cal_cfg.get("dimension_step", 0.05))
    train_ratio = float(cal_cfg.get("train_ratio", 0.6))
    min_dates = int(cal_cfg.get("min_dates_train", 10))

    # Check required columns
    dim_cols = ["valuation_score", "clause_score", "linkage_score", "structure_score"]
    available = [c for c in dim_cols if c in scores_df.columns]
    if len(available) < 4:
        return {"error": "missing_dimension_scores", "available": available}

    # Merge scores with returns
    merged = scores_df.merge(returns_df, on=["trade_date", "bond_code"], how="inner")
    if merged.empty:
        return {"error": "no_merged_data"}

    # Walk-forward split by date
    dates = sorted(merged["trade_date"].unique())
    if len(dates) < min_dates:
        return {"error": "insufficient_dates", "n_dates": len(dates), "min_required": min_dates}

    split_idx = max(min_dates, int(len(dates) * train_ratio))
    train_dates = set(dates[:split_idx])
    valid_dates = set(dates[split_idx:])
    train = merged[merged["trade_date"].isin(train_dates)]
    valid = merged[merged["trade_date"].isin(valid_dates)]

    if len(train["trade_date"].unique()) < min_dates:
        return {"error": "insufficient_train_dates"}

    # Generate grid
    grid = _generate_simplex_grid(4, step)
    best_weights = None
    best_ic = -999.0

    for w1, w2, w3, w4 in grid:
        train_copy = train.copy()
        train_copy["composite"] = (
            w1 * train_copy["valuation_score"] / 100
            + w2 * train_copy["clause_score"] / 100
            + w3 * train_copy["linkage_score"] / 100
            + w4 * train_copy["structure_score"] / 100
        )
        # Compute IC per date
        ic_vals = []
        for date, grp in train_copy.groupby("trade_date"):
            if len(grp) >= 10:
                try:
                    ic, _ = spearmanr(grp["composite"], grp["fwd_return"])
                    if not np.isnan(ic):
                        ic_vals.append(ic)
                except Exception:
                    logger.debug("IC computation skipped for one date (calibrate train)", exc_info=True)
        if ic_vals:
            mean_ic = float(np.mean(ic_vals))
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_weights = (w1, w2, w3, w4)

    if best_weights is None:
        return {"error": "no_valid_ic", "total_combinations": len(grid)}

    # Evaluate on validation set
    valid_copy = valid.copy()
    w1, w2, w3, w4 = best_weights
    valid_copy["composite"] = (
        w1 * valid_copy["valuation_score"] / 100
        + w2 * valid_copy["clause_score"] / 100
        + w3 * valid_copy["linkage_score"] / 100
        + w4 * valid_copy["structure_score"] / 100
    )
    valid_ic_vals = []
    for date, grp in valid_copy.groupby("trade_date"):
        if len(grp) >= 10:
            try:
                ic, _ = spearmanr(grp["composite"], grp["fwd_return"])
                if not np.isnan(ic):
                    valid_ic_vals.append(ic)
            except Exception:
                logger.debug("IC computation skipped for one date (calibrate valid)", exc_info=True)
    valid_ic = float(np.mean(valid_ic_vals)) if valid_ic_vals else 0.0

    return {
        "optimal_weights": {
            "valuation": round(w1, 4),
            "clause": round(w2, 4),
            "linkage": round(w3, 4),
            "structure": round(w4, 4),
        },
        "train_ic": round(best_ic, 6),
        "valid_ic": round(valid_ic, 6),
        "total_combinations": len(grid),
        "train_dates": len(train_dates),
        "valid_dates": len(valid_dates),
    }


# ---------------------------------------------------------------------------
# Dynamic IC weighting
# ---------------------------------------------------------------------------

# Mapping from config detector_weights keys → sig_ column names in history
SIGNAL_KEY_MAP: dict[str, str] = {
    "double_low": "sig_double_low",
    "ytm_defense": "sig_ytm_defense",
    "bond_floor": "sig_bond_floor",
    "premium_percentile": "sig_premium_percentile",
    "redemption_progress": "sig_redemption",
    "downward_revision": "sig_downward_revision",
    "putback_progress": "sig_putback",
    "maturity_alert": "sig_maturity",
    "stock_momentum": "sig_stock_momentum",
    "cb_stock_deviation": "sig_cb_stock_deviation",
    "delta_elasticity": "sig_delta",
    "stock_pattern": "sig_stock_pattern",
    "iv_percentile": "sig_iv_percentile",
    "hv_iv_divergence": "sig_hv_iv_divergence",
    "vol_expansion": "sig_vol_expansion",
    "bs_delta": "sig_bs_delta",
    "volume_active": "sig_volume",
    "balance_trend": "sig_balance_trend",
}


def compute_rolling_signal_ic(
    history_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    signal_keys: list[str],
    forward_days: int = 5,
    rolling_window: int = 20,
    min_periods: int = 10,
) -> dict[str, float]:
    """Compute rolling-window Spearman IC for each signal vs forward returns.

    For each signal, merges history data with forward returns, computes
    cross-sectional Spearman IC per date, then takes the mean over the
    most recent ``rolling_window`` dates.

    Args:
        history_df: [trade_date, bond_code, sig_*, ...] from HistoryStore.
        prices_df: [trade_date, bond_code, cb_price] for forward returns.
        signal_keys: Config-level signal keys (e.g. 'double_low').
        forward_days: Forward horizon for returns.
        rolling_window: Number of recent dates to average IC over.
        min_periods: Minimum dates required per signal for valid IC.

    Returns:
        Dict mapping signal_key → rolling_mean_IC. Signals with insufficient
        data are absent from the result.
    """
    if history_df.empty:
        return {}

    fwd_df = compute_forward_returns(prices_df, forward_days)
    if fwd_df.empty:
        return {}

    result: dict[str, float] = {}
    for key in signal_keys:
        sig_col = SIGNAL_KEY_MAP.get(key, f"sig_{key}")
        if sig_col not in history_df.columns:
            continue

        sig_data = history_df[["trade_date", "bond_code", sig_col]].dropna(subset=[sig_col]).copy()
        sig_data = sig_data.rename(columns={sig_col: "signal_value"})
        merged = sig_data.merge(fwd_df, on=["trade_date", "bond_code"], how="inner")
        if merged.empty:
            continue

        ic_vals = []
        for date, grp in merged.groupby("trade_date"):
            if len(grp) < 10:
                continue
            try:
                ic, _ = spearmanr(grp["signal_value"], grp["fwd_return"])
                if not np.isnan(ic):
                    ic_vals.append(ic)
            except Exception:
                logger.debug("IC computation skipped for one date (rolling signal)", exc_info=True)

        if len(ic_vals) >= min_periods:
            # Rolling mean: average over the most recent rolling_window values
            recent = ic_vals[-rolling_window:] if len(ic_vals) > rolling_window else ic_vals
            result[key] = round(float(np.mean(recent)), 6)

    return result


def compute_dynamic_weights(
    rolling_ics: dict[str, float],
    base_weights: dict[str, float],
    floor_ic: float = 0.0,
) -> dict[str, float]:
    """Adjust detector weights based on rolling signal IC.

    Formula:
        adj = base_weight * max(floor_ic, rolling_IC)
        w_signal = adj / sum(all_adjusted)  [normalize to sum 1]

    Signals with rolling_IC ≤ floor_ic get zero effective weight.
    If all signals have IC ≤ floor_ic, falls back to base weights unchanged.

    Args:
        rolling_ics: {signal_key: rolling_mean_IC}.
        base_weights: {signal_key: base_weight} from config.
        floor_ic: Minimum IC threshold (default 0.0).

    Returns:
        {signal_key: dynamic_weight} normalized to sum to 1.
    """
    if not rolling_ics or not base_weights:
        return dict(base_weights)

    adjusted: dict[str, float] = {}
    total_adj = 0.0
    for key, base_w in base_weights.items():
        ic_val = rolling_ics.get(key, 0.0)
        effective_ic = max(floor_ic, ic_val)
        adj = base_w * effective_ic
        adjusted[key] = adj
        total_adj += adj

    if total_adj <= 0:
        # All ICs ≤ floor_ic → fall back to base weights
        return dict(base_weights)

    return {k: round(v / total_adj, 6) for k, v in adjusted.items()}


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
        self._benchmark_enabled = bool(bt.get("benchmark_enabled", True))
        self._benchmark_code = str(bt.get("benchmark_code", "000832"))
        self._calibrate_weights = bool(bt.get("calibrate_weights", False))
        dw_cfg = bt.get("dynamic_weights", {})
        self._dynamic_weights_enabled = bool(dw_cfg.get("enabled", False))
        self._dw_rolling_window = int(dw_cfg.get("rolling_window", 20))
        self._dw_min_periods = int(dw_cfg.get("min_periods", 10))
        self._dw_floor_ic = float(dw_cfg.get("floor_ic", 0.0))
        cost_cfg = bt.get("cost_model", {})
        self._cost_model = CostModel(
            enabled=bool(cost_cfg.get("enabled", True)),
            stamp_duty=float(cost_cfg.get("stamp_duty", 0.0005)),
            commission=float(cost_cfg.get("commission", 0.0001)),
            slippage=float(cost_cfg.get("slippage", 0.0001)),
            min_daily_turnover=float(cost_cfg.get("min_daily_turnover", 100)),
            filter_limit_hit=bool(cost_cfg.get("filter_limit_hit", True)),
        )

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

        # 3. Compute forward returns (primary horizon, with cost filtering)
        fwd_df = compute_forward_returns(prices_df, self._forward_days, cost_model=self._cost_model)

        # 4. IC decay analysis (multi-horizon, also uses cost model for consistency)
        ic_decay = compute_ic_decay(scores_df, prices_df, self._ic_horizons)

        # 5. IC analysis (primary horizon, with cost-filtered returns)
        ic_summary = compute_ic(scores_df, fwd_df)

        # 6. Stratified backtest (primary horizon, with costs)
        cost_pct = self._cost_model.round_trip_cost if self._cost_model.enabled else 0.0
        quintile_data = stratified_backtest(
            scores_df, fwd_df, self._n_quintiles,
            risk_free_rate=self._risk_free_rate,
            cost_pct=cost_pct,
        )

        # Extract backward-compatible quintile_returns
        quintile_rets = quintile_data.get("cumulative", {}) if quintile_data else {}
        quintile_metrics = quintile_data.get("metrics", {}) if quintile_data else {}
        quintile_cum_df = None
        if quintile_data and quintile_data.get("daily"):
            try:
                daily = quintile_data["daily"]
                cum_series = {}
                for q, rets in daily.items():
                    if rets:
                        cum_returns = [1.0]
                        for r in rets:
                            cum_returns.append(cum_returns[-1] * (1 + r))
                        cum_series[q] = cum_returns
                if cum_series:
                    quintile_cum_df = pd.DataFrame(cum_series)
            except Exception:
                logger.debug("Quintile cumulative frame construction failed", exc_info=True)

        # 6.5 Benchmark comparison (if enabled)
        benchmark_data: dict = {}
        benchmark_comparison: dict = {}
        if self._benchmark_enabled:
            try:
                idx_df = load_index_prices_from_cache(
                    self._cache_dir, available_dates, self._benchmark_code,
                )
                if not idx_df.empty:
                    benchmark_data = compute_benchmark_returns(
                        idx_df, available_dates, self._forward_days,
                    )
                    if quintile_data and 1 in quintile_data.get("daily", {}):
                        q1_daily = quintile_data["daily"][1]
                        bm_daily = benchmark_data.get("daily_returns", [])
                        if q1_daily and bm_daily:
                            benchmark_comparison = _compute_benchmark_comparison(q1_daily, bm_daily)
            except Exception as e:
                errors.append(f"Benchmark comparison failed: {e}")

        # 7. Factor attribution (if enabled)
        factor_attr = {}
        fa_cfg = self._config.get("factor_analysis", {})
        if fa_cfg.get("enabled", False):
            try:
                factor_attr = fama_macbeth(scores_df, fwd_df)
            except Exception as e:
                errors.append(f"Factor attribution failed: {e}")

        # 7.5 Weight calibration (if enabled)
        weight_cal = {}
        if self._calibrate_weights:
            try:
                weight_cal = calibrate_dimension_weights(scores_df, fwd_df, self._config)
            except Exception as e:
                errors.append(f"Weight calibration failed: {e}")

        # 7.6 Dynamic IC weighting (if enabled)
        dynamic_weights = {}
        if self._dynamic_weights_enabled:
            try:
                hist_file = Path("data") / "cb_history.parquet"
                if hist_file.exists():
                    history_df = pd.read_parquet(hist_file)
                    base_weights = self._config.get("detector_weights", {})
                    signal_keys = list(base_weights.keys())
                    if signal_keys and not history_df.empty:
                        rolling_ics = compute_rolling_signal_ic(
                            history_df, prices_df, signal_keys,
                            forward_days=self._forward_days,
                            rolling_window=self._dw_rolling_window,
                            min_periods=self._dw_min_periods,
                        )
                        dynamic_weights = compute_dynamic_weights(
                            rolling_ics, base_weights, floor_ic=self._dw_floor_ic,
                        )
                else:
                    errors.append("Dynamic weights enabled but no history file found")
            except Exception as e:
                errors.append(f"Dynamic weight computation failed: {e}")

        # 7.7 Delisting / survivorship analysis (v1.7)
        delisting_results: dict = {}
        try:
            delisting_results = analyze_delisting_survivorship(
                scores_df, prices_df, output_dir=self._output_dir,
            )
        except Exception as e:
            errors.append(f"Delisting analysis failed: {e}")

        # 8. Summary
        summary_parts = []
        # Add delisting note to summary
        if delisting_results.get("analyzed"):
            summary_parts.append(
                f"退市分析：期初{delisting_results['bonds_start']}只→"
                f"期末{delisting_results['bonds_end']}只，"
                f"生存偏差：{delisting_results['bias_direction']}"
            )

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
            quintile_cumulative=quintile_cum_df,
            quintile_risk_metrics=quintile_metrics,
            factor_attribution=factor_attr,
            benchmark_returns=benchmark_data,
            benchmark_comparison=benchmark_comparison,
            cost_model={
                "enabled": self._cost_model.enabled,
                "stamp_duty": self._cost_model.stamp_duty,
                "commission": self._cost_model.commission,
                "slippage": self._cost_model.slippage,
                "round_trip_cost": self._cost_model.round_trip_cost,
                "min_daily_turnover": self._cost_model.min_daily_turnover,
                "filter_limit_hit": self._cost_model.filter_limit_hit,
            },
            weight_calibration=weight_cal,
            dynamic_weights=dynamic_weights,
            delisting_analysis=delisting_results,
            summary=" | ".join(summary_parts) if summary_parts else "",
            errors=errors,
        )
