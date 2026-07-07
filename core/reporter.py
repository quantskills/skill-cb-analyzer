"""Report generator: Markdown (10-section) + JSON for CB daily analysis.

Report sections:
    1. 市场概览 — Market overview
    2. 双低策略精选 — Double-Low Top picks
    3. 高YTM防御组合 — High-YTM defensive bonds
    4. 条款事件监控 — Clause events (redemption/revision/putback/maturity)
    5. 正股联动精选 — Stock-CB linkage opportunities
    6. 行业分布 — Industry distribution
    7. 信用风险告警 — Credit risk alerts
    8. 综合评分排名 — Composite score ranking (Top N)
    9. AI逐券研判 — LLM per-bond analysis (placeholder)
    10. 数据溯源 — Data provenance
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from core.scorer import ScoreResult

logger = logging.getLogger(__name__)


def _safe_float(value: float) -> float:
    """Return 0.0 if value is NaN or Inf, else value."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return float(value)


class Reporter:
    """Generates Markdown + JSON reports for CB daily analysis."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        out_cfg = cfg.get("output", {})
        self._output_dir = Path(out_cfg.get("dir", "output"))
        self._config = cfg

        # Read thresholds from config (same keys as detectors)
        val = cfg.get("valuation", {})
        self._dl_price_max = float(val.get("double_low_price_max", 120))
        self._dl_premium_max = float(val.get("double_low_premium_max", 20))
        self._ytm_threshold = float(val.get("ytm_threshold", 3.0)) / 100  # config in %, code in decimal

        clause = cfg.get("clause", {})
        self._warn_ratio = float(clause.get("redemption_warn_ratio", 1.20))
        self._danger_ratio = float(clause.get("redemption_danger_ratio", 1.28))

    def generate(
        self,
        trade_date: str,
        cb_df: pd.DataFrame,
        ranked: list[ScoreResult],
        all_results: list[ScoreResult],
        val_results: dict,
        clause_results: dict,
        link_results: dict,
        struct_results: dict,
        top_n: int = 20,
        stock_info: pd.DataFrame | None = None,
        llm_analyses: dict[str, str] | None = None,
        backtest_result=None,
        vol_results: dict | None = None,
        signal_correlation: dict | None = None,
    ) -> tuple[str, str]:
        """Generate both MD and JSON reports.

        Returns:
            (md_path, json_path) as strings.
        """
        date_formatted = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
        out_dir = self._output_dir / date_formatted
        out_dir.mkdir(parents=True, exist_ok=True)

        md_path = out_dir / f"cb_daily_{trade_date}.md"
        json_path = out_dir / f"cb_daily_{trade_date}.json"

        # Generate
        md_content = self._render_md(trade_date, date_formatted, cb_df, ranked,
                                      all_results, val_results, clause_results,
                                      link_results, struct_results, top_n,
                                      stock_info=stock_info,
                                      llm_analyses=llm_analyses or {},
                                      backtest_result=backtest_result,
                                      vol_results=vol_results,
                                      signal_correlation=signal_correlation)
        json_content = self._render_json(trade_date, cb_df, ranked, all_results,
                                          llm_analyses=llm_analyses or {})

        md_path.write_text(md_content, encoding="utf-8")
        json_path.write_text(json_content, encoding="utf-8")

        logger.info("Report saved: %s, %s", md_path, json_path)
        return str(md_path), str(json_path)

    # -- Markdown ---------------------------------------------------

    def _render_md(
        self, trade_date: str, date_formatted: str,
        cb_df: pd.DataFrame, ranked: list[ScoreResult],
        all_results: list[ScoreResult],
        val_results: dict, clause_results: dict,
        link_results: dict, struct_results: dict,
        top_n: int,
        stock_info: pd.DataFrame | None = None,
        llm_analyses: dict[str, str] | None = None,
        backtest_result=None,
        vol_results: dict | None = None,
        signal_correlation: dict | None = None,
    ) -> str:
        excluded = [s for s in all_results if s.excluded]
        active = [s for s in all_results if not s.excluded]
        top = ranked[:top_n]

        lines = []
        lines.append(f"# 可转债每日分析报告 — {date_formatted}")
        lines.append("")
        lines.append(f"> 扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"全市场转债: {len(cb_df)} 只 | 入选: {len(active)} 只 | "
                      f"排除(强赎/信用): {len(excluded)} 只")
        lines.append("")

        # ── 1. 市场概览 ──
        lines.append("## 1. 市场概览")
        lines.append("")
        avg_price = cb_df["cb_price"].mean() if "cb_price" in cb_df.columns else 0
        avg_premium = cb_df["premium_rate"].mean() if "premium_rate" in cb_df.columns else 0
        avg_ytm = cb_df["ytm"].mean() * 100 if "ytm" in cb_df.columns else 0
        # Guard against NaN/Inf when column exists but all values are empty
        avg_price = _safe_float(avg_price)
        avg_premium = _safe_float(avg_premium)
        avg_ytm = _safe_float(avg_ytm)
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 转债总数 | {len(cb_df)} |")
        lines.append(f"| 平均价格 | {avg_price:.2f} 元 |")
        lines.append(f"| 平均转股溢价率 | {avg_premium:.1f}% |")
        lines.append(f"| 平均到期收益率 | {avg_ytm:.2f}% |")
        lines.append(f"| 入选数量 | {len(active)} |")
        lines.append(f"| 排除数量 | {len(excluded)} |")
        lines.append("")

        # Signal statistics
        lines.append("### 信号触发统计")
        lines.append("")
        lines.append(f"| 信号组 | 检测器 | 触发数 |")
        lines.append(f"|--------|--------|--------|")

        signal_groups = [("估值", val_results), ("条款", clause_results),
                         ("正股联动", link_results), ("波动率与期权", vol_results or {}),
                         ("市场结构", struct_results)]
        for group_name, results in signal_groups:
            for key, sig_list in results.items():
                triggered_n = sum(1 for s in sig_list if s.triggered)
                label = sig_list[0].label if sig_list else key
                lines.append(f"| {group_name} | {label} | {triggered_n} |")
        lines.append("")

        # ── 2. 双低策略精选 ──
        lines.append(f"## 2. 双低策略精选 (价格<{self._dl_price_max:.0f}, 溢价率<{self._dl_premium_max:.0f}%)")
        lines.append("")
        if "premium_rate" in cb_df.columns and "cb_price" in cb_df.columns and "double_low" in cb_df.columns:
            double_low = cb_df[
                (cb_df["cb_price"] < self._dl_price_max) &
                (cb_df["premium_rate"] < self._dl_premium_max) &
                (cb_df["premium_rate"] > -100)
            ].nsmallest(15, "double_low")

            if not double_low.empty:
                lines.append("| 排名 | 转债 | 价格 | 溢价率 | 双低值 | 正股 |")
                lines.append("|------|------|------|--------|--------|------|")
                for i, (_, row) in enumerate(double_low.iterrows()):
                    name = row.get("bond_name", row.get("转债名称", ""))
                    price = float(row.get("cb_price", 0)) or 0
                    prem = float(row.get("premium_rate", 0)) or 0
                    dl = float(row.get("double_low", 0)) or 0
                    stock = row.get("stock_name", row.get("正股名称", ""))
                    lines.append(f"| {i+1} | {name} | {price:.2f} | {prem:.1f}% | {dl:.1f} | {stock} |")
                lines.append("")
            else:
                lines.append("当日无转债满足双低条件。")
                lines.append("")

        # ── 3. 高YTM防御组合 ──
        ytm_threshold_pct = self._ytm_threshold * 100  # convert decimal to %
        lines.append(f"## 3. 高YTM防御组合 (YTM > {ytm_threshold_pct:.0f}%)")
        lines.append("")
        if "ytm" in cb_df.columns:
            ytm_df = cb_df[cb_df["ytm"] > self._ytm_threshold].nsmallest(15, "premium_rate" if "premium_rate" in cb_df.columns else "ytm")
            if not ytm_df.empty:
                lines.append("| 排名 | 转债 | 价格 | YTM | 溢价率 | 评级 |")
                lines.append("|------|------|------|-----|--------|------|")
                for i, (_, row) in enumerate(ytm_df.iterrows()):
                    name = row.get("bond_name", row.get("转债名称", ""))
                    price = float(row.get("cb_price", 0)) or 0
                    ytm = float(row.get("ytm", 0)) or 0
                    ytm_pct = ytm * 100
                    prem = float(row.get("premium_rate", 0)) or 0
                    rating = row.get("credit_rating", row.get("债券评级", ""))
                    lines.append(f"| {i+1} | {name} | {price:.2f} | {ytm_pct:.1f}% | {prem:.1f}% | {rating} |")
                lines.append("")
            else:
                lines.append("当日无转债YTM超过3%。")
                lines.append("")

        # ── 4. 条款事件监控 ──
        lines.append("## 4. 条款事件监控")
        lines.append("")
        lines.append(f"### 4.1 强赎预警 (正股价/转股价 > {self._warn_ratio:.2f})")
        lines.append("")
        if "redemption_ratio" in cb_df.columns:
            warn = cb_df[cb_df["redemption_ratio"] >= self._warn_ratio].nlargest(10, "redemption_ratio")
            if not warn.empty:
                lines.append("| 转债 | 触发比 | 状态 |")
                lines.append("|------|--------|------|")
                for _, row in warn.iterrows():
                    name = row.get("bond_name", row.get("转债名称", ""))
                    ratio = float(row.get("redemption_ratio", 0)) or 0
                    level = "🚨高危" if ratio >= self._danger_ratio else "⚠️预警"
                    lines.append(f"| {name} | {ratio:.2f} | {level} |")
                lines.append("")
            else:
                lines.append("当日无强赎预警信号。")
                lines.append("")

        lines.append("### 4.2 下修候选")
        lines.append("")
        down_rev = []
        for sig_list in clause_results.get("downward_revision", []):
            if sig_list.triggered and sig_list.detail.get("probability", 0) >= 0.15:
                down_rev.append(sig_list)
        if down_rev:
            lines.append(f"共 {len(down_rev)} 只转债存在一定下修可能。")
            lines.append("")
        else:
            lines.append("当日无明显下修候选。")
            lines.append("")

        # ── 5. 正股联动精选 ──
        lines.append("## 5. 正股联动精选")
        lines.append("")
        lines.append("基于正股动量与期权 Delta 维度的股债联动机会：")
        lines.append("")

        # Use detector results when available; fall back to simple premium filter
        linkage_bonds: list[dict] = []
        if link_results and vol_results:
            bs_d_list = vol_results.get("bs_delta", [])
            iv_list = vol_results.get("iv_percentile", [])
            for idx, row in cb_df.iterrows():
                try:
                    mom = link_results["stock_momentum"][idx]
                    delta = link_results["delta"][idx]
                    dev = link_results["cb_stock_deviation"][idx]
                    bs_d = bs_d_list[idx] if idx < len(bs_d_list) else None
                    iv_perc = iv_list[idx] if idx < len(iv_list) else None

                    # stock_momentum triggered AND (delta OR bs_delta triggered)
                    condition1 = mom.triggered and (
                        delta.triggered or (bs_d is not None and bs_d.triggered)
                    )
                    # cb_stock_deviation triggered (bullish lag)
                    condition2 = dev.triggered and dev.detail.get("signal") == "bullish_lag"

                    if condition1 or condition2:
                        name = row.get("bond_name", row.get("转债名称", ""))
                        sig_parts = []
                        if mom.triggered:
                            sig_parts.append("正股动量")
                        if delta.triggered:
                            sig_parts.append(f"Delta={delta.strength:.2f}")
                        if bs_d is not None and bs_d.triggered:
                            sig_parts.append(f"BS Delta={bs_d.strength:.2f}")
                        if dev.triggered:
                            sig_parts.append("股债偏离")
                        linkage_bonds.append({
                            "name": name,
                            "price": float(row.get("cb_price", 0)) or 0,
                            "premium": float(row.get("premium_rate", 0)) or 0,
                            "delta_val": delta.strength if delta else 0,
                            "iv": iv_perc.strength if iv_perc else 0,
                            "summary": "、".join(sig_parts) if sig_parts else "联动信号",
                        })
                except (KeyError, IndexError):
                    continue

        if linkage_bonds:
            linkage_bonds.sort(key=lambda x: abs(x["delta_val"]) + abs(x["premium"] * 0.1), reverse=True)
            shown = linkage_bonds[:10]
            lines.append("| 转债 | 价格 | 溢价率 | Delta | IV | 信号摘要 |")
            lines.append("|------|------|--------|-------|----|----------|")
            for b in shown:
                iv_str = f"{b['iv']:.2f}" if b["iv"] else "—"
                lines.append(
                    f"| {b['name']} | {b['price']:.2f} | {b['premium']:.1f}% | "
                    f"{b['delta_val']:.2f} | {iv_str} | {b['summary']} |"
                )
            lines.append("")
        elif "premium_rate" in cb_df.columns:
            # Fallback: simple premium filter
            arbitrage = cb_df[cb_df["premium_rate"] < -1].head(10)
            if not arbitrage.empty:
                lines.append("当日无联动信号，以下为折价套利机会（负溢价）：")
                lines.append("")
                lines.append("| 转债 | 价格 | 溢价率 | 转股价值 | 正股 |")
                lines.append("|------|------|--------|----------|------|")
                for _, row in arbitrage.iterrows():
                    name = row.get("bond_name", row.get("转债名称", ""))
                    price = float(row.get("cb_price", 0)) or 0
                    prem = float(row.get("premium_rate", 0)) or 0
                    cv = float(row.get("conversion_value", 0)) or 0
                    stock = row.get("stock_name", row.get("正股名称", ""))
                    lines.append(f"| {name} | {price:.2f} | {prem:.1f}% | {cv:.2f} | {stock} |")
                lines.append("")
            else:
                lines.append("当日无显著股债联动信号，也无折价套利机会。")
                lines.append("")
        else:
            lines.append("当日无显著股债联动信号。")
            lines.append("")

        # ── 6. 行业分布 ──
        lines.append("## 6. 行业分布")
        lines.append("")

        # Build stock_code → industry mapping from stock_info
        industry_map: dict[str, str] = {}
        if stock_info is not None and not stock_info.empty:
            for _, srow in stock_info.iterrows():
                sym = str(srow.get("symbol", ""))
                ind = str(srow.get("industry", "未知"))
                # Strip exchange suffix for matching
                if "." in sym:
                    sym = sym.split(".")[0]
                industry_map[sym] = ind

        # Count industries across all CBs
        from collections import Counter
        industry_counts: Counter = Counter()
        for _, row in cb_df.iterrows():
            sc = str(row.get("stock_code", row.get("正股代码", ""))).strip().zfill(6)
            ind = industry_map.get(sc, "未知")
            industry_counts[ind] += 1

        if industry_counts:
            lines.append("| 行业 | 转债数量 | 占比 |")
            lines.append("|------|----------|------|")
            total = sum(industry_counts.values())
            for ind, count in industry_counts.most_common():
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"| {ind} | {count} | {pct:.1f}% |")
            lines.append("")
        else:
            lines.append("暂无行业分布数据。")
            lines.append("")

        # ── 7. 信用风险告警 ──
        lines.append("## 7. 信用风险告警")
        lines.append("")
        # Pre-build bond name list matching risk_filter iteration order
        bond_names = [
            str(row.get("bond_name", row.get("转债名称", "")))
            for _, row in cb_df.iterrows()
        ]
        credit_alerts = []
        for i, sig in enumerate(struct_results.get("credit_risk", [])):
            if sig.triggered and sig.strength < 0:
                name = bond_names[i] if i < len(bond_names) else "—"
                credit_alerts.append((name, sig))
        if credit_alerts:
            lines.append(f"| 转债 | 风险描述 |")
            lines.append(f"|------|----------|")
            for name, sig in credit_alerts:
                lines.append(f"| {name} | {sig.summary} |")
            lines.append("")
        else:
            lines.append("当日无明显信用风险信号。")
            lines.append("")

        # ── 8. 综合评分排名 ──
        lines.append("## 8. 综合评分排名 (Top {})".format(len(top)))
        lines.append("")
        lines.append("| 排名 | 转债 | 价格 | 溢价率 | 双低值 | 估值 | 条款 | 联动 | 结构 | 总分 | 等级 | 触发信号 |")
        lines.append("|------|------|------|--------|--------|------|------|------|------|------|------|----------|")
        for s in top:
            signals_str = "、".join(s.triggered_signals[:3]) if s.triggered_signals else "—"
            lines.append(
                f"| {s.rank} | {s.bond_name}({s.bond_code}) | {s.cb_price:.2f} | "
                f"{s.premium_rate:.1f}% | {s.double_low:.1f} | "
                f"{s.valuation_score:.0f} | {s.clause_score:.0f} | {s.linkage_score:.0f} | "
                f"{s.structure_score:.0f} | **{s.composite_score:.0f}** | {s.grade} | {signals_str} |"
            )
        lines.append("")
        if self._config.get("scoring", {}).get("industry_neutralize", False):
            lines.append("> 注：以上排名已进行行业中性化处理（行业内 Z-score 标准化）。")
            lines.append("")

        # ── 9. AI 逐券研判 ──
        lines.append("## 9. AI 逐券研判")
        lines.append("")
        if llm_analyses:
            for sr in top:
                code = sr.bond_code
                if code in llm_analyses and llm_analyses[code]:
                    lines.append(f"### {sr.bond_name}（{code}）")
                    lines.append("")
                    lines.append(llm_analyses[code])
                    lines.append("")
        else:
            lines.append("> （LLM 分析结果将由 analyst 模块填充，使用 --llm 参数启用）")
            lines.append("")

        # ── 10. 数据溯源 ──
        lines.append("## 10. 数据溯源")
        lines.append("")
        lines.append("| 数据类别 | 来源 | 状态 |")
        lines.append("|----------|------|------|")
        lines.append("| 可转债行情 | AKShare 集思录 (bond_cb_jsl) | ✅ |")
        lines.append("| 正股K线 | Pandadata get_stock_daily | ✅ |")
        lines.append("| 正股信息 | Pandadata get_stock_detail | ✅ |")
        lines.append("| 条款数据 | 基于行情数据计算 | ✅ |")
        lines.append("| LLM分析 | DeepSeek / Claude API | ✅ |")
        lines.append("")

        # ── 11. Strategy Backtest ──
        if backtest_result is not None:
            bt = backtest_result
            lines.append("---")
            lines.append("")
            lines.append("## 11. 策略回测")
            lines.append("")

            ic = bt.ic_summary
            has_ic = ic.get("num_periods", 0) > 0
            has_quintile = bool(bt.quintile_returns)

            if not has_ic and not has_quintile:
                lines.append(f"> 回测数据不足：{bt.num_periods} 个交易日记录，"
                             f"但缺少足够的前向收益数据（需 forward_days={bt.forward_days} 天）。"
                             f"随着每日运行将自动积累。")
                lines.append("")
            else:
                # -- IC Decay table --
                ic_decay = getattr(bt, "ic_decay", {}) or {}
                if ic_decay:
                    lines.append("### IC 衰减分析")
                    lines.append("")
                    lines.append("| 前瞻天数 | 平均Rank IC | IC标准差 | IC IR | 胜率 | 样本期数 |")
                    lines.append("|----------|------------|---------|-------|------|---------|")
                    max_ic = 0.0
                    for horizon in sorted(ic_decay.keys()):
                        d = ic_decay[horizon]
                        mean_ic = d.get("mean_ic", 0)
                        max_ic = max(max_ic, abs(mean_ic))
                        lines.append(
                            f"| {horizon}D | {mean_ic:.4f} | {d.get('ic_std', 0):.4f} | "
                            f"{d.get('ic_ir', 0):.2f} | {d.get('ic_win_rate', 0):.1%} | "
                            f"{d.get('num_periods', 0)} |"
                        )
                    lines.append("")

                    # ASCII sparkline
                    if max_ic > 0:
                        lines.append("IC 衰减曲线 (mean_IC):")
                        lines.append("```")
                        bar_max = 20
                        for horizon in sorted(ic_decay.keys()):
                            mean_ic = ic_decay[horizon].get("mean_ic", 0)
                            bar_len = max(1, int(abs(mean_ic) / max_ic * bar_max))
                            bar = "█" * bar_len
                            lines.append(f"  {horizon:>3}D: {bar} {mean_ic:.4f}")
                        lines.append("```")
                        lines.append("")
                elif has_ic:
                    # Single IC table (backward compat — no decay data)
                    lines.append("### IC 分析")
                    lines.append("")
                    lines.append("| 指标 | 数值 |")
                    lines.append("|------|------|")
                    lines.append(f"| 回测区间 | {bt.start_date} ~ {bt.end_date} ({ic.get('num_periods', 0)} 期) |")
                    lines.append(f"| 平均 Rank IC | {ic.get('mean_ic', 0):.4f} |")
                    lines.append(f"| IC 标准差 | {ic.get('ic_std', 0):.4f} |")
                    lines.append(f"| IC IR | {ic.get('ic_ir', 0):.2f} |")
                    lines.append(f"| IC 胜率 | {ic.get('ic_win_rate', 0):.1%} |")
                    lines.append("")

                # -- Stratified backtest with risk metrics --
                if bt.quintile_returns:
                    n_q = len(bt.quintile_returns)
                    risk_metrics = getattr(bt, "quintile_risk_metrics", {}) or {}
                    has_metrics = bool(risk_metrics)

                    lines.append("### 分层回测 (按综合评分分 " + str(n_q) + " 组)")
                    lines.append("")
                    if has_metrics:
                        lines.append("| 分位 | 累计收益 | 年化收益 | 年化波动 | Sharpe | 最大回撤 |")
                        lines.append("|------|----------|----------|----------|--------|----------|")
                    else:
                        lines.append("| 分位 | 累计收益 |")
                        lines.append("|------|----------|")
                    for q in sorted(bt.quintile_returns.keys()):
                        label = f"Q{q}" + (" (最高分)" if q == 1 else " (最低分)" if q == n_q else "")
                        if has_metrics and q in risk_metrics:
                            m = risk_metrics[q]
                            lines.append(
                                f"| {label} | {bt.quintile_returns[q]:.2%} | "
                                f"{m.get('annualized_return', 0):.2%} | "
                                f"{m.get('annualized_volatility', 0):.2%} | "
                                f"{m.get('sharpe_ratio', 0):.2f} | "
                                f"{m.get('max_drawdown', 0):.2%} |"
                            )
                        else:
                            lines.append(f"| {label} | {bt.quintile_returns[q]:.2%} |")
                    if 1 in bt.quintile_returns and n_q in bt.quintile_returns:
                        spread = bt.quintile_returns[1] - bt.quintile_returns[n_q]
                        lines.append(f"| Q1-Q{n_q} 多空 | {spread:.2%} |")
                    lines.append("")

                # -- Benchmark comparison --
                bm_comp = getattr(bt, "benchmark_comparison", {}) or {}
                if bm_comp and "error" not in bm_comp:
                    lines.append("### 基准对比 (vs 中证转债指数 000832)")
                    lines.append("")
                    lines.append("| 指标 | 策略Q1 | 基准指数 | 超额 |")
                    lines.append("|------|--------|----------|------|")
                    s_cum = bm_comp.get("strategy_cumulative", 0)
                    b_cum = bm_comp.get("benchmark_cumulative", 0)
                    excess = bm_comp.get("excess_return", 0)
                    lines.append(f"| 累计收益 | {s_cum:.2%} | {b_cum:.2%} | {excess:.2%} |")
                    ir_val = bm_comp.get("information_ratio", 0)
                    te_val = bm_comp.get("tracking_error", 0)
                    wr_val = bm_comp.get("win_rate", 0)
                    lines.append(f"| 信息比率 | — | — | {ir_val:.2f} |")
                    lines.append(f"| 跟踪误差 | — | — | {te_val:.1%} |")
                    lines.append(f"| 跑赢胜率 | — | — | {wr_val:.0%} |")
                    lines.append(
                        f"> 基于 {bm_comp.get('n_periods', 0)} 期重叠数据。"
                        f"Q1代表最高分组合，基准为中证转债指数(000832)。"
                    )
                    lines.append("")
                elif bm_comp and "error" in bm_comp:
                    lines.append("### 基准对比 (vs 中证转债指数 000832)")
                    lines.append("")
                    lines.append(f"> 基准对比数据不足（{bm_comp.get('error', 'unknown')}），"
                                 f"需积累更多历史数据后分析。")
                    lines.append("")

                # -- Factor attribution --
                factor_attr = getattr(bt, "factor_attribution", {}) or {}
                if factor_attr and "error" not in factor_attr:
                    premiums = factor_attr.get("factor_premiums", {})
                    t_stats = factor_attr.get("t_stats", {})
                    nw_t_stats = factor_attr.get("nw_t_stats", {})
                    if premiums:
                        lines.append("### 因子收益归因 (Fama-MacBeth)")
                        lines.append("")
                        if nw_t_stats:
                            lines.append(f"| 因子 | 风险溢价 | t统计量 | NW-t | 显著性 |")
                            lines.append(f"|------|----------|--------|------|--------|")
                        else:
                            lines.append(f"| 因子 | 风险溢价 | t统计量 | 显著性 |")
                            lines.append(f"|------|----------|---------|--------|")
                        for f_name, premium in premiums.items():
                            if f_name == "intercept":
                                continue
                            t_val = t_stats.get(f_name, 0)
                            nw_val = nw_t_stats.get(f_name, t_val) if nw_t_stats else t_val
                            # Use NW t-stat for significance marking when available
                            sig = ""
                            if abs(nw_val) >= 2.58:
                                sig = "***"
                            elif abs(nw_val) >= 1.96:
                                sig = "**"
                            elif abs(nw_val) >= 1.64:
                                sig = "*"
                            label_map = {
                                "valuation_score": "估值因子",
                                "clause_score": "条款因子",
                                "linkage_score": "联动因子",
                                "structure_score": "结构因子",
                            }
                            label = label_map.get(f_name, f_name)
                            if nw_t_stats:
                                lines.append(f"| {label} | {premium:.6f} | {t_val:.2f} | {nw_val:.2f} | {sig} |")
                            else:
                                lines.append(f"| {label} | {premium:.6f} | {t_val:.2f} | {sig} |")
                        nw_note = " (Newey-West HAC)" if nw_t_stats else ""
                        lines.append(
                            f"> 基于 {factor_attr.get('n_dates', 0)} 个交易日的横截面回归。"
                            f"平均R²={factor_attr.get('r_squared_avg', 0):.4f}。"
                            f"NW-t{nw_note}用于显著性标记。"
                            f"* p<0.1, ** p<0.05, *** p<0.01"
                        )
                        lines.append("")
                elif factor_attr and "error" in factor_attr:
                    lines.append("### 因子收益归因 (Fama-MacBeth)")
                    lines.append("")
                    lines.append(f"> 因子归因数据不足（{factor_attr.get('error', 'unknown')}），"
                                 f"需积累更多历史数据后分析。")
                    lines.append("")

                # -- Cost model note --
                cost_model = getattr(bt, "cost_model", {}) or {}
                if cost_model.get("enabled", False):
                    lines.append(f"> **交易成本假设**: 印花税 {cost_model.get('stamp_duty', 0.0005):.2%}(卖出单边)"
                                 f" + 佣金 {cost_model.get('commission', 0.0001):.2%}(双边)"
                                 f" + 滑点 {cost_model.get('slippage', 0.0001):.2%}，"
                                 f"合计约 {cost_model.get('round_trip_cost', 0.0009):.2%} round-trip。")
                    if cost_model.get('min_daily_turnover', 0) > 0:
                        lines.append(f"> 流动性过滤: 日成交额 >= {cost_model['min_daily_turnover']:.0f}万元"
                                     f" | 涨跌停过滤: {'已启用' if cost_model.get('filter_limit_hit') else '未启用'}")
                    lines.append("")

                # -- Weight calibration --
                weight_cal = getattr(bt, "weight_calibration", {}) or {}
                if weight_cal and "error" not in weight_cal:
                    lines.append("### 权重校准 (Grid Search)")
                    lines.append("")
                    opt_w = weight_cal.get("optimal_weights", {})
                    lines.append("| 维度 | 原始权重 | 最优权重 |")
                    lines.append("|------|----------|----------|")
                    scoring_cfg = self._config.get("scoring", {})
                    orig = {
                        "valuation": scoring_cfg.get("valuation_weight", 0.40),
                        "clause": scoring_cfg.get("clause_weight", 0.30),
                        "linkage": scoring_cfg.get("linkage_weight", 0.20),
                        "structure": scoring_cfg.get("structure_weight", 0.10),
                    }
                    for dim in ["valuation", "clause", "linkage", "structure"]:
                        ow = opt_w.get(dim, orig.get(dim, 0))
                        lines.append(f"| {dim} | {orig.get(dim, 0):.2f} | {ow:.4f} |")
                    lines.append("")
                    train_ic = weight_cal.get("train_ic", 0)
                    valid_ic = weight_cal.get("valid_ic", 0)
                    n_combos = weight_cal.get("total_combinations", 0)
                    lines.append(f"| 指标 | 数值 |")
                    lines.append(f"|------|------|")
                    lines.append(f"| 训练集IC | {train_ic:.4f} |")
                    lines.append(f"| 验证集IC | {valid_ic:.4f} |")
                    lines.append(f"| 搜索组合数 | {n_combos} |")
                    lines.append(
                        f"> 在{n_combos}个权重组合中搜索，步长={self._config.get('backtest', {}).get('calibration', {}).get('dimension_step', 0.05)}。"
                        f"训练集({weight_cal.get('train_dates', '?')}期) / 验证集({weight_cal.get('valid_dates', '?')}期)。"
                    )
                    lines.append("")
                elif weight_cal and "error" in weight_cal:
                    lines.append("### 权重校准 (Grid Search)")
                    lines.append("")
                    lines.append(f"> 权重校准失败：{weight_cal.get('error', 'unknown')}")
                    lines.append("")

                # -- Dynamic weights --
                dyn_weights = getattr(bt, "dynamic_weights", {}) or {}
                if dyn_weights and "error" not in dyn_weights:
                    lines.append("### 动态IC权重调整")
                    lines.append("")
                    lines.append("| 信号 | 静态权重 | 动态权重 | 变化 |")
                    lines.append("|------|----------|----------|------|")
                    base_w = self._config.get("detector_weights", {})
                    for key, dw in sorted(dyn_weights.items()):
                        bw = base_w.get(key, 0)
                        change = dw - bw
                        sign = "+" if change > 0 else ""
                        lines.append(f"| {key} | {bw:.2f} | {dw:.4f} | {sign}{change:.4f} |")
                    lines.append("")
                    dw_cfg = self._config.get("backtest", {}).get("dynamic_weights", {})
                    lines.append(
                        f"> 基于滚动IC（窗口={dw_cfg.get('rolling_window', 20)}期）调整信号权重。"
                        f"IC ≤ {dw_cfg.get('floor_ic', 0)} 的信号被降权为零。"
                    )
                    lines.append("")

                if bt.summary:
                    lines.append(f"> {bt.summary}")
                    lines.append("")

        # ── 12. Signal Correlation Matrix ──
        if signal_correlation:
            lines.append("---")
            lines.append("")
            lines.append("## 12. 信号相关性矩阵")
            lines.append("")
            if "error" in signal_correlation:
                lines.append(f"> 数据不足，无法计算相关性矩阵"
                             f"（{signal_correlation.get('n_bonds', 0)} 只转债）。")
                lines.append("")
            else:
                top_pairs = signal_correlation.get("top_pairs", [])
                n_bonds = signal_correlation.get("n_bonds", 0)
                if top_pairs:
                    lines.append("### 高相关信号对 (|r| > 0.5)")
                    lines.append("")
                    lines.append("| 信号A | 信号B | Spearman ρ |")
                    lines.append("|-------|-------|------------|")
                    for pair in top_pairs:
                        lines.append(
                            f"| {pair['signal_a']} | {pair['signal_b']} | "
                            f"{pair['correlation']:.3f} |"
                        )
                    lines.append(
                        f"> 共 {n_bonds} 只转债参与计算。高相关信号对可能需考虑降权或合并以减少多重共线性。"
                    )
                    lines.append("")
                else:
                    lines.append(f"> 未发现高相关信号对 (|r| > 0.5)，"
                                 f"共 {n_bonds} 只转债参与计算。")
                    lines.append("")

        # ── Disclaimer ──
        lines.append("---")
        lines.append("")
        lines.append("## 免责声明")
        lines.append("")
        lines.append("本报告仅供研究参考，**不构成任何投资建议**。可转债交易存在市场风险、信用风险和条款风险，投资者应独立判断并承担交易风险。过往表现不代表未来收益。")
        lines.append("")
        lines.append(f"> Generated by skill-cb-analyzer v1.6.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        return "\n".join(lines)

    # -- JSON -------------------------------------------------------

    def _render_json(
        self, trade_date: str,
        cb_df: pd.DataFrame,
        ranked: list[ScoreResult],
        all_results: list[ScoreResult],
        llm_analyses: dict[str, str] | None = None,
    ) -> str:
        excluded = [s for s in all_results if s.excluded]
        active = [s for s in all_results if not s.excluded]

        output = {
            "trade_date": trade_date,
            "scan_time": datetime.now().isoformat(),
            "total_cb": len(cb_df),
            "selected_count": len(active),
            "excluded_count": len(excluded),
            "data_provenance": {
                "cb_quote": {"source": "AKShare bond_cb_jsl (集思录)", "status": "real"},
                "stock_kline": {"source": "Pandadata get_stock_daily", "status": "real"},
                "stock_info": {"source": "Pandadata get_stock_detail", "status": "real"},
                "llm": {"source": "DeepSeek / Claude API", "status": "real"},
            },
            "market_summary": {
                "avg_price": round(_safe_float(cb_df["cb_price"].mean()) if "cb_price" in cb_df.columns else 0, 2),
                "avg_premium_rate": round(_safe_float(cb_df["premium_rate"].mean()) if "premium_rate" in cb_df.columns else 0, 2),
                "avg_ytm": round(_safe_float(cb_df["ytm"].mean() * 100) if "ytm" in cb_df.columns else 0, 2),
            },
            "excluded": [
                {
                    "bond_code": s.bond_code,
                    "bond_name": s.bond_name,
                    "reason": s.exclude_reason,
                }
                for s in excluded
            ],
            "rankings": [
                {
                    "rank": s.rank,
                    "bond_code": s.bond_code,
                    "bond_name": s.bond_name,
                    "stock_code": s.stock_code,
                    "stock_name": s.stock_name,
                    "cb_price": s.cb_price,
                    "premium_rate": s.premium_rate,
                    "conversion_value": s.conversion_value,
                    "double_low": s.double_low,
                    "ytm": s.ytm,
                    "valuation_score": s.valuation_score,
                    "clause_score": s.clause_score,
                    "linkage_score": s.linkage_score,
                    "structure_score": s.structure_score,
                    "composite_score": s.composite_score,
                    "neutralized_score": s.neutralized_score,
                    "risk_penalty": s.risk_penalty,
                    "grade": s.grade,
                    "triggered_signals": s.triggered_signals,
                    "risk_flags": s.risk_flags,
                }
                for s in ranked
            ],
            "llm_analyses": llm_analyses or {},
            "disclaimer": "本报告仅供研究参考，不构成任何投资建议。",
        }

        return json.dumps(output, ensure_ascii=False, indent=2)
