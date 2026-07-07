"""End-to-end pipeline orchestrator for CB daily analysis.

Data flow:
    1. Determine trade date
    2. Check cache / fetch fresh data
    3. Compute CB metrics (bond_calculator)
    4. Run 17 signal detectors (valuation + clause + linkage + risk)
    5. Score and rank all CBs
    6. Generate report (MD + JSON)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from core._types import SignalResult
from core.cache import CacheManager
from core.data_fetcher import DataFetcher, _load_config
from core.bond_calculator import compute_cb_metrics
from core.data_quality import validate_cb_data
from core.history_store import HistoryStore
from core.valuation import ValuationDetector
from core.clause_monitor import ClauseMonitor
from core.stock_linkage import StockLinkageDetector
from core.options_pricing import VolatilityDetector
from core.risk_filter import RiskFilter
from core.scorer import ScoreResult, Scorer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Output of a pipeline run."""
    trade_date: str = ""
    total_cb: int = 0
    selected_count: int = 0
    ranked_stocks: list[dict] = field(default_factory=list)
    md_path: str = ""
    json_path: str = ""
    errors: list[str] = field(default_factory=list)
    backtest: any = None  # BacktestResult | None


class CBPipeline:
    """End-to-end convertible bond analysis pipeline."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_config()
        self._fetcher = DataFetcher(self._config)
        self._cache = CacheManager("cache")
        self._history = HistoryStore("data")

        # Detectors
        self._valuation = ValuationDetector(self._config)
        self._clause = ClauseMonitor(self._config)
        self._linkage = StockLinkageDetector(self._config)
        self._volatility = VolatilityDetector(self._config)
        self._risk = RiskFilter(self._config)

        # Config validation (non-blocking)
        from core.config_validator import validate_config
        _cfg_warnings = validate_config(self._config)
        if _cfg_warnings:
            for w in _cfg_warnings:
                logger.warning("Config: %s", w)

        # Scoring
        self._scorer = Scorer(self._config)

        # State cache for --llm report regeneration (avoids double pipeline run)
        self._last_trade_date: str = ""
        self._last_cb_df: pd.DataFrame | None = None
        self._last_ranked: list[ScoreResult] = []
        self._last_score_results: list[ScoreResult] = []
        self._last_val_results: dict = {}
        self._last_clause_results: dict = {}
        self._last_link_results: dict = {}
        self._last_vol_results: dict = {}
        self._last_struct_results: dict = {}
        self._last_stock_info: pd.DataFrame | None = None
        self._last_signal_correlation: dict = {}

    def run(
        self,
        trade_date: Optional[str] = None,
        use_cache: bool = True,
        top_n: int = 20,
        summary_mode: bool = False,
        llm_analyses: dict[str, str] | None = None,
    ) -> PipelineResult:
        """Run the full CB analysis pipeline.

        Args:
            trade_date: Target date (YYYYMMDD). Default: latest trading day.
            use_cache: If True, skip fetch when cached data exists.
            top_n: Number of top CBs in report.
            summary_mode: Shorter LLM output.
            llm_analyses: Optional dict of bond_code → LLM analysis text.

        Returns:
            PipelineResult with ranked stocks and output paths.
        """
        t0 = time.time()
        errors: list[str] = []

        # 1. Determine trade date
        try:
            self._fetcher.init_api()
            if trade_date is None:
                trade_date = self._fetcher.get_last_trade_date()
        except RuntimeError as e:
            errors.append(f"API 初始化失败: {e}")
            return PipelineResult(errors=errors)
        except Exception as e:
            errors.append(f"API 调用异常: {e}")
            logger.exception("Unexpected error during API init")
            return PipelineResult(errors=errors)

        logger.info("Target trade date: %s", trade_date)

        # Check trading day
        try:
            is_trade = self._fetcher.is_trading_day(trade_date)
        except Exception as e:
            logger.warning("is_trading_day check failed (non-fatal): %s", e)
            is_trade = False  # Assume non-trading day on error to avoid stale runs

        if not is_trade:
            msg = f"{trade_date} 休市，无需分析。"
            logger.info(msg)
            return PipelineResult(trade_date=trade_date, errors=[msg])

        # 2. Data acquisition
        cb_df, stock_kline, stock_info, cb_stock_map = self._load_data(
            trade_date, use_cache
        )

        if cb_df.empty:
            errors.append("可转债行情数据为空")
            return PipelineResult(trade_date=trade_date, errors=errors)

        # 3. Compute CB metrics
        logger.info("Computing CB metrics ...")
        cb_df = compute_cb_metrics(cb_df, stock_kline, trade_date)

        # 3a. Data quality validation (non-blocking)
        cb_df, quality_warnings = validate_cb_data(cb_df)
        if quality_warnings:
            errors.extend(quality_warnings)

        # 3b. Save current snapshot to history (for future A4/D2 use)
        try:
            self._history.save_snapshot(trade_date, cb_df)
        except Exception as e:
            logger.warning("History save failed (non-fatal): %s", e)

        # 3c. Build historical data dicts for detectors
        hist_df = self._history.load_history()
        premium_history: dict[str, pd.Series] = {}
        previous_balances: dict[str, float] = {}
        if not hist_df.empty:
            bond_col = next((c for c in ["bond_code", "转债代码", "代码"] if c in cb_df.columns), None)
            if bond_col is not None:
                for bond in cb_df[bond_col].astype(str).unique():
                    premium_history[bond] = self._history.get_premium_history(bond, df=hist_df)
                    prev_bal = self._history.get_previous_balance(bond, trade_date, df=hist_df)
                    if prev_bal is not None:
                        previous_balances[bond] = prev_bal

        # 4. Run detectors
        logger.info("Running signal detectors ...")

        # Valuation (A4 now receives premium history)
        val_results = self._valuation.run_all(cb_df, premium_history=premium_history)

        # Clause (B1/B3 now use history for consecutive-day tracking)
        clause_results = self._clause.run_all(cb_df, trade_date,
                                               history_store=self._history)

        # Stock linkage
        # Compute daily stock changes from K-line
        stock_changes = {}
        if not stock_kline.empty:
            date_col = next((c for c in ["date", "trade_date"] if c in stock_kline.columns), None)
            sym_col = "symbol" if "symbol" in stock_kline.columns else None
            if date_col and sym_col:
                sorted_k = stock_kline.sort_values(date_col)
                for sym, grp in sorted_k.groupby(sym_col):
                    if len(grp) >= 2:
                        close = grp["close"].astype(float)
                        stock_changes[sym] = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100

        link_results = self._linkage.run_all(cb_df, stock_kline, cb_stock_map, stock_changes)

        # Volatility / options (extends stock-linkage dimension)
        vol_results = self._volatility.run_all(cb_df, stock_kline)

        # Risk / structure (D2 now receives previous balance data)
        struct_results = self._risk.run_all(cb_df, stock_info,
                                            previous_balances=previous_balances)

        # 5. Score each CB
        logger.info("Scoring and ranking ...")
        score_results: list[ScoreResult] = []
        all_bond_signals: list[dict[str, SignalResult]] = []

        for idx, row in cb_df.iterrows():
            # Check exclusion
            excluded = False
            exclude_reason = ""
            if self._risk.is_redemption_announced(row):
                excluded = True
                exclude_reason = "已公告强赎，面临赎回风险"
            elif self._risk.exclude_st:
                # Use pre-computed credit_risk result from struct_results (BUG-2 fix)
                credit_signal = struct_results["credit_risk"][idx]
                if credit_signal.triggered:
                    severity = credit_signal.detail.get("severity", 0)
                    if severity >= 0.5:
                        excluded = True
                        exclude_reason = "信用风险过高（评级/ST）"

            if excluded:
                sr = ScoreResult(
                    bond_code=str(row.get("bond_code", row.get("转债代码", ""))),
                    bond_name=str(row.get("bond_name", row.get("转债名称", ""))),
                    excluded=True,
                    exclude_reason=exclude_reason,
                )
                score_results.append(sr)
                continue

            # Collect per-signal results for this bond
            bond_signals_val = {
                "double_low": val_results["double_low"][idx],
                "ytm_defense": val_results["ytm_defense"][idx],
                "bond_floor": val_results["bond_floor"][idx],
                "premium_percentile": val_results["premium_percentile"][idx],
            }
            bond_signals_clause = {
                "redemption": clause_results["redemption"][idx],
                "downward_revision": clause_results["downward_revision"][idx],
                "putback": clause_results["putback"][idx],
                "maturity": clause_results["maturity"][idx],
            }
            bond_signals_link = {
                "stock_momentum": link_results["stock_momentum"][idx],
                "cb_stock_deviation": link_results["cb_stock_deviation"][idx],
                "delta": link_results["delta"][idx],
                "stock_pattern": link_results["stock_pattern"][idx],
                "iv_percentile": vol_results["iv_percentile"][idx],
                "hv_iv_divergence": vol_results["hv_iv_divergence"][idx],
                "vol_expansion": vol_results["vol_expansion"][idx],
                "bs_delta": vol_results["bs_delta"][idx],
            }
            bond_signals_struct = {
                "volume": struct_results["volume"][idx],
                "balance_trend": struct_results["balance_trend"][idx],
                "credit_risk": struct_results["credit_risk"][idx],
                "liquidity": struct_results["liquidity"][idx],
            }

            # Composites
            val_comp = self._valuation.composite_score(bond_signals_val)
            clause_comp = self._clause.composite_score(bond_signals_clause)
            link_comp = self._linkage.composite_score(bond_signals_link)
            struct_comp = self._risk.composite_score(bond_signals_struct)

            # Risk penalty
            risk_pen = self._risk.risk_penalty(bond_signals_struct)

            # Collect triggered and risk flags
            all_signals = {
                **bond_signals_val, **bond_signals_clause,
                **bond_signals_link, **bond_signals_struct,
            }
            triggered = [sig.label for sig in all_signals.values()
                         if sig.triggered and sig.strength > 0]
            risk_flags = [sig.summary for sig in all_signals.values()
                          if sig.triggered and sig.strength < 0]

            # Collect for signal correlation analysis
            if not excluded:
                all_bond_signals.append(all_signals)

            sr = self._scorer.compute(
                idx, row,
                bond_signals_val, bond_signals_clause,
                bond_signals_link, bond_signals_struct,
                risk_pen,
                val_comp, clause_comp, link_comp, struct_comp,
                triggered, risk_flags,
            )
            score_results.append(sr)

        # Rank (non-excluded only)
        active = [s for s in score_results if not s.excluded]
        ranked = self._scorer.rank(
            active,
            stock_info=stock_info,
            industry_neutralize=self._config.get("scoring", {}).get("industry_neutralize", False),
        )

        # Signal correlation analysis
        signal_correlation = self._compute_signal_correlation(all_bond_signals)

        # Store intermediate state for --llm report regeneration (P2.1)
        self._last_trade_date = trade_date
        self._last_cb_df = cb_df
        self._last_ranked = ranked
        self._last_score_results = score_results
        self._last_val_results = val_results
        self._last_clause_results = clause_results
        self._last_link_results = link_results
        self._last_vol_results = vol_results
        self._last_struct_results = struct_results
        self._last_stock_info = stock_info
        self._last_signal_correlation = signal_correlation

        # 6. Generate report
        logger.info("Generating report ...")
        from core.reporter import Reporter
        reporter = Reporter(self._config)
        md_path, json_path = reporter.generate(
            trade_date, cb_df, ranked, score_results,
            val_results, clause_results, link_results, struct_results,
            top_n,
            stock_info=stock_info,
            llm_analyses=llm_analyses or {},
            vol_results=vol_results,
            signal_correlation=signal_correlation,
        )

        elapsed = time.time() - t0
        logger.info("Pipeline complete in %.1fs: %d/%d CBs ranked",
                     elapsed, len(ranked), len(cb_df))

        # Convert top N to dicts for CLI display + LLM analysis
        top_dicts = [s.to_summary_dict() for s in ranked[:top_n]]

        return PipelineResult(
            trade_date=trade_date,
            total_cb=len(cb_df),
            selected_count=len(ranked),
            ranked_stocks=top_dicts,
            md_path=md_path,
            json_path=json_path,
            errors=errors,
        )

    # -- Internal ---------------------------------------------------

    def _load_data(
        self, trade_date: str, use_cache: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Load data from cache or fetch fresh."""
        if use_cache and self._cache.has(trade_date):
            cb_df, stock_kline, stock_info = self._cache.load(trade_date)
            # Rebuild CB-stock map from CB data
            cb_stock_map = {}
            stock_col = next((c for c in ["stock_code", "正股代码"] if c in cb_df.columns), None)
            bond_col = next((c for c in ["代码", "bond_code", "转债代码"] if c in cb_df.columns), None)
            if stock_col and bond_col:
                for _, row in cb_df.iterrows():
                    bond = str(row.get(bond_col, ""))
                    stock = str(row.get(stock_col, ""))
                    if bond and stock:
                        from core.exchange_utils import resolve_exchange_suffix
                        stock = resolve_exchange_suffix(stock)
                        cb_stock_map[bond] = stock
            if not cb_df.empty:
                return cb_df, stock_kline, stock_info, cb_stock_map

        # Fetch fresh
        logger.info("Fetching fresh data for %s ...", trade_date)
        cb_df, stock_kline, stock_info, cb_stock_map = self._fetcher.fetch_all_data(trade_date)

        # Save to cache
        try:
            fetch_time = datetime.now().isoformat()
            self._cache.save(trade_date, cb_df, stock_kline, stock_info, fetch_time)
        except Exception as e:
            logger.warning("Cache save failed (non-fatal): %s", e)

        return cb_df, stock_kline, stock_info, cb_stock_map

    def _compute_signal_correlation(
        self,
        all_signal_dicts: list[dict[str, SignalResult]],
    ) -> dict:
        """Compute pairwise Spearman correlation of signal strengths.

        Args:
            all_signal_dicts: List of per-bond signal dicts (21 signals each).

        Returns:
            {"top_pairs": [...], "n_bonds": N} or {"error": "...", "n_bonds": N}.
        """
        if len(all_signal_dicts) < 5:
            return {"error": "insufficient_data", "n_bonds": len(all_signal_dicts)}

        signal_keys = sorted(all_signal_dicts[0].keys())
        data = {}
        for key in signal_keys:
            data[key] = [sig_dict[key].strength for sig_dict in all_signal_dicts]

        corr_df = pd.DataFrame(data).corr(method="spearman")

        pairs = []
        for i in range(len(signal_keys)):
            for j in range(i + 1, len(signal_keys)):
                r = corr_df.iloc[i, j]
                if abs(r) >= 0.5:
                    pairs.append({
                        "signal_a": signal_keys[i],
                        "signal_b": signal_keys[j],
                        "correlation": round(float(r), 4),
                    })
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        top_n = int(self._config.get("correlation", {}).get("top_n_pairs", 15))
        return {
            "top_pairs": pairs[:top_n],
            "n_bonds": len(all_signal_dicts),
        }

    def cleanup_cache(self, keep_days: int = 30) -> int:
        """Remove old cache directories."""
        return self._cache.clear_old(keep_days)

    def run_backtest(self) -> "BacktestResult":
        """Run backtest on historical scores from output directory.

        Uses the Backtester to compute IC analysis and stratified returns
        from previously generated daily reports.

        Returns:
            BacktestResult with IC summary and quintile returns.
        """
        from core.backtester import Backtester, BacktestResult
        bt = Backtester(self._config)
        return bt.run()

    def regenerate_report(
        self,
        top_n: int = 20,
        llm_analyses: dict[str, str] | None = None,
        backtest_result=None,
    ) -> tuple[str, str]:
        """Re-generate MD+JSON reports from cached pipeline state.

        Use this after running LLM analysis on top bonds to embed the
        analysis text without re-running data fetch, detectors, or scoring.

        Args:
            top_n: Number of top CBs in report.
            llm_analyses: Dict of bond_code -> LLM analysis text.
            backtest_result: Optional BacktestResult for Section 11.

        Returns:
            (md_path, json_path) tuple.

        Raises:
            RuntimeError: If run() has not been called yet.
        """
        if self._last_cb_df is None or not self._last_ranked:
            raise RuntimeError(
                "No cached pipeline state. Call pipeline.run() before regenerate_report()."
            )

        from core.reporter import Reporter
        reporter = Reporter(self._config)
        md_path, json_path = reporter.generate(
            self._last_trade_date,
            self._last_cb_df,
            self._last_ranked,
            self._last_score_results,
            self._last_val_results,
            self._last_clause_results,
            self._last_link_results,
            self._last_struct_results,
            top_n,
            stock_info=self._last_stock_info,
            llm_analyses=llm_analyses or {},
            backtest_result=backtest_result,
            vol_results=self._last_vol_results,
            signal_correlation=self._last_signal_correlation,
        )
        logger.info("Reports regenerated with LLM analyses: %s, %s", md_path, json_path)
        return md_path, json_path
