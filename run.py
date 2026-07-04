#!/usr/bin/env python
"""Convertible Bond Daily Analyzer — CLI entry point.

Usage::

    python run.py                       # Latest trading day
    python run.py --date 20260701       # Specific date
    python run.py --date 20260701 --no-cache  # Force fresh data
    python run.py --top-n 30            # Top 30 in rankings

Output::

    output/YYYY-MM-DD/
        cb_daily_YYYYMMDD.md
        cb_daily_YYYYMMDD.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-7s] %(name)s — %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    for noisy in ["urllib3", "requests", "panda_data", "akshare"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A-share Convertible Bond Daily Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                       # Latest trading day (LLM analysis enabled)
  python run.py --date 20260701       # Specific date
  python run.py --date 20260701 --no-cache   # Force fresh data
  python run.py --top-n 30 --verbose         # Top 30, debug logging
  python run.py --no-llm                     # Skip LLM, rule-based only
        """.strip(),
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Target trade date (YYYYMMDD). Default: latest trading day.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip cache — always fetch fresh data.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of bonds in rankings (default: 20).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug-level logging.")
    parser.add_argument("--cleanup-cache", type=int, default=None, metavar="DAYS",
                        help="Remove cached data older than DAYS days, then exit.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM analysis (rule-based fallback only).")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest analysis on historical scores (IC + stratified returns).")

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("run")

    from core.pipeline import CBPipeline

    pipeline = CBPipeline()

    # --cleanup-cache mode
    if args.cleanup_cache is not None:
        removed = pipeline.cleanup_cache(keep_days=args.cleanup_cache)
        logger.info("Cache cleanup: removed %d directories older than %d days",
                     removed, args.cleanup_cache)
        print(f"Cache cleanup complete: {removed} directories removed.")
        return

    # Normal run
    logger.info("Starting CB daily analyzer ...")
    if args.date:
        logger.info("Target date: %s", args.date)
    else:
        logger.info("Target date: latest trading day (auto-detect)")

    use_cache = not args.no_cache

    if args.output_dir:
        pipeline._config.setdefault("output", {})["dir"] = args.output_dir

    result = pipeline.run(
        trade_date=args.date,
        use_cache=use_cache,
        top_n=args.top_n,
    )

    # Print summary
    print()
    print("=" * 60)
    print(f"  CB Daily Analyzer — {result.trade_date}")
    print("=" * 60)
    print(f"  Total CBs:        {result.total_cb}")
    print(f"  Ranked:           {result.selected_count}")
    print(f"  Top 5:")
    for s in result.ranked_stocks[:5]:
        risk_flags = s.get("risk_flags", [])
        risk_str = f" [! {risk_flags[0]}]" if risk_flags else ""
        print(f"    {s['rank']:>2}. {s['name']} ({s['code']}) "
              f"得分 {s['score']:.0f} | {s['grade']}{risk_str}")
    if len(result.ranked_stocks) > 5:
        print(f"    ... (共 {len(result.ranked_stocks)} 只)")
    print("-" * 60)
    if result.md_path:
        print(f"  Report (MD):      {result.md_path}")
    if result.json_path:
        print(f"  Report (JSON):    {result.json_path}")
    if result.errors:
        for e in result.errors:
            print(f"  [WARN] {e}")
    print("=" * 60)
    print()

    # LLM analysis (default; use --no-llm to skip)
    if not args.no_llm and result.ranked_stocks:
        from core.data_fetcher import _load_config
        from llm.analyst import CBAnalyst
        config = _load_config()
        analyst = CBAnalyst(config)
        llm_top_n = int(config.get("llm", {}).get("top_n", 5))
        top_bonds = result.ranked_stocks[:llm_top_n]
        bonds_data = [
            {
                "bond_name": s.get("name", ""),
                "bond_code": s.get("code", ""),
                "stock_name": s.get("stock_name", ""),
                "stock_code": s.get("stock_code", ""),
                "cb_price": s.get("cb_price", 0),
                "premium_rate": s.get("premium_rate", 0),
                "conversion_value": s.get("conversion_value", 0),
                "double_low": s.get("double_low", 0),
                "ytm": s.get("ytm", 0),
                "signal_summary": "、".join(s.get("triggered_patterns", [])[:3]) or "无明显信号",
                "risk_summary": "；".join(s.get("risk_flags", [])) or "无明显风险信号",
                "val_score": s.get("valuation_score", 0),
                "clause_score": s.get("clause_score", 0),
                "link_score": s.get("linkage_score", 0),
                "struct_score": s.get("structure_score", 0),
                "composite": s.get("score", 0),
                "grade": s.get("grade", ""),
            }
            for s in top_bonds
        ]
        print(f"Running LLM analysis on Top {llm_top_n} ...")
        analyses = analyst.analyze_batch(bonds_data)
        for code, analysis in analyses.items():
            print(f"\n--- {code} ---")
            print(analysis)
        print()

        # Re-generate reports with LLM analyses embedded (uses cached state, no re-run)
        logger.info("Regenerating reports with LLM analyses ...")
        md_path, json_path = pipeline.regenerate_report(
            top_n=args.top_n,
            llm_analyses=analyses,
        )
        result.md_path = md_path
        result.json_path = json_path

    # Backtest analysis
    if args.backtest:
        print()
        print("Running backtest on historical scores ...")
        try:
            bt_result = pipeline.run_backtest()
            result.backtest = bt_result
            print(f"  Backtest: {bt_result.start_date} ~ {bt_result.end_date} ({bt_result.num_periods} periods)")
            ic = bt_result.ic_summary
            if ic.get("num_periods", 0) > 0:
                print(f"  Mean Rank IC: {ic['mean_ic']:.4f} | IC IR: {ic['ic_ir']:.2f} | Win Rate: {ic['ic_win_rate']:.1%}")
            if bt_result.quintile_returns:
                q_keys = sorted(bt_result.quintile_returns.keys())
                if 1 in q_keys and len(q_keys) in q_keys:
                    spread = bt_result.quintile_returns[1] - bt_result.quintile_returns[len(q_keys)]
                    print(f"  Q1-Q{len(q_keys)} Spread: {spread:.2%}")
            if bt_result.errors:
                for e in bt_result.errors:
                    print(f"  [WARN] {e}")

            # Re-generate reports with backtest section embedded
            logger.info("Regenerating reports with backtest results ...")
            bt_md, bt_json = pipeline.regenerate_report(
                top_n=args.top_n,
                llm_analyses=analyses if not args.no_llm and result.ranked_stocks else {},
                backtest_result=bt_result,
            )
            result.md_path = bt_md
            result.json_path = bt_json
        except Exception as e:
            logger.warning("Backtest failed (non-fatal): %s", e)
            print(f"  [WARN] Backtest failed: {e}")

    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
