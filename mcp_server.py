#!/usr/bin/env python3
"""MCP server exposing the CB analyzer as callable tools.

Start with:
    python mcp_server.py

Or install and run:
    pip install -e .
    cb-mcp

LLMs can then call:
    - run_cb_analyzer: Full daily CB analysis (with optional LLM analysis)
    - get_latest_report: Read full content of the most recent CB report
    - check_trading_day: Verify if a date is an A-share trading day
    - search_bonds: Search ranked bonds by name/code in latest report
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from core.data_fetcher import DataFetcher, _load_config
from core.pipeline import CBPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_server")

mcp = FastMCP("cb-analyzer")


# -- helpers -----------------------------------------------------------

def _get_output_dir(config: dict) -> Path:
    """Resolve output directory from config, relative to project root."""
    out_dir = config.get("output", {}).get("dir", "output")
    path = Path(out_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


# -- tools -------------------------------------------------------------

@mcp.tool()
def run_cb_analyzer(
    date: str = "",
    top_n: int = 20,
    use_llm: bool = True,
    llm_top_n: int = 5,
) -> dict:
    """Run daily convertible bond analysis.

    This is the main tool — it fetches market data, runs 17 signal detectors,
    scores and ranks all CBs, and generates Markdown + JSON reports.

    Args:
        date: Target trade date (YYYYMMDD). Leave empty for latest trading day.
        top_n: Number of top bonds in the report ranking (default: 20).
        use_llm: If True (default), runs LLM per-bond analysis on top N bonds.
                 Falls back to rule-based analysis when LLM is unavailable.
                 Set to False to skip LLM and use rule-based only.
        llm_top_n: How many top bonds to analyze with LLM when use_llm=True
                   (default: 5).

    Returns:
        Summary dict with ranked bonds, report paths, and optional LLM analyses.
    """
    config = _load_config()
    pipeline = CBPipeline(config)
    trade_date = date if date else None
    result = pipeline.run(trade_date=trade_date, top_n=top_n)

    if result.errors and result.total_cb == 0:
        return {
            "status": "error",
            "message": "; ".join(result.errors),
            "trade_date": result.trade_date,
        }

    response = {
        "status": "ok",
        "trade_date": result.trade_date,
        "total_cb": result.total_cb,
        "selected_count": result.selected_count,
        "top10": [
            {
                "rank": s["rank"],
                "name": s["name"],
                "code": s["code"],
                "score": s["score"],
                "grade": s["grade"],
                "price": s.get("cb_price", 0),
                "premium": s.get("premium_rate", 0),
                "triggered_signals": s["triggered_patterns"][:3],
                "risk_flags": s["risk_flags"],
            }
            for s in result.ranked_stocks[:10]
        ],
        "md_report": result.md_path,
        "json_report": result.json_path,
    }

    # Optional LLM analysis
    if use_llm and result.ranked_stocks:
        try:
            from llm.analyst import CBAnalyst

            analyst = CBAnalyst(config)
            llm_n = min(llm_top_n, len(result.ranked_stocks))
            top_bonds = result.ranked_stocks[:llm_n]

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

            analyses = analyst.analyze_batch(bonds_data)
            response["llm_analyses"] = {
                s.get("code", "?"): analysis
                for s, analysis in zip(top_bonds, analyses.values())
            }

            # Re-generate reports with LLM analyses embedded
            md_path, json_path = pipeline.regenerate_report(
                top_n=top_n, llm_analyses=analyses,
            )
            response["md_report"] = md_path
            response["json_report"] = json_path
        except Exception as e:
            logger.warning("LLM analysis failed: %s", e)
            response["llm_error"] = str(e)

    return response


@mcp.tool()
def get_latest_report(full_content: bool = False) -> dict:
    """Get the most recent CB analysis report.

    Args:
        full_content: If True, returns the complete markdown report text.
                      If False (default), returns metadata + preview (first 80 lines).

    Returns:
        Dict with report metadata, paths, and content.
    """
    config = _load_config()
    output_dir = _get_output_dir(config)

    if not output_dir.exists():
        return {"status": "error", "message": "No reports found (output/ directory empty)"}

    date_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
    if not date_dirs:
        return {"status": "error", "message": "No report dates found"}

    latest_dir = date_dirs[0]
    md_files = list(latest_dir.glob("cb_daily_*.md"))
    json_files = list(latest_dir.glob("cb_daily_*.json"))

    result = {
        "status": "ok",
        "date": latest_dir.name,
        "md_path": str(md_files[0]) if md_files else "",
        "json_path": str(json_files[0]) if json_files else "",
    }

    if md_files:
        content = md_files[0].read_text(encoding="utf-8")
        if full_content:
            result["content"] = content
        else:
            lines = content.split("\n")
            result["preview"] = "\n".join(lines[:80])
            result["total_lines"] = len(lines)

    # Also load JSON for structured data
    if json_files:
        try:
            jdata = json.loads(json_files[0].read_text(encoding="utf-8"))
            result["market_summary"] = jdata.get("market_summary", {})
            result["top5_rankings"] = [
                {
                    "rank": r["rank"],
                    "bond_code": r["bond_code"],
                    "bond_name": r["bond_name"],
                    "score": r["composite_score"],
                    "grade": r["grade"],
                }
                for r in jdata.get("rankings", [])[:5]
            ]
        except Exception:
            pass

    return result


@mcp.tool()
def check_trading_day(date: str) -> dict:
    """Check if a given date is an A-share trading day.

    Args:
        date: Date to check (YYYYMMDD).

    Returns:
        Dict with is_trading_day flag and last trade date for reference.
    """
    config = _load_config()
    fetcher = DataFetcher(config)
    try:
        fetcher.init_api()
    except RuntimeError as e:
        return {"status": "error", "message": str(e), "date": date}

    is_trade = fetcher.is_trading_day(date)

    return {
        "status": "ok",
        "date": date,
        "is_trading_day": is_trade,
        "last_trade_date": fetcher.get_last_trade_date(),
    }


@mcp.tool()
def search_bonds(query: str, top_n: int = 10) -> dict:
    """Search for bonds by name or code in the latest report rankings.

    Use this to look up specific bonds mentioned in conversation or check
    their ranking, score, and signals.

    Args:
        query: Bond name (partial match) or bond code (exact match).
        top_n: Max number of results to return (default: 10).

    Returns:
        Dict with matching bonds, their ranks, scores, and signal details.
    """
    config = _load_config()
    output_dir = _get_output_dir(config)

    if not output_dir.exists():
        return {"status": "error", "message": "No reports found"}

    date_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
    if not date_dirs:
        return {"status": "error", "message": "No report dates found"}

    # Load the latest JSON report
    json_files = list(date_dirs[0].glob("cb_daily_*.json"))
    if not json_files:
        return {"status": "error", "message": f"No JSON report found for {date_dirs[0].name}"}

    try:
        jdata = json.loads(json_files[0].read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "error", "message": f"Failed to read report: {e}"}

    rankings = jdata.get("rankings", [])
    query_lower = query.lower().strip()

    matches = []
    for r in rankings:
        code = str(r.get("bond_code", ""))
        name = str(r.get("bond_name", ""))
        if query_lower in name.lower() or query_lower == code:
            matches.append({
                "rank": r["rank"],
                "bond_code": code,
                "bond_name": name,
                "stock_name": r.get("stock_name", ""),
                "cb_price": r.get("cb_price", 0),
                "premium_rate": r.get("premium_rate", 0),
                "composite_score": r.get("composite_score", 0),
                "grade": r.get("grade", ""),
                "valuation_score": r.get("valuation_score", 0),
                "clause_score": r.get("clause_score", 0),
                "linkage_score": r.get("linkage_score", 0),
                "triggered_signals": r.get("triggered_signals", []),
                "risk_flags": r.get("risk_flags", []),
            })

    return {
        "status": "ok",
        "report_date": date_dirs[0].name,
        "query": query,
        "matches_count": len(matches),
        "matches": matches[:top_n],
    }


def main():
    """Entry point for console_scripts."""
    mcp.run()


if __name__ == "__main__":
    main()
