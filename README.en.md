# skill-cb-analyzer

> A-share Convertible Bond Daily Comprehensive Analyzer: double-low strategy + clause-driven events + stock linkage + Black-Scholes options pricing (Delta/Gamma/Vega) + historical/implied volatility + IC stratified backtesting + delisting tracking. 21 signal detectors, 4-dimension weighted scoring (0-100), LLM per-bond deep analysis, Markdown + JSON dual-format daily report, MCP protocol support for AI Agent invocation.

<p align="center">
  <img alt="detectors" src="https://img.shields.io/badge/signal_detectors-21-brightgreen">
  <img alt="dimensions" src="https://img.shields.io/badge/scoring_dimensions-4-blue">
  <img alt="greeks" src="https://img.shields.io/badge/BS_Greeks-Delta_Gamma_Vega-purple">
  <img alt="backtest" src="https://img.shields.io/badge/backtest-IC_%2B_stratified-orange">
  <img alt="tests" src="https://img.shields.io/badge/tests-375-green">
  <img alt="mcp" src="https://img.shields.io/badge/MCP-4_tools-purple">
  <img alt="license" src="https://img.shields.io/badge/license-GPLv3-blue">
</p>

---

## What is this?

`skill-cb-analyzer` is an **Agent Skill**: after each trading day, it automatically analyzes all convertible bonds in the A-share market (~330 bonds), using 21 signal detectors × 4-dimension weighted scoring to find double-low, clause-game, stock-linkage, and volatility arbitrage opportunities, combined with Black-Scholes options pricing and IC stratified backtesting for strategy validation.

Differentiation from other CB tools:

| Existing Tools | Why This is Different |
|---|---|
| Jisilu (集思录) | Jisilu provides double-low rankings and market data; this Skill additionally provides **BS Greeks, HV/IV, IC backtesting, LLM per-bond analysis** |
| Ningwen (宁稳网) | Ningwen provides CB clauses and valuations; this Skill is **fully automated daily scanning + scoring + callable by AI Agents** |
| Lude (禄得网) | Lude focuses on double-low and discount arbitrage; this Skill adds **Black-Scholes pricing + volatility signals + delisting survivorship bias tracking** |

Core differentiation: **Convertible bond derivatives pricing model + quantitative backtest validation + AI Agent native integration**.

---

## Analysis Pipeline

```
Trigger (manual or cron 15:45) → Trading Day Check
  → Data Acquisition (CB quotes via Jisilu + Underlying K-line via Pandadata + CB clauses via THS)
  → 21 Signal Detectors (parallel)
    → Group A: Valuation (4)
    → Group B: Clause Events (4)
    → Group C: Linkage + Volatility (8)
    → Group D: Structure Risk (5)
  → 4-Dimension Weighted Scoring (0.40×Val + 0.30×Clause + 0.20×Linkage + 0.10×Structure)
  → Risk Penalties + Credit Exclusions
  → Composite Score 0-100 + Grade A+~E
  → LLM Per-Bond Analysis (DeepSeek/Claude API)
  → Markdown 11-Section Daily + JSON
  → (Optional) Backtest: IC Analysis + Stratified Returns + Delisting Tracking
  → Save to output/YYYY-MM-DD/
```

---

## Signal Detectors (21)

### Valuation (Group A, Weight 40%)

| # | Detector | Trigger | Weight |
|---|---|---|---|
| 1 | Double-Low | Price < 120 AND premium < 20% | 4 |
| 2 | YTM Defense | YTM > bond + 2% | 3 |
| 3 | Net Bond Premium | CB price / net bond value < 1.05 | 3 |
| 4 | Premium Percentile | Current premium at historical low percentile | 2 |

### Clause Events (Group B, Weight 30%)

| # | Detector | Trigger | Weight |
|---|---|---|---|
| 5 | Call Progress | Stock price / conversion price > 1.20 (warning) / > 1.28 (danger) | 4 |
| 6 | Put Probability | Composite score > 0.4 | 3 |
| 7 | Resale Progress | Stock price < conversion price × 70% | 2 |
| 8 | Near Maturity | Remaining term < 1 year | 1 |

### Stock Linkage (Group C, Weight 20%)

| # | Detector | Trigger | Weight |
|---|---|---|---|
| 9 | Underlying Momentum | 20-day return ± MA alignment | 3 |
| 10 | CB Divergence | CB return − underlying return < −3% | 2 |
| 11 | Delta Elasticity | BS Delta > 0.70 (high equity sensitivity) or < 0.30 (bond-like) | 2 |
| 12 | Underlying Pattern | MA golden/death cross, volume breakout/breakdown | 1 |

### Volatility & Options (included in Linkage dimension)

| # | Detector | Trigger | Weight |
|---|---|---|---|
| 13 | IV Percentile | IV at historical low/high percentile | 2 |
| 14 | Vol Divergence | IV − HV deviation > 8% (v1.7 lowered from 10%) | 2 |
| 15 | Vol Expansion | Current HV vs 20-day change > ±20% | 2 |
| 16 | Delta Quality | **BS Gamma > 0.05** (option sensitivity quality, v1.7 changed to Gamma signal) | 1 |

### Market Structure & Risk (Group D, Weight 10%)

| # | Detector | Trigger | Weight |
|---|---|---|---|
| 17 | Volume Active | Daily turnover > 50M | 2 |
| 18 | Balance Trend | Balance decrease > 5% | 1 |
| 19 | Credit Risk | ST underlying → **direct exclusion** | — |
| 20 | Call Announcement Exclusion | Announced call → **direct exclusion** | — |
| 21 | Liquidity Risk | Turnover < 1M → penalty (−10) | — |

---

## Black-Scholes Options Pricing

This Skill is **currently the only** free CB tool providing full BS Greeks:

| Greek | Formula | Usage |
|---|---|---|
| **Delta** | N(d₁) | Equity/bond sensitivity assessment, replaces simplified cv/cb_price approximation |
| **Gamma** | N'(d₁) / (S·σ·√T) | Option sensitivity quality (v1.7: changed to Gamma signal) |
| **Vega** | S·N'(d₁)·√T / 100 | Price impact per 1% volatility change |
| **HV** | std(log_returns) × √252 | Historical volatility (annualized, 60-day window) |
| **IV** | Bisection inversion | Implied volatility from CB option value |

---

## Backtest Framework

`--backtest` flag enables full backtesting:

| Metric | Description |
|---|---|
| Rank IC | Spearman rank correlation (score vs N-day forward return) |
| IC IR | Mean IC / IC standard deviation |
| IC Win Rate | Proportion of trading days with IC > 0 |
| IC Decay | Multi-horizon analysis (1/3/5/10/20 day) |
| Stratified Backtest | 5 groups equal-weight forward return + cumulative return curve |
| Newey-West t-test | HAC standard error correction for autocorrelation |
| Fama-MacBeth | 4-dimension factor risk premium attribution |
| Weight Calibration | Simplex grid search for optimal dimension weights |
| Benchmark Comparison | CSI Convertible Bond Index (000832) excess return / IR / tracking error |
| **Delisting Tracking v1.7** | Categorize by call/maturity/underlying-delisting, quantify survivorship bias direction & magnitude |

---

## Quick Start

### Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # Edit and fill in credentials

python run.py                       # Latest trading day (LLM enabled by default)
python run.py --date 20260701       # Specify date
python run.py --no-llm              # Skip LLM, rule-engine only
python run.py --top-n 30 --verbose  # Custom params
python run.py --backtest            # Enable backtest analysis
```

### MCP Server (AI Agent)

```bash
python mcp_server.py
# or after pip install:
cb-mcp
```

MCP Tools:

| Tool | Description |
|---|---|
| `run_cb_analyzer` | Run full daily CB analysis |
| `get_latest_report` | Get latest full report |
| `check_trading_day` | Check A-share trading day |
| `search_bonds` | Search bonds by name/code |

### Using with Other AI Agents

| Agent | Integration |
|---|---|
| **Claude Code** | Add MCP config, or place in `.claude/skills/` |
| **Cursor** | MCP config |
| **Codex / OpenAI** | MCP standard protocol |
| **Other LLMs** | Inject Portable Loader Prompt |

---

## Report Structure (11 Sections)

1. **Market Overview** — Total CBs, avg price, avg premium, avg YTM
2. **Double-Low Strategy Picks** — Top 15 double-low CBs
3. **High YTM Defense Portfolio** — YTM > 3% bond-like CBs
4. **Clause Event Monitoring** — Call warnings/put candidates/resale triggers/maturity
5. **Stock Linkage Picks** — Discount arbitrage opportunities + IV/HV columns
6. **Industry Distribution** — Industry concentration
7. **Credit Risk Alerts** — ST/rating downgrade/par-value delisting risk
8. **Composite Score Ranking** — Top N with BS Delta/volatility signals
9. **AI Per-Bond Analysis** — LLM deep analysis
10. **Data Provenance** — Source for each data category
11. **Strategy Backtest** — IC analysis + stratified backtest + delisting tracking (`--backtest`)

---

## Data Sources

| Data | Source | Notes |
|---|---|---|
| CB Quotes | AKShare bond_cb_jsl (Jisilu) | Price/premium/conversion price/trigger prices |
| CB Terms | AKShare THS | Maturity date/issue scale/coupon |
| Underlying K-line | Pandadata get_stock_daily_post (post-adjusted) | ~500 underlyings, 120 days |
| Underlying Info | Pandadata get_stock_detail | Industry, list_status |
| Options Pricing | scipy.stats | BS norm.cdf/pdf |
| LLM Analysis | DeepSeek / Claude API | Pluggable backend |

---

## Core Constraints

| Constraint | Description |
|---|---|
| Announced call exclusion | CBs with announced calls are directly excluded |
| Credit risk exclusion | ST/\*ST/PT/delisting-period → direct exclusion |
| BS Delta replacement | Delta Elasticity uses real N(d₁), falls back to cv/cb_price when HV unavailable |
| No bond recommendations | Use "值得关注" / "可跟踪", forbid "买入" / "目标价" |
| Trading day aware | Skip holidays |
| Audit trail | JSON output includes scoring sub-items |
| Delisting tracking v1.7 | Auto-categorize delisting reasons, quantify survivorship bias |

---

## Directory Structure

```
skill-cb-analyzer/
├── SKILL.md                    # Agent workflow entry point
├── README.md                   # Project introduction (Chinese)
├── README.en.md                # Project introduction (English)
├── LICENSE                     # GPLv3
├── config.json                 # Runtime config
├── pyproject.toml              # Python project metadata
├── requirements.txt            # Python dependencies
├── .env.example                # Credential template
├── run.py                      # CLI entry point
├── mcp_server.py               # MCP server (4 tools)
├── core/
│   ├── _types.py               # Shared types + safe_float()
│   ├── bond_calculator.py      # CB calculation engine
│   ├── data_fetcher.py         # Data acquisition
│   ├── exchange_utils.py       # Exchange suffix mapping
│   ├── valuation.py            # Group A: Valuation signals (4)
│   ├── clause_monitor.py       # Group B: Clause events (4)
│   ├── stock_linkage.py        # Group C: Stock linkage (4)
│   ├── options_pricing.py      # BS pricing + HV/IV + volatility detectors
│   ├── risk_filter.py          # Group D: Structure + risk (5)
│   ├── scorer.py               # 4-dimension weighted scoring
│   ├── pipeline.py             # End-to-end pipeline
│   ├── backtester.py           # Backtest framework (incl. delisting tracking)
│   ├── reporter.py             # Markdown + JSON report
│   ├── cache.py                # Data cache
│   ├── history_store.py        # Per-bond time-series store
│   ├── config_validator.py     # Config validation
│   └── data_quality.py         # Data quality validation
├── llm/
│   └── analyst.py              # LLM analysis (pluggable backend)
├── tests/                      # 375 test cases
├── references/                 # Reference documents
├── agents/
│   ├── openai.yaml             # OpenAI/Codex adapter
│   ├── cursor-rule.mdc         # Cursor IDE adapter
│   └── portable-loader.md      # Generic loader for any agent
├── data/                       # HistoryStore persistent data
├── cache/                      # Data cache
└── output/                     # Date-grouped reports
    └── YYYY-MM-DD/
        ├── cb_daily_YYYYMMDD.md
        └── cb_daily_YYYYMMDD.json
```

---

## Disclaimer

This Skill's output is for research reference only. It does **not constitute any investment advice**. Convertible bond trading involves market risk, credit risk, and clause risk. Investors should make independent judgments and bear trading risks. Past performance does not guarantee future results.

## License

GPLv3
