"""Tests for core.reporter — MD/JSON report generation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from core.reporter import Reporter, _safe_float
from core.scorer import ScoreResult


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_nan_returns_zero(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf_returns_zero(self):
        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf")) == 0.0

    def test_normal_passthrough(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float(0.0) == 0.0
        assert _safe_float(-5.0) == -5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_score_result(bond_code="123001", bond_name="Test CB", rank=1, **overrides):
    defaults = {
        "bond_code": bond_code,
        "bond_name": bond_name,
        "composite_score": 80.0,
        "valuation_score": 35.0,
        "clause_score": 25.0,
        "linkage_score": 10.0,
        "structure_score": 10.0,
        "risk_penalty": 0.0,
        "neutralized_score": 0.0,
        "grade": "B",
        "rank": rank,
        "stock_code": "000001",
        "stock_name": "Test Stock",
        "cb_price": 110.0,
        "premium_rate": 15.0,
        "conversion_value": 95.0,
        "double_low": 125.0,
        "ytm": 0.02,
        "triggered_signals": ["double_low"],
        "risk_flags": [],
        "excluded": False,
        "exclude_reason": "",
    }
    defaults.update(overrides)
    return ScoreResult(**defaults)


def _make_cb_df(n=5):
    return pd.DataFrame({
        "bond_code": [f"12300{i}" for i in range(1, n + 1)],
        "bond_name": [f"Test CB {i}" for i in range(1, n + 1)],
        "cb_price": [100 + i * 5 for i in range(n)],
        "premium_rate": [10 + i * 5 for i in range(n)],
        "ytm": [0.02 + i * 0.005 for i in range(n)],
        "double_low": [110 + i * 10 for i in range(n)],
        "conversion_value": [90 + i * 5 for i in range(n)],
        "stock_code": [f"00000{i}" for i in range(1, n + 1)],
        "stock_name": [f"Stock {i}" for i in range(1, n + 1)],
        "credit_rating": ["AA"] * n,
    })


# ---------------------------------------------------------------------------
# Render Markdown
# ---------------------------------------------------------------------------

class TestRenderMd:
    def test_header_present(self):
        reporter = Reporter()
        cb_df = _make_cb_df(5)
        ranked = [_make_score_result(f"12300{i}", f"CB {i}", rank=i + 1) for i in range(1, 6)]
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 20,
        )
        assert "# 可转债每日分析报告" in md
        assert "2026-07-01" in md

    def test_market_overview_table(self):
        reporter = Reporter()
        cb_df = _make_cb_df(5)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 20,
        )
        assert "## 1. 市场概览" in md
        assert "转债总数" in md
        assert "5" in md  # total CBs

    def test_section_8_rankings(self):
        reporter = Reporter()
        cb_df = _make_cb_df(5)
        ranked = [_make_score_result(f"12300{i}", f"CB {i}", rank=i + 1) for i in range(1, 4)]
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
        )
        assert "## 8. 综合评分排名" in md
        assert "CB 1" in md
        assert "CB 2" in md

    def test_llm_section_empty(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
            llm_analyses={},
        )
        assert "LLM 分析" in md or "## 9. AI" in md

    def test_llm_section_filled(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
            llm_analyses={"123001": "This is LLM analysis text for CB 1."},
        )
        assert "This is LLM analysis text" in md

    def test_backtest_section_absent(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
        )
        # Without backtest_result, section 11 should not appear
        assert "## 11. 策略回测" not in md

    def test_benchmark_section_with_backtest(self):
        """When backtest_result is provided, section 11 should appear."""
        reporter = Reporter()
        cb_df = _make_cb_df(3)

        class FakeBacktest:
            ic_summary = {"num_periods": 0}
            quintile_returns = {}
            num_periods = 0
            forward_days = 5
            start_date = "2026-01-01"
            end_date = "2026-07-01"
            summary = ""
            factor_attribution = {}
            benchmark_comparison = {}
            cost_model = {}
            weight_calibration = {}
            dynamic_weights = {}
            ic_decay = {}

        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
            backtest_result=FakeBacktest(),
        )
        assert "## 11. 策略回测" in md

    def test_signal_correlation_section(self):
        reporter = Reporter()
        cb_df = _make_cb_df(5)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        corr = {
            "top_pairs": [
                {"signal_a": "double_low", "signal_b": "ytm_defense", "correlation": 0.75},
            ],
            "n_bonds": 5,
        }
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
            signal_correlation=corr,
        )
        assert "## 12. 信号相关性矩阵" in md
        assert "double_low" in md
        assert "0.750" in md

    def test_signal_correlation_insufficient(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        corr = {"error": "insufficient_data", "n_bonds": 3}
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
            signal_correlation=corr,
        )
        assert "数据不足" in md

    def test_disclaimer_present(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        md = reporter._render_md(
            "20260701", "2026-07-01", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
        )
        assert "免责声明" in md
        assert "不构成任何投资建议" in md


# ---------------------------------------------------------------------------
# Render JSON
# ---------------------------------------------------------------------------

class TestRenderJson:
    def test_valid_json_output(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        json_str = reporter._render_json("20260701", cb_df, ranked, ranked)
        data = json.loads(json_str)
        assert data["trade_date"] == "20260701"
        assert data["total_cb"] == 3

    def test_rankings_array(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        ranked = [
            _make_score_result("123001", "CB 1", rank=1),
            _make_score_result("123002", "CB 2", rank=2),
        ]
        json_str = reporter._render_json("20260701", cb_df, ranked, ranked)
        data = json.loads(json_str)
        assert len(data["rankings"]) == 2
        assert data["rankings"][0]["rank"] == 1
        assert data["rankings"][0]["bond_code"] == "123001"

    def test_excluded_array(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        excluded_list = [_make_score_result("999999", "Bad CB", rank=0,
                                            excluded=True, exclude_reason="强赎风险")]
        json_str = reporter._render_json("20260701", cb_df, [], excluded_list)
        data = json.loads(json_str)
        assert len(data["excluded"]) == 1
        assert data["excluded"][0]["reason"] == "强赎风险"

    def test_market_summary_keys(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        json_str = reporter._render_json("20260701", cb_df, [], [])
        data = json.loads(json_str)
        assert "market_summary" in data
        assert "avg_price" in data["market_summary"]
        assert "avg_premium_rate" in data["market_summary"]

    def test_disclaimer_present(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        json_str = reporter._render_json("20260701", cb_df, [], [])
        data = json.loads(json_str)
        assert "disclaimer" in data

    def test_llm_analyses_included(self):
        reporter = Reporter()
        cb_df = _make_cb_df(3)
        analyses = {"123001": "Strong buy signal"}
        json_str = reporter._render_json("20260701", cb_df, [], [],
                                          llm_analyses=analyses)
        data = json.loads(json_str)
        assert data["llm_analyses"]["123001"] == "Strong buy signal"


# ---------------------------------------------------------------------------
# Generate (filesystem)
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_files_written(self, tmp_path):
        cfg = {"output": {"dir": str(tmp_path)}}
        reporter = Reporter(cfg)
        cb_df = _make_cb_df(2)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        md_path, json_path = reporter.generate(
            "20260701", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
        )
        assert Path(md_path).exists()
        assert Path(json_path).exists()
        assert md_path.endswith(".md")
        assert json_path.endswith(".json")

    def test_md_has_content(self, tmp_path):
        cfg = {"output": {"dir": str(tmp_path)}}
        reporter = Reporter(cfg)
        cb_df = _make_cb_df(2)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        md_path, _ = reporter.generate(
            "20260701", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
        )
        content = Path(md_path).read_text(encoding="utf-8")
        assert len(content) > 100

    def test_json_is_parseable(self, tmp_path):
        cfg = {"output": {"dir": str(tmp_path)}}
        reporter = Reporter(cfg)
        cb_df = _make_cb_df(2)
        ranked = [_make_score_result("123001", "CB 1", rank=1)]
        _, json_path = reporter.generate(
            "20260701", cb_df, ranked, ranked,
            {}, {}, {}, {}, 10,
        )
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert data["trade_date"] == "20260701"
