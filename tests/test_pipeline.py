"""Tests for core.pipeline — PipelineResult, signal correlation, integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core._types import SignalResult
from core.pipeline import CBPipeline, PipelineResult


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult()
        assert r.trade_date == ""
        assert r.total_cb == 0
        assert r.selected_count == 0
        assert r.ranked_stocks == []
        assert r.md_path == ""
        assert r.json_path == ""
        assert r.errors == []

    def test_with_errors(self):
        r = PipelineResult(errors=["error 1", "error 2"])
        assert len(r.errors) == 2

    def test_with_data(self):
        r = PipelineResult(
            trade_date="20260701",
            total_cb=100,
            selected_count=80,
            ranked_stocks=[{"code": "123001"}],
            md_path="/out/report.md",
            json_path="/out/report.json",
        )
        assert r.trade_date == "20260701"
        assert r.total_cb == 100
        assert r.selected_count == 80
        assert len(r.ranked_stocks) == 1


# ---------------------------------------------------------------------------
# Signal correlation (internal method)
# ---------------------------------------------------------------------------

class TestSignalCorrelation:
    def test_insufficient_bonds(self):
        """Less than 5 bonds → error."""
        pipeline = CBPipeline()
        result = pipeline._compute_signal_correlation([])
        assert "error" in result
        assert result["n_bonds"] == 0

        few_signals = [
            {"s1": SignalResult("s1", "S1", False, 0, "neutral", ""),
             "s2": SignalResult("s2", "S2", False, 0, "neutral", "")},
        ] * 3  # 3 bonds < 5
        result = pipeline._compute_signal_correlation(few_signals)
        assert "error" in result
        assert result["n_bonds"] == 3

    def test_enough_bonds_returns_pairs(self):
        """>=5 bonds → top_pairs with correlations."""
        pipeline = CBPipeline()
        rng = np.random.RandomState(42)

        n_bonds = 10
        signal_dicts = []
        for _ in range(n_bonds):
            s1_strength = rng.uniform(0, 1)
            signal_dicts.append({
                "s1": SignalResult("s1", "S1", True, s1_strength, "bullish", ""),
                "s2": SignalResult("s2", "S2", True, s1_strength * 0.8, "bullish", ""),
                "s3": SignalResult("s3", "S3", False, 0, "neutral", ""),
            })

        result = pipeline._compute_signal_correlation(signal_dicts)
        assert "error" not in result
        assert "top_pairs" in result
        assert result["n_bonds"] == n_bonds

        # s1 and s2 should be highly correlated
        pairs = result["top_pairs"]
        assert len(pairs) >= 1

    def test_pairs_sorted_by_abs_correlation(self):
        """Pairs should be sorted by |correlation| descending."""
        pipeline = CBPipeline()
        rng = np.random.RandomState(99)

        signal_dicts = []
        for _ in range(10):
            signal_dicts.append({
                "a": SignalResult("a", "A", True, rng.uniform(0, 1), "bullish", ""),
                "b": SignalResult("b", "B", True, rng.uniform(0, 1), "bullish", ""),
            })

        result = pipeline._compute_signal_correlation(signal_dicts)
        if "top_pairs" in result:
            pairs = result["top_pairs"]
            for i in range(len(pairs) - 1):
                assert abs(pairs[i]["correlation"]) >= abs(pairs[i + 1]["correlation"])


# ---------------------------------------------------------------------------
# Pipeline init
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_creates_detectors(self):
        pipeline = CBPipeline()
        assert pipeline._valuation is not None
        assert pipeline._clause is not None
        assert pipeline._linkage is not None
        assert pipeline._volatility is not None
        assert pipeline._risk is not None
        assert pipeline._scorer is not None

    def test_state_cache_empty_initially(self):
        pipeline = CBPipeline()
        assert pipeline._last_trade_date == ""
        assert pipeline._last_cb_df is None
        assert pipeline._last_ranked == []


# ---------------------------------------------------------------------------
# regenerate_report
# ---------------------------------------------------------------------------

class TestRegenerateReport:
    def test_raises_without_prior_run(self):
        pipeline = CBPipeline()
        with pytest.raises(RuntimeError, match="No cached pipeline state"):
            pipeline.regenerate_report()

    def test_succeeds_after_run(self, tmp_path):
        """regenerate_report should work after run() caches state."""
        pipeline = CBPipeline({"output": {"dir": str(tmp_path)}})
        cb_df = pd.DataFrame({
            "bond_code": ["123001", "123002"],
            "bond_name": ["CB A", "CB B"],
            "cb_price": [100, 110],
            "premium_rate": [15, 20],
            "ytm": [0.02, 0.03],
            "double_low": [115, 130],
            "conversion_value": [90, 95],
            "stock_code": ["000001", "000002"],
            "stock_name": ["Stock A", "Stock B"],
            "credit_rating": ["AA", "AA-"],
            "outstanding_balance": [50000, 30000],
            "redemption_ratio": [1.0, 1.0],
            "putback_ratio": [0.5, 0.5],
            "_data_quality_flag": [False, False],
        })

        from core.scorer import ScoreResult
        sr1 = ScoreResult(
            bond_code="123001", bond_name="CB A",
            composite_score=80, valuation_score=35, clause_score=25,
            linkage_score=10, structure_score=10,
            risk_penalty=0, neutralized_score=0, grade="B", rank=1,
            stock_code="000001", stock_name="Stock A",
            cb_price=100, premium_rate=15, conversion_value=90,
            double_low=115, ytm=0.02, triggered_signals=[], risk_flags=[],
        )
        sr2 = ScoreResult(
            bond_code="123002", bond_name="CB B",
            composite_score=75, valuation_score=30, clause_score=25,
            linkage_score=10, structure_score=10,
            risk_penalty=0, neutralized_score=0, grade="B", rank=2,
            stock_code="000002", stock_name="Stock B",
            cb_price=110, premium_rate=20, conversion_value=95,
            double_low=130, ytm=0.03, triggered_signals=[], risk_flags=[],
        )

        # Manually populate state cache
        pipeline._last_trade_date = "20260701"
        pipeline._last_cb_df = cb_df
        pipeline._last_ranked = [sr1, sr2]
        pipeline._last_score_results = [sr1, sr2]
        pipeline._last_val_results = {}
        pipeline._last_clause_results = {}
        pipeline._last_link_results = {}
        pipeline._last_struct_results = {}
        pipeline._last_vol_results = {}
        pipeline._last_stock_info = None
        pipeline._last_signal_correlation = {}

        md_path, json_path = pipeline.regenerate_report(top_n=5)
        assert Path(md_path).exists()
        assert Path(json_path).exists()
