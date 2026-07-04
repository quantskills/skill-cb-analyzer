"""Tests for backtester — IC analysis, stratified backtest, data loading."""

import numpy as np
import pandas as pd
import pytest
from core.backtester import (
    BacktestResult,
    Backtester,
    compute_forward_returns,
    compute_ic,
    compute_ic_decay,
    compute_risk_metrics,
    fama_macbeth,
    stratified_backtest,
    load_scores_from_output,
)


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

class TestForwardReturns:
    @pytest.fixture
    def simple_prices(self):
        return pd.DataFrame({
            "trade_date": ["20260701", "20260702", "20260703", "20260704", "20260705",
                           "20260701", "20260702", "20260703", "20260704", "20260705"],
            "bond_code": ["A", "A", "A", "A", "A",
                          "B", "B", "B", "B", "B"],
            "cb_price": [100, 102, 104, 106, 108,    # A: +2% per day
                         200, 198, 196, 194, 192],   # B: -1% per day
        })

    def test_forward_2d(self, simple_prices):
        fwd = compute_forward_returns(simple_prices, forward_days=2)
        # A on 20260701: (104-100)/100 = 0.04
        row = fwd[(fwd["trade_date"] == "20260701") & (fwd["bond_code"] == "A")]
        assert len(row) == 1
        assert row.iloc[0]["fwd_return"] == pytest.approx(0.04)

    def test_forward_exceeds_data(self, simple_prices):
        """Dates near the end should not appear (no forward data)."""
        fwd = compute_forward_returns(simple_prices, forward_days=3)
        # 20260704: no N+3 data → excluded
        dates_a = fwd[fwd["bond_code"] == "A"]["trade_date"].tolist()
        assert "20260704" not in dates_a
        assert "20260705" not in dates_a

    def test_empty_prices(self):
        fwd = compute_forward_returns(pd.DataFrame(columns=["trade_date", "bond_code", "cb_price"]))
        assert fwd.empty


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

class TestICComputation:
    @pytest.fixture
    def perfect_positive(self):
        """Scores and returns perfectly positively correlated."""
        scores = pd.DataFrame({
            "trade_date": ["20260701"] * 20,
            "bond_code": [f"B{i:03d}" for i in range(20)],
            "composite_score": [i + 1 for i in range(20)],  # 1..20
        })
        returns = pd.DataFrame({
            "trade_date": ["20260701"] * 20,
            "bond_code": [f"B{i:03d}" for i in range(20)],
            "fwd_return": [float(i + 1) / 100 for i in range(20)],  # 0.01..0.20
        })
        return scores, returns

    def test_perfect_positive_ic(self, perfect_positive):
        ic = compute_ic(*perfect_positive)
        assert ic["num_periods"] == 1
        assert ic["mean_ic"] == pytest.approx(1.0, abs=0.01)
        assert ic["ic_win_rate"] == 1.0

    @pytest.fixture
    def perfect_negative(self):
        scores = pd.DataFrame({
            "trade_date": ["20260701"] * 20,
            "bond_code": [f"B{i:03d}" for i in range(20)],
            "composite_score": [i + 1 for i in range(20)],
        })
        returns = pd.DataFrame({
            "trade_date": ["20260701"] * 20,
            "bond_code": [f"B{i:03d}" for i in range(20)],
            "fwd_return": [float(20 - i) / 100 for i in range(20)],  # descending
        })
        return scores, returns

    def test_perfect_negative_ic(self, perfect_negative):
        ic = compute_ic(*perfect_negative)
        assert ic["mean_ic"] == pytest.approx(-1.0, abs=0.01)
        assert ic["ic_win_rate"] == 0.0

    def test_empty_merge(self):
        scores = pd.DataFrame({"trade_date": ["20260701"], "bond_code": ["A"], "composite_score": [50]})
        returns = pd.DataFrame({"trade_date": ["20260702"], "bond_code": ["A"], "fwd_return": [0.01]})
        ic = compute_ic(scores, returns)
        assert ic["num_periods"] == 0

    def test_few_bonds_skipped(self):
        """Dates with < 10 bonds should be skipped."""
        scores = pd.DataFrame({
            "trade_date": ["20260701"] * 5,
            "bond_code": [f"B{i}" for i in range(5)],
            "composite_score": list(range(5)),
        })
        returns = pd.DataFrame({
            "trade_date": ["20260701"] * 5,
            "bond_code": [f"B{i}" for i in range(5)],
            "fwd_return": [0.01] * 5,
        })
        ic = compute_ic(scores, returns)
        assert ic["num_periods"] == 0  # < 10 bonds → skipped


# ---------------------------------------------------------------------------
# Stratified backtest
# ---------------------------------------------------------------------------

class TestStratifiedBacktest:
    @pytest.fixture
    def two_date_data(self):
        """Two days with 30 bonds each, monotonic score → return relationship."""
        records_s, records_r = [], []
        for day_idx, date in enumerate(["20260701", "20260705"]):
            for i in range(30):
                code = f"B{i:03d}"
                score = float(i + 1)  # 1..30
                ret = float(score) / 1000 * (1 if day_idx == 0 else 1.1)
                records_s.append({"trade_date": date, "bond_code": code, "composite_score": score})
                records_r.append({"trade_date": date, "bond_code": code, "fwd_return": ret})
        return pd.DataFrame(records_s), pd.DataFrame(records_r)

    def test_quintile_order(self, two_date_data):
        result = stratified_backtest(*two_date_data, n_quintiles=5)
        assert result is not None
        qr = result["cumulative"]
        assert len(qr) == 5
        # Q5 (highest score) should have higher return than Q1 (lowest score)
        assert qr[5] > qr[1]
        # Risk metrics should exist per quintile
        assert "metrics" in result
        for q in range(1, 6):
            assert q in result["metrics"]
            assert "sharpe_ratio" in result["metrics"][q]
            assert "max_drawdown" in result["metrics"][q]

    def test_insufficient_data(self):
        """Single date with few bonds → None."""
        scores = pd.DataFrame({
            "trade_date": ["20260701"] * 5,
            "bond_code": [f"B{i}" for i in range(5)],
            "composite_score": list(range(5)),
        })
        returns = pd.DataFrame({
            "trade_date": ["20260701"] * 5,
            "bond_code": [f"B{i}" for i in range(5)],
            "fwd_return": [0.01] * 5,
        })
        result = stratified_backtest(scores, returns, n_quintiles=5)
        assert result is None  # < n_quintiles*3 bonds → None


# ---------------------------------------------------------------------------
# Backtester orchestrator
# ---------------------------------------------------------------------------

class TestBacktester:
    def test_insufficient_data(self, tmp_path):
        """Backtester with no output → returns error result."""
        config = {
            "output": {"dir": str(tmp_path / "no_output")},
            "backtest": {"forward_days": 5, "n_quintiles": 5, "min_periods": 2},
        }
        bt = Backtester(config)
        result = bt.run()
        assert isinstance(result, BacktestResult)
        assert len(result.errors) > 0
        assert result.num_periods == 0

    def test_empty_output_dir(self, tmp_path):
        out_dir = tmp_path / "empty_output"
        out_dir.mkdir()
        config = {
            "output": {"dir": str(out_dir)},
            "backtest": {"forward_days": 5, "n_quintiles": 5, "min_periods": 2},
        }
        bt = Backtester(config)
        result = bt.run()
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# Load scores from output
# ---------------------------------------------------------------------------

class TestLoadScoresFromOutput:
    def test_empty_dir(self, tmp_path):
        df = load_scores_from_output(tmp_path / "nonexistent")
        assert df.empty

    def test_with_data(self, tmp_path):
        import json
        date_dir = tmp_path / "2026-07-01"
        date_dir.mkdir(parents=True)
        jdata = {
            "rankings": [
                {"bond_code": "123456", "bond_name": "Test CB", "composite_score": 65.0},
                {"bond_code": "789012", "bond_name": "Test CB2", "composite_score": 42.0},
            ]
        }
        (date_dir / "cb_daily_20260701.json").write_text(json.dumps(jdata))
        df = load_scores_from_output(tmp_path)
        assert len(df) == 2
        assert set(df["bond_code"]) == {"123456", "789012"}
        assert df[df["bond_code"] == "123456"]["composite_score"].iloc[0] == 65.0


# ---------------------------------------------------------------------------
# IC decay
# ---------------------------------------------------------------------------

class TestICDecay:
    @pytest.fixture
    def multi_horizon_data(self):
        """Scores + prices for 3 dates with 20 bonds each."""
        records_s, records_p = [], []
        for day_idx, date in enumerate(["20260701", "20260702", "20260703"]):
            for i in range(20):
                code = f"B{i:03d}"
                score = float(i + 1)
                price = 100 + float(i) + day_idx * 2
                records_s.append({"trade_date": date, "bond_code": code, "composite_score": score})
                records_p.append({"trade_date": date, "bond_code": code, "cb_price": price})
        return pd.DataFrame(records_s), pd.DataFrame(records_p)

    def test_two_horizons_returned(self, multi_horizon_data):
        """Both short and long horizons produce results."""
        scores, prices = multi_horizon_data
        decay = compute_ic_decay(scores, prices, horizons=[1, 2])
        assert 1 in decay
        assert 2 in decay

    def test_long_horizon_skipped(self, multi_horizon_data):
        """Horizon exceeding data range is skipped gracefully."""
        scores, prices = multi_horizon_data
        decay = compute_ic_decay(scores, prices, horizons=[1, 30])
        assert 1 in decay
        assert 30 not in decay  # No forward data for 30-day horizon

    def test_empty_data(self):
        """Empty data → empty result."""
        empty = pd.DataFrame(columns=["trade_date", "bond_code", "composite_score"])
        prices = pd.DataFrame(columns=["trade_date", "bond_code", "cb_price"])
        decay = compute_ic_decay(empty, prices, horizons=[1, 3, 5])
        assert decay == {}


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    def test_known_returns(self):
        """Five days of +1% — known cumulative return, zero drawdown."""
        rets = [0.01, 0.01, 0.01, 0.01, 0.01]
        m = compute_risk_metrics(rets)
        assert m["n_days"] == 5
        assert m["cumulative_return"] == pytest.approx(0.05101, abs=1e-4)
        assert m["max_drawdown"] == 0.0  # No drawdown
        assert m["annualized_volatility"] == 0.0  # Identical returns → no variance
        assert m["sharpe_ratio"] == 0.0  # Zero volatility → Sharpe undefined → 0

    def test_drawdown_detected(self):
        """Returns with a drop should produce negative max_drawdown."""
        rets = [0.02, 0.02, -0.05, 0.01, 0.01]
        m = compute_risk_metrics(rets)
        assert m["max_drawdown"] < 0

    def test_empty_returns(self):
        m = compute_risk_metrics([])
        assert m["n_days"] == 0
        assert m["sharpe_ratio"] == 0.0

    def test_single_value(self):
        m = compute_risk_metrics([0.01])
        assert m["n_days"] == 1
        assert m["annualized_return"] == 0.0

    def test_all_positive_no_drawdown(self):
        rets = [0.005, 0.003, 0.008, 0.002, 0.006]
        m = compute_risk_metrics(rets)
        assert m["max_drawdown"] == 0.0
        assert m["sharpe_ratio"] > 0


# ---------------------------------------------------------------------------
# Fama-MacBeth factor attribution
# ---------------------------------------------------------------------------

class TestFamaMacBeth:
    @pytest.fixture
    def factor_data(self):
        """3 dates, 20 bonds each. valuation_score perfectly predicts fwd_return."""
        records_s, records_r = [], []
        for date in ["20260701", "20260702", "20260703"]:
            for i in range(20):
                code = f"B{i:03d}"
                val_score = float(i + 1)
                clause_score = 30.0
                link_score = 40.0
                struct_score = 50.0
                records_s.append({
                    "trade_date": date, "bond_code": code,
                    "composite_score": val_score,
                    "valuation_score": val_score,
                    "clause_score": clause_score,
                    "linkage_score": link_score,
                    "structure_score": struct_score,
                })
                records_r.append({
                    "trade_date": date, "bond_code": code,
                    "fwd_return": val_score / 1000,
                })
        return pd.DataFrame(records_s), pd.DataFrame(records_r)

    def test_factor_premiums_returned(self, factor_data):
        result = fama_macbeth(
            *factor_data,
            factor_columns=["valuation_score", "clause_score", "linkage_score", "structure_score"],
            min_dates=2,
        )
        assert "factor_premiums" in result
        assert "t_stats" in result
        assert result["n_dates"] == 3
        # valuation_score should have significant premium
        assert "valuation_score" in result["factor_premiums"]

    def test_insufficient_data(self):
        """Less than min_dates → error."""
        scores = pd.DataFrame({
            "trade_date": ["20260701"] * 20,
            "bond_code": [f"B{i:03d}" for i in range(20)],
            "composite_score": [float(i) for i in range(20)],
            "valuation_score": [float(i) for i in range(20)],
        })
        returns = pd.DataFrame({
            "trade_date": ["20260701"] * 20,
            "bond_code": [f"B{i:03d}" for i in range(20)],
            "fwd_return": [0.01] * 20,
        })
        result = fama_macbeth(scores, returns, factor_columns=["valuation_score"], min_dates=5)
        assert "error" in result

    def test_missing_columns(self, factor_data):
        """Missing factor columns → error."""
        scores, returns = factor_data
        result = fama_macbeth(scores, returns, factor_columns=["nonexistent_factor"])
        assert "error" in result
