"""Tests for backtester — IC analysis, stratified backtest, data loading."""

import numpy as np
import pandas as pd
import pytest
from core.backtester import (
    BacktestResult,
    Backtester,
    CostModel,
    compute_forward_returns,
    compute_ic,
    compute_ic_decay,
    compute_risk_metrics,
    fama_macbeth,
    stratified_backtest,
    load_scores_from_output,
    load_index_prices_from_cache,
    compute_benchmark_returns,
    _compute_benchmark_comparison,
    _generate_simplex_grid,
    calibrate_dimension_weights,
    compute_rolling_signal_ic,
    compute_dynamic_weights,
    SIGNAL_KEY_MAP,
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


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_default_round_trip(self):
        cm = CostModel()
        assert cm.enabled is True
        assert cm.round_trip_cost == pytest.approx(0.0009)

    def test_disabled_no_cost(self):
        cm = CostModel(enabled=False)
        assert cm.round_trip_cost == 0.0

    def test_custom_params(self):
        cm = CostModel(stamp_duty=0.001, commission=0.0002, slippage=0.0002)
        # 0.001 + 2*0.0002 + 2*0.0002 = 0.0018
        assert cm.round_trip_cost == pytest.approx(0.0018)

    def test_filters_default(self):
        cm = CostModel()
        assert cm.filter_limit_hit is True
        assert cm.min_daily_turnover == 100


# ---------------------------------------------------------------------------
# Cost-filtered forward returns
# ---------------------------------------------------------------------------

class TestCostFilteredForwardReturns:
    @pytest.fixture
    def prices_with_extras(self):
        """Prices with turnover and pre_close columns."""
        return pd.DataFrame({
            "trade_date": ["20260701", "20260702", "20260703",
                           "20260701", "20260702", "20260703"],
            "bond_code": ["A", "A", "A", "B", "B", "B"],
            "cb_price": [100, 102, 104, 200, 198, 196],
            "turnover": [500, 500, 500, 50, 50, 50],    # B is low turnover
            "pre_close": [99, 101, 103, 199, 197, 195],
        })

    def test_low_turnover_filtered(self, prices_with_extras):
        """Bond B with turnover < 100 should be filtered."""
        cm = CostModel(min_daily_turnover=100)
        fwd = compute_forward_returns(prices_with_extras, forward_days=1, cost_model=cm)
        # Only bond A should remain
        assert "A" in fwd["bond_code"].values
        assert "B" not in fwd["bond_code"].values

    def test_limit_up_filtered(self):
        """Bond at ~10% limit up should be filtered."""
        df = pd.DataFrame({
            "trade_date": ["20260701", "20260702",
                           "20260701", "20260702"],
            "bond_code": ["A", "A", "B", "B"],
            "cb_price": [110, 112, 100, 102],
            "pre_close": [100, 111, 99, 101],
        })
        cm = CostModel(filter_limit_hit=True)
        fwd = compute_forward_returns(df, forward_days=1, cost_model=cm)
        # A on day1: 110/100-1 = 10% → filtered
        assert len(fwd[fwd["bond_code"] == "A"]) == 0
        # B should remain
        assert len(fwd[fwd["bond_code"] == "B"]) > 0

    def test_no_cost_model_no_filter(self, prices_with_extras):
        """Without cost_model, all bonds pass through."""
        fwd = compute_forward_returns(prices_with_extras, forward_days=1)
        assert "B" in fwd["bond_code"].values


# ---------------------------------------------------------------------------
# Benchmark: index loading + returns + comparison
# ---------------------------------------------------------------------------

class TestBenchmarkIndex:
    def test_empty_dir(self, tmp_path):
        df = load_index_prices_from_cache(tmp_path, ["20260701"])
        assert df.empty

    def test_with_parquet(self, tmp_path):
        cache_file = tmp_path / "index_000832.parquet"
        pd.DataFrame({
            "trade_date": ["20260701", "20260702", "20260703"],
            "close": [1000, 1010, 1020],
        }).to_parquet(cache_file)
        df = load_index_prices_from_cache(tmp_path, ["20260701", "20260702"])
        assert len(df) == 2

    def test_date_filter(self, tmp_path):
        cache_file = tmp_path / "index_000832.parquet"
        pd.DataFrame({
            "trade_date": ["20260701", "20260702", "20260703"],
            "close": [1000, 1010, 1020],
        }).to_parquet(cache_file)
        df = load_index_prices_from_cache(tmp_path, ["20260701"])
        assert len(df) == 1


class TestBenchmarkReturns:
    def test_monotonic_up(self):
        idx_df = pd.DataFrame({
            "trade_date": ["20260701", "20260702", "20260703", "20260704", "20260705", "20260706"],
            "close": [1000, 1005, 1010, 1015, 1020, 1025],
        })
        result = compute_benchmark_returns(
            idx_df, ["20260701", "20260702", "20260703", "20260704", "20260705", "20260706"], forward_days=2,
        )
        assert result["cumulative_return"] > 0
        assert result["n_periods"] > 0

    def test_insufficient_data(self):
        result = compute_benchmark_returns(
            pd.DataFrame(columns=["trade_date", "close"]), ["20260701"], forward_days=5,
        )
        assert "error" in result

    def test_forward_exceeds_range(self):
        idx_df = pd.DataFrame({
            "trade_date": ["20260701", "20260702"],
            "close": [1000, 1005],
        })
        result = compute_benchmark_returns(idx_df, ["20260701", "20260702"], forward_days=5)
        assert "error" in result


class TestBenchmarkComparison:
    def test_perfect_outperform(self):
        """Q1 returns beat benchmark every period."""
        q1_daily = [0.02, 0.03, 0.01, 0.04, 0.02]
        bm_daily = [0.01, 0.01, 0.00, 0.02, 0.01]
        comp = _compute_benchmark_comparison(q1_daily, bm_daily)
        assert comp["win_rate"] == 1.0
        assert comp["excess_return"] > 0
        assert comp["information_ratio"] > 0

    def test_insufficient_overlap(self):
        comp = _compute_benchmark_comparison([0.01], [0.01])
        assert "error" in comp

    def test_equal_performance(self):
        """Same returns → zero excess, zero IR."""
        rets = [0.01, 0.01, 0.01, 0.01]
        comp = _compute_benchmark_comparison(rets, rets)
        assert comp["excess_return"] == pytest.approx(0.0, abs=0.001)
        assert comp["win_rate"] == 0.0  # excess > 0 never true (equal)


# ---------------------------------------------------------------------------
# Simplex grid + weight calibration
# ---------------------------------------------------------------------------

class TestSimplexGrid:
    def test_4var_step_005(self):
        grid = _generate_simplex_grid(4, 0.05)
        assert len(grid) == 1771  # C(20+4-1, 4-1) = C(23, 3) = 1771

    def test_all_sum_to_one(self):
        grid = _generate_simplex_grid(3, 0.1)
        for w in grid:
            assert sum(w) == pytest.approx(1.0)

    def test_small_step(self):
        grid = _generate_simplex_grid(2, 0.5)
        # 2 vars, step=0.5 → 3 combos: (0,1), (0.5,0.5), (1,0)
        assert len(grid) == 3
        assert (1.0, 0.0) in grid
        assert (0.0, 1.0) in grid
        assert (0.5, 0.5) in grid


class TestCalibrateWeights:
    @pytest.fixture
    def calibration_data(self):
        """10 dates, 20 bonds each. valuation_score strongly predicts returns."""
        records_s, records_r = [], []
        for day_idx, date in enumerate([f"202607{i:02d}" for i in range(1, 11)]):
            for b in range(20):
                code = f"B{b:03d}"
                val_score = float(b + 1)
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

    def test_returns_optimal_weights(self, calibration_data):
        scores, returns = calibration_data
        config = {
            "backtest": {
                "calibration": {
                    "dimension_step": 0.2,  # Coarse for speed
                    "train_ratio": 0.6,
                    "min_dates_train": 4,
                }
            }
        }
        result = calibrate_dimension_weights(scores, returns, config)
        assert "error" not in result
        assert "optimal_weights" in result
        assert len(result["optimal_weights"]) == 4
        opt = result["optimal_weights"]
        # Weights should sum to ~1.0
        assert sum(opt.values()) == pytest.approx(1.0, abs=0.01)
        # Train IC should be positive (valuation_score predicts fwd_return)
        assert result["train_ic"] > 0

    def test_insufficient_data(self):
        scores = pd.DataFrame({
            "trade_date": ["20260701"] * 20,
            "bond_code": [f"B{i:03d}" for i in range(20)],
            "composite_score": [float(i) for i in range(20)],
            "valuation_score": [float(i) for i in range(20)],
            "clause_score": [30.0] * 20,
            "linkage_score": [40.0] * 20,
            "structure_score": [50.0] * 20,
        })
        returns = pd.DataFrame({
            "trade_date": ["20260701"] * 20,
            "bond_code": [f"B{i:03d}" for i in range(20)],
            "fwd_return": [0.01] * 20,
        })
        result = calibrate_dimension_weights(scores, returns, {"backtest": {"calibration": {"min_dates_train": 10}}})
        assert "error" in result


# ---------------------------------------------------------------------------
# Rolling signal IC + dynamic weights
# ---------------------------------------------------------------------------

class TestRollingSignalIC:
    @pytest.fixture
    def signal_history(self):
        """3 dates, 20 bonds. sig_double_low perfectly predicts fwd_return."""
        records = []
        for day_idx, date in enumerate(["20260701", "20260702", "20260703"]):
            for b in range(20):
                records.append({
                    "trade_date": date,
                    "bond_code": f"B{b:03d}",
                    "sig_double_low": float(b + 1),
                    "sig_volume": 0.5,
                })
        return pd.DataFrame(records)

    @pytest.fixture
    def signal_prices(self):
        """Prices structured so higher b → higher return (positive IC)."""
        records = []
        for day_idx, date in enumerate(["20260701", "20260702", "20260703", "20260704"]):
            for b in range(20):
                # Higher b gets higher price growth over time
                base = 100.0
                price = base + float(b) * 0.1 * (day_idx + 1)
                records.append({
                    "trade_date": date,
                    "bond_code": f"B{b:03d}",
                    "cb_price": price,
                })
        return pd.DataFrame(records)

    def test_positive_ic(self, signal_history, signal_prices):
        ic = compute_rolling_signal_ic(
            signal_history, signal_prices,
            signal_keys=["double_low"],
            forward_days=1,
            min_periods=2,
        )
        assert "double_low" in ic
        assert ic["double_low"] > 0

    def test_unknown_signal_skipped(self, signal_history, signal_prices):
        ic = compute_rolling_signal_ic(
            signal_history, signal_prices,
            signal_keys=["nonexistent_signal"],
            forward_days=1,
        )
        assert "nonexistent_signal" not in ic

    def test_empty_history(self):
        empty = pd.DataFrame(columns=["trade_date", "bond_code"])
        prices = pd.DataFrame({"trade_date": ["20260701"], "bond_code": ["A"], "cb_price": [100]})
        ic = compute_rolling_signal_ic(empty, prices, ["double_low"])
        assert ic == {}


class TestDynamicWeights:
    def test_positive_ic_increases_weight(self):
        base = {"double_low": 4, "ytm_defense": 3, "bond_floor": 3}
        ic = {"double_low": 0.8, "ytm_defense": 0.1, "bond_floor": 0.05}
        dyn = compute_dynamic_weights(ic, base)
        assert sum(dyn.values()) == pytest.approx(1.0)
        # double_low should get the highest dynamic weight
        assert dyn["double_low"] == max(dyn.values())

    def test_negative_ic_zeroed(self):
        base = {"double_low": 4, "ytm_defense": 3}
        ic = {"double_low": -0.1, "ytm_defense": 0.3}
        dyn = compute_dynamic_weights(ic, base, floor_ic=0.0)
        # double_low has negative IC → zeroed
        assert dyn["double_low"] == 0.0
        assert dyn["ytm_defense"] > 0.9  # gets all weight

    def test_all_negative_fallback(self):
        base = {"double_low": 4, "ytm_defense": 3}
        ic = {"double_low": -0.1, "ytm_defense": -0.2}
        dyn = compute_dynamic_weights(ic, base)
        # Fallback to base weights normalized
        assert dyn["double_low"] > 0
        assert dyn["ytm_defense"] > 0

    def test_empty_ic_passthrough(self):
        base = {"double_low": 4, "ytm_defense": 3}
        dyn = compute_dynamic_weights({}, base)
        assert dyn == base

    def test_normalization(self):
        base = {"a": 2, "b": 2}
        ic = {"a": 0.5, "b": 0.5}
        dyn = compute_dynamic_weights(ic, base)
        assert sum(dyn.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SIGNAL_KEY_MAP
# ---------------------------------------------------------------------------

class TestSignalKeyMap:
    def test_all_config_keys_mapped(self):
        """Every key in default detector_weights has a corresponding sig_ column."""
        config_keys = [
            "double_low", "ytm_defense", "bond_floor", "premium_percentile",
            "redemption_progress", "downward_revision", "putback_progress", "maturity_alert",
            "stock_momentum", "cb_stock_deviation", "delta_elasticity", "stock_pattern",
            "iv_percentile", "hv_iv_divergence", "vol_expansion", "bs_delta",
            "volume_active", "balance_trend",
        ]
        for key in config_keys:
            assert key in SIGNAL_KEY_MAP, f"Missing: {key}"
            col = SIGNAL_KEY_MAP[key]
            assert col.startswith("sig_"), f"Bad prefix: {col}"
