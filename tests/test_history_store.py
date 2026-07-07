"""Tests for core.history_store — HistoryStore persistence and querying."""

import os
from pathlib import Path

import pandas as pd
import pytest

from core.history_store import ALL_COLUMNS, HISTORY_COLUMNS, HistoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """HistoryStore pointed at a temp directory."""
    return HistoryStore(data_dir=tmp_path)


def _make_cb_df(bonds=None, trade_date=None):
    """Minimal CB DataFrame for save_snapshot."""
    if bonds is None:
        bonds = [
            {"bond_code": "123001", "premium_rate": 15.0, "outstanding_balance": 50000},
            {"bond_code": "123002", "premium_rate": 30.0, "outstanding_balance": 30000},
        ]
    rows = []
    for b in bonds:
        row = {"bond_code": b["bond_code"]}
        row["premium_rate"] = b.get("premium_rate", 20.0)
        row["outstanding_balance"] = b.get("outstanding_balance", 10000)
        row["redemption_ratio"] = b.get("redemption_ratio", 0.0)
        row["putback_ratio"] = b.get("putback_ratio", 0.0)
        if trade_date:
            row["trade_date"] = trade_date
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load history
# ---------------------------------------------------------------------------

class TestLoadHistory:
    def test_empty_dir_returns_empty_df(self, store):
        df = store.load_history()
        assert df.empty

    def test_with_data(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        df = store.load_history()
        assert len(df) == 2
        assert "123001" in df["bond_code"].values

    def test_cache_hit(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        df1 = store.load_history()
        df2 = store.load_history()
        # Same object (cached)
        assert df1.equals(df2)

    def test_cache_bypass(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        store.load_history()  # populate cache
        df_fresh = store.load_history(use_cache=False)
        assert len(df_fresh) == 2

    def test_corrupt_file_recovery(self, store, tmp_path):
        # Write invalid parquet content
        store._file.write_text("not a parquet file", encoding="utf-8")
        df = store.load_history()
        assert df.empty


# ---------------------------------------------------------------------------
# Save snapshot
# ---------------------------------------------------------------------------

class TestSaveSnapshot:
    def test_normal_append(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        store.save_snapshot("20260702", _make_cb_df())
        df = store.load_history(use_cache=False)
        assert len(df) == 4

    def test_dedup_same_date(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        # Save again with different data for same date
        bonds = [
            {"bond_code": "123003", "premium_rate": 10.0, "outstanding_balance": 10000},
        ]
        store.save_snapshot("20260701", _make_cb_df(bonds=bonds))
        df = store.load_history(use_cache=False)
        # Old rows for 20260701 replaced, only 1 bond remains
        assert len(df[df["trade_date"] == "20260701"]) == 1

    def test_missing_bond_code_column(self, store, caplog):
        df = pd.DataFrame({"premium_rate": [10.0]})
        store.save_snapshot("20260701", df)
        assert "missing bond_code" in caplog.text.lower()

    def test_empty_df_skipped(self, store):
        df = pd.DataFrame({"bond_code": []})
        store.save_snapshot("20260701", df)
        df_loaded = store.load_history(use_cache=False)
        assert df_loaded.empty

    def test_with_signal_strengths(self, store):
        cb_df = _make_cb_df()
        signals = {
            "123001": {"double_low": 0.8, "ytm_defense": 0.3},
            "123002": {"redemption": -0.5},
        }
        store.save_snapshot("20260701", cb_df, signal_strengths=signals)
        df = store.load_history(use_cache=False)
        row1 = df[df["bond_code"] == "123001"].iloc[0]
        assert row1["sig_double_low"] == 0.8
        assert row1["sig_ytm_defense"] == 0.3

    def test_with_score_results(self, store):
        from core.scorer import ScoreResult
        cb_df = _make_cb_df()
        sr1 = ScoreResult(
            bond_code="123001", bond_name="Test1",
            composite_score=85.0, valuation_score=40.0,
            clause_score=25.0, linkage_score=10.0,
            structure_score=10.0, risk_penalty=0.0, grade="A",
        )
        store.save_snapshot("20260701", cb_df, score_results=[sr1])
        df = store.load_history(use_cache=False)
        row = df[df["bond_code"] == "123001"].iloc[0]
        assert row.get("composite_score") == 85.0
        assert row.get("grade") == "A"


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------

class TestGetPremiumHistory:
    def test_found(self, store):
        bonds1 = _make_cb_df([{"bond_code": "123001", "premium_rate": 15.0}])
        bonds2 = _make_cb_df([{"bond_code": "123001", "premium_rate": 20.0}])
        store.save_snapshot("20260701", bonds1)
        store.save_snapshot("20260702", bonds2)
        series = store.get_premium_history("123001")
        assert len(series) == 2

    def test_not_found(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        series = store.get_premium_history("999999")
        assert series.empty

    def test_with_passed_df(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        df = store.load_history()
        series = store.get_premium_history("123001", df=df)
        assert len(series) == 1


class TestGetConsecutiveDays:
    def test_above_threshold(self, store):
        bonds = [
            {"bond_code": "123001", "redemption_ratio": 1.25},
            {"bond_code": "123001", "redemption_ratio": 1.30},
            {"bond_code": "123001", "redemption_ratio": 1.15},
        ]
        for i, b in enumerate(bonds):
            store.save_snapshot(f"2026070{i+1}", _make_cb_df(bonds=[b]))
        crossed, total = store.get_consecutive_days(
            "123001", "redemption_ratio", threshold=1.20, direction="above",
        )
        assert crossed == 2
        assert total == 3

    def test_below_threshold(self, store):
        bonds = [
            {"bond_code": "123001", "putback_ratio": 0.5},
            {"bond_code": "123001", "putback_ratio": 0.6},
            {"bond_code": "123001", "putback_ratio": 0.8},
        ]
        for i, b in enumerate(bonds):
            store.save_snapshot(f"2026070{i+1}", _make_cb_df(bonds=[b]))
        crossed, total = store.get_consecutive_days(
            "123001", "putback_ratio", threshold=0.7, direction="below",
        )
        assert crossed == 2

    def test_no_history(self, store):
        crossed, total = store.get_consecutive_days(
            "123001", "redemption_ratio", threshold=1.20,
        )
        assert crossed == 0
        assert total == 0

    def test_missing_field(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        crossed, total = store.get_consecutive_days(
            "123001", "nonexistent_field", threshold=1.0,
        )
        assert crossed == 0


class TestGetPreviousBalance:
    def test_found(self, store):
        bonds = _make_cb_df([{"bond_code": "123001", "outstanding_balance": 50000}])
        store.save_snapshot("20260701", bonds)
        bonds2 = _make_cb_df([{"bond_code": "123001", "outstanding_balance": 45000}])
        store.save_snapshot("20260702", bonds2)
        prev = store.get_previous_balance("123001", "20260702")
        assert prev == 50000.0

    def test_not_found(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        prev = store.get_previous_balance("123001", "20260701")
        assert prev is None

    def test_empty_history(self, store):
        prev = store.get_previous_balance("123001", "20260701")
        assert prev is None


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_valid_file(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        ok, msg = store.validate_integrity()
        assert ok is True
        assert "Valid" in msg

    def test_no_file(self, store):
        ok, msg = store.validate_integrity()
        assert ok is True

    def test_future_dates(self, store):
        bonds = [{"bond_code": "123001", "premium_rate": 15.0}]
        store.save_snapshot("20991231", _make_cb_df(bonds=bonds))
        ok, msg = store.validate_integrity()
        # 20991231 > 20991231 → "Future trade dates" — actually check boundary
        # The check is > "20991231", so 20991231 passes
        assert ok is True

    def test_beyond_future(self, store):
        bonds = [{"bond_code": "123001", "premium_rate": 15.0}]
        store.save_snapshot("21000101", _make_cb_df(bonds=bonds))
        ok, msg = store.validate_integrity()
        assert ok is False
        assert "Future trade dates" in msg

    def test_implausible_premium(self, store):
        bonds = [{"bond_code": "123001", "premium_rate": -300.0}]
        store.save_snapshot("20260701", _make_cb_df(bonds=bonds))
        ok, msg = store.validate_integrity()
        assert ok is False
        assert "premium_rate" in msg

    def test_backup_and_repair(self, store):
        store.save_snapshot("20260701", _make_cb_df())
        assert store.backup_and_repair() is True
        bak = store._file.with_suffix(".parquet.bak")
        assert bak.exists()

    def test_backup_no_file(self, store):
        assert store.backup_and_repair() is False
