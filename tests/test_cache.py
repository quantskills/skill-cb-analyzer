"""Tests for cache manager."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from core.cache import CacheManager


@pytest.fixture
def cache_dir():
    """Create a temporary cache root directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_dfs():
    """Create sample DataFrames for cache testing."""
    cb_df = pd.DataFrame([
        {"bond_code": "113001", "bond_name": "测试A", "cb_price": 115.0},
        {"bond_code": "123456", "bond_name": "测试B", "cb_price": 105.0},
    ])
    stock_kline = pd.DataFrame([
        {"symbol": "600001", "date": "20260701", "close": 12.0},
        {"symbol": "600001", "date": "20260702", "close": 12.1},
    ])
    stock_info = pd.DataFrame([
        {"symbol": "600001", "name": "测试正股", "industry": "制造业"},
    ])
    return cb_df, stock_kline, stock_info


class TestCacheRoundtrip:
    def test_save_and_load(self, cache_dir, sample_dfs):
        """Save and load returns equivalent DataFrames."""
        cb_df, stock_kline, stock_info = sample_dfs
        cm = CacheManager(cache_dir)
        cm.save("20260701", cb_df, stock_kline, stock_info, "2026-07-01T12:00")

        loaded_cb, loaded_kl, loaded_si = cm.load("20260701")
        assert len(loaded_cb) == len(cb_df)
        assert len(loaded_kl) == len(stock_kline)
        assert len(loaded_si) == len(stock_info)
        # Check key column values survived
        assert loaded_cb.iloc[0]["bond_code"] == "113001"
        assert loaded_kl.iloc[0]["symbol"] == "600001"

    def test_has_positive(self, cache_dir, sample_dfs):
        """has() returns True after save."""
        cb_df, stock_kline, stock_info = sample_dfs
        cm = CacheManager(cache_dir)
        cm.save("20260701", cb_df, stock_kline, stock_info)
        assert cm.has("20260701")

    def test_has_negative(self, cache_dir):
        """has() returns False for unsaved date."""
        cm = CacheManager(cache_dir)
        assert not cm.has("20991231")

    def test_load_meta(self, cache_dir, sample_dfs):
        """load_meta() returns metadata dict."""
        cb_df, stock_kline, stock_info = sample_dfs
        cm = CacheManager(cache_dir)
        cm.save("20260701", cb_df, stock_kline, stock_info, "2026-07-01T12:00")

        meta = cm.load_meta("20260701")
        assert meta["trade_date"] == "20260701"
        assert meta["cb_quote_rows"] == 2
        assert meta["fetch_time"] == "2026-07-01T12:00"

    def test_load_meta_missing(self, cache_dir):
        """load_meta() returns empty dict for missing cache."""
        cm = CacheManager(cache_dir)
        assert cm.load_meta("20991231") == {}

    def test_load_nonexistent(self, cache_dir):
        """Load nonexistent cache returns empty DataFrames."""
        cm = CacheManager(cache_dir)
        cb, kl, si = cm.load("20991231")
        assert cb.empty
        assert kl.empty
        assert si.empty

    def test_save_overwrites(self, cache_dir, sample_dfs):
        """Saving twice overwrites previous data."""
        cb_df, stock_kline, stock_info = sample_dfs
        cm = CacheManager(cache_dir)
        cm.save("20260701", cb_df, stock_kline, stock_info)

        # Modify and save again
        cb_df2 = pd.DataFrame([{"bond_code": "999999", "bond_name": "新债", "cb_price": 200.0}])
        cm.save("20260701", cb_df2, stock_kline, stock_info)

        loaded_cb, _, _ = cm.load("20260701")
        assert len(loaded_cb) == 1
        assert loaded_cb.iloc[0]["bond_code"] == "999999"


class TestClearOld:
    def test_clear_old_removes(self, cache_dir, sample_dfs):
        """clear_old() removes directories older than keep_days."""
        cb_df, stock_kline, stock_info = sample_dfs
        cm = CacheManager(cache_dir)
        # Save data from 100 days ago
        cm.save("20260325", cb_df, stock_kline, stock_info)
        # Save data from today
        cm.save("20260703", cb_df, stock_kline, stock_info)

        removed = cm.clear_old(keep_days=30)
        assert removed >= 1
        assert not cm.has("20260325")
        assert cm.has("20260703")

    def test_clear_old_empty(self, cache_dir):
        """clear_old() with no cache returns 0."""
        cm = CacheManager(cache_dir)
        assert cm.clear_old(keep_days=30) == 0

    def test_non_date_dirs_ignored(self, cache_dir, sample_dfs):
        """Directories not matching YYYYMMDD are ignored."""
        cb_df, stock_kline, stock_info = sample_dfs
        cm = CacheManager(cache_dir)
        cm.save("20260701", cb_df, stock_kline, stock_info)
        # Create a non-date directory manually
        (cache_dir / "not_a_date").mkdir(exist_ok=True)

        removed = cm.clear_old(keep_days=365)
        # The non-date dir should be ignored; only date dirs counted
        assert removed == 0  # 20260701 is within 365 days


class TestCachedDates:
    def test_cached_dates_sorted(self, cache_dir, sample_dfs):
        """cached_dates returns sorted date list."""
        cb_df, stock_kline, stock_info = sample_dfs
        cm = CacheManager(cache_dir)
        cm.save("20260701", cb_df, stock_kline, stock_info)
        cm.save("20260625", cb_df, stock_kline, stock_info)
        cm.save("20260630", cb_df, stock_kline, stock_info)

        dates = cm.cached_dates
        assert dates == ["20260625", "20260630", "20260701"]
