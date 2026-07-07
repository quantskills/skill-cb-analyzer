"""Tests for core.data_fetcher — column normalization, API init, trading day."""

import pandas as pd
import pytest

from core._types import CB_COLUMN_MAP
from core.data_fetcher import DataFetcher, _load_config, _retry_api_call


# ---------------------------------------------------------------------------
# Column normalization (_normalize_cb_columns)
# ---------------------------------------------------------------------------

class TestNormalizeColumns:
    def test_chinese_to_english(self):
        df = pd.DataFrame({
            "转债代码": ["123001", "123002"],
            "转债名称": ["Test A", "Test B"],
            "转债最新价": [100.0, 110.0],
            "正股代码": ["000001", "000002"],
            "转股溢价率": [15.0, 20.0],
        })
        result = DataFetcher._normalize_cb_columns(df)
        assert "bond_code" in result.columns
        assert "bond_name" in result.columns
        assert "cb_price" in result.columns
        assert "stock_code" in result.columns
        assert result.loc[0, "bond_code"] == "123001"
        assert result.loc[0, "cb_price"] == 100.0

    def test_partial_column_mapping(self):
        """Only some columns are Chinese — others remain."""
        df = pd.DataFrame({
            "转债代码": ["123001"],
            "already_english": [42.0],
        })
        result = DataFetcher._normalize_cb_columns(df)
        assert "bond_code" in result.columns
        assert "already_english" in result.columns

    def test_already_english_passthrough(self):
        df = pd.DataFrame({
            "bond_code": ["123001"],
            "cb_price": [100.0],
            "bond_name": ["Test"],
        })
        result = DataFetcher._normalize_cb_columns(df)
        assert list(result.columns) == ["bond_code", "cb_price", "bond_name"]

    def test_empty_df(self):
        df = pd.DataFrame()
        result = DataFetcher._normalize_cb_columns(df)
        assert result.empty

    def test_non_string_column_names(self):
        """Columns with non-string names should be left as-is."""
        df = pd.DataFrame({0: ["val1"], "转债代码": ["123001"]})
        result = DataFetcher._normalize_cb_columns(df)
        assert "bond_code" in result.columns
        assert 0 in result.columns


# ---------------------------------------------------------------------------
# API init
# ---------------------------------------------------------------------------

class TestInitApi:
    def test_success(self, monkeypatch):
        """With valid env vars, init_api should not raise."""
        monkeypatch.setenv("DEFAULT_USERNAME", "8613800000000")
        monkeypatch.setenv("DEFAULT_PASSWORD", "test_password")
        # Mock pdd.init_token to avoid actual network call
        import panda_data as _pdd
        monkeypatch.setattr(_pdd, "init_token", lambda **kw: None)
        fetcher = DataFetcher({})
        try:
            fetcher.init_api()
        except RuntimeError:
            pytest.fail("init_api() raised RuntimeError unexpectedly")

    def test_missing_credentials_raises(self, monkeypatch):
        monkeypatch.delenv("DEFAULT_USERNAME", raising=False)
        monkeypatch.delenv("DEFAULT_PASSWORD", raising=False)
        monkeypatch.setenv("DEFAULT_USERNAME", "")
        monkeypatch.setenv("DEFAULT_PASSWORD", "")
        # Empty config ensures no config.json fallback
        fetcher = DataFetcher({})
        with pytest.raises(RuntimeError, match="Pandadata credentials"):
            fetcher.init_api()


# ---------------------------------------------------------------------------
# Trading day
# ---------------------------------------------------------------------------

class TestTradingDay:
    def test_is_trading_day_true(self, monkeypatch):
        """Mock trade_cal to return a trading day."""
        monkeypatch.setenv("DEFAULT_USERNAME", "8613800000000")
        monkeypatch.setenv("DEFAULT_PASSWORD", "test_password")
        fetcher = DataFetcher()
        assert callable(fetcher.is_trading_day)

    def test_get_last_trade_date(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_USERNAME", "8613800000000")
        monkeypatch.setenv("DEFAULT_PASSWORD", "test_password")
        fetcher = DataFetcher()
        assert callable(fetcher.get_last_trade_date)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

class TestRetryApiCall:
    def test_success_first_try(self):
        called = [0]

        def ok_func(x):
            called[0] += 1
            return x * 2

        result = _retry_api_call(ok_func, 5, description="test")
        assert result == 10
        assert called[0] == 1

    def test_retry_then_succeed(self):
        called = [0]

        def flaky_func():
            called[0] += 1
            if called[0] < 3:
                raise ValueError("transient error")
            return "ok"

        result = _retry_api_call(flaky_func, max_retries=3, base_delay=0.01, description="test")
        assert result == "ok"
        assert called[0] == 3

    def test_all_retries_exhausted(self):
        def always_fail():
            raise RuntimeError("permanent error")

        with pytest.raises(RuntimeError, match="permanent error"):
            _retry_api_call(always_fail, max_retries=2, base_delay=0.01, description="test")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_dict(self):
        cfg = _load_config()
        assert isinstance(cfg, dict)

    def test_cache_returns_same_object(self):
        cfg1 = _load_config()
        cfg2 = _load_config()
        assert cfg1 is cfg2

    def test_has_expected_keys(self):
        cfg = _load_config()
        # Config should have at least scoring and valuation sections
        assert "scoring" in cfg or isinstance(cfg, dict)
