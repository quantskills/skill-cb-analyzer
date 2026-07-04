"""Shared fixtures for CB analyzer tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_cb_df() -> pd.DataFrame:
    """Create a sample CB quote DataFrame for testing."""
    return pd.DataFrame([
        {
            "bond_code": "113001.SH",
            "bond_name": "测试转债01",
            "stock_code": "600001.SH",
            "stock_name": "测试正股A",
            "cb_price": 115.0,
            "conversion_price": 10.0,
            "conversion_value": 120.0,
            "premium_rate": -4.17,
            "double_low": 115.0,
            "ytm": 0.015,
            "bond_floor_value": 98.0,
            "redemption_ratio": 1.25,
            "putback_ratio": 1.5,
            "credit_rating": "AA",
            "maturity_date": "2028-06-15",
            "remaining_years_raw": 2.0,
            "turnover": 5000,
            "outstanding_balance": 100000,
        },
        {
            "bond_code": "123456.SZ",
            "bond_name": "测试转债02",
            "stock_code": "000001.SZ",
            "stock_name": "测试正股B",
            "cb_price": 105.0,
            "conversion_price": 12.0,
            "conversion_value": 85.0,
            "premium_rate": 23.53,
            "double_low": 128.5,
            "ytm": 0.045,
            "bond_floor_value": 102.0,
            "redemption_ratio": 0.75,
            "putback_ratio": 1.07,
            "credit_rating": "AA-",
            "maturity_date": "2029-03-20",
            "remaining_years_raw": 2.7,
            "turnover": 800,
            "outstanding_balance": 50000,
        },
        {
            "bond_code": "127000.SZ",
            "bond_name": "测试转债03",
            "stock_code": "300001.SZ",
            "stock_name": "测试正股C",
            "cb_price": 140.0,
            "conversion_price": 8.0,
            "conversion_value": 165.0,
            "premium_rate": -15.15,
            "double_low": 40.0,
            "ytm": -0.02,
            "bond_floor_value": 80.0,
            "redemption_ratio": 1.65,
            "putback_ratio": 3.0,
            "credit_rating": "AAA",
            "maturity_date": "2027-01-10",
            "remaining_years_raw": 0.5,
            "turnover": 50000,
            "outstanding_balance": 30000,
        },
    ])


@pytest.fixture
def sample_stock_kline() -> pd.DataFrame:
    """Create sample stock K-line data for testing."""
    import numpy as np
    dates = pd.date_range("2026-01-01", "2026-07-03", freq="B")
    records = []
    for sym in ["600001.SH", "000001.SZ", "300001.SZ"]:
        base_price = {"600001.SH": 12.0, "000001.SZ": 8.5, "300001.SZ": 13.2}[sym]
        for i, d in enumerate(dates):
            noise = np.random.normal(0, 0.02)
            price = base_price * (1 + i * 0.001 + noise)
            records.append({
                "symbol": sym,
                "date": d.strftime("%Y%m%d"),
                "open": price * 0.99,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": np.random.randint(1000000, 10000000),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_config() -> dict:
    """Load sample config for testing."""
    import json
    from pathlib import Path
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}
