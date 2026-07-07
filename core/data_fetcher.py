"""Data fetcher: Pandadata (stock K-line + info) + AKShare (CB quotes).

Data sources:
  - CB quotes (primary): AKShare ``bond_cb_jsl`` — 集思录可转债实时行情
  - CB daily history:    AKShare ``bond_cb_daily`` — 转债日线
  - Stock K-line:        Pandadata ``get_stock_daily_post`` — 正股K线（后复权）
  - Stock info:          Pandadata ``get_stock_detail`` — 正股基本信息
  - Trading calendar:    Pandadata ``get_trade_cal`` / ``get_last_trade_date``
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

import panda_data as pdd

logger = logging.getLogger(__name__)

DEFAULT_FIELDS_KLINE = [
    "open", "high", "low", "close", "volume", "amount", "pre_close",
]

_config_cache: dict | None = None


def _load_config() -> dict:
    """Load config.json with in-memory caching."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if config_path.exists():
        _config_cache = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        _config_cache = {}
    return _config_cache


def _retry_api_call(
    func, *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    description: str = "API call",
    **kwargs,
):
    """Call *func* with retry and exponential backoff."""
    import time as _time

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    description, attempt + 1, max_retries + 1, e, delay,
                )
                _time.sleep(delay)
            else:
                logger.error("%s failed after %d attempts: %s", description, max_retries + 1, e)
                raise


class DataFetcher:
    """Encapsulates all data acquisition for the CB analyzer."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_config()
        self._initialized = False

    # -- Pandadata init -----------------------------------------------

    def init_api(self) -> None:
        """Log into Pandadata service.

        Credentials resolved from env vars or config.json.
        Username must use '86' prefix.
        """
        if self._initialized:
            return
        pd_cfg = self._config.get("pandadata", {})

        username = os.getenv("DEFAULT_USERNAME") or pd_cfg.get("username", "")
        password = os.getenv("DEFAULT_PASSWORD") or pd_cfg.get("password", "")
        base_url = pd_cfg.get("base_url", "http://pandadata.pandaaiquant.com")

        if username and not username.startswith("86") and not username.startswith("086"):
            username = "86" + username
            logger.info("Auto-added 86 prefix: %s", username)

        if not username or not password:
            raise RuntimeError(
                "Pandadata credentials not configured. Set:\n"
                "  - Env: DEFAULT_USERNAME / DEFAULT_PASSWORD\n"
                "  - config.json: pandadata.username / pandadata.password"
            )

        pdd.init_token(username=username, password=password, base_url=base_url)
        self._initialized = True
        logger.info("Pandadata API initialized (base_url=%s)", base_url)

    # -- Trading calendar ---------------------------------------------

    def get_last_trade_date(self, exchange: str = "sh") -> str:
        """Return the latest completed A-share trading day."""
        return pdd.get_last_trade_date(exchange=exchange)

    def is_trading_day(self, date_str: str) -> bool:
        """Check if date_str is an A-share trading day."""
        cal = pdd.get_trade_cal(start_date=date_str, end_date=date_str, exchange="sh")
        if cal.empty:
            return False
        val = cal.iloc[0].get("is_trade", cal.iloc[0].get("is_trading_day", "0"))
        return val in ("1", 1, True)

    def get_next_trade_date(self, date_str: str, offset: int = 1) -> str:
        """Get the Nth trading day after date_str."""
        end_dt = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=offset * 3)
        end_str = end_dt.strftime("%Y%m%d")
        cal = pdd.get_trade_cal(start_date=date_str, end_date=end_str, exchange="sh")
        if cal.empty:
            return date_str
        col = "is_trade" if "is_trade" in cal.columns else "is_trading_day"
        trade_days = cal[cal[col].isin(("1", 1, True))]
        if "date" in trade_days.columns:
            dates = sorted(trade_days["date"].astype(str).tolist())
        else:
            dates = sorted(trade_days.iloc[:, 0].astype(str).tolist())
        dates = [d for d in dates if d > date_str]
        if len(dates) >= offset:
            return dates[offset - 1]
        return date_str

    # -- Index data ---------------------------------------------------

    def fetch_index_daily(
        self,
        index_code: str = "000832",
        start_date: str = "20000101",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLC for an index (default: 中证转债指数 000832).

        Uses Pandadata index daily API. Falls back to stock daily interface.
        Caches result as ``cache/index_<code>.parquet``.

        Returns:
            DataFrame with [trade_date, close] or empty on failure.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        logger.info("Fetching index daily: %s (%s ~ %s)", index_code, start_date, end_date)
        try:
            raw = _retry_api_call(
                lambda: pdd.get_index_daily(
                    symbol=index_code, start_date=start_date, end_date=end_date,
                ),
                desc=f"index daily {index_code}",
            )
        except Exception:
            logger.warning("get_index_daily failed, trying stock daily fallback")
            try:
                raw = _retry_api_call(
                    lambda: pdd.get_stock_daily(
                        symbol=[index_code], start_date=start_date, end_date=end_date,
                    ),
                    desc=f"index daily fallback {index_code}",
                )
            except Exception as e:
                logger.warning("Index fetch failed: %s", e)
                return pd.DataFrame(columns=["trade_date", "close"])

        if raw is None or raw.empty:
            return pd.DataFrame(columns=["trade_date", "close"])

        df = raw.copy()
        date_col = next((c for c in ["date", "trade_date"] if c in df.columns), None)
        close_col = next((c for c in ["close"] if c in df.columns), None)
        if date_col and close_col:
            df = df[[date_col, close_col]].copy()
            df.rename(columns={date_col: "trade_date", close_col: "close"}, inplace=True)
            df["trade_date"] = df["trade_date"].astype(str)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["close"])
            df.sort_values("trade_date", inplace=True)

        # Cache to parquet
        try:
            cache_dir = Path("cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"index_{index_code}.parquet"
            df.to_parquet(cache_path, index=False)
            logger.info("Index cache saved: %s (%d rows)", cache_path, len(df))
        except Exception as e:
            logger.warning("Index cache save failed: %s", e)

        return df

    # -- CB data: 东方财富 (primary) + 同花顺 (supplement) ------------

    def fetch_cb_quotes(self) -> pd.DataFrame:
        """Fetch CB quotes from 东方财富 via AKShare.

        Uses ``ak.bond_zh_cov()`` (东方财富) as primary — returns ~1000 bonds
        with price, conversion price, conversion value, premium rate, stock info,
        credit rating. Falls back to 集思录 if unavailable.

        Returns:
            DataFrame with CB quote data. Attrs['source'] = 'akshare_eastmoney'.
        """
        import akshare as ak

        logger.info("Fetching CB quotes from 东方财富 (AKShare bond_zh_cov) ...")
        try:
            raw = ak.bond_zh_cov()
        except Exception as e:
            logger.warning("bond_zh_cov failed: %s — trying 集思录 fallback", e)
            raw = self._fetch_cb_quotes_fallback()

        if raw is None or raw.empty:
            logger.warning("Empty CB data from 东方财富, trying fallback")
            raw = self._fetch_cb_quotes_fallback()

        df = raw.copy()
        df.attrs["source"] = "akshare_eastmoney"

        # Filter: keep only bonds with conversion price set (= listed & active)
        cp_col = next((c for c in ["转股价", "conversion_price"] if c in df.columns), None)
        if cp_col:
            df = df[df[cp_col].notna()].copy()
        # Also ensure we have a valid price
        price_col = next((c for c in ["债现价", "cb_price"] if c in df.columns), None)
        if price_col:
            df = df[df[price_col].notna() & (df[price_col] > 0)].copy()

        logger.info("CB quotes: %d bonds (listed & active)", len(df))
        return df

    def _fetch_cb_quotes_fallback(self) -> pd.DataFrame:
        """Fallback: try 集思录 bond_cb_jsl, then bond_cb_daily."""
        import akshare as ak

        logger.info("Trying 集思录 (bond_cb_jsl) fallback ...")
        try:
            raw = ak.bond_cb_jsl()
            if raw is not None and not raw.empty:
                raw.attrs["source"] = "akshare_jisilu"
                return raw
        except Exception:
            logger.debug("bond_cb_jsl failed, trying next fallback", exc_info=True)

        logger.info("Trying bond_cb_daily fallback ...")
        try:
            df = ak.bond_cb_daily()
            if df is not None and not df.empty:
                df.attrs["source"] = "akshare_cb_daily_fallback"
                return df
        except Exception as e:
            logger.warning("bond_cb_daily also failed: %s", e)
        return pd.DataFrame()

    def _fetch_cb_maturity_info(self) -> pd.DataFrame | None:
        """Fetch maturity dates and coupon info from 同花顺 CB info.

        Uses ``ak.bond_zh_cov_info_ths()`` which returns ~937 bonds with:
        - 债券代码, 转股价格, 到期时间, 上市日期, 票面利率, 利率说明, etc.

        Returns:
            DataFrame with all available columns or None.
            Preserves coupon-related columns (票面利率, 利率说明) when present.
        """
        import akshare as ak

        logger.info("Fetching CB maturity info from 同花顺 (bond_zh_cov_info_ths) ...")
        try:
            df = ak.bond_zh_cov_info_ths()
            if df is not None and not df.empty:
                keep_cols = ["债券代码", "到期时间"]
                if "转股价格" in df.columns:
                    keep_cols.append("转股价格")
                # Preserve coupon rate columns if the API returns them
                for col in ["票面利率", "利率说明", "coupon_rate", "赎回价", "到期赎回价"]:
                    if col in df.columns:
                        keep_cols.append(col)
                return df[[c for c in keep_cols if c in df.columns]].copy()
        except Exception as e:
            logger.warning("bond_zh_cov_info_ths failed: %s", e)
        return None

    def _fetch_cb_turnover_data(self) -> pd.DataFrame | None:
        """Fetch real-time CB quotes with volume from 同花顺.

        Uses ``ak.bond_zh_hs_cov_spot()`` which returns ~320 bonds with:
        - code (bond code), trade (price), volume, amount (turnover), changepercent

        Returns:
            DataFrame with columns [code, trade, volume, amount, changepercent] or None.
        """
        import akshare as ak

        logger.info("Fetching CB turnover data from 同花顺 (bond_zh_hs_cov_spot) ...")
        try:
            df = ak.bond_zh_hs_cov_spot()
            if df is not None and not df.empty:
                # Normalize code column
                if "code" in df.columns:
                    df["code"] = df["code"].astype(str).str.strip()
                # Normalize amount: 同花顺 returns 元 → convert to 万元
                if "amount" in df.columns:
                    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") / 10000
                keep = ["code", "trade", "volume", "amount", "changepercent"]
                return df[[c for c in keep if c in df.columns]].copy()
        except Exception as e:
            logger.warning("bond_zh_hs_cov_spot failed: %s", e)
        return None

    def fetch_cb_stock_map(self) -> dict[str, str]:
        """Fetch CB→stock code mapping.

        Uses ``ak.bond_cb_stock_map()`` if available, otherwise extracts from
        the CB quote data itself.

        Returns:
            Dict mapping bond_code → stock_code (with .SH/.SZ suffix).
        """
        import akshare as ak

        try:
            map_df = ak.bond_cb_stock_map()
            if map_df is not None and not map_df.empty:
                mapping = {}
                for _, row in map_df.iterrows():
                    bond = str(row.get("代码", row.get("转债代码", row.get("bond_code", ""))))
                    stock = str(row.get("正股代码", row.get("stock_code", "")))
                    if bond and stock:
                        from core.exchange_utils import resolve_exchange_suffix
                        stock = resolve_exchange_suffix(stock)
                        mapping[bond] = stock
                logger.info("CB-stock map: %d pairs", len(mapping))
                return mapping
        except Exception as e:
            logger.warning("bond_cb_stock_map failed: %s", e)

        return {}

    # -- Stock data: Pandadata ---------------------------------------

    def fetch_stock_kline(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        fields: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch daily K-line for underlying stocks."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not symbols:
            return pd.DataFrame()
        fields = fields or DEFAULT_FIELDS_KLINE
        chunk_size = 200
        frames: list[pd.DataFrame] = []

        def _fetch_chunk(chunk_symbols: list[str], chunk_idx: int) -> pd.DataFrame:
            try:
                df = _retry_api_call(
                    pdd.get_stock_daily_post,  # 后复权：分红/送股/配股不产生价格跳空
                    symbol=chunk_symbols,
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields,
                    description=f"K-line chunk {chunk_idx}",
                )
                return df if not df.empty else pd.DataFrame()
            except Exception as e:
                logger.warning("K-line chunk %d failed: %s", chunk_idx, e)
                return pd.DataFrame()

        chunks = [
            (symbols[i:i + chunk_size], i)
            for i in range(0, len(symbols), chunk_size)
        ]

        max_workers = min(4, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_chunk, chunk, idx): idx
                for chunk, idx in chunks
            }
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    frames.append(df)

        if frames:
            result = pd.concat(frames, ignore_index=True)
            logger.info("Stock K-line: %d rows, %d stocks", len(result),
                         result["symbol"].nunique() if "symbol" in result.columns else 0)
            return result
        return pd.DataFrame()

    def fetch_stock_info(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch stock basic info: name, industry, list_status."""
        if not symbols:
            return pd.DataFrame()

        chunk_size = 200
        frames: list[pd.DataFrame] = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            try:
                df = _retry_api_call(
                    pdd.get_stock_detail,
                    symbol=chunk,
                    description=f"Stock detail chunk {i}",
                )
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning("Stock detail chunk %d failed: %s", i, e)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        col_map = {"sector_code_name": "industry", "status": "list_status"}
        for old, new in col_map.items():
            if old in result.columns and new not in result.columns:
                result[new] = result[old]

        for col in ["name", "industry", "list_status"]:
            if col not in result.columns:
                result[col] = "未知"

        keep = ["symbol", "name", "industry", "list_status"]
        result = result[[c for c in keep if c in result.columns]]
        logger.info("Stock info: %d stocks", len(result))
        return result

    # -- Column name normalization -----------------------------------

    @staticmethod
    def _normalize_cb_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename Chinese column names to English before caching.

        Parquet on Windows may garble Chinese characters in column names,
        so we normalize to ASCII-safe names early.
        """
        from core._types import CB_COLUMN_MAP

        df = df.copy()
        for old, new in CB_COLUMN_MAP.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        return df

    # -- High-level convenience --------------------------------------

    def fetch_all_data(
        self, trade_date: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
        """Fetch all data for the CB analysis.

        Returns:
            (cb_quote_df, stock_kline_df, stock_info_df, cb_stock_map)
        """
        self.init_api()

        # 1. CB quotes from 东方财富 (primary)
        cb_quote_df = self.fetch_cb_quotes()
        if cb_quote_df.empty:
            logger.warning("No CB quote data available")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

        # 1a. Merge maturity dates from 同花顺
        maturity_df = self._fetch_cb_maturity_info()
        if maturity_df is not None and not maturity_df.empty:
            # Find bond code columns on both sides (in case of normalization differences)
            bond_col = next((c for c in ["债券代码", "bond_code", "代码"] if c in cb_quote_df.columns), None)
            mat_bond_col = next((c for c in ["债券代码", "bond_code", "代码"] if c in maturity_df.columns), None)
            if bond_col and mat_bond_col and "到期时间" in maturity_df.columns:
                cb_quote_df[bond_col] = cb_quote_df[bond_col].astype(str).str.strip()
                maturity_df[mat_bond_col] = maturity_df[mat_bond_col].astype(str).str.strip()
                maturity_map = dict(zip(
                    maturity_df[mat_bond_col],
                    maturity_df["到期时间"]
                ))
                if "到期时间" not in cb_quote_df.columns:
                    cb_quote_df["到期时间"] = cb_quote_df[bond_col].map(maturity_map)
                logger.info("Merged maturity dates from 同花顺 (%d entries)", len(maturity_map))
            else:
                logger.warning("Maturity merge skipped: missing bond code or maturity column")

        # 1b. Merge turnover/volume from 同花顺 spot
        turnover_df = self._fetch_cb_turnover_data()
        if turnover_df is not None and not turnover_df.empty:
            bond_col = next((c for c in ["债券代码", "bond_code", "代码"] if c in cb_quote_df.columns), None)
            to_bond_col = next((c for c in ["code", "债券代码", "bond_code"] if c in turnover_df.columns), None)
            if bond_col and to_bond_col:
                cb_quote_df[bond_col] = cb_quote_df[bond_col].astype(str).str.strip()
                turnover_df[to_bond_col] = turnover_df[to_bond_col].astype(str).str.strip()
                for col in ["volume", "amount"]:
                    if col in turnover_df.columns:
                        tmap = dict(zip(turnover_df[to_bond_col], turnover_df[col]))
                        if col not in cb_quote_df.columns:
                            cb_quote_df[col] = cb_quote_df[bond_col].map(tmap)
                logger.info("Merged turnover data from 同花顺 spot (%d entries)", len(turnover_df))
            else:
                logger.warning("Turnover merge skipped: missing bond code column")

        # 2. CB→stock mapping (extract from CB data directly)
        cb_stock_map = self.fetch_cb_stock_map()
        if not cb_stock_map:
            stock_col = next((c for c in ["正股代码", "stock_code", "symbol"] if c in cb_quote_df.columns), None)
            bond_col = next((c for c in ["债券代码", "代码", "bond_code"] if c in cb_quote_df.columns), None)
            if stock_col and bond_col:
                for _, row in cb_quote_df.iterrows():
                    bond = str(row.get(bond_col, ""))
                    stock = str(row.get(stock_col, ""))
                    if bond and stock:
                        from core.exchange_utils import resolve_exchange_suffix
                        stock = resolve_exchange_suffix(stock)
                        cb_stock_map[bond] = stock

        # 3. Stock K-line
        stock_symbols = list(set(cb_stock_map.values()))
        dt = datetime.strptime(trade_date, "%Y%m%d")
        lookback_days = int(self._config.get("scan", {}).get("lookback_days", 120))
        start_dt = dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y%m%d")
        stock_kline_df = self.fetch_stock_kline(stock_symbols, start_date, trade_date)

        # 4. Stock info
        stock_info_df = self.fetch_stock_info(stock_symbols)

        # 5. Normalize CB column names to English (prevents parquet encoding issues on Windows)
        cb_quote_df = self._normalize_cb_columns(cb_quote_df)

        return cb_quote_df, stock_kline_df, stock_info_df, cb_stock_map
