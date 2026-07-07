"""Local Parquet cache for convertible bond data.

Avoids re-fetching data on every run during the same day.
Cache is keyed by trade date (YYYYMMDD).

Cache directory::

    cache/
      20260701/
        cb_quote.parquet
        stock_kline.parquet
        stock_info.parquet
        .cache_meta.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

META_FILENAME = ".cache_meta.json"


class CacheManager:
    """Date-keyed Parquet cache for CB analyzer data."""

    def __init__(self, cache_root: str | Path = "cache") -> None:
        self._root = Path(cache_root)

    # -- public API --------------------------------------------------

    def has(self, trade_date: str) -> bool:
        """Check if cached data exists for *trade_date*."""
        if not self._root.exists():
            return False
        date_dir = self._root / trade_date
        return date_dir.is_dir() and (date_dir / "cb_quote.parquet").exists()

    def load(self, trade_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached (cb_quote, stock_kline, stock_info) for *trade_date*.

        Returns empty DataFrames on corrupt or missing cache files.
        """
        date_dir = self._root / trade_date

        def _safe_read(name: str) -> pd.DataFrame:
            try:
                return pd.read_parquet(date_dir / name)
            except Exception:
                logger.warning("Corrupt or unreadable %s cache for %s, ignoring", name, trade_date)
                return pd.DataFrame()

        cb_quote = _safe_read("cb_quote.parquet")
        stock_kline = _safe_read("stock_kline.parquet")
        stock_info = _safe_read("stock_info.parquet")
        if not cb_quote.empty:
            logger.info("Cache hit: %s (%d CB quotes)", trade_date, len(cb_quote))
        else:
            logger.info("Cache miss: %s (no data found)", trade_date)
        return cb_quote, stock_kline, stock_info

    def load_meta(self, trade_date: str) -> dict:
        """Load cache metadata without reading parquet files."""
        date_dir = self._root / trade_date
        meta_path = date_dir / META_FILENAME
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                logger.debug("Failed to read cache metadata %s", meta_path, exc_info=True)
        return {}

    def save(
        self,
        trade_date: str,
        cb_quote_df: pd.DataFrame,
        stock_kline_df: pd.DataFrame,
        stock_info_df: pd.DataFrame,
        fetch_time: str = "",
    ) -> None:
        """Persist data to cache."""
        date_dir = self._root / trade_date
        date_dir.mkdir(parents=True, exist_ok=True)

        def _safe_parquet(df: pd.DataFrame, path: Path) -> None:
            df = df.copy()
            for col in df.columns:
                if df[col].dtype == object:
                    mask = df[col].isna()
                    df.loc[mask, col] = None
            df.to_parquet(path, index=False)

        _safe_parquet(cb_quote_df, date_dir / "cb_quote.parquet")
        _safe_parquet(stock_kline_df, date_dir / "stock_kline.parquet")
        _safe_parquet(stock_info_df, date_dir / "stock_info.parquet")

        cb_source = cb_quote_df.attrs.get("source", "unknown") if not cb_quote_df.empty else "none"
        meta = {
            "trade_date": trade_date,
            "cb_quote_rows": len(cb_quote_df),
            "stock_kline_rows": len(stock_kline_df),
            "stock_info_rows": len(stock_info_df),
            "cb_source": cb_source,
        }
        if fetch_time:
            meta["fetch_time"] = fetch_time
        (date_dir / META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Cache saved: %s", date_dir)

    def is_stale(self, trade_date: str, max_age_hours: int = 6) -> bool:
        """Check if cached data for *trade_date* is older than *max_age_hours*.

        Args:
            trade_date: YYYYMMDD cache key.
            max_age_hours: Maximum acceptable age in hours.

        Returns:
            True if cache is missing or older than *max_age_hours*.
        """
        meta = self.load_meta(trade_date)
        if not meta or "fetch_time" not in meta:
            return True  # No metadata — assume stale
        from datetime import datetime, timedelta
        try:
            fetch_time = datetime.fromisoformat(meta["fetch_time"])
            age = datetime.now() - fetch_time
            return age > timedelta(hours=max_age_hours)
        except (ValueError, TypeError):
            return True

    def clear_old(self, keep_days: int = 30) -> int:
        """Remove cached dates older than *keep_days* calendar days."""
        from datetime import datetime, timedelta

        if not self._root.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=keep_days)
        removed = 0
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            try:
                dt = datetime.strptime(child.name, "%Y%m%d")
                if dt < cutoff:
                    for f in child.iterdir():
                        f.unlink()
                    child.rmdir()
                    removed += 1
                    logger.info("Cache cleanup: removed %s", child.name)
            except ValueError:
                continue
        return removed

    @property
    def cached_dates(self) -> list[str]:
        """Return sorted list of cached dates."""
        if not self._root.exists():
            return []
        dates = []
        for child in self._root.iterdir():
            if child.is_dir() and (child / "cb_quote.parquet").exists():
                dates.append(child.name)
        return sorted(dates)
