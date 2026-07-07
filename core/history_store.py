"""Per-bond time-series history store for detector history.

Powers A4 (premium_rate percentile), D2 (outstanding balance trend),
and B1/B3 (redemption/putback consecutive-day tracking).

File: ``data/cb_history.parquet``

Columns:
    trade_date          — YYYYMMDD
    bond_code           — CB code
    premium_rate        — Current premium rate (for A4 percentile)
    outstanding_balance — Current outstanding balance in 万元 (for D2 trend)
    redemption_ratio    — stock_price / conversion_price (for B1 consecutive days)
    putback_ratio       — stock_price / (conversion_price * 0.70) (for B3 consecutive days)
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from core._types import safe_float

logger = logging.getLogger(__name__)

HISTORY_COLUMNS = [
    "trade_date", "bond_code", "premium_rate",
    "outstanding_balance", "redemption_ratio", "putback_ratio",
]

SIGNAL_COLUMNS = [
    "sig_double_low", "sig_ytm_defense", "sig_bond_floor", "sig_premium_percentile",
    "sig_redemption", "sig_downward_revision", "sig_putback", "sig_maturity",
    "sig_stock_momentum", "sig_cb_stock_deviation", "sig_delta", "sig_stock_pattern",
    "sig_iv_percentile", "sig_hv_iv_divergence", "sig_vol_expansion", "sig_bs_delta",
    "sig_volume", "sig_balance_trend",
]

ALL_COLUMNS = HISTORY_COLUMNS + SIGNAL_COLUMNS


class HistoryStore:
    """Persistent per-bond time-series store for detector history."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._dir = Path(data_dir)
        self._file = self._dir / "cb_history.parquet"
        self._cache: pd.DataFrame | None = None

    # -- public API ----------------------------------------------------

    def load_history(self, use_cache: bool = True) -> pd.DataFrame:
        """Load all historical data.

        Args:
            use_cache: If True (default), returns a cached copy when available.
                       Set to False to force a fresh read from disk.

        Returns:
            DataFrame with columns [trade_date, bond_code, premium_rate,
            outstanding_balance], or empty DataFrame on first run.
        """
        if use_cache and self._cache is not None:
            return self._cache.copy()
        if not self._file.exists():
            self._cache = pd.DataFrame(columns=ALL_COLUMNS)
            return self._cache.copy()
        try:
            df = pd.read_parquet(self._file)
            # Backward compat: ensure all expected columns exist
            for col in ALL_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            self._cache = df
            return df.copy()
        except Exception:
            logger.warning("Corrupt history file, starting fresh")
            self._cache = pd.DataFrame(columns=ALL_COLUMNS)
            return self._cache.copy()

    def save_snapshot(self, trade_date: str, cb_df: pd.DataFrame,
                       score_results: list | None = None,
                       signal_strengths: dict[str, dict[str, float]] | None = None) -> None:
        """Append current date's per-bond metrics to history.

        Deduplicates: if trade_date already exists, removes old rows first.
        Uses atomic write (temp file + rename) to prevent partial writes.

        Args:
            trade_date: YYYYMMDD string.
            cb_df: CB DataFrame with columns bond_code, premium_rate,
                   and optionally outstanding_balance.
            score_results: Optional list of ScoreResult objects; if provided,
                           score columns are merged into stored rows.
            signal_strengths: Optional dict of bond_code → {sig_key: strength}
                              for per-signal strength persistence.
        """
        if "bond_code" not in cb_df.columns:
            logger.warning("Cannot save history: cb_df missing bond_code column")
            return

        # Build score lookup if provided
        score_lookup: dict[str, dict] = {}
        if score_results:
            for sr in score_results:
                code = getattr(sr, "bond_code", getattr(sr, "code", ""))
                if code:
                    score_lookup[str(code)] = {
                        "composite_score": getattr(sr, "composite_score", getattr(sr, "score", 0)),
                        "valuation_score": getattr(sr, "valuation_score", getattr(sr, "val_score", 0)),
                        "clause_score": getattr(sr, "clause_score", 0),
                        "linkage_score": getattr(sr, "linkage_score", getattr(sr, "link_score", 0)),
                        "structure_score": getattr(sr, "structure_score", getattr(sr, "struct_score", 0)),
                        "risk_penalty": getattr(sr, "risk_penalty", 0),
                        "grade": getattr(sr, "grade", ""),
                    }

        rows = []
        for _, row in cb_df.iterrows():
            bond = str(row.get("bond_code", ""))
            if not bond:
                continue
            premium = safe_float(row.get("premium_rate", 0), 0.0)
            balance = safe_float(row.get("outstanding_balance",
                                row.get("issue_scale",
                                row.get("发行规模", 0))), 0.0)
            redemption = safe_float(row.get("redemption_ratio", 0), 0.0)
            putback = safe_float(row.get("putback_ratio", 0), 0.0)
            row_data = {
                "trade_date": trade_date,
                "bond_code": bond,
                "premium_rate": premium,
                "outstanding_balance": balance,
                "redemption_ratio": redemption,
                "putback_ratio": putback,
            }
            # Merge score data if available
            if bond in score_lookup:
                row_data.update(score_lookup[bond])
            # Merge signal strengths if available
            if signal_strengths and bond in signal_strengths:
                for sig_key, sig_val in signal_strengths[bond].items():
                    col_name = f"sig_{sig_key}"
                    if col_name in SIGNAL_COLUMNS or col_name.startswith("sig_"):
                        row_data[col_name] = sig_val
            rows.append(row_data)

        if not rows:
            return

        new_df = pd.DataFrame(rows)

        # Deduplicate: remove existing rows for this trade_date, then append
        existing = self.load_history()
        existing = existing[existing["trade_date"] != trade_date]
        combined = pd.concat([existing, new_df], ignore_index=True)

        # Ensure expected columns exist (backward compat with old files)
        for col in ALL_COLUMNS:
            if col not in combined.columns:
                combined[col] = None

        self._dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file first, then rename
        tmp_path = ""
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".parquet", prefix="cb_hist_", dir=str(self._dir),
            )
            os.close(fd)
            combined.to_parquet(tmp_path, index=False)
            # On Windows, the target must not exist for os.replace
            if self._file.exists():
                bak_path = self._file.with_suffix(".parquet.bak")
                shutil.copy2(self._file, bak_path)
                self._file.unlink()
            os.replace(tmp_path, self._file)
        except Exception:
            # Clean up temp file on failure
            if tmp_path:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass
            raise

        self._cache = None  # Invalidate cache after write
        logger.info("History saved: %d bonds for %s (total %d rows)",
                     len(rows), trade_date, len(combined))

    # -- integrity helpers -----------------------------------------------

    def validate_integrity(self) -> tuple[bool, str]:
        """Validate history parquet file integrity.

        Returns:
            (is_valid, message) — True if file is readable and internally
            consistent, False with a reason otherwise.
        """
        if not self._file.exists():
            return True, "No history file yet"
        try:
            df = pd.read_parquet(self._file)
            for col in ALL_COLUMNS:
                if col not in df.columns:
                    return False, f"Missing column: {col}"
            # Sanity checks
            dates = df["trade_date"].astype(str)
            if (dates > "20991231").any():
                return False, "Future trade dates detected"
            pr = pd.to_numeric(df.get("premium_rate", pd.Series()), errors="coerce")
            if (pr < -200).any():
                return False, "Implausible premium_rate values (< -200%)"
            return True, f"Valid: {len(df)} rows, {df['bond_code'].nunique()} bonds"
        except Exception as e:
            return False, f"Corrupt or unreadable: {e}"

    def backup_and_repair(self) -> bool:
        """Create .bak backup. Returns True if the original is readable."""
        if not self._file.exists():
            return False
        bak_path = self._file.with_suffix(".parquet.bak")
        shutil.copy2(self._file, bak_path)
        logger.info("History backup saved: %s", bak_path)
        try:
            pd.read_parquet(self._file)
            return True
        except Exception:
            logger.warning("History file corrupt; backup created, starting fresh")
            self._cache = pd.DataFrame(columns=ALL_COLUMNS)
            return False

    def get_premium_history(self, bond_code: str,
                             df: pd.DataFrame | None = None) -> pd.Series:
        """Return historical premium_rate series for a single bond.

        Args:
            bond_code: CB code.
            df: Pre-loaded history DataFrame (avoids re-reading parquet).

        Returns:
            pandas Series of premium rates, indexed by trade_date.
        """
        if df is None:
            df = self.load_history()
        if df.empty:
            return pd.Series(dtype=float)
        mask = df["bond_code"].astype(str) == str(bond_code)
        hist = df[mask].copy()
        if hist.empty:
            return pd.Series(dtype=float)
        hist = hist.sort_values("trade_date")
        return pd.Series(hist["premium_rate"].values, index=hist["trade_date"].values)

    def get_consecutive_days(
        self, bond_code: str, field: str,
        threshold: float, direction: str = "above",
        window: int = 30,
        df: pd.DataFrame | None = None,
    ) -> tuple[int, int]:
        """Count how many days in the last *window* days *field* crossed *threshold*.

        Used for redemption/putback consecutive-day tracking (clause B1/B3).

        Args:
            bond_code: CB code.
            field: Column name to check (e.g. 'redemption_ratio', 'putback_ratio').
            threshold: Threshold value.
            direction: 'above' (field >= threshold) or 'below' (field <= threshold).
            window: Lookback window in trading days (default 30).
            df: Pre-loaded history DataFrame (avoids re-reading parquet).

        Returns:
            (count_crossed, total_days_in_window) — count is how many days crossed
            the threshold, total is the number of days with data available.
        """
        if df is None:
            df = self.load_history()
        if df.empty or field not in df.columns:
            return 0, 0

        mask = df["bond_code"].astype(str) == str(bond_code)
        hist = df[mask].sort_values("trade_date")
        if hist.empty:
            return 0, 0

        # Take last N days
        recent = hist.tail(window)
        total_days = len(recent)

        ratios = recent[field].dropna()
        if ratios.empty:
            return 0, total_days

        if direction == "above":
            crossed = (ratios >= threshold).sum()
        else:
            crossed = (ratios <= threshold).sum()

        return int(crossed), total_days

    def get_previous_balance(self, bond_code: str, trade_date: str,
                              df: pd.DataFrame | None = None) -> float | None:
        """Get outstanding balance from the most recent date before trade_date.

        Args:
            bond_code: CB code.
            trade_date: Current trade date (YYYYMMDD).
            df: Pre-loaded history DataFrame (avoids re-reading parquet).

        Returns:
            Previous balance (万元) or None if no history exists.
        """
        if df is None:
            df = self.load_history()
        if df.empty:
            return None
        mask = (df["bond_code"].astype(str) == str(bond_code)) & (df["trade_date"] < trade_date)
        hist = df[mask].sort_values("trade_date")
        if hist.empty:
            return None
        val = hist.iloc[-1].get("outstanding_balance", 0)
        return float(val) if pd.notna(val) and val > 0 else None
