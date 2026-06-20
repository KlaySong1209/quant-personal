"""Latest end-of-day quote sources for paper simulation.

Only local/manual quote files are implemented. No live data, broker API, or
network feed is connected here.

Quote sources:
  - ManualQuoteSource: reads a CSV of close prices (backward-compatible).
  - SnapshotQuoteSource: reads a landed snapshot file with required metadata fields.
  - RealtimeQuoteSource: reserved, raises NotImplementedError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_REQUIRED_COLUMNS = ("fetched_at", "source", "as_of_date", "timestamp", "symbol", "close")
SNAPSHOT_OPTIONAL_COLUMNS = ("open",)


class QuoteSource(ABC):
    @abstractmethod
    def latest(self, symbols: list[str], *, as_of: str | pd.Timestamp | None = None) -> dict[str, float]:
        raise NotImplementedError


class ManualQuoteSource(QuoteSource):
    """Read close prices from a local CSV with configurable column mapping.

    This is the primary quote source for manual paper-account day steps.
    The CSV must have at least timestamp, symbol, and close columns.
    An optional *open* column enables next_day_open fills.
    """

    def __init__(self, path: str | Path, *, column_mapping: dict[str, str] | None = None):
        self.path = Path(path)
        self.column_mapping = column_mapping or {"timestamp": "timestamp", "symbol": "symbol", "close": "close"}

    def latest(self, symbols: list[str], *, as_of: str | pd.Timestamp | None = None) -> dict[str, float]:
        """Return the latest close prices for the requested symbols."""
        df = self._read_and_validate(symbols, as_of)
        return {str(row["symbol"]): float(row["close"]) for _, row in df.iterrows()}

    def snapshot(self, symbols: list[str], *, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """Return a snapshot DataFrame with timestamp, symbol, close, and optional open.

        This is the preferred method when open prices are needed for next_day_open fills.
        """
        df = self._read_and_validate(symbols, as_of)
        result_cols = ["timestamp", "symbol", "close"]
        if "open" in df.columns:
            result_cols.append("open")
        return df[result_cols].reset_index(drop=True)

    def _read_and_validate(self, symbols: list[str], as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
        if not (3 <= len(symbols) <= 5):
            raise ValueError("manual quote source requires a 3-5 symbols universe")
        if not self.path.exists():
            raise FileNotFoundError(f"manual quote file not found: {self.path}")
        raw = pd.read_csv(self.path, dtype=str)
        required = ["timestamp", "symbol", "close"]
        missing = [
            f"{canonical}->{self.column_mapping.get(canonical)}"
            for canonical in required
            if not self.column_mapping.get(canonical) or self.column_mapping[canonical] not in raw.columns
        ]
        if missing:
            raise ValueError(f"missing mapped quote column(s): {missing}")
        df = pd.DataFrame(
            {
                "timestamp": raw[self.column_mapping["timestamp"]],
                "symbol": raw[self.column_mapping["symbol"]],
                "close": raw[self.column_mapping["close"]],
            }
        )
        # Optional open
        open_col = self.column_mapping.get("open")
        if open_col and open_col in raw.columns:
            df["open"] = raw[open_col]

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = df["symbol"].astype(str)
        df["close"] = pd.to_numeric(df["close"], errors="raise").astype(float)
        if not np.isfinite(df["close"]).all() or (df["close"] <= 0).any():
            raise ValueError("manual quote closes must be finite and strictly positive")
        if "open" in df.columns:
            df["open"] = pd.to_numeric(df["open"], errors="raise").astype(float)
            if not np.isfinite(df["open"]).all() or (df["open"] <= 0).any():
                raise ValueError("manual quote opens must be finite and strictly positive")
        if as_of is not None:
            ts = pd.Timestamp(as_of)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            df = df[df["timestamp"] <= ts]
        if df.empty:
            raise ValueError("manual quote file has no quote at or before requested date")
        latest_ts = df["timestamp"].max()
        latest = df[df["timestamp"] == latest_ts]
        actual = set(latest["symbol"])
        expected = set(symbols)
        if actual != expected:
            raise ValueError(f"manual quote symbols must match requested universe: expected {sorted(expected)}, got {sorted(actual)}")
        return latest


FileQuoteSource = ManualQuoteSource


class RealtimeQuoteSource(QuoteSource):
    def latest(self, symbols: list[str], *, as_of: str | pd.Timestamp | None = None) -> dict[str, float]:
        del symbols, as_of
        raise NotImplementedError("RealtimeQuoteSource is reserved for future broker/quote APIs; no live feed is implemented.")


class SnapshotQuoteSource(QuoteSource):
    """Read a landed quote snapshot file. No network access.

    The snapshot file MUST contain:
      - fetched_at: when the snapshot was fetched
      - source: where the data came from
      - as_of_date: the business date the snapshot represents
      - timestamp: per-row timestamp
      - symbol: per-row symbol
      - close: per-row close price

    Optional:
      - open: per-row open price (enables next_day_open fills)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _validate_snapshot(self, df: pd.DataFrame) -> None:
        """Fail-fast if required metadata columns are missing."""
        missing = [col for col in SNAPSHOT_REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"snapshot file missing required column(s): {missing}")

    def latest(self, symbols: list[str], *, as_of: str | pd.Timestamp | None = None) -> dict[str, float]:
        """Return close prices from the snapshot.

        This is the QuoteSource interface. For open prices, use snapshot().
        """
        df = self._read(symbols, as_of)
        return {str(row["symbol"]): float(row["close"]) for _, row in df.iterrows()}

    def snapshot(self, symbols: list[str], *, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """Return full snapshot DataFrame with all columns.

        Includes metadata (fetched_at, source, as_of_date) and optional open.
        """
        return self._read(symbols, as_of)

    def _read(self, symbols: list[str], as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"snapshot file not found: {self.path}")
        raw = pd.read_csv(self.path, dtype=str)
        self._validate_snapshot(raw)

        df = pd.DataFrame()
        df["fetched_at"] = pd.to_datetime(raw["fetched_at"], utc=True)
        df["source"] = raw["source"].astype(str)
        df["as_of_date"] = pd.to_datetime(raw["as_of_date"], utc=True)
        df["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        df["symbol"] = raw["symbol"].astype(str)
        df["close"] = pd.to_numeric(raw["close"], errors="raise").astype(float)
        if "open" in raw.columns:
            df["open"] = pd.to_numeric(raw["open"], errors="raise").astype(float)

        if not np.isfinite(df["close"]).all() or (df["close"] <= 0).any():
            raise ValueError("snapshot closes must be finite and strictly positive")
        if "open" in df.columns:
            if not np.isfinite(df["open"]).all() or (df["open"] <= 0).any():
                raise ValueError("snapshot opens must be finite and strictly positive")

        if as_of is not None:
            ts = pd.Timestamp(as_of)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            df = df[df["timestamp"] <= ts]

        if df.empty:
            raise ValueError("snapshot has no data at or before requested date")

        # Filter to requested symbols
        df = df[df["symbol"].isin(symbols)]
        actual = set(df["symbol"])
        expected = set(symbols)
        if actual != expected:
            raise ValueError(f"snapshot symbols must match requested universe: expected {sorted(expected)}, got {sorted(actual)}")

        return df.reset_index(drop=True)
