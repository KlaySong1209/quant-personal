"""Canonical OHLCV schema helpers."""

from __future__ import annotations

import pandas as pd

OHLCV_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")


def coerce_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, list(OHLCV_COLUMNS)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["symbol"] = out["symbol"].astype(str)
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out

