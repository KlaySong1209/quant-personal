"""Point-in-time universe membership helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant.data.adjust.loaders import read_mapped_csv


@dataclass(frozen=True)
class UniverseColumnMap:
    symbol: str
    start: str
    end: str | None = None


def load_universe_membership(path: str | Path, mapping: UniverseColumnMap) -> pd.DataFrame:
    membership = read_mapped_csv(path, mapping, required=("symbol", "start"), optional=("end",))
    membership["symbol"] = membership["symbol"].astype(str)
    membership["start"] = pd.to_datetime(membership["start"], utc=True)
    if "end" not in membership:
        membership["end"] = pd.NaT
    membership["end"] = pd.to_datetime(membership["end"], utc=True)
    bad = membership["end"].notna() & (membership["end"] < membership["start"])
    if bad.any():
        raise ValueError("universe membership end date must be >= start date")
    return membership[["symbol", "start", "end"]].sort_values(["symbol", "start"]).reset_index(drop=True)


def build_universe_mask(
    index: pd.DatetimeIndex,
    symbols: list[str] | tuple[str, ...] | pd.Index,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    out = pd.DataFrame(False, index=idx, columns=list(symbols))
    mem = membership.copy()
    if mem.empty:
        return out
    mem["start"] = pd.to_datetime(mem["start"], utc=True)
    mem["end"] = pd.to_datetime(mem["end"], utc=True)
    for _, row in mem.iterrows():
        sym = str(row["symbol"])
        if sym not in out.columns:
            continue
        mask = idx >= pd.Timestamp(row["start"])
        if pd.notna(row["end"]):
            mask &= idx <= pd.Timestamp(row["end"])
        out.loc[mask, sym] = True
    return out


def all_symbols_universe(index: pd.DatetimeIndex, symbols: list[str] | pd.Index) -> pd.DataFrame:
    return pd.DataFrame(True, index=pd.DatetimeIndex(index), columns=list(symbols))


def synthetic_universe_with_delisting(
    *,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    delisted_symbol: str,
    delist_date: pd.Timestamp,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": sym,
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(delist_date) if sym == delisted_symbol else pd.Timestamp(end),
            }
            for sym in symbols
        ]
    )

