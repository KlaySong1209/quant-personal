"""Trading calendar validation and alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant.data.adjust.loaders import read_mapped_csv


@dataclass(frozen=True)
class CalendarColumnMap:
    timestamp: str


@dataclass(frozen=True)
class TradingCalendar:
    exchange: str
    sessions: pd.DatetimeIndex

    def __post_init__(self) -> None:
        sessions = pd.DatetimeIndex(pd.to_datetime(self.sessions, utc=True)).sort_values().unique()
        if len(sessions) == 0:
            raise ValueError("trading calendar must contain at least one session")
        object.__setattr__(self, "sessions", sessions)

    @classmethod
    def synthetic(cls, start: str | pd.Timestamp, end: str | pd.Timestamp, exchange: str = "SYNTH") -> "TradingCalendar":
        return cls(exchange=exchange, sessions=pd.bdate_range(start=start, end=end, tz="UTC"))

    def validate_timestamps(self, timestamps: pd.Series | pd.DatetimeIndex) -> None:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).normalize()
        sessions = pd.DatetimeIndex(self.sessions).normalize()
        off = ts.difference(sessions)
        if len(off):
            raise ValueError(f"timestamp(s) off trading calendar {self.exchange}: {[x.isoformat() for x in off[:5]]}")


def load_calendar(path: str | Path, mapping: CalendarColumnMap, *, exchange: str) -> TradingCalendar:
    mapped = read_mapped_csv(path, mapping, required=("timestamp",))
    return TradingCalendar(exchange=exchange, sessions=pd.to_datetime(mapped["timestamp"], utc=True))


def align_close_panel_with_tradable(
    ohlcv: pd.DataFrame,
    *,
    symbols: list[str],
    calendar: TradingCalendar,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = ohlcv.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    calendar.validate_timestamps(df["timestamp"])
    raw = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    raw = raw.reindex(index=calendar.sessions, columns=list(symbols))
    tradable = raw.notna()
    filled = raw.ffill().bfill()
    bad = filled.columns[filled.isna().any()].tolist()
    if bad:
        raise ValueError(f"no usable close prices for symbol(s): {bad}")
    if (filled <= 0).any().any():
        raise ValueError("aligned close prices must be strictly positive")
    return filled.astype(float), tradable.astype(bool)

