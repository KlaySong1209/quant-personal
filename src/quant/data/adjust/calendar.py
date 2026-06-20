"""Trading calendar validation and alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd

from quant.data.adjust.loaders import read_mapped_csv


@dataclass(frozen=True)
class CalendarColumnMap:
    timestamp: str
    is_open: str | None = None


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


def load_calendar(
    path: str | Path,
    mapping: CalendarColumnMap,
    *,
    exchange: str,
    date_format: str | None = None,
    timezone: str = "UTC",
) -> TradingCalendar:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"calendar file not found: {p}")
    mapped = read_mapped_csv(p, mapping, required=("timestamp",), optional=("is_open",))
    if "is_open" in mapped.columns:
        open_mask = mapped["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes", "y", "open"})
        mapped = mapped.loc[open_mask].copy()
    parsed = pd.to_datetime(mapped["timestamp"], format=date_format, errors="raise")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(timezone)
    parsed = parsed.dt.tz_convert("UTC")
    return TradingCalendar(exchange=exchange, sessions=parsed)


def build_trading_calendar(
    *,
    mode: str = "synthetic",
    file: str | Path | None = None,
    column_mapping: dict[str, str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    exchange: str = "SYNTH",
    date_format: str | None = None,
    timezone: str = "UTC",
    production_data: bool = False,
) -> TradingCalendar:
    if mode == "file":
        if file is None:
            raise ValueError("calendar file required when calendar mode is file")
        mapping = _calendar_mapping(column_mapping or {})
        return load_calendar(file, mapping, exchange=exchange, date_format=date_format, timezone=timezone)
    if mode == "synthetic":
        if production_data:
            raise ValueError("synthetic calendar is forbidden when production_data is true")
        if start is None or end is None:
            raise ValueError("start and end required for synthetic calendar")
        warnings.warn(
            "Using synthetic business-day calendar; A-share holidays are not included.",
            RuntimeWarning,
            stacklevel=2,
        )
        return TradingCalendar.synthetic(start, end, exchange=exchange)
    raise ValueError(f"unknown calendar mode: {mode}")


def _calendar_mapping(mapping: dict[str, str]) -> CalendarColumnMap:
    if "timestamp" in mapping:
        return CalendarColumnMap(timestamp=mapping["timestamp"], is_open=mapping.get("is_open"))
    if "date" in mapping:
        return CalendarColumnMap(timestamp=mapping["date"], is_open=mapping.get("is_open"))
    return CalendarColumnMap(timestamp="date", is_open=mapping.get("is_open"))


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
