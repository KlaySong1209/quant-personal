"""Corporate-action adjustment for local daily equity data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data.adjust.loaders import read_mapped_csv

PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class CorporateActionColumnMap:
    symbol: str
    timestamp: str
    event_type: str
    amount: str | None = None
    ratio: str | None = None


def load_corporate_actions(path: str | Path, mapping: CorporateActionColumnMap) -> pd.DataFrame:
    events = read_mapped_csv(
        path,
        mapping,
        required=("symbol", "timestamp", "event_type"),
        optional=("amount", "ratio"),
    )
    events["symbol"] = events["symbol"].astype(str)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    if "amount" not in events:
        events["amount"] = 0.0
    if "ratio" not in events:
        events["ratio"] = 1.0
    events["amount"] = pd.to_numeric(events["amount"], errors="raise").fillna(0.0)
    events["ratio"] = pd.to_numeric(events["ratio"], errors="raise").fillna(1.0)
    bad = set(events["event_type"]) - {"cash_dividend", "split"}
    if bad:
        raise ValueError(f"unsupported corporate action type(s): {sorted(bad)}")
    return events[["timestamp", "symbol", "event_type", "amount", "ratio"]].sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)


def _event_factor(sub: pd.DataFrame, ts: pd.Timestamp, event_type: str, amount: float, ratio: float) -> float:
    if event_type == "split":
        if ratio <= 0:
            raise ValueError("split ratio must be positive")
        return 1.0 / ratio
    if event_type == "cash_dividend":
        prior = sub[sub["timestamp"] < ts].sort_values("timestamp")
        if prior.empty:
            raise ValueError(f"cash dividend at {ts} has no prior close")
        prev_close = float(prior["raw_close"].iloc[-1])
        if amount < 0 or amount >= prev_close:
            raise ValueError("cash dividend amount must be non-negative and smaller than prior close")
        return (prev_close - amount) / prev_close
    raise ValueError(f"unsupported corporate action type: {event_type}")


def adjust_ohlcv_for_corporate_actions(ohlcv: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = ohlcv.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["symbol"] = out["symbol"].astype(str)
    for col in PRICE_COLUMNS:
        out[f"raw_{col}"] = out[col].astype(float)
        out[f"adjusted_{col}"] = out[f"raw_{col}"]
    out["adjustment_applied"] = False
    if events.empty:
        return out
    ev = events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"], utc=True)
    parts = []
    for sym, sub in out.groupby("symbol", sort=False):
        sub = sub.sort_values("timestamp").copy()
        for _, event in ev[ev["symbol"] == sym].sort_values("timestamp").iterrows():
            ts = pd.Timestamp(event["timestamp"])
            factor = _event_factor(
                sub,
                ts,
                str(event["event_type"]),
                float(event.get("amount", 0.0)),
                float(event.get("ratio", 1.0)),
            )
            mask = sub["timestamp"] < ts
            for col in PRICE_COLUMNS:
                sub.loc[mask, f"adjusted_{col}"] *= factor
            sub.loc[mask, "adjustment_applied"] = True
        for col in PRICE_COLUMNS:
            sub[col] = sub[f"adjusted_{col}"]
        parts.append(sub)
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def flag_implausible_unadjusted_jumps(ohlcv: pd.DataFrame, *, max_log_return: float = 0.10) -> None:
    df = ohlcv.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for sym, sub in df.groupby("symbol", sort=False):
        sub = sub.sort_values("timestamp")
        log_r = np.log(sub["close"].astype(float)).diff().abs()
        if log_r.max(skipna=True) > max_log_return:
            pos = int(log_r.to_numpy().argmax())
            raise ValueError(
                f"implausible overnight jump for {sym} at {sub['timestamp'].iloc[pos]} "
                f"({float(log_r.iloc[pos]):.4f} > {max_log_return:.4f})"
            )

