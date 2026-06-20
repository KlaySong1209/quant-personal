"""Corporate-action declarations for price adjustment and account-layer events.

Corporate actions are declared as standardized events with UTC timestamps.
The same declarations drive both (a) backward price adjustment and (b) account-layer
cash/share corrections so that ex-date equity does not falsely jump.

See docs/data_ingestion.md for how user exports map into these fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant.data.adjust.loaders import read_mapped_csv

# ---------------------------------------------------------------------------
# Backward-compatible price-adjustment API (used by experiment/run.py, tests)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# New standardized corporate action model (account-layer + price adjustment)
# ---------------------------------------------------------------------------

ActionType = Literal[
    "cash_dividend",
    "stock_dividend",
    "capitalization",
    "split",
    "reverse_split",
    "rights_issue",
]


@dataclass(frozen=True)
class CorporateAction:
    """One corporate action event, keyed by (timestamp, symbol, action_type).

    All timestamps are UTC ex-dates.
    """

    timestamp: pd.Timestamp
    symbol: str
    action_type: ActionType

    # --- cash dividend ---
    cash_per_share: float = 0.0

    # --- stock dividend / capitalization (e.g. 10-for-10 is share_ratio=1.0) ---
    share_ratio: float = 0.0

    # --- split / reverse split (2-for-1 split is split_ratio=2.0) ---
    split_ratio: float = 1.0

    # --- rights issue ---
    subscription_ratio: float = 0.0
    subscription_price: float = 0.0
    participate: bool = False  # whether the account holder chose to participate

    tax_treatment: Literal["pre_tax", "post_tax", "none"] = "none"
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, pd.Timestamp):
            object.__setattr__(self, "timestamp", pd.Timestamp(self.timestamp))
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, "timestamp", self.timestamp.tz_localize("UTC"))
        else:
            object.__setattr__(self, "timestamp", self.timestamp.tz_convert("UTC"))
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        valid_types = {
            "cash_dividend", "stock_dividend", "capitalization",
            "split", "reverse_split", "rights_issue",
        }
        if self.action_type not in valid_types:
            raise ValueError(f"action_type must be one of {sorted(valid_types)}: {self.action_type}")
        # Per-type validation
        if self.action_type == "cash_dividend":
            if self.cash_per_share <= 0:
                raise ValueError(f"cash_dividend cash_per_share must be positive: {self.cash_per_share}")
        elif self.action_type in ("stock_dividend", "capitalization"):
            if self.share_ratio <= 0:
                raise ValueError(f"{self.action_type} share_ratio must be positive: {self.share_ratio}")
        elif self.action_type == "split":
            if self.split_ratio <= 1.0:
                raise ValueError(f"split split_ratio must be > 1.0: {self.split_ratio}")
        elif self.action_type == "reverse_split":
            if self.split_ratio <= 0 or self.split_ratio >= 1.0:
                raise ValueError(f"reverse_split split_ratio must be 0 < ratio < 1.0: {self.split_ratio}")
        elif self.action_type == "rights_issue":
            if self.subscription_ratio <= 0 or self.subscription_price <= 0:
                raise ValueError(f"rights_issue requires positive subscription_ratio and subscription_price")

    @property
    def event_key(self) -> str:
        """Deterministic, idempotent key for dedup at account layer."""
        ts = self.timestamp.isoformat()
        return f"{ts}::{self.symbol}::{self.action_type}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorporateAction":
        return cls(
            timestamp=pd.Timestamp(data["timestamp"]),
            symbol=str(data["symbol"]),
            action_type=data["action_type"],
            cash_per_share=float(data.get("cash_per_share", 0.0)),
            share_ratio=float(data.get("share_ratio", 0.0)),
            split_ratio=float(data.get("split_ratio", 1.0)),
            subscription_ratio=float(data.get("subscription_ratio", 0.0)),
            subscription_price=float(data.get("subscription_price", 0.0)),
            participate=bool(data.get("participate", False)),
            tax_treatment=data.get("tax_treatment", "none"),
            source=str(data.get("source", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action_type": self.action_type,
            "cash_per_share": self.cash_per_share,
            "share_ratio": self.share_ratio,
            "split_ratio": self.split_ratio,
            "subscription_ratio": self.subscription_ratio,
            "subscription_price": self.subscription_price,
            "participate": self.participate,
            "tax_treatment": self.tax_treatment,
            "source": self.source,
        }


def validate_corporate_actions(actions: list[CorporateAction]) -> None:
    """Fail-fast on missing required fields or invalid values."""
    seen_keys: set[str] = set()
    for ca in actions:
        if not isinstance(ca.timestamp, pd.Timestamp):
            raise ValueError(f"corporate action missing timestamp: {ca}")
        if not ca.symbol:
            raise ValueError(f"corporate action missing symbol: {ca}")
        if ca.action_type not in {
            "cash_dividend", "stock_dividend", "capitalization",
            "split", "reverse_split", "rights_issue",
        }:
            raise ValueError(f"corporate action has unknown action_type: {ca.action_type}")

        if ca.action_type == "cash_dividend":
            if ca.cash_per_share <= 0:
                raise ValueError(f"cash_dividend cash_per_share must be positive: {ca}")
        elif ca.action_type in ("stock_dividend", "capitalization"):
            if ca.share_ratio <= 0:
                raise ValueError(f"{ca.action_type} share_ratio must be positive: {ca}")
        elif ca.action_type in ("split", "reverse_split"):
            if ca.split_ratio <= 0:
                raise ValueError(f"{ca.action_type} split_ratio must be positive: {ca}")
            if ca.action_type == "reverse_split" and ca.split_ratio >= 1.0:
                raise ValueError(f"reverse_split split_ratio must be < 1.0: {ca}")
            if ca.action_type == "split" and ca.split_ratio <= 1.0:
                raise ValueError(f"split split_ratio must be > 1.0: {ca}")
        elif ca.action_type == "rights_issue":
            if ca.subscription_ratio <= 0 or ca.subscription_price <= 0:
                raise ValueError(f"rights_issue requires positive subscription_ratio and subscription_price: {ca}")

        key = ca.event_key
        if key in seen_keys:
            raise ValueError(f"duplicate corporate action event key: {key}")
        seen_keys.add(key)


def corporate_actions_for_date(
    actions: list[CorporateAction],
    symbol: str,
    as_of: pd.Timestamp,
) -> list[CorporateAction]:
    """Return corporate actions for *symbol* whose ex-date equals *as_of* (date only)."""
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    as_of_date = as_of.normalize()
    return [
        ca for ca in actions
        if ca.symbol == symbol and ca.timestamp.normalize() == as_of_date
    ]
