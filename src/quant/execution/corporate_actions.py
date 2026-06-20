"""Account-layer corporate action application.

This module applies corporate-action events to a simulated account on the ex-date,
*before* trading, so that equity does not falsely jump.

Price-level backward adjustment lives in `quant.data.adjust.corporate_actions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant.data.adjust.corporate_actions import (
    CorporateAction,
    corporate_actions_for_date,
)


class CorporateActionError(Exception):
    """Raised when a corporate action cannot be applied."""


@dataclass(frozen=True)
class AppliedCorporateAction:
    """Record of a corporate action applied at the account layer."""
    event_key: str
    timestamp: pd.Timestamp
    symbol: str
    action_type: str
    shares_delta: float
    cash_delta: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_key": self.event_key,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action_type": self.action_type,
            "shares_delta": self.shares_delta,
            "cash_delta": self.cash_delta,
            "note": self.note,
        }


def apply_corporate_action(
    ca: CorporateAction,
    positions: dict[str, float],
    cash: float,
) -> tuple[dict[str, float], float, AppliedCorporateAction]:
    """Apply a single corporate action to positions and cash.

    Returns (updated_positions, updated_cash, record).

    Rules:
      - cash_dividend: cash += shares * cash_per_share
      - stock_dividend / capitalization: shares += shares * share_ratio
      - split: shares *= split_ratio
      - reverse_split: shares *= split_ratio (where split_ratio < 1)
      - rights_issue: if participate=True, cash -= new_shares * subscription_price,
        shares += shares * subscription_ratio
    """
    current_shares = positions.get(ca.symbol, 0.0)
    shares_delta = 0.0
    cash_delta = 0.0
    note = ""

    if ca.action_type == "cash_dividend":
        if current_shares <= 0:
            note = f"no position for {ca.symbol}, cash dividend skipped"
            return dict(positions), cash, AppliedCorporateAction(
                event_key=ca.event_key, timestamp=ca.timestamp,
                symbol=ca.symbol, action_type=ca.action_type,
                shares_delta=0.0, cash_delta=0.0, note=note,
            )
        cash_delta = current_shares * ca.cash_per_share
        note = f"cash dividend {ca.cash_per_share}/share x {current_shares} shares"

    elif ca.action_type in ("stock_dividend", "capitalization"):
        if current_shares <= 0:
            note = f"no position for {ca.symbol}, {ca.action_type} skipped"
            return dict(positions), cash, AppliedCorporateAction(
                event_key=ca.event_key, timestamp=ca.timestamp,
                symbol=ca.symbol, action_type=ca.action_type,
                shares_delta=0.0, cash_delta=0.0, note=note,
            )
        shares_delta = current_shares * ca.share_ratio
        note = f"{ca.action_type} {ca.share_ratio}/share x {current_shares} shares"

    elif ca.action_type in ("split", "reverse_split"):
        if current_shares <= 0:
            note = f"no position for {ca.symbol}, {ca.action_type} skipped"
            return dict(positions), cash, AppliedCorporateAction(
                event_key=ca.event_key, timestamp=ca.timestamp,
                symbol=ca.symbol, action_type=ca.action_type,
                shares_delta=0.0, cash_delta=0.0, note=note,
            )
        new_shares = current_shares * ca.split_ratio
        shares_delta = new_shares - current_shares
        note = f"{ca.action_type} ratio={ca.split_ratio}: {current_shares} -> {new_shares} shares"

    elif ca.action_type == "rights_issue":
        if current_shares <= 0:
            note = f"no position for {ca.symbol}, rights_issue skipped"
            return dict(positions), cash, AppliedCorporateAction(
                event_key=ca.event_key, timestamp=ca.timestamp,
                symbol=ca.symbol, action_type=ca.action_type,
                shares_delta=0.0, cash_delta=0.0, note=note,
            )
        if not ca.participate:
            note = f"rights_issue for {ca.symbol} not participated; no change"
            return dict(positions), cash, AppliedCorporateAction(
                event_key=ca.event_key, timestamp=ca.timestamp,
                symbol=ca.symbol, action_type=ca.action_type,
                shares_delta=0.0, cash_delta=0.0, note=note,
            )
        new_shares = current_shares * ca.subscription_ratio
        cost = new_shares * ca.subscription_price
        shares_delta = new_shares
        cash_delta = -cost
        note = f"rights_issue participate: +{new_shares} shares, -{cost} cash"

    new_positions = dict(positions)
    if shares_delta != 0.0:
        new_positions[ca.symbol] = current_shares + shares_delta

    return new_positions, cash + cash_delta, AppliedCorporateAction(
        event_key=ca.event_key, timestamp=ca.timestamp,
        symbol=ca.symbol, action_type=ca.action_type,
        shares_delta=shares_delta, cash_delta=cash_delta, note=note,
    )


def apply_corporate_actions_for_date(
    actions: list[CorporateAction],
    timestamp: pd.Timestamp,
    positions: dict[str, float],
    cash: float,
    *,
    applied_keys: set[str] | None = None,
) -> tuple[dict[str, float], float, list[AppliedCorporateAction]]:
    """Apply all corporate actions for a given date, before trading.

    Returns (new_positions, new_cash, records).
    Uses *applied_keys* for idempotency — each event_key is only applied once.
    """
    applied = applied_keys or set()
    records: list[AppliedCorporateAction] = []
    new_positions = dict(positions)
    new_cash = cash

    for ca in actions:
        if ca.event_key in applied:
            continue
        if ca.timestamp.normalize() != timestamp.normalize():
            continue
        new_positions, new_cash, record = apply_corporate_action(ca, new_positions, new_cash)
        records.append(record)

    return new_positions, new_cash, records
