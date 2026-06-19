"""Continuous futures contract construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ROLL_METHODS = Literal["back_adjusted", "none"]
ROLL_RULES = Literal["calendar", "volume_crossover"]


@dataclass(frozen=True)
class RollMetadata:
    method: ROLL_METHODS
    rule: ROLL_RULES
    calendar_days_before_expiry: int = 0
    contract_codes: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "rule": self.rule,
            "calendar_days_before_expiry": self.calendar_days_before_expiry,
            "contract_codes": list(self.contract_codes),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ContractSeries:
    code: str
    expiry: pd.Timestamp
    df: pd.DataFrame


def _roll_dates(series: list[ContractSeries], n_days_before: int) -> dict[str, pd.Timestamp]:
    out = {}
    for s in series[:-1]:
        days = pd.DatetimeIndex(s.df["timestamp"]).sort_values()
        idx = len(days) - 1 - n_days_before
        if idx < 0:
            raise ValueError(f"contract {s.code} has too few days")
        out[s.code] = days[idx]
    return out


def build_continuous_contract(
    series: list[ContractSeries],
    *,
    method: ROLL_METHODS = "back_adjusted",
    rule: ROLL_RULES = "calendar",
    calendar_days_before_expiry: int = 5,
    symbol: str = "CONT",
) -> tuple[pd.DataFrame, RollMetadata]:
    if not series:
        raise ValueError("at least one contract series is required")
    if method != "back_adjusted":
        raise ValueError("continuous contracts require method='back_adjusted'")
    series = sorted(series, key=lambda s: s.expiry)
    if rule != "calendar":
        raise ValueError("only calendar roll is supported in this V1 path")
    roll_dates = _roll_dates(series, calendar_days_before_expiry)
    pieces = [series[-1].df.copy()]
    codes = [series[-1].code]
    cumulative = 0.0
    for i in range(len(series) - 2, -1, -1):
        cur = series[i]
        nxt = series[i + 1]
        roll_date = roll_dates[cur.code]
        cur_window = cur.df[cur.df["timestamp"] <= roll_date].copy()
        nxt_at = nxt.df[nxt.df["timestamp"] >= roll_date].sort_values("timestamp")
        if cur_window.empty or nxt_at.empty:
            raise ValueError("cannot compute roll gap")
        gap = float(nxt_at["close"].iloc[0]) - float(cur_window.sort_values("timestamp")["close"].iloc[-1])
        cumulative += gap
        for col in ("open", "high", "low", "close"):
            cur_window[col] = cur_window[col] + cumulative
        pieces.append(cur_window)
        codes.append(cur.code)
    cont = pd.concat(reversed(pieces), ignore_index=True).sort_values("timestamp")
    cont = cont.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    cont["symbol"] = symbol
    cont = cont[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
    return cont, RollMetadata(
        method="back_adjusted",
        rule=rule,
        calendar_days_before_expiry=calendar_days_before_expiry,
        contract_codes=tuple(reversed(codes)),
    )


def assert_no_implausible_overnight_jumps(
    df: pd.DataFrame, *, max_log_return: float = 0.10, symbol_col: str = "symbol"
) -> None:
    for sym, sub in df.groupby(symbol_col, sort=False):
        sub = sub.sort_values("timestamp")
        log_r = np.log(sub["close"].astype(float)).diff().abs()
        if log_r.max(skipna=True) > max_log_return:
            raise ValueError(f"implausible overnight jump in {sym}")

