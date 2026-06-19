"""Configurable loader for local daily futures contract exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant.data.adjust.loaders import read_mapped_csv
from quant.data.futures_roll import ContractSeries, ROLL_RULES, build_continuous_contract


@dataclass(frozen=True)
class FuturesColumnMap:
    contract: str
    timestamp: str
    expiry: str
    open: str
    high: str
    low: str
    close: str
    volume: str


def load_contract_series_from_file(path: str | Path, mapping: FuturesColumnMap) -> list[ContractSeries]:
    df = read_mapped_csv(
        path,
        mapping,
        required=("contract", "timestamp", "expiry", "open", "high", "low", "close", "volume"),
    )
    df["contract"] = df["contract"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    contracts = []
    for code, sub in df.groupby("contract", sort=False):
        sub = sub.sort_values("timestamp").copy()
        contracts.append(
            ContractSeries(
                code=str(code),
                expiry=pd.Timestamp(sub["expiry"].max()),
                df=sub[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True),
            )
        )
    return sorted(contracts, key=lambda c: c.expiry)


def continuous_from_local_file(
    path: str | Path,
    mapping: FuturesColumnMap,
    *,
    continuous_symbol: str,
    roll_rule: ROLL_RULES = "calendar",
    calendar_days_before_expiry: int = 5,
):
    return build_continuous_contract(
        load_contract_series_from_file(path, mapping),
        method="back_adjusted",
        rule=roll_rule,
        calendar_days_before_expiry=calendar_days_before_expiry,
        symbol=continuous_symbol,
    )

