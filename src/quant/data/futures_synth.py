"""Synthetic futures contracts for local tests and demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.data.futures_roll import ContractSeries
from quant.data.sources import DataSource


@dataclass
class SyntheticFuturesSource(DataSource):
    symbol_root: str
    start: str
    n_contracts: int = 4
    days_per_contract: int = 60
    overlap_days: int = 10
    seed: int = 42
    initial_underlying_price: float = 500.0
    annual_drift: float = 0.02
    annual_vol: float = 0.15
    basis_per_contract: float = 2.0

    def load(self) -> pd.DataFrame:
        return pd.concat(
            [c.df.assign(symbol=c.code) for c in self.contracts()], ignore_index=True
        )[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

    def contracts(self) -> list[ContractSeries]:
        if self.overlap_days >= self.days_per_contract:
            raise ValueError("overlap_days must be < days_per_contract")
        total_days = self.days_per_contract + (self.n_contracts - 1) * (
            self.days_per_contract - self.overlap_days
        )
        idx = pd.bdate_range(self.start, periods=total_days, tz="UTC")
        rng = np.random.default_rng(self.seed)
        dt = 1.0 / 252.0
        log_r = (
            (self.annual_drift - 0.5 * self.annual_vol**2) * dt
            + self.annual_vol * np.sqrt(dt) * rng.standard_normal(total_days)
        )
        underlying = self.initial_underlying_price * np.exp(np.cumsum(log_r))
        step = self.days_per_contract - self.overlap_days
        out = []
        for i in range(self.n_contracts):
            start = i * step
            end = min(start + self.days_per_contract, total_days)
            sub_idx = idx[start:end]
            close = underlying[start:end] + i * self.basis_per_contract
            open_ = np.r_[close[0], close[:-1]]
            high = np.maximum(open_, close)
            low = np.minimum(open_, close)
            volume = rng.integers(5_000, 50_000, size=len(close)).astype(float)
            df = pd.DataFrame(
                {
                    "timestamp": sub_idx,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
            out.append(ContractSeries(code=f"{self.symbol_root}_C{i+1:02d}", expiry=sub_idx[-1], df=df))
        return out

