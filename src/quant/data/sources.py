"""Data sources: local CSV and seeded synthetic daily OHLCV."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data.schema import OHLCV_COLUMNS


class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError


class CSVSource(DataSource):
    def __init__(self, csv_path: str | Path, symbols: list[str]):
        if not symbols:
            raise ValueError("symbols must be non-empty")
        self.csv_path = Path(csv_path)
        self.symbols = list(symbols)

    def load(self) -> pd.DataFrame:
        frames = []
        for sym in self.symbols:
            path = self.csv_path / f"{sym}.csv"
            if not path.exists():
                raise FileNotFoundError(f"CSV not found for {sym}: {path}")
            df = pd.read_csv(path)
            if "symbol" not in df.columns:
                df["symbol"] = sym
            frames.append(df.loc[:, list(OHLCV_COLUMNS)])
        return pd.concat(frames, ignore_index=True)


class SyntheticSource(DataSource):
    def __init__(
        self,
        symbols: list[str],
        start: str,
        end: str,
        *,
        initial_price: float = 100.0,
        annual_drift: float = 0.05,
        annual_vol: float = 0.20,
        seed: int = 42,
    ):
        if not symbols:
            raise ValueError("symbols must be non-empty")
        self.symbols = list(symbols)
        self.start = start
        self.end = end
        self.initial_price = float(initial_price)
        self.annual_drift = float(annual_drift)
        self.annual_vol = float(annual_vol)
        self.seed = int(seed)

    def load(self) -> pd.DataFrame:
        idx = pd.date_range(self.start, self.end, freq="B", tz="UTC")
        if len(idx) == 0:
            raise ValueError("empty date range")
        dt = 1.0 / 252.0
        master = np.random.default_rng(self.seed)
        seeds = master.integers(0, 2**31 - 1, size=len(self.symbols))
        frames = []
        for sym, seed in zip(self.symbols, seeds):
            rng = np.random.default_rng(int(seed))
            shocks = rng.standard_normal(len(idx))
            log_r = (
                (self.annual_drift - 0.5 * self.annual_vol**2) * dt
                + self.annual_vol * np.sqrt(dt) * shocks
            )
            close = self.initial_price * np.exp(np.cumsum(log_r))
            band = abs(rng.standard_normal(len(idx))) * self.annual_vol * np.sqrt(dt) * close
            open_ = np.empty(len(idx))
            open_[0] = self.initial_price
            open_[1:] = close[:-1]
            high = np.maximum(open_, close) + band
            low = np.maximum(np.minimum(open_, close) - band, 1e-6)
            volume = rng.integers(1_000, 10_000, size=len(idx)).astype(float)
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": idx,
                        "symbol": sym,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

