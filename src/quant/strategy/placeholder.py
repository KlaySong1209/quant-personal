"""Educational placeholder strategy. Not real alpha."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategy.base import Strategy


class PlaceholderStrategy(Strategy):
    def __init__(self, fast_window: int = 10, slow_window: int = 30):
        if fast_window <= 0 or slow_window <= 0 or fast_window >= slow_window:
            raise ValueError("fast_window must be positive and < slow_window")
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast = prices.rolling(self.fast_window, min_periods=self.fast_window).mean()
        slow = prices.rolling(self.slow_window, min_periods=self.slow_window).mean()
        signal = (fast > slow).astype(float)
        counts = signal.sum(axis=1).replace(0, np.nan)
        weights = signal.div(counts, axis=0).fillna(0.0)
        return weights.reindex_like(prices).fillna(0.0)

