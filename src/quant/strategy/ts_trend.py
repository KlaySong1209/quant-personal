"""Simple educational time-series trend strategy. Not real alpha."""

from __future__ import annotations

import pandas as pd

from quant.strategy.base import Strategy


class TimeSeriesTrendStrategy(Strategy):
    def __init__(self, lookback: int = 60, target_weight: float = 1.0):
        self.lookback = int(lookback)
        self.target_weight = float(target_weight)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        momentum = prices.pct_change(self.lookback)
        weights = (momentum > 0).astype(float) * self.target_weight
        weights[momentum < 0] = -self.target_weight
        return weights.fillna(0.0)

