"""Educational placeholder strategy. Not real alpha or investment advice."""

from __future__ import annotations

import pandas as pd

from quant.strategy.base import Strategy


class PlaceholderStrategy(Strategy):
    """Trivial wiring placeholder: equal-weight hold across available symbols."""

    def __init__(self, mode: str = "equal_weight"):
        if mode not in {"equal_weight", "all_flat"}:
            raise ValueError("placeholder mode must be 'equal_weight' or 'all_flat'")
        self.mode = mode

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "all_flat":
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if prices.shape[1] == 0:
            raise ValueError("prices must contain at least one symbol")
        weight = 1.0 / prices.shape[1]
        return pd.DataFrame(weight, index=prices.index, columns=prices.columns)
