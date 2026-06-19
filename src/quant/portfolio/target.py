"""Translate weights to target positions."""

from __future__ import annotations

import pandas as pd


def weights_to_target_dollars(weights: pd.DataFrame, equity: float) -> pd.DataFrame:
    if equity <= 0:
        raise ValueError("equity must be positive")
    return weights.astype(float) * float(equity)


def target_dollars_to_shares(target_dollars: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if not target_dollars.index.equals(prices.index):
        raise ValueError("target_dollars and prices must share an index")
    if list(target_dollars.columns) != list(prices.columns):
        raise ValueError("target_dollars and prices must share columns")
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    return target_dollars / prices

