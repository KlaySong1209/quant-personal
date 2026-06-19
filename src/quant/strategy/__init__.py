"""Strategies."""

from __future__ import annotations

from quant.strategy.placeholder import PlaceholderStrategy
from quant.strategy.ts_trend import TimeSeriesTrendStrategy


def build_strategy(name: str, params: dict, *, specs: dict | None = None):
    if name == "placeholder":
        return PlaceholderStrategy(**params)
    if name == "ts_trend":
        return TimeSeriesTrendStrategy(**params)
    raise ValueError(f"unknown strategy: {name}")

