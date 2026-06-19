"""Linear bps cost model."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class CostModel(ABC):
    @abstractmethod
    def cost(self, traded_notional: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def per_symbol_cost(self, traded_notional: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class BpsCostModel(CostModel):
    def __init__(
        self,
        bps: float,
        slippage_bps: float = 0.0,
        spread_bps: float = 0.0,
        *,
        allow_zero_cost_for_tests: bool = False,
    ):
        if bps < 0 or slippage_bps < 0 or spread_bps < 0:
            raise ValueError("bps, slippage_bps, and spread_bps must be non-negative")
        if bps + slippage_bps + spread_bps == 0 and not allow_zero_cost_for_tests:
            raise ValueError("zero-cost BpsCostModel is not allowed")
        self.bps = float(bps)
        self.slippage_bps = float(slippage_bps)
        self.spread_bps = float(spread_bps)

    @property
    def total_rate(self) -> float:
        return (self.bps + self.slippage_bps + self.spread_bps) / 10_000.0

    def cost(self, traded_notional: pd.DataFrame) -> pd.Series:
        return traded_notional.abs().sum(axis=1) * self.total_rate

    def per_symbol_cost(self, traded_notional: pd.DataFrame) -> pd.DataFrame:
        return traded_notional.abs() * self.total_rate

