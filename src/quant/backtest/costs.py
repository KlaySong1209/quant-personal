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


class AShareCostModel(CostModel):
    """A-share style linear costs with sell-side stamp duty."""

    def __init__(
        self,
        *,
        commission_bps: float = 1.0,
        stamp_duty_bps: float = 5.0,
        slippage_bps: float = 1.0,
        allow_zero_cost_for_tests: bool = False,
    ):
        if commission_bps < 0 or stamp_duty_bps < 0 or slippage_bps < 0:
            raise ValueError("commission_bps, stamp_duty_bps, and slippage_bps must be non-negative")
        if commission_bps + stamp_duty_bps + slippage_bps == 0 and not allow_zero_cost_for_tests:
            raise ValueError("zero-cost AShareCostModel is not allowed")
        self.commission_bps = float(commission_bps)
        self.stamp_duty_bps = float(stamp_duty_bps)
        self.slippage_bps = float(slippage_bps)

    @property
    def buy_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    @property
    def sell_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps + self.stamp_duty_bps) / 10_000.0

    def cost(self, traded_notional: pd.DataFrame) -> pd.Series:
        return self.per_symbol_cost(traded_notional).sum(axis=1)

    def per_symbol_cost(self, traded_notional: pd.DataFrame) -> pd.DataFrame:
        buys = traded_notional.clip(lower=0.0) * self.buy_rate
        sells = (-traded_notional.clip(upper=0.0)) * self.sell_rate
        return buys + sells

    def per_symbol_breakdown(self, traded_notional: pd.Series) -> dict[str, float]:
        buys = traded_notional.clip(lower=0.0)
        sells = -traded_notional.clip(upper=0.0)
        commission = traded_notional.abs().sum() * (self.commission_bps / 10_000.0)
        slippage = traded_notional.abs().sum() * (self.slippage_bps / 10_000.0)
        stamp_duty = sells.sum() * (self.stamp_duty_bps / 10_000.0)
        return {
            "commission": float(commission),
            "slippage": float(slippage),
            "stamp_duty": float(stamp_duty),
            "total": float(commission + slippage + stamp_duty),
            "buy_notional": float(buys.sum()),
            "sell_notional": float(sells.sum()),
        }
