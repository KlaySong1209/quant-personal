"""Virtual paper broker. No real broker integration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.execution.broker import BrokerAdapter


class PaperBrokerError(Exception):
    pass


@dataclass
class _Event:
    timestamp: pd.Timestamp
    symbol: str
    shares_delta: float
    price: float
    cash_delta: float
    cash_after: float


class PaperBroker(BrokerAdapter):
    def __init__(
        self,
        starting_cash: float,
        *,
        allow_short: bool = False,
        allow_margin: bool = False,
        max_gross_leverage: float = 1.0,
    ):
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")
        self.starting_cash = float(starting_cash)
        self.allow_short = bool(allow_short)
        self.allow_margin = bool(allow_margin)
        self.max_gross_leverage = float(max_gross_leverage)
        self._cash = float(starting_cash)
        self._positions: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}
        self._events: list[_Event] = []

    @property
    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def update_prices(self, prices: dict[str, float]) -> None:
        for sym, px in prices.items():
            if px <= 0:
                raise ValueError(f"non-positive price for {sym}: {px}")
            self._last_prices[sym] = float(px)

    def _validate_target(self, target_shares: dict[str, float]) -> dict[str, float]:
        final = dict(self._positions)
        for sym, target in target_shares.items():
            final[sym] = float(target)
        if not self.allow_short:
            shorts = {s: q for s, q in final.items() if q < 0}
            if shorts:
                raise PaperBrokerError(f"short positions disallowed: {shorts}")
        cash_after = self._cash
        for sym, target in final.items():
            if sym not in self._last_prices:
                raise PaperBrokerError(f"no known price for {sym}")
            cash_after -= (target - self._positions.get(sym, 0.0)) * self._last_prices[sym]
        if not self.allow_margin and cash_after < -1e-9:
            raise PaperBrokerError("target basket would overdraft cash")
        equity_after = cash_after + sum(q * self._last_prices[s] for s, q in final.items())
        if equity_after <= 0:
            raise PaperBrokerError("target basket would non-positive equity")
        gross = sum(abs(q * self._last_prices[s]) for s, q in final.items())
        if gross / equity_after > self.max_gross_leverage + 1e-9:
            raise PaperBrokerError("target basket gross leverage exceeds max")
        return final

    def submit_target(
        self,
        timestamp: pd.Timestamp,
        target_shares: dict[str, float],
        prices: dict[str, float] | None = None,
    ) -> None:
        if prices is not None:
            self.update_prices(prices)
        final = self._validate_target(target_shares)
        for sym, target in final.items():
            current = self._positions.get(sym, 0.0)
            delta = target - current
            if delta == 0:
                continue
            price = self._last_prices[sym]
            cash_delta = -delta * price
            self._cash += cash_delta
            self._positions[sym] = target
            self._events.append(_Event(pd.Timestamp(timestamp), sym, delta, price, cash_delta, self._cash))

    def mark_to_market(self, timestamp: pd.Timestamp) -> float:
        del timestamp
        return float(self._cash + sum(q * self._last_prices[s] for s, q in self._positions.items()))

    def ledger(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self._events])

    def is_balanced(self, tol: float = 1e-6) -> bool:
        ledger = self.ledger()
        delta = float(ledger["cash_delta"].sum()) if not ledger.empty else 0.0
        return abs(self.starting_cash + delta - self._cash) < tol

