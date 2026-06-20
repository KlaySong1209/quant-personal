"""Virtual paper broker. No real broker integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
            if (not np.isfinite(float(px))) or px <= 0:
                raise ValueError(f"price for {sym} must be finite and positive: {px}")
            self._last_prices[sym] = float(px)

    def _validate_target(self, target_shares: dict[str, float]) -> dict[str, float]:
        final = dict(self._positions)
        for sym, target in target_shares.items():
            if not np.isfinite(float(target)):
                raise PaperBrokerError(f"target shares must be finite for {sym}: {target}")
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

    def apply_cash_fee(self, timestamp: pd.Timestamp, amount: float, *, label: str = "SIMULATED_COST") -> None:
        fee = float(amount)
        if not np.isfinite(fee) or fee < 0:
            raise ValueError("cash fee must be finite and non-negative")
        if fee == 0:
            return
        self._cash -= fee
        self._events.append(_Event(pd.Timestamp(timestamp), label, 0.0, 0.0, -fee, self._cash))

    def apply_cash_adjustment(
        self,
        timestamp: pd.Timestamp,
        amount: float,
        *,
        label: str = "CORPORATE_ACTION",
    ) -> None:
        """Apply a corporate-action cash flow (positive or negative)."""
        adj = float(amount)
        if not np.isfinite(adj):
            raise ValueError("cash adjustment must be finite")
        if adj == 0:
            return
        self._cash += adj
        self._events.append(_Event(pd.Timestamp(timestamp), label, 0.0, 0.0, adj, self._cash))

    def apply_share_adjustment(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        shares_delta: float,
        *,
        label: str = "CORPORATE_ACTION",
    ) -> None:
        """Adjust share count for a corporate action without a buy/sell trade."""
        delta = float(shares_delta)
        if not np.isfinite(delta):
            raise ValueError("share adjustment must be finite")
        if delta == 0:
            return
        if symbol not in self._last_prices:
            raise PaperBrokerError(f"no known price for {symbol}")
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta
        self._events.append(_Event(pd.Timestamp(timestamp), symbol, delta, 0.0, 0.0, self._cash))

    def mark_to_market(self, timestamp: pd.Timestamp) -> float:
        del timestamp
        return float(self._cash + sum(q * self._last_prices[s] for s, q in self._positions.items()))

    def ledger(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self._events])

    def is_balanced(self, tol: float = 1e-6) -> bool:
        ledger = self.ledger()
        delta = float(ledger["cash_delta"].sum()) if not ledger.empty else 0.0
        return abs(self.starting_cash + delta - self._cash) < tol

    def to_dict(self) -> dict:
        return {
            "starting_cash": self.starting_cash,
            "allow_short": self.allow_short,
            "allow_margin": self.allow_margin,
            "max_gross_leverage": self.max_gross_leverage,
            "cash": self._cash,
            "positions": dict(self._positions),
            "last_prices": dict(self._last_prices),
            "events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "symbol": event.symbol,
                    "shares_delta": event.shares_delta,
                    "price": event.price,
                    "cash_delta": event.cash_delta,
                    "cash_after": event.cash_after,
                }
                for event in self._events
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperBroker":
        for key in ("starting_cash", "cash"):
            if not np.isfinite(float(data[key])):
                raise ValueError(f"non-finite broker state value: {key}")
        broker = cls(
            float(data["starting_cash"]),
            allow_short=bool(data.get("allow_short", False)),
            allow_margin=bool(data.get("allow_margin", False)),
            max_gross_leverage=float(data.get("max_gross_leverage", 1.0)),
        )
        broker._cash = float(data["cash"])
        broker._positions = {str(k): _finite_float(v, f"position {k}") for k, v in data.get("positions", {}).items()}
        broker._last_prices = {str(k): _finite_float(v, f"last price {k}") for k, v in data.get("last_prices", {}).items()}
        broker._events = [
            _Event(
                timestamp=pd.Timestamp(event["timestamp"]),
                symbol=str(event["symbol"]),
                shares_delta=_finite_float(event["shares_delta"], "event shares_delta"),
                price=_finite_float(event["price"], "event price"),
                cash_delta=_finite_float(event["cash_delta"], "event cash_delta"),
                cash_after=_finite_float(event["cash_after"], "event cash_after"),
            )
            for event in data.get("events", [])
        ]
        return broker


def _finite_float(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"non-finite broker state value: {name}")
    return out
