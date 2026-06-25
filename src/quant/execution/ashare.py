"""A-share daily execution rules for local simulation only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest.costs import CostModel

EXECUTION_ADJUSTMENTS = ("none", "hfq")


@dataclass(frozen=True)
class AShareExecutionConfig:
    lot_size: int = 100
    tick_size: float = 0.01
    max_gross_leverage: float = 1.0


@dataclass(frozen=True)
class ExecutionResult:
    cash: float
    positions: dict[str, float]
    available_shares: dict[str, float]
    ledger_events: list[dict[str, Any]]
    fill_results: list[dict[str, Any]]


def validate_execution_adjustment(adjustment: str) -> str:
    if adjustment == "qfq":
        raise ValueError("qfq prices are not valid for account/matching simulation")
    if adjustment not in EXECUTION_ADJUSTMENTS:
        raise ValueError(f"execution adjustment must be one of {EXECUTION_ADJUSTMENTS}, got {adjustment!r}")
    return adjustment


def align_to_tick(price: float, tick_size: float = 0.01) -> float:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    return round(round(float(price) / tick_size) * tick_size, 10)


def execute_rebalance(
    *,
    timestamp: pd.Timestamp,
    cash: float,
    positions: dict[str, float],
    available_shares: dict[str, float],
    target_weights: dict[str, float],
    fill_prices: dict[str, float],
    mark_prices: dict[str, float] | None,
    cost_model: CostModel,
    config: AShareExecutionConfig = AShareExecutionConfig(),
    can_buy: dict[str, bool] | None = None,
    can_sell: dict[str, bool] | None = None,
    order_id: str | None = None,
) -> ExecutionResult:
    if cash < -1e-9:
        raise ValueError("cash cannot be negative before execution")
    if config.lot_size <= 0:
        raise ValueError("lot_size must be positive")
    symbols = [str(s) for s in fill_prices]
    prices = {str(s): align_to_tick(px, config.tick_size) for s, px in fill_prices.items()}
    marks = {str(s): float(px) for s, px in (mark_prices or fill_prices).items()}
    pos = {str(s): float(qty) for s, qty in positions.items()}
    avail = {str(s): float(qty) for s, qty in available_shares.items()}
    current_cash = float(cash)
    ledger: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    equity_base = current_cash + sum(pos.get(sym, 0.0) * prices[sym] for sym in symbols)
    if equity_base <= 0:
        raise ValueError("account equity must be positive before execution")
    gross_target = sum(abs(float(target_weights.get(sym, 0.0))) for sym in symbols)
    if gross_target > config.max_gross_leverage + 1e-9:
        raise ValueError("target weights exceed account gross leverage")

    target_shares = {
        sym: (float(target_weights.get(sym, 0.0)) * equity_base) / prices[sym]
        for sym in symbols
    }
    deltas = {sym: target_shares[sym] - pos.get(sym, 0.0) for sym in symbols}

    def add_event(
        *,
        symbol: str,
        shares_delta: float,
        price: float,
        cash_delta: float,
        status: str,
        reason: str = "",
        fee: float = 0.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        stamp_duty: float = 0.0,
    ) -> None:
        nonlocal current_cash
        current_cash += cash_delta
        event = {
            "timestamp": pd.Timestamp(timestamp),
            "symbol": str(symbol),
            "shares_delta": float(shares_delta),
            "price": float(price),
            "cash_delta": float(cash_delta),
            "cash_after": float(current_cash),
            "status": status,
            "reason": reason,
            "order_id": order_id or "",
            "fee": float(fee),
            "commission": float(commission),
            "slippage": float(slippage),
            "stamp_duty": float(stamp_duty),
        }
        ledger.append(event)
        results.append(
            {
                "order_id": order_id,
                "symbol": str(symbol),
                "status": status,
                "reason": reason,
                "shares_delta": float(shares_delta),
                "price": float(price),
                "cash_delta": float(cash_delta),
                "fee": float(fee),
                "commission": float(commission),
                "slippage": float(slippage),
                "stamp_duty": float(stamp_duty),
            }
        )

    for sym in symbols:
        delta = deltas[sym]
        if delta >= -1e-9:
            continue
        price = prices[sym]
        if can_sell is not None and not can_sell.get(sym, True):
            add_event(symbol=sym, shares_delta=0.0, price=price, cash_delta=0.0, status="rejected", reason="limit_down_no_sell")
            continue
        sell_qty = float(np.floor(abs(delta) + 1e-9))
        available = avail.get(sym, 0.0)
        if sell_qty > available + 1e-9:
            add_event(symbol=sym, shares_delta=0.0, price=price, cash_delta=0.0, status="rejected", reason="insufficient_available_shares")
            continue
        notional = sell_qty * price
        cost = _single_notional_breakdown(cost_model, -notional)
        fee = cost["total"]
        pos[sym] = pos.get(sym, 0.0) - sell_qty
        avail[sym] = max(available - sell_qty, 0.0)
        add_event(
            symbol=sym,
            shares_delta=-sell_qty,
            price=price,
            cash_delta=notional - fee,
            status="filled",
            fee=fee,
            commission=cost.get("commission", 0.0),
            slippage=cost.get("slippage", 0.0),
            stamp_duty=cost.get("stamp_duty", 0.0),
        )

    for sym in symbols:
        delta = deltas[sym]
        if delta <= 1e-9:
            continue
        price = prices[sym]
        if can_buy is not None and not can_buy.get(sym, True):
            add_event(symbol=sym, shares_delta=0.0, price=price, cash_delta=0.0, status="rejected", reason="limit_up_no_buy")
            continue
        buy_qty = float(np.floor(delta / config.lot_size) * config.lot_size)
        if buy_qty <= 0:
            add_event(symbol=sym, shares_delta=0.0, price=price, cash_delta=0.0, status="rejected", reason="below_lot_size")
            continue
        notional = buy_qty * price
        cost = _single_notional_breakdown(cost_model, notional)
        fee = cost["total"]
        required = notional + fee
        if required > current_cash + 1e-9:
            add_event(symbol=sym, shares_delta=0.0, price=price, cash_delta=0.0, status="rejected", reason="insufficient_cash")
            continue
        pos[sym] = pos.get(sym, 0.0) + buy_qty
        avail.setdefault(sym, avail.get(sym, 0.0))
        add_event(
            symbol=sym,
            shares_delta=buy_qty,
            price=price,
            cash_delta=-required,
            status="filled",
            fee=fee,
            commission=cost.get("commission", 0.0),
            slippage=cost.get("slippage", 0.0),
            stamp_duty=cost.get("stamp_duty", 0.0),
        )

    for sym in marks:
        pos.setdefault(sym, pos.get(sym, 0.0))

    return ExecutionResult(
        cash=float(current_cash),
        positions={sym: qty for sym, qty in pos.items() if abs(qty) > 1e-12},
        available_shares={sym: qty for sym, qty in avail.items() if abs(qty) > 1e-12},
        ledger_events=ledger,
        fill_results=results,
    )


def _single_notional_breakdown(cost_model: CostModel, signed_notional: float) -> dict[str, float]:
    frame = pd.Series({"_": float(signed_notional)})
    if hasattr(cost_model, "per_symbol_breakdown"):
        raw = cost_model.per_symbol_breakdown(frame)  # type: ignore[attr-defined]
        return {str(k): float(v) for k, v in raw.items()}
    total = float(cost_model.cost(frame.to_frame().T).iloc[0])
    return {"total": total, "commission": 0.0, "slippage": 0.0, "stamp_duty": 0.0}
