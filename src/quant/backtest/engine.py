"""Vectorized daily backtest engine with one-period execution shift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.costs import CostModel
from quant.execution.account import SimAccount
from quant.risk.checks import RiskConfig, apply_risk_checks


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    weights_effective: pd.DataFrame
    target_weights: pd.DataFrame
    trades_long: pd.DataFrame
    costs: pd.Series
    equity_curve: pd.Series
    initial_equity: float


def _validate_inputs(prices: pd.DataFrame, target_weights: pd.DataFrame) -> None:
    if not prices.index.equals(target_weights.index):
        raise ValueError("prices and target_weights must share an index")
    if list(prices.columns) != list(target_weights.columns):
        raise ValueError("prices and target_weights must share columns")
    if prices.isna().any().any():
        raise ValueError("prices contain NaN")
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")


def _validate_price_panel(panel: pd.DataFrame, prices: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not panel.index.equals(prices.index):
        raise ValueError(f"{name} index must match prices index")
    if list(panel.columns) != list(prices.columns):
        raise ValueError(f"{name} columns must match prices columns")
    if panel.isna().any().any():
        raise ValueError(f"{name} contains NaN")
    if (panel <= 0).any().any():
        raise ValueError(f"{name} must be strictly positive")
    return panel.astype(float)


def _validate_bool_panel(panel: pd.DataFrame, prices: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not panel.index.equals(prices.index):
        raise ValueError(f"{name} index must match prices index")
    if list(panel.columns) != list(prices.columns):
        raise ValueError(f"{name} columns must match prices columns")
    if not (panel.dtypes == bool).all():
        raise TypeError(f"{name} must have bool dtype for every column")
    return panel


def _apply_tradable_mask(weights_effective: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    arr = weights_effective.to_numpy(copy=True)
    mask = tradable.to_numpy()
    if len(arr) == 0:
        return weights_effective.copy()
    arr[0] = np.where(mask[0], arr[0], 0.0)
    for t in range(1, len(arr)):
        arr[t] = np.where(mask[t], arr[t], arr[t - 1])
    return pd.DataFrame(arr, index=weights_effective.index, columns=weights_effective.columns)


def _build_trades_long(
    weights_effective: pd.DataFrame,
    weight_delta: pd.DataFrame,
    traded_notional: pd.DataFrame,
    per_symbol_cost: pd.DataFrame,
) -> pd.DataFrame:
    prev = weights_effective.shift(1).fillna(0.0)
    frames = []
    for ts in weights_effective.index:
        for sym in weights_effective.columns:
            delta = float(weight_delta.loc[ts, sym])
            if abs(delta) <= 1e-12:
                continue
            frames.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "prev_weight": float(prev.loc[ts, sym]),
                    "weight": float(weights_effective.loc[ts, sym]),
                    "weight_delta": delta,
                    "traded_notional": float(traded_notional.loc[ts, sym]),
                    "cost": float(per_symbol_cost.loc[ts, sym]),
                }
            )
    return pd.DataFrame(
        frames,
        columns=[
            "timestamp",
            "symbol",
            "prev_weight",
            "weight",
            "weight_delta",
            "traded_notional",
            "cost",
        ],
    )


def run_backtest(
    *,
    prices: pd.DataFrame,
    open_prices: pd.DataFrame | None = None,
    target_weights: pd.DataFrame,
    cost_model: CostModel,
    risk: RiskConfig | None,
    initial_equity: float,
    tradable: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    fill_price_rule: str = "vectorized",
    execution_adjustment: str = "none",
    lot_size: int = 100,
    tick_size: float = 0.01,
) -> BacktestResult:
    _validate_inputs(prices, target_weights)
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if fill_price_rule not in {"vectorized", "same_day_close", "next_day_open"}:
        raise ValueError("fill_price_rule must be 'vectorized', 'same_day_close', or 'next_day_open'")
    if tradable is not None:
        tradable = _validate_bool_panel(tradable, prices, name="tradable")
    if universe is not None:
        universe = _validate_bool_panel(universe, prices, name="universe")
    if fill_price_rule in {"same_day_close", "next_day_open"}:
        return _run_account_backtest(
            prices=prices,
            open_prices=open_prices,
            target_weights=target_weights,
            cost_model=cost_model,
            risk=risk,
            initial_equity=initial_equity,
            tradable=tradable,
            universe=universe,
            fill_price_rule=fill_price_rule,
            execution_adjustment=execution_adjustment,
            lot_size=lot_size,
            tick_size=tick_size,
        )

    if risk is None:
        checked_targets = target_weights.astype(float)
    else:
        checked_targets = apply_risk_checks(target_weights, risk)
    weights_effective = checked_targets.shift(1).fillna(0.0)
    if tradable is not None:
        weights_effective = _apply_tradable_mask(weights_effective, tradable)
    if universe is not None:
        weights_effective = weights_effective.where(universe, 0.0)
    apply_risk_checks(weights_effective, risk)

    returns_panel = prices.pct_change().fillna(0.0)
    gross_returns = (weights_effective * returns_panel).sum(axis=1)
    equity_pre_cost = initial_equity * (1.0 + gross_returns).cumprod()
    equity_base = equity_pre_cost.shift(1).fillna(initial_equity)
    delta_w = weights_effective.diff()
    delta_w.iloc[0] = weights_effective.iloc[0]
    traded_notional = delta_w.mul(equity_base, axis=0)
    costs = cost_model.cost(traded_notional)
    cost_drag = costs.div(equity_base.replace(0.0, np.nan)).fillna(0.0)
    portfolio_returns = gross_returns - cost_drag
    equity = initial_equity * (1.0 + portfolio_returns).cumprod()
    per_symbol_cost = cost_model.per_symbol_cost(traded_notional)
    trades = _build_trades_long(weights_effective, delta_w, traded_notional, per_symbol_cost)
    return BacktestResult(
        returns=portfolio_returns,
        weights_effective=weights_effective,
        target_weights=checked_targets,
        trades_long=trades,
        costs=costs,
        equity_curve=equity,
        initial_equity=float(initial_equity),
    )


def _run_account_backtest(
    *,
    prices: pd.DataFrame,
    open_prices: pd.DataFrame | None,
    target_weights: pd.DataFrame,
    cost_model: CostModel,
    risk: RiskConfig | None,
    initial_equity: float,
    tradable: pd.DataFrame | None,
    universe: pd.DataFrame | None,
    fill_price_rule: str,
    execution_adjustment: str,
    lot_size: int,
    tick_size: float,
) -> BacktestResult:
    open_panel = _validate_price_panel(open_prices if open_prices is not None else prices, prices, name="open_prices")
    checked_targets = target_weights.astype(float) if risk is None else apply_risk_checks(target_weights, risk)
    if universe is not None:
        checked_targets = checked_targets.where(universe, 0.0)

    account = SimAccount(
        account_id="backtest-ledger",
        starting_cash=initial_equity,
        fill_price_rule=fill_price_rule,
        allow_zero_cost_for_tests=True,
        execution_adjustment=execution_adjustment,
        lot_size=lot_size,
        tick_size=tick_size,
    )
    account.cost_model = cost_model

    for ts in prices.index:
        can_trade = tradable.loc[ts].to_dict() if tradable is not None else None
        account.step(
            ts,
            prices=prices.loc[ts].to_dict(),
            open_prices=open_panel.loc[ts].to_dict() if fill_price_rule == "next_day_open" else None,
            target_weights=checked_targets.loc[ts].to_dict(),
            can_buy=can_trade,
            can_sell=can_trade,
        )

    history = account.equity_history()
    equity = pd.Series(history["equity"].astype(float).to_numpy(), index=prices.index, name="equity")
    returns = equity.pct_change().fillna(0.0)
    weights_effective = _weights_from_history(history, prices)
    trades = account.broker.ledger()
    costs = _costs_from_ledger(trades, prices.index)
    return BacktestResult(
        returns=returns,
        weights_effective=weights_effective,
        target_weights=checked_targets,
        trades_long=trades,
        costs=costs,
        equity_curve=equity,
        initial_equity=float(initial_equity),
    )


def _weights_from_history(history: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in history.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        equity = float(row["equity"])
        positions = row.get("positions") or {}
        values = {
            sym: (float(positions.get(sym, 0.0)) * float(prices.loc[ts, sym]) / equity) if equity else 0.0
            for sym in prices.columns
        }
        rows.append(values)
    return pd.DataFrame(rows, index=prices.index, columns=prices.columns).fillna(0.0)


def _costs_from_ledger(ledger: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    if ledger.empty or "fee" not in ledger.columns:
        return pd.Series(0.0, index=index)
    work = ledger.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    return work.groupby("timestamp")["fee"].sum().reindex(index, fill_value=0.0).astype(float)
