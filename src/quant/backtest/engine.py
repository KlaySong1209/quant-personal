"""Vectorized daily backtest engine with one-period execution shift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.costs import CostModel
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
    target_weights: pd.DataFrame,
    cost_model: CostModel,
    risk: RiskConfig,
    initial_equity: float,
    tradable: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
) -> BacktestResult:
    _validate_inputs(prices, target_weights)
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if tradable is not None:
        tradable = _validate_bool_panel(tradable, prices, name="tradable")
    if universe is not None:
        universe = _validate_bool_panel(universe, prices, name="universe")

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

