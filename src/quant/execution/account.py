"""Persistent simulated paper account. No real broker or order routing.

Supports:
  - same_day_close fill (default, backward-compatible).
  - next_day_open pending-order state machine.
  - Account-layer corporate action application before each step.
  - Demo / no-state status for dashboard honesty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant.backtest.costs import AShareCostModel
from quant.data.adjust.corporate_actions import (
    CorporateAction,
    validate_corporate_actions,
    corporate_actions_for_date,
)
from quant.execution.corporate_actions import (
    apply_corporate_action,
    AppliedCorporateAction,
)
from quant.execution.paper import PaperBroker

PAPER_SIMULATION_LABEL = "SIMULATED / PAPER -- NOT REAL"
DEMO_LABEL = "DEMO / SYNTHETIC DATA -- NOT YOUR DATA"

AccountMode = Literal["paper_simulation", "demo"]
MissingOpenPolicy = Literal["skip", "fallback_to_prev_close", "fail"]
PendingStatus = Literal["pending", "filled", "skipped", "failed"]


@dataclass
class PendingOrder:
    """A pending next_day_open order created at T close, to be filled at T+1 open."""
    order_id: str
    created_on: pd.Timestamp
    target_weights: dict[str, float]
    decision_prices: dict[str, float]
    fill_rule: str  # "next_day_open"
    status: PendingStatus = "pending"
    filled_on: pd.Timestamp | None = None
    fill_prices: dict[str, float] | None = None
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "created_on": self.created_on.isoformat(),
            "target_weights": self.target_weights,
            "decision_prices": self.decision_prices,
            "fill_rule": self.fill_rule,
            "status": self.status,
            "filled_on": self.filled_on.isoformat() if self.filled_on else None,
            "fill_prices": self.fill_prices,
            "degraded": self.degraded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingOrder":
        return cls(
            order_id=str(data["order_id"]),
            created_on=pd.Timestamp(data["created_on"]),
            target_weights={str(k): float(v) for k, v in data["target_weights"].items()},
            decision_prices={str(k): float(v) for k, v in data["decision_prices"].items()},
            fill_rule=str(data.get("fill_rule", "next_day_open")),
            status=data.get("status", "pending"),
            filled_on=pd.Timestamp(data["filled_on"]) if data.get("filled_on") else None,
            fill_prices={str(k): float(v) for k, v in (data.get("fill_prices") or {}).items()},
            degraded=bool(data.get("degraded", False)),
        )


@dataclass
class SimAccount:
    account_id: str
    starting_cash: float
    allow_short: bool = False
    allow_margin: bool = False
    max_gross_leverage: float = 1.0
    commission_bps: float = 1.0
    stamp_duty_bps: float = 5.0
    slippage_bps: float = 1.0
    fill_price_rule: str = "same_day_close"
    missing_open_policy: MissingOpenPolicy = "skip"
    allow_zero_cost_for_tests: bool = False
    mode: AccountMode = "paper_simulation"
    broker: PaperBroker = field(init=False)
    _history: list[dict[str, Any]] = field(default_factory=list)
    _completed_steps: set[str] = field(default_factory=set)
    _applied_corporate_actions: set[str] = field(default_factory=set)
    _corporate_actions: list[CorporateAction] = field(default_factory=list)
    _applied_ca_records: list[AppliedCorporateAction] = field(default_factory=list)
    _pending_orders: list[PendingOrder] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fill_price_rule not in {"same_day_close", "next_day_open"}:
            raise ValueError("fill_price_rule must be 'same_day_close' or 'next_day_open'")
        if self.missing_open_policy not in {"skip", "fallback_to_prev_close", "fail"}:
            raise ValueError("missing_open_policy must be 'skip', 'fallback_to_prev_close', or 'fail'")
        if self.mode not in {"paper_simulation", "demo"}:
            raise ValueError("mode must be 'paper_simulation' or 'demo'")
        self.cost_model = AShareCostModel(
            commission_bps=self.commission_bps,
            stamp_duty_bps=self.stamp_duty_bps,
            slippage_bps=self.slippage_bps,
            allow_zero_cost_for_tests=self.allow_zero_cost_for_tests,
        )
        self.broker = PaperBroker(
            self.starting_cash,
            allow_short=self.allow_short,
            allow_margin=self.allow_margin,
            max_gross_leverage=self.max_gross_leverage,
        )

    # ------------------------------------------------------------------
    # Corporate actions
    # ------------------------------------------------------------------

    def set_corporate_actions(self, actions: list[CorporateAction]) -> None:
        """Register corporate actions for account-layer application.

        Validation is fail-fast.  Call before the first step() or after
        loading state that carries actions.
        """
        validate_corporate_actions(actions)
        self._corporate_actions = list(actions)

    @property
    def applied_corporate_actions(self) -> set[str]:
        return set(self._applied_corporate_actions)

    def _apply_corporate_actions_before_trading(self, timestamp: pd.Timestamp) -> None:
        """Apply ex-date corporate actions before trading, ensuring idempotency."""
        if not self._corporate_actions:
            return
        ts_normalized = timestamp.normalize()
        for ca in self._corporate_actions:
            if ca.event_key in self._applied_corporate_actions:
                continue
            if ca.timestamp.normalize() != ts_normalized:
                continue
            new_positions, new_cash, record = apply_corporate_action(
                ca, self.broker.positions(), self.broker.cash,
            )
            # Apply share adjustments
            for sym, qty in new_positions.items():
                old_qty = self.broker.positions().get(sym, 0.0)
                delta = qty - old_qty
                if delta != 0:
                    self.broker.apply_share_adjustment(
                        timestamp, sym, delta,
                        label=f"CORP_ACT:{ca.action_type}",
                    )
            # Apply cash adjustment
            cash_delta = new_cash - self.broker.cash
            if cash_delta != 0:
                self.broker.apply_cash_adjustment(
                    timestamp, cash_delta,
                    label=f"CORP_ACT:{ca.action_type}",
                )
            self._applied_corporate_actions.add(ca.event_key)
            self._applied_ca_records.append(record)

    # ------------------------------------------------------------------
    # Pending orders (next_day_open)
    # ------------------------------------------------------------------

    def _fill_pending_orders(self, timestamp: pd.Timestamp, prices: dict[str, float]) -> list[dict[str, Any]]:
        """Attempt to fill any pending next_day_open orders using today's open prices.

        The open prices must be passed as *prices*; for next_day_open, the caller
        provides T+1 open as the prices dict, and this method fills orders created
        at T close.

        Returns a list of fill-result dicts (one per filled/skipped/failed order).
        """
        results: list[dict[str, Any]] = []
        still_pending: list[PendingOrder] = []

        for order in self._pending_orders:
            if order.status != "pending":
                still_pending.append(order)
                continue

            missing_open = [sym for sym in order.target_weights if sym not in prices or not np.isfinite(prices[sym]) or prices[sym] <= 0]
            if missing_open:
                if self.missing_open_policy == "fail":
                    raise ValueError(f"missing open prices for symbols {missing_open} on {timestamp}; policy=fail")
                elif self.missing_open_policy == "skip":
                    order.status = "skipped"
                    still_pending.append(order)
                    results.append({"order_id": order.order_id, "status": "skipped", "reason": f"missing open: {missing_open}"})
                    continue
                elif self.missing_open_policy == "fallback_to_prev_close":
                    order.degraded = True
                    for sym in missing_open:
                        if sym in order.decision_prices:
                            prices = dict(prices)
                            prices[sym] = order.decision_prices[sym]

            # Fill the order at open prices
            order.fill_prices = {sym: prices[sym] for sym in order.target_weights}
            order.filled_on = timestamp
            order.status = "filled"

            # Execute the fill through the broker
            try:
                self._execute_fill(timestamp, order.target_weights, order.fill_prices)
            except Exception as exc:
                order.status = "failed"
                still_pending.append(order)
                results.append({"order_id": order.order_id, "status": "failed", "reason": str(exc)})
                continue

            still_pending.append(order)
            results.append({
                "order_id": order.order_id,
                "status": "filled",
                "fill_prices": order.fill_prices,
                "degraded": order.degraded,
            })

        self._pending_orders = still_pending
        return results

    def _create_pending_order(self, timestamp: pd.Timestamp, target_weights: dict[str, float], prices: dict[str, float]) -> PendingOrder:
        """Create a pending order for next_day_open fill."""
        order_id = f"pending-{timestamp.isoformat()}-{len(self._pending_orders)}"
        order = PendingOrder(
            order_id=order_id,
            created_on=timestamp,
            target_weights=dict(target_weights),
            decision_prices=dict(prices),
            fill_rule="next_day_open",
            status="pending",
        )
        self._pending_orders.append(order)
        return order

    def _execute_fill(self, timestamp: pd.Timestamp, target_weights: dict[str, float], fill_prices: dict[str, float]) -> None:
        """Execute a fill for given target weights at given fill prices.

        This is the core trading logic, shared between same_day_close and
        next_day_open fills.
        """
        self.broker.update_prices(fill_prices)
        equity_before = self.broker.mark_to_market(timestamp)
        cost_buffer_rate = max(self.cost_model.buy_rate, self.cost_model.sell_rate)
        investable_equity = equity_before * (1.0 - cost_buffer_rate)
        target_shares = {
            sym: (target_weights.get(sym, 0.0) * investable_equity) / fill_prices[sym]
            for sym in fill_prices
        }
        current = self.broker.positions()
        traded_notional = pd.Series(
            {
                sym: (target_shares[sym] - current.get(sym, 0.0)) * fill_prices[sym]
                for sym in fill_prices
            },
            dtype="float64",
        )
        cost_breakdown = self.cost_model.per_symbol_breakdown(traded_notional)
        self.broker.submit_target(timestamp, target_shares)
        self.broker.apply_cash_fee(timestamp, cost_breakdown["total"])

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        timestamp: pd.Timestamp | str,
        *,
        prices: dict[str, float],
        target_weights: dict[str, float],
        save_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Advance the account by one day-step.

        For *same_day_close*: trades execute immediately at *prices*.
        For *next_day_open*:
          1. Fill any pending orders from the previous step using today's *prices* as open.
          2. Create a new pending order from *target_weights* (to be filled next step).
        """
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        step_key = ts.isoformat()
        if step_key in self._completed_steps:
            if save_path is not None:
                self.save(save_path)
            return self._history_by_timestamp(step_key)

        clean_prices = {str(sym): float(price) for sym, price in prices.items()}
        clean_weights = {str(sym): float(weight) for sym, weight in target_weights.items()}
        if any((not np.isfinite(price)) or price <= 0 for price in clean_prices.values()):
            raise ValueError("prices must be finite and strictly positive")
        if any(not np.isfinite(weight) for weight in clean_weights.values()):
            raise ValueError("target weights must be finite")
        gross = sum(abs(weight) for weight in clean_weights.values())
        if gross > self.max_gross_leverage + 1e-9:
            raise ValueError("target weights exceed account gross leverage")

        # 1. Apply corporate actions before trading
        self._apply_corporate_actions_before_trading(ts)

        fill_results: list[dict[str, Any]] = []
        if self.fill_price_rule == "next_day_open":
            # 2a. Fill pending orders from previous step using today's open
            fill_results = self._fill_pending_orders(ts, clean_prices)
            # 2b. Create new pending order from today's close (will fill at T+1 open)
            self._create_pending_order(ts, clean_weights, clean_prices)
            # For next_day_open, we don't execute the target_weights now.
            # Use current broker state for equity history.
            equity = self.broker.mark_to_market(ts)
            position_value = equity - self.broker.cash
            row = {
                "timestamp": step_key,
                "label": self._label(),
                "cash": float(self.broker.cash),
                "position_value": float(position_value),
                "equity": float(equity),
                "positions": self.broker.positions(),
                "prices": clean_prices,
                "target_weights": clean_weights,
                "pending_orders": [o.to_dict() for o in self._pending_orders],
                "fill_results": fill_results,
                "costs": {"total": 0.0},
                "assumptions": self.assumptions(),
            }
        else:
            # same_day_close: execute immediately
            self.broker.update_prices(clean_prices)
            equity_before = self.broker.mark_to_market(ts)
            cost_buffer_rate = max(self.cost_model.buy_rate, self.cost_model.sell_rate)
            investable_equity = equity_before * (1.0 - cost_buffer_rate)
            target_shares = {
                sym: (clean_weights.get(sym, 0.0) * investable_equity) / price
                for sym, price in clean_prices.items()
            }
            current = self.broker.positions()
            traded_notional = pd.Series(
                {
                    sym: (target_shares[sym] - current.get(sym, 0.0)) * clean_prices[sym]
                    for sym in clean_prices
                },
                dtype="float64",
            )
            cost_breakdown = self.cost_model.per_symbol_breakdown(traded_notional)
            self.broker.submit_target(ts, target_shares)
            self.broker.apply_cash_fee(ts, cost_breakdown["total"])
            position_value = sum(self.broker.positions().get(sym, 0.0) * clean_prices[sym] for sym in clean_prices)
            equity = self.broker.cash + position_value
            row = {
                "timestamp": step_key,
                "label": self._label(),
                "cash": float(self.broker.cash),
                "position_value": float(position_value),
                "equity": float(equity),
                "positions": self.broker.positions(),
                "prices": clean_prices,
                "costs": cost_breakdown,
                "assumptions": self.assumptions(),
            }

        self._history.append(row)
        self._completed_steps.add(step_key)
        if save_path is not None:
            self.save(save_path)
        return row

    def _label(self) -> str:
        if self.mode == "demo":
            return DEMO_LABEL
        return PAPER_SIMULATION_LABEL

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def equity_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        _assert_no_non_finite(data, path="account")
        target.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SimAccount":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _assert_no_non_finite(data, path="account")
        account = cls(
            account_id=str(data["account_id"]),
            starting_cash=float(data["starting_cash"]),
            allow_short=bool(data.get("allow_short", False)),
            allow_margin=bool(data.get("allow_margin", False)),
            max_gross_leverage=float(data.get("max_gross_leverage", 1.0)),
            commission_bps=float(data.get("assumptions", {}).get("commission_bps", 1.0)),
            stamp_duty_bps=float(data.get("assumptions", {}).get("stamp_duty_bps", 5.0)),
            slippage_bps=float(data.get("assumptions", {}).get("slippage_bps", 1.0)),
            fill_price_rule=str(data.get("assumptions", {}).get("fill_price_rule", "same_day_close")),
            missing_open_policy=str(data.get("assumptions", {}).get("missing_open_policy", "skip")),
            allow_zero_cost_for_tests=bool(data.get("assumptions", {}).get("allow_zero_cost_for_tests", False)),
            mode=str(data.get("mode", "paper_simulation")),
        )
        account.broker = PaperBroker.from_dict(data["broker"])
        account._history = list(data.get("history", []))
        account._completed_steps = set(data.get("completed_steps", []))
        account._applied_corporate_actions = set(data.get("applied_corporate_actions", []))
        account._applied_ca_records = [
            AppliedCorporateAction(
                event_key=r["event_key"],
                timestamp=pd.Timestamp(r["timestamp"]),
                symbol=r["symbol"],
                action_type=r["action_type"],
                shares_delta=r["shares_delta"],
                cash_delta=r["cash_delta"],
                note=r["note"],
            )
            for r in data.get("applied_ca_records", [])
        ]
        account._pending_orders = [
            PendingOrder.from_dict(o) for o in data.get("pending_orders", [])
        ]
        # Restore corporate actions from saved state
        if "corporate_actions" in data:
            account._corporate_actions = [
                CorporateAction.from_dict(ca) for ca in data["corporate_actions"]
            ]
        return account

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self._label(),
            "account_id": self.account_id,
            "starting_cash": self.starting_cash,
            "allow_short": self.allow_short,
            "allow_margin": self.allow_margin,
            "max_gross_leverage": self.max_gross_leverage,
            "assumptions": self.assumptions(),
            "broker": self.broker.to_dict(),
            "history": list(self._history),
            "completed_steps": sorted(self._completed_steps),
            "applied_corporate_actions": sorted(self._applied_corporate_actions),
            "applied_ca_records": [r.to_dict() for r in self._applied_ca_records],
            "corporate_actions": [ca.to_dict() for ca in self._corporate_actions],
            "pending_orders": [o.to_dict() for o in self._pending_orders],
            "paper_only": True,
        }

    def assumptions(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self._label(),
            "commission_bps": self.commission_bps,
            "stamp_duty_bps": self.stamp_duty_bps,
            "slippage_bps": self.slippage_bps,
            "fill_price_rule": self.fill_price_rule,
            "missing_open_policy": self.missing_open_policy,
            "allow_zero_cost_for_tests": self.allow_zero_cost_for_tests,
            "order_routing": "none; all fills are local ledger simulations",
        }

    def _history_by_timestamp(self, step_key: str) -> dict[str, Any]:
        for row in self._history:
            if row["timestamp"] == step_key:
                _assert_no_non_finite(row, path=f"history[{step_key}]")
                return row
        raise RuntimeError(f"completed step missing history row: {step_key}")


def _assert_no_non_finite(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"non-finite value in saved account state at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_non_finite(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _assert_no_non_finite(item, path=f"{path}[{i}]")
        return
