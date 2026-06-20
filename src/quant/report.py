"""Status report module. Pure view over account and backtest state.

Produces HUMAN-READABLE summaries of paper account status and backtest results.
Every assumption surfaced comes from recorded metadata/state, never inferred.
This module computes NO new trading logic — it is a view only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECORDED_MISSING = "recorded-missing"


def _recorded(assumptions: dict[str, Any], key: str) -> Any:
    """Return a recorded assumption value, never a guessed default."""
    if key not in assumptions:
        return RECORDED_MISSING
    return assumptions[key]


def account_report(state_path: str | Path) -> dict[str, Any]:
    """Produce a structured status report dict from a paper account state file.

    Returns a dict with keys:
      - report_type: "account_status"
      - generated_at: ISO timestamp
      - account: {account_id, mode, label, advanced_to, steps}
      - equity: {cash, position_value, total_equity, ledger_balanced}
      - pending_orders: list of pending order summaries
      - assumptions: dict of recorded assumptions
      - positions: list of {symbol, shares, price, value}
      - flags: list of warning/abnormal-condition strings
      - error: None or error string
    """
    path = Path(state_path)
    if not path.exists():
        return {
            "report_type": "account_status",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "account": None,
            "equity": None,
            "pending_orders": [],
            "assumptions": None,
            "positions": [],
            "flags": ["no account state file found"],
            "error": None,
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "report_type": "account_status",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "account": None,
            "equity": None,
            "pending_orders": [],
            "assumptions": None,
            "positions": [],
            "flags": [],
            "error": f"failed to read account state: {exc}",
        }

    assumptions = data.get("assumptions", {}) or {}
    history = data.get("history", []) or []
    last = history[-1] if history else {}
    positions_raw = last.get("positions", {}) or {}
    last_prices = data.get("broker", {}).get("last_prices", {}) or {}

    # Build position summaries
    positions = []
    for sym, shares in sorted(positions_raw.items()):
        if abs(float(shares)) < 1e-12:
            continue
        price = float(last_prices.get(sym, 0.0))
        value = float(shares) * price
        positions.append({
            "symbol": str(sym),
            "shares": float(shares),
            "price": price,
            "value": value,
        })

    # Build pending order summaries
    pending_orders = []
    for po in data.get("pending_orders", []) or []:
        status = po.get("status", "pending")
        reason = ""
        if status == "pending":
            reason = "waiting for next trading day open"
        elif status == "skipped":
            reason = po.get("reason", "missing open prices; skipped by policy")
        elif status == "failed":
            reason = po.get("reason", "execution failed")
        degraded = bool(po.get("degraded", False))
        if degraded:
            reason += " (degraded: used fallback prices)"
        pending_orders.append({
            "order_id": str(po.get("order_id", "")),
            "created_on": str(po.get("created_on", "")),
            "status": str(status),
            "reason": reason,
            "degraded": degraded,
        })

    # Build flags
    flags = []
    if pending_orders:
        pending_count = sum(1 for o in pending_orders if o["status"] == "pending")
        if pending_count:
            flags.append(f"{pending_count} pending order(s) waiting for next-day open fill")
        skipped = [o for o in pending_orders if o["status"] == "skipped"]
        if skipped:
            flags.append(f"{len(skipped)} order(s) skipped due to missing open prices")
        failed = [o for o in pending_orders if o["status"] == "failed"]
        if failed:
            flags.append(f"{len(failed)} order(s) failed to execute")

    ledger_balanced = None
    # Check broker ledger balance from cash delta
    broker = data.get("broker", {})
    if broker:
        events = broker.get("events", []) or []
        starting = float(broker.get("starting_cash", 0))
        cash_now = float(broker.get("cash", 0))
        cash_delta_sum = sum(float(e.get("cash_delta", 0)) for e in events)
        ledger_balanced = abs(starting + cash_delta_sum - cash_now) < 1e-6

    return {
        "report_type": "account_status",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account": {
            "account_id": str(data.get("account_id", "")),
            "mode": str(data.get("mode", "paper_simulation")),
            "label": str(data.get("label", "")),
            "advanced_to": str(last.get("timestamp", "")) if last else None,
            "steps": len(history),
        },
        "equity": {
            "cash": float(last.get("cash", 0)) if last else None,
            "position_value": float(last.get("position_value", 0)) if last else None,
            "total_equity": float(last.get("equity", 0)) if last else None,
            "ledger_balanced": ledger_balanced,
        },
        "pending_orders": pending_orders,
        "assumptions": {
            "fill_price_rule": str(_recorded(assumptions, "fill_price_rule")),
            "missing_open_policy": str(_recorded(assumptions, "missing_open_policy")),
            "commission_bps": _recorded(assumptions, "commission_bps"),
            "stamp_duty_bps": _recorded(assumptions, "stamp_duty_bps"),
            "slippage_bps": _recorded(assumptions, "slippage_bps"),
            "order_routing": str(_recorded(assumptions, "order_routing")),
        },
        "positions": positions,
        "flags": flags,
        "error": None,
    }


def format_account_report(report: dict[str, Any]) -> str:
    """Format an account report dict as printable plain text."""
    lines = []
    lines.append("=" * 60)
    lines.append("  quant-personal  Account Status")
    lines.append("=" * 60)

    if report.get("error"):
        lines.append(f"\n  ERROR: {report['error']}")
        return "\n".join(lines)

    account = report.get("account") or {}
    equity = report.get("equity") or {}
    assumptions = report.get("assumptions") or {}
    positions = report.get("positions") or []
    pending = report.get("pending_orders") or []
    flags = report.get("flags") or []

    # Account
    lines.append("")
    lines.append("─" * 40)
    lines.append("  ACCOUNT")
    lines.append(f"    ID:       {account.get('account_id', 'N/A')}")
    lines.append(f"    Mode:     {account.get('mode', 'N/A')}")
    lines.append(f"    Label:    {account.get('label', 'N/A')}")
    lines.append(f"    Advanced: {account.get('advanced_to', 'never')}")
    lines.append(f"    Steps:    {account.get('steps', 0)}")

    # Equity
    lines.append("")
    lines.append("─" * 40)
    lines.append("  EQUITY")
    cash_str = f"¥{equity['cash']:,.2f}" if equity.get("cash") is not None else "N/A"
    pv_str = f"¥{equity['position_value']:,.2f}" if equity.get("position_value") is not None else "N/A"
    eq_str = f"¥{equity['total_equity']:,.2f}" if equity.get("total_equity") is not None else "N/A"
    lines.append(f"    Cash:           {cash_str}")
    lines.append(f"    Position Value: {pv_str}")
    lines.append(f"    Total Equity:   {eq_str}")
    ledger = equity.get("ledger_balanced")
    if ledger is True:
        lines.append("    Ledger:         BALANCED")
    elif ledger is False:
        lines.append("    Ledger:         CHECK — imbalance detected")
    else:
        lines.append("    Ledger:         unknown")

    # Pending orders
    lines.append("")
    lines.append("─" * 40)
    lines.append("  PENDING ORDERS")
    if not pending:
        lines.append("    (none)")
    else:
        for po in pending:
            lines.append(f"    [{po['status'].upper()}] {po['order_id']}")
            lines.append(f"      Created: {po['created_on']}")
            if po["reason"]:
                lines.append(f"      Reason:  {po['reason']}")

    # Assumptions
    lines.append("")
    lines.append("─" * 40)
    lines.append("  ASSUMPTIONS")
    lines.append(f"    fill_price_rule:    {assumptions.get('fill_price_rule', 'N/A')}")
    lines.append(f"    missing_open_policy:{assumptions.get('missing_open_policy', 'N/A')}")
    lines.append(f"    commission_bps:     {assumptions.get('commission_bps', 'N/A')}")
    lines.append(f"    stamp_duty_bps:     {assumptions.get('stamp_duty_bps', 'N/A')}")
    lines.append(f"    slippage_bps:       {assumptions.get('slippage_bps', 'N/A')}")
    lines.append(f"    order_routing:      {assumptions.get('order_routing', 'N/A')}")

    # Positions
    lines.append("")
    lines.append("─" * 40)
    lines.append("  POSITIONS")
    if not positions:
        lines.append("    (no open positions)")
    else:
        for pos in positions:
            lines.append(
                f"    {pos['symbol']}: {pos['shares']:,.0f} shares"
                f" @ ¥{pos['price']:,.2f} = ¥{pos['value']:,.2f}"
            )

    # Flags
    if flags:
        lines.append("")
        lines.append("─" * 40)
        lines.append("  FLAGS")
        for flag in flags:
            lines.append(f"    ! {flag}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def backtest_report(run_dir: str | Path) -> dict[str, Any]:
    """Produce a structured report dict from a backtest run directory."""
    from quant.app import load_run_summary, metric_explanations

    path = Path(run_dir)
    if not path.exists():
        return {
            "report_type": "backtest",
            "error": f"run directory not found: {run_dir}",
            "metrics": None,
            "metric_lines": [],
        }

    try:
        summary = load_run_summary(path)
    except Exception as exc:
        return {
            "report_type": "backtest",
            "error": f"failed to load run: {exc}",
            "metrics": None,
            "metric_lines": [],
        }

    metrics = summary.get("metrics", {})
    explanations = metric_explanations()
    metric_lines = []
    for key, value in sorted(metrics.items()):
        desc = explanations.get(key, key)
        metric_lines.append({
            "key": key,
            "value": float(value),
            "description": desc,
        })

    return {
        "report_type": "backtest",
        "error": None,
        "run_dir": str(path),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "metric_lines": metric_lines,
    }


def format_backtest_report(report: dict[str, Any]) -> str:
    """Format a backtest report dict as printable plain text."""
    lines = []
    lines.append("")
    lines.append("─" * 40)
    lines.append("  BACKTEST")

    if report.get("error"):
        lines.append(f"    ERROR: {report['error']}")
        return "\n".join(lines)

    lines.append(f"    Run: {report.get('run_dir', 'N/A')}")
    lines.append("")
    for ml in report.get("metric_lines", []):
        lines.append(f"    {ml['key']:>20s}: {ml['value']:+.6f}")
        lines.append(f"    {'':>20s}  {ml['description']}")
    return "\n".join(lines)


def combined_report(
    state_path: str | Path,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Produce a combined account + optional backtest report dict."""
    acct = account_report(state_path)
    result: dict[str, Any] = {
        "report_type": "combined",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "account": acct,
    }
    if run_dir:
        result["backtest"] = backtest_report(run_dir)
    else:
        result["backtest"] = None
    return result


def format_combined_report(report: dict[str, Any]) -> str:
    """Format a combined report dict as printable plain text."""
    parts = [format_account_report(report["account"])]
    if report.get("backtest") and not report["backtest"].get("error"):
        parts.append(format_backtest_report(report["backtest"]))
    return "\n".join(parts)
