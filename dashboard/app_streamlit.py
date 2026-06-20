"""quant-personal dashboard.

A Streamlit view layer that only calls quant.app — no data processing,
strategy logic, or account calculations live here.
"""

from __future__ import annotations

import html
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from quant import app
from quant.execution.account import PAPER_SIMULATION_LABEL

STREAMLIT_INSTALL_COMMAND = app.STREAMLIT_INSTALL_COMMAND
PAPER_MODE_LABEL = PAPER_SIMULATION_LABEL

# Ensure the project root is on sys.path so that quant imports work.
_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from quant.app import paper_account_status, format_metrics_plain, load_run_summary, list_runs  # noqa: E402
from quant.report import account_report, backtest_report, combined_report  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers (used by tests and pages)
# ---------------------------------------------------------------------------

def _escape(value: Any) -> str:
    return html.escape(str(value))


def _format_money(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):,.2f}"


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):,.{digits}f}"


def _format_percent(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:+.2f}%"


def _terminal_css() -> str:
    return """
<style>
:root {
  --qp-ink: #20242a;
  --qp-muted: #6d7580;
  --qp-line: #d9dee5;
  --qp-soft: #f6f7f9;
  --qp-blue: #dcecf8;
  --qp-blue-strong: #477da8;
  --qp-red: #f2cfc4;
  --qp-red-strong: #923f2e;
  --qp-green: #dfeee4;
  --qp-green-strong: #3e7653;
}
.quant-terminal,
.terminal-panel,
.terminal-strip,
.terminal-statusbar {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}
.quant-terminal {
  border: 1px solid var(--qp-line);
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 8px 28px rgba(19, 31, 44, 0.08);
  overflow: hidden;
}
.terminal-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 18px;
  border-bottom: 1px solid var(--qp-line);
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}
.terminal-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}
.terminal-logo {
  width: 36px;
  height: 36px;
  border: 1px solid #bfc8d2;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-size: 15px;
  color: #2a313a;
  background: #fbfcfd;
}
.terminal-kicker {
  color: var(--qp-muted);
  font-size: 10px;
  letter-spacing: 0;
  text-transform: uppercase;
  line-height: 1.3;
}
.terminal-title {
  color: var(--qp-ink);
  font-size: 18px;
  line-height: 1.15;
  font-weight: 800;
  letter-spacing: 0;
}
.terminal-pills {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.terminal-pill {
  border: 1px solid var(--qp-line);
  border-radius: 4px;
  padding: 7px 10px;
  background: #fff;
  color: #313943;
  font-size: 11px;
  font-weight: 700;
  white-space: nowrap;
}
.terminal-pill.blue {
  background: var(--qp-blue);
  border-color: #c6dced;
  color: #315e7d;
}
.terminal-pill.red {
  background: var(--qp-red);
  border-color: #e3b3a5;
  color: var(--qp-red-strong);
}
.terminal-panel {
  border: 1px solid var(--qp-line);
  border-radius: 8px;
  background: #ffffff;
  padding: 14px 16px;
  min-height: 100%;
}
.terminal-panel.tight {
  padding: 10px 12px;
}
.terminal-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding-bottom: 9px;
  margin-bottom: 10px;
  border-bottom: 1px solid #e8ebef;
}
.terminal-panel-title {
  color: #232a32;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0;
  text-transform: uppercase;
}
.terminal-panel-note {
  color: var(--qp-muted);
  font-size: 10px;
  text-transform: uppercase;
}
.metric-tile {
  border: 1px solid #e0e5eb;
  border-radius: 6px;
  background: #fbfcfd;
  padding: 12px 13px;
  min-height: 96px;
}
.metric-tile.blue {
  background: linear-gradient(180deg, #fbfdff 0%, #eef6fd 100%);
}
.metric-tile.red {
  background: linear-gradient(180deg, #fffdfc 0%, #f9ebe6 100%);
}
.metric-label {
  color: var(--qp-muted);
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.metric-value {
  color: var(--qp-ink);
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(24px, 2.6vw, 36px);
  line-height: 1;
  letter-spacing: 0;
  white-space: nowrap;
}
.metric-detail {
  color: var(--qp-muted);
  font-size: 11px;
  line-height: 1.35;
  margin-top: 8px;
}
.terminal-mini-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  border: 1px solid #e2e6eb;
  border-radius: 6px;
  overflow: hidden;
}
.terminal-mini-grid > div {
  padding: 9px 10px;
  border-right: 1px solid #e2e6eb;
  background: #fbfcfd;
}
.terminal-mini-grid > div:last-child {
  border-right: 0;
}
.mini-label {
  color: var(--qp-muted);
  font-size: 9px;
  text-transform: uppercase;
  font-weight: 800;
}
.mini-value {
  color: var(--qp-blue-strong);
  font-size: 13px;
  font-weight: 900;
  margin-top: 4px;
}
.terminal-list {
  display: grid;
  gap: 7px;
  margin: 0;
}
.terminal-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid #edf0f3;
  font-size: 12px;
}
.terminal-row:last-child {
  border-bottom: 0;
}
.terminal-row span:first-child {
  color: var(--qp-muted);
}
.terminal-row span:last-child {
  color: var(--qp-ink);
  font-weight: 800;
  text-align: right;
}
.terminal-statusbar {
  border: 1px solid var(--qp-line);
  border-radius: 6px;
  background: #ffffff;
  color: #59616b;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 11px;
  font-size: 10px;
  text-transform: uppercase;
  overflow-x: auto;
}
.terminal-alert {
  border: 1px solid #e6c5bc;
  background: #fff8f5;
  color: var(--qp-red-strong);
  border-radius: 6px;
  padding: 10px 12px;
  font-size: 12px;
  font-weight: 800;
}
div[data-testid="stMetric"] {
  border: 1px solid var(--qp-line);
  border-radius: 6px;
  padding: 10px 12px;
  background: #ffffff;
}
.stButton > button {
  border-radius: 5px;
  border: 1px solid #bdc7d2;
  background: #ffffff;
  color: #222a33;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-weight: 800;
}
.stButton > button:hover {
  border-color: var(--qp-blue-strong);
  color: var(--qp-blue-strong);
}
@media (max-width: 760px) {
  .terminal-topbar {
    align-items: flex-start;
    flex-direction: column;
  }
  .terminal-pills {
    justify-content: flex-start;
  }
  .terminal-mini-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .terminal-mini-grid > div:nth-child(2) {
    border-right: 0;
  }
  .metric-value {
    font-size: 28px;
  }
}
</style>
"""


def _paper_snapshot(summary: dict[str, Any]) -> dict[str, str]:
    positions = summary.get("positions", {}) or {}
    open_positions = sum(1 for shares in positions.values() if abs(float(shares)) > 1e-12)
    universe_size = len(positions)
    assumptions = summary.get("assumptions", {}) or {}
    ledger_balanced = summary.get("ledger_balanced")
    ledger = "NO STATE" if ledger_balanced is None else ("BALANCED" if ledger_balanced else "CHECK")
    return {
        "label": str(summary.get("label") or PAPER_MODE_LABEL),
        "cash": _format_money(summary.get("final_cash")),
        "equity": _format_money(summary.get("final_equity")),
        "steps": str(summary.get("steps", 0)),
        "ledger": ledger,
        "positions": f"{open_positions} / {universe_size}",
        "assumptions": ", ".join(f"{k}={v}" for k, v in sorted(assumptions.items())) or "N/A",
    }


st.set_page_config(
    page_title="quant-personal",
    page_icon=":chart:",
    layout="wide",
)


def _render_demo_banner() -> None:
    """Persistent demo banner shown at top and on account card."""
    st.warning(":warning: **DEMO / SYNTHETIC DATA — NOT YOUR DATA**")
    st.caption("This account was generated from synthetic example data and does not represent real trading.")


def _render_paper_simulation_banner() -> None:
    st.info(":test_tube: **SIMULATED / PAPER — NOT REAL**")
    st.caption("All fills are virtual ledger entries. No orders are routed to any venue.")


def _render_onboarding() -> None:
    st.title("quant-personal")
    st.markdown("""
    ### Welcome to quant-personal

    This is a local daily-frequency research tool. To get started:

    1. **Generate example data** — `python -m quant --generate-example-data`
    2. **Ingest your local data** — `python -m quant --ingest-local-data ...`
    3. **Run the daily loop** — `python -m quant --daily ...`
    4. **Or run a paper session** — `python scripts/run_paper_session.py ...`

    See [Getting Started](docs/GETTING_STARTED.md) for full instructions.

    Once you have a paper account, reload this dashboard to see your portfolio.
    """)
    st.info("No account state found. Start a paper session to populate the dashboard.")


def _status_report_view_data(report: dict[str, Any]) -> dict[str, Any]:
    """Shape the status report into dashboard view data without computing state."""
    account = report.get("account") or {}
    equity = report.get("equity") or {}
    return {
        "error": report.get("error"),
        "account_id": account.get("account_id", "Unknown"),
        "advanced_to": account.get("advanced_to"),
        "steps": account.get("steps", 0),
        "flags": report.get("flags", []) or [],
        "metrics": {
            "total_equity": equity.get("total_equity"),
            "cash": equity.get("cash"),
            "position_value": equity.get("position_value"),
            "ledger_balanced": equity.get("ledger_balanced"),
        },
        "pending_orders": report.get("pending_orders", []) or [],
        "assumptions": report.get("assumptions") or {},
        "positions": report.get("positions", []) or [],
        "has_account": bool(account),
    }


def _render_status_report(report: dict[str, Any]) -> None:
    """Render the Task-2 status report as the primary dashboard view."""
    view_data = _status_report_view_data(report)
    if view_data.get("error"):
        st.error(f"Report error: {view_data['error']}")
        return

    # --- Account identity ---
    if not view_data["has_account"]:
        st.info("No account data available.")
        return

    st.subheader(f"Account: {view_data['account_id']}")
    if view_data.get("advanced_to"):
        st.caption(f"Advanced to: {view_data['advanced_to']} ({view_data.get('steps', 0)} steps)")

    # --- Flags / warnings ---
    flags = view_data["flags"]
    for flag in flags:
        if "skipped" in flag.lower() or "failed" in flag.lower():
            st.warning(f":warning: {flag}")
        else:
            st.info(flag)

    # --- Equity decomposition ---
    metrics = view_data["metrics"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total = metrics.get("total_equity")
        st.metric("Total Equity", f"¥{total:,.2f}" if total is not None else "N/A")
    with col2:
        cash = metrics.get("cash")
        st.metric("Cash", f"¥{cash:,.2f}" if cash is not None else "N/A")
    with col3:
        pv = metrics.get("position_value")
        st.metric("Position Value", f"¥{pv:,.2f}" if pv is not None else "N/A")
    with col4:
        ledger = metrics.get("ledger_balanced")
        if ledger is True:
            st.metric("Ledger", ":white_check_mark: BALANCED")
        elif ledger is False:
            st.metric("Ledger", ":warning: CHECK")
        else:
            st.metric("Ledger", "N/A")

    # --- Pending orders ---
    pending = view_data["pending_orders"]
    if pending:
        st.markdown("---")
        st.subheader("Pending Orders")
        for po in pending:
            status_emoji = {"pending": ":hourglass:", "filled": ":white_check_mark:", "skipped": ":no_entry:", "failed": ":x:"}
            emoji = status_emoji.get(po.get("status", ""), "")
            st.markdown(
                f"{emoji} **[{po.get('status', '').upper()}]** `{po.get('order_id', '')}` — "
                f"created {po.get('created_on', '')}"
            )
            if po.get("reason"):
                st.caption(f"  {po['reason']}")

    # --- Assumptions ---
    st.markdown("---")
    st.subheader("Assumptions")
    assumptions = view_data["assumptions"]
    cols = st.columns(3)
    for i, (key, value) in enumerate(sorted(assumptions.items())):
        with cols[i % 3]:
            st.metric(key, str(value))

    # --- Positions ---
    st.markdown("---")
    st.subheader("Positions")
    positions = view_data["positions"]
    if positions:
        pos_data = {
            "Symbol": [p["symbol"] for p in positions],
            "Shares": [f"{p['shares']:,.0f}" for p in positions],
            "Price": [f"¥{p['price']:,.2f}" for p in positions],
            "Value": [f"¥{p['value']:,.2f}" for p in positions],
        }
        st.dataframe(pos_data, use_container_width=True)
    else:
        st.caption("No open positions.")


def main() -> None:
    st.sidebar.title("quant-personal")
    st.sidebar.markdown("Local daily-frequency research tool.")

    # --- Account Status ---
    status = paper_account_status()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Account")
    state_type = status["state_type"]

    if state_type == "no_state":
        st.sidebar.warning("No account state")
    elif state_type == "demo":
        st.sidebar.warning("DEMO / SYNTHETIC DATA")
        st.sidebar.caption("NOT YOUR DATA")
    elif state_type == "paper_simulation":
        st.sidebar.info("SIMULATED / PAPER")
        st.sidebar.caption("NOT REAL")

    if status.get("error"):
        st.sidebar.error(f"State error: {status['error']}")

    # --- Main content ---
    if state_type == "no_state":
        _render_onboarding()
        return

    # Banner
    if state_type == "demo":
        _render_demo_banner()
    else:
        _render_paper_simulation_banner()

    # === TASK 4: Status report as primary top view ===
    report = account_report(app.PROJECT_ROOT / "state" / "paper_account.json")
    _render_status_report(report)

    # --- Charts / equity curve below ---
    status_path = app.PROJECT_ROOT / "state" / "paper_account.json"
    if status_path.exists():
        st.markdown("---")
        st.subheader("Equity History")
        try:
            import json as _json
            data = _json.loads(status_path.read_text(encoding="utf-8"))
            history = data.get("history", [])
            if history:
                import pandas as pd
                eq_df = pd.DataFrame(history)
                if "timestamp" in eq_df.columns and "equity" in eq_df.columns:
                    eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
                    eq_df = eq_df.set_index("timestamp").sort_index()
                    st.line_chart(eq_df[["equity"]], use_container_width=True)
                    # Also show cash + position_value if available
                    if "cash" in eq_df.columns and "position_value" in eq_df.columns:
                        st.caption("Cash and position value over time")
                        st.line_chart(eq_df[["cash", "position_value"]], use_container_width=True)
        except Exception:
            st.caption("Equity history not available.")

    # --- Run Results ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtest Results")
    runs = list_runs()
    if runs:
        selected = st.sidebar.selectbox("Select run", [str(r) for r in runs])
        if selected:
            try:
                bt_report = backtest_report(selected)
                st.markdown("---")
                st.subheader("Latest Backtest")
                if bt_report.get("error"):
                    st.error(bt_report["error"])
                else:
                    for ml in bt_report.get("metric_lines", []):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(ml["key"], f"{ml['value']:+.4f}")
                        with col2:
                            st.caption(ml["description"])
            except Exception as exc:
                st.sidebar.error(f"Failed to load run: {exc}")
    else:
        st.sidebar.caption("No backtest results yet.")

    st.sidebar.markdown("---")
    st.sidebar.caption("quant-personal | local daily research tool")


if __name__ == "__main__":
    main()
