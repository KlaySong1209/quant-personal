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

from quant.app import (  # noqa: E402
    create_bundle,
    get_bundle_status,
    list_bundles,
    list_runs,
    paper_account_status,
    run_bundle_quote_step,
    update_bundle,
)
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
  --qp-bg: #0b1016;
  --qp-panel: #101821;
  --qp-panel-2: #0f1720;
  --qp-line: #233241;
  --qp-green: #35d07f;
  --qp-yellow: #f2c94c;
  --qp-red: #ff6b6b;
  --qp-blue: #56a8ff;
  --qp-text: #e6edf3;
  --qp-muted: #8b98a5;
}
.stApp {
  background: radial-gradient(circle at top left, #13202d 0%, #0b1016 36%, #070b10 100%);
  color: var(--qp-text);
}
.main .block-container {
  padding-top: 2rem;
  max-width: 1280px;
}
.qp-hero {
  border: 1px solid var(--qp-line);
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(16,24,33,.96), rgba(10,16,23,.96));
  padding: 22px 24px;
  margin-bottom: 18px;
  box-shadow: 0 18px 50px rgba(0,0,0,.28);
}
.qp-kicker {
  color: var(--qp-green);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  letter-spacing: .18em;
  text-transform: uppercase;
}
.qp-title {
  color: var(--qp-text);
  font-size: 32px;
  font-weight: 800;
  line-height: 1.12;
  margin: 8px 0 6px 0;
}
.qp-subtitle { color: var(--qp-muted); font-size: 14px; }
.qp-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 14px 0 18px 0;
}
.qp-card {
  border: 1px solid var(--qp-line);
  border-radius: 14px;
  background: rgba(16,24,33,.94);
  padding: 18px;
  min-height: 190px;
  box-shadow: 0 12px 36px rgba(0,0,0,.22);
}
.qp-card-title {
  color: var(--qp-text);
  font-size: 15px;
  font-weight: 800;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}
.qp-card-title span {
  color: var(--qp-green);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.qp-metric {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 26px;
  color: var(--qp-green);
  font-weight: 800;
  margin: 6px 0;
}
.qp-muted { color: var(--qp-muted); font-size: 13px; line-height: 1.5; }
.qp-row {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid rgba(35,50,65,.7);
  padding: 7px 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
}
.qp-row:last-child { border-bottom: 0; }
.qp-row b { color: var(--qp-text); }
.qp-status-ok { color: var(--qp-green); font-weight: 800; }
.qp-status-warn { color: var(--qp-yellow); font-weight: 800; }
.qp-status-bad { color: var(--qp-red); font-weight: 800; }
.qp-terminal {
  border: 1px solid var(--qp-line);
  border-radius: 12px;
  background: #070b10;
  padding: 14px 16px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  color: #b7c4d1;
  margin-top: 12px;
}
.stButton > button {
  border-radius: 8px;
  border: 1px solid #2f465c;
  background: linear-gradient(180deg, #172433, #101923);
  color: #e6edf3;
  font-weight: 800;
  min-height: 42px;
}
.stButton > button:hover { border-color: var(--qp-green); color: var(--qp-green); }
div[data-testid="stTextInput"] input {
  background: #080d13;
  color: var(--qp-text);
  border: 1px solid var(--qp-line);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
@media (max-width: 900px) { .qp-grid { grid-template-columns: 1fr; } }
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


# ---------------------------------------------------------------------------
# Data page (bundle view) — Stage 2
# ---------------------------------------------------------------------------


_FRESHNESS_BADGE = {
    "fresh":     ("🟢", "最新"),
    "stale":     ("🟡", "需要更新"),
    "no_data":   ("⚪", "还没有数据"),
    "no_bundle": ("⚪", "还没有股票池"),
    "error":     ("🔴", "数据异常"),
}


def _human_symbol(symbol: str) -> str:
    names = {
        "SH600519": "贵州茅台",
        "SZ000001": "平安银行",
        "SZ000002": "万科A",
    }
    return names.get(symbol, symbol)


def _symbol_input_to_list(text: str) -> list[str]:
    """Parse user-entered stock codes from comma/space/newline separated text."""
    raw = text.replace("，", ",").replace("\n", ",").replace(" ", ",")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _symbols_to_input(symbols: list[str]) -> str:
    if not symbols:
        return "600519, 000001, 000002"
    # For readability let users see bare six-digit codes where possible.
    out = []
    for sym in symbols:
        if sym.startswith(("SH", "SZ")) and len(sym) == 8:
            out.append(sym[2:])
        else:
            out.append(sym)
    return ", ".join(out)


def _stock_pool_label(name: str | None) -> str:
    if not name:
        return "未选择股票池"
    if name == "default":
        return "默认股票池"
    return str(name)


def _coverage_text(date_range: dict[str, Any]) -> str:
    if not date_range:
        return "暂无数据"
    first = date_range.get("first", "?")
    last = date_range.get("last", "?")
    return f"{first} 至 {last}"


def _bundle_view_data(status: dict[str, Any]) -> dict[str, Any]:
    """Shape a bundle status report (from ``app.get_bundle_status``) into a view model."""
    manifest = status.get("manifest") or {}
    icon, label = _FRESHNESS_BADGE.get(status.get("status", "error"), _FRESHNESS_BADGE["error"])
    return {
        "freshness_icon": icon,
        "freshness_label": label,
        "name": status.get("name"),
        "error": status.get("error"),
        "manifest": manifest,
        "symbols": manifest.get("symbols") or [],
        "date_range": manifest.get("date_range") or {},
        "source_chain": manifest.get("source_chain") or [],
        "adjustment": manifest.get("adjustment") or {},
        "calendar": manifest.get("calendar") or {},
        "row_count": manifest.get("row_count"),
        "updated_at": manifest.get("updated_at"),
        "recent_provenance": status.get("recent_provenance") or [],
    }


def _render_terminal_header() -> None:
    st.markdown(
        '<div class="qp-hero"><div class="qp-kicker">quant-personal</div>'
        '<div class="qp-title">A股量化研究终端</div>'
        '<div class="qp-subtitle">本地日频数据 · 模拟账户 · 不连接实盘</div></div>',
        unsafe_allow_html=True,
    )


def _render_data_onboarding() -> None:
    """Shown on the main operation page when no stock pool exists."""
    _render_terminal_header()
    st.markdown('<div class="qp-grid">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="qp-card-title"><span>01</span> 股票池</div>', unsafe_allow_html=True)
        codes = st.text_input(
            "输入股票代码",
            value="600519, 000001, 000002",
            help="用逗号分隔，支持 6 位数字或 SH600519 格式",
        )
        parsed = _symbol_input_to_list(codes)
        st.caption(f"将创建 {len(parsed)} 只股票的默认股票池")
        if st.button("创建股票池", type="primary"):
            result = create_bundle("default", symbols=parsed)
            if result.get("status") in {"ok", "already_exists"}:
                st.success("股票池已准备好。下一步点击“更新今日数据”。")
                st.rerun()
            else:
                st.error(f"创建失败：{result.get('error')}")
    with col2:
        st.markdown('<div class="qp-card-title"><span>02</span> 数据状态</div>', unsafe_allow_html=True)
        st.markdown('<div class="qp-metric">未创建</div><div class="qp-muted">创建股票池后即可拉取最新日线数据。</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="qp-card-title"><span>03</span> 模拟账户</div>', unsafe_allow_html=True)
        st.markdown('<div class="qp-metric">未开始</div><div class="qp-muted">数据更新后，可以推进一次本地模拟账户。</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("说明：本页面只做本地模拟研究，不连接实盘账户，不下真实订单。")

def _load_stock_pool_view(bundle_name: str | None) -> dict[str, Any] | None:
    if not bundle_name:
        return None
    status = get_bundle_status(bundle_name)
    if status.get("status") == "no_bundle":
        return {"error": f"股票池 `{bundle_name}` 不存在，请刷新页面。"}
    view = _bundle_view_data(status)
    if view["error"]:
        return {"error": f"数据异常：{view['error']}"}
    return view


def _render_data_page(bundle_name: str | None) -> None:
    """Main operation page: user-facing, card dashboard, no internal jargon."""
    if not bundle_name:
        _render_data_onboarding()
        return

    view = _load_stock_pool_view(bundle_name)
    if not view:
        _render_data_onboarding()
        return
    if view.get("error"):
        st.error(view["error"])
        return

    _render_terminal_header()
    icon = view["freshness_icon"]
    label = view["freshness_label"]
    last_date = (view["date_range"] or {}).get("last", "暂无数据")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="qp-card-title"><span>01</span> 股票池</div>', unsafe_allow_html=True)
        codes = st.text_input(
            "股票代码",
            value=_symbols_to_input(view["symbols"]),
            help="当前版本会使用已创建的股票池；如需换股，请先创建新的默认池（后续会做重置/多股票池管理）。",
        )
        rows = []
        for sym in view["symbols"]:
            rows.append(f'<div class="qp-row"><span>{_human_symbol(sym)}</span><b>{sym}</b></div>')
        st.markdown(''.join(rows) if rows else '<div class="qp-muted">还没有股票</div>', unsafe_allow_html=True)

    with col2:
        status_class = "qp-status-ok" if label == "最新" else "qp-status-warn"
        st.markdown('<div class="qp-card-title"><span>02</span> 数据状态</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="qp-metric">{icon} {label}</div>'
            f'<div class="qp-row"><span>最新交易日</span><b>{last_date}</b></div>'
            f'<div class="qp-row"><span>覆盖范围</span><b>{_coverage_text(view["date_range"])}</b></div>'
            f'<div class="qp-row"><span>股票数量</span><b>{len(view["symbols"])}</b></div>',
            unsafe_allow_html=True,
        )
        _render_update_section(view["name"])

    with col3:
        st.markdown('<div class="qp-card-title"><span>03</span> 模拟账户</div>', unsafe_allow_html=True)
        acct = paper_account_status()
        eq = acct.get("final_equity")
        if eq is None:
            st.markdown('<div class="qp-metric">未开始</div><div class="qp-muted">更新数据后，点击下方按钮推进本地模拟账户。</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="qp-metric">¥{float(eq):,.2f}</div>'
                f'<div class="qp-row"><span>步数</span><b>{acct.get("steps") or 0}</b></div>'
                f'<div class="qp-row"><span>现金</span><b>¥{float(acct.get("final_cash") or 0):,.2f}</b></div>',
                unsafe_allow_html=True,
            )
        _render_advance_section(view["name"])

    st.markdown(
        '<div class="qp-terminal">'
        '操作顺序：1 输入/确认股票池 → 2 更新今日数据 → 3 推进模拟账户。'
        '<br>所有操作只写入本地文件，不连接实盘，不下真实订单。'
        '</div>',
        unsafe_allow_html=True,
    )

def _render_data_detail_page(bundle_name: str | None) -> None:
    """Advanced data details for debugging — bundle/manifest/provenance live here."""
    if not bundle_name:
        _render_data_onboarding()
        return
    view = _load_stock_pool_view(bundle_name)
    if not view:
        _render_data_onboarding()
        return
    if view.get("error"):
        st.error(view["error"])
        return

    st.title("数据详情（高级）")
    st.caption("这里保留工程细节，用于排查数据问题。日常操作请使用“今日操作”。")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bundle", view["name"])
    with col2:
        st.metric("Freshness", f"{view['freshness_icon']} {view['freshness_label']}")
    with col3:
        rc = view["row_count"]
        st.metric("Rows", f"{rc:,}" if isinstance(rc, int) else "N/A")
    with col4:
        st.metric("Sources", " → ".join(view["source_chain"]) if view["source_chain"] else "N/A")

    st.markdown("---")
    st.subheader("Symbols")
    st.dataframe({"Canonical code": view["symbols"]}, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Adjustment**")
        st.write(view["adjustment"])
    with col_b:
        st.markdown("**Calendar**")
        st.write(view["calendar"])
    if view["updated_at"]:
        st.caption(f"Manifest last updated: {view['updated_at']}")

    st.markdown("---")
    st.subheader("Recent operations / provenance")
    prov = view["recent_provenance"]
    if not prov:
        st.caption("No recorded operations yet.")
    else:
        rows = []
        for r in prov:
            rows.append({
                "Time":   r.get("ts", ""),
                "Op":     r.get("op", ""),
                "Status": r.get("status", ""),
                "Source": r.get("source", ""),
                "Rows":   r.get("rows", ""),
                "Error":  r.get("error", "") or "",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


_UPDATE_STATE_KEY = "_qp_last_update_result"


def _render_update_section(bundle_name: str) -> None:
    """Render the data-update button and the last result (across reruns)."""
    last = st.session_state.get(_UPDATE_STATE_KEY)

    cols = st.columns([1, 4])
    with cols[0]:
        clicked = st.button("更新今日数据", type="primary", key=f"update_btn_{bundle_name}")
    with cols[1]:
        st.caption(
            "从通达信行情源拉取当前股票池的最新日线数据，并保存到本地。"
        )

    if clicked:
        with st.spinner("正在更新行情数据…"):
            try:
                result = update_bundle(bundle_name)
            except Exception as exc:  # noqa: BLE001 — surface the cause to the user
                result = {
                    "status": "failed",
                    "bundle": bundle_name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "rows_added": 0, "rows_skipped": 0, "rows_conflicting": 0,
                    "symbols_ok": [], "symbols_failed": {},
                    "new_last_date": None, "raw_paths": [],
                }
        st.session_state[_UPDATE_STATE_KEY] = result
        st.rerun()

    if last and last.get("bundle") == bundle_name:
        _render_update_result(last)


def _render_update_result(result: dict[str, Any]) -> None:
    status = result.get("status")
    if status == "ok":
        added = result.get("rows_added", 0)
        skipped = result.get("rows_skipped", 0)
        last_date = result.get("new_last_date") or "未变化"
        if added:
            st.success(f"数据更新完成：新增 {added} 行，最新交易日 {last_date}。")
        else:
            st.info(f"数据已经是最新：没有新增行，已匹配 {skipped} 行。")
    elif status == "partial":
        st.warning(
            f"部分更新成功：新增 {result.get('rows_added', 0)} 行，"
            f"有 {len(result.get('symbols_failed') or {})} 只股票失败。"
        )
    elif status == "no_bundle":
        st.error(result.get("error") or "没有找到股票池。")
    else:
        st.error(f"更新失败：{result.get('error') or status}")

    failed = result.get("symbols_failed") or {}
    if failed:
        st.markdown("**失败股票：**")
        st.dataframe(
            [{"股票代码": k, "原因": v} for k, v in failed.items()],
            use_container_width=True, hide_index=True,
        )


_ADVANCE_STATE_KEY = "_qp_last_advance_result"


def _render_advance_section(bundle_name: str) -> None:
    """Render the bundle-backed account advance button and last result."""
    last = st.session_state.get(_ADVANCE_STATE_KEY)

    cols = st.columns([1, 4])
    with cols[0]:
        clicked = st.button("推进模拟账户", key=f"advance_btn_{bundle_name}")
    with cols[1]:
        st.caption(
            "使用最新收盘价，按当前策略推进一次本地模拟账户。不会下真实订单。"
        )

    if clicked:
        with st.spinner("正在推进模拟账户…"):
            try:
                result = run_bundle_quote_step(bundle_name=bundle_name)
                result = {"status": "ok", **result}
            except Exception as exc:  # noqa: BLE001 — surface the cause to the user
                result = {
                    "status": "failed",
                    "bundle_name": bundle_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        st.session_state[_ADVANCE_STATE_KEY] = result
        st.rerun()

    if last and last.get("bundle_name") == bundle_name:
        _render_advance_result(last)


def _render_advance_result(result: dict[str, Any]) -> None:
    if result.get("status") == "ok":
        equity = result.get("final_equity", 0)
        advanced_to = result.get("advanced_to") or "未知日期"
        st.success(
            f"模拟账户已推进到 {advanced_to}。当前权益 {equity:,.2f}，"
            f"账本平衡：{result.get('ledger_balanced')}。"
        )
    else:
        st.error(f"推进失败：{result.get('error')}")


# ---------------------------------------------------------------------------
# Sidebar helpers
# ---------------------------------------------------------------------------


def _render_bundle_sidebar(bundles: list[dict[str, Any]]) -> str | None:
    """Render the stock-pool selector and return the chosen bundle name."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("股票池")
    if not bundles:
        st.sidebar.caption("还没有股票池")
        return None

    names = [b["name"] for b in bundles]
    labels = {_stock_pool_label(n): n for n in names}
    selected_label = st.sidebar.selectbox(
        "选择股票池",
        list(labels.keys()),
        label_visibility="collapsed",
    )
    selected = labels[selected_label]
    chosen = next((b for b in bundles if b["name"] == selected), bundles[0])
    icon, label = _FRESHNESS_BADGE.get(chosen.get("freshness_status", "error"),
                                       _FRESHNESS_BADGE["error"])
    st.sidebar.caption(f"{icon} {label}")
    dr = chosen.get("date_range") or {}
    if dr:
        st.sidebar.caption(f"数据：{dr.get('first', '?')} 至 {dr.get('last', '?')}")
    return selected


def main() -> None:
    st.markdown(_terminal_css(), unsafe_allow_html=True)
    st.sidebar.title("quant-personal")
    st.sidebar.markdown("本地日频量化研究工具")

    page = st.sidebar.radio(
        "页面",
        ["今日操作", "模拟账户", "数据详情（高级）"],
        label_visibility="collapsed",
    )

    bundles = list_bundles()
    selected_bundle = _render_bundle_sidebar(bundles)

    if page == "今日操作":
        _render_data_page(selected_bundle)
        return
    if page == "数据详情（高级）":
        _render_data_detail_page(selected_bundle)
        return

    _render_account_page()


def _render_account_page() -> None:
    """The original Account view; unchanged behavior."""
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
