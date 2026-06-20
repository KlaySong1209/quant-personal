"""Actionable error messages for intermittent use.

Converts fail-fast errors into messages that name the step, cause, and fix action.
No silent fallbacks — every error condition produces an explanation a returning
user can act on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def diagnose_stale_quotes(
    quote_latest: str | None,
    calendar_latest: str | None,
    quote_path: str | None = None,
) -> str | None:
    """Return an actionable message if quotes end before the calendar expects, or None."""
    if quote_latest is None or calendar_latest is None:
        return None
    if quote_latest < calendar_latest:
        path_hint = f" (file: {quote_path})" if quote_path else ""
        return (
            f"quotes end {quote_latest} but calendar expects {calendar_latest}; "
            f"append missing rows and retry{path_hint}"
        )
    return None


def diagnose_missing_open(
    missing_symbols: list[str],
    timestamp: str,
    policy: str,
) -> str:
    """Return an actionable message for unfilled pending orders due to missing open."""
    syms = ", ".join(missing_symbols)
    action = {
        "skip": "order was skipped; add open prices to quote data and re-run to fill",
        "fail": "execution halted; add open prices for these symbols and retry",
        "fallback_to_prev_close": "order filled using previous close as fallback (degraded); consider adding real open data",
    }.get(policy, "unknown policy")
    return (
        f"pending order cannot fill at {timestamp}: missing open prices for [{syms}]. "
        f"policy={policy}: {action}"
    )


def diagnose_config_drift(
    current_config: dict[str, Any] | None,
    saved_config: dict[str, Any] | None,
    config_path: str | None = None,
) -> str | None:
    """Return an actionable message if the current config differs from the saved run config, or None."""
    if current_config is None or saved_config is None:
        return None
    diffs = []
    for key in sorted(set(current_config.keys()) | set(saved_config.keys())):
        cv = current_config.get(key)
        sv = saved_config.get(key)
        if cv != sv:
            diffs.append(f"  {key}: saved={sv!r}, current={cv!r}")
    if not diffs:
        return None
    path_hint = f" (config: {config_path})" if config_path else ""
    return (
        f"config has changed since last run{path_hint}:\n"
        + "\n".join(diffs)
        + "\naction: review the differences above. To accept new config, start a fresh run. "
        "To keep old config, restore the saved version."
    )


def diagnose_no_data(
    data_dir: str | None = None,
    quotes_dir: str | None = None,
) -> str:
    """Return an actionable message when no local data is found."""
    hints = []
    if data_dir:
        hints.append(f"processed data directory: {data_dir}")
    if quotes_dir:
        hints.append(f"quotes directory: {quotes_dir}")
    hint_str = " or ".join(hints) if hints else "expected locations"
    return (
        f"no local data found in {hint_str}. "
        "Run: python -m quant --generate-example-data to create synthetic data, "
        "or python -m quant --ingest-local-data <path> --column-mapping <mapping> --symbols A B C "
        "to ingest your own CSV."
    )


def diagnose_account_state_corrupt(
    state_path: str,
    reason: str,
) -> str:
    """Return an actionable message when the account state file is unreadable."""
    return (
        f"account state file is corrupt or unreadable: {state_path}\n"
        f"reason: {reason}\n"
        f"action: delete or rename {state_path} and start a new paper session, "
        f"or restore from a backup."
    )


def diagnose_symbol_mismatch(
    requested: list[str],
    available: list[str],
    source: str = "data file",
) -> str:
    """Return an actionable message when requested symbols don't match available data."""
    missing = sorted(set(requested) - set(available))
    extra = sorted(set(available) - set(requested))
    parts = [f"symbol mismatch in {source}:"]
    if missing:
        parts.append(f"  requested but not in data: {missing}")
    if extra:
        parts.append(f"  in data but not requested: {extra}")
    parts.append(
        "action: update your symbol list to match the data, "
        "or re-ingest with the correct symbols."
    )
    return "\n".join(parts)


def actionable_error(
    step: str,
    cause: str,
    fix: str,
    *,
    details: dict[str, Any] | None = None,
) -> str:
    """Format an actionable error message with step, cause, and fix.

    Args:
        step: which step of the process failed (e.g. "data detection", "account advance")
        cause: what went wrong (e.g. "no quotes for 2026-06-19")
        fix: what the user should do (e.g. "append missing rows to quotes.csv")
        details: optional extra context dict
    """
    lines = [
        "=" * 60,
        f"  ERROR — step: {step}",
        f"  cause:  {cause}",
        f"  fix:    {fix}",
    ]
    if details:
        lines.append("  details:")
        for k, v in sorted(details.items()):
            lines.append(f"    {k}: {v}")
    lines.append("=" * 60)
    return "\n".join(lines)
