"""Daily research loop. Thin orchestration over existing src/quant functions.

Walks the end-of-day routine:
  1. Detect whether new local data/quotes exist
  2. Advance the persistent paper account to the latest available date
  3. Optionally run the current config's backtest for reference
  4. Emit a "today summary"

Idempotent: re-running on the same day must not double-advance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant.app import (
    run_paper_session,
    run_manual_quote_step,
    run_backtest_experiment,
    paper_account_status,
    list_experiment_configs,
    PROJECT_ROOT,
    CONFIG_DIR,
)
from quant.config.loader import load_config
from quant.data.local import read_processed_ohlcv
from quant.data.validate import validate_ohlcv
from quant.errors import actionable_error, diagnose_config_drift, diagnose_no_data
from quant.report import combined_report, format_combined_report

DEFAULT_STATE_PATH = PROJECT_ROOT / "state" / "paper_account.json"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_QUOTES_DIR = PROJECT_ROOT / "data" / "quotes"


def _detect_latest_data_date(data_dir: Path) -> pd.Timestamp | None:
    """Find the latest date with processed OHLCV data in *data_dir*."""
    latest_ts, _latest_path, _error = _detect_latest_data_file(data_dir)
    return latest_ts


def _detect_latest_data_file(data_dir: Path) -> tuple[pd.Timestamp | None, Path | None, str | None]:
    """Find the latest processed OHLCV file in *data_dir*."""
    ohlcv_path = data_dir / "local_daily_ohlcv.parquet"
    if not ohlcv_path.exists():
        ohlcv_path = data_dir / "local_daily_ohlcv.csv"
    if not ohlcv_path.exists():
        return None, None, None
    try:
        ohlcv = validate_ohlcv(read_processed_ohlcv(ohlcv_path), max_missing_ratio=0.0)
        return ohlcv["timestamp"].max(), ohlcv_path, None
    except Exception as exc:
        return None, None, actionable_error(
            "data detection",
            f"cannot read processed data file {ohlcv_path}: {exc}",
            "re-run ingestion to create a valid local_daily_ohlcv parquet/csv file, then retry",
        )


def _detect_latest_quote_date(quotes_dir: Path) -> pd.Timestamp | None:
    """Find the latest quote CSV in *quotes_dir* and return its max timestamp."""
    latest_ts, _latest_path, _error = _detect_latest_quote_file(quotes_dir)
    return latest_ts


def _detect_latest_quote_file(quotes_dir: Path) -> tuple[pd.Timestamp | None, Path | None, str | None]:
    """Find the latest valid quote CSV in *quotes_dir*.

    Returns (latest timestamp, path, actionable error).  Malformed quote files are
    not ignored here because the daily loop is an operator-facing command.
    """
    if not quotes_dir.exists():
        return None, None, None
    csv_files = sorted(quotes_dir.glob("*.csv"))
    if not csv_files:
        return None, None, None
    latest_ts: pd.Timestamp | None = None
    latest_path: Path | None = None
    for csv_path in csv_files:
        try:
            dates = _read_quote_dates(csv_path)
        except Exception as exc:
            return None, None, actionable_error(
                "quote detection",
                f"cannot read quote file {csv_path}: {exc}",
                "fix the quote CSV so it has a date/timestamp/as_of_date column, symbol, and close values; then retry",
            )
        if not dates:
            continue
        ts = max(dates)
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_path = csv_path
    return latest_ts, latest_path, None


def _read_quote_dates(quote_path: str | Path) -> list[pd.Timestamp]:
    """Return sorted unique quote dates from a local quote CSV."""
    path = Path(quote_path)
    if not path.exists():
        raise FileNotFoundError(f"quote file not found: {path}")
    df = pd.read_csv(path)
    ts_col = None
    for col in ("timestamp", "date", "as_of_date"):
        if col in df.columns:
            ts_col = col
            break
    if ts_col is None:
        raise ValueError("missing quote date column: expected one of timestamp, date, as_of_date")
    if "symbol" not in df.columns:
        raise ValueError("missing quote symbol column")
    if "close" not in df.columns:
        raise ValueError("missing quote close column")
    timestamps = pd.to_datetime(df[ts_col], utc=True, errors="raise")
    return sorted(pd.Timestamp(ts).normalize() for ts in timestamps.dropna().unique())


def _account_advanced_to(state_path: Path) -> pd.Timestamp | None:
    """Return the last date the account was advanced to, or None."""
    status = paper_account_status(state_path)
    history_len = status.get("steps", 0) or 0
    if history_len == 0:
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        history = data.get("history", []) or []
        if history:
            last_ts = history[-1].get("timestamp", "")
            if last_ts:
                return pd.Timestamp(last_ts)
    except Exception:
        pass
    return None


def run_daily(
    *,
    state_path: str | Path = DEFAULT_STATE_PATH,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    quotes_dir: str | Path = DEFAULT_QUOTES_DIR,
    data_path: str | Path | None = None,
    quote_path: str | Path | None = None,
    symbols: list[str] | None = None,
    run_backtest: bool = False,
    config_path: str | Path | None = None,
    starting_cash: float = 100000.0,
    fill_price_rule: str = "next_day_open",
    missing_open_policy: str = "skip",
    mode: str = "paper_simulation",
    production_data: bool = False,
) -> dict[str, Any]:
    """Run the daily end-of-day routine.

    Returns a dict with:
      - status: "ok" | "no_new_data" | "error"
      - account_advanced: bool
      - latest_data_date: ISO string or None
      - account_date: ISO string or None
      - steps_run: number of steps advanced
      - backtest_result: dict or None
      - report: the combined report dict
      - report_text: printable text report
      - error: error message or None
    """
    state = Path(state_path)
    data = Path(data_dir)
    quotes = Path(quotes_dir)

    # 1. Detect latest available data date
    latest_data, detected_data_path, data_detection_error = _detect_latest_data_file(data)
    if data_detection_error:
        return _error_result(data_detection_error)
    latest_quote, detected_quote_path, quote_detection_error = _detect_latest_quote_file(quotes)
    if quote_detection_error:
        return _error_result(quote_detection_error)

    # Use explicit data_path / quote_path if provided
    if data_path is not None:
        ohlcv_path = Path(data_path)
        try:
            ohlcv = validate_ohlcv(read_processed_ohlcv(ohlcv_path), max_missing_ratio=0.0)
            latest_data = ohlcv["timestamp"].max()
            detected_data_path = ohlcv_path
        except Exception as exc:
            return _error_result(
                actionable_error(
                    "data detection",
                    f"failed to read data_path {data_path}: {exc}",
                    "provide a valid processed OHLCV parquet/csv file, or re-run ingestion and retry",
                )
            )

    selected_quote_path = detected_quote_path
    if quote_path is not None:
        qp = Path(quote_path)
        try:
            quote_dates = _read_quote_dates(qp)
        except Exception as exc:
            return _error_result(
                actionable_error(
                    "quote detection",
                    f"cannot read quote file {qp}: {exc}",
                    "fix the quote CSV so it has date, symbol, close, and optional open columns; then retry",
                )
            )
        if quote_dates:
            qts = max(quote_dates)
            if latest_quote is None or qts > latest_quote:
                latest_quote = qts
            selected_quote_path = qp

    # Determine effective latest date
    effective_latest = latest_data
    if latest_quote is not None:
        if effective_latest is None or latest_quote > effective_latest:
            effective_latest = latest_quote

    if effective_latest is None:
        return _error_result(
            actionable_error(
                "data detection",
                f"no processed data or local quote CSV found in {data_dir} or {quotes_dir}",
                diagnose_no_data(str(data_dir), str(quotes_dir)),
            )
        )

    drift_error = _account_config_drift_error(
        state,
        fill_price_rule=fill_price_rule,
        missing_open_policy=missing_open_policy,
        mode=mode,
    )
    if drift_error:
        return _error_result(drift_error)

    # 2. Check if account needs advancing
    account_date = _account_advanced_to(state)
    if account_date is not None and account_date >= effective_latest:
        # Already up to date
        report = combined_report(state, None)
        if run_backtest:
            bt_result = _maybe_run_backtest(config_path)
            if bt_result:
                report = combined_report(state, bt_result.get("run_dir"))
        return {
            "status": "no_new_data",
            "account_advanced": False,
            "latest_data_date": effective_latest.isoformat(),
            "account_date": account_date.isoformat(),
            "steps_run": 0,
            "backtest_result": bt_result if run_backtest else None,
            "report": report,
            "report_text": format_combined_report(report),
            "error": None,
        }

    # 3. Advance the account
    # Determine symbols from existing state or default config
    if symbols is None:
        if state.exists():
            try:
                state_data = json.loads(state.read_text(encoding="utf-8"))
                broker = state_data.get("broker", {})
                symbols = list(broker.get("positions", {}).keys()) or list(
                    broker.get("last_prices", {}).keys()
                )
            except Exception:
                pass
        if not symbols:
            # Try loading from config
            configs = list_experiment_configs()
            if configs:
                cfg = load_config(configs[0])
                symbols = cfg.universe.symbols if hasattr(cfg.universe, "symbols") else []
        if not symbols:
            return _error_result(
                "cannot determine symbols. Provide --symbols or create an account state first."
            )

    if len(symbols) < 3 or len(symbols) > 5:
        return _error_result(
            f"daily loop requires 3-5 symbols, got {len(symbols)}: {symbols}"
        )

    steps_run = 0
    try:
        use_quotes = False
        if selected_quote_path is not None:
            if quote_path is not None:
                use_quotes = True
            elif data_path is None and (latest_data is None or (latest_quote is not None and latest_quote >= latest_data)):
                use_quotes = True

        if use_quotes:
            # Use manual quote stepping
            dates_to_run = _quote_dates_after(selected_quote_path, account_date, effective_latest)
            if not dates_to_run:
                return _error_result(
                    actionable_error(
                        "quote advance",
                        f"no quote rows newer than account date {account_date}",
                        "append a newer dated row for every configured symbol and retry",
                    )
                )
            for as_of_str in dates_to_run:
                result = run_manual_quote_step(
                    quote_path=selected_quote_path,
                    symbols=symbols,
                    state_path=state,
                    output_dir=PROJECT_ROOT / "results" / "paper_session",
                    starting_cash=starting_cash,
                    as_of=as_of_str,
                    fill_price_rule=fill_price_rule,
                    missing_open_policy=missing_open_policy,
                    mode=mode,
                    production_data=production_data,
                )
                steps_run += 1
        elif data_path is not None:
            # Use processed data for full paper session
            result = run_paper_session(
                data_path=data_path,
                symbols=symbols,
                state_path=state,
                output_dir=PROJECT_ROOT / "results" / "paper_session",
                starting_cash=starting_cash,
                fill_price_rule=fill_price_rule,
                missing_open_policy=missing_open_policy,
                mode=mode,
                production_data=production_data,
            )
            steps_run = result.get("steps", 0)
        else:
            # Try auto-detecting data file
            ohlcv_path = detected_data_path
            if ohlcv_path is not None and ohlcv_path.exists():
                result = run_paper_session(
                    data_path=ohlcv_path,
                    symbols=symbols,
                    state_path=state,
                    output_dir=PROJECT_ROOT / "results" / "paper_session",
                    starting_cash=starting_cash,
                    fill_price_rule=fill_price_rule,
                    missing_open_policy=missing_open_policy,
                    mode=mode,
                    production_data=production_data,
                )
                steps_run = result.get("steps", 0)
            else:
                return _error_result(
                    actionable_error(
                        "data detection",
                        f"no processed data file found at {data}",
                        "run ingestion to create local_daily_ohlcv.parquet/csv, or provide a quote CSV",
                    )
                )
    except Exception as exc:
        return _error_result(
            actionable_error(
                "account advance",
                str(exc),
                "check the data/quote file for the named date and symbols, fix the missing or invalid rows, then retry",
            )
        )

    # 4. Optionally run backtest
    bt_result = None
    if run_backtest:
        bt_result = _maybe_run_backtest(config_path)
        if bt_result and bt_result.get("error"):
            return _error_result(bt_result["error"])

    # 5. Generate report
    report = combined_report(
        state,
        bt_result.get("run_dir") if bt_result else None,
    )

    return {
        "status": "ok",
        "account_advanced": True,
        "latest_data_date": effective_latest.isoformat(),
        "account_date": str(_account_advanced_to(state)),
        "steps_run": steps_run,
        "backtest_result": bt_result,
        "report": report,
        "report_text": format_combined_report(report),
        "error": None,
    }


def _iter_dates(
    from_date: pd.Timestamp | None,
    to_date: pd.Timestamp,
) -> list[str]:
    """List dates from *from_date*+1 day through *to_date* as ISO date strings."""
    dates = []
    start = from_date + pd.Timedelta(days=1) if from_date is not None else to_date - pd.Timedelta(days=1)
    # Generate a range and filter to dates <= to_date
    current = pd.Timestamp(start.date(), tz="UTC")
    end_date = pd.Timestamp(to_date.date(), tz="UTC")
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += pd.Timedelta(days=1)
    return dates


def _quote_dates_after(
    quote_path: str | Path,
    account_date: pd.Timestamp | None,
    latest_date: pd.Timestamp,
) -> list[str]:
    """Return quote dates that should be applied to the account."""
    dates = _read_quote_dates(quote_path)
    end = pd.Timestamp(latest_date).normalize()
    result = []
    for ts in dates:
        ts = pd.Timestamp(ts).normalize()
        if ts > end:
            continue
        if account_date is not None and ts <= pd.Timestamp(account_date).normalize():
            continue
        result.append(ts.strftime("%Y-%m-%d"))
    return result


def _account_config_drift_error(
    state_path: Path,
    *,
    fill_price_rule: str,
    missing_open_policy: str,
    mode: str,
) -> str | None:
    """Return an actionable config-drift error for an existing account."""
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return actionable_error(
            "config check",
            f"cannot read existing account state {state_path}: {exc}",
            "repair, rename, or restore the account state file, then retry",
        )
    saved_assumptions = data.get("assumptions") or {}
    saved: dict[str, Any] = {}
    current: dict[str, Any] = {}
    for key, value in {
        "fill_price_rule": fill_price_rule,
        "missing_open_policy": missing_open_policy,
    }.items():
        if key in saved_assumptions:
            saved[key] = saved_assumptions[key]
            current[key] = value
    if "mode" in data:
        saved["mode"] = data["mode"]
        current["mode"] = mode
    drift = diagnose_config_drift(current, saved, config_path=str(state_path))
    if not drift:
        return None
    return actionable_error(
        "config check",
        "current daily command differs from the existing paper account state",
        drift,
    )


def _maybe_run_backtest(config_path: str | Path | None) -> dict[str, Any] | None:
    """Run backtest if a config is available, returning a summary dict or None."""
    try:
        if config_path:
            artifacts = run_backtest_experiment(load_config(config_path))
        else:
            configs = list_experiment_configs()
            if not configs:
                return None
            artifacts = run_backtest_experiment(load_config(configs[0]))
        return {
            "run_dir": str(artifacts.run_dir),
            "metrics": artifacts.metrics,
        }
    except Exception as exc:
        return {
            "error": actionable_error(
                "backtest",
                str(exc),
                "review the selected config and local data paths, then retry with --with-backtest",
            )
        }


def _error_result(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "account_advanced": False,
        "latest_data_date": None,
        "account_date": None,
        "steps_run": 0,
        "backtest_result": None,
        "report": None,
        "report_text": "",
        "error": message,
    }
