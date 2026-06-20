"""Local app orchestration for CLI menu and dashboard."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from quant.config.loader import load_config
from quant.data.adjust.calendar import build_trading_calendar
from quant.data.local import ingest_local_file, load_column_mapping, read_processed_ohlcv, write_synthetic_local_export
from quant.data.pipeline import build_source, load_and_validate, to_close_panel
from quant.data.quotes import ManualQuoteSource
from quant.data.validate import validate_ohlcv
from quant.execution.account import SimAccount, AccountMode, MissingOpenPolicy
from quant.execution.paper import PaperBroker
from quant.experiment.run import RESULTS_ROOT, RunArtifacts, run_experiment
from quant.portfolio.target import target_dollars_to_shares, weights_to_target_dollars
from quant.risk.checks import RiskConfig, apply_risk_checks
from quant.strategy import build_strategy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
EXAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "example"
STREAMLIT_INSTALL_COMMAND = "pip3 install -e .[dashboard]"


def metric_explanations() -> dict[str, str]:
    return {
        "total_return": "Total return: total compounded gain or loss over the run.",
        "annualized_return": "Annualized return: average yearly return implied by the run.",
        "annualized_volatility": "Volatility: how much returns fluctuate; lower is steadier.",
        "sharpe": "Sharpe: risk-adjusted return; higher is better, below 0 means it lost money after risk.",
        "max_drawdown": "Max drawdown: worst peak-to-trough loss; closer to 0 is better.",
        "turnover": "Turnover: how much the portfolio traded; higher usually means higher costs.",
        "excess_return": "Excess return: compounded return above the benchmark.",
        "tracking_error": "Tracking error: how differently the strategy moved versus the benchmark.",
        "information_ratio": "Information ratio: benchmark-adjusted return per unit of active risk.",
        "beta": "Beta: sensitivity to the benchmark; 1 moves roughly with it.",
    }


def list_experiment_configs(config_dir: Path = CONFIG_DIR) -> list[Path]:
    return sorted(config_dir.glob("*.yaml"))


def generate_example_data(config_path: str | Path = CONFIG_DIR / "exp_placeholder.yaml") -> list[Path]:
    cfg = load_config(config_path)
    df = build_source(cfg).load()
    EXAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for sym, sub in df.groupby("symbol"):
        path = EXAMPLE_DATA_DIR / f"{sym}.csv"
        sub.to_csv(path, index=False)
        paths.append(path)
    return paths


def run_backtest_experiment(config_path: str | Path) -> RunArtifacts:
    return run_experiment(load_config(config_path))


def run_paper_demo(config_path: str | Path = CONFIG_DIR / "exp_placeholder.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    ohlcv = load_and_validate(cfg)
    prices = to_close_panel(ohlcv)

    strategy = build_strategy(cfg.strategy.name, dict(cfg.strategy.params))
    raw_weights = strategy.generate_weights(prices)

    risk = RiskConfig(
        max_symbol_weight=cfg.risk.max_symbol_weight,
        max_gross_leverage=cfg.risk.max_gross_leverage,
        reject_nan=cfg.risk.reject_nan,
    )
    weights = apply_risk_checks(raw_weights, risk)
    effective = weights.shift(1).fillna(0.0)

    target_dollars = weights_to_target_dollars(effective, cfg.portfolio.initial_equity)
    target_shares = target_dollars_to_shares(target_dollars, prices)

    broker = PaperBroker(
        starting_cash=cfg.execution.paper.starting_cash,
        allow_short=cfg.execution.paper.allow_short,
        allow_margin=cfg.execution.paper.allow_margin,
        max_gross_leverage=cfg.execution.paper.max_gross_leverage,
    )

    for ts in prices.index:
        px_row = prices.loc[ts]
        broker.update_prices(px_row.to_dict())
        broker.submit_target(ts, target_shares.loc[ts].to_dict())

    last_ts = prices.index[-1]
    equity = broker.mark_to_market(last_ts)
    ledger = broker.ledger()
    balanced = broker.is_balanced()
    if not balanced:
        raise RuntimeError("paper-broker ledger is not balanced")
    return {
        "events": len(ledger),
        "final_cash": broker.cash,
        "final_positions": broker.positions(),
        "final_equity": equity,
        "ledger_balanced": balanced,
    }


def ingest_local_data(
    *,
    data_path: str | Path,
    mapping_path: str | Path,
    symbols: list[str],
    output_dir: str | Path = PROJECT_ROOT / "data" / "processed",
    calendar_file: str | Path | None = None,
    calendar_mapping: dict[str, str] | None = None,
    calendar_exchange: str = "SYNTH",
    adjustment_convention: str | None = None,
    has_adjustment_factor: bool | None = None,
    dividend_tax_treatment: str | None = None,
    production_data: bool = False,
) -> dict[str, Any]:
    calendar = None
    if calendar_file is not None:
        calendar = build_trading_calendar(
            mode="file",
            file=calendar_file,
            column_mapping=calendar_mapping or {"date": "date"},
            exchange=calendar_exchange,
            production_data=production_data,
        )
    result = ingest_local_file(
        data_path,
        mapping=load_column_mapping(mapping_path),
        symbols=symbols,
        output_dir=output_dir,
        calendar=calendar,
        adjustment_convention=adjustment_convention,
        has_adjustment_factor=has_adjustment_factor,
        dividend_tax_treatment=dividend_tax_treatment,
        production_data=production_data,
    )
    return {
        "processed_path": result.processed_path,
        "metadata_path": result.metadata_path,
        "rows": len(result.ohlcv),
        "symbols": result.metadata["universe"]["symbols"],
        "adjustment_method": result.metadata["adjustment"]["method"],
        "calendar_source": result.metadata["calendar"]["source"],
    }


def generate_synthetic_local_export(
    *,
    output_path: str | Path,
    mapping_path: str | Path,
    symbols: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> Path:
    return write_synthetic_local_export(
        output_path,
        symbols=symbols,
        start=start,
        end=end,
        mapping=load_column_mapping(mapping_path),
        seed=seed,
    )


def _to_price_panel(ohlcv: pd.DataFrame, column: str) -> pd.DataFrame:
    panel = ohlcv.pivot(index="timestamp", columns="symbol", values=column).sort_index()
    if panel.isna().any().any():
        bad = panel.columns[panel.isna().any()].tolist()
        raise ValueError(f"{column} panel has NaNs in columns: {bad}")
    return panel


def run_paper_session(
    *,
    data_path: str | Path,
    symbols: list[str],
    state_path: str | Path = PROJECT_ROOT / "state" / "paper_account.json",
    output_dir: str | Path = PROJECT_ROOT / "results" / "paper_session",
    starting_cash: float = 100000.0,
    start: str | None = None,
    end: str | None = None,
    commission_bps: float = 1.0,
    stamp_duty_bps: float = 5.0,
    slippage_bps: float = 1.0,
    fill_price_rule: str = "same_day_close",
    missing_open_policy: str = "skip",
    mode: str = "paper_simulation",
    production_data: bool = False,
) -> dict[str, Any]:
    if not (3 <= len(symbols) <= 5):
        raise ValueError("paper sessions require a 3-5 symbols universe")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")
    if mode not in ("paper_simulation", "demo"):
        raise ValueError("mode must be 'paper_simulation' or 'demo'")
    if fill_price_rule not in ("same_day_close", "next_day_open"):
        raise ValueError("fill_price_rule must be 'same_day_close' or 'next_day_open'")
    if missing_open_policy not in ("skip", "fallback_to_prev_close", "fail"):
        raise ValueError("missing_open_policy must be 'skip', 'fallback_to_prev_close', or 'fail'")
    if production_data and mode == "demo":
        raise ValueError("demo mode is forbidden when production_data is true")
    ohlcv = validate_ohlcv(read_processed_ohlcv(data_path), max_missing_ratio=0.0)
    available = set(ohlcv["symbol"].astype(str))
    missing = [sym for sym in symbols if sym not in available]
    if missing:
        raise ValueError(f"missing requested symbol(s) in processed data: {missing}")
    ohlcv = ohlcv[ohlcv["symbol"].isin(symbols)].copy()
    prices = to_close_panel(ohlcv).reindex(columns=symbols)
    open_prices = _to_price_panel(ohlcv, "open").reindex(columns=symbols)
    if start:
        prices = prices.loc[prices.index >= pd.Timestamp(start, tz="UTC")]
        open_prices = open_prices.loc[open_prices.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        prices = prices.loc[prices.index <= pd.Timestamp(end, tz="UTC")]
        open_prices = open_prices.loc[open_prices.index <= pd.Timestamp(end, tz="UTC")]
    if prices.empty:
        raise ValueError("paper session has no prices for the requested date range")
    if prices.isna().any().any():
        bad = prices.columns[prices.isna().any()].tolist()
        raise ValueError(f"paper session prices contain missing values for: {bad}")
    if not np.isfinite(prices.to_numpy(dtype="float64")).all():
        raise ValueError("paper session prices must be finite")
    if fill_price_rule == "next_day_open":
        if open_prices.isna().any().any():
            bad = open_prices.columns[open_prices.isna().any()].tolist()
            raise ValueError(f"paper session open prices contain missing values for: {bad}")
        if not np.isfinite(open_prices.to_numpy(dtype="float64")).all():
            raise ValueError("paper session open prices must be finite")

    strategy = build_strategy("placeholder", {"mode": "equal_weight"})
    weights = strategy.generate_weights(prices)
    state = Path(state_path)
    if state.exists():
        account = SimAccount.load(state)
    else:
        account = SimAccount(
            account_id="paper-session",
            starting_cash=starting_cash,
            commission_bps=commission_bps,
            stamp_duty_bps=stamp_duty_bps,
            slippage_bps=slippage_bps,
            fill_price_rule=fill_price_rule,
            missing_open_policy=missing_open_policy,
            mode=mode,
        )

    for ts in prices.index:
        account.step(
            ts,
            prices=prices.loc[ts].to_dict(),
            open_prices=open_prices.loc[ts].to_dict() if fill_price_rule == "next_day_open" else None,
            target_weights=weights.loc[ts].to_dict(),
            save_path=state,
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = account.equity_history()
    equity_history_path = _write_dataframe(history, out / "equity_history")
    ledger_path = _write_dataframe(account.broker.ledger(), out / "ledger")
    last = history.iloc[-1]
    return {
        "label": account.to_dict()["label"],
        "assumptions": account.assumptions(),
        "state_path": state,
        "equity_history_path": equity_history_path,
        "ledger_path": ledger_path,
        "steps": int(len(history)),
        "final_cash": float(last["cash"]),
        "final_equity": float(last["equity"]),
        "positions": account.broker.positions(),
        "ledger_balanced": account.broker.is_balanced(),
    }


def run_manual_quote_step(
    *,
    quote_path: str | Path,
    symbols: list[str],
    state_path: str | Path = PROJECT_ROOT / "state" / "paper_account.json",
    output_dir: str | Path = PROJECT_ROOT / "results" / "paper_session",
    starting_cash: float = 100000.0,
    as_of: str | None = None,
    commission_bps: float = 1.0,
    stamp_duty_bps: float = 5.0,
    slippage_bps: float = 1.0,
    fill_price_rule: str = "same_day_close",
    missing_open_policy: str = "skip",
    mode: str = "paper_simulation",
    production_data: bool = False,
) -> dict[str, Any]:
    if production_data and mode == "demo":
        raise ValueError("demo mode is forbidden when production_data is true")
    column_mapping = {"timestamp": "date", "symbol": "symbol", "close": "close", "open": "open"}
    quotes = ManualQuoteSource(quote_path, column_mapping=column_mapping)
    snapshot = quotes.snapshot(symbols, as_of=as_of)
    prices = {str(row["symbol"]): float(row["close"]) for _, row in snapshot.iterrows()}
    open_prices = (
        {str(row["symbol"]): float(row["open"]) for _, row in snapshot.iterrows()}
        if "open" in snapshot.columns
        else {}
    )
    state = Path(state_path)
    if state.exists():
        account = SimAccount.load(state)
    else:
        account = SimAccount(
            account_id="manual-quote-paper-session",
            starting_cash=starting_cash,
            commission_bps=commission_bps,
            stamp_duty_bps=stamp_duty_bps,
            slippage_bps=slippage_bps,
            fill_price_rule=fill_price_rule,
            missing_open_policy=missing_open_policy,
            mode=mode,
        )
    weights = build_strategy("placeholder", {"mode": "equal_weight"}).generate_weights(
        pd.DataFrame([prices], index=[pd.Timestamp(as_of or pd.Timestamp.utcnow())])
    )
    ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.utcnow()
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    account.step(
        ts,
        prices=prices,
        open_prices=open_prices if fill_price_rule == "next_day_open" else None,
        target_weights=weights.iloc[0].to_dict(),
        save_path=state,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = account.equity_history()
    equity_history_path = _write_dataframe(history, out / "equity_history")
    ledger_path = _write_dataframe(account.broker.ledger(), out / "ledger")
    last = history.iloc[-1]
    return {
        "label": account.to_dict()["label"],
        "assumptions": account.assumptions(),
        "state_path": state,
        "equity_history_path": equity_history_path,
        "ledger_path": ledger_path,
        "steps": int(len(history)),
        "final_cash": float(last["cash"]),
        "final_equity": float(last["equity"]),
        "positions": account.broker.positions(),
        "ledger_balanced": account.broker.is_balanced(),
    }


def format_paper_demo_plain(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"events: {summary['events']}",
            f"final cash:      {summary['final_cash']:,.2f}",
            f"final positions: {summary['final_positions']}",
            f"final equity:    {summary['final_equity']:,.2f}",
            f"ledger balanced: {summary['ledger_balanced']}",
        ]
    )


def format_ingestion_plain(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"processed data: {summary['processed_path']}",
            f"metadata:       {summary['metadata_path']}",
            f"rows:           {summary['rows']}",
            f"symbols:        {summary['symbols']}",
            f"adjustment:     {summary['adjustment_method']}",
            f"calendar:       {summary['calendar_source']}",
        ]
    )


def format_paper_session_plain(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(summary["label"]),
            f"state:          {summary['state_path']}",
            f"equity history: {summary['equity_history_path']}",
            f"ledger:         {summary['ledger_path']}",
            f"steps:          {summary['steps']}",
            f"final cash:     {summary['final_cash']:,.2f}",
            f"final equity:   {summary['final_equity']:,.2f}",
            f"positions:      {summary['positions']}",
            f"ledger balanced:{summary['ledger_balanced']}",
            f"assumptions:    {summary['assumptions']}",
        ]
    )


def list_runs(results_root: Path = RESULTS_ROOT) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.iterdir() if p.is_dir()], reverse=True)


def latest_run(results_root: Path = RESULTS_ROOT) -> Path | None:
    runs = list_runs(results_root)
    return runs[0] if runs else None


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {"run_dir": str(run), "metrics": metrics, "metadata": metadata}


def load_equity_curve(run_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / "equity_curve.parquet")


def load_trades(run_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / "trades.parquet")


def _write_dataframe(df: pd.DataFrame, base_path: Path) -> Path:
    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def format_metrics_plain(metrics: dict[str, float]) -> str:
    explanations = metric_explanations()
    return "\n".join(
        f"{key}: {value:+.6f} -- {explanations.get(key, key)}"
        for key, value in metrics.items()
    )


def show_latest_results() -> str:
    run = latest_run()
    if run is None:
        return "No results found yet. Run a backtest first."
    return f"Run: {run}\n" + format_metrics_plain(load_run_summary(run)["metrics"])


def paper_account_status(state_path: str | Path | None = None, *, production_data: bool = False) -> dict[str, Any]:
    """Return a dashboard-ready view model for paper account status.

    State types:
      - no_state: no account state file exists
      - demo: account was created with demo/synthetic data
      - paper_simulation: real paper account with user data

    Dashboard MUST only render this view model; it must not guess state.
    """
    if state_path is None:
        state_path = PROJECT_ROOT / "state" / "paper_account.json"
    path = Path(state_path)
    if not path.exists():
        return {
            "state_type": "no_state",
            "label": "No Account",
            "mode": None,
            "account_id": None,
            "final_equity": None,
            "final_cash": None,
            "positions": None,
            "steps": None,
            "assumptions": None,
            "error": None,
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mode = data.get("mode", "paper_simulation")
        if production_data and mode == "demo":
            return {
                "state_type": "no_state",
                "label": "Demo Rejected",
                "mode": None,
                "account_id": None,
                "final_equity": None,
                "final_cash": None,
                "positions": None,
                "steps": None,
                "assumptions": None,
                "error": "demo account state is forbidden when production_data is true",
            }
        history = data.get("history", [])
        last = history[-1] if history else {}
        return {
            "state_type": mode,
            "label": data.get("label", ""),
            "mode": mode,
            "account_id": data.get("account_id"),
            "final_equity": last.get("equity"),
            "final_cash": last.get("cash"),
            "positions": last.get("positions"),
            "steps": len(history),
            "assumptions": data.get("assumptions"),
            "error": None,
        }
    except Exception as exc:
        return {
            "state_type": "no_state",
            "label": "Error Loading State",
            "mode": None,
            "account_id": None,
            "final_equity": None,
            "final_cash": None,
            "positions": None,
            "steps": None,
            "assumptions": None,
            "error": str(exc),
        }


def dashboard_status() -> str:
    if importlib.util.find_spec("streamlit") is None:
        return f"Streamlit is not installed. Install it with: {STREAMLIT_INSTALL_COMMAND}"
    return "Streamlit is installed. Run: streamlit run dashboard/app_streamlit.py"


def launch_dashboard() -> int:
    if importlib.util.find_spec("streamlit") is None:
        print(dashboard_status())
        return 1
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "dashboard/app_streamlit.py"])


def _choose_config() -> Path:
    configs = list_experiment_configs()
    for i, path in enumerate(configs, 1):
        print(f"{i}) {path.name}")
    return configs[int(input("Choose config number: ").strip()) - 1]


def interactive_menu() -> int:
    while True:
        print("\nquant-personal main panel")
        print("1) generate example data")
        print("2) run a backtest")
        print("3) show last results")
        print("4) run paper demo")
        print("5) launch dashboard")
        print("6) run tests")
        print("7) ingest local data")
        print("8) run/resume paper account")
        print("0) exit")
        choice = input("Select: ").strip()
        if choice == "0":
            return 0
        if choice == "1":
            for path in generate_example_data():
                print(f"wrote {path}")
        elif choice == "2":
            artifacts = run_backtest_experiment(_choose_config())
            print(f"run_dir: {artifacts.run_dir}")
            print(format_metrics_plain(artifacts.metrics))
        elif choice == "3":
            print(show_latest_results())
        elif choice == "4":
            print(format_paper_demo_plain(run_paper_demo()))
        elif choice == "5":
            return launch_dashboard()
        elif choice == "6":
            return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
        elif choice == "7":
            data_path = input("Local CSV/parquet path: ").strip()
            mapping_path = input("Column mapping YAML path: ").strip()
            symbols = [x.strip() for x in input("Symbols, comma-separated: ").split(",") if x.strip()]
            print(format_ingestion_plain(ingest_local_data(data_path=data_path, mapping_path=mapping_path, symbols=symbols)))
        elif choice == "8":
            data_path = input("Processed OHLCV path: ").strip()
            symbols = [x.strip() for x in input("Symbols, comma-separated: ").split(",") if x.strip()]
            print(format_paper_session_plain(run_paper_session(data_path=data_path, symbols=symbols)))
        else:
            print("Unknown choice.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="quant-personal main panel",
        epilog=(
            "Menu items: 1) generate example data  2) run a backtest  "
            "3) show last results  4) run paper demo  5) launch dashboard  6) run tests  "
            "7) ingest local data  8) run/resume paper account"
        ),
    )
    parser.add_argument("--generate-example-data", action="store_true")
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--paper-demo", action="store_true")
    parser.add_argument("--show-last-results", action="store_true")
    parser.add_argument("--dashboard-status", action="store_true")
    parser.add_argument("--ingest-local-data", type=Path)
    parser.add_argument("--column-mapping", type=Path)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--processed-output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--calendar-file", type=Path)
    parser.add_argument("--calendar-date-column", default="date")
    parser.add_argument("--calendar-is-open-column")
    parser.add_argument("--calendar-exchange", default="SYNTH")
    parser.add_argument("--adjustment-convention", choices=["forward", "backward", "none"])
    parser.add_argument("--has-adjustment-factor", action="store_true")
    parser.add_argument("--adjusted-price-column", action="store_true")
    parser.add_argument("--dividend-tax-treatment", choices=["pre_tax", "post_tax"])
    parser.add_argument("--production-data", action="store_true")
    parser.add_argument("--write-synthetic-local-export", type=Path)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2020-01-31")
    parser.add_argument("--run-paper-session", type=Path)
    parser.add_argument("--manual-quotes", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--account-state", type=Path, default=PROJECT_ROOT / "state" / "paper_account.json")
    parser.add_argument("--paper-output-dir", type=Path, default=PROJECT_ROOT / "results" / "paper_session")
    parser.add_argument("--starting-cash", type=float, default=100000.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--fill-price-rule", choices=["same_day_close", "next_day_open"], default="same_day_close")
    parser.add_argument("--missing-open-policy", choices=["skip", "fallback_to_prev_close", "fail"], default="skip")
    parser.add_argument("--mode", choices=["paper_simulation", "demo"], default="paper_simulation")
    args = parser.parse_args(argv)
    if args.generate_example_data:
        for path in generate_example_data():
            print(f"wrote {path}")
        return 0
    if args.run_config:
        artifacts = run_backtest_experiment(args.run_config)
        print(f"run_dir: {artifacts.run_dir}")
        print(format_metrics_plain(artifacts.metrics))
        return 0
    if args.paper_demo:
        print(format_paper_demo_plain(run_paper_demo()))
        return 0
    if args.show_last_results:
        print(show_latest_results())
        return 0
    if args.dashboard_status:
        print(dashboard_status())
        return 0
    if args.ingest_local_data:
        if not args.column_mapping or not args.symbols:
            parser.error("--ingest-local-data requires --column-mapping and --symbols")
        print(
            format_ingestion_plain(
                ingest_local_data(
                    data_path=args.ingest_local_data,
                    mapping_path=args.column_mapping,
                    symbols=args.symbols,
                    output_dir=args.processed_output_dir,
                    calendar_file=args.calendar_file,
                    calendar_mapping={
                        k: v
                        for k, v in {
                            "date": args.calendar_date_column,
                            "is_open": args.calendar_is_open_column,
                        }.items()
                        if v
                    },
                    calendar_exchange=args.calendar_exchange,
                    adjustment_convention=args.adjustment_convention,
                    has_adjustment_factor=True if args.has_adjustment_factor else (False if args.adjusted_price_column else None),
                    dividend_tax_treatment=args.dividend_tax_treatment,
                    production_data=args.production_data,
                )
            )
        )
        return 0
    if args.write_synthetic_local_export:
        if not args.column_mapping or not args.symbols:
            parser.error("--write-synthetic-local-export requires --column-mapping and --symbols")
        path = generate_synthetic_local_export(
            output_path=args.write_synthetic_local_export,
            mapping_path=args.column_mapping,
            symbols=args.symbols,
            start=args.start,
            end=args.end,
            seed=42,
        )
        print(f"wrote {path}")
        return 0
    if args.run_paper_session:
        if not args.symbols:
            parser.error("--run-paper-session requires --symbols")
        print(
            format_paper_session_plain(
                run_paper_session(
                    data_path=args.run_paper_session,
                    symbols=args.symbols,
                    state_path=args.account_state,
                    output_dir=args.paper_output_dir,
                    starting_cash=args.starting_cash,
                    commission_bps=args.commission_bps,
                    stamp_duty_bps=args.stamp_duty_bps,
                    slippage_bps=args.slippage_bps,
                    fill_price_rule=args.fill_price_rule,
                    missing_open_policy=args.missing_open_policy,
                    mode=args.mode,
                    production_data=args.production_data,
                )
            )
        )
        return 0
    if args.manual_quotes:
        if not args.symbols:
            parser.error("--manual-quotes requires --symbols")
        print(
            format_paper_session_plain(
                run_manual_quote_step(
                    quote_path=args.manual_quotes,
                    symbols=args.symbols,
                    state_path=args.account_state,
                    output_dir=args.paper_output_dir,
                    starting_cash=args.starting_cash,
                    as_of=args.as_of,
                    commission_bps=args.commission_bps,
                    stamp_duty_bps=args.stamp_duty_bps,
                    slippage_bps=args.slippage_bps,
                    fill_price_rule=args.fill_price_rule,
                    missing_open_policy=args.missing_open_policy,
                    mode=args.mode,
                    production_data=args.production_data,
                )
            )
        )
        return 0
    return interactive_menu()
