"""Experiment runner."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from quant.backtest.costs import BpsCostModel
from quant.backtest.engine import BacktestResult, run_backtest
from quant.backtest.metrics import compute_metrics
from quant.config.loader import config_to_dict
from quant.config.schema import AppConfig, FuturesAppConfig
from quant.data.adjust.calendar import CalendarColumnMap, TradingCalendar, align_close_panel_with_tradable, load_calendar
from quant.data.adjust.corporate_actions import CorporateActionColumnMap, adjust_ohlcv_for_corporate_actions, flag_implausible_unadjusted_jumps, load_corporate_actions
from quant.data.adjust.universe import UniverseColumnMap, all_symbols_universe, build_universe_mask, load_universe_membership, synthetic_universe_with_delisting
from quant.data.futures_local import FuturesColumnMap, continuous_from_local_file
from quant.data.futures_roll import assert_no_implausible_overnight_jumps, build_continuous_contract
from quant.data.futures_synth import SyntheticFuturesSource
from quant.data.pipeline import load_and_validate
from quant.experiment.logger import get_run_logger
from quant.experiment.metadata import build_metadata, hash_dataframe, hash_file, write_metadata
from quant.risk.checks import RiskConfig
from quant.strategy import build_strategy
from quant.utils.timeutils import make_run_id

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    metrics: dict
    backtest: BacktestResult


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _save_plots(plot_dir: Path, equity: pd.Series, returns: pd.Series) -> None:
    mpl_cache = PROJECT_ROOT / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*pyparsing.*")
        warnings.filterwarnings("ignore", message=r".*oneOf.*deprecated.*")
        warnings.filterwarnings("ignore", category=Warning, module=r"pyparsing\..*")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    equity.plot(ax=ax)
    ax.set_title("Equity curve")
    fig.tight_layout()
    fig.savefig(plot_dir / "equity_curve.png", dpi=120)
    plt.close(fig)
    curve = (1.0 + returns).cumprod()
    dd = curve / curve.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(8, 4))
    dd.plot(ax=ax)
    ax.set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(plot_dir / "drawdown.png", dpi=120)
    plt.close(fig)


def run_experiment(
    cfg: AppConfig | FuturesAppConfig,
    *,
    results_root: Path | None = None,
    now: datetime | None = None,
) -> RunArtifacts:
    if isinstance(cfg, FuturesAppConfig):
        return _run_futures_experiment(cfg, results_root=results_root, now=now)
    return _run_equity_experiment(cfg, results_root=results_root, now=now)


def _run_equity_experiment(cfg: AppConfig, *, results_root: Path | None, now: datetime | None) -> RunArtifacts:
    run_dir, run_id = _make_run_dir(results_root, now)
    logger = get_run_logger(run_dir)
    ohlcv = load_and_validate(cfg)
    ohlcv = _apply_equity_data_honesty(cfg, ohlcv)
    calendar = _build_equity_calendar(cfg)
    prices, tradable = align_close_panel_with_tradable(ohlcv, symbols=list(cfg.data.symbols), calendar=calendar)
    universe = _build_equity_universe(cfg, prices.index, prices.columns)
    snapshot = config_to_dict(cfg)
    snapshot["_resolved"] = {
        "data_honesty": {
            "adjusted_prices": bool(cfg.data.corporate_actions.enabled),
            "calendar_exchange": calendar.exchange,
            "pit_universe": cfg.data.universe.source,
            "tradable_mask": True,
        }
    }
    _write_yaml(run_dir / "config_snapshot.yaml", snapshot)
    logger.info("loaded prices: %d rows x %d symbols", len(prices), prices.shape[1])
    strategy = build_strategy(cfg.strategy.name, dict(cfg.strategy.params))
    target_weights = strategy.generate_weights(prices)
    cost_model = BpsCostModel(
        cfg.costs.bps,
        cfg.costs.slippage_bps,
        cfg.costs.spread_bps,
        allow_zero_cost_for_tests=cfg.costs.allow_zero_cost_for_tests,
    )
    risk = RiskConfig(cfg.risk.max_symbol_weight, cfg.risk.max_gross_leverage, cfg.risk.reject_nan)
    result = run_backtest(
        prices=prices,
        target_weights=target_weights,
        cost_model=cost_model,
        risk=risk,
        initial_equity=cfg.portfolio.initial_equity,
        tradable=tradable,
        universe=universe,
    )
    return _write_common_artifacts(run_dir, run_id, cfg, ohlcv, result)


def _run_futures_experiment(cfg: FuturesAppConfig, *, results_root: Path | None, now: datetime | None) -> RunArtifacts:
    run_dir, run_id = _make_run_dir(results_root, now)
    if cfg.data.source == "local_file_futures":
        if not cfg.data.path:
            raise ValueError("local_file_futures requires data.path")
        continuous, roll_meta = continuous_from_local_file(
            _project_path(cfg.data.path),
            FuturesColumnMap(**cfg.data.column_map),
            continuous_symbol=cfg.data.continuous_symbol,
            roll_rule=cfg.data.roll.rule,
            calendar_days_before_expiry=cfg.data.roll.calendar_days_before_expiry,
        )
    else:
        synth = SyntheticFuturesSource(
            symbol_root=cfg.data.symbol,
            start=cfg.data.start,
            n_contracts=cfg.data.synthetic.n_contracts,
            days_per_contract=cfg.data.synthetic.days_per_contract,
            overlap_days=cfg.data.synthetic.overlap_days,
            seed=cfg.run.seed,
            initial_underlying_price=cfg.data.synthetic.initial_underlying_price,
            annual_drift=cfg.data.synthetic.annual_drift,
            annual_vol=cfg.data.synthetic.annual_vol,
            basis_per_contract=cfg.data.synthetic.basis_per_contract,
        )
        continuous, roll_meta = build_continuous_contract(
            synth.contracts(),
            method=cfg.data.roll.method,
            rule=cfg.data.roll.rule,
            calendar_days_before_expiry=cfg.data.roll.calendar_days_before_expiry,
            symbol=cfg.data.continuous_symbol,
        )
    assert_no_implausible_overnight_jumps(continuous, max_log_return=cfg.data.roll.max_implausible_log_return)
    snapshot = config_to_dict(cfg)
    snapshot["_resolved"] = {"roll_metadata": roll_meta.to_dict()}
    _write_yaml(run_dir / "config_snapshot.yaml", snapshot)
    prices = continuous.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    strategy = build_strategy(cfg.strategy.name, dict(cfg.strategy.params))
    target = strategy.generate_weights(prices)
    cost = BpsCostModel(
        cfg.costs.commission_bps,
        cfg.costs.slippage_bps,
        cfg.costs.spread_bps,
        allow_zero_cost_for_tests=cfg.costs.allow_zero_cost_for_tests,
    )
    risk = RiskConfig(1.0, 10.0, cfg.risk.reject_nan)
    result = run_backtest(
        prices=prices,
        target_weights=target,
        cost_model=cost,
        risk=risk,
        initial_equity=cfg.portfolio.initial_equity,
    )
    artifacts = _write_common_artifacts(run_dir, run_id, cfg, continuous, result)
    continuous.to_parquet(run_dir / "contracts.parquet", index=False)
    return artifacts


def _make_run_dir(results_root: Path | None, now: datetime | None) -> tuple[Path, str]:
    run_id = make_run_id(now or datetime.now(timezone.utc))
    root = results_root or RESULTS_ROOT
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def _write_common_artifacts(run_dir: Path, run_id: str, cfg, input_df: pd.DataFrame, result: BacktestResult) -> RunArtifacts:
    metrics = compute_metrics(result.returns, result.weights_effective)
    _write_json(run_dir / "metrics.json", metrics)
    eq = result.equity_curve.rename("equity").to_frame()
    eq.index.name = "timestamp"
    eq.to_parquet(run_dir / "equity_curve.parquet")
    result.trades_long.to_parquet(run_dir / "trades.parquet", index=False)
    _save_plots(run_dir / "plots", result.equity_curve, result.returns)
    metadata = build_metadata(
        run_id=run_id,
        config_snapshot_relpath="config_snapshot.yaml",
        input_data_hashes=_compute_input_hashes(cfg, input_df),
        repo_root=PROJECT_ROOT,
    )
    write_metadata(run_dir / "metadata.json", metadata)
    return RunArtifacts(run_id=run_id, run_dir=run_dir, metrics=metrics, backtest=result)


def _build_equity_calendar(cfg: AppConfig) -> TradingCalendar:
    cal = cfg.data.calendar
    if cal.source == "file":
        if not cal.path:
            raise ValueError("calendar.path required")
        return load_calendar(_project_path(cal.path), CalendarColumnMap(**cal.column_map), exchange=cal.exchange)
    return TradingCalendar.synthetic(cfg.data.start, cfg.data.end, exchange=cal.exchange)


def _apply_equity_data_honesty(cfg: AppConfig, ohlcv: pd.DataFrame) -> pd.DataFrame:
    ca = cfg.data.corporate_actions
    out = ohlcv
    if ca.enabled:
        if not ca.path:
            raise ValueError("corporate_actions.path required")
        events = load_corporate_actions(_project_path(ca.path), CorporateActionColumnMap(**ca.column_map))
        out = adjust_ohlcv_for_corporate_actions(out, events)
    flag_implausible_unadjusted_jumps(out, max_log_return=ca.max_unadjusted_log_return)
    return out


def _build_equity_universe(cfg: AppConfig, index: pd.DatetimeIndex, symbols: pd.Index) -> pd.DataFrame:
    uni = cfg.data.universe
    symbol_list = list(symbols)
    if uni.source == "all":
        return all_symbols_universe(index, symbol_list)
    if uni.source == "file":
        if not uni.path:
            raise ValueError("universe.path required")
        membership = load_universe_membership(_project_path(uni.path), UniverseColumnMap(**uni.column_map))
    else:
        membership = synthetic_universe_with_delisting(
            symbols=symbol_list,
            start=index[0],
            end=index[-1],
            delisted_symbol=uni.delisted_symbol or symbol_list[-1],
            delist_date=pd.Timestamp(uni.delist_date, tz="UTC") if uni.delist_date else index[len(index) // 2],
        )
    return build_universe_mask(index, symbol_list, membership)


def _compute_input_hashes(cfg, input_df: pd.DataFrame) -> dict[str, str]:
    if hasattr(cfg, "data") and getattr(cfg.data, "source", "") == "csv":
        out = {}
        csv_dir = _project_path(cfg.data.csv_path)
        for sym in cfg.data.symbols:
            path = csv_dir / f"{sym}.csv"
            if path.exists():
                out[path.name] = hash_file(path)
        return out
    return {"input_frame": hash_dataframe(input_df)}
