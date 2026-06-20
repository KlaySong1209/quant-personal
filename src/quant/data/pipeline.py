"""End-to-end load + validate helpers."""

from __future__ import annotations

import pandas as pd

from quant.config.schema import AppConfig
from quant.data.adjust.calendar import build_trading_calendar
from quant.data.local import LocalFileSource, ingest_local_file, load_column_mapping
from quant.data.sources import CSVSource, DataSource, SyntheticSource
from quant.data.validate import validate_ohlcv


def build_source(cfg: AppConfig) -> DataSource:
    d = cfg.data
    if d.source == "synthetic":
        s = d.synthetic
        return SyntheticSource(
            symbols=list(d.symbols),
            start=d.start,
            end=d.end,
            initial_price=s.initial_price,
            annual_drift=s.annual_drift,
            annual_vol=s.annual_vol,
            seed=cfg.run.seed,
        )
    if d.source == "csv":
        return CSVSource(d.csv_path, list(d.symbols))
    if d.source == "local_file":
        if not d.local_path or not d.column_mapping_path:
            raise ValueError("local_file data source requires local_path and column_mapping_path")
        return LocalFileSource(d.local_path, load_column_mapping(d.column_mapping_path))
    raise ValueError(f"unknown data source: {d.source}")


def load_and_validate(cfg: AppConfig) -> pd.DataFrame:
    if cfg.data.source == "local_file":
        if not cfg.data.local_path or not cfg.data.column_mapping_path:
            raise ValueError("local_file data source requires local_path and column_mapping_path")
        cal_cfg = cfg.data.calendar
        cal_mode = cal_cfg.mode or cal_cfg.source
        cal_file = cal_cfg.file or cal_cfg.path
        cal_mapping = cal_cfg.column_mapping or cal_cfg.column_map
        calendar = build_trading_calendar(
            mode=cal_mode,
            file=cal_file,
            column_mapping=cal_mapping,
            start=cfg.data.start,
            end=cfg.data.end,
            exchange=cal_cfg.exchange,
            date_format=cal_cfg.date_format,
            timezone=cal_cfg.timezone,
            production_data=cfg.data.production_data,
        )
        result = ingest_local_file(
            cfg.data.local_path,
            mapping=load_column_mapping(cfg.data.column_mapping_path),
            symbols=list(cfg.data.symbols),
            output_dir=cfg.data.processed_output_dir,
            calendar=calendar,
            max_unadjusted_log_return=cfg.data.corporate_actions.max_unadjusted_log_return,
            adjustment_convention=cfg.data.corporate_actions.adjustment_convention,
            has_adjustment_factor=cfg.data.corporate_actions.has_adjustment_factor,
            dividend_tax_treatment=cfg.data.corporate_actions.dividend_tax_treatment,
            production_data=cfg.data.production_data,
        )
        return result.ohlcv
    return validate_ohlcv(
        build_source(cfg).load(),
        max_missing_ratio=cfg.data.validation.max_missing_ratio,
    )


def to_close_panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    panel = ohlcv.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    if panel.isna().any().any():
        bad = panel.columns[panel.isna().any()].tolist()
        raise ValueError(f"close panel has NaNs in columns: {bad}")
    return panel
