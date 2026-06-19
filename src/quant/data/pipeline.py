"""End-to-end load + validate helpers."""

from __future__ import annotations

import pandas as pd

from quant.config.schema import AppConfig
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
    raise ValueError(f"unknown data source: {d.source}")


def load_and_validate(cfg: AppConfig) -> pd.DataFrame:
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

