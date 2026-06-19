"""Fail-fast OHLCV validation."""

from __future__ import annotations

import pandas as pd

from quant.data.schema import NUMERIC_COLUMNS, OHLCV_COLUMNS, coerce_ohlcv


class DataValidationError(ValueError):
    pass


def _ensure_input_tz_aware(df: pd.DataFrame) -> None:
    ts = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        tz = getattr(ts.dt, "tz", None)
        if tz is None:
            raise DataValidationError("timestamp column must be tz-aware")
    elif ts.dtype == object:
        sample = ts.dropna().astype(str)
        if not sample.empty and not sample.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$").all():
            raise DataValidationError("timestamp strings must include an explicit UTC offset")


def validate_ohlcv(df: pd.DataFrame, *, max_missing_ratio: float = 0.01) -> pd.DataFrame:
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")
    _ensure_input_tz_aware(df)
    out = coerce_ohlcv(df).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    for sym, sub in out.groupby("symbol", sort=False):
        if not sub["timestamp"].is_monotonic_increasing:
            raise DataValidationError(f"timestamps not monotonic for {sym}")
        if sub["timestamp"].duplicated().any():
            raise DataValidationError(f"duplicate timestamps for {sym}")
    for col in NUMERIC_COLUMNS:
        if (out[col] < 0).fillna(False).any():
            raise DataValidationError(f"negative values found in {col}")
        ratio = out[col].isna().mean()
        if ratio > max_missing_ratio:
            raise DataValidationError(
                f"missing-value ratio for {col} = {ratio:.4f} exceeds {max_missing_ratio:.4f}"
            )
    if (out["high"] < out[["open", "close"]].max(axis=1)).any():
        raise DataValidationError("high < max(open, close)")
    if (out[["open", "close"]].min(axis=1) < out["low"]).any():
        raise DataValidationError("min(open, close) < low")
    return out

