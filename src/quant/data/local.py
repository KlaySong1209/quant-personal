"""Local daily file ingestion with configurable column mappings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import yaml

from quant.data.adjust.calendar import TradingCalendar, align_close_panel_with_tradable
from quant.data.schema import OHLCV_COLUMNS
from quant.data.sources import DataSource
from quant.data.validate import validate_ohlcv

REQUIRED_LOCAL_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
OPTIONAL_LOCAL_COLUMNS = ("adjusted_close", "adjustment_factor", "dividend", "dividend_ex_date", "split", "bonus", "conversion")
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class LocalIngestionResult:
    ohlcv: pd.DataFrame
    prices: pd.DataFrame
    tradable: pd.DataFrame
    processed_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


class LocalFileSource(DataSource):
    """Read one user-exported daily OHLCV file and normalize vendor columns."""

    def __init__(self, path: str | Path, mapping: dict[str, str]):
        self.path = Path(path)
        self.mapping = dict(mapping)

    def load(self) -> pd.DataFrame:
        raw = _read_table(self.path)
        missing = [
            f"{canonical}->{self.mapping.get(canonical)}"
            for canonical in REQUIRED_LOCAL_COLUMNS
            if not self.mapping.get(canonical) or self.mapping[canonical] not in raw.columns
        ]
        if missing:
            raise ValueError(f"missing mapped column(s): {missing}")
        out = pd.DataFrame({canonical: raw[self.mapping[canonical]] for canonical in REQUIRED_LOCAL_COLUMNS})
        for canonical in OPTIONAL_LOCAL_COLUMNS:
            vendor = self.mapping.get(canonical)
            if vendor and vendor in raw.columns:
                out[canonical] = raw[vendor]
        return out


def load_column_mapping(path: str | Path) -> dict[str, str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("column mapping file must contain a mapping")
    columns = data.get("columns", data)
    if not isinstance(columns, dict):
        raise ValueError("column mapping file must contain a columns mapping")
    return {str(k): str(v) for k, v in columns.items() if v is not None}


def ingest_local_file(
    path: str | Path,
    *,
    mapping: dict[str, str],
    symbols: list[str],
    output_dir: str | Path = "data/processed",
    calendar: TradingCalendar | None = None,
    max_unadjusted_log_return: float = 0.25,
    adjustment_convention: str | None = None,
    has_adjustment_factor: bool | None = None,
    dividend_tax_treatment: str | None = None,
    production_data: bool = False,
) -> LocalIngestionResult:
    if not (3 <= len(symbols) <= 5):
        raise ValueError("local daily ingestion is limited to 3-5 symbols")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must be unique")

    source = LocalFileSource(path, mapping)
    mapped = source.load()
    cleaned, adjustment = _apply_adjustment_handling(
        mapped,
        max_unadjusted_log_return=max_unadjusted_log_return,
        adjustment_convention=adjustment_convention,
        has_adjustment_factor=has_adjustment_factor,
        dividend_tax_treatment=dividend_tax_treatment,
    )
    cleaned = validate_ohlcv(cleaned, max_missing_ratio=0.0)
    _validate_symbol_universe(cleaned, symbols)

    if calendar is None:
        if production_data:
            raise ValueError("synthetic calendar is forbidden when production_data is true")
        start = cleaned["timestamp"].min().date().isoformat()
        end = cleaned["timestamp"].max().date().isoformat()
        calendar = TradingCalendar.synthetic(start, end, exchange="SYNTH")
        calendar_meta = {
            "source": "synthetic",
            "exchange": calendar.exchange,
            "session_count": int(len(calendar.sessions)),
            "warning": "Using synthetic business-day calendar; A-share holidays are not included.",
        }
        warnings.warn(calendar_meta["warning"], RuntimeWarning, stacklevel=2)
    else:
        calendar_meta = {
            "source": "file",
            "exchange": calendar.exchange,
            "session_count": int(len(calendar.sessions)),
            "warning": None,
        }
    prices, tradable = align_close_panel_with_tradable(cleaned, symbols=list(symbols), calendar=calendar)
    alignment = _alignment_metadata(tradable)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    processed_path = _write_processed(cleaned, output / "local_daily_ohlcv")
    metadata = {
        "source": {"path": str(Path(path)), "format": Path(path).suffix.lower().lstrip(".") or "csv"},
        "mapping": dict(mapping),
        "universe": {"symbols": list(symbols), "symbol_count": len(symbols)},
        "adjustment": adjustment,
        "calendar": calendar_meta,
        "alignment": alignment,
        "processed_path": str(processed_path),
    }
    metadata_path = output / "local_daily_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return LocalIngestionResult(
        ohlcv=cleaned,
        prices=prices,
        tradable=tradable,
        processed_path=processed_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def write_synthetic_local_export(
    path: str | Path,
    *,
    symbols: list[str],
    start: str,
    end: str,
    mapping: dict[str, str],
    seed: int = 42,
) -> Path:
    if not (3 <= len(symbols) <= 5):
        raise ValueError("synthetic local export is limited to 3-5 symbols")
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, end=end)
    if len(idx) == 0:
        raise ValueError("empty date range")
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        shocks = rng.normal(0.0002, 0.01, len(idx))
        close = 100.0 * np.exp(np.cumsum(shocks))
        open_ = np.empty(len(idx))
        open_[0] = close[0]
        open_[1:] = close[:-1]
        spread = np.maximum(close * 0.002, 0.01)
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.integers(1_000, 10_000, len(idx))
        for i, ts in enumerate(idx):
            canonical = {
                "timestamp": ts.date().isoformat(),
                "symbol": sym,
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "volume": int(volume[i]),
                "adjusted_close": float(close[i]),
            }
            rows.append(
                {
                    vendor: canonical[canonical_name]
                    for canonical_name, vendor in mapping.items()
                    if canonical_name in canonical
                }
            )
    out = pd.DataFrame(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return target


def read_processed_ohlcv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"local data file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    raise ValueError(f"unsupported local data file format: {path.suffix}")


def _apply_adjustment_handling(
    mapped: pd.DataFrame,
    *,
    max_unadjusted_log_return: float,
    adjustment_convention: str | None,
    has_adjustment_factor: bool | None,
    dividend_tax_treatment: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = mapped.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["symbol"] = out["symbol"].astype(str)
    for col in ("open", "high", "low", "close", "volume", "adjusted_close", "adjustment_factor", "dividend", "split", "bonus", "conversion"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise")

    declarations = _adjustment_declarations(
        out,
        adjustment_convention=adjustment_convention,
        has_adjustment_factor=has_adjustment_factor,
        dividend_tax_treatment=dividend_tax_treatment,
    )
    if "adjustment_factor" in out.columns:
        if has_adjustment_factor is not True:
            raise ValueError("adjustment_factor column requires has_adjustment_factor=true")
        factor = out["adjustment_factor"].astype(float)
        for col in PRICE_COLUMNS:
            out[col] = out[col].astype(float) * factor
        meta = {"method": "provided_adjustment_factor", "warning": None, "implausible_jumps": [], "declarations": declarations}
        return out.loc[:, list(OHLCV_COLUMNS)], meta

    if "adjusted_close" in out.columns:
        if has_adjustment_factor is not False:
            raise ValueError("adjusted price column requires has_adjustment_factor=false")
        ratio = out["adjusted_close"].astype(float) / out["close"].astype(float)
        for col in PRICE_COLUMNS:
            out[col] = out[col].astype(float) * ratio
        meta = {"method": "provided_adjusted_close", "warning": None, "implausible_jumps": [], "declarations": declarations}
        return out.loc[:, list(OHLCV_COLUMNS)], meta

    if "dividend" in out.columns or "split" in out.columns:
        adjusted = _adjust_from_dividends_splits(out)
        meta = {"method": "built_from_dividends_splits", "warning": None, "implausible_jumps": [], "declarations": declarations}
        return adjusted.loc[:, list(OHLCV_COLUMNS)], meta

    jumps = _find_implausible_jumps(out, max_log_return=max_unadjusted_log_return)
    meta = {
        "method": "raw_unadjusted",
        "warning": "No adjusted prices, dividends, or splits were provided; raw prices are used.",
        "implausible_jumps": jumps,
        "declarations": declarations,
    }
    return out.loc[:, list(OHLCV_COLUMNS)], meta


def _adjustment_declarations(
    out: pd.DataFrame,
    *,
    adjustment_convention: str | None,
    has_adjustment_factor: bool | None,
    dividend_tax_treatment: str | None,
) -> dict[str, Any]:
    adjusted_path = any(col in out.columns for col in ("adjusted_close", "adjustment_factor", "dividend", "split", "bonus", "conversion"))
    if not adjusted_path:
        return {
            "adjustment_convention": "none",
            "has_adjustment_factor": False,
            "adjustment_factor_semantics": None,
            "dividend_date_role": None,
            "dividend_tax_treatment": None,
        }
    if adjustment_convention not in {"forward", "backward", "none"}:
        raise ValueError("adjustment_convention is required and must be one of forward|backward|none")
    if "adjustment_factor" in out.columns and has_adjustment_factor is not True:
        raise ValueError("has_adjustment_factor must be true when an adjustment_factor column is mapped")
    if "adjusted_close" in out.columns and has_adjustment_factor is not False:
        raise ValueError("has_adjustment_factor must be false when adjusted price columns are mapped")
    if "dividend" in out.columns:
        if "dividend_ex_date" not in out.columns:
            raise ValueError("cash dividend data requires an explicit ex-dividend date column")
        if dividend_tax_treatment not in {"pre_tax", "post_tax"}:
            raise ValueError("dividend_tax_treatment must be declared as pre_tax or post_tax")
        out["dividend_ex_date"] = pd.to_datetime(out["dividend_ex_date"], utc=True, errors="raise")
    return {
        "adjustment_convention": adjustment_convention,
        "has_adjustment_factor": bool(has_adjustment_factor),
        "adjustment_factor_semantics": "raw_times_factor" if has_adjustment_factor else None,
        "dividend_date_role": "ex_date" if "dividend" in out.columns else None,
        "dividend_tax_treatment": dividend_tax_treatment if "dividend" in out.columns else None,
    }


def _adjust_from_dividends_splits(mapped: pd.DataFrame) -> pd.DataFrame:
    out = mapped.copy()
    out["dividend"] = out.get("dividend", 0.0)
    out["split"] = out.get("split", 1.0)
    parts = []
    for _, sub in out.groupby("symbol", sort=False):
        sub = sub.sort_values("timestamp").copy()
        factor = pd.Series(1.0, index=sub.index)
        cumulative = 1.0
        for idx in reversed(sub.index.tolist()):
            factor.loc[idx] = cumulative
            split = float(sub.loc[idx, "split"]) if pd.notna(sub.loc[idx, "split"]) else 1.0
            dividend = float(sub.loc[idx, "dividend"]) if pd.notna(sub.loc[idx, "dividend"]) else 0.0
            if split <= 0:
                raise ValueError("split ratio must be positive")
            if split != 1.0:
                cumulative *= 1.0 / split
            if dividend:
                close = float(sub.loc[idx, "close"])
                if dividend < 0 or dividend >= close:
                    raise ValueError("dividend must be non-negative and smaller than close")
                cumulative *= (close - dividend) / close
        for col in PRICE_COLUMNS:
            sub[col] = sub[col].astype(float) * factor
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def _find_implausible_jumps(ohlcv: pd.DataFrame, *, max_log_return: float) -> list[dict[str, Any]]:
    jumps: list[dict[str, Any]] = []
    for sym, sub in ohlcv.groupby("symbol", sort=False):
        ordered = sub.sort_values("timestamp")
        log_r = np.log(ordered["close"].astype(float)).diff().abs()
        for idx, value in log_r[log_r > max_log_return].items():
            jumps.append(
                {
                    "symbol": str(sym),
                    "timestamp": pd.Timestamp(ordered.loc[idx, "timestamp"]).isoformat(),
                    "abs_log_return": float(value),
                    "threshold": float(max_log_return),
                }
            )
    return jumps


def _validate_symbol_universe(ohlcv: pd.DataFrame, symbols: list[str]) -> None:
    actual = set(ohlcv["symbol"].astype(str))
    expected = set(symbols)
    if actual != expected:
        raise ValueError(f"local file symbols must match configured 3-5 symbol universe: expected {sorted(expected)}, got {sorted(actual)}")


def _alignment_metadata(tradable: pd.DataFrame) -> dict[str, Any]:
    gaps = []
    for ts in tradable.index:
        missing = tradable.columns[~tradable.loc[ts]].tolist()
        if missing:
            gaps.append({"timestamp": pd.Timestamp(ts).isoformat(), "missing_symbols": missing})
    return {
        "filled_gap_count": int(sum(len(item["missing_symbols"]) for item in gaps)),
        "gaps": gaps,
        "fill_method": "ffill_then_bfill_recorded",
    }


def _write_processed(df: pd.DataFrame, base_path: Path) -> Path:
    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
