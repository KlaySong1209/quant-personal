"""AkShare daily OHLCV fetcher with source fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quant.data.fetchers.base import FetchError, FetchResult, Fetcher
from quant.data.schema import OHLCV_COLUMNS
from quant.data.symbols import Exchange, SymbolError, normalize, parse_symbol, to_tencent

DataSource = Literal["eastmoney", "sina", "tencent"]
SOURCE = "akshare"

_EASTMONEY_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


class AkshareFetcher(Fetcher):
    """Pull A-share daily bars through optional AkShare APIs.

    The fetcher is deliberately separate from the bundle writer. It writes one
    canonical raw parquet file and returns a FetchResult so the existing bundle
    ingestion layer can decide how to persist provenance.
    """

    source = SOURCE

    def __init__(
        self,
        *,
        datasource: str = "auto",
        akshare_module: Any = None,
    ):
        if datasource not in {"auto", "eastmoney", "sina", "tencent"}:
            raise ValueError("datasource must be one of 'auto', 'eastmoney', 'sina', or 'tencent'")
        self.datasource = datasource
        self._akshare = akshare_module

    def fetch_daily_ohlcv(
        self,
        symbols: list[str],
        *,
        raw_dir: Path,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> FetchResult:
        canonical_symbols = [normalize(sym) for sym in symbols]
        if not canonical_symbols:
            return FetchResult(source=SOURCE, status="ok", raw_paths=[], symbols_ok=[], rows_total=0)

        ak = self._load()
        failures_by_source: dict[str, dict[str, str]] = {}
        for datasource in self._source_order():
            result = self._fetch_with_source(
                ak,
                datasource,
                canonical_symbols,
                raw_dir=raw_dir,
                start=start,
                end=end,
            )
            if result.symbols_ok:
                return result
            failures_by_source[datasource] = result.symbols_failed

        raise FetchError(f"akshare returned no data from any source: {failures_by_source}")

    def _source_order(self) -> list[DataSource]:
        if self.datasource == "auto":
            return ["sina", "tencent", "eastmoney"]
        return [self.datasource]  # type: ignore[list-item]

    def _load(self):
        if self._akshare is not None:
            return self._akshare
        try:
            import akshare  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("akshare is not installed. Install it with: pip install akshare") from exc
        return akshare

    def _fetch_with_source(
        self,
        ak,
        datasource: DataSource,
        symbols: list[str],
        *,
        raw_dir: Path,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> FetchResult:
        start_str = _fmt_date(start)
        end_str = _fmt_date(end)
        ok: list[str] = []
        failed: dict[str, str] = {}
        frames: list[pd.DataFrame] = []

        for sym in symbols:
            try:
                raw = self._fetch_one(ak, datasource, sym, start_str, end_str)
                parsed = parse_akshare_bars(raw, canonical_symbol=sym)
            except Exception as exc:  # noqa: BLE001
                failed[sym] = f"{type(exc).__name__}: {exc}"
                continue
            if parsed.empty:
                failed[sym] = "empty response"
                continue
            frames.append(parsed)
            ok.append(sym)

        raw_paths: list[Path] = []
        rows_total = int(sum(len(frame) for frame in frames))
        if frames:
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / _filename_for(symbols, datasource)
            pd.concat(frames, ignore_index=True).to_parquet(raw_path, index=False)
            raw_paths.append(raw_path)

        return FetchResult.from_per_symbol(
            source=f"{SOURCE}/{datasource}",
            raw_paths=raw_paths,
            ok=ok,
            failed=failed,
            rows_total=rows_total,
            route_note=f"akshare datasource={datasource}",
        )

    def _fetch_one(self, ak, datasource: DataSource, symbol: str, start: str, end: str) -> pd.DataFrame:
        exchange, raw = parse_symbol(symbol)
        if exchange == Exchange.SYNTH:
            raise SymbolError(f"akshare does not support synthetic symbol {symbol!r}")
        if datasource == "eastmoney":
            return ak.stock_zh_a_hist(
                symbol=raw,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="",
            )
        if datasource == "sina":
            return ak.stock_zh_a_daily(
                symbol=to_tencent(symbol),
                start_date=start,
                end_date=end,
                adjust="",
            )
        return ak.stock_zh_a_hist_tx(
            symbol=to_tencent(symbol),
            start_date=start,
            end_date=end,
            adjust="",
        )


def parse_akshare_bars(raw: pd.DataFrame, *, canonical_symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    df = raw.rename(columns=_EASTMONEY_COLUMN_MAP).copy()
    if "date" not in df.columns and "日期" in df.columns:
        df = df.rename(columns={"日期": "date"})
    missing = [col for col in ("date", "open", "high", "low", "close") if col not in df.columns]
    if missing:
        raise ValueError(f"akshare frame for {canonical_symbol!r} missing columns {missing}")

    ts = pd.to_datetime(df["date"], errors="raise")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Shanghai")
    ts = ts.dt.tz_convert("UTC").dt.normalize()
    volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "timestamp": ts.reset_index(drop=True),
            "symbol": canonical_symbol,
            "open": pd.to_numeric(df["open"], errors="raise").astype(float).reset_index(drop=True),
            "high": pd.to_numeric(df["high"], errors="raise").astype(float).reset_index(drop=True),
            "low": pd.to_numeric(df["low"], errors="raise").astype(float).reset_index(drop=True),
            "close": pd.to_numeric(df["close"], errors="raise").astype(float).reset_index(drop=True),
            "volume": pd.to_numeric(volume, errors="coerce").fillna(0.0).astype(float).reset_index(drop=True),
        }
    )
    return out.drop_duplicates(subset=["timestamp", "symbol"], keep="last").sort_values("timestamp").reset_index(drop=True)


def _fmt_date(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return ""
    return pd.Timestamp(ts).strftime("%Y%m%d")


def _filename_for(symbols: list[str], datasource: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-akshare-{datasource}-{len(symbols)}syms.parquet"
