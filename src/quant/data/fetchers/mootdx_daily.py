"""MootdxFetcher: pull daily OHLCV from Tongdaxin via the mootdx library.

Split into two layers:
- :func:`parse_mootdx_bars` — pure function from a raw mootdx DataFrame to
  the canonical OHLCV schema. Fixture-testable, no network.
- :class:`MootdxFetcher` — transport. Drives the client per symbol, writes
  one parquet per ``fetch_daily_ohlcv`` call, and reports partial failures.

Why "unadjusted only" right now (see [plan](robust-swinging-falcon.md) §阶段 3):
- mootdx 0.11.x ``adjust='qfq'`` path is broken against pandas >=2.2
  (uses removed ``fillna(method=...)``).
- More importantly: this project's existing
  :class:`quant.data.adjust.corporate_actions.CorporateAction` schema is the
  source of truth for adjustments. Letting the fetcher silently apply qfq
  would hide what corporate actions were used — which violates the project's
  "诚实优先" principle. A future ``MootdxCorporateActionFetcher`` will pull
  the xdxr endpoint into the bundle's ``corporate_actions.parquet`` instead.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from quant.data.fetchers.base import FetchError, FetchResult, Fetcher
from quant.data.fetchers.tdx_client import TdxClientError, open_client
from quant.data.schema import OHLCV_COLUMNS
from quant.data.symbols import SymbolError, parse_symbol, to_mootdx

SOURCE = "mootdx"
DEFAULT_OFFSET = 60  # ~3 months of trading days


class _TdxLike(Protocol):
    """Minimal duck-type of ``mootdx.quotes.Quotes`` we depend on. Lets unit
    tests inject a stub without importing mootdx."""

    def bars(self, *, symbol: str, frequency: int, offset: int) -> pd.DataFrame: ...


def parse_mootdx_bars(raw: pd.DataFrame, *, canonical_symbol: str) -> pd.DataFrame:
    """Convert a raw mootdx ``bars`` DataFrame into the canonical OHLCV schema.

    mootdx returns columns like::

        open, close, high, low, vol, amount, year, month, day, hour, minute,
        datetime, volume

    where ``volume`` and ``vol`` are duplicates (both are 手 = lots). The
    DataFrame index is a ``DatetimeIndex`` set from ``datetime`` (HH:MM 15:00
    close time, no timezone).

    The canonical schema is ``(timestamp UTC, symbol, open, high, low, close, volume)``.
    We normalise the close-time to midnight UTC so it lines up with synthetic
    business-day calendars used elsewhere.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    # mootdx indexes by datetime; bring it back as a column so we control casts.
    df = raw.reset_index(drop=False) if "datetime" not in raw.columns else raw.copy()
    if "datetime" not in df.columns:
        raise ValueError("mootdx bars frame is missing 'datetime' column")
    if "volume" not in df.columns and "vol" not in df.columns:
        raise ValueError("mootdx bars frame is missing 'vol'/'volume' column")

    ts = pd.to_datetime(df["datetime"], errors="raise")
    # mootdx datetimes are naive Asia/Shanghai 15:00 close. We anchor on the
    # trading date (UTC midnight) — matches the rest of the project.
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Shanghai")
    ts = ts.dt.tz_convert("UTC").dt.normalize()

    volume = df["volume"] if "volume" in df.columns else df["vol"]

    # Build via a tz-aware Series (passing .values to a DataFrame strips tz).
    out = pd.DataFrame({
        "timestamp": ts.reset_index(drop=True),
        "symbol": canonical_symbol,
        "open": pd.to_numeric(df["open"], errors="raise").astype(float).reset_index(drop=True),
        "high": pd.to_numeric(df["high"], errors="raise").astype(float).reset_index(drop=True),
        "low": pd.to_numeric(df["low"], errors="raise").astype(float).reset_index(drop=True),
        "close": pd.to_numeric(df["close"], errors="raise").astype(float).reset_index(drop=True),
        "volume": pd.to_numeric(volume, errors="raise").astype(float).reset_index(drop=True),
    })
    # Dedup just in case mootdx returns dupes around server hand-off.
    out = out.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _utc_today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).normalize()


def _filename_for(symbols: list[str]) -> str:
    """Stable per-batch filename: ``YYYYMMDDTHHMMSSZ-<count>syms.parquet``.

    We do NOT key by symbol — one fetch call writes one parquet so the
    raw file's lineage matches the FetchResult.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{len(symbols)}syms.parquet"


class MootdxFetcher(Fetcher):
    """Pull unadjusted daily OHLCV via mootdx (TDX TCP).

    Construct with no args for the default behavior (lazy-open the client).
    Tests inject a stub via ``client_factory``.
    """

    source = SOURCE

    def __init__(
        self,
        *,
        client_factory: Any = None,
        default_offset: int = DEFAULT_OFFSET,
    ):
        self._client_factory = client_factory  # callable returning (client, route)
        self._default_offset = int(default_offset)

    def fetch_daily_ohlcv(
        self,
        symbols: list[str],
        *,
        raw_dir: Path,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> FetchResult:
        if not symbols:
            return FetchResult(
                source=SOURCE, status="ok", raw_paths=[],
                symbols_ok=[], rows_total=0,
            )

        # Translate canonical → mootdx pre-flight so we fail fast on bad input.
        try:
            tdx_codes = {s: to_mootdx(s) for s in symbols}
        except SymbolError as exc:
            raise FetchError(f"unsupported symbol for mootdx: {exc}") from exc

        # Open client (real or injected stub).
        if self._client_factory is not None:
            client, route = self._client_factory()
        else:
            try:
                client, route = open_client()
            except TdxClientError as exc:
                raise FetchError(str(exc)) from exc

        offset = self._infer_offset(start, end)
        rows_total = 0
        ok: list[str] = []
        failed: dict[str, str] = {}
        frames: list[pd.DataFrame] = []

        for canonical, (market, raw_code) in tdx_codes.items():
            try:
                raw = client.bars(symbol=raw_code, frequency=9, offset=offset)
            except Exception as exc:  # noqa: BLE001 — mootdx raises many shapes
                failed[canonical] = f"{type(exc).__name__}: {exc}"
                continue

            try:
                parsed = parse_mootdx_bars(raw, canonical_symbol=canonical)
            except Exception as exc:  # noqa: BLE001
                failed[canonical] = f"parse: {type(exc).__name__}: {exc}"
                continue

            if parsed.empty:
                failed[canonical] = "empty response"
                continue

            # Apply explicit date bounds if given.
            if start is not None:
                parsed = parsed[parsed["timestamp"] >= pd.Timestamp(start).tz_convert("UTC")]
            if end is not None:
                parsed = parsed[parsed["timestamp"] <= pd.Timestamp(end).tz_convert("UTC")]
            if parsed.empty:
                failed[canonical] = "no rows in requested range"
                continue

            frames.append(parsed)
            ok.append(canonical)
            rows_total += len(parsed)

        raw_paths: list[Path] = []
        if frames:
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / _filename_for(symbols)
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(raw_path, index=False)
            raw_paths.append(raw_path)

        return FetchResult.from_per_symbol(
            source=SOURCE,
            raw_paths=raw_paths,
            ok=ok,
            failed=failed,
            rows_total=rows_total,
            route_note=f"tdx route={route.method}"
                       + (f" via {route.server}" if route.server else ""),
        )

    # ------------------------------------------------------------------

    def _infer_offset(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> int:
        """Translate (start, end) into mootdx's ``offset`` (# of bars from now).

        mootdx's daily bars API only supports a "last N bars" offset; there's
        no explicit start_date. We over-fetch with a generous margin when
        start is given, and filter to range after parsing.
        """
        if start is None and end is None:
            return self._default_offset
        ref_end = pd.Timestamp(end) if end is not None else _utc_today()
        if start is None:
            return self._default_offset
        ref_start = pd.Timestamp(start)
        if ref_start.tzinfo is None:
            ref_start = ref_start.tz_localize("UTC")
        if ref_end.tzinfo is None:
            ref_end = ref_end.tz_localize("UTC")
        days = max(1, int((ref_end - ref_start).days))
        # ~5/7 of calendar days are trading days; pad 20 bars for holidays.
        return max(self._default_offset, days * 5 // 7 + 20)
