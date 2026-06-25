"""Fetcher abstraction.

A Fetcher pulls OHLCV from one external source for a given list of canonical
symbols + date range. It writes raw parquet files to disk and returns a
``FetchResult`` describing what worked, what didn't, and where the raw bytes
landed. It does NOT touch bundles — that's the Ingestor's job in
:mod:`quant.data.ingestors`.

Failure model (fail-loud, never silent):
- A symbol that completely failed shows up in ``symbols_failed`` with a message.
- A symbol that returned zero rows is also a failure (the upstream said "no
  data for you" — that's information, not silence).
- Partial success is OK: ``status="partial"`` plus a non-empty failure map.
- The caller decides whether to retry, fall back, or surface the error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

FetchStatus = Literal["ok", "partial", "failed"]


class FetchError(Exception):
    """Raised by fetchers when the entire batch could not be attempted
    (e.g. no TDX server reachable). Per-symbol failures are reported via
    :class:`FetchResult`, not raised."""


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one ``Fetcher.fetch_daily_ohlcv`` call.

    Conventions:
      - ``raw_paths`` is empty when ``status == "failed"``.
      - ``symbols_failed`` keys are canonical symbols (``SH600519`` ...).
      - ``route_note`` is an optional one-liner used for provenance (e.g.
        "tdx via 119.147.212.81:7709" or "bestip").
    """

    source: str
    status: FetchStatus
    raw_paths: list[Path]
    symbols_ok: list[str]
    symbols_failed: dict[str, str] = field(default_factory=dict)
    rows_total: int = 0
    route_note: str | None = None
    error: str | None = None  # set when status == "failed"

    @classmethod
    def from_per_symbol(
        cls,
        *,
        source: str,
        raw_paths: list[Path],
        ok: list[str],
        failed: dict[str, str],
        rows_total: int,
        route_note: str | None = None,
    ) -> "FetchResult":
        if not ok and failed:
            status: FetchStatus = "failed"
            error: str | None = "; ".join(f"{s}: {m}" for s, m in failed.items())
        elif failed:
            status = "partial"
            error = None
        else:
            status = "ok"
            error = None
        return cls(
            source=source,
            status=status,
            raw_paths=raw_paths,
            symbols_ok=ok,
            symbols_failed=dict(failed),
            rows_total=rows_total,
            route_note=route_note,
            error=error,
        )


class Fetcher(ABC):
    """Concrete fetchers subclass this. ``source`` is the provenance label."""

    source: str

    @abstractmethod
    def fetch_daily_ohlcv(
        self,
        symbols: list[str],
        *,
        raw_dir: Path,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> FetchResult:
        """Pull daily OHLCV for *symbols* into ``raw_dir``.

        Args:
            symbols: Canonical symbols (``SH600519`` / ``SZ000001`` / ...).
            raw_dir: Where this call should drop its raw parquet file(s).
                The fetcher creates the directory if needed.
            start, end: Optional UTC-naive or UTC-aware timestamps that bound
                what to fetch. Concrete fetchers may translate to ``offset``
                or ``start_date`` semantics depending on the upstream API.
                ``None`` means "use a sensible default" (e.g. last N days).

        Raises:
            FetchError: When the entire batch could not even be attempted
                (transport down, upstream auth failure, no IP route).
        """
        raise NotImplementedError
