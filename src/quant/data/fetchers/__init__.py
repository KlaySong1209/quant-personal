"""Fetchers: pull external market data into local raw files.

Design (see [plan](robust-swinging-falcon.md) §"Fetcher 接口"):
- Every fetcher writes ``data/raw/<source>/<...>.parquet`` first;
- Ingestors (separate module) later read those raw files and merge into a bundle.
- This separation is the project's "诚实优先" principle applied to data:
  when a row in the bundle looks wrong, the raw file is still on disk for diff.

This module exposes only the **shape** — concrete fetchers live in sibling files.
"""

from quant.data.fetchers.base import (
    FetchError,
    FetchResult,
    FetchStatus,
    Fetcher,
)
from quant.data.fetchers.akshare_daily import AkshareFetcher, parse_akshare_bars

__all__ = [
    "AkshareFetcher",
    "FetchError",
    "FetchResult",
    "FetchStatus",
    "Fetcher",
    "parse_akshare_bars",
]
