"""Ingestors: take fetcher-produced raw files and merge them into a bundle.

Why fetcher + ingestor instead of one step:
- Fetcher cares about transport (mootdx TCP, HTTP, …) and produces raw files.
- Ingestor cares about schema (canonical symbols, dedup, manifest update).
- The two-step keeps raw files on disk for debugging when bundle data looks
  wrong — diff the raw file vs. the bundle's ``ohlcv.parquet``.
- It also lets one ingestor accept raw files from several fetchers
  (mootdx, tencent, ...) — they all produce the same canonical schema.
"""

from quant.data.ingestors.base import IngestResult, Ingestor
from quant.data.ingestors.mootdx_ingestor import MootdxIngestor

__all__ = ["IngestResult", "Ingestor", "MootdxIngestor"]
