"""Ingestor abstraction: raw file → bundle increment.

An ingestor reads one or more raw parquet files (produced by a fetcher),
merges them into a bundle's ``ohlcv.parquet``, and updates the manifest's
date range / row count / source chain / freshness.

Conservatism rules:
- New rows can extend a bundle's date range (forward in time) but must NOT
  contradict existing data on overlapping dates. Conflicts fail loudly.
- New symbols beyond the manifest's declared universe are rejected.
- ``provenance.jsonl`` gets one ``ingest`` line per merge call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

IngestStatus = Literal["ok", "no_op", "failed"]


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one ingest call.

    - ``rows_added``: rows that newly entered the bundle.
    - ``rows_skipped``: rows already present (same timestamp+symbol+values).
    - ``rows_conflicting``: rows that disagreed with existing bundle values
      on the same (timestamp, symbol). Non-zero ⇒ ``status="failed"``.
    """

    status: IngestStatus
    bundle: str
    source: str
    rows_added: int = 0
    rows_skipped: int = 0
    rows_conflicting: int = 0
    new_last_date: str | None = None
    error: str | None = None


class Ingestor(ABC):
    """Concrete ingestors subclass this."""

    source: str  # provenance label, matches the originating fetcher

    @abstractmethod
    def ingest_into_bundle(
        self,
        raw_paths: list[Path],
        *,
        bundle_root: Path,
    ) -> IngestResult:
        """Merge *raw_paths* into the bundle at *bundle_root*.

        Returns an :class:`IngestResult`. Never raises for per-row conflicts —
        report them via ``rows_conflicting`` instead, so the caller can decide
        whether to retry, repair, or surface to the user.
        """
        raise NotImplementedError
