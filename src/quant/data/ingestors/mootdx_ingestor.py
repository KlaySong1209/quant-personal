"""MootdxIngestor: merge MootdxFetcher raw parquet files into a bundle.

This is also generic enough to consume any fetcher whose raw files conform
to the canonical OHLCV schema (which is the goal — only the ``source`` label
differs). For now it ships as ``MootdxIngestor`` for clarity; rename when a
second fetcher appears.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant.data.bundle.freshness import judge
from quant.data.bundle.manifest import BundleManifest, DateRange
from quant.data.bundle.provenance import ProvenanceRecord
from quant.data.bundle.store import BundleStore
from quant.data.ingestors.base import IngestResult, Ingestor
from quant.data.schema import OHLCV_COLUMNS, coerce_ohlcv

# Float comparison tolerance for "same OHLCV row" detection. Bigger than
# float-eps because mootdx's prices are 2-decimal and tencent/eastmoney can
# round differently.
_VALUE_TOL = 1e-4


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MootdxIngestor(Ingestor):
    """Merge canonical-schema raw parquet into a bundle.

    The ``source`` label is what shows up in the bundle's ``source_chain``
    and in provenance lines. Pass a different label when this same class
    is reused for another fetcher.
    """

    def __init__(self, *, source: str = "mootdx"):
        self.source = source

    def ingest_into_bundle(
        self,
        raw_paths: list[Path],
        *,
        bundle_root: Path,
    ) -> IngestResult:
        store = BundleStore(bundle_root)
        if not store.exists():
            return IngestResult(
                status="failed",
                bundle=bundle_root.name,
                source=self.source,
                error=f"bundle does not exist: {bundle_root}",
            )
        if not raw_paths:
            return IngestResult(
                status="no_op",
                bundle=bundle_root.name,
                source=self.source,
            )

        manifest = store.manifest()
        try:
            incoming = _load_raw(raw_paths)
        except Exception as exc:  # noqa: BLE001
            store.record(ProvenanceRecord(
                op="ingest", status="failed",
                source=self.source, bundle=manifest.name,
                error=f"raw read: {type(exc).__name__}: {exc}",
            ))
            return IngestResult(
                status="failed", bundle=manifest.name, source=self.source,
                error=f"raw read: {type(exc).__name__}: {exc}",
            )

        if incoming.empty:
            store.record(ProvenanceRecord(
                op="ingest", status="ok",
                source=self.source, bundle=manifest.name, rows=0,
                details={"note": "no rows in raw"},
            ))
            return IngestResult(
                status="no_op", bundle=manifest.name, source=self.source,
            )

        # Reject rows for symbols outside the declared universe (fail-loud).
        outside = sorted(set(incoming["symbol"]) - set(manifest.symbols))
        if outside:
            err = f"raw contains symbols outside bundle universe: {outside}"
            store.record(ProvenanceRecord(
                op="ingest", status="failed",
                source=self.source, bundle=manifest.name, error=err,
            ))
            return IngestResult(
                status="failed", bundle=manifest.name, source=self.source,
                error=err,
            )

        existing = store.ohlcv()
        merged, added, skipped, conflicting = _merge(existing, incoming)
        if conflicting > 0:
            err = (
                f"{conflicting} row(s) disagree with bundle on (timestamp, symbol); "
                "ingest aborted to preserve existing data. See raw parquet for diff."
            )
            store.record(ProvenanceRecord(
                op="ingest", status="failed",
                source=self.source, bundle=manifest.name,
                rows=conflicting, error=err,
            ))
            return IngestResult(
                status="failed", bundle=manifest.name, source=self.source,
                rows_conflicting=conflicting, error=err,
            )

        # Build updated manifest before writing OHLCV (so write_ohlcv can
        # validate against the new date_range).
        first_iso = merged["timestamp"].min().date().isoformat()
        last_iso = merged["timestamp"].max().date().isoformat()
        new_chain = list(manifest.source_chain)
        if self.source not in new_chain:
            new_chain.append(self.source)
        new_manifest = manifest.model_copy(update={
            "updated_at": _utcnow_iso(),
            "date_range": DateRange(first=first_iso, last=last_iso),
            "source_chain": new_chain,
            "row_count": int(len(merged)),
        })
        new_manifest = new_manifest.model_copy(update={"freshness": judge(new_manifest)})

        store.write_ohlcv(merged, manifest=new_manifest)
        store.write_manifest(new_manifest)

        store.record(ProvenanceRecord(
            op="ingest",
            status="ok",
            source=self.source,
            bundle=manifest.name,
            rows=added,
            details={
                "raw_paths": [str(p) for p in raw_paths],
                "rows_skipped": skipped,
                "new_last_date": last_iso,
            },
        ))
        return IngestResult(
            status="ok",
            bundle=manifest.name,
            source=self.source,
            rows_added=added,
            rows_skipped=skipped,
            new_last_date=last_iso,
        )


# ---------------------------------------------------------------------------
# Pure helpers (testable without disk)
# ---------------------------------------------------------------------------


def _load_raw(raw_paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for p in raw_paths:
        df = pd.read_parquet(p)
        frames.append(coerce_ohlcv(df))
    if not frames:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    # Within-raw dedup: if mootdx returned the same row twice, take last.
    out = out.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return out


def _merge(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """Return ``(merged_df, rows_added, rows_skipped, rows_conflicting)``.

    Logic:
      - Index both by (timestamp, symbol) as a MultiIndex.
      - For overlapping keys: compare OHLCV values within ``_VALUE_TOL``.
        - Equal → counted as skipped.
        - Unequal → counted as conflicting (caller must abort).
      - For incoming-only keys: counted as added.
      - Volume disagreement is treated like other columns (TDX sometimes
        republishes a row with slightly different volume; we still consider
        that a conflict — fail loud, don't paper over).
    """
    if existing.empty:
        return incoming.copy(), len(incoming), 0, 0

    e_idx = existing.set_index(["timestamp", "symbol"])
    i_idx = incoming.set_index(["timestamp", "symbol"])

    overlap_keys = i_idx.index.intersection(e_idx.index)
    new_keys = i_idx.index.difference(e_idx.index)

    rows_conflicting = 0
    rows_skipped = 0
    if len(overlap_keys) > 0:
        e_over = e_idx.loc[overlap_keys, ["open", "high", "low", "close", "volume"]]
        i_over = i_idx.loc[overlap_keys, ["open", "high", "low", "close", "volume"]]
        diff = (e_over.to_numpy(dtype="float64") - i_over.to_numpy(dtype="float64"))
        per_row_equal = np.all(np.abs(diff) <= _VALUE_TOL, axis=1)
        rows_skipped = int(per_row_equal.sum())
        rows_conflicting = int((~per_row_equal).sum())

    if len(new_keys) > 0:
        additions = i_idx.loc[new_keys].reset_index()
        merged = pd.concat([existing, additions], ignore_index=True)
    else:
        merged = existing.copy()
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return merged, int(len(new_keys)), rows_skipped, rows_conflicting


def _ohlcv_equal(a: pd.Series, b: pd.Series, tol: float = _VALUE_TOL) -> bool:
    """Retained as a public helper for unit tests."""
    for col in ("open", "high", "low", "close", "volume"):
        av = float(a[col])
        bv = float(b[col])
        if not (np.isfinite(av) and np.isfinite(bv)):
            return False
        if abs(av - bv) > tol:
            return False
    return True
