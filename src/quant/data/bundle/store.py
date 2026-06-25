"""BundleStore: the single read/write API for one bundle directory.

A bundle on disk:

  data/bundles/<name>/
    manifest.json                # validated against manifest.schema.json
    ohlcv.parquet                # canonical OHLCV (timestamp, symbol, OHLCV[, source])
    corporate_actions.parquet    # optional, CorporateAction.to_dict() rows
    calendar.parquet             # optional, only when calendar.source=='file'
    provenance.jsonl             # append-only audit log

This class is the ONLY thing that touches files inside that directory.
Higher layers (``quant.app``, dashboard, fetchers) go through here.

Schema invariants enforced here (not at the schema layer):
- ``ohlcv.parquet`` symbols ⊆ ``manifest.symbols``
- ``ohlcv.parquet`` date range ⊆ ``manifest.date_range``
- ``timestamp`` column is tz-aware UTC
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant.data.bundle.freshness import judge
from quant.data.bundle.manifest import (
    AdjustmentMeta,
    BundleManifest,
    CalendarMeta,
    DateRange,
    FreshnessMeta,
    SCHEMA_VERSION,
)
from quant.data.bundle.provenance import ProvenanceLog, ProvenanceRecord
from quant.data.schema import OHLCV_COLUMNS, coerce_ohlcv
from quant.data.symbols import normalize_many

MANIFEST_FILENAME = "manifest.json"
OHLCV_FILENAME = "ohlcv.parquet"
CORPORATE_ACTIONS_FILENAME = "corporate_actions.parquet"
CALENDAR_FILENAME = "calendar.parquet"
PROVENANCE_FILENAME = "provenance.jsonl"


class BundleError(Exception):
    """Raised when a bundle is malformed or an operation would corrupt it."""


@dataclass(frozen=True)
class BundleLayout:
    """Resolved on-disk paths for a single bundle. Pure value object."""

    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILENAME

    @property
    def ohlcv_path(self) -> Path:
        return self.root / OHLCV_FILENAME

    @property
    def corporate_actions_path(self) -> Path:
        return self.root / CORPORATE_ACTIONS_FILENAME

    @property
    def calendar_path(self) -> Path:
        return self.root / CALENDAR_FILENAME

    @property
    def provenance_path(self) -> Path:
        return self.root / PROVENANCE_FILENAME


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BundleStore:
    """Read/write a single bundle directory."""

    def __init__(self, root: str | Path):
        self.layout = BundleLayout(Path(root))
        self.provenance = ProvenanceLog(self.layout.provenance_path)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self.layout.manifest_path.exists()

    def manifest(self) -> BundleManifest:
        if not self.exists():
            raise BundleError(f"bundle does not exist: {self.layout.root}")
        return BundleManifest.load(self.layout.manifest_path)

    def write_manifest(self, manifest: BundleManifest) -> Path:
        return manifest.save(self.layout.manifest_path)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def ohlcv(self) -> pd.DataFrame:
        """Return the bundle's OHLCV frame (canonical schema)."""
        path = self.layout.ohlcv_path
        if not path.exists():
            return pd.DataFrame(columns=list(OHLCV_COLUMNS))
        df = pd.read_parquet(path)
        return coerce_ohlcv(df)

    def write_ohlcv(self, ohlcv: pd.DataFrame, *, manifest: BundleManifest) -> Path:
        """Replace the bundle's OHLCV file. Manifest invariants are enforced."""
        df = coerce_ohlcv(ohlcv)
        _validate_ohlcv_against_manifest(df, manifest)
        self.layout.root.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.layout.ohlcv_path, index=False)
        return self.layout.ohlcv_path

    # ------------------------------------------------------------------
    # Corporate actions (optional)
    # ------------------------------------------------------------------

    def corporate_actions(self) -> pd.DataFrame:
        path = self.layout.corporate_actions_path
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write_corporate_actions(self, df: pd.DataFrame) -> Path:
        self.layout.root.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.layout.corporate_actions_path, index=False)
        return self.layout.corporate_actions_path

    # ------------------------------------------------------------------
    # Provenance helpers (thin pass-through for callers that already hold a store)
    # ------------------------------------------------------------------

    def record(self, record: ProvenanceRecord) -> None:
        self.provenance.append(record)

    # ------------------------------------------------------------------
    # Create from scratch
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        name: str,
        symbols: Iterable[str],
        ohlcv: pd.DataFrame,
        source: str,
        adjustment: AdjustmentMeta,
        calendar: CalendarMeta,
        corporate_actions: pd.DataFrame | None = None,
    ) -> "BundleStore":
        """Materialise a brand-new bundle at *root*.

        - Symbols are normalized to canonical form before persisting.
        - OHLCV is coerced and validated against the resulting manifest.
        - The store appends a ``create`` provenance record.
        """
        store = cls(root)
        if store.exists():
            raise BundleError(f"bundle already exists: {store.layout.root}")
        canonical_symbols = normalize_many(list(symbols))

        df = coerce_ohlcv(ohlcv)
        # Translate any non-canonical symbols inside the dataframe before storing.
        df = _canonicalize_symbol_column(df)

        actual_symbols = sorted(df["symbol"].astype(str).unique())
        unexpected = sorted(set(actual_symbols) - set(canonical_symbols))
        if unexpected:
            raise BundleError(
                f"OHLCV contains symbols not in the declared universe: {unexpected}"
            )

        if df.empty:
            first_iso = last_iso = pd.Timestamp.now(tz="UTC").date().isoformat()
            row_count = 0
        else:
            first_iso = df["timestamp"].min().date().isoformat()
            last_iso = df["timestamp"].max().date().isoformat()
            row_count = int(len(df))

        now = _utcnow_iso()
        manifest = BundleManifest(
            name=name,
            schema_version=SCHEMA_VERSION,
            market="a_share_cn",
            created_at=now,
            updated_at=now,
            symbols=canonical_symbols,
            date_range=DateRange(first=first_iso, last=last_iso),
            source_chain=[source],
            adjustment=adjustment,
            row_count=row_count,
            calendar=calendar,
            freshness=FreshnessMeta(
                expected_through=last_iso,
                actual_through=last_iso,
                status="fresh" if row_count > 0 else "no_data",
            ),
        )
        # Refine freshness against the calendar fallback (synthetic bdays for now).
        manifest = manifest.model_copy(update={"freshness": judge(manifest)})

        store.layout.root.mkdir(parents=True, exist_ok=True)
        store.write_ohlcv(df, manifest=manifest)
        if corporate_actions is not None and not corporate_actions.empty:
            store.write_corporate_actions(corporate_actions)
        store.write_manifest(manifest)
        store.record(
            ProvenanceRecord(
                op="create",
                status="ok",
                source=source,
                bundle=name,
                symbols=canonical_symbols,
                rows=row_count,
            )
        )
        return store

    # ------------------------------------------------------------------
    # Freshness recompute (cheap, no I/O besides reading the manifest)
    # ------------------------------------------------------------------

    def refresh_freshness(
        self,
        *,
        as_of: pd.Timestamp | None = None,
    ) -> BundleManifest:
        """Recompute freshness against *as_of* and persist the updated manifest."""
        m = self.manifest()
        new_freshness = judge(m, as_of=as_of)
        if new_freshness == m.freshness:
            return m
        updated = m.model_copy(update={
            "freshness": new_freshness,
            "updated_at": _utcnow_iso(),
        })
        self.write_manifest(updated)
        return updated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonicalize_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the ``symbol`` column to canonical form (idempotent)."""
    from quant.data.symbols import normalize

    if df.empty:
        return df
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).map(normalize)
    return out


def _validate_ohlcv_against_manifest(df: pd.DataFrame, manifest: BundleManifest) -> None:
    if df.empty:
        return
    actual_syms = set(df["symbol"].astype(str).unique())
    declared = set(manifest.symbols)
    extra = actual_syms - declared
    if extra:
        raise BundleError(
            f"ohlcv contains symbols outside the manifest universe: {sorted(extra)}"
        )
    first = df["timestamp"].min().date().isoformat()
    last = df["timestamp"].max().date().isoformat()
    if first < manifest.date_range.first or last > manifest.date_range.last:
        raise BundleError(
            f"ohlcv date range [{first}, {last}] exceeds manifest "
            f"[{manifest.date_range.first}, {manifest.date_range.last}]"
        )
    if not np.isfinite(df[["open", "high", "low", "close"]].to_numpy(dtype="float64")).all():
        raise BundleError("ohlcv contains non-finite price values")
