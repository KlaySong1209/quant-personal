"""One-time migration: legacy processed files → default bundle.

When a user has been on the old layout (``data/processed/local_daily_ohlcv.parquet``
+ ``local_daily_metadata.json``), this module wraps those files into a
``data/bundles/default/`` bundle so the rest of the new bundle code can run
unchanged.

Idempotent. Safe to call every startup — if the default bundle already exists
(and points at a manifest), this is a no-op. If only the legacy file exists,
we materialise the bundle. If neither exists, we do nothing (system will
behave exactly like before for users who never ingested anything).

The legacy file is NOT deleted — users may still rely on it (e.g.
``scripts/run_paper_session.py --data data/processed/...``). See the
compatibility table in [plan](robust-swinging-falcon.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from quant.data.bundle import AdjustmentMeta, CalendarMeta
from quant.data.bundle.catalog import BundleCatalog
from quant.data.bundle.store import BundleStore
from quant.data.local import read_processed_ohlcv
from quant.data.symbols import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_BUNDLES_ROOT = PROJECT_ROOT / "data" / "bundles"
DEFAULT_BUNDLE_NAME = "default"


@dataclass(frozen=True)
class MigrationResult:
    status: Literal["created", "already_exists", "no_legacy", "skipped"]
    bundle_name: str | None = None
    bundle_path: Path | None = None
    legacy_path: Path | None = None
    rows: int = 0
    error: str | None = None


def _detect_adjustment(metadata: dict) -> AdjustmentMeta:
    method = (metadata.get("adjustment") or {}).get("method", "raw_unadjusted")
    valid_methods = {
        "provided_adjusted_close",
        "provided_adjustment_factor",
        "built_from_dividends_splits",
        "raw_unadjusted",
    }
    if method not in valid_methods:
        method = "raw_unadjusted"
    declarations = (metadata.get("adjustment") or {}).get("declarations") or {}
    convention = declarations.get("adjustment_convention") or "none"
    if convention not in {"forward", "backward", "none"}:
        convention = "none"
    return AdjustmentMeta(convention=convention, method=method)


def _detect_calendar(metadata: dict) -> CalendarMeta:
    cal = metadata.get("calendar") or {}
    source = cal.get("source") or "synthetic"
    if source not in {"synthetic", "file"}:
        source = "synthetic"
    exchange = cal.get("exchange") or "SYNTH"
    return CalendarMeta(source=source, exchange=exchange)


def migrate_legacy_processed(
    *,
    bundles_root: Path = DEFAULT_BUNDLES_ROOT,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    bundle_name: str = DEFAULT_BUNDLE_NAME,
) -> MigrationResult:
    """Materialise a default bundle from legacy ``data/processed/`` files.

    Returns a :class:`MigrationResult` describing what happened.
    """
    catalog = BundleCatalog.load(bundles_root)
    if catalog.get(bundle_name) is not None:
        return MigrationResult(
            status="already_exists",
            bundle_name=bundle_name,
            bundle_path=bundles_root / bundle_name,
        )

    # Detect legacy files.
    legacy_parquet = processed_dir / "local_daily_ohlcv.parquet"
    legacy_csv = processed_dir / "local_daily_ohlcv.csv"
    metadata_path = processed_dir / "local_daily_metadata.json"
    if legacy_parquet.exists():
        legacy_path = legacy_parquet
    elif legacy_csv.exists():
        legacy_path = legacy_csv
    else:
        return MigrationResult(status="no_legacy")

    try:
        # ``read_processed_ohlcv`` uses default dtype inference; for CSV legacy
        # files a numeric-only ``symbol`` column would become int and lose
        # leading zeros (``000001`` → ``1``). Coerce CSV explicitly so
        # leading-zero A-share codes survive the round-trip.
        if legacy_path.suffix.lower() == ".csv":
            ohlcv = pd.read_csv(legacy_path, dtype={"symbol": str})
        else:
            ohlcv = read_processed_ohlcv(legacy_path)
    except Exception as exc:
        return MigrationResult(
            status="skipped",
            legacy_path=legacy_path,
            error=f"cannot read legacy file: {exc}",
        )

    metadata: dict = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    # Canonicalize symbols in the OHLCV frame.
    # Force str dtype first — CSV roundtrips lose dtype and bare codes like
    # ``000001`` come back as int. Parquet preserves str but coercing here is
    # cheap and uniform.
    ohlcv = ohlcv.copy()
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).map(normalize)
    declared_symbols = sorted(ohlcv["symbol"].unique())
    if not declared_symbols:
        return MigrationResult(
            status="skipped",
            legacy_path=legacy_path,
            error="legacy file has no symbols",
        )

    store = BundleStore.create(
        bundles_root / bundle_name,
        name=bundle_name,
        symbols=declared_symbols,
        ohlcv=ohlcv,
        source="legacy-processed",
        adjustment=_detect_adjustment(metadata),
        calendar=_detect_calendar(metadata),
    )
    catalog.register(name=bundle_name)
    return MigrationResult(
        status="created",
        bundle_name=bundle_name,
        bundle_path=store.layout.root,
        legacy_path=legacy_path,
        rows=int(len(ohlcv)),
    )


def auto_migrate_if_needed(
    *,
    bundles_root: Path = DEFAULT_BUNDLES_ROOT,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> MigrationResult:
    """Idempotent guard: only runs migration when it would actually do work.

    Designed to be called once at app startup from ``quant.app``.
    """
    catalog = BundleCatalog.load(bundles_root)
    if catalog.get(DEFAULT_BUNDLE_NAME) is not None:
        return MigrationResult(
            status="already_exists",
            bundle_name=DEFAULT_BUNDLE_NAME,
            bundle_path=bundles_root / DEFAULT_BUNDLE_NAME,
        )
    return migrate_legacy_processed(bundles_root=bundles_root, processed_dir=processed_dir)
