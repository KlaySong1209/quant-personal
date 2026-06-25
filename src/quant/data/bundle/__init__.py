"""Bundle subsystem: stateful, on-disk datasets that own provenance and freshness.

A *bundle* is a directory under ``data/bundles/<name>/`` holding canonical OHLCV
(+ corporate actions + calendar) plus a ``manifest.json`` describing what's in
it and how fresh it is.

This package owns the schema (validated against
``data/bundles/manifest.schema.json``), low-level read/write
(:class:`BundleStore`), the multi-bundle index (:class:`BundleCatalog`), the
append-only audit log (:mod:`provenance`), and the one-time legacy migration
(:mod:`migrate`).

Higher layers (``quant.app``, the dashboard, fetchers) consume bundles through
:class:`BundleStore` / :class:`BundleCatalog` only — they MUST NOT touch the
bundle directory directly.
"""

from quant.data.bundle.manifest import (
    AdjustmentMeta,
    BundleManifest,
    CalendarMeta,
    DateRange,
    FreshnessMeta,
    SCHEMA_VERSION,
    SUPPORTED_MARKETS,
)

__all__ = [
    "AdjustmentMeta",
    "BundleManifest",
    "CalendarMeta",
    "DateRange",
    "FreshnessMeta",
    "SCHEMA_VERSION",
    "SUPPORTED_MARKETS",
]
