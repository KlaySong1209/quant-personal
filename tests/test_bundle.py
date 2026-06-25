"""Stage 1 tests: bundle abstraction (symbols, manifest, store, catalog,
freshness, provenance, migration).

All tests are in-process — no network, no real disk outside ``tempfile``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.data.bundle import (
    AdjustmentMeta,
    BundleManifest,
    CalendarMeta,
    DateRange,
    FreshnessMeta,
)
from quant.data.bundle.catalog import BundleCatalog, CatalogError
from quant.data.bundle.freshness import judge
from quant.data.bundle.migrate import (
    auto_migrate_if_needed,
    migrate_legacy_processed,
)
from quant.data.bundle.provenance import ProvenanceLog, ProvenanceRecord
from quant.data.bundle.store import BundleError, BundleStore
from quant.data.symbols import (
    Exchange,
    SymbolError,
    normalize,
    normalize_many,
    parse_symbol,
    to_eastmoney,
    to_mootdx,
    to_tencent,
)


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


class TestSymbols(unittest.TestCase):
    def test_parse_canonical_forms_roundtrip(self) -> None:
        self.assertEqual(parse_symbol("SH600519"), (Exchange.SSE, "600519"))
        self.assertEqual(parse_symbol("SZ000001"), (Exchange.SZSE, "000001"))
        self.assertEqual(parse_symbol("SYNTHAAA"), (Exchange.SYNTH, "AAA"))
        self.assertEqual(parse_symbol("sh600519"), (Exchange.SSE, "600519"))

    def test_parse_bare_a_share_infers_exchange(self) -> None:
        self.assertEqual(parse_symbol("600519"), (Exchange.SSE, "600519"))
        self.assertEqual(parse_symbol("688981"), (Exchange.SSE, "688981"))
        self.assertEqual(parse_symbol("605358"), (Exchange.SSE, "605358"))
        self.assertEqual(parse_symbol("000001"), (Exchange.SZSE, "000001"))
        self.assertEqual(parse_symbol("002475"), (Exchange.SZSE, "002475"))
        self.assertEqual(parse_symbol("300750"), (Exchange.SZSE, "300750"))

    def test_parse_bare_alpha_goes_to_synth(self) -> None:
        self.assertEqual(parse_symbol("AAA"), (Exchange.SYNTH, "AAA"))
        self.assertEqual(parse_symbol("bbb"), (Exchange.SYNTH, "BBB"))

    def test_parse_rejects_unsupported(self) -> None:
        for bad in ["", "12345", "1234567", "400000", "900000", "BJ831234"]:
            with self.subTest(bad=bad):
                with self.assertRaises(SymbolError):
                    parse_symbol(bad)

    def test_normalize_idempotent(self) -> None:
        for s in ["600519", "SH600519", "sh600519"]:
            self.assertEqual(normalize(s), "SH600519")

    def test_normalize_many_rejects_dups_post_normalize(self) -> None:
        with self.assertRaises(SymbolError):
            normalize_many(["600519", "SH600519"])

    def test_transforms_match_external_apis(self) -> None:
        # mootdx (market=1 for SSE, 0 for SZSE)
        self.assertEqual(to_mootdx("SH600519"), (1, "600519"))
        self.assertEqual(to_mootdx("SZ000001"), (0, "000001"))
        # tencent: lowercase prefix
        self.assertEqual(to_tencent("SH600519"), "sh600519")
        self.assertEqual(to_tencent("SZ300750"), "sz300750")
        # eastmoney: market.code (1 for SH, 0 for SZ)
        self.assertEqual(to_eastmoney("SH688981"), "1.688981")
        self.assertEqual(to_eastmoney("SZ002475"), "0.002475")

    def test_synth_cannot_go_to_real_fetchers(self) -> None:
        for fn in (to_mootdx, to_tencent, to_eastmoney):
            with self.subTest(fn=fn.__name__):
                with self.assertRaises(SymbolError):
                    fn("SYNTHAAA")


# ---------------------------------------------------------------------------
# Manifest (schema validation, roundtrip)
# ---------------------------------------------------------------------------


def _sample_manifest(name: str = "default") -> BundleManifest:
    return BundleManifest(
        name=name,
        created_at="2026-06-23T00:00:00+00:00",
        updated_at="2026-06-23T00:00:00+00:00",
        symbols=["SH600519", "SZ000001", "SZ000002"],
        date_range=DateRange(first="2020-01-01", last="2026-06-20"),
        source_chain=["mootdx"],
        adjustment=AdjustmentMeta(convention="backward", method="mootdx_hfq"),
        row_count=4521,
        calendar=CalendarMeta(source="mootdx", exchange="SSE_SZSE"),
        freshness=FreshnessMeta(
            expected_through="2026-06-20",
            actual_through="2026-06-20",
            status="fresh",
        ),
    )


class TestManifest(unittest.TestCase):
    def test_roundtrip_through_dict(self) -> None:
        m = _sample_manifest()
        self.assertEqual(BundleManifest.from_dict(m.to_dict()), m)

    def test_save_and_load_roundtrip(self) -> None:
        m = _sample_manifest()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            m.save(path)
            self.assertEqual(BundleManifest.load(path), m)

    def test_rejects_unsupported_market(self) -> None:
        d = _sample_manifest().to_dict()
        d["market"] = "us_equity"
        with self.assertRaises(Exception):  # jsonschema.ValidationError
            BundleManifest.from_dict(d)

    def test_rejects_extra_fields(self) -> None:
        d = _sample_manifest().to_dict()
        d["unknown_field"] = "boom"
        with self.assertRaises(Exception):
            BundleManifest.from_dict(d)

    def test_rejects_lowercase_symbol(self) -> None:
        d = _sample_manifest().to_dict()
        d["symbols"] = ["sh600519"]
        with self.assertRaises(Exception):
            BundleManifest.from_dict(d)


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


class TestFreshness(unittest.TestCase):
    def test_fresh_when_actual_at_or_after_expected(self) -> None:
        m = _sample_manifest()
        # actual_through 2026-06-20; ask freshness as_of 2026-06-19
        f = judge(m, as_of=pd.Timestamp("2026-06-19", tz="UTC"))
        self.assertEqual(f.status, "fresh")
        self.assertEqual(f.actual_through, "2026-06-20")

    def test_stale_when_actual_before_expected(self) -> None:
        m = _sample_manifest()
        # actual_through 2026-06-20; ask freshness as_of 2026-06-23 (Mon→Tue→Wed→Mon)
        f = judge(m, as_of=pd.Timestamp("2026-06-23", tz="UTC"))
        self.assertEqual(f.status, "stale")
        self.assertEqual(f.actual_through, "2026-06-20")

    def test_no_data_when_row_count_zero(self) -> None:
        m = _sample_manifest().model_copy(update={"row_count": 0})
        f = judge(m)
        self.assertEqual(f.status, "no_data")


# ---------------------------------------------------------------------------
# Provenance log (append-only)
# ---------------------------------------------------------------------------


class TestProvenance(unittest.TestCase):
    def test_append_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = ProvenanceLog(Path(td) / "provenance.jsonl")
            for i in range(5):
                log.append(ProvenanceRecord(op="fetch", status="ok", rows=i))
            tail = log.tail(3)
            self.assertEqual([r["rows"] for r in tail], [2, 3, 4])

    def test_corrupt_line_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "provenance.jsonl"
            path.write_text('{"ok":1}\nthis-is-not-json\n', encoding="utf-8")
            log = ProvenanceLog(path)
            with self.assertRaises(ValueError):
                log.read_all()


# ---------------------------------------------------------------------------
# BundleStore (create, read, invariants)
# ---------------------------------------------------------------------------


def _toy_ohlcv(symbols: list[str], n: int = 10) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for i, d in enumerate(pd.date_range("2026-06-01", periods=n, freq="B", tz="UTC")):
            p = 100.0 + i
            rows.append({
                "timestamp": d, "symbol": sym,
                "open": p, "high": p + 1, "low": p - 1, "close": p, "volume": 1000,
            })
    return pd.DataFrame(rows)


class TestBundleStore(unittest.TestCase):
    def test_create_and_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = BundleStore.create(
                Path(td) / "alpha",
                name="alpha",
                symbols=["SH600519", "SZ000001", "SZ000002"],
                ohlcv=_toy_ohlcv(["SH600519", "SZ000001", "SZ000002"]),
                source="test",
                adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
                calendar=CalendarMeta(source="synthetic", exchange="SYNTH"),
            )
            m = store.manifest()
            self.assertEqual(m.name, "alpha")
            self.assertEqual(m.market, "a_share_cn")
            self.assertEqual(sorted(m.symbols), ["SH600519", "SZ000001", "SZ000002"])
            df = store.ohlcv()
            self.assertEqual(len(df), 30)
            self.assertEqual(set(df["symbol"]), set(m.symbols))

    def test_create_canonicalizes_symbols_in_ohlcv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # Feed bare A-share codes; expect canonical SH/SZ in the bundle.
            store = BundleStore.create(
                Path(td) / "alpha",
                name="alpha",
                symbols=["600519", "000001", "000002"],
                ohlcv=_toy_ohlcv(["600519", "000001", "000002"]),
                source="test",
                adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
                calendar=CalendarMeta(source="synthetic", exchange="SYNTH"),
            )
            m = store.manifest()
            self.assertEqual(sorted(m.symbols), ["SH600519", "SZ000001", "SZ000002"])
            self.assertEqual(
                set(store.ohlcv()["symbol"]),
                {"SH600519", "SZ000001", "SZ000002"},
            )

    def test_rejects_create_when_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            BundleStore.create(
                Path(td) / "alpha", name="alpha",
                symbols=["SH600519", "SZ000001", "SZ000002"],
                ohlcv=_toy_ohlcv(["SH600519", "SZ000001", "SZ000002"]),
                source="t",
                adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
                calendar=CalendarMeta(source="synthetic", exchange="SYNTH"),
            )
            with self.assertRaises(BundleError):
                BundleStore.create(
                    Path(td) / "alpha", name="alpha",
                    symbols=["SH600519", "SZ000001", "SZ000002"],
                    ohlcv=_toy_ohlcv(["SH600519", "SZ000001", "SZ000002"]),
                    source="t",
                    adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
                    calendar=CalendarMeta(source="synthetic", exchange="SYNTH"),
                )

    def test_rejects_write_ohlcv_with_unknown_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = BundleStore.create(
                Path(td) / "alpha", name="alpha",
                symbols=["SH600519", "SZ000001", "SZ000002"],
                ohlcv=_toy_ohlcv(["SH600519", "SZ000001", "SZ000002"]),
                source="t",
                adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
                calendar=CalendarMeta(source="synthetic", exchange="SYNTH"),
            )
            m = store.manifest()
            bad = _toy_ohlcv(["SH600519", "SZ000001", "SH000999"])
            with self.assertRaises(BundleError):
                store.write_ohlcv(bad, manifest=m)


# ---------------------------------------------------------------------------
# Catalog (multi-bundle index)
# ---------------------------------------------------------------------------


class TestCatalog(unittest.TestCase):
    def test_empty_catalog_when_no_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cat = BundleCatalog.load(Path(td) / "no_such_root")
            self.assertEqual(cat.names(), [])

    def test_register_persist_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cat = BundleCatalog.load(Path(td))
            cat.register(name="alpha")
            cat.register(name="beta")
            cat2 = BundleCatalog.load(Path(td))
            self.assertEqual(cat2.names(), ["alpha", "beta"])

    def test_rejects_dup_register(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cat = BundleCatalog.load(Path(td))
            cat.register(name="alpha")
            with self.assertRaises(CatalogError):
                cat.register(name="alpha")

    def test_rescan_rebuilds_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            BundleStore.create(
                root / "alpha", name="alpha",
                symbols=["SH600519", "SZ000001", "SZ000002"],
                ohlcv=_toy_ohlcv(["SH600519", "SZ000001", "SZ000002"]),
                source="t",
                adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
                calendar=CalendarMeta(source="synthetic", exchange="SYNTH"),
            )
            # No catalog.json yet; rescan should find alpha.
            cat = BundleCatalog.rescan(root)
            self.assertEqual(cat.names(), ["alpha"])

    def test_path_for_unknown_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cat = BundleCatalog.load(Path(td))
            with self.assertRaises(CatalogError):
                cat.path_for("ghost")


# ---------------------------------------------------------------------------
# Migration (idempotent legacy → default bundle)
# ---------------------------------------------------------------------------


def _write_legacy(processed: Path, *, parquet: bool = True, with_metadata: bool = True) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    df = _toy_ohlcv(["000001", "600519", "000002"])
    if parquet:
        df.to_parquet(processed / "local_daily_ohlcv.parquet", index=False)
    else:
        df.to_csv(processed / "local_daily_ohlcv.csv", index=False)
    if with_metadata:
        (processed / "local_daily_metadata.json").write_text(
            json.dumps({
                "adjustment": {
                    "method": "provided_adjusted_close",
                    "declarations": {"adjustment_convention": "backward"},
                },
                "calendar": {"source": "synthetic", "exchange": "SSE_SZSE"},
            })
        )


class TestMigration(unittest.TestCase):
    def test_no_legacy_returns_no_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            r = migrate_legacy_processed(
                bundles_root=Path(td) / "bundles",
                processed_dir=Path(td) / "processed",
            )
            self.assertEqual(r.status, "no_legacy")

    def test_legacy_parquet_creates_default_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            processed = Path(td) / "processed"
            _write_legacy(processed)
            r = migrate_legacy_processed(
                bundles_root=Path(td) / "bundles",
                processed_dir=processed,
            )
            self.assertEqual(r.status, "created")
            self.assertEqual(r.bundle_name, "default")
            store = BundleStore(r.bundle_path)
            m = store.manifest()
            self.assertEqual(sorted(m.symbols), ["SH600519", "SZ000001", "SZ000002"])
            self.assertEqual(m.adjustment.method, "provided_adjusted_close")
            self.assertEqual(m.adjustment.convention, "backward")
            self.assertEqual(m.calendar.exchange, "SSE_SZSE")
            # provenance has a 'create' record
            records = store.provenance.read_all()
            self.assertTrue(any(r["op"] == "create" for r in records))

    def test_legacy_csv_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            processed = Path(td) / "processed"
            _write_legacy(processed, parquet=False)
            r = migrate_legacy_processed(
                bundles_root=Path(td) / "bundles",
                processed_dir=processed,
            )
            self.assertEqual(r.status, "created")

    def test_auto_migrate_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            processed = Path(td) / "processed"
            _write_legacy(processed)
            r1 = auto_migrate_if_needed(
                bundles_root=Path(td) / "bundles",
                processed_dir=processed,
            )
            self.assertEqual(r1.status, "created")
            r2 = auto_migrate_if_needed(
                bundles_root=Path(td) / "bundles",
                processed_dir=processed,
            )
            self.assertEqual(r2.status, "already_exists")

    def test_migration_handles_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            processed = Path(td) / "processed"
            _write_legacy(processed, with_metadata=False)
            r = migrate_legacy_processed(
                bundles_root=Path(td) / "bundles",
                processed_dir=processed,
            )
            self.assertEqual(r.status, "created")
            store = BundleStore(r.bundle_path)
            m = store.manifest()
            # Defaults when no metadata
            self.assertEqual(m.adjustment.method, "raw_unadjusted")
            self.assertEqual(m.calendar.source, "synthetic")


# ---------------------------------------------------------------------------
# quant.app view models
# ---------------------------------------------------------------------------


class TestAppBundleViews(unittest.TestCase):
    def test_list_bundles_empty(self) -> None:
        from quant.app import list_bundles
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(list_bundles(bundles_root=Path(td)), [])

    def test_get_bundle_status_no_bundle(self) -> None:
        from quant.app import get_bundle_status
        with tempfile.TemporaryDirectory() as td:
            r = get_bundle_status("default", bundles_root=Path(td))
            self.assertEqual(r["status"], "no_bundle")
            self.assertIsNone(r["manifest"])

    def test_list_and_get_after_create(self) -> None:
        from quant.app import get_bundle_status, list_bundles
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            BundleStore.create(
                root / "alpha", name="alpha",
                symbols=["SH600519", "SZ000001", "SZ000002"],
                ohlcv=_toy_ohlcv(["SH600519", "SZ000001", "SZ000002"]),
                source="test",
                adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
                calendar=CalendarMeta(source="synthetic", exchange="SYNTH"),
            )
            BundleCatalog.load(root).register(name="alpha")

            summaries = list_bundles(bundles_root=root)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["name"], "alpha")
            self.assertEqual(summaries[0]["source_chain"], ["test"])

            status = get_bundle_status("alpha", bundles_root=root)
            self.assertIn(status["status"], {"fresh", "stale"})
            self.assertEqual(status["manifest"]["market"], "a_share_cn")
            self.assertEqual(
                sorted(status["manifest"]["symbols"]),
                ["SH600519", "SZ000001", "SZ000002"],
            )


if __name__ == "__main__":
    unittest.main()
