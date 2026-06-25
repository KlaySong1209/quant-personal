"""Stage 3 tests: mootdx fetcher + ingestor + app.update_bundle().

Two layers:
  - **Unit tests** (default): pure-function parser + ingestor + update_bundle
    driven by fixtures and stub clients. NO network.
  - **Integration tests** (``@pytest.mark.network``, opt-in): real mootdx
    TCP calls to TDX servers. Run only with: ``pytest -m network``.

Fixture: ``tests/fixtures/mootdx/sh600519_5bars.parquet`` is a real mootdx
response captured once; it lets the parser tests roundtrip against shape
the upstream actually returns.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pytest

from quant.app import update_bundle
from quant.data.bundle import AdjustmentMeta, CalendarMeta
from quant.data.bundle.catalog import BundleCatalog
from quant.data.bundle.store import BundleStore
from quant.data.fetchers.base import FetchError, FetchResult
from quant.data.fetchers.mootdx_daily import MootdxFetcher, parse_mootdx_bars
from quant.data.fetchers.tdx_client import TdxRoute
from quant.data.ingestors.mootdx_ingestor import (
    MootdxIngestor,
    _merge,
    _ohlcv_equal,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "mootdx"


# ---------------------------------------------------------------------------
# Parser (fixture-driven, no network)
# ---------------------------------------------------------------------------


class TestMootdxParser(unittest.TestCase):
    def test_parses_real_fixture(self) -> None:
        raw = pd.read_parquet(FIXTURE_DIR / "sh600519_5bars.parquet")
        parsed = parse_mootdx_bars(raw, canonical_symbol="SH600519")
        self.assertEqual(
            list(parsed.columns),
            ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
        )
        self.assertEqual(len(parsed), 5)
        self.assertEqual(set(parsed["symbol"]), {"SH600519"})
        self.assertIsNotNone(parsed["timestamp"].dt.tz, msg="timestamp must be tz-aware")
        # All times must be normalised to midnight UTC (no hh:mm:ss leftover).
        for ts in parsed["timestamp"]:
            self.assertEqual(ts.hour, 0)
            self.assertEqual(ts.minute, 0)
        self.assertTrue((parsed["close"] > 0).all())

    def test_empty_response_returns_empty_frame(self) -> None:
        parsed = parse_mootdx_bars(pd.DataFrame(), canonical_symbol="SH600519")
        self.assertEqual(len(parsed), 0)
        self.assertEqual(
            list(parsed.columns),
            ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
        )

    def test_parser_deduplicates_within_response(self) -> None:
        raw = pd.DataFrame({
            "open":   [10.0, 10.0],
            "high":   [10.6, 10.6],
            "low":    [9.9,  9.9],
            "close":  [10.4, 10.5],   # second wins
            "vol":    [1000, 1000],
            "volume": [1000, 1000],
            "datetime": ["2026-06-16 15:00", "2026-06-16 15:00"],
        })
        raw = raw.set_index(pd.to_datetime(raw["datetime"]))
        parsed = parse_mootdx_bars(raw, canonical_symbol="SH600519")
        self.assertEqual(len(parsed), 1)
        self.assertAlmostEqual(float(parsed["close"].iloc[0]), 10.5)


# ---------------------------------------------------------------------------
# Ingestor merge logic (pure function, no network, no disk)
# ---------------------------------------------------------------------------


def _df(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date_iso, sym, o, h, l, c, v)"""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([r[0] for r in rows], utc=True),
        "symbol":   [r[1] for r in rows],
        "open":     [r[2] for r in rows],
        "high":     [r[3] for r in rows],
        "low":      [r[4] for r in rows],
        "close":    [r[5] for r in rows],
        "volume":   [float(r[6]) for r in rows],
    })


class TestIngestorMerge(unittest.TestCase):
    def test_empty_existing_keeps_all_incoming(self) -> None:
        incoming = _df([
            ("2026-06-16", "SH600519", 10.0, 10.6, 9.9, 10.4, 1000),
            ("2026-06-17", "SH600519", 10.5, 11.0, 10.3, 10.8, 1100),
        ])
        merged, added, skipped, conflicting = _merge(pd.DataFrame(), incoming)
        self.assertEqual(added, 2)
        self.assertEqual(skipped, 0)
        self.assertEqual(conflicting, 0)
        self.assertEqual(len(merged), 2)

    def test_overlap_with_matching_values_skips(self) -> None:
        existing = _df([("2026-06-16", "SH600519", 10.0, 10.6, 9.9, 10.4, 1000)])
        incoming = _df([
            ("2026-06-16", "SH600519", 10.0, 10.6, 9.9, 10.4, 1000),  # dup
            ("2026-06-17", "SH600519", 10.5, 11.0, 10.3, 10.8, 1100),
        ])
        merged, added, skipped, conflicting = _merge(existing, incoming)
        self.assertEqual(added, 1)
        self.assertEqual(skipped, 1)
        self.assertEqual(conflicting, 0)
        self.assertEqual(len(merged), 2)

    def test_overlap_with_disagreement_flags_conflict(self) -> None:
        existing = _df([("2026-06-16", "SH600519", 10.0, 10.6, 9.9, 10.4, 1000)])
        incoming = _df([("2026-06-16", "SH600519", 10.0, 10.6, 9.9, 99.99, 1000)])  # bad close
        _, added, skipped, conflicting = _merge(existing, incoming)
        self.assertEqual(conflicting, 1)
        self.assertEqual(skipped, 0)

    def test_ohlcv_equal_tolerates_float_noise(self) -> None:
        a = pd.Series({"open": 10.0, "high": 10.6, "low": 9.9, "close": 10.4, "volume": 1000.0})
        b = pd.Series({"open": 10.00001, "high": 10.6, "low": 9.9, "close": 10.4, "volume": 1000.0})
        self.assertTrue(_ohlcv_equal(a, b))


# ---------------------------------------------------------------------------
# End-to-end: stub fetcher + real ingestor + update_bundle()
# ---------------------------------------------------------------------------


def _make_stub_fetcher(raw_frame: pd.DataFrame) -> MootdxFetcher:
    raw_idx = raw_frame.set_index(pd.to_datetime(raw_frame["datetime"]))

    class StubClient:
        def bars(self, *, symbol, frequency, offset):
            return raw_idx.copy()

    return MootdxFetcher(client_factory=lambda: (StubClient(), TdxRoute(method="stub", server=None)))


class TestUpdateBundle(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.bundles = self.root / "bundles"
        self.raw = self.root / "raw"
        # Seed a bundle with 2 rows for SH600519.
        BundleStore.create(
            self.bundles / "default", name="default",
            symbols=["SH600519"],
            ohlcv=_df([
                ("2026-06-16", "SH600519", 10.0, 10.6, 9.9, 10.4, 1000),
                ("2026-06-17", "SH600519", 10.5, 11.0, 10.3, 10.8, 1100),
            ]),
            source="seed",
            adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
            calendar=CalendarMeta(source="synthetic", exchange="SSE_SZSE"),
        )
        BundleCatalog.load(self.bundles).register(name="default")

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_update_appends_new_rows(self) -> None:
        raw = pd.DataFrame({
            "open":   [10.0, 10.5, 10.7, 10.9],
            "close":  [10.4, 10.8, 11.0, 11.1],
            "high":   [10.6, 11.0, 11.1, 11.2],
            "low":    [9.9,  10.3, 10.5, 10.8],
            "vol":    [1000, 1100, 1200, 1300],
            "volume": [1000, 1100, 1200, 1300],
            "datetime": [
                "2026-06-16 15:00", "2026-06-17 15:00",
                "2026-06-18 15:00", "2026-06-19 15:00",
            ],
        })
        result = update_bundle(
            "default",
            bundles_root=self.bundles, raw_root=self.raw,
            fetcher=_make_stub_fetcher(raw),
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rows_added"], 2)
        self.assertEqual(result["rows_skipped"], 2)
        self.assertEqual(result["new_last_date"], "2026-06-19")

    def test_update_is_idempotent(self) -> None:
        raw = pd.DataFrame({
            "open":   [10.7], "close": [11.0], "high": [11.1], "low": [10.5],
            "vol":    [1200], "volume": [1200],
            "datetime": ["2026-06-18 15:00"],
        })
        f = _make_stub_fetcher(raw)
        r1 = update_bundle("default", bundles_root=self.bundles, raw_root=self.raw, fetcher=f)
        r2 = update_bundle("default", bundles_root=self.bundles, raw_root=self.raw, fetcher=f)
        self.assertEqual(r1["rows_added"], 1)
        self.assertEqual(r2["rows_added"], 0)
        self.assertEqual(r2["rows_skipped"], 1)

    def test_update_aborts_on_conflict(self) -> None:
        # Try to overwrite 2026-06-16 close with a different value.
        raw = pd.DataFrame({
            "open":   [10.0], "close": [99.99], "high": [10.6], "low": [9.9],
            "vol":    [1000], "volume": [1000],
            "datetime": ["2026-06-16 15:00"],
        })
        result = update_bundle(
            "default", bundles_root=self.bundles, raw_root=self.raw,
            fetcher=_make_stub_fetcher(raw),
        )
        self.assertEqual(result["status"], "failed")
        self.assertGreaterEqual(result["rows_conflicting"], 1)
        # Bundle untouched.
        store = BundleStore(self.bundles / "default")
        self.assertEqual(store.manifest().row_count, 2)

    def test_unknown_bundle_returns_no_bundle(self) -> None:
        result = update_bundle(
            "ghost",
            bundles_root=self.bundles, raw_root=self.raw,
            fetcher=_make_stub_fetcher(pd.DataFrame(columns=[
                "open", "close", "high", "low", "vol", "volume", "datetime",
            ])),
        )
        self.assertEqual(result["status"], "no_bundle")

    def test_fetcher_transport_error_records_failure(self) -> None:
        class BoomFetcher:
            source = "mootdx"
            def fetch_daily_ohlcv(self, *a, **kw):
                raise FetchError("no TDX server reachable")
        result = update_bundle(
            "default",
            bundles_root=self.bundles, raw_root=self.raw,
            fetcher=BoomFetcher(),
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("no TDX", result["error"])
        # Bundle untouched.
        store = BundleStore(self.bundles / "default")
        self.assertEqual(store.manifest().row_count, 2)
        # Provenance recorded the failed fetch.
        prov = store.provenance.read_all()
        self.assertTrue(any(p["op"] == "fetch" and p["status"] == "failed" for p in prov))


# ---------------------------------------------------------------------------
# Integration tests (real mootdx TCP — opt-in via `pytest -m network`)
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestMootdxLive(unittest.TestCase):
    def test_pulls_real_bars_for_moutai(self) -> None:
        from quant.data.fetchers.tdx_client import open_client
        client, route = open_client()
        df = client.bars(symbol="600519", frequency=9, offset=3)
        self.assertGreater(len(df), 0)
        self.assertIn("close", df.columns)
        self.assertTrue((df["close"] > 0).all())

    def test_fetcher_writes_raw_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fetcher = MootdxFetcher()
            res = fetcher.fetch_daily_ohlcv(
                ["SH600519", "SZ000001", "SZ000002"],
                raw_dir=Path(td),
            )
            self.assertIn(res.status, {"ok", "partial"})
            self.assertGreater(len(res.raw_paths), 0)
            written = pd.read_parquet(res.raw_paths[0])
            self.assertGreater(len(written), 0)
            self.assertTrue(set(written["symbol"]).issubset(
                {"SH600519", "SZ000001", "SZ000002"}
            ))


if __name__ == "__main__":
    unittest.main()
