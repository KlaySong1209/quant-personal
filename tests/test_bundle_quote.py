"""Stage 4 tests: bundle-backed quote source and account stepping."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.app import run_bundle_quote_step
from quant.data.bundle import AdjustmentMeta, CalendarMeta
from quant.data.bundle.catalog import BundleCatalog
from quant.data.bundle.store import BundleStore
from quant.data.quotes import BundleQuoteSource


def _bundle_df(*, gap_latest: bool = False) -> pd.DataFrame:
    rows = []
    for sym in ["SH600519", "SZ000001", "SZ000002"]:
        for i, d in enumerate(pd.date_range("2026-06-16", periods=3, freq="B", tz="UTC")):
            if gap_latest and sym == "SZ000002" and i == 2:
                continue
            p = 100.0 + i + (1 if sym == "SH600519" else 0)
            rows.append({
                "timestamp": d,
                "symbol": sym,
                "open": p,
                "high": p + 1,
                "low": p - 1,
                "close": p + 0.25,
                "volume": 1000.0 + i,
            })
    return pd.DataFrame(rows)


def _create_bundle(root: Path, *, gap_latest: bool = False) -> BundleStore:
    store = BundleStore.create(
        root / "default",
        name="default",
        symbols=["SH600519", "SZ000001", "SZ000002"],
        ohlcv=_bundle_df(gap_latest=gap_latest),
        source="test",
        adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
        calendar=CalendarMeta(source="synthetic", exchange="SSE_SZSE"),
    )
    BundleCatalog.load(root).register(name="default")
    return store


class TestBundleQuoteSource(unittest.TestCase):
    def test_latest_and_snapshot_use_latest_bundle_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _create_bundle(Path(td))
            q = BundleQuoteSource(store.layout.root)

            latest = q.latest(["600519", "000001", "000002"])
            self.assertEqual(set(latest), {"SH600519", "SZ000001", "SZ000002"})
            # SH has +1 offset and close has +0.25; last row i=2.
            self.assertAlmostEqual(latest["SH600519"], 103.25)
            self.assertAlmostEqual(latest["SZ000001"], 102.25)

            snap = q.snapshot(["SH600519", "SZ000001", "SZ000002"])
            self.assertEqual(list(snap.columns), ["timestamp", "symbol", "close", "open"])
            self.assertEqual(len(snap), 3)
            self.assertEqual(set(snap["symbol"]), {"SH600519", "SZ000001", "SZ000002"})

    def test_as_of_uses_prior_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _create_bundle(Path(td))
            q = BundleQuoteSource(store.layout.root)
            snap = q.snapshot(["SH600519", "SZ000001", "SZ000002"], as_of="2026-06-17")
            self.assertTrue((snap["timestamp"].dt.date.astype(str) == "2026-06-17").all())

    def test_rejects_unknown_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _create_bundle(Path(td))
            q = BundleQuoteSource(store.layout.root)
            with self.assertRaises(ValueError):
                q.latest(["SH600519", "SZ000001", "SZ999999"])

    def test_rejects_misaligned_latest_dates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = _create_bundle(Path(td), gap_latest=True)
            q = BundleQuoteSource(store.layout.root)
            with self.assertRaises(ValueError):
                q.latest(["SH600519", "SZ000001", "SZ000002"])

    def test_empty_bundle_rejects_until_updated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = BundleStore.create(
                root / "default",
                name="default",
                symbols=["SH600519", "SZ000001", "SZ000002"],
                ohlcv=pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]),
                source="empty",
                adjustment=AdjustmentMeta(convention="none", method="raw_unadjusted"),
                calendar=CalendarMeta(source="synthetic", exchange="SSE_SZSE"),
            )
            q = BundleQuoteSource(store.layout.root)
            with self.assertRaises(ValueError):
                q.latest(["SH600519", "SZ000001", "SZ000002"])


class TestRunBundleQuoteStep(unittest.TestCase):
    def test_advances_bundle_account_and_is_idempotent_for_same_date(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundles_root = root / "bundles"
            state_path = root / "state" / "account.json"
            output_dir = root / "results"
            _create_bundle(bundles_root)

            first = run_bundle_quote_step(
                bundle_name="default",
                bundles_root=bundles_root,
                state_path=state_path,
                output_dir=output_dir,
                starting_cash=100000.0,
            )
            self.assertTrue(state_path.exists())
            self.assertTrue(first["ledger_balanced"])
            self.assertEqual(first["steps"], 1)
            self.assertEqual(first["bundle_name"], "default")
            self.assertIn("SH600519", first["positions"])

            # Same latest date again: SimAccount should return the existing step,
            # not append a duplicate history row.
            second = run_bundle_quote_step(
                bundle_name="default",
                bundles_root=bundles_root,
                state_path=state_path,
                output_dir=output_dir,
                starting_cash=100000.0,
            )
            self.assertEqual(second["steps"], 1)
            self.assertTrue(second["ledger_balanced"])
            self.assertEqual(second["advanced_to"], first["advanced_to"])

    def test_unknown_bundle_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                run_bundle_quote_step(
                    bundle_name="ghost",
                    bundles_root=Path(td) / "bundles",
                    state_path=Path(td) / "state.json",
                    output_dir=Path(td) / "results",
                )


if __name__ == "__main__":
    unittest.main()
