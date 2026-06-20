"""Tests for SnapshotQuoteSource and ManualQuoteSource snapshot().

Covers:
  - SnapshotQuoteSource required columns fail-fast
  - Snapshot reads are deterministic (same file → same result)
  - Snapshot with open column
  - RealtimeQuoteSource continues to raise NotImplementedError
"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data.quotes import (
    SnapshotQuoteSource,
    ManualQuoteSource,
    RealtimeQuoteSource,
)


class TestSnapshotQuoteSource(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_snapshot(self, name: str, content: str) -> Path:
        path = Path(self.tmpdir) / name
        path.write_text(content)
        return path

    def test_snapshot_with_all_required_columns(self):
        path = self._write_snapshot("valid.csv",
            "fetched_at,source,as_of_date,timestamp,symbol,close\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000001,10.50\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000002,20.00\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000003,30.00\n"
        )
        source = SnapshotQuoteSource(path)
        prices = source.latest(["000001", "000002", "000003"])
        self.assertEqual(prices, {"000001": 10.50, "000002": 20.00, "000003": 30.00})

    def test_snapshot_missing_required_column_fails(self):
        path = self._write_snapshot("missing_fetched_at.csv",
            "source,as_of_date,timestamp,symbol,close\n"
            "manual,2020-01-15,2020-01-15T00:00:00Z,000001,10.50\n"
            "manual,2020-01-15,2020-01-15T00:00:00Z,000002,20.00\n"
            "manual,2020-01-15,2020-01-15T00:00:00Z,000003,30.00\n"
        )
        source = SnapshotQuoteSource(path)
        with self.assertRaises(ValueError):
            source.latest(["000001", "000002", "000003"])

    def test_snapshot_missing_source_fails(self):
        path = self._write_snapshot("missing_source.csv",
            "fetched_at,as_of_date,timestamp,symbol,close\n"
            "2020-01-15T09:00:00Z,2020-01-15,2020-01-15T00:00:00Z,000001,10.50\n"
            "2020-01-15T09:00:00Z,2020-01-15,2020-01-15T00:00:00Z,000002,20.00\n"
            "2020-01-15T09:00:00Z,2020-01-15,2020-01-15T00:00:00Z,000003,30.00\n"
        )
        source = SnapshotQuoteSource(path)
        with self.assertRaises(ValueError):
            source.latest(["000001", "000002", "000003"])

    def test_snapshot_missing_as_of_date_fails(self):
        path = self._write_snapshot("missing_as_of.csv",
            "fetched_at,source,timestamp,symbol,close\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15T00:00:00Z,000001,10.50\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15T00:00:00Z,000002,20.00\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15T00:00:00Z,000003,30.00\n"
        )
        source = SnapshotQuoteSource(path)
        with self.assertRaises(ValueError):
            source.latest(["000001", "000002", "000003"])

    def test_snapshot_read_twice_returns_same_result(self):
        path = self._write_snapshot("stable.csv",
            "fetched_at,source,as_of_date,timestamp,symbol,close\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000001,10.50\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000002,20.00\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000003,30.00\n"
        )
        source = SnapshotQuoteSource(path)
        prices1 = source.latest(["000001", "000002", "000003"])
        prices2 = source.latest(["000001", "000002", "000003"])
        self.assertEqual(prices1, prices2)

    def test_snapshot_with_open_column(self):
        path = self._write_snapshot("with_open.csv",
            "fetched_at,source,as_of_date,timestamp,symbol,close,open\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000001,10.50,10.20\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000002,20.00,19.80\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000003,30.00,30.10\n"
        )
        source = SnapshotQuoteSource(path)
        snapshot = source.snapshot(["000001", "000002", "000003"])
        self.assertIn("open", snapshot.columns)
        self.assertEqual(snapshot.loc[snapshot["symbol"] == "000001", "open"].iloc[0], 10.20)

    def test_snapshot_symbols_must_match(self):
        path = self._write_snapshot("subset.csv",
            "fetched_at,source,as_of_date,timestamp,symbol,close\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000001,10.50\n"
            "2020-01-15T09:00:00Z,manual,2020-01-15,2020-01-15T00:00:00Z,000002,20.00\n"
        )
        source = SnapshotQuoteSource(path)
        with self.assertRaises(ValueError):
            source.latest(["000001", "000002", "000003"])  # 000003 missing

    def test_snapshot_file_not_found(self):
        source = SnapshotQuoteSource(Path(self.tmpdir) / "does_not_exist.csv")
        with self.assertRaises(FileNotFoundError):
            source.latest(["000001", "000002", "000003"])

    def test_realtime_quote_source_raises_not_implemented(self):
        source = RealtimeQuoteSource()
        with self.assertRaises(NotImplementedError):
            source.latest(["000001", "000002", "000003"])


if __name__ == "__main__":
    unittest.main()
