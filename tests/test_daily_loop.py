"""Tests for daily research loop.

Covers:
  - Daily loop advances account correctly
  - Idempotent on same-day re-run
  - Handles "no new data" without error
  - CLI --daily flag works
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.loop import (
    run_daily,
    _detect_latest_data_date,
    _detect_latest_quote_date,
    _account_advanced_to,
)
from quant.execution.account import SimAccount


class TestDailyLoopDetection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_detect_latest_data_date_from_processed(self):
        root = Path(self.tmpdir)
        data_dir = root / "processed"
        data_dir.mkdir(parents=True)
        ohlcv = pd.DataFrame({
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"], utc=True),
            "symbol": ["AAA", "BBB", "CCC"],
            "open": [10.0, 20.0, 30.0],
            "high": [10.0, 20.0, 30.0],
            "low": [10.0, 20.0, 30.0],
            "close": [10.0, 20.0, 30.0],
            "volume": [1000, 1000, 1000],
        })
        ohlcv.to_parquet(data_dir / "local_daily_ohlcv.parquet")
        result = _detect_latest_data_date(data_dir)
        self.assertEqual(result, pd.Timestamp("2020-01-03", tz="UTC"))

    def test_detect_latest_data_date_empty_dir(self):
        root = Path(self.tmpdir)
        data_dir = root / "empty"
        data_dir.mkdir(parents=True)
        result = _detect_latest_data_date(data_dir)
        self.assertIsNone(result)

    def test_detect_latest_quote_date(self):
        root = Path(self.tmpdir)
        quotes_dir = root / "quotes"
        quotes_dir.mkdir(parents=True)
        (quotes_dir / "quotes1.csv").write_text(
            "date,symbol,close\n2020-01-01,AAA,10.0\n2020-01-01,BBB,20.0\n"
        )
        (quotes_dir / "quotes2.csv").write_text(
            "date,symbol,close\n2020-01-05,AAA,11.0\n2020-01-05,BBB,21.0\n"
        )
        result = _detect_latest_quote_date(quotes_dir)
        self.assertEqual(result, pd.Timestamp("2020-01-05", tz="UTC"))

    def test_detect_latest_quote_date_empty_dir(self):
        root = Path(self.tmpdir)
        quotes_dir = root / "no_quotes"
        quotes_dir.mkdir(parents=True)
        result = _detect_latest_quote_date(quotes_dir)
        self.assertIsNone(result)

    def test_account_advanced_to_returns_none_for_no_state(self):
        root = Path(self.tmpdir)
        state_path = root / "nonexistent.json"
        result = _account_advanced_to(state_path)
        self.assertIsNone(result)

    def test_account_advanced_to_returns_last_date(self):
        root = Path(self.tmpdir)
        state_path = root / "account.json"
        account = SimAccount("test-dl", 100000.0, allow_zero_cost_for_tests=True)
        account.step(
            pd.Timestamp("2020-01-03", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        result = _account_advanced_to(state_path)
        self.assertEqual(result, pd.Timestamp("2020-01-03", tz="UTC"))


class TestDailyLoopIdempotency(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)

    def _setup_data(self):
        """Create processed data, quotes, and initial account state."""
        data_dir = self.root / "processed"
        data_dir.mkdir(parents=True)
        quotes_dir = self.root / "quotes"
        quotes_dir.mkdir(parents=True)
        state_path = self.root / "account.json"

        # Processed data — 2 dates x 3 symbols = 6 rows
        ohlcv = pd.DataFrame({
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-01",
                 "2020-01-02", "2020-01-02", "2020-01-02"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "open": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
            "high": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
            "low": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
            "close": [10.0, 20.0, 30.0, 11.0, 21.0, 31.0],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000],
        })
        ohlcv.to_parquet(data_dir / "local_daily_ohlcv.parquet")

        return data_dir, quotes_dir, state_path

    def test_daily_loop_no_new_data_returns_no_new_data(self):
        data_dir, quotes_dir, state_path = self._setup_data()

        # First, advance account to latest date
        result1 = run_daily(
            state_path=state_path,
            data_dir=data_dir,
            quotes_dir=quotes_dir,
            symbols=["AAA", "BBB", "CCC"],
            starting_cash=100000.0,
        )
        self.assertIn(result1["status"], ("ok", "no_new_data", "error"))
        if result1["status"] == "ok":
            self.assertTrue(result1["account_advanced"])

        # Second run — should be no_new_data
        result2 = run_daily(
            state_path=state_path,
            data_dir=data_dir,
            quotes_dir=quotes_dir,
            symbols=["AAA", "BBB", "CCC"],
            starting_cash=100000.0,
        )
        self.assertEqual(result2["status"], "no_new_data")
        self.assertFalse(result2["account_advanced"])
        self.assertEqual(result2["steps_run"], 0)

    def test_daily_loop_re_run_idempotent(self):
        data_dir, quotes_dir, state_path = self._setup_data()

        # Run twice in a row — same account date, same result
        result1 = run_daily(
            state_path=state_path,
            data_dir=data_dir,
            quotes_dir=quotes_dir,
            symbols=["AAA", "BBB", "CCC"],
            starting_cash=100000.0,
        )
        result2 = run_daily(
            state_path=state_path,
            data_dir=data_dir,
            quotes_dir=quotes_dir,
            symbols=["AAA", "BBB", "CCC"],
            starting_cash=100000.0,
        )
        # Both should report the same account date (first advances, second is idempotent)
        self.assertIsNotNone(result1.get("account_date"))
        self.assertIsNotNone(result2.get("account_date"))
        # Normalize to Timestamp for comparison (isoformat vs str may differ)
        self.assertEqual(
            pd.Timestamp(result1["account_date"]),
            pd.Timestamp(result2["account_date"]),
        )

    def test_daily_loop_advances_from_quotes_directory_without_processed_data(self):
        data_dir = self.root / "processed"
        data_dir.mkdir(parents=True)
        quotes_dir = self.root / "quotes"
        quotes_dir.mkdir(parents=True)
        state_path = self.root / "account.json"
        (quotes_dir / "manual.csv").write_text(
            "date,symbol,close,open\n"
            "2020-01-02,AAA,10.0,10.1\n"
            "2020-01-02,BBB,20.0,20.1\n"
            "2020-01-02,CCC,30.0,30.1\n",
            encoding="utf-8",
        )

        result = run_daily(
            state_path=state_path,
            data_dir=data_dir,
            quotes_dir=quotes_dir,
            symbols=["AAA", "BBB", "CCC"],
            starting_cash=100000.0,
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["account_advanced"])
        self.assertEqual(result["steps_run"], 1)
        self.assertTrue(state_path.exists())
        self.assertIn("Account Status", result["report_text"])

    def test_daily_loop_no_data_error_names_step_cause_and_fix(self):
        empty_dir = self.root / "empty"
        empty_dir.mkdir(parents=True)

        result = run_daily(
            state_path=self.root / "no_account.json",
            data_dir=empty_dir,
            quotes_dir=empty_dir,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("step:", result["error"])
        self.assertIn("cause:", result["error"])
        self.assertIn("fix:", result["error"])


class TestDailyLoopCLI(unittest.TestCase):
    def test_daily_flag_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "quant", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--daily", proc.stdout)

    def test_report_flag_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "quant", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--report", proc.stdout)

    def test_daily_with_manual_quotes_uses_daily_loop_and_next_day_open_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            quote_path = root / "manual.csv"
            state_path = root / "account.json"
            quote_path.write_text(
                "date,symbol,close,open\n"
                "2020-01-02,AAA,10.0,10.1\n"
                "2020-01-02,BBB,20.0,20.1\n"
                "2020-01-02,CCC,30.0,30.1\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "quant",
                    "--daily",
                    "--manual-quotes",
                    str(quote_path),
                    "--account-state",
                    str(state_path),
                    "--symbols",
                    "AAA",
                    "BBB",
                    "CCC",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertIn("Account Status", proc.stdout)
            self.assertNotIn("equity history:", proc.stdout)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["assumptions"]["fill_price_rule"], "next_day_open")


class TestDailyLoopErrors(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)

    def test_daily_loop_no_data_error(self):
        empty_dir = self.root / "empty"
        empty_dir.mkdir(parents=True)
        result = run_daily(
            state_path=self.root / "no_account.json",
            data_dir=empty_dir,
            quotes_dir=empty_dir,
        )
        self.assertEqual(result["status"], "error")
        self.assertIsNotNone(result["error"])

    def test_daily_loop_bad_quote_file_error_is_actionable(self):
        quotes_dir = self.root / "quotes"
        quotes_dir.mkdir(parents=True)
        data_dir = self.root / "processed"
        data_dir.mkdir(parents=True)
        (quotes_dir / "bad.csv").write_text("symbol,close\nAAA,10\n", encoding="utf-8")

        result = run_daily(
            state_path=self.root / "account.json",
            data_dir=data_dir,
            quotes_dir=quotes_dir,
            symbols=["AAA", "BBB", "CCC"],
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("quote detection", result["error"])
        self.assertIn("step:", result["error"])
        self.assertIn("fix:", result["error"])

    def test_daily_loop_bad_processed_data_error_is_actionable(self):
        data_dir = self.root / "processed"
        data_dir.mkdir(parents=True)
        quotes_dir = self.root / "quotes"
        quotes_dir.mkdir(parents=True)
        (data_dir / "local_daily_ohlcv.csv").write_text(
            "date,symbol,close\n2020-01-02,AAA,10\n",
            encoding="utf-8",
        )

        result = run_daily(
            state_path=self.root / "account.json",
            data_dir=data_dir,
            quotes_dir=quotes_dir,
            symbols=["AAA", "BBB", "CCC"],
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("data detection", result["error"])
        self.assertIn("cannot read processed data", result["error"])
        self.assertIn("step:", result["error"])
        self.assertIn("fix:", result["error"])

    def test_daily_loop_config_drift_error_is_actionable(self):
        data_dir = self.root / "processed"
        data_dir.mkdir(parents=True)
        quotes_dir = self.root / "quotes"
        quotes_dir.mkdir(parents=True)
        state_path = self.root / "account.json"
        account = SimAccount(
            "drift-test",
            100000.0,
            fill_price_rule="same_day_close",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        (quotes_dir / "manual.csv").write_text(
            "date,symbol,close,open\n"
            "2020-01-02,AAA,10.0,10.1\n"
            "2020-01-02,BBB,20.0,20.1\n"
            "2020-01-02,CCC,30.0,30.1\n",
            encoding="utf-8",
        )

        result = run_daily(
            state_path=state_path,
            data_dir=data_dir,
            quotes_dir=quotes_dir,
            symbols=["AAA", "BBB", "CCC"],
            fill_price_rule="next_day_open",
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("config", result["error"])
        self.assertIn("fill_price_rule", result["error"])
        self.assertIn("fix:", result["error"])


if __name__ == "__main__":
    unittest.main()
