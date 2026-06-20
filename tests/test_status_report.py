"""Tests for status report module.

Covers:
  - Account report fields come from recorded state/metadata
  - Pending orders surfaced correctly
  - Assumptions surfaced correctly
  - Plain-language metric lines present
  - Backtest report loads from run dir
  - Combined report works
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.execution.account import SimAccount
from quant.report import (
    account_report,
    format_account_report,
    backtest_report,
    combined_report,
    format_combined_report,
)


class TestAccountReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_report_no_state_file(self):
        path = Path(self.tmpdir) / "nonexistent.json"
        report = account_report(path)
        self.assertEqual(report["report_type"], "account_status")
        self.assertIsNone(report["account"])
        self.assertEqual(report["flags"], ["no account state file found"])

    def test_report_fields_from_recorded_state(self):
        state_path = Path(self.tmpdir) / "account.json"
        account = SimAccount(
            "rpt-test", 100000.0,
            fill_price_rule="next_day_open",
            missing_open_policy="skip",
            mode="paper_simulation",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        report = account_report(state_path)
        self.assertIsNone(report["error"])
        self.assertEqual(report["account"]["account_id"], "rpt-test")
        self.assertEqual(report["account"]["mode"], "paper_simulation")
        self.assertEqual(report["account"]["steps"], 1)
        self.assertIsNotNone(report["equity"]["total_equity"])
        self.assertIsNotNone(report["equity"]["cash"])
        self.assertIsNotNone(report["equity"]["position_value"])

    def test_report_assumptions_surfaced(self):
        state_path = Path(self.tmpdir) / "account.json"
        account = SimAccount(
            "rpt-assume", 100000.0,
            fill_price_rule="same_day_close",
            commission_bps=2.0,
            stamp_duty_bps=10.0,
            slippage_bps=3.0,
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        report = account_report(state_path)
        assumptions = report["assumptions"]
        self.assertEqual(assumptions["fill_price_rule"], "same_day_close")
        self.assertEqual(assumptions["commission_bps"], 2.0)
        self.assertEqual(assumptions["stamp_duty_bps"], 10.0)
        self.assertEqual(assumptions["slippage_bps"], 3.0)
        self.assertIn("none", assumptions["order_routing"])

    def test_report_does_not_infer_missing_assumptions(self):
        state_path = Path(self.tmpdir) / "account.json"
        state_path.write_text(
            json.dumps(
                {
                    "account_id": "missing-assumptions",
                    "mode": "paper_simulation",
                    "label": "SIMULATED / PAPER -- NOT REAL",
                    "broker": {
                        "starting_cash": 100000.0,
                        "cash": 100000.0,
                        "events": [],
                        "last_prices": {"AAA": 10.0},
                    },
                    "history": [
                        {
                            "timestamp": "2020-01-02T00:00:00+00:00",
                            "cash": 100000.0,
                            "position_value": 0.0,
                            "equity": 100000.0,
                            "positions": {},
                        }
                    ],
                    "pending_orders": [],
                }
            ),
            encoding="utf-8",
        )

        report = account_report(state_path)

        self.assertEqual(report["assumptions"]["fill_price_rule"], "recorded-missing")
        self.assertEqual(report["assumptions"]["missing_open_policy"], "recorded-missing")
        self.assertEqual(report["assumptions"]["commission_bps"], "recorded-missing")

    def test_report_pending_orders_surfaced(self):
        state_path = Path(self.tmpdir) / "account.json"
        account = SimAccount(
            "rpt-po", 100000.0,
            fill_price_rule="next_day_open",
            missing_open_policy="skip",
            allow_zero_cost_for_tests=True,
        )
        # T day: creates pending order
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        report = account_report(state_path)
        self.assertGreater(len(report["pending_orders"]), 0)
        pending = report["pending_orders"][0]
        self.assertEqual(pending["status"], "pending")
        self.assertIn("waiting", pending["reason"].lower())

    def test_report_positions_included(self):
        state_path = Path(self.tmpdir) / "account.json"
        account = SimAccount(
            "rpt-pos", 100000.0,
            fill_price_rule="same_day_close",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        report = account_report(state_path)
        self.assertGreater(len(report["positions"]), 0)
        self.assertIn("symbol", report["positions"][0])
        self.assertIn("shares", report["positions"][0])
        self.assertIn("value", report["positions"][0])


class TestFormatAccountReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_format_includes_key_sections(self):
        state_path = Path(self.tmpdir) / "account.json"
        account = SimAccount(
            "fmt-test", 100000.0,
            fill_price_rule="same_day_close",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        report = account_report(state_path)
        text = format_account_report(report)
        self.assertIn("ACCOUNT", text)
        self.assertIn("EQUITY", text)
        self.assertIn("ASSUMPTIONS", text)
        self.assertIn("POSITIONS", text)
        self.assertIn("fmt-test", text)

    def test_format_includes_plain_language_metric_lines(self):
        state_path = Path(self.tmpdir) / "account.json"
        account = SimAccount(
            "lang-test", 100000.0,
            fill_price_rule="next_day_open",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        report = account_report(state_path)
        text = format_account_report(report)
        self.assertIn("fill_price_rule", text)
        self.assertIn("commission_bps", text)
        self.assertIn("slippage_bps", text)
        self.assertIn("order_routing", text)

    def test_format_corrupt_file(self):
        state_path = Path(self.tmpdir) / "corrupt.json"
        state_path.write_text("not json {{{")
        report = account_report(state_path)
        self.assertIsNotNone(report["error"])
        text = format_account_report(report)
        self.assertIn("ERROR", text)


class TestBacktestReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_backtest_report_nonexistent_dir(self):
        report = backtest_report(Path(self.tmpdir) / "no_such_run")
        self.assertIsNotNone(report["error"])
        self.assertIsNone(report["metrics"])


class TestCombinedReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_combined_report_account_only(self):
        state_path = Path(self.tmpdir) / "account.json"
        account = SimAccount(
            "combined-test", 100000.0,
            fill_price_rule="same_day_close",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        report = combined_report(state_path)
        self.assertEqual(report["report_type"], "combined")
        self.assertIsNotNone(report["account"])
        self.assertIsNone(report["backtest"])

    def test_combined_report_format(self):
        state_path = Path(self.tmpdir) / "account.json"
        account = SimAccount(
            "combined-fmt", 100000.0,
            fill_price_rule="same_day_close",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            save_path=state_path,
        )
        report = combined_report(state_path)
        text = format_combined_report(report)
        self.assertIn("ACCOUNT", text)
        self.assertIn("EQUITY", text)


if __name__ == "__main__":
    unittest.main()
