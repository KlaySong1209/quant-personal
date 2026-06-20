"""Tests for actionable error messages.

Covers:
  - Stale quote detection produces actionable message
  - Missing open produces actionable message with policy-specific fix
  - Config drift detection
  - No data message
  - Account state corrupt message
  - Symbol mismatch message
  - actionable_error formatter
"""

from __future__ import annotations

import unittest

from quant.errors import (
    diagnose_stale_quotes,
    diagnose_missing_open,
    diagnose_config_drift,
    diagnose_no_data,
    diagnose_account_state_corrupt,
    diagnose_symbol_mismatch,
    actionable_error,
)


class TestStaleQuotes(unittest.TestCase):
    def test_quotes_behind_calendar_produces_message(self):
        msg = diagnose_stale_quotes("2026-06-15", "2026-06-19", quote_path="/data/quotes.csv")
        self.assertIsNotNone(msg)
        self.assertIn("2026-06-15", msg)
        self.assertIn("2026-06-19", msg)
        self.assertIn("append missing rows", msg)
        self.assertIn("/data/quotes.csv", msg)

    def test_quotes_ahead_no_message(self):
        msg = diagnose_stale_quotes("2026-06-20", "2026-06-19")
        self.assertIsNone(msg)

    def test_none_inputs_no_message(self):
        self.assertIsNone(diagnose_stale_quotes(None, "2026-06-19"))
        self.assertIsNone(diagnose_stale_quotes("2026-06-19", None))


class TestMissingOpen(unittest.TestCase):
    def test_missing_open_skip_policy(self):
        msg = diagnose_missing_open(["AAA", "BBB"], "2020-01-15", "skip")
        self.assertIn("AAA", msg)
        self.assertIn("BBB", msg)
        self.assertIn("skip", msg.lower())
        self.assertIn("add open prices", msg)

    def test_missing_open_fail_policy(self):
        msg = diagnose_missing_open(["CCC"], "2020-01-16", "fail")
        self.assertIn("CCC", msg)
        self.assertIn("fail", msg.lower())
        self.assertIn("execution halted", msg)

    def test_missing_open_fallback_policy(self):
        msg = diagnose_missing_open(["DDD"], "2020-01-17", "fallback_to_prev_close")
        self.assertIn("DDD", msg)
        self.assertIn("degraded", msg)
        self.assertIn("real open data", msg)


class TestConfigDrift(unittest.TestCase):
    def test_no_drift(self):
        msg = diagnose_config_drift(
            {"fill_price_rule": "same_day_close"},
            {"fill_price_rule": "same_day_close"},
        )
        self.assertIsNone(msg)

    def test_drift_detected(self):
        msg = diagnose_config_drift(
            {"fill_price_rule": "next_day_open", "mode": "demo"},
            {"fill_price_rule": "same_day_close", "mode": "demo"},
            config_path="/configs/test.yaml",
        )
        self.assertIsNotNone(msg)
        self.assertIn("fill_price_rule", msg)
        self.assertIn("same_day_close", msg)
        self.assertIn("next_day_open", msg)
        self.assertIn("review the differences", msg)

    def test_drift_new_key(self):
        msg = diagnose_config_drift(
            {"new_field": "value"},
            {},
        )
        self.assertIsNotNone(msg)
        self.assertIn("new_field", msg)


class TestNoData(unittest.TestCase):
    def test_no_data_message(self):
        msg = diagnose_no_data(data_dir="/data/processed", quotes_dir="/data/quotes")
        self.assertIn("/data/processed", msg)
        self.assertIn("/data/quotes", msg)
        self.assertIn("generate-example-data", msg)
        self.assertIn("ingest-local-data", msg)


class TestAccountStateCorrupt(unittest.TestCase):
    def test_corrupt_message(self):
        msg = diagnose_account_state_corrupt("/state/bad.json", "JSONDecodeError at line 1")
        self.assertIn("/state/bad.json", msg)
        self.assertIn("JSONDecodeError", msg)
        self.assertIn("delete or rename", msg)
        self.assertIn("restore from a backup", msg)


class TestSymbolMismatch(unittest.TestCase):
    def test_symbol_mismatch_message(self):
        msg = diagnose_symbol_mismatch(
            ["AAA", "BBB", "ZZZ"],
            ["AAA", "BBB", "CCC"],
            source="processed data",
        )
        self.assertIn("ZZZ", msg)
        self.assertIn("CCC", msg)
        self.assertIn("symbol mismatch", msg)
        self.assertIn("update your symbol list", msg)


class TestActionableError(unittest.TestCase):
    def test_formatted_error_includes_all_parts(self):
        msg = actionable_error(
            step="data detection",
            cause="no quotes for 2026-06-19",
            fix="run: python -m quant --ingest-local-data ...",
            details={"data_dir": "/data", "latest_date": "2026-06-15"},
        )
        self.assertIn("ERROR", msg)
        self.assertIn("data detection", msg)
        self.assertIn("no quotes for 2026-06-19", msg)
        self.assertIn("python -m quant", msg)
        self.assertIn("data_dir", msg)
        self.assertIn("2026-06-15", msg)

    def test_formatted_error_no_details(self):
        msg = actionable_error(
            step="account advance",
            cause="missing open prices",
            fix="add open column to quotes",
        )
        self.assertIn("account advance", msg)
        self.assertIn("missing open prices", msg)
        self.assertIn("add open column", msg)


if __name__ == "__main__":
    unittest.main()
