"""Tests for next_day_open pending-order state machine.

Covers:
  - T day creates pending order, does NOT execute
  - T+1 open fills pending order
  - Reload preserves pending state
  - Same day repeated step is idempotent
  - missing_open_policy: skip, fail, fallback_to_prev_close
  - ManualQuoteSource with open column
"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quant.execution.account import SimAccount, PendingOrder


class TestNextDayOpenStateMachine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_path = Path(self.tmpdir) / "account.json"

    def _make_account(self, **kwargs) -> SimAccount:
        defaults = {
            "account_id": "test-ndo",
            "starting_cash": 100000.0,
            "fill_price_rule": "next_day_open",
            "missing_open_policy": "skip",
            "allow_zero_cost_for_tests": True,
        }
        defaults.update(kwargs)
        return SimAccount(**defaults)

    def test_t_day_creates_pending_does_not_execute(self):
        account = self._make_account()
        result = account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        # No trades executed — positions still empty
        self.assertEqual(account.broker.positions(), {})
        self.assertEqual(account.broker.cash, 100000.0)
        # One pending order created
        self.assertEqual(len(account._pending_orders), 1)
        self.assertEqual(account._pending_orders[0].status, "pending")
        self.assertEqual(account._pending_orders[0].fill_rule, "next_day_open")

    def test_t_plus_1_open_fills_pending(self):
        account = self._make_account()
        # T day
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        # T+1 day — open prices fill the pending order, new pending created
        result = account.step(
            pd.Timestamp("2020-01-02", tz="UTC"),
            prices={"000001": 10.5, "000002": 19.5, "000003": 30.0},
            target_weights={"000001": 0.4, "000002": 0.3, "000003": 0.3},
            save_path=self.state_path,
        )
        # First order should be filled
        filled = [o for o in account._pending_orders if o.status == "filled"]
        self.assertEqual(len(filled), 1)
        self.assertIsNotNone(filled[0].filled_on)
        # Positions should now be non-empty
        self.assertNotEqual(account.broker.positions(), {})
        # A new pending order for the next day
        pending = [o for o in account._pending_orders if o.status == "pending"]
        self.assertEqual(len(pending), 1)

    def test_reload_preserves_pending_state(self):
        account = self._make_account()
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        loaded = SimAccount.load(self.state_path)
        self.assertEqual(len(loaded._pending_orders), 1)
        self.assertEqual(loaded._pending_orders[0].status, "pending")
        self.assertEqual(loaded._pending_orders[0].fill_rule, "next_day_open")

    def test_same_day_repeated_step_is_idempotent(self):
        account = self._make_account()
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        first_pending_count = len(account._pending_orders)
        # Repeat same day
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        # No duplicate pending order
        self.assertEqual(len(account._pending_orders), first_pending_count)

    def test_missing_open_policy_skip(self):
        account = self._make_account(missing_open_policy="skip")
        # T day with 3 symbols
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        # T+1 day — missing open for one symbol
        account.step(
            pd.Timestamp("2020-01-02", tz="UTC"),
            prices={"000001": 10.5, "000003": 30.0},  # missing 000002
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        skipped = [o for o in account._pending_orders if o.status == "skipped"]
        self.assertEqual(len(skipped), 1)

    def test_missing_open_policy_fail(self):
        account = self._make_account(missing_open_policy="fail")
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        with self.assertRaises(ValueError):
            account.step(
                pd.Timestamp("2020-01-02", tz="UTC"),
                prices={"000001": 10.5, "000003": 30.0},  # missing 000002
                target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
                save_path=self.state_path,
            )

    def test_missing_open_policy_fallback_to_prev_close(self):
        account = self._make_account(missing_open_policy="fallback_to_prev_close")
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        # T+1 missing 000002 open — falls back to previous close
        account.step(
            pd.Timestamp("2020-01-02", tz="UTC"),
            prices={"000001": 10.5, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        filled = [o for o in account._pending_orders if o.status == "filled"]
        self.assertEqual(len(filled), 1)
        self.assertTrue(filled[0].degraded)

    def test_same_day_close_unchanged(self):
        """same_day_close behavior must not regress."""
        account = SimAccount(
            "test-sdc", 100000.0,
            fill_price_rule="same_day_close",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=self.state_path,
        )
        # Trades executed immediately
        self.assertNotEqual(account.broker.positions(), {})
        self.assertEqual(len(account._pending_orders), 0)


class TestManualQuoteSourceWithOpen(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_snapshot_includes_open(self):
        from quant.data.quotes import ManualQuoteSource

        csv_path = Path(self.tmpdir) / "quotes.csv"
        csv_path.write_text(
            "date,symbol,close,open\n"
            "2020-01-15,000001,10.50,10.20\n"
            "2020-01-15,000002,20.00,19.80\n"
            "2020-01-15,000003,30.00,30.10\n"
        )
        source = ManualQuoteSource(
            csv_path,
            column_mapping={"timestamp": "date", "symbol": "symbol", "close": "close", "open": "open"},
        )
        snapshot = source.snapshot(["000001", "000002", "000003"])
        self.assertIn("open", snapshot.columns)
        self.assertIn("close", snapshot.columns)

    def test_latest_still_returns_close_only(self):
        from quant.data.quotes import ManualQuoteSource

        csv_path = Path(self.tmpdir) / "quotes2.csv"
        csv_path.write_text(
            "date,symbol,close,open\n"
            "2020-01-15,000001,10.50,10.20\n"
            "2020-01-15,000002,20.00,19.80\n"
            "2020-01-15,000003,30.00,30.10\n"
        )
        source = ManualQuoteSource(
            csv_path,
            column_mapping={"timestamp": "date", "symbol": "symbol", "close": "close", "open": "open"},
        )
        prices = source.latest(["000001", "000002", "000003"])
        self.assertEqual(prices, {"000001": 10.50, "000002": 20.00, "000003": 30.00})

    def test_latest_without_open_still_works(self):
        from quant.data.quotes import ManualQuoteSource

        csv_path = Path(self.tmpdir) / "quotes3.csv"
        csv_path.write_text(
            "date,symbol,close\n"
            "2020-01-15,000001,10.50\n"
            "2020-01-15,000002,20.00\n"
            "2020-01-15,000003,30.00\n"
        )
        source = ManualQuoteSource(csv_path, column_mapping={"timestamp": "date", "symbol": "symbol", "close": "close"})
        prices = source.latest(["000001", "000002", "000003"])
        self.assertEqual(prices, {"000001": 10.50, "000002": 20.00, "000003": 30.00})
        # snapshot without open column is fine
        snapshot = source.snapshot(["000001", "000002", "000003"])
        self.assertNotIn("open", snapshot.columns)


if __name__ == "__main__":
    unittest.main()
