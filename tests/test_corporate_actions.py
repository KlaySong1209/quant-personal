"""Tests for corporate action declarations and account-layer application.

Covers:
  - CorporateAction event creation and validation
  - Account-layer application (cash dividend, stock dividend, split, rights)
  - Idempotency (same event applied twice = no double count)
  - Fail-fast on missing required fields
"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data.adjust.corporate_actions import (
    CorporateAction,
    validate_corporate_actions,
    corporate_actions_for_date,
)
from quant.execution.corporate_actions import (
    apply_corporate_action,
    apply_corporate_actions_for_date,
)
from quant.execution.account import SimAccount


class TestCorporateActionDeclaration(unittest.TestCase):
    def test_cash_dividend_event_key_is_deterministic(self):
        ca = CorporateAction(
            timestamp=pd.Timestamp("2020-01-15", tz="UTC"),
            symbol="000001",
            action_type="cash_dividend",
            cash_per_share=0.50,
        )
        self.assertEqual(ca.event_key, "2020-01-15T00:00:00+00:00::000001::cash_dividend")

    def test_stock_dividend_10_for_10(self):
        ca = CorporateAction(
            timestamp=pd.Timestamp("2020-06-01", tz="UTC"),
            symbol="000002",
            action_type="stock_dividend",
            share_ratio=1.0,
        )
        self.assertEqual(ca.share_ratio, 1.0)
        self.assertEqual(ca.action_type, "stock_dividend")

    def test_split_2_for_1(self):
        ca = CorporateAction(
            timestamp=pd.Timestamp("2020-03-15", tz="UTC"),
            symbol="000003",
            action_type="split",
            split_ratio=2.0,
        )
        self.assertEqual(ca.split_ratio, 2.0)

    def test_fail_on_missing_action_type(self):
        with self.assertRaises(ValueError):
            CorporateAction(
                timestamp=pd.Timestamp("2020-01-01"),
                symbol="X",
                action_type="invalid_type",
            )

    def test_fail_on_empty_symbol(self):
        with self.assertRaises(ValueError):
            CorporateAction(
                timestamp=pd.Timestamp("2020-01-01"),
                symbol="",
                action_type="cash_dividend",
                cash_per_share=0.5,
            )

    def test_naive_timestamp_gets_utc(self):
        ca = CorporateAction(
            timestamp=pd.Timestamp("2020-01-01"),
            symbol="A",
            action_type="cash_dividend",
            cash_per_share=1.0,
        )
        self.assertIsNotNone(ca.timestamp.tzinfo)

    def test_validate_rejects_duplicate_keys(self):
        ca1 = CorporateAction("2020-01-01", "A", "cash_dividend", cash_per_share=0.5)
        ca2 = CorporateAction("2020-01-01", "A", "cash_dividend", cash_per_share=0.5)
        with self.assertRaises(ValueError):
            validate_corporate_actions([ca1, ca2])

    def test_validate_rejects_negative_cash_dividend(self):
        with self.assertRaises(ValueError):
            CorporateAction("2020-01-01", "A", "cash_dividend", cash_per_share=-0.5)

    def test_validate_rejects_zero_share_ratio_for_stock_dividend(self):
        with self.assertRaises(ValueError):
            CorporateAction("2020-01-01", "A", "stock_dividend", share_ratio=0.0)

    def test_validate_rejects_zero_split_ratio(self):
        with self.assertRaises(ValueError):
            CorporateAction("2020-01-01", "A", "split", split_ratio=0.0)

    def test_validate_rejects_reverse_split_ge_1(self):
        with self.assertRaises(ValueError):
            CorporateAction("2020-01-01", "A", "reverse_split", split_ratio=1.5)

    def test_corporate_actions_for_date_filters_correctly(self):
        ca1 = CorporateAction("2020-01-15", "000001", "cash_dividend", cash_per_share=0.5)
        ca2 = CorporateAction("2020-01-16", "000001", "cash_dividend", cash_per_share=0.3)
        actions = [ca1, ca2]
        result = corporate_actions_for_date(actions, "000001", pd.Timestamp("2020-01-15", tz="UTC"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].cash_per_share, 0.5)


class TestCorporateActionApplication(unittest.TestCase):
    def test_cash_dividend_increases_cash(self):
        ca = CorporateAction("2020-01-15", "000001", "cash_dividend", cash_per_share=0.50)
        positions = {"000001": 1000.0}
        cash = 10000.0
        new_pos, new_cash, record = apply_corporate_action(ca, positions, cash)
        self.assertEqual(new_pos["000001"], 1000.0)
        self.assertEqual(new_cash, 10000.0 + 1000 * 0.50)
        self.assertEqual(record.cash_delta, 500.0)
        self.assertEqual(record.shares_delta, 0.0)

    def test_stock_dividend_10_for_10_doubles_shares(self):
        ca = CorporateAction("2020-06-01", "000002", "stock_dividend", share_ratio=1.0)
        positions = {"000002": 1000.0}
        cash = 10000.0
        new_pos, new_cash, record = apply_corporate_action(ca, positions, cash)
        self.assertEqual(new_pos["000002"], 2000.0)
        self.assertEqual(new_cash, 10000.0)
        self.assertEqual(record.shares_delta, 1000.0)
        self.assertEqual(record.cash_delta, 0.0)

    def test_capitalization_increases_shares_no_cash(self):
        ca = CorporateAction("2020-07-01", "000003", "capitalization", share_ratio=0.5)
        positions = {"000003": 1000.0}
        cash = 10000.0
        new_pos, new_cash, record = apply_corporate_action(ca, positions, cash)
        self.assertEqual(new_pos["000003"], 1500.0)
        self.assertEqual(new_cash, 10000.0)

    def test_split_2_for_1_doubles_shares(self):
        ca = CorporateAction("2020-03-15", "000004", "split", split_ratio=2.0)
        positions = {"000004": 500.0}
        cash = 10000.0
        new_pos, new_cash, record = apply_corporate_action(ca, positions, cash)
        self.assertEqual(new_pos["000004"], 1000.0)
        self.assertEqual(new_cash, 10000.0)

    def test_reverse_split_reduces_shares(self):
        ca = CorporateAction("2020-04-01", "000005", "reverse_split", split_ratio=0.5)
        positions = {"000005": 1000.0}
        cash = 10000.0
        new_pos, new_cash, record = apply_corporate_action(ca, positions, cash)
        self.assertEqual(new_pos["000005"], 500.0)
        self.assertEqual(new_cash, 10000.0)

    def test_rights_issue_participate_adds_shares_deducts_cash(self):
        ca = CorporateAction(
            "2020-08-01", "000006", "rights_issue",
            subscription_ratio=0.3, subscription_price=5.0, participate=True,
        )
        positions = {"000006": 1000.0}
        cash = 10000.0
        new_pos, new_cash, record = apply_corporate_action(ca, positions, cash)
        expected_new_shares = 1000 * 0.3
        expected_cost = expected_new_shares * 5.0
        self.assertEqual(new_pos["000006"], 1000.0 + expected_new_shares)
        self.assertEqual(new_cash, 10000.0 - expected_cost)

    def test_rights_issue_no_participate_no_change(self):
        ca = CorporateAction(
            "2020-08-01", "000006", "rights_issue",
            subscription_ratio=0.3, subscription_price=5.0, participate=False,
        )
        positions = {"000006": 1000.0}
        cash = 10000.0
        new_pos, new_cash, record = apply_corporate_action(ca, positions, cash)
        self.assertEqual(new_pos["000006"], 1000.0)
        self.assertEqual(new_cash, 10000.0)

    def test_no_position_skips_action(self):
        ca = CorporateAction("2020-01-15", "000007", "cash_dividend", cash_per_share=0.50)
        positions = {}
        cash = 10000.0
        new_pos, new_cash, record = apply_corporate_action(ca, positions, cash)
        self.assertEqual(new_cash, 10000.0)


class TestAccountLayerCorporateActions(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state_path = Path(self.tmpdir) / "account.json"

    def test_corporate_action_applied_before_trading(self):
        ca = CorporateAction("2020-01-02", "000001", "cash_dividend", cash_per_share=1.0)
        account = SimAccount("test-ca", 100000.0)
        account.set_corporate_actions([ca])
        # Step on ex-date — corporate action should apply first
        result = account.step(
            pd.Timestamp("2020-01-02", tz="UTC"),
            prices={"000001": 10.0},
            target_weights={"000001": 1.0},
            save_path=self.state_path,
        )
        # Cash dividend only adds cash if there are shares from a previous step.
        # On first step, positions are empty, so dividend has no effect.
        self.assertIn("2020-01-02T00:00:00+00:00::000001::cash_dividend", account.applied_corporate_actions)

    def test_same_corporate_action_applied_only_once(self):
        ca = CorporateAction("2020-01-02", "000001", "cash_dividend", cash_per_share=1.0)
        account = SimAccount("test-ca-idem", 100000.0)
        account.set_corporate_actions([ca])
        account.step(
            pd.Timestamp("2020-01-02", tz="UTC"),
            prices={"000001": 10.0},
            target_weights={"000001": 1.0},
            save_path=self.state_path,
        )
        first_applied = account.applied_corporate_actions.copy()
        # Second step same day is idempotent (returns history)
        account.step(
            pd.Timestamp("2020-01-02", tz="UTC"),
            prices={"000001": 10.0},
            target_weights={"000001": 1.0},
            save_path=self.state_path,
        )
        self.assertEqual(account.applied_corporate_actions, first_applied)

    def test_equity_continuous_through_stock_dividend(self):
        """After a 10-for-10 stock dividend, with price halving, equity should be continuous."""
        ca = CorporateAction("2020-01-02", "000001", "stock_dividend", share_ratio=1.0)
        account = SimAccount("test-continuity", 100000.0, allow_zero_cost_for_tests=True)
        account.set_corporate_actions([ca])
        # Step 1: buy shares at price 20
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 20.0},
            target_weights={"000001": 1.0},
            save_path=self.state_path,
        )
        equity_before = account.equity_history().iloc[-1]["equity"]
        # Step 2: ex-date, price halves to 10 (adjusted), corporate action applies
        account.step(
            pd.Timestamp("2020-01-02", tz="UTC"),
            prices={"000001": 10.0},
            target_weights={"000001": 1.0},
            save_path=self.state_path,
        )
        equity_after = account.equity_history().iloc[-1]["equity"]
        # Equity should be roughly continuous (small cost differences acceptable)
        self.assertAlmostEqual(equity_before, equity_after, delta=equity_before * 0.02)

    def test_load_preserves_corporate_actions(self):
        ca = CorporateAction("2020-01-02", "000001", "cash_dividend", cash_per_share=0.5)
        account = SimAccount("test-load-ca", 100000.0)
        account.set_corporate_actions([ca])
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0},
            target_weights={"000001": 1.0},
            save_path=self.state_path,
        )
        loaded = SimAccount.load(self.state_path)
        self.assertEqual(len(loaded._corporate_actions), 1)
        self.assertEqual(loaded._corporate_actions[0].event_key, ca.event_key)


if __name__ == "__main__":
    unittest.main()
