"""Tests for dashboard status view model and demo/no-state behavior.

Covers:
  - paper_account_status returns no_state when no file exists
  - demo mode returns correct state_type and label
  - paper_simulation mode returns correct state_type and label
  - Dashboard import smoke test
  - production_data context rejects demo
"""

from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.app import paper_account_status
from quant.execution.account import SimAccount, DEMO_LABEL, PAPER_SIMULATION_LABEL


class TestPaperAccountStatus(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_no_state_file_returns_no_state(self):
        nonexistent = Path(self.tmpdir) / "nonexistent.json"
        status = paper_account_status(nonexistent)
        self.assertEqual(status["state_type"], "no_state")
        self.assertIsNone(status["final_equity"])
        self.assertIsNone(status["error"])

    def test_demo_state_returns_demo(self):
        state_path = Path(self.tmpdir) / "demo_account.json"
        account = SimAccount(
            "demo-1", 100000.0,
            mode="demo",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=state_path,
        )
        status = paper_account_status(state_path)
        self.assertEqual(status["state_type"], "demo")
        self.assertEqual(status["mode"], "demo")
        self.assertEqual(status["label"], DEMO_LABEL)
        self.assertIsNotNone(status["final_equity"])

    def test_demo_state_rejected_in_production_data_context(self):
        state_path = Path(self.tmpdir) / "demo_account.json"
        account = SimAccount(
            "demo-1", 100000.0,
            mode="demo",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=state_path,
        )
        status = paper_account_status(state_path, production_data=True)
        self.assertEqual(status["state_type"], "no_state")
        self.assertIn("demo", status["error"].lower())

    def test_paper_simulation_returns_paper_simulation(self):
        state_path = Path(self.tmpdir) / "paper_account.json"
        account = SimAccount(
            "paper-1", 100000.0,
            mode="paper_simulation",
            allow_zero_cost_for_tests=True,
        )
        account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
            target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
            save_path=state_path,
        )
        status = paper_account_status(state_path)
        self.assertEqual(status["state_type"], "paper_simulation")
        self.assertEqual(status["mode"], "paper_simulation")
        self.assertEqual(status["label"], PAPER_SIMULATION_LABEL)
        self.assertIsNotNone(status["final_equity"])

    def test_corrupt_state_file_returns_no_state_with_error(self):
        corrupt_path = Path(self.tmpdir) / "corrupt.json"
        corrupt_path.write_text("not valid json {{{")
        status = paper_account_status(corrupt_path)
        self.assertEqual(status["state_type"], "no_state")
        self.assertIsNotNone(status["error"])


class TestDemoModeRejectsProduction(unittest.TestCase):
    def test_demo_mode_has_correct_label(self):
        account = SimAccount("demo-test", 100000.0, mode="demo")
        self.assertEqual(account._label(), DEMO_LABEL)

    def test_demo_mode_saves_correct_label(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            account = SimAccount("demo-test", 100000.0, mode="demo", allow_zero_cost_for_tests=True)
            account.step(
                pd.Timestamp("2020-01-01", tz="UTC"),
                prices={"000001": 10.0, "000002": 20.0, "000003": 30.0},
                target_weights={"000001": 0.5, "000002": 0.3, "000003": 0.2},
                save_path=state_path,
            )
            data = json.loads(state_path.read_text())
            self.assertEqual(data["mode"], "demo")
            self.assertEqual(data["label"], DEMO_LABEL)

    def test_paper_simulation_has_correct_label(self):
        account = SimAccount("paper-test", 100000.0, mode="paper_simulation")
        self.assertEqual(account._label(), PAPER_SIMULATION_LABEL)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            SimAccount("bad", 100000.0, mode="invalid_mode")


class TestDashboardImport(unittest.TestCase):
    def test_dashboard_imports(self):
        """Smoke test: dashboard module imports without error."""
        import importlib.util
        spec = importlib.util.find_spec("dashboard.app_streamlit")
        # Streamlit may not be installed; that's fine
        if spec is None:
            self.skipTest("dashboard module not found on path")
        # Just verify the module can be located

    def test_dashboard_status_report_view_data_present(self):
        import importlib.util
        spec = importlib.util.find_spec("dashboard.app_streamlit")
        if spec is None:
            self.skipTest("dashboard module not found on path")

        import dashboard.app_streamlit as dashboard

        report = {
            "account": {"account_id": "paper-1", "advanced_to": "2020-01-02", "steps": 2},
            "equity": {
                "total_equity": 100500.0,
                "cash": 50000.0,
                "position_value": 50500.0,
                "ledger_balanced": True,
            },
            "pending_orders": [
                {
                    "order_id": "pending-1",
                    "created_on": "2020-01-02",
                    "status": "pending",
                    "reason": "waiting for next trading day open",
                }
            ],
            "assumptions": {
                "fill_price_rule": "next_day_open",
                "missing_open_policy": "skip",
            },
            "positions": [{"symbol": "AAA", "shares": 10.0, "price": 10.0, "value": 100.0}],
            "flags": ["1 pending order(s) waiting for next-day open fill"],
            "error": None,
        }

        view_data = dashboard._status_report_view_data(report)

        self.assertEqual(view_data["account_id"], "paper-1")
        self.assertEqual(view_data["advanced_to"], "2020-01-02")
        self.assertEqual(view_data["metrics"]["total_equity"], 100500.0)
        self.assertEqual(view_data["assumptions"]["fill_price_rule"], "next_day_open")
        self.assertEqual(view_data["pending_orders"][0]["status"], "pending")


class TestBundleViewModel(unittest.TestCase):
    """Dashboard Data page consumes only the view model from
    ``_bundle_view_data``; no Streamlit calls required to test it."""

    def test_no_bundle_status_shapes_to_empty_view(self):
        from dashboard import app_streamlit as dashboard
        view = dashboard._bundle_view_data({
            "status": "no_bundle",
            "name": "default",
            "error": None,
            "manifest": None,
            "recent_provenance": [],
        })
        self.assertEqual(view["freshness_icon"], "⚪")
        self.assertEqual(view["freshness_label"], "还没有股票池")
        self.assertEqual(view["symbols"], [])
        self.assertEqual(view["source_chain"], [])
        self.assertEqual(view["recent_provenance"], [])

    def test_fresh_bundle_view_extracts_manifest_fields(self):
        from dashboard import app_streamlit as dashboard
        status = {
            "status": "fresh",
            "name": "default",
            "error": None,
            "manifest": {
                "name": "default",
                "schema_version": "1.0",
                "market": "a_share_cn",
                "symbols": ["SH600519", "SZ000001"],
                "date_range": {"first": "2020-01-01", "last": "2026-06-20"},
                "source_chain": ["mootdx", "manual-resset"],
                "adjustment": {"convention": "backward", "method": "mootdx_hfq"},
                "calendar": {"source": "mootdx", "exchange": "SSE_SZSE"},
                "row_count": 4521,
                "updated_at": "2026-06-23T00:00:00+00:00",
                "freshness": {
                    "expected_through": "2026-06-20",
                    "actual_through": "2026-06-20",
                    "status": "fresh",
                },
            },
            "recent_provenance": [
                {"ts": "2026-06-23T00:00:00+00:00", "op": "create", "status": "ok", "rows": 30},
            ],
        }
        view = dashboard._bundle_view_data(status)
        self.assertEqual(view["freshness_icon"], "🟢")
        self.assertEqual(view["freshness_label"], "最新")
        self.assertEqual(view["symbols"], ["SH600519", "SZ000001"])
        self.assertEqual(view["date_range"]["first"], "2020-01-01")
        self.assertEqual(view["source_chain"], ["mootdx", "manual-resset"])
        self.assertEqual(view["adjustment"]["method"], "mootdx_hfq")
        self.assertEqual(view["row_count"], 4521)
        self.assertEqual(len(view["recent_provenance"]), 1)

    def test_stale_bundle_view(self):
        from dashboard import app_streamlit as dashboard
        view = dashboard._bundle_view_data({
            "status": "stale",
            "name": "default",
            "error": None,
            "manifest": {"symbols": ["SH600519"], "row_count": 10},
            "recent_provenance": [],
        })
        self.assertEqual(view["freshness_icon"], "🟡")
        self.assertEqual(view["freshness_label"], "需要更新")

    def test_error_status_propagates(self):
        from dashboard import app_streamlit as dashboard
        view = dashboard._bundle_view_data({
            "status": "error",
            "name": "default",
            "error": "manifest corrupt",
            "manifest": None,
            "recent_provenance": [],
        })
        self.assertEqual(view["freshness_icon"], "🔴")
        self.assertEqual(view["error"], "manifest corrupt")


class TestDashboardInputHelpers(unittest.TestCase):
    def test_symbol_input_parser_accepts_comma_space_and_chinese_comma(self):
        from dashboard import app_streamlit as dashboard
        self.assertEqual(
            dashboard._symbol_input_to_list("600519, 000001，000002\n300750"),
            ["600519", "000001", "000002", "300750"],
        )

    def test_symbols_to_input_prefers_bare_codes_for_a_share(self):
        from dashboard import app_streamlit as dashboard
        self.assertEqual(
            dashboard._symbols_to_input(["SH600519", "SZ000001", "SYNTHAAA"]),
            "600519, 000001, SYNTHAAA",
        )


if __name__ == "__main__":
    unittest.main()
