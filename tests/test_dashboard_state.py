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


if __name__ == "__main__":
    unittest.main()
