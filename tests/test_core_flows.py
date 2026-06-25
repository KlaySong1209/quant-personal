"""Core regression tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant import app
from quant.backtest.costs import BpsCostModel
from quant.backtest.engine import run_backtest
from quant.config.loader import load_config
from quant.execution.paper import PaperBroker, PaperBrokerError
from quant.experiment.run import run_experiment
from quant.risk.checks import RiskConfig


class TestBacktestCore(unittest.TestCase):
    def test_shift_blocks_same_day_peek(self) -> None:
        idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
        prices = pd.DataFrame({"A": [100.0, 110.0, 100.0, 90.0, 99.0]}, index=idx)
        rets = prices.pct_change().fillna(0.0)
        peek = (rets > 0).astype(float) * 2 - 1
        peek.iloc[0] = 0.0
        res = run_backtest(
            prices=prices,
            target_weights=peek,
            cost_model=BpsCostModel(0, 0, allow_zero_cost_for_tests=True),
            risk=RiskConfig(1.0, 1.0),
            initial_equity=1000.0,
        )
        self.assertEqual(res.weights_effective.iloc[1, 0], 0.0)
        self.assertNotEqual(float(res.returns.iloc[2]), abs(float(rets.iloc[2, 0])))

    def test_tradable_mask_rechecked_for_hidden_leverage(self) -> None:
        idx = pd.date_range("2020-01-01", periods=3, freq="B", tz="UTC")
        prices = pd.DataFrame(100.0, index=idx, columns=["A", "B"])
        weights = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=idx)
        tradable = pd.DataFrame(True, index=idx, columns=prices.columns)
        tradable.loc[idx[2], "A"] = False
        with self.assertRaisesRegex(ValueError, "gross leverage"):
            run_backtest(
                prices=prices,
                target_weights=weights,
                cost_model=BpsCostModel(0, 0, allow_zero_cost_for_tests=True),
                risk=RiskConfig(1.0, 1.0),
                initial_equity=1000.0,
                tradable=tradable,
            )


class TestPaperBroker(unittest.TestCase):
    def test_default_rejects_short_and_overdraft(self) -> None:
        broker = PaperBroker(1000.0)
        broker.update_prices({"AAA": 100.0})
        with self.assertRaises(PaperBrokerError):
            broker.submit_target(pd.Timestamp("2020-01-01", tz="UTC"), {"AAA": -1.0})
        with self.assertRaises(PaperBrokerError):
            broker.submit_target(pd.Timestamp("2020-01-01", tz="UTC"), {"AAA": 20.0})


class TestAppAndEndToEnd(unittest.TestCase):
    def test_python_module_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "quant", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("generate example data", proc.stdout.lower())

    def test_paper_session_script_help_exposes_mode_and_missing_open_policy(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/run_paper_session.py", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--missing-open-policy", proc.stdout)
        self.assertIn("--mode", proc.stdout)

    def test_dashboard_import(self) -> None:
        import dashboard.app_streamlit as dashboard_app

        self.assertTrue(hasattr(dashboard_app, "main"))
        self.assertTrue(hasattr(dashboard_app, "STREAMLIT_INSTALL_COMMAND"))

    def test_dashboard_terminal_contract(self) -> None:
        import dashboard.app_streamlit as dashboard_app

        css = dashboard_app._terminal_css()
        self.assertIn("qp-card", css)
        self.assertIn("qp-terminal", css)
        self.assertIn("SIMULATED / PAPER -- NOT REAL", dashboard_app.PAPER_MODE_LABEL)

        snapshot = dashboard_app._paper_snapshot(
            {
                "label": "SIMULATED / PAPER -- NOT REAL",
                "final_cash": 900.0,
                "final_equity": 1000.0,
                "steps": 3,
                "ledger_balanced": True,
                "positions": {"AAA": 1.0, "BBB": 0.0, "CCC": 2.0},
                "assumptions": {"fill_price_rule": "same_day_close", "order_routing": "none"},
            }
        )
        self.assertEqual(snapshot["label"], "SIMULATED / PAPER -- NOT REAL")
        self.assertEqual(snapshot["ledger"], "BALANCED")
        self.assertEqual(snapshot["positions"], "2 / 3")
        self.assertIn("same_day_close", snapshot["assumptions"])

    def test_metric_explanations(self) -> None:
        self.assertIn("higher", app.metric_explanations()["sharpe"].lower())

    def test_equity_run_writes_required_artifacts(self) -> None:
        cfg = load_config("configs/experiments/exp_placeholder.yaml")
        with tempfile.TemporaryDirectory() as td:
            artifacts = run_experiment(cfg, results_root=Path(td))
            for name in ("config_snapshot.yaml", "metrics.json", "metadata.json", "equity_curve.parquet", "trades.parquet", "run.log"):
                self.assertTrue((artifacts.run_dir / name).exists(), name)
            snap = (artifacts.run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
            self.assertIn("corporate_actions:", snap)
            self.assertIn("universe:", snap)
            self.assertIn("calendar:", snap)
            metrics = json.loads((artifacts.run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("sharpe", metrics)
