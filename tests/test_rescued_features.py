"""Regression coverage for selected features rescued from old research branches."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.backtest.costs import AShareCostModel
from quant.execution.account import SimAccount


class TestDailyQualityReport(unittest.TestCase):
    def test_reports_price_shape_duplicates_staleness_and_calendar_requirements(self) -> None:
        from quant.data.quality import run_quality_checks

        rows = [
            ["2020-01-01", "AAA", 10.0, 11.0, 9.0, 10.0],
            ["2020-01-01", "AAA", 10.0, 9.0, 9.5, 20.0],
            ["2020-01-03", "BBB", -1.0, 2.0, 1.0, 1.5],
        ]
        df = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close"])

        report = run_quality_checks(
            df,
            symbols=["AAA", "BBB", "CCC"],
            stale_after_days=1,
            as_of=pd.Timestamp("2020-01-06"),
            max_abs_log_return=0.05,
        )

        codes = {issue.code for issue in report.issues}
        self.assertFalse(report.ok)
        self.assertIn("missing_symbol", codes)
        self.assertIn("duplicate_date", codes)
        self.assertIn("zero_or_negative_price", codes)
        self.assertIn("ohlc_inconsistent", codes)
        self.assertIn("abnormal_return", codes)
        self.assertIn("stale_data", codes)

        prod_report = run_quality_checks(df, production_data=True)
        self.assertIn("missing_calendar", {issue.code for issue in prod_report.issues})


class TestSignalConsistency(unittest.TestCase):
    def test_shared_signal_entry_and_adjustment_mismatch_guard(self) -> None:
        from quant.signal import (
            AdjustmentMismatchError,
            SignalConfig,
            assert_adjustment_consistent,
            generate_target_weights,
        )

        idx = pd.date_range("2020-01-01", periods=2, freq="B", tz="UTC")
        prices = pd.DataFrame(10.0, index=idx, columns=["AAA", "BBB", "CCC"])

        weights = generate_target_weights(prices, SignalConfig())

        self.assertEqual(list(weights.columns), ["AAA", "BBB", "CCC"])
        self.assertAlmostEqual(float(weights.iloc[-1].sum()), 1.0)
        assert_adjustment_consistent("none", "none")
        with self.assertRaises(AdjustmentMismatchError):
            assert_adjustment_consistent("qfq", "none")


class TestAShareExecutionRules(unittest.TestCase):
    def test_lot_rounding_tick_alignment_and_rejection_reasons_are_recorded(self) -> None:
        account = SimAccount(
            account_id="ashare",
            starting_cash=10_000.0,
            allow_zero_cost_for_tests=True,
            execution_adjustment="none",
            lot_size=100,
            tick_size=0.01,
        )

        row = account.step(
            pd.Timestamp("2020-01-01", tz="UTC"),
            prices={"AAA": 10.006, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 1.0, "BBB": 0.0, "CCC": 0.0},
        )

        self.assertEqual(row["positions"]["AAA"], 900.0)
        filled = [r for r in row["fill_results"] if r["status"] == "filled"]
        self.assertEqual(filled[0]["price"], 10.01)

        rejected = account.step(
            pd.Timestamp("2020-01-02", tz="UTC"),
            prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            target_weights={"AAA": 0.0, "BBB": 1.0, "CCC": 0.0},
            can_buy={"AAA": True, "BBB": False, "CCC": True},
        )
        reasons = {r["reason"] for r in rejected["fill_results"] if r["status"] == "rejected"}
        self.assertIn("limit_up_no_buy", reasons)

        with self.assertRaisesRegex(ValueError, "qfq"):
            SimAccount(account_id="bad", starting_cash=10_000.0, execution_adjustment="qfq")


class TestAkshareFallbackFetcher(unittest.TestCase):
    def test_auto_datasource_falls_back_and_writes_canonical_raw_file(self) -> None:
        from quant.data.fetchers.akshare_daily import AkshareFetcher

        class FakeAkshare:
            def stock_zh_a_daily(self, **kwargs):
                raise RuntimeError("sina unavailable")

            def stock_zh_a_hist_tx(self, **kwargs):
                return pd.DataFrame(
                    [
                        {
                            "date": "2020-01-02",
                            "open": 10.0,
                            "high": 11.0,
                            "low": 9.0,
                            "close": 10.5,
                            "amount": 1000.0,
                        }
                    ]
                )

            def stock_zh_a_hist(self, **kwargs):
                raise RuntimeError("eastmoney should not be needed")

        with tempfile.TemporaryDirectory() as td:
            result = AkshareFetcher(akshare_module=FakeAkshare()).fetch_daily_ohlcv(
                ["SZ000001"],
                raw_dir=Path(td),
                start=pd.Timestamp("2020-01-01", tz="UTC"),
                end=pd.Timestamp("2020-01-31", tz="UTC"),
            )

            self.assertEqual(result.status, "ok")
            self.assertEqual(result.source, "akshare/tencent")
            self.assertEqual(result.symbols_ok, ["SZ000001"])
            df = pd.read_parquet(result.raw_paths[0])
            self.assertEqual(df.loc[0, "symbol"], "SZ000001")
            self.assertEqual(float(df.loc[0, "close"]), 10.5)


class TestAShareBacktestParity(unittest.TestCase):
    def test_backtest_uses_same_lot_execution_as_paper_account(self) -> None:
        from quant.backtest.engine import run_backtest

        idx = pd.date_range("2020-01-01", periods=2, freq="B", tz="UTC")
        prices = pd.DataFrame(10.006, index=idx, columns=["AAA", "BBB", "CCC"])
        weights = pd.DataFrame(
            [{"AAA": 1.0, "BBB": 0.0, "CCC": 0.0}, {"AAA": 1.0, "BBB": 0.0, "CCC": 0.0}],
            index=idx,
        )

        result = run_backtest(
            prices=prices,
            target_weights=weights,
            cost_model=AShareCostModel(allow_zero_cost_for_tests=True),
            risk=None,
            initial_equity=10_000.0,
            fill_price_rule="same_day_close",
            execution_adjustment="none",
        )

        self.assertEqual(float(result.trades_long.iloc[0]["shares_delta"]), 900.0)
        self.assertEqual(float(result.trades_long.iloc[0]["price"]), 10.01)


class TestArtifactHelpers(unittest.TestCase):
    def test_dataframe_artifact_roundtrip_and_missing_run_artifacts(self) -> None:
        from quant.artifacts import load_equity_curve, load_trades, read_dataframe, write_dataframe

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            frame = pd.DataFrame({"timestamp": ["2020-01-01"], "equity": [100.0]})
            written = write_dataframe(frame, root / "equity_history")
            loaded = read_dataframe(written)
            self.assertEqual(float(loaded.loc[0, "equity"]), 100.0)

            run_dir = root / "run"
            run_dir.mkdir()
            frame.to_csv(run_dir / "equity_curve.csv", index=False)
            pd.DataFrame({"symbol": ["AAA"]}).to_csv(run_dir / "trades.csv", index=False)
            self.assertEqual(float(load_equity_curve(run_dir).iloc[0]["equity"]), 100.0)
            self.assertEqual(load_trades(run_dir).loc[0, "symbol"], "AAA")
