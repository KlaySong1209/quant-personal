"""Daily data ingestion foundation tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.backtest.costs import BpsCostModel
from quant.backtest.engine import run_backtest
from quant.data.adjust.calendar import CalendarColumnMap, TradingCalendar, align_close_panel_with_tradable, load_calendar
from quant.data.adjust.corporate_actions import CorporateActionColumnMap, adjust_ohlcv_for_corporate_actions, flag_implausible_unadjusted_jumps, load_corporate_actions
from quant.data.adjust.universe import UniverseColumnMap, build_universe_mask, load_universe_membership, synthetic_universe_with_delisting
from quant.data.futures_local import FuturesColumnMap, continuous_from_local_file
from quant.risk.checks import RiskConfig


def _ts(values: list[str]) -> pd.DatetimeIndex:
    return pd.to_datetime(values, utc=True)


class TestDailyDataFoundation(unittest.TestCase):
    def test_corporate_action_adjustment(self) -> None:
        raw = pd.DataFrame(
            {
                "timestamp": _ts(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "symbol": ["AAA", "AAA", "AAA"],
                "open": [100.0, 90.0, 50.0],
                "high": [100.0, 90.0, 50.0],
                "low": [100.0, 90.0, 50.0],
                "close": [100.0, 90.0, 50.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        events = pd.DataFrame(
            {
                "timestamp": _ts(["2020-01-02", "2020-01-03"]),
                "symbol": ["AAA", "AAA"],
                "event_type": ["cash_dividend", "split"],
                "amount": [10.0, 0.0],
                "ratio": [1.0, 2.0],
            }
        )
        adjusted = adjust_ohlcv_for_corporate_actions(raw, events)
        self.assertEqual(list(adjusted["raw_close"]), [100.0, 90.0, 50.0])
        self.assertEqual(list(adjusted["adjusted_close"]), [45.0, 45.0, 50.0])
        self.assertEqual(list(adjusted["close"]), [45.0, 45.0, 50.0])

    def test_implausible_jump_flagged(self) -> None:
        raw = pd.DataFrame(
            {
                "timestamp": _ts(["2020-01-01", "2020-01-02"]),
                "symbol": ["AAA", "AAA"],
                "open": [100.0, 80.0],
                "high": [100.0, 80.0],
                "low": [100.0, 80.0],
                "close": [100.0, 80.0],
                "volume": [1.0, 1.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "implausible"):
            flag_implausible_unadjusted_jumps(raw, max_log_return=0.05)

    def test_column_mapping_loaders_keep_leading_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.csv"
            pd.DataFrame(
                {"Stkcd": ["000001"], "ExDate": ["2020-01-02"], "EventKind": ["cash_dividend"], "CashDiv": [1.5]}
            ).to_csv(path, index=False)
            events = load_corporate_actions(
                path,
                CorporateActionColumnMap("Stkcd", "ExDate", "EventKind", amount="CashDiv"),
            )
        self.assertEqual(events.loc[0, "symbol"], "000001")

    def test_universe_delisting_blocks_holding(self) -> None:
        idx = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
        symbols = ["AAA", "BBB"]
        membership = synthetic_universe_with_delisting(
            symbols=symbols, start=idx[0], end=idx[-1], delisted_symbol="BBB", delist_date=idx[1]
        )
        universe = build_universe_mask(idx, symbols, membership)
        prices = pd.DataFrame(100.0, index=idx, columns=symbols)
        weights = pd.DataFrame(0.5, index=idx, columns=symbols)
        res = run_backtest(
            prices=prices,
            target_weights=weights,
            cost_model=BpsCostModel(0, 0, allow_zero_cost_for_tests=True),
            risk=RiskConfig(1.0, 1.0),
            initial_equity=1000.0,
            universe=universe,
        )
        self.assertEqual(res.weights_effective.loc[idx[2], "BBB"], 0.0)

    def test_universe_loader_and_calendar_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "membership.csv"
            pd.DataFrame({"Stkcd": ["000001"], "InDate": ["2020-01-01"], "OutDate": ["2020-01-31"]}).to_csv(path, index=False)
            membership = load_universe_membership(path, UniverseColumnMap("Stkcd", "InDate", "OutDate"))
            self.assertEqual(membership.loc[0, "symbol"], "000001")
            cal_path = Path(td) / "calendar.csv"
            pd.DataFrame({"TradingDate": ["2020-01-01", "2020-01-02", "2020-01-03"]}).to_csv(cal_path, index=False)
            calendar = load_calendar(cal_path, CalendarColumnMap("TradingDate"), exchange="TEST")
        with self.assertRaisesRegex(ValueError, "off trading calendar"):
            calendar.validate_timestamps(_ts(["2020-01-04"]))
        ohlcv = pd.DataFrame(
            {
                "timestamp": _ts(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "symbol": ["AAA", "AAA", "BBB"],
                "open": [10.0, 11.0, 20.0],
                "high": [10.0, 11.0, 20.0],
                "low": [10.0, 11.0, 20.0],
                "close": [10.0, 11.0, 20.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        panel, tradable = align_close_panel_with_tradable(ohlcv, symbols=["AAA", "BBB"], calendar=calendar)
        self.assertFalse(bool(tradable.loc[pd.Timestamp("2020-01-01", tz="UTC"), "BBB"]))
        self.assertEqual(panel.loc[pd.Timestamp("2020-01-01", tz="UTC"), "BBB"], 20.0)

    def test_futures_local_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "futures.csv"
            rows = []
            for code, start, price in [("AU2406", "2020-01-01", 100.0), ("AU2407", "2020-01-03", 110.0)]:
                for i, ts in enumerate(pd.date_range(start, periods=3, freq="B", tz="UTC")):
                    rows.append(
                        {
                            "Contract": code,
                            "TradeDate": ts.date().isoformat(),
                            "Expiry": (ts + pd.Timedelta(days=10)).date().isoformat(),
                            "OpenPx": price + i,
                            "HighPx": price + i,
                            "LowPx": price + i,
                            "ClosePx": price + i,
                            "Volume": 1000,
                        }
                    )
            pd.DataFrame(rows).to_csv(path, index=False)
            cont, meta = continuous_from_local_file(
                path,
                FuturesColumnMap("Contract", "TradeDate", "Expiry", "OpenPx", "HighPx", "LowPx", "ClosePx", "Volume"),
                continuous_symbol="AU_CONT",
                calendar_days_before_expiry=0,
            )
        self.assertEqual(set(cont["symbol"]), {"AU_CONT"})
        self.assertEqual(meta.method, "back_adjusted")

