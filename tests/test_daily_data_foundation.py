"""Daily data ingestion foundation tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd

from quant.config.loader import load_config
from quant.backtest.costs import BpsCostModel
from quant.backtest.engine import run_backtest
from quant.data.adjust.calendar import CalendarColumnMap, TradingCalendar, align_close_panel_with_tradable, load_calendar
from quant.data.adjust.corporate_actions import CorporateActionColumnMap, adjust_ohlcv_for_corporate_actions, flag_implausible_unadjusted_jumps, load_corporate_actions
from quant.data.adjust.universe import UniverseColumnMap, build_universe_mask, load_universe_membership, synthetic_universe_with_delisting
from quant.data.futures_local import FuturesColumnMap, continuous_from_local_file
from quant.data.local import ingest_local_file, load_column_mapping, write_synthetic_local_export
from quant.execution.account import SimAccount
from quant.experiment.run import run_experiment
from quant.risk.checks import RiskConfig


def _ts(values: list[str]) -> pd.DatetimeIndex:
    return pd.to_datetime(values, utc=True)


class TestDailyDataFoundation(unittest.TestCase):
    def test_local_file_mapping_loads_and_persists_resset_shaped_export(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mapping_path = root / "mapping.yaml"
            mapping_path.write_text(
                "\n".join(
                    [
                        "columns:",
                        "  timestamp: Trddt",
                        "  symbol: Stkcd",
                        "  open: Opnprc",
                        "  high: Hiprc",
                        "  low: Loprc",
                        "  close: Clsprc",
                        "  volume: Dnshrtrd",
                        "  adjusted_close: AdjClsprc",
                    ]
                ),
                encoding="utf-8",
            )
            export_path = root / "resset.csv"
            write_synthetic_local_export(
                export_path,
                symbols=["000001", "000002", "000003"],
                start="2020-01-01",
                end="2020-01-03",
                mapping=load_column_mapping(mapping_path),
                seed=7,
            )

            result = ingest_local_file(
                export_path,
                mapping=load_column_mapping(mapping_path),
                symbols=["000001", "000002", "000003"],
                output_dir=root / "processed",
                adjustment_convention="backward",
                has_adjustment_factor=False,
            )

            self.assertTrue(result.processed_path.exists())
            self.assertEqual(set(result.ohlcv["symbol"]), {"000001", "000002", "000003"})
            self.assertEqual(result.metadata["adjustment"]["method"], "provided_adjusted_close")
            self.assertEqual(result.metadata["adjustment"]["declarations"]["adjustment_convention"], "backward")
            self.assertEqual(result.metadata["universe"]["symbols"], ["000001", "000002", "000003"])

    def test_adjustment_declarations_required_and_factor_path_applies_raw_times_factor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "factor.csv"
            pd.DataFrame(
                [
                    ["2020-01-01", "AAA", 10.0, 11.0, 9.0, 10.0, 1000, 2.0],
                    ["2020-01-01", "BBB", 20.0, 22.0, 18.0, 20.0, 1000, 3.0],
                    ["2020-01-01", "CCC", 30.0, 33.0, 27.0, 30.0, 1000, 4.0],
                ],
                columns=["date", "code", "open", "high", "low", "close", "volume", "factor"],
            ).to_csv(path, index=False)
            mapping = {
                "timestamp": "date",
                "symbol": "code",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "adjustment_factor": "factor",
            }
            with self.assertRaisesRegex(ValueError, "adjustment_convention"):
                ingest_local_file(path, mapping=mapping, symbols=["AAA", "BBB", "CCC"], output_dir=root / "bad")

            result = ingest_local_file(
                path,
                mapping=mapping,
                symbols=["AAA", "BBB", "CCC"],
                output_dir=root / "ok",
                adjustment_convention="backward",
                has_adjustment_factor=True,
            )
            closes = result.ohlcv.sort_values("symbol")["close"].tolist()
            self.assertEqual(closes, [20.0, 60.0, 120.0])
            self.assertEqual(result.metadata["adjustment"]["method"], "provided_adjustment_factor")
            self.assertTrue(result.metadata["adjustment"]["declarations"]["has_adjustment_factor"])

    def test_dividend_adjustment_requires_ex_date_and_tax_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "dividend.csv"
            pd.DataFrame(
                [
                    ["2020-01-01", "AAA", 10.0, 10.0, 10.0, 10.0, 1000, 0.1],
                    ["2020-01-01", "BBB", 20.0, 20.0, 20.0, 20.0, 1000, 0.0],
                    ["2020-01-01", "CCC", 30.0, 30.0, 30.0, 30.0, 1000, 0.0],
                ],
                columns=["date", "code", "open", "high", "low", "close", "volume", "cash_div"],
            ).to_csv(path, index=False)
            mapping = {"timestamp": "date", "symbol": "code", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "dividend": "cash_div"}
            with self.assertRaisesRegex(ValueError, "ex-dividend"):
                ingest_local_file(
                    path,
                    mapping=mapping,
                    symbols=["AAA", "BBB", "CCC"],
                    output_dir=root / "processed",
                    adjustment_convention="backward",
                )

    def test_local_file_mapping_fails_clearly_and_enforces_small_universe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "bad.csv"
            pd.DataFrame(
                {
                    "Trddt": ["2020-01-01T00:00:00Z"],
                    "Stkcd": ["000001"],
                    "Opnprc": [10.0],
                    "Hiprc": [10.0],
                    "Loprc": [10.0],
                    "Clsprc": [10.0],
                    "Dnshrtrd": [100],
                }
            ).to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing mapped column"):
                ingest_local_file(
                    csv_path,
                    mapping={"timestamp": "NoSuchColumn", "symbol": "Stkcd", "open": "Opnprc", "high": "Hiprc", "low": "Loprc", "close": "Clsprc", "volume": "Dnshrtrd"},
                    symbols=["000001", "000002", "000003"],
                    output_dir=root / "processed",
                )
            with self.assertRaisesRegex(ValueError, "3-5 symbols"):
                ingest_local_file(
                    csv_path,
                    mapping={"timestamp": "Trddt", "symbol": "Stkcd", "open": "Opnprc", "high": "Hiprc", "low": "Loprc", "close": "Clsprc", "volume": "Dnshrtrd"},
                    symbols=["000001", "000002"],
                    output_dir=root / "processed",
                )

    def test_local_file_unadjusted_jump_and_calendar_gaps_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_path = root / "raw.csv"
            rows = [
                ["2020-01-01", "AAA", 100.0, 100.0, 100.0, 100.0, 1000],
                ["2020-01-02", "AAA", 50.0, 50.0, 50.0, 50.0, 1000],
                ["2020-01-01", "BBB", 20.0, 20.0, 20.0, 20.0, 1000],
                ["2020-01-02", "BBB", 21.0, 21.0, 21.0, 21.0, 1000],
                ["2020-01-01", "CCC", 30.0, 30.0, 30.0, 30.0, 1000],
            ]
            pd.DataFrame(rows, columns=["date", "code", "open", "high", "low", "close", "vol"]).to_csv(csv_path, index=False)
            result = ingest_local_file(
                csv_path,
                mapping={"timestamp": "date", "symbol": "code", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "vol"},
                symbols=["AAA", "BBB", "CCC"],
                output_dir=root / "processed",
                calendar=TradingCalendar.synthetic("2020-01-01", "2020-01-02"),
                max_unadjusted_log_return=0.05,
            )
            self.assertEqual(result.metadata["adjustment"]["method"], "raw_unadjusted")
            self.assertGreater(len(result.metadata["adjustment"]["implausible_jumps"]), 0)
            self.assertGreater(result.metadata["alignment"]["filled_gap_count"], 0)

            saturday_path = root / "saturday.csv"
            pd.DataFrame(
                [["2020-01-04", "AAA", 1, 1, 1, 1, 1], ["2020-01-04", "BBB", 1, 1, 1, 1, 1], ["2020-01-04", "CCC", 1, 1, 1, 1, 1]],
                columns=["date", "code", "open", "high", "low", "close", "vol"],
            ).to_csv(saturday_path, index=False)
            with self.assertRaisesRegex(ValueError, "off trading calendar"):
                ingest_local_file(
                    saturday_path,
                    mapping={"timestamp": "date", "symbol": "code", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "vol"},
                    symbols=["AAA", "BBB", "CCC"],
                    output_dir=root / "processed",
                    calendar=TradingCalendar.synthetic("2020-01-01", "2020-01-06"),
                )

    def test_sim_account_round_trip_idempotent_step_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "account.json"
            account = SimAccount(account_id="paper-test", starting_cash=1000.0)
            account.step(
                pd.Timestamp("2020-01-01", tz="UTC"),
                prices={"AAA": 100.0, "BBB": 50.0, "CCC": 25.0},
                target_weights={"AAA": 0.5, "BBB": 0.25, "CCC": 0.25},
                save_path=state_path,
            )
            ledger_events = len(account.broker.ledger())
            account.step(
                pd.Timestamp("2020-01-01", tz="UTC"),
                prices={"AAA": 100.0, "BBB": 50.0, "CCC": 25.0},
                target_weights={"AAA": 0.5, "BBB": 0.25, "CCC": 0.25},
                save_path=state_path,
            )
            self.assertEqual(len(account.broker.ledger()), ledger_events)
            loaded = SimAccount.load(state_path)
            self.assertEqual(loaded.to_dict(), account.to_dict())
            loaded.step(
                pd.Timestamp("2020-01-02", tz="UTC"),
                prices={"AAA": 110.0, "BBB": 50.0, "CCC": 25.0},
                target_weights={"AAA": 0.0, "BBB": 0.5, "CCC": 0.5},
                save_path=state_path,
            )
            self.assertEqual(len(loaded.equity_history()), 2)
            last = loaded.equity_history().iloc[-1]
            self.assertAlmostEqual(float(last["equity"]), float(last["cash"] + last["position_value"]))

    def test_sim_account_rejects_non_finite_saved_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "bad_account.json"
            state_path.write_text(
                json.dumps(
                    {
                        "account_id": "bad",
                        "starting_cash": 1000.0,
                        "allow_short": False,
                        "allow_margin": False,
                        "max_gross_leverage": 1.0,
                        "broker": {
                            "starting_cash": 1000.0,
                            "allow_short": False,
                            "allow_margin": False,
                            "max_gross_leverage": 1.0,
                            "cash": float("nan"),
                            "positions": {"AAA": 1.0},
                            "last_prices": {"AAA": 100.0},
                            "events": [],
                        },
                        "history": [
                            {
                                "timestamp": "2020-01-01T00:00:00+00:00",
                                "cash": float("nan"),
                                "position_value": 100.0,
                                "equity": float("nan"),
                                "positions": {"AAA": 1.0},
                                "prices": {"AAA": 100.0},
                            }
                        ],
                        "completed_steps": ["2020-01-01T00:00:00+00:00"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                SimAccount.load(state_path)

    def test_ingest_to_paper_session_end_to_end(self) -> None:
        from quant.app import run_paper_session

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mapping = {"timestamp": "date", "symbol": "code", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "adjusted_close": "adj_close"}
            export_path = root / "export.csv"
            write_synthetic_local_export(
                export_path,
                symbols=["AAA", "BBB", "CCC"],
                start="2020-01-01",
                end="2020-01-07",
                mapping=mapping,
                seed=42,
            )
            ingest = ingest_local_file(
                export_path,
                mapping=mapping,
                symbols=["AAA", "BBB", "CCC"],
                output_dir=root / "processed",
                adjustment_convention="backward",
                has_adjustment_factor=False,
            )
            first = run_paper_session(
                data_path=ingest.processed_path,
                symbols=["AAA", "BBB", "CCC"],
                state_path=root / "state" / "account.json",
                output_dir=root / "paper",
                starting_cash=1000.0,
            )
            second = run_paper_session(
                data_path=ingest.processed_path,
                symbols=["AAA", "BBB", "CCC"],
                state_path=root / "state" / "account.json",
                output_dir=root / "paper",
                starting_cash=1000.0,
            )
            self.assertTrue(first["state_path"].exists())
            self.assertTrue(first["equity_history_path"].exists())
            self.assertEqual(first["final_equity"], second["final_equity"])
            self.assertEqual(first["steps"], second["steps"])

    def test_paper_session_rejects_missing_or_wrong_sized_symbol_universe(self) -> None:
        from quant.app import run_paper_session

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mapping = {"timestamp": "date", "symbol": "code", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "adjusted_close": "adj_close"}
            export_path = root / "export.csv"
            write_synthetic_local_export(
                export_path,
                symbols=["AAA", "BBB", "CCC"],
                start="2020-01-01",
                end="2020-01-03",
                mapping=mapping,
                seed=42,
            )
            ingest = ingest_local_file(
                export_path,
                mapping=mapping,
                symbols=["AAA", "BBB", "CCC"],
                output_dir=root / "processed",
                adjustment_convention="backward",
                has_adjustment_factor=False,
            )
            with self.assertRaisesRegex(ValueError, "3-5 symbols"):
                run_paper_session(
                    data_path=ingest.processed_path,
                    symbols=["AAA", "BBB"],
                    state_path=root / "state" / "bad.json",
                    output_dir=root / "paper",
                )
            with self.assertRaisesRegex(ValueError, "missing requested symbol"):
                run_paper_session(
                    data_path=ingest.processed_path,
                    symbols=["AAA", "BBB", "ZZZ"],
                    state_path=root / "state" / "bad.json",
                    output_dir=root / "paper",
                )

    def test_local_file_config_path_persists_processed_data_and_metadata(self) -> None:
        from quant.data.pipeline import load_and_validate

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mapping_path = root / "mapping.yaml"
            mapping_path.write_text(
                "\n".join(
                    [
                        "columns:",
                        "  timestamp: date",
                        "  symbol: code",
                        "  open: open",
                        "  high: high",
                        "  low: low",
                        "  close: close",
                        "  volume: volume",
                        "  adjusted_close: adj_close",
                    ]
                ),
                encoding="utf-8",
            )
            export_path = root / "export.csv"
            write_synthetic_local_export(
                export_path,
                symbols=["AAA", "BBB", "CCC"],
                start="2020-01-01",
                end="2020-01-03",
                mapping=load_column_mapping(mapping_path),
                seed=42,
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "run:",
                        "  name: local_file_test",
                        "  seed: 42",
                        "data:",
                        "  source: local_file",
                        f"  local_path: {export_path}",
                        f"  column_mapping_path: {mapping_path}",
                        f"  processed_output_dir: {root / 'processed'}",
                        "  symbols: [AAA, BBB, CCC]",
                        "  start: '2020-01-01'",
                        "  end: '2020-01-03'",
                        "  corporate_actions:",
                        "    adjustment_convention: backward",
                        "    has_adjustment_factor: false",
                        "strategy:",
                        "  name: placeholder",
                        "  params:",
                        "    mode: equal_weight",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_config(config_path)
            loaded = load_and_validate(cfg)
            self.assertEqual(len(loaded), 9)
            self.assertTrue((root / "processed" / "local_daily_metadata.json").exists())
            self.assertTrue(
                (root / "processed" / "local_daily_ohlcv.parquet").exists()
                or (root / "processed" / "local_daily_ohlcv.csv").exists()
            )
            artifacts = run_experiment(cfg, results_root=root / "results")
            snapshot = (artifacts.run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
            self.assertIn("local_ingestion:", snapshot)
            self.assertIn("provided_adjusted_close", snapshot)

    def test_calendar_file_enforced_and_synthetic_forbidden_for_production_data(self) -> None:
        from quant.data.adjust.calendar import build_trading_calendar

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            calendar_path = root / "calendar.csv"
            pd.DataFrame({"TradingDate": ["2020-01-02"], "IsOpen": [1]}).to_csv(calendar_path, index=False)
            calendar = build_trading_calendar(
                mode="file",
                file=calendar_path,
                column_mapping={"date": "TradingDate", "is_open": "IsOpen"},
                exchange="SSE",
            )
            calendar.validate_timestamps(pd.to_datetime(["2020-01-02"], utc=True))
            with self.assertRaisesRegex(ValueError, "off trading calendar"):
                calendar.validate_timestamps(pd.to_datetime(["2020-01-03"], utc=True))
            with self.assertRaisesRegex(FileNotFoundError, "calendar file"):
                build_trading_calendar(mode="file", file=root / "missing.csv", column_mapping={"date": "TradingDate"})
            with self.assertRaisesRegex(ValueError, "synthetic calendar"):
                build_trading_calendar(mode="synthetic", start="2020-01-01", end="2020-01-03", production_data=True)

    def test_manual_quote_source_advances_account_and_realtime_is_reserved(self) -> None:
        from quant.data.quotes import ManualQuoteSource, RealtimeQuoteSource

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            quote_path = root / "quotes.csv"
            pd.DataFrame(
                [
                    ["2020-01-01", "AAA", 10.0],
                    ["2020-01-01", "BBB", 20.0],
                    ["2020-01-01", "CCC", 30.0],
                    ["2020-01-02", "AAA", 11.0],
                    ["2020-01-02", "BBB", 21.0],
                    ["2020-01-02", "CCC", 31.0],
                ],
                columns=["date", "symbol", "close"],
            ).to_csv(quote_path, index=False)
            quotes = ManualQuoteSource(quote_path, column_mapping={"timestamp": "date", "symbol": "symbol", "close": "close"})
            account = SimAccount(account_id="manual-quotes", starting_cash=1000.0, allow_zero_cost_for_tests=True)
            row = account.step(
                pd.Timestamp("2020-01-02", tz="UTC"),
                prices=quotes.latest(["AAA", "BBB", "CCC"], as_of="2020-01-02"),
                target_weights={"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3},
            )
            self.assertIn("SIMULATED / PAPER -- NOT REAL", account.to_dict()["label"])
            self.assertAlmostEqual(float(row["equity"]), float(row["cash"] + row["position_value"]))
            with self.assertRaises(NotImplementedError):
                RealtimeQuoteSource().latest(["AAA", "BBB", "CCC"])

    def test_sim_account_applies_nonzero_costs_and_labels_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = Path(td) / "account.json"
            with self.assertRaisesRegex(ValueError, "zero-cost"):
                SimAccount(
                    account_id="zero-cost",
                    starting_cash=1000.0,
                    commission_bps=0.0,
                    stamp_duty_bps=0.0,
                    slippage_bps=0.0,
                )
            account = SimAccount(account_id="costed", starting_cash=1000.0, commission_bps=1.0, stamp_duty_bps=10.0, slippage_bps=1.0)
            first = account.step(
                pd.Timestamp("2020-01-01", tz="UTC"),
                prices={"AAA": 100.0, "BBB": 100.0, "CCC": 100.0},
                target_weights={"AAA": 0.3, "BBB": 0.3, "CCC": 0.0},
            )
            second = account.step(
                pd.Timestamp("2020-01-02", tz="UTC"),
                prices={"AAA": 100.0, "BBB": 100.0, "CCC": 100.0},
                target_weights={"AAA": 0.0, "BBB": 0.3, "CCC": 0.3},
                save_path=state_path,
            )
            self.assertGreater(first["costs"]["total"], 0.0)
            self.assertEqual(first["costs"]["stamp_duty"], 0.0)
            self.assertGreater(second["costs"]["stamp_duty"], 0.0)
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "paper_simulation")
            self.assertIn("SIMULATED / PAPER -- NOT REAL", saved["label"])
            self.assertEqual(saved["assumptions"]["fill_price_rule"], "same_day_close")

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
