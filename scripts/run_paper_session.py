"""Thin CLI wrapper for a local paper-account session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant.app import format_paper_session_plain, run_manual_quote_step, run_paper_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or resume a simulated paper account.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="Processed OHLCV parquet/csv from local ingestion.")
    source.add_argument("--manual-quotes", type=Path, help="Manual quote CSV with date,symbol,close rows.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Small 3-5 symbol universe.")
    parser.add_argument("--state", type=Path, default=Path("state/paper_account.json"), help="Account state JSON path.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_session"), help="Equity history and ledger output folder.")
    parser.add_argument("--starting-cash", type=float, default=100000.0)
    parser.add_argument("--as-of", help="Quote date to use with --manual-quotes.")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--fill-price-rule", choices=["same_day_close", "next_day_open"], default="same_day_close")
    args = parser.parse_args(argv)
    if args.manual_quotes:
        summary = run_manual_quote_step(
            quote_path=args.manual_quotes,
            symbols=args.symbols,
            state_path=args.state,
            output_dir=args.output_dir,
            starting_cash=args.starting_cash,
            as_of=args.as_of,
            commission_bps=args.commission_bps,
            stamp_duty_bps=args.stamp_duty_bps,
            slippage_bps=args.slippage_bps,
            fill_price_rule=args.fill_price_rule,
        )
    else:
        summary = run_paper_session(
            data_path=args.data,
            symbols=args.symbols,
            state_path=args.state,
            output_dir=args.output_dir,
            starting_cash=args.starting_cash,
            commission_bps=args.commission_bps,
            stamp_duty_bps=args.stamp_duty_bps,
            slippage_bps=args.slippage_bps,
            fill_price_rule=args.fill_price_rule,
        )
    print(format_paper_session_plain(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
