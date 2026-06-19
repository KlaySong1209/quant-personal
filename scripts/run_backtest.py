"""Run a backtest from a YAML experiment config.

Thin orchestration only — all logic lives under ``quant``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quant.app import format_metrics_plain, run_backtest_experiment


def main() -> None:
    p = argparse.ArgumentParser(description="Run a quant-personal backtest.")
    p.add_argument("--config", required=True, type=Path, help="Path to experiment YAML.")
    args = p.parse_args()

    artifacts = run_backtest_experiment(args.config)

    print(f"\nrun_id: {artifacts.run_id}")
    print(f"run_dir: {artifacts.run_dir}")
    print("metrics:")
    print(format_metrics_plain(artifacts.metrics))


if __name__ == "__main__":
    main()

