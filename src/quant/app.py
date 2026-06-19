"""Local app orchestration for CLI menu and dashboard."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from quant.config.loader import load_config
from quant.data.pipeline import build_source
from quant.experiment.run import RESULTS_ROOT, RunArtifacts, run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
EXAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "example"
STREAMLIT_INSTALL_COMMAND = "pip3 install -e .[dashboard]"


def metric_explanations() -> dict[str, str]:
    return {
        "total_return": "Total return: total compounded gain or loss over the run.",
        "annualized_return": "Annualized return: average yearly return implied by the run.",
        "annualized_volatility": "Volatility: how much returns fluctuate; lower is steadier.",
        "sharpe": "Sharpe: risk-adjusted return; higher is better, below 0 means it lost money after risk.",
        "max_drawdown": "Max drawdown: worst peak-to-trough loss; closer to 0 is better.",
        "turnover": "Turnover: how much the portfolio traded; higher usually means higher costs.",
        "excess_return": "Excess return: compounded return above the benchmark.",
        "tracking_error": "Tracking error: how differently the strategy moved versus the benchmark.",
        "information_ratio": "Information ratio: benchmark-adjusted return per unit of active risk.",
        "beta": "Beta: sensitivity to the benchmark; 1 moves roughly with it.",
    }


def list_experiment_configs(config_dir: Path = CONFIG_DIR) -> list[Path]:
    return sorted(config_dir.glob("*.yaml"))


def generate_example_data(config_path: str | Path = CONFIG_DIR / "exp_placeholder.yaml") -> list[Path]:
    cfg = load_config(config_path)
    df = build_source(cfg).load()
    EXAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for sym, sub in df.groupby("symbol"):
        path = EXAMPLE_DATA_DIR / f"{sym}.csv"
        sub.to_csv(path, index=False)
        paths.append(path)
    return paths


def run_backtest_experiment(config_path: str | Path) -> RunArtifacts:
    return run_experiment(load_config(config_path))


def list_runs(results_root: Path = RESULTS_ROOT) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted([p for p in results_root.iterdir() if p.is_dir()], reverse=True)


def latest_run(results_root: Path = RESULTS_ROOT) -> Path | None:
    runs = list_runs(results_root)
    return runs[0] if runs else None


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {"run_dir": str(run), "metrics": metrics, "metadata": metadata}


def load_equity_curve(run_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / "equity_curve.parquet")


def load_trades(run_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / "trades.parquet")


def format_metrics_plain(metrics: dict[str, float]) -> str:
    explanations = metric_explanations()
    return "\n".join(
        f"{key}: {value:+.6f} -- {explanations.get(key, key)}"
        for key, value in metrics.items()
    )


def show_latest_results() -> str:
    run = latest_run()
    if run is None:
        return "No results found yet. Run a backtest first."
    return f"Run: {run}\n" + format_metrics_plain(load_run_summary(run)["metrics"])


def dashboard_status() -> str:
    if importlib.util.find_spec("streamlit") is None:
        return f"Streamlit is not installed. Install it with: {STREAMLIT_INSTALL_COMMAND}"
    return "Streamlit is installed. Run: streamlit run dashboard/app_streamlit.py"


def launch_dashboard() -> int:
    if importlib.util.find_spec("streamlit") is None:
        print(dashboard_status())
        return 1
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "dashboard/app_streamlit.py"])


def _choose_config() -> Path:
    configs = list_experiment_configs()
    for i, path in enumerate(configs, 1):
        print(f"{i}) {path.name}")
    return configs[int(input("Choose config number: ").strip()) - 1]


def interactive_menu() -> int:
    while True:
        print("\nquant-personal main panel")
        print("1) generate example data")
        print("2) run a backtest")
        print("3) show last results")
        print("4) run paper demo")
        print("5) launch dashboard")
        print("6) run tests")
        print("0) exit")
        choice = input("Select: ").strip()
        if choice == "0":
            return 0
        if choice == "1":
            for path in generate_example_data():
                print(f"wrote {path}")
        elif choice == "2":
            artifacts = run_backtest_experiment(_choose_config())
            print(f"run_dir: {artifacts.run_dir}")
            print(format_metrics_plain(artifacts.metrics))
        elif choice == "3":
            print(show_latest_results())
        elif choice == "4":
            return subprocess.call([sys.executable, "scripts/run_paper_demo.py"])
        elif choice == "5":
            return launch_dashboard()
        elif choice == "6":
            return subprocess.call([sys.executable, "-m", "pytest"])
        else:
            print("Unknown choice.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="quant-personal main panel",
        epilog=(
            "Menu items: 1) generate example data  2) run a backtest  "
            "3) show last results  4) run paper demo  5) launch dashboard  6) run tests"
        ),
    )
    parser.add_argument("--generate-example-data", action="store_true")
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--show-last-results", action="store_true")
    parser.add_argument("--dashboard-status", action="store_true")
    args = parser.parse_args(argv)
    if args.generate_example_data:
        for path in generate_example_data():
            print(f"wrote {path}")
        return 0
    if args.run_config:
        artifacts = run_backtest_experiment(args.run_config)
        print(f"run_dir: {artifacts.run_dir}")
        print(format_metrics_plain(artifacts.metrics))
        return 0
    if args.show_last_results:
        print(show_latest_results())
        return 0
    if args.dashboard_status:
        print(dashboard_status())
        return 0
    return interactive_menu()
