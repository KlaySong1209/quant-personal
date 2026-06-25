"""Small file-artifact helpers used by CLI, dashboard, and paper sessions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def write_dataframe(df: pd.DataFrame, base_path: str | Path) -> Path:
    base = Path(base_path)
    parquet_path = base.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def load_equity_curve(run_dir: str | Path) -> pd.DataFrame:
    run = Path(run_dir)
    parquet_path = run / "equity_curve.parquet"
    csv_path = run / "equity_curve.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        return df
    raise FileNotFoundError(f"missing equity curve artifact in {run}")


def load_trades(run_dir: str | Path) -> pd.DataFrame:
    run = Path(run_dir)
    parquet_path = run / "trades.parquet"
    csv_path = run / "trades.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"missing trades artifact in {run}")
