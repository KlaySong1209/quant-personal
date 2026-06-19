# Getting Started

This is a local research tool. It does not connect real money, brokers,
credentials, live feeds, or validated alpha.

## 10-Minute First Run

1. Install the package:

```bash
pip install -e .[dev]
```

2. Start the main panel:

```bash
python -m quant
```

3. Pick:

- `1` to generate example data.
- `2` to run a backtest.
- `3` to show the latest result in plain language.

Each run writes a folder under `results/<run_id>/` with:

- `config_snapshot.yaml`
- `metrics.json`
- `metadata.json`
- `equity_curve.parquet`
- `trades.parquet`
- plots and logs

## Reading Metrics

- Sharpe: risk-adjusted return; higher is better, below 0 means it lost money after risk.
- Max drawdown: worst peak-to-trough loss.
- Turnover: how much the portfolio traded; higher usually means higher costs.

## Dashboard

```bash
pip install -e .[dashboard]
streamlit run dashboard/app_streamlit.py
```

## Using Your Own RESSET Data

See `docs/data_ingestion.md`. Data must be local files. Configure column maps in
YAML; do not change source code for vendor column names.

## Writing Your Own Strategy

See `docs/how_to_build_a_strategy.md`.

