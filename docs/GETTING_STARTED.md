# Getting Started

This is a local research tool. It does not connect real money, brokers,
credentials, live feeds, or validated alpha.

Current scope: daily data, 3-5 stocks, local files, simulated paper account.
Real-time data, whole-market universes, live brokers, and real strategies are
future work.

## 10-Minute First Run

1. Install the package:

```bash
pip install -e .
```

2. Start the main panel:

```bash
python -m quant
```

3. Pick:

- `1` to generate example data.
- `2` to run a backtest.
- `3` to show the latest result in plain language.
- `7` to ingest a mapped local data file.
- `8` to run or resume the simulated paper account.

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

Minimal command path:

```bash
python -m quant --ingest-local-data path/to/your_export.csv --column-mapping configs/data_mappings/resset_daily_illustrative.yaml --symbols 000001 000002 000003 --calendar-file path/to/calendar.csv --calendar-date-column TradingDate --calendar-is-open-column IsOpen --adjustment-convention backward --adjusted-price-column --production-data
python scripts/run_paper_session.py --data data/processed/local_daily_ohlcv.parquet --symbols 000001 000002 000003
python scripts/run_paper_session.py --manual-quotes data/raw/manual_quotes.csv --symbols 000001 000002 000003 --as-of 2020-01-31
```

## Writing Your Own Strategy

See `docs/where_strategy_goes.md`.
