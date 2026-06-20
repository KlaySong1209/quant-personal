# quant-personal

Start here for non-engineers: see [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

Current scope:

- Daily end-of-day data only.
- A small user-chosen stock universe: 3-5 symbols.
- Local CSV/parquet files exported by the user, with configurable column mapping.
- A persistent simulated paper account advanced from local data or a manual quote CSV.
- Strategy logic is only a trivial placeholder so the data and account flow can run.

Reserved for future work: real-time or intraday trading, real-time gold, whole-market
universes, real strategies, live brokers, and real order routing.

Start the local menu:

```bash
python -m quant
```

Optional local dashboard:

```bash
pip install -e .[dashboard]
streamlit run dashboard/app_streamlit.py
```

This is a personal-use quantitative research system. It does not connect real
money, real brokers, credentials, live feeds, or validated alpha. Data is read
from local files or generated synthetically.

## Advanced CLI

```bash
python -m quant --generate-example-data
python -m quant --write-synthetic-local-export data/raw/resset_daily.csv --column-mapping configs/data_mappings/resset_daily_illustrative.yaml --symbols 000001 000002 000003 --start 2020-01-01 --end 2020-01-31
python -m quant --ingest-local-data data/raw/resset_daily.csv --column-mapping configs/data_mappings/resset_daily_illustrative.yaml --symbols 000001 000002 000003 --calendar-file data/raw/sse_calendar.csv --calendar-date-column TradingDate --calendar-is-open-column IsOpen --adjustment-convention backward --adjusted-price-column --production-data
python scripts/run_paper_session.py --data data/processed/local_daily_ohlcv.parquet --symbols 000001 000002 000003
python scripts/run_paper_session.py --manual-quotes data/raw/manual_quotes.csv --symbols 000001 000002 000003 --as-of 2020-01-31
python -m quant --run-config configs/experiments/exp_placeholder.yaml
python -m quant --paper-demo
python -m unittest discover -s tests
```

Every run writes `results/<run_id>/` with config snapshot, metrics, equity curve,
trades, plots, logs, and metadata.
