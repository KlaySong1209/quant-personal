# Local Daily Data Ingestion

This project reads real market data only from files already on your computer.
It does not download data, store credentials, connect to brokers, or read live
feeds.

The current workflow is for daily stock data and a small universe of 3-5 symbols.
RESSET-style exports are historical bulk data only. Latest paper-account prices
come from the separate manual QuoteSource path described below, not from RESSET.

## Step 1: Export A Daily File

From RESSET or another data vendor, export one daily stock price file containing
your selected symbols. A typical file should have one row per date and symbol.

Required fields after mapping:

- trade date
- stock symbol
- open
- high
- low
- close
- volume

Optional but preferred fields:

- adjusted close
- adjustment factor
- dividend
- dividend ex-date
- split ratio

Adjusted data is never guessed. If adjusted close is present, declare
`--adjustment-convention backward --adjusted-price-column`. If an adjustment
factor is present, declare `--adjustment-convention backward
--has-adjustment-factor`; the factor path means adjusted OHLC = raw OHLC * factor.
If dividends are present, the mapping must include an ex-dividend date column and
you must declare `--dividend-tax-treatment pre_tax` or `post_tax`.

## Step 2: Fill The Column Mapping

Use `configs/data_mappings/resset_daily_illustrative.yaml` as a starting point.
It is illustrative only. Check your actual RESSET export and change the right side
of each mapping to match your file.

```yaml
columns:
  timestamp: Trddt
  symbol: Stkcd
  open: Opnprc
  high: Hiprc
  low: Loprc
  close: Clsprc
  volume: Dnshrtrd
  adjusted_close: AdjClsprc
  adjustment_factor: AdjFactor
  dividend: Dvdnt
  dividend_ex_date: ExDate
  split: SplitRatio
  bonus: BonusShareRatio
  conversion: TransferShareRatio
```

The left side is the internal name. Do not change it. The right side is the
column name in your export.

## Step 3: Test With Synthetic Local Export

This writes a RESSET-shaped example file, then sends it through the same mapping
and validation path used for real local files.

```bash
python -m quant --write-synthetic-local-export data/raw/resset_daily.csv --column-mapping configs/data_mappings/resset_daily_illustrative.yaml --symbols 000001 000002 000003 --start 2020-01-01 --end 2020-01-31
python -m quant --ingest-local-data data/raw/resset_daily.csv --column-mapping configs/data_mappings/resset_daily_illustrative.yaml --symbols 000001 000002 000003 --adjustment-convention backward --adjusted-price-column
```

Expected outputs:

- `data/processed/local_daily_ohlcv.parquet` when parquet support is available.
- `data/processed/local_daily_ohlcv.csv` as fallback.
- `data/processed/local_daily_metadata.json` with mapping, adjustment, calendar,
  and gap facts.

## Step 4: Ingest Your Real File

Replace `data/raw/resset_daily.csv` with your local export path:

```bash
python -m quant --ingest-local-data path/to/your_export.csv --column-mapping configs/data_mappings/resset_daily_illustrative.yaml --symbols 000001 000002 000003 --calendar-file path/to/sse_or_szse_calendar.csv --calendar-date-column TradingDate --calendar-is-open-column IsOpen --adjustment-convention backward --adjusted-price-column --production-data
```

The ingestion fails fast when:

- a mapped column is missing
- the file contains symbols outside the configured 3-5 symbol universe
- any configured symbol is missing
- timestamps are duplicated per symbol
- prices or volume are invalid
- OHLC values are inconsistent
- dates fall outside the trading calendar
- adjusted prices or factors are present without explicit declarations
- dividends are present without an ex-dividend date and tax declaration

## Trading Calendar

For real data, provide an exchange calendar file. The file can be a shared
calendar or separate SSE/SZSE files; the system does not assume they are
identical. A minimal file has one date column, and may also have an open/closed
flag.

Illustrative mapping:

```yaml
columns:
  date: TradingDate
  is_open: IsOpen
```

With a file calendar, trading days are judged strictly by that file. Missing file,
bad mapping, or unparseable dates fail fast. Without a file, the system uses a
synthetic business-day approximation and records a warning that A-share holidays
are not included. `--production-data` forbids the synthetic calendar.

Date gaps across symbols are handled explicitly and recorded in metadata. They
are not silently ignored.

## Manual Quote File For Paper Account

The paper account can advance one day from a small local quote CSV:

```csv
date,symbol,close
2020-01-31,000001,10.25
2020-01-31,000002,12.40
2020-01-31,000003,8.90
```

Run:

```bash
python scripts/run_paper_session.py --manual-quotes data/raw/manual_quotes.csv --symbols 000001 000002 000003 --as-of 2020-01-31
```

`RealtimeQuoteSource` is reserved for future broker or quote APIs. It is not
implemented, and adding it later should not require account or strategy changes.

## Futures Contracts

Daily local futures exports are supported separately through the existing
continuous-contract path. Real-time, intraday, and tick workflows are future work
and need a different engine.
