# Local Daily Data Ingestion

All real data must come from local files exported by the user. The project does
not contain network, credential, live-feed, or broker code.

## RESSET-Style Equity Prices

Export one CSV per symbol or build files compatible with `data.csv_path`.
Required canonical OHLCV columns are:

- `timestamp`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `volume`

Timestamps must include an explicit UTC offset such as `2020-01-01T00:00:00Z`.

## Corporate Actions

Use `data.corporate_actions.column_map` to map vendor columns:

```yaml
corporate_actions:
  enabled: true
  path: "local/div_split.csv"
  column_map:
    symbol: "Stkcd"
    timestamp: "ExDate"
    event_type: "EventKind"
    amount: "CashDiv"
    ratio: "SplitRatio"
```

Canonical event types are `cash_dividend` and `split`.

## Point-In-Time Universe

Use a membership file instead of today's constituents applied to the past:

```yaml
universe:
  source: "file"
  path: "local/membership.csv"
  column_map:
    symbol: "Stkcd"
    start: "InDate"
    end: "OutDate"
```

The backtest may only hold names whose membership is active on that date.

## Trading Calendar

Use a local calendar file when you have an exchange calendar:

```yaml
calendar:
  source: "file"
  exchange: "SSE"
  path: "local/calendar.csv"
  column_map:
    timestamp: "TradingDate"
```

Off-calendar timestamps fail fast.

## Futures Contracts

For daily futures exports, map individual contracts into a back-adjusted
continuous series:

```yaml
data:
  source: "local_file_futures"
  path: "local/futures.csv"
  column_map:
    contract: "Contract"
    timestamp: "TradeDate"
    expiry: "Expiry"
    open: "OpenPx"
    high: "HighPx"
    low: "LowPx"
    close: "ClosePx"
    volume: "Volume"
```

The roll method and rule are recorded in `config_snapshot.yaml`.

