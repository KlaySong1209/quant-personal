# Architecture

`quant-personal` is a local daily-frequency research tool.

Core logic lives under `src/quant/`.

- `quant.data`: separate historical bulk sources and latest quote sources.
  RESSET-style local exports are HistoricalSource inputs for research/backtests.
  ManualQuoteSource is the only current QuoteSource for paper-account day steps;
  RealtimeQuoteSource is reserved and not implemented.
- `quant.data.adjust`: corporate-action declarations, point-in-time universe,
  and trading calendar validation.
- `quant.backtest`: vectorized daily backtest with one-period execution shift.
- `quant.risk`: reject-only risk checks.
- `quant.execution`: virtual PaperBroker and persistent SimAccount only, labeled
  as simulated paper trading with no route to real venues.
- `quant.experiment`: run orchestration and artifacts.
- `quant.app`: shared orchestration for CLI and dashboard.

The dashboard and scripts are thin views over `quant.app`; they must not
reimplement calculations.

No real money, live data, broker credentials, microservices, or event-driven
tick engine are included.
