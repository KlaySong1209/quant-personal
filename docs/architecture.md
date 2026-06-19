# Architecture

`quant-personal` is a local daily-frequency research tool.

Core logic lives under `src/quant/`.

- `quant.data`: local files, synthetic data, validation, corporate actions,
  point-in-time universe, trading calendar, futures continuous contracts.
- `quant.backtest`: vectorized daily backtest with one-period execution shift.
- `quant.risk`: reject-only risk checks.
- `quant.execution`: virtual PaperBroker only.
- `quant.experiment`: run orchestration and artifacts.
- `quant.app`: shared orchestration for CLI and dashboard.

The dashboard and scripts are thin views over `quant.app`; they must not
reimplement calculations.

No real money, live data, broker credentials, microservices, or event-driven
tick engine are included.

