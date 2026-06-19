"""Drive the PaperBroker through placeholder target positions and print a ledger.

Thin orchestration only — all logic lives under ``quant``.
"""

from __future__ import annotations

from pathlib import Path

from quant.config.loader import load_config
from quant.data.pipeline import load_and_validate, to_close_panel
from quant.execution.paper import PaperBroker
from quant.portfolio.target import target_dollars_to_shares, weights_to_target_dollars
from quant.risk.checks import RiskConfig, apply_risk_checks
from quant.strategy import build_strategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "experiments" / "exp_placeholder.yaml"


def main() -> None:
    cfg = load_config(CONFIG)
    ohlcv = load_and_validate(cfg)
    prices = to_close_panel(ohlcv)

    strategy = build_strategy(cfg.strategy.name, dict(cfg.strategy.params))
    raw_weights = strategy.generate_weights(prices)

    risk = RiskConfig(
        max_symbol_weight=cfg.risk.max_symbol_weight,
        max_gross_leverage=cfg.risk.max_gross_leverage,
        reject_nan=cfg.risk.reject_nan,
    )
    weights = apply_risk_checks(raw_weights, risk)
    effective = weights.shift(1).fillna(0.0)

    target_dollars = weights_to_target_dollars(effective, cfg.portfolio.initial_equity)
    target_shares = target_dollars_to_shares(target_dollars, prices)

    broker = PaperBroker(
        starting_cash=cfg.execution.paper.starting_cash,
        allow_short=cfg.execution.paper.allow_short,
        allow_margin=cfg.execution.paper.allow_margin,
        max_gross_leverage=cfg.execution.paper.max_gross_leverage,
    )

    for ts in prices.index:
        px_row = prices.loc[ts]
        broker.update_prices(px_row.to_dict())
        broker.submit_target(ts, target_shares.loc[ts].to_dict())

    last_ts = prices.index[-1]
    equity = broker.mark_to_market(last_ts)
    ledger = broker.ledger()

    print(f"events: {len(ledger)}")
    print(f"final cash:      {broker.cash:,.2f}")
    print(f"final positions: {broker.positions()}")
    print(f"final equity:    {equity:,.2f}")
    print(f"ledger balanced: {broker.is_balanced()}")
    if not broker.is_balanced():
        raise SystemExit("paper-broker ledger is NOT balanced — investigate.")


if __name__ == "__main__":
    main()

