"""Shared target-weight signal entry point."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant.risk.checks import RiskConfig, apply_risk_checks
from quant.strategy import build_strategy


class AdjustmentMismatchError(ValueError):
    """Raised when signal and execution prices use incompatible adjustments."""


@dataclass(frozen=True)
class SignalConfig:
    strategy_name: str = "placeholder"
    strategy_params: dict = field(default_factory=lambda: {"mode": "equal_weight"})
    risk: RiskConfig = field(
        default_factory=lambda: RiskConfig(max_symbol_weight=1.0, max_gross_leverage=1.0)
    )
    adjustment: str = "none"


def generate_target_weights(prices: pd.DataFrame, signal_config: SignalConfig) -> pd.DataFrame:
    strategy = build_strategy(signal_config.strategy_name, dict(signal_config.strategy_params))
    raw = strategy.generate_weights(prices)
    return apply_risk_checks(raw, signal_config.risk)


def assert_adjustment_consistent(signal_adjustment: str, *others: str) -> None:
    values = [signal_adjustment, *others]
    if len(set(values)) > 1:
        raise AdjustmentMismatchError(
            "price adjustment mismatch: signal uses "
            f"{signal_adjustment!r} but a matching layer uses {list(others)!r}; "
            "signal and matching must use the same adjustment"
        )
