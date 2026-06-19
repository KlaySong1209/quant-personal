"""Reject-only risk checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class RiskCheckError(ValueError):
    pass


@dataclass(frozen=True)
class RiskConfig:
    max_symbol_weight: float
    max_gross_leverage: float
    reject_nan: bool = True

    def __post_init__(self) -> None:
        if not (0 < self.max_symbol_weight <= 1):
            raise ValueError("max_symbol_weight must be in (0, 1]")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")


@dataclass(frozen=True)
class FuturesRiskConfig:
    max_symbol_notional: float
    max_gross_notional: float
    max_margin_use: float
    reject_nan: bool = True


def apply_risk_checks(weights: pd.DataFrame, cfg: RiskConfig, *, tol: float = 1e-9) -> pd.DataFrame:
    arr = weights.to_numpy(dtype="float64")
    if cfg.reject_nan and not np.isfinite(arr).all():
        raise RiskCheckError("weights contain NaN/inf")
    abs_w = weights.abs()
    if (abs_w > cfg.max_symbol_weight + tol).any().any():
        raise RiskCheckError("per-symbol weight exceeds max")
    gross = abs_w.sum(axis=1)
    if (gross > cfg.max_gross_leverage + tol).any():
        raise RiskCheckError("gross leverage exceeds max")
    return weights

