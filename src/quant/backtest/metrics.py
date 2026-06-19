"""Performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0) if len(returns) else 0.0


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(returns)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) >= 2 else 0.0


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    return 0.0 if std == 0 else float(np.sqrt(periods_per_year) * returns.mean() / std)


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1.0 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    curve = equity_curve(returns)
    return float((curve / curve.cummax() - 1.0).min())


def turnover(weights: pd.DataFrame) -> float:
    if len(weights) < 2:
        return 0.0
    return float(weights.diff().abs().sum(axis=1).iloc[1:].mean() * 0.5)


def _align(portfolio: pd.Series, benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = portfolio.index.intersection(benchmark.index)
    if len(idx) == 0:
        raise ValueError("portfolio and benchmark series have no overlapping dates")
    return portfolio.loc[idx], benchmark.loc[idx]


def excess_return(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    p, b = _align(portfolio_returns, benchmark_returns)
    return p - b


def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    ex = excess_return(portfolio_returns, benchmark_returns)
    return float(ex.std(ddof=1) * np.sqrt(periods_per_year)) if len(ex) >= 2 else 0.0


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252) -> float:
    ex = excess_return(portfolio_returns, benchmark_returns)
    if len(ex) < 2:
        return 0.0
    sd = float(ex.std(ddof=1))
    return 0.0 if sd == 0 else float(np.sqrt(periods_per_year) * ex.mean() / sd)


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    p, b = _align(portfolio_returns, benchmark_returns)
    if len(p) < 2:
        return 0.0
    var = float(b.var(ddof=1))
    return 0.0 if var == 0 else float(p.cov(b, ddof=1) / var)


def compute_metrics(
    returns: pd.Series,
    weights: pd.DataFrame,
    *,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    out = {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "turnover": turnover(weights),
    }
    if benchmark_returns is not None:
        ex = excess_return(returns, benchmark_returns)
        out["excess_return"] = float((1.0 + ex).prod() - 1.0)
        out["tracking_error"] = tracking_error(returns, benchmark_returns)
        out["information_ratio"] = information_ratio(returns, benchmark_returns)
        out["beta"] = beta(returns, benchmark_returns)
    return out

