"""Performance metrics with Rust acceleration and Python fallback.

Three-tier fallback: Rust (.so via maturin) > Pure Python

This wrapper imports from the locally-owned _hb_rust extension
(built via ``pixi run rust-build``).
"""

from __future__ import annotations

import numpy as np

try:
    from _hb_rust.metrics import (
        calculate_all_metrics as _rs_calculate_all_metrics,
        calculate_max_drawdown as _rs_calculate_max_drawdown,
        calculate_profit_factor as _rs_calculate_profit_factor,
        calculate_sharpe_ratio as _rs_calculate_sharpe_ratio,
    )

    _ACCELERATED = True
except ImportError:
    _ACCELERATED = False


def is_accelerated() -> bool:
    """Return True if Rust acceleration is available."""
    return _ACCELERATED


def calculate_max_drawdown(
    pnl_series: np.ndarray,
) -> tuple[float, float]:
    """Calculate max drawdown and max drawdown percentage.

    :param pnl_series: Cumulative P&L values as numpy array
    :return: (max_drawdown_abs, max_drawdown_pct)
    """
    if _ACCELERATED:
        return _rs_calculate_max_drawdown(pnl_series)

    n = len(pnl_series)
    if n == 0:
        return (0.0, 0.0)

    peak = pnl_series[0]
    max_dd = 0.0
    max_dd_pct = 0.0

    for i in range(1, n):
        if pnl_series[i] > peak:
            peak = pnl_series[i]
        dd = peak - pnl_series[i]
        if dd > max_dd:
            max_dd = dd
            if peak != 0.0:
                max_dd_pct = dd / peak

    return (max_dd, max_dd_pct)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio from returns series.

    :param returns: Array of period returns
    :param periods_per_year: Annualization factor (default 252 for daily)
    :return: Annualized Sharpe ratio
    """
    if _ACCELERATED:
        return _rs_calculate_sharpe_ratio(returns, periods_per_year)

    n = len(returns)
    if n < 2:
        return 0.0

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))

    if std == 0.0:
        return 0.0

    return (mean / std) * (periods_per_year**0.5)


def calculate_profit_factor(pnl_values: np.ndarray) -> float:
    """Calculate profit factor (sum of wins / abs sum of losses).

    :param pnl_values: Array of individual trade P&L values
    :return: Profit factor (inf if no losses)
    """
    if _ACCELERATED:
        return _rs_calculate_profit_factor(pnl_values)

    gross_profit = float(np.sum(pnl_values[pnl_values > 0]))
    gross_loss = float(np.sum(np.abs(pnl_values[pnl_values < 0])))

    if gross_loss == 0.0:
        return float("inf")

    return gross_profit / gross_loss


def calculate_all_metrics(
    pnl_series: np.ndarray,
    periods_per_year: int = 252,
) -> tuple[float, float, float, float]:
    """Calculate all performance metrics.

    :param pnl_series: Cumulative P&L values as numpy array
    :param periods_per_year: Annualization factor for Sharpe
    :return: (max_drawdown, max_drawdown_pct, sharpe_ratio, profit_factor)
    """
    if _ACCELERATED:
        return _rs_calculate_all_metrics(pnl_series, periods_per_year)

    max_dd, max_dd_pct = calculate_max_drawdown(pnl_series)
    sharpe = calculate_sharpe_ratio(pnl_series, periods_per_year)
    pf = calculate_profit_factor(pnl_series)

    return (max_dd, max_dd_pct, sharpe, pf)
