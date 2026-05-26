"""Tests for hummingbot.core.rust_metrics — covers Python fallback path.

In CI (Rust not built), _ACCELERATED=False so all Python fallback
functions execute. These tests verify correctness of the fallback
implementations and ensure 100% diff-cover on the module.
"""

import numpy as np
import pytest

from hummingbot.core.rust_metrics import (
    calculate_all_metrics,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    is_accelerated,
)

# ---------------------------------------------------------------------------
# is_accelerated
# ---------------------------------------------------------------------------


def test_is_accelerated_returns_bool():
    result = is_accelerated()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# calculate_max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_empty_array():
    result = calculate_max_drawdown(np.array([]))
    assert result == (0.0, 0.0)


def test_max_drawdown_single_element():
    result = calculate_max_drawdown(np.array([5.0]))
    assert result == (0.0, 0.0)


def test_max_drawdown_monotone_up():
    pnl = np.array([1.0, 2.0, 3.0, 4.0])
    dd, dd_pct = calculate_max_drawdown(pnl)
    assert dd == pytest.approx(0.0)
    assert dd_pct == pytest.approx(0.0)


def test_max_drawdown_simple_drop():
    # Peak=4, drops to 2 → drawdown=2, pct=0.5
    pnl = np.array([1.0, 4.0, 2.0])
    dd, dd_pct = calculate_max_drawdown(pnl)
    assert dd == pytest.approx(2.0)
    assert dd_pct == pytest.approx(0.5)


def test_max_drawdown_peak_zero():
    # Peak starts at 0 — pct should remain 0 (no division)
    pnl = np.array([0.0, -1.0])
    dd, dd_pct = calculate_max_drawdown(pnl)
    assert dd == pytest.approx(1.0)
    assert dd_pct == pytest.approx(0.0)


def test_max_drawdown_multiple_drops():
    pnl = np.array([0.0, 10.0, 5.0, 15.0, 8.0])
    dd, dd_pct = calculate_max_drawdown(pnl)
    # peak=15, min=8, dd=7, pct=7/15
    assert dd == pytest.approx(7.0)
    assert dd_pct == pytest.approx(7.0 / 15.0)


# ---------------------------------------------------------------------------
# calculate_sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_empty_returns():
    result = calculate_sharpe_ratio(np.array([]))
    assert result == 0.0


def test_sharpe_single_return():
    result = calculate_sharpe_ratio(np.array([0.01]))
    assert result == 0.0


def test_sharpe_zero_std():
    # Constant returns → std=0 → Sharpe=0
    returns = np.array([0.01, 0.01, 0.01])
    result = calculate_sharpe_ratio(returns)
    assert result == 0.0


def test_sharpe_positive_ratio():
    returns = np.array([0.01, 0.02, 0.015, 0.03, 0.01])
    result = calculate_sharpe_ratio(returns, periods_per_year=252)
    assert result > 0.0


def test_sharpe_negative_ratio():
    returns = np.array([-0.01, -0.02, -0.015])
    result = calculate_sharpe_ratio(returns)
    assert result < 0.0


def test_sharpe_custom_periods():
    returns = np.array([0.01, 0.02, 0.015])
    r252 = calculate_sharpe_ratio(returns, periods_per_year=252)
    r52 = calculate_sharpe_ratio(returns, periods_per_year=52)
    # More periods = larger annualization factor = larger absolute ratio
    assert abs(r252) > abs(r52)


# ---------------------------------------------------------------------------
# calculate_profit_factor
# ---------------------------------------------------------------------------


def test_profit_factor_all_wins():
    pnl = np.array([10.0, 5.0, 3.0])
    result = calculate_profit_factor(pnl)
    assert result == float("inf")


def test_profit_factor_all_losses():
    pnl = np.array([-5.0, -3.0])
    result = calculate_profit_factor(pnl)
    assert result == pytest.approx(0.0)


def test_profit_factor_mixed():
    # wins=15, losses=5 → pf=3.0
    pnl = np.array([10.0, 5.0, -3.0, -2.0])
    result = calculate_profit_factor(pnl)
    assert result == pytest.approx(3.0)


def test_profit_factor_no_losses_zero_sum():
    pnl = np.array([0.0, 0.0])
    result = calculate_profit_factor(pnl)
    assert result == float("inf")


# ---------------------------------------------------------------------------
# calculate_all_metrics
# ---------------------------------------------------------------------------


def test_all_metrics_empty():
    dd, dd_pct, sharpe, pf = calculate_all_metrics(np.array([]))
    assert dd == 0.0
    assert dd_pct == 0.0
    assert sharpe == 0.0
    assert pf == float("inf")


def test_all_metrics_growing():
    pnl = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dd, dd_pct, sharpe, pf = calculate_all_metrics(pnl, periods_per_year=252)
    assert dd == pytest.approx(0.0)
    assert dd_pct == pytest.approx(0.0)
    # pnl series used as returns → all positive → pf=inf
    assert pf == float("inf")


def test_all_metrics_with_drawdown():
    pnl = np.array([0.0, 10.0, 5.0, 8.0])
    dd, dd_pct, sharpe, pf = calculate_all_metrics(pnl)
    assert dd == pytest.approx(5.0)
    assert dd_pct == pytest.approx(0.5)
    assert isinstance(sharpe, float)
    assert isinstance(pf, float)


def test_all_metrics_returns_tuple_of_four():
    result = calculate_all_metrics(np.array([1.0, 2.0, 1.5]))
    assert len(result) == 4
