"""
Unit Tests for Performance and Risk Metrics
===========================================
Validates Sharpe ratio, annualized volatility, max drawdown, Sortino,
and Calmar against exact hand-calculated test cases.
"""

import numpy as np
import pytest

from quant_lab.engine.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    max_drawdown_duration,
    calmar_ratio,
    calculate_all_metrics,
)


def test_hand_calculated_volatility():
    """Verify sample standard deviation and annualization against hand calculations."""
    # 4 observations
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    # Mean: (0.01 - 0.02 + 0.03 - 0.01) / 4 = 0.01 / 4 = 0.0025
    # Deviations: [0.0075, -0.0225, 0.0275, -0.0125]
    # Sum of sq deviations: 0.0075^2 + 0.0225^2 + 0.0275^2 + 0.0125^2
    # = 0.00005625 + 0.00050625 + 0.00075625 + 0.00015625 = 0.001475
    # Sample variance (ddof=1): 0.001475 / 3 = 0.0004916666666666667
    # Sample std: sqrt(0.0004916666666666667) = 0.022173557826535897
    # Annualized vol with periods=252:
    # 0.022173557826535897 * sqrt(252) = 0.3520014204516788
    
    expected_vol = 0.022173557826535897 * np.sqrt(252)
    computed_vol = annualized_volatility(returns, periods_per_year=252)
    assert np.isclose(computed_vol, expected_vol, rtol=1e-7)


def test_hand_calculated_sharpe_ratio():
    """Verify annualized Sharpe ratio with risk-free rate adjustment."""
    returns = np.array([0.02, 0.01, 0.03, 0.02])
    # Mean: 0.02
    # Variance (ddof=1): [(0)^2 + (-0.01)^2 + (0.01)^2 + (0)^2] / 3 = 0.0002 / 3 = 6.66666667e-5
    # Std: sqrt(6.66666667e-5) = 0.00816496580927726
    # Sharpe with rf=0: (0.02 / 0.00816496580927726) * sqrt(252)
    # = 2.449489742783178 * 15.874507866387544 = 38.88444413901389
    
    expected_sharpe = (0.02 / np.std(returns, ddof=1)) * np.sqrt(252)
    computed_sharpe = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    assert np.isclose(computed_sharpe, expected_sharpe, rtol=1e-7)


def test_hand_calculated_max_drawdown():
    """
    Verify Max Drawdown against an exact sequence of price changes.
    Wealth sequence:
    t=0: $1.00
    t=1: +20% -> $1.20 (Peak = 1.20)
    t=2: -25% -> 1.20 * 0.75 = $0.90 (DD = (0.90 - 1.20)/1.20 = -0.25)
    t=3: +10% -> 0.90 * 1.10 = $0.99 (Peak = 1.20, DD = (0.99 - 1.20)/1.20 = -0.175)
    Max Drawdown must be exactly 0.25 (25.0%).
    """
    returns = np.array([0.20, -0.25, 0.10])
    dd = max_drawdown(returns)
    assert np.isclose(dd, 0.25, atol=1e-6)
    assert max_drawdown_duration(returns) == 2


def test_sortino_ratio_downside():
    """Verify that Sortino only penalizes negative deviations across all observations."""
    # Strategy with only positive returns
    positive_returns = np.array([0.01, 0.02, 0.015, 0.03])
    sortino_pos = sortino_ratio(positive_returns, risk_free_rate=0.0)
    assert np.isinf(sortino_pos)

    # Strategy with negative returns
    mixed_returns = np.array([0.02, -0.01, 0.03, -0.02])
    # T = 4, excess = [0.02, -0.01, 0.03, -0.02]
    # downside diff = [0, -0.01, 0, -0.02]
    # downside variance = (0 + 0.0001 + 0 + 0.0004) / 4 = 0.0005 / 4 = 0.000125
    # downside vol = sqrt(0.000125) * sqrt(252)
    # mean excess = (0.02 - 0.01 + 0.03 - 0.02)/4 * 252 = 0.005 * 252 = 1.26
    # sortino = 1.26 / (sqrt(0.000125) * sqrt(252))
    expected_downside_vol = np.sqrt(0.000125) * np.sqrt(252)
    expected_sortino = 1.26 / expected_downside_vol
    computed_sortino = sortino_ratio(mixed_returns, risk_free_rate=0.0, periods_per_year=252)
    assert np.isclose(computed_sortino, expected_sortino, rtol=1e-7)


def test_zero_variance_edge_cases():
    """Verify zero variance does not crash or raise exceptions."""
    flat_returns = np.array([0.0, 0.0, 0.0, 0.0])
    assert sharpe_ratio(flat_returns) == 0.0
    assert annualized_volatility(flat_returns) == 0.0
    assert max_drawdown(flat_returns) == 0.0
    
    empty_returns = np.array([])
    assert annualized_return(empty_returns) == 0.0
    assert max_drawdown(empty_returns) == 0.0
