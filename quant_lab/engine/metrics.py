"""
Performance and Risk Metrics Module
===================================
Calculates risk-adjusted return and drawdown statistics for quantitative strategies.
Implementation follows established academic standards and handles edge cases safely.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def to_numpy_returns(returns: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Sanitize and validate return series, removing NaNs."""
    arr = np.asarray(returns, dtype=float)
    clean = arr[~np.isnan(arr)]
    return clean


def annualized_return(
    returns: Union[np.ndarray, pd.Series, list],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    geometric: bool = True
) -> float:
    """
    Calculate annualized return.
    
    Parameters
    ----------
    returns : array-like
        Periodic rate of return series (e.g. daily returns).
    periods_per_year : int, default 252
        Number of observations per year.
    geometric : bool, default True
        If True, calculates CAGR. If False, calculates annualized arithmetic mean.
    """
    arr = to_numpy_returns(returns)
    n = len(arr)
    if n == 0:
        return 0.0
    
    if geometric:
        wealth = np.cumprod(1.0 + arr)
        final_wealth = wealth[-1]
        if final_wealth <= 0:
            return -1.0
        return float(final_wealth ** (periods_per_year / n) - 1.0)
    else:
        return float(np.mean(arr) * periods_per_year)


def annualized_volatility(
    returns: Union[np.ndarray, pd.Series, list],
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized volatility (sample standard deviation scaled by sqrt(periods)).
    """
    arr = to_numpy_returns(returns)
    if len(arr) < 2:
        return 0.0
    vol = np.std(arr, ddof=1)
    return float(vol * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized Sharpe ratio:
    Sharpe = (E[R] - Rf) / Vol(R) * sqrt(periods_per_year)
    """
    arr = to_numpy_returns(returns)
    n = len(arr)
    if n < 2:
        return 0.0
    
    rf_per_period = risk_free_rate / periods_per_year
    excess = arr - rf_per_period
    std_excess = np.std(excess, ddof=1)
    
    if std_excess <= 1e-12:
        return 0.0
    
    return float((np.mean(excess) / std_excess) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate annualized Sortino ratio using full-sample downside semi-deviation.
    
    Downside deviation is defined across ALL T observations:
        sigma_down = sqrt( (1 / T) * sum( min(0, r_t - r_f)^2 ) ) * sqrt(periods)
    """
    arr = to_numpy_returns(returns)
    n = len(arr)
    if n < 2:
        return 0.0
    
    rf_per_period = risk_free_rate / periods_per_year
    excess = arr - rf_per_period
    downside_diff = np.minimum(0.0, excess)
    downside_variance = np.sum(downside_diff ** 2) / n
    downside_vol = np.sqrt(downside_variance) * np.sqrt(periods_per_year)
    
    mean_annual_excess = np.mean(excess) * periods_per_year
    
    if downside_vol <= 1e-12:
        return float(np.inf) if mean_annual_excess > 0 else 0.0
        
    return float(mean_annual_excess / downside_vol)


def max_drawdown(returns: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Calculate maximum peak-to-trough drawdown as a positive fraction in [0, 1].
    Example: 0.25 represents a 25% peak-to-trough decline.
    """
    arr = to_numpy_returns(returns)
    if len(arr) == 0:
        return 0.0
    
    wealth = np.cumprod(1.0 + arr)
    wealth = np.insert(wealth, 0, 1.0)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (wealth - running_max) / np.where(running_max > 0, running_max, 1.0)
    min_dd = np.min(drawdowns)
    return float(abs(min_dd))


def max_drawdown_duration(returns: Union[np.ndarray, pd.Series, list]) -> int:
    """
    Calculate the longest drawdown duration in periods (bars).
    """
    arr = to_numpy_returns(returns)
    if len(arr) == 0:
        return 0
    
    wealth = np.cumprod(1.0 + arr)
    wealth = np.insert(wealth, 0, 1.0)
    running_max = np.maximum.accumulate(wealth)
    
    max_duration = 0
    current_duration = 0
    for w, m in zip(wealth, running_max):
        if w < m:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0
            
    return max_duration


def calmar_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate Calmar ratio = Annualized Return / Maximum Drawdown.
    """
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year, geometric=True)
    mdd = max_drawdown(returns)
    if mdd <= 1e-12:
        return float(np.inf) if ann_ret > 0 else 0.0
    return float(ann_ret / mdd)


def win_rate(returns: Union[np.ndarray, pd.Series, list]) -> float:
    """Calculate the fraction of non-zero returns that are positive."""
    arr = to_numpy_returns(returns)
    active = arr[arr != 0]
    if len(active) == 0:
        return 0.0
    return float(np.sum(active > 0) / len(active))


def profit_factor(returns: Union[np.ndarray, pd.Series, list]) -> float:
    """Calculate gross profit divided by gross loss."""
    arr = to_numpy_returns(returns)
    gains = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())
    if losses <= 1e-12:
        return float(np.inf) if gains > 0 else 0.0
    return float(gains / losses)


def calculate_all_metrics(
    returns: Union[np.ndarray, pd.Series, list],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> Dict[str, float]:
    """
    Compute a consolidated dictionary of risk, return, and drawdown metrics.
    """
    arr = to_numpy_returns(returns)
    cagr = annualized_return(arr, periods_per_year, geometric=True)
    arithmetic_ret = annualized_return(arr, periods_per_year, geometric=False)
    vol = annualized_volatility(arr, periods_per_year)
    sharpe = sharpe_ratio(arr, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(arr, risk_free_rate, periods_per_year)
    mdd = max_drawdown(arr)
    mdd_duration = max_drawdown_duration(arr)
    calmar = calmar_ratio(arr, periods_per_year)
    wr = win_rate(arr)
    pf = profit_factor(arr)
    
    return {
        "annualized_return_cagr": cagr,
        "annualized_return_arithmetic": arithmetic_ret,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "max_drawdown_duration_bars": mdd_duration,
        "calmar_ratio": calmar,
        "win_rate": wr,
        "profit_factor": pf,
        "total_observations": len(arr)
    }
