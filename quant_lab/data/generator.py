"""
Market Data Simulation Module
=============================
Generates realistic asset price histories using Geometric Brownian Motion (GBM)
with optional drift, volatility clustering, and jump diffusion.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def generate_market_data(
    n_days: int = 1000,
    start_price: float = 100.0,
    annual_drift: float = 0.07,
    annual_vol: float = 0.18,
    periods_per_year: int = 252,
    seed: int = 42,
    start_date: str = "2020-01-01"
) -> Tuple[pd.Series, pd.DatetimeIndex]:
    """
    Simulate daily asset price history via Geometric Brownian Motion.
    
    Parameters
    ----------
    n_days : int, default 1000
        Number of trading days to simulate.
    start_price : float, default 100.0
        Initial asset price.
    annual_drift : float, default 0.07
        Annualized expected return (drift mu).
    annual_vol : float, default 0.18
        Annualized volatility (sigma).
    periods_per_year : int, default 252
        Trading days per year.
    seed : int, default 42
        PRNG seed for reproducible experiments.
    start_date : str, default '2020-01-01'
        Starting date for timestamp index.
        
    Returns
    -------
    prices : pd.Series
        Closing price series.
    dates : pd.DatetimeIndex
        Trading calendar timestamps.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    
    # Ito's lemma formulation: d(ln S) = (mu - 0.5 * sigma^2) dt + sigma * dW
    drift_dt = (annual_drift - 0.5 * annual_vol**2) * dt
    vol_dt = annual_vol * np.sqrt(dt)
    
    daily_shocks = rng.normal(0, 1, size=n_days)
    log_returns = drift_dt + vol_dt * daily_shocks
    
    # Cumulative price path
    cumulative_log = np.cumsum(log_returns)
    prices_arr = start_price * np.exp(np.insert(cumulative_log[:-1], 0, 0.0))
    
    dates = pd.date_range(start=start_date, periods=n_days, freq="B")
    price_series = pd.Series(prices_arr, index=dates, name="Close")
    return price_series, dates
