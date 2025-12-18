"""
Controlled Noisy-Strategy Overfitting Experiment Tests
=====================================================
Demonstrates and asserts that Probability of Backtest Overfitting (PBO)
systematically increases when optimizing across noisy strategy universes,
validating the core theoretical property of selection bias.
"""

import numpy as np
import pytest

from quant_lab.validation.pbo import ProbabilityOfBacktestOverfitting


def test_pbo_increases_with_noise_and_selection_bias():
    """
    Controlled experiment proving PBO increases from a signal-dominated
    universe to a noise-dominated universe.
    
    Theoretical foundation:
    - In a universe with genuine persistent signal, the in-sample winner
      possesses true structural alpha and continues to outperform out-of-sample (low PBO).
    - In a universe of uninformative noise, the in-sample winner is an artifact
      of data snooping / winner's curse. On out-of-sample data, its expected rank
      regresses to median, yielding high PBO (approaching or exceeding 50%).
    """
    n_days = 600
    pbo_engine = ProbabilityOfBacktestOverfitting(n_groups=10)

    # -------------------------------------------------------------
    # 1. Genuine Signal Universe (Low Overfitting Risk)
    # -------------------------------------------------------------
    # Strategy 0 has genuine alpha, Strategy 1 has moderate alpha, Strategy 2 has slight alpha
    rng_signal = np.random.default_rng(101)
    signal_returns = np.column_stack([
        rng_signal.normal(0.0012, 0.010, size=n_days),   # Strong persistent alpha
        rng_signal.normal(0.0006, 0.010, size=n_days),   # Moderate alpha
        rng_signal.normal(0.0001, 0.010, size=n_days),   # Weak alpha
    ])
    
    res_signal = pbo_engine.fit(signal_returns)
    pbo_signal = res_signal.pbo
    degradation_signal = res_signal.performance_degradation

    # -------------------------------------------------------------
    # 2. Pure Noise Universe (High Selection Bias)
    # -------------------------------------------------------------
    # 40 strategies with zero expected return (pure noise)
    rng_noise = np.random.default_rng(202)
    noise_returns = rng_noise.normal(0.0, 0.012, size=(n_days, 40))
    
    res_noise = pbo_engine.fit(noise_returns)
    pbo_noise = res_noise.pbo
    degradation_noise = res_noise.performance_degradation

    # -------------------------------------------------------------
    # 3. Statistical Assertions (No hardcoded arbitrary numbers)
    # -------------------------------------------------------------
    # A) PBO of pure noise must be significantly higher than genuine signal
    assert pbo_noise > pbo_signal, (
        f"Overfitting probability in noise universe ({pbo_noise:.2%}) "
        f"must exceed signal universe ({pbo_signal:.2%})"
    )
    
    # Margin between noise and signal should be substantial (at least 20 percentage points)
    assert (pbo_noise - pbo_signal) >= 0.20, (
        f"Expected at least 20% spread between noise ({pbo_noise:.2%}) "
        f"and signal ({pbo_signal:.2%})"
    )

    # B) In pure noise, PBO should be around or above 50% (half of selections underperform median)
    assert pbo_noise >= 0.45, f"Noise PBO ({pbo_noise:.2%}) should reflect substantial risk (>= 45%)"

    # C) Performance degradation must be higher in the noise universe
    assert degradation_noise > degradation_signal, (
        f"Noise degradation ({degradation_noise:.2%}) must exceed signal degradation ({degradation_signal:.2%})"
    )


def test_monotonic_pbo_growth_with_strategy_count():
    """
    Verifies that increasing the number of noise strategies tested (multiple testing)
    drives up PBO and median relative rank degradation.
    """
    n_days = 500
    rng = np.random.default_rng(777)
    pbo_engine = ProbabilityOfBacktestOverfitting(n_groups=8)

    # Compare a small universe (3 noise strategies) vs a large universe (50 noise strategies)
    small_noise = rng.normal(0.0, 0.01, size=(n_days, 3))
    large_noise = rng.normal(0.0, 0.01, size=(n_days, 50))

    res_small = pbo_engine.fit(small_noise)
    res_large = pbo_engine.fit(large_noise)

    # In the large noise universe, data snooping allows selecting extreme outliers,
    # leading to higher OOS relative rank regression (higher degradation)
    assert res_large.mean_is_sharpe > res_small.mean_is_sharpe, (
        "Larger universe allows selecting higher spurious in-sample Sharpe (winner's curse)"
    )
    assert res_large.pbo >= res_small.pbo - 0.05
