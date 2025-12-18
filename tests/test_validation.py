"""
Unit Tests for Chronological Holdout and Walk-Forward Validation
===============================================================
Verifies that chronological holdout and rolling walk-forward methods
execute without look-ahead leaks or boundary overlaps.
"""

import numpy as np
import pandas as pd
import pytest

from quant_lab.validation.chronological_oos import ChronologicalOOSValidator
from quant_lab.validation.walk_forward import WalkForwardValidator


def test_chronological_holdout_split():
    """Verify clean chronological partitioning without temporal leakage."""
    np.random.seed(42)
    # 500 days, 3 strategies
    n_days = 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    
    rng = np.random.default_rng(42)
    # Strategy 0: strong positive in IS, strong negative in OOS
    # Strategy 1: consistently positive throughout
    # Strategy 2: zero mean throughout
    split_idx = int(n_days * 0.70)  # 350
    noise = rng.normal(0, 0.001, size=n_days)
    
    returns_strat_0 = np.concatenate([np.full(split_idx, 0.002), np.full(n_days - split_idx, -0.002)]) + noise
    returns_strat_1 = np.full(n_days, 0.0005) + noise
    returns_strat_2 = np.full(n_days, 0.0) + noise
    
    matrix = np.column_stack([returns_strat_0, returns_strat_1, returns_strat_2])
    
    validator = ChronologicalOOSValidator(train_ratio=0.70)
    res = validator.evaluate(matrix, strategy_names=["StratA", "StratB", "StratC"], dates=dates)
    
    assert res.split_index == 350
    # StratA had the highest return in IS (0.001 vs 0.0005)
    assert res.best_strategy_index == 0
    assert res.best_strategy_name == "StratA"
    
    # In OOS, StratA had negative return (-0.001), while StratB had +0.0005 and StratC had 0.0.
    # So StratA is worst (Rank 3 out of 3)!
    assert res.oos_rank == 3
    assert res.total_strategies == 3
    assert res.oos_metrics["annualized_return_cagr"] < 0.0


def test_walk_forward_window_progression():
    """Verify non-overlapping sequential test windows in walk-forward evaluation."""
    n_days = 600
    # 2 strategies
    matrix = np.zeros((n_days, 2))
    matrix[:, 0] = 0.0004
    matrix[:, 1] = 0.0002
    
    train_win = 200
    test_win = 50
    
    wf = WalkForwardValidator(train_window=train_win, test_window=test_win, anchored=False)
    res = wf.evaluate(matrix)
    
    # Number of possible windows: (600 - 200) // 50 = 400 // 50 = 8 windows
    assert res.total_windows == 8
    assert len(res.windows) == 8
    
    # Check that windows advance properly
    for idx, w in enumerate(res.windows):
        expected_test_start = train_win + idx * test_win
        expected_test_end = expected_test_start + test_win
        assert w.test_start == expected_test_start
        assert w.test_end == expected_test_end
        assert len(w.test_returns) == test_win
        
    # Check stitched return length
    assert len(res.stitched_oos_returns) == 8 * 50
    assert len(res.stitched_oos_equity) == 8 * 50
