"""
Hand-Calculated Verification and Unit Tests for CSCV and PBO
============================================================
Verifies:
1. Exact hand-calculated CSCV partitioning and PBO on a deterministic matrix.
2. Vectorized Sharpe calculations matching single-series Sharpe calculations.
3. Partitioner bounds and combination counts.
"""

import numpy as np
import pytest

from quant_lab.engine.metrics import sharpe_ratio
from quant_lab.validation.cscv import CSCVPartitioner
from quant_lab.validation.pbo import ProbabilityOfBacktestOverfitting, vectorized_sharpe


def test_cscv_partitioner_bounds():
    """Verify exact group boundaries and partition counts."""
    # S = 4, T = 10 -> base_size = 2, remainder = 2 -> sizes: 3, 3, 2, 2
    part = CSCVPartitioner(n_groups=4)
    bounds = part.get_group_bounds(10)
    assert bounds == [(0, 3), (3, 6), (6, 8), (8, 10)]
    assert part.num_splits == 6  # C(4, 2) = 6
    
    splits = part.generate_splits()
    assert len(splits) == 6
    for train_g, test_g in splits:
        assert len(train_g) == 2
        assert len(test_g) == 2
        assert set(train_g).intersection(set(test_g)) == set()
        assert set(train_g).union(set(test_g)) == {0, 1, 2, 3}


def test_vectorized_sharpe_matches_scalar():
    """Verify that vectorized_sharpe yields identical numbers to scalar sharpe_ratio."""
    np.random.seed(123)
    matrix = np.random.normal(0.001, 0.015, size=(250, 5))
    
    vec_sr = vectorized_sharpe(matrix, risk_free_rate=0.02, periods_per_year=252)
    scalar_sr = [sharpe_ratio(matrix[:, col], risk_free_rate=0.02, periods_per_year=252) for col in range(5)]
    
    assert np.allclose(vec_sr, scalar_sr, rtol=1e-7)


def test_exact_hand_calculated_pbo():
    """
    Exhaustive verification against an exact hand-calculated 8x3 matrix.
    
    Setup:
    - S = 4 groups (each 2 rows).
    - C(4, 2) = 6 splits:
        Split 0: Train={0, 1}, Test={2, 3}
        Split 1: Train={0, 2}, Test={1, 3}
        Split 2: Train={0, 3}, Test={1, 2}
        Split 3: Train={1, 2}, Test={0, 3}
        Split 4: Train={1, 3}, Test={0, 2}
        Split 5: Train={2, 3}, Test={0, 1}
    - N = 3 strategies:
        Strat 0: G0=[0.03, 0.01], G1=[0.03, 0.01], G2=[-0.01, -0.03], G3=[-0.01, -0.03]
        Strat 1: G0=[0.015, 0.005], G1=[0.015, 0.005], G2=[0.015, 0.005], G3=[0.015, 0.005]
        Strat 2: G0=[-0.01, -0.03], G1=[-0.01, -0.03], G2=[0.03, 0.01], G3=[0.03, 0.01]
        
    Theoretical derivations by hand:
    - Split 0: IS winner is Strat 0. OOS ranks: Strat 2 (1), Strat 1 (2), Strat 0 (3).
               Strat 0 is rank 3 (> 1.5 -> bottom half).
    - Split 5: IS winner is Strat 2. OOS ranks: Strat 0 (1), Strat 1 (2), Strat 2 (3).
               Strat 2 is rank 3 (> 1.5 -> bottom half).
    - Splits 1, 2, 3, 4: In Train, Strat 0 and Strat 2 have mean 0, Strat 1 has mean +0.01.
               IS winner is Strat 1. OOS rank for Strat 1 is rank 1 (top half).
               
    Hand-calculated PBO:
    Exactly 2 out of 6 splits rank in bottom half.
    PBO = 2 / 6 = 1/3 ~ 0.33333333.
    """
    g0 = np.array([[0.03, 0.015, -0.01], [0.01, 0.001, -0.03]])
    g1 = np.array([[0.03, 0.015, -0.01], [0.01, 0.001, -0.03]])
    g2 = np.array([[-0.01, 0.015, 0.03], [-0.03, 0.001, 0.01]])
    g3 = np.array([[-0.01, 0.015, 0.03], [-0.03, 0.001, 0.01]])
    
    matrix = np.vstack([g0, g1, g2, g3])  # Shape (8, 3)
    assert matrix.shape == (8, 3)

    pbo_engine = ProbabilityOfBacktestOverfitting(n_groups=4, risk_free_rate=0.0)
    res = pbo_engine.fit(matrix)

    # 1. Total splits must be exactly 6
    assert res.num_splits == 6
    assert res.num_strategies == 3

    # 2. Check exact selected strategies per split
    # Split 0: Train {0, 1} -> best is Strat 0
    # Split 5: Train {2, 3} -> best is Strat 2
    # Splits 1, 2, 3, 4: Train has mixed -> best is Strat 1
    assert res.oos_ranks_selected[0] == 3.0
    assert res.oos_ranks_selected[5] == 3.0
    for split_i in [1, 2, 3, 4]:
        assert res.oos_ranks_selected[split_i] == 1.0

    # 3. Exact PBO must be 2 / 6 = 1/3
    expected_pbo = 2.0 / 6.0
    assert np.isclose(res.pbo, expected_pbo, atol=1e-7)

    # 4. Check relative ranks: omega = r / (N + 1) = r / 4
    # rank 3 -> 3/4 = 0.75
    # rank 1 -> 1/4 = 0.25
    assert np.isclose(res.relative_ranks_selected[0], 0.75, atol=1e-7)
    assert np.isclose(res.relative_ranks_selected[5], 0.75, atol=1e-7)
    assert np.isclose(res.relative_ranks_selected[1], 0.25, atol=1e-7)

    # 5. Check logit values: lambda = ln(omega / (1 - omega))
    # omega = 0.75 -> ln(0.75 / 0.25) = ln(3) ~ 1.098612
    # omega = 0.25 -> ln(0.25 / 0.75) = ln(1/3) ~ -1.098612
    assert np.isclose(res.logits[0], np.log(3.0), atol=1e-7)
    assert np.isclose(res.logits[1], np.log(1.0 / 3.0), atol=1e-7)
