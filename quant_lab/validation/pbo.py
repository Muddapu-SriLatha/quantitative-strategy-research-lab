"""
Probability of Backtest Overfitting (PBO) Module
=================================================
Implementation of Probability of Backtest Overfitting (PBO) via
Combinatorially Symmetric Cross-Validation (CSCV).

References:
-----------
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. V. (2015/2016).
  "The Probability of Backtest Overfitting", Journal of Computational Finance, 20(4), 39–69.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quant_lab.validation.cscv import CSCVPartitioner


@dataclass
class PBOResult:
    """Consolidated PBO and CSCV diagnostic results."""
    pbo: float
    probability_of_loss: float
    mean_is_sharpe: float
    mean_oos_sharpe: float
    performance_degradation: float
    median_relative_rank: float
    best_is_worst_oos_prob: float
    num_splits: int
    num_strategies: int
    n_groups: int
    is_sharpes_selected: np.ndarray
    oos_sharpes_selected: np.ndarray
    oos_ranks_selected: np.ndarray
    relative_ranks_selected: np.ndarray
    logits: np.ndarray
    all_is_sharpes: np.ndarray
    all_oos_sharpes: np.ndarray

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """Return human-readable summary metrics."""
        return {
            "PBO (Bottom 50% OOS Risk)": f"{self.pbo:.2%}",
            "Probability of OOS Loss (Sharpe <= 0)": f"{self.probability_of_loss:.2%}",
            "Mean IS Sharpe (Selected)": f"{self.mean_is_sharpe:.3f}",
            "Mean OOS Sharpe (Realized)": f"{self.mean_oos_sharpe:.3f}",
            "Performance Degradation": f"{self.performance_degradation:.2%}",
            "Median OOS Relative Rank": f"{self.median_relative_rank:.3f}",
            "Best IS is Worst OOS Prob": f"{self.best_is_worst_oos_prob:.2%}",
            "Strategy Universe Size (N)": self.num_strategies,
            "Combinatorial Splits (C)": self.num_splits,
            "CSCV Groups (S)": self.n_groups,
        }


def vectorized_sharpe(
    matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> np.ndarray:
    """
    Compute annualized Sharpe ratio across columns of a 2D return matrix in parallel.
    
    Parameters
    ----------
    matrix : np.ndarray, shape (T, N)
        Return series of N strategies over T time steps.
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess = matrix - rf_per_period
    n_obs = excess.shape[0]
    
    if n_obs < 2:
        return np.zeros(matrix.shape[1])
        
    mean_excess = np.mean(excess, axis=0)
    std_excess = np.std(excess, axis=0, ddof=1)
    
    # Avoid division by zero
    zero_vol = std_excess <= 1e-12
    safe_std = np.where(zero_vol, 1.0, std_excess)
    sharpes = (mean_excess / safe_std) * np.sqrt(periods_per_year)
    sharpes[zero_vol] = 0.0
    return sharpes


class ProbabilityOfBacktestOverfitting:
    """
    High-performance, vectorized PBO analysis engine.
    
    Parameters
    ----------
    n_groups : int, default 10
        Number of contiguous CSCV partitions (must be even >= 4).
    risk_free_rate : float, default 0.0
        Annualized risk-free rate for Sharpe calculations.
    periods_per_year : int, default 252
        Number of trading intervals per year.
    """

    def __init__(
        self,
        n_groups: int = 10,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        self.partitioner = CSCVPartitioner(n_groups=n_groups)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def fit(
        self,
        returns_matrix: Union[np.ndarray, pd.DataFrame],
        verbose: bool = False
    ) -> PBOResult:
        """
        Execute PBO estimation across all combinatorial splits.
        
        Parameters
        ----------
        returns_matrix : np.ndarray or pd.DataFrame, shape (T, N)
            Strategy returns where rows are time periods and columns are candidate strategies.
        verbose : bool, default False
            If True, prints progress details.
            
        Returns
        -------
        PBOResult
            Rich container with statistical diagnostics and distributions.
        """
        if isinstance(returns_matrix, pd.DataFrame):
            matrix = returns_matrix.values.astype(float)
        else:
            matrix = np.asarray(returns_matrix, dtype=float)
            
        if matrix.ndim != 2:
            raise ValueError(f"returns_matrix must be 2-dimensional (T, N), got shape {matrix.shape}")
            
        n_obs, n_strategies = matrix.shape
        if n_strategies < 2:
            raise ValueError(f"PBO requires at least 2 strategies to evaluate selection bias, got {n_strategies}")
        if n_obs < self.partitioner.n_groups * 2:
            raise ValueError(f"Observations ({n_obs}) too small for {self.partitioner.n_groups} CSCV groups")

        # 1. Pre-partition matrix into S contiguous block arrays
        blocks = self.partitioner.partition_matrix(matrix)
        splits = self.partitioner.generate_splits()
        num_splits = len(splits)

        if verbose:
            print(f"Running PBO: {n_strategies} strategies, {n_obs} observations, {num_splits} CSCV splits...")

        best_is_idx = np.empty(num_splits, dtype=int)
        best_is_sharpes = np.empty(num_splits, dtype=float)
        best_is_oos_sharpes = np.empty(num_splits, dtype=float)
        oos_ranks_selected = np.empty(num_splits, dtype=float)
        relative_ranks = np.empty(num_splits, dtype=float)
        logits = np.empty(num_splits, dtype=float)

        all_is_sharpes = np.empty((num_splits, n_strategies), dtype=float)
        all_oos_sharpes = np.empty((num_splits, n_strategies), dtype=float)

        # 2. Iterate through all C(S, S/2) splits with vectorized Sharpe calculations
        for split_idx, (train_groups, test_groups) in enumerate(splits):
            train_data = np.concatenate([blocks[g] for g in train_groups], axis=0)
            test_data = np.concatenate([blocks[g] for g in test_groups], axis=0)

            # Vectorized Sharpe across all N strategies simultaneously
            is_sr = vectorized_sharpe(
                train_data,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year
            )
            oos_sr = vectorized_sharpe(
                test_data,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year
            )

            all_is_sharpes[split_idx, :] = is_sr
            all_oos_sharpes[split_idx, :] = oos_sr

            # Identify the best In-Sample strategy
            # Tie-breaking defaults to first encountered
            best_idx = int(np.argmax(is_sr))
            best_is_idx[split_idx] = best_idx
            best_is_sharpes[split_idx] = is_sr[best_idx]
            best_is_oos_sharpes[split_idx] = oos_sr[best_idx]

            # Compute Out-of-Sample ranks:
            # Rank 1 = highest OOS Sharpe (best)
            # Rank N = lowest OOS Sharpe (worst)
            # rankdata with -oos_sr gives rank 1 to highest Sharpe
            oos_ranks = rankdata(-oos_sr, method="average")
            r_c = oos_ranks[best_idx]
            oos_ranks_selected[split_idx] = r_c

            # Normalized relative rank: omega_bar = r_c / (N + 1) in (0, 1)
            # Following Bailey et al. (2016):
            omega_bar = r_c / (n_strategies + 1.0)
            relative_ranks[split_idx] = omega_bar

            # Logit lambda_c = ln(omega_bar / (1 - omega_bar))
            logit_val = np.log(omega_bar / (1.0 - omega_bar))
            logits[split_idx] = logit_val

        # 3. Calculate PBO and core statistics
        # In descending ranking (1=best, N=worst):
        # omega_bar > 0.5 means strategy performed in the bottom half out-of-sample!
        # Equivalently, oos_rank > n_strategies / 2
        pbo = float(np.mean(oos_ranks_selected > (n_strategies / 2.0)))
        
        # Probability of loss: OOS Sharpe of selected strategy <= 0
        prob_loss = float(np.mean(best_is_oos_sharpes <= 0.0))

        # Expected performance degradation
        mean_is = float(np.mean(best_is_sharpes))
        mean_oos = float(np.mean(best_is_oos_sharpes))
        if abs(mean_is) > 1e-12:
            degradation = (mean_is - mean_oos) / abs(mean_is)
        else:
            degradation = 0.0

        median_rel_rank = float(np.median(relative_ranks))
        best_is_worst_prob = float(np.mean(oos_ranks_selected == n_strategies))

        return PBOResult(
            pbo=pbo,
            probability_of_loss=prob_loss,
            mean_is_sharpe=mean_is,
            mean_oos_sharpe=mean_oos,
            performance_degradation=degradation,
            median_relative_rank=median_rel_rank,
            best_is_worst_oos_prob=best_is_worst_prob,
            num_splits=num_splits,
            num_strategies=n_strategies,
            n_groups=self.partitioner.n_groups,
            is_sharpes_selected=best_is_sharpes,
            oos_sharpes_selected=best_is_oos_sharpes,
            oos_ranks_selected=oos_ranks_selected,
            relative_ranks_selected=relative_ranks,
            logits=logits,
            all_is_sharpes=all_is_sharpes,
            all_oos_sharpes=all_oos_sharpes
        )
