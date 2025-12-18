"""
Combinatorially Symmetric Cross-Validation (CSCV) Partitioner
============================================================
Mathematical implementation of CSCV following:
Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. V. (2015/2016).
"The Probability of Backtest Overfitting", Journal of Computational Finance.

CSCV partitions a return matrix of T observations into S contiguous slices,
forming all C(S, S/2) symmetric combinations of size S/2.
"""

from itertools import combinations
from math import comb
from typing import List, Tuple
import numpy as np


class CSCVPartitioner:
    """
    Partitions time series data into S contiguous blocks and generates
    all symmetric C(S, S/2) train/test split combinations.
    
    Parameters
    ----------
    n_groups : int, default 10
        Number of contiguous groups S (must be even and >= 4).
    """

    def __init__(self, n_groups: int = 10):
        if n_groups % 2 != 0:
            raise ValueError(f"n_groups must be an even integer, got {n_groups}")
        if n_groups < 4:
            raise ValueError(f"n_groups must be >= 4, got {n_groups}")
            
        self.n_groups = n_groups
        self.half_groups = n_groups // 2
        self.num_splits = comb(n_groups, self.half_groups)

    def get_group_bounds(self, n_obs: int) -> List[Tuple[int, int]]:
        """
        Compute start and end indices for each of the S contiguous blocks.
        
        Parameters
        ----------
        n_obs : int
            Total observations T.
            
        Returns
        -------
        List of (start_idx, end_idx) tuples for S contiguous slices.
        """
        if n_obs < self.n_groups:
            raise ValueError(f"Observations ({n_obs}) must be >= n_groups ({self.n_groups})")
            
        base_size = n_obs // self.n_groups
        remainder = n_obs % self.n_groups
        
        bounds = []
        start = 0
        for i in range(self.n_groups):
            # Distribute remainder evenly across the first blocks
            size = base_size + (1 if i < remainder else 0)
            end = start + size
            bounds.append((start, end))
            start = end
            
        return bounds

    def generate_splits(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """
        Generate all C(S, S/2) symmetric (train_indices, test_indices) tuples.
        
        Returns
        -------
        splits : List of (train_group_ids, test_group_ids)
            Length is C(S, S/2).
        """
        all_ids = set(range(self.n_groups))
        combos = list(combinations(range(self.n_groups), self.half_groups))
        
        splits = []
        for train_groups in combos:
            test_groups = tuple(sorted(all_ids - set(train_groups)))
            splits.append((train_groups, test_groups))
            
        return splits

    def partition_matrix(self, matrix: np.ndarray) -> List[np.ndarray]:
        """
        Split a (T, N) matrix into S contiguous sub-arrays along axis 0.
        """
        n_obs = matrix.shape[0]
        bounds = self.get_group_bounds(n_obs)
        return [matrix[start:end] for start, end in bounds]
