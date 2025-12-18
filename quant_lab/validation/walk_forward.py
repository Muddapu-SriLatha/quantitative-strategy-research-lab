"""
Walk-Forward Validation Engine
==============================
Simulates continuous adaptive strategy selection over sequential time windows:
- Rolling or anchored In-Sample training windows
- Forward testing on adjacent unseen test slices
- Stitched cumulative walk-forward equity curve
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from quant_lab.engine.metrics import calculate_all_metrics
from quant_lab.validation.pbo import vectorized_sharpe


@dataclass
class WalkForwardWindow:
    """Detailed record for an individual walk-forward step."""
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    selected_strategy_index: int
    selected_strategy_name: str
    is_sharpe: float
    oos_sharpe: float
    test_returns: np.ndarray


@dataclass
class WalkForwardResult:
    """Comprehensive walk-forward execution diagnostics."""
    windows: List[WalkForwardWindow]
    stitched_oos_returns: np.ndarray
    stitched_oos_equity: np.ndarray
    walk_forward_metrics: Dict[str, float]
    strategy_switch_count: int
    total_windows: int
    anchored: bool

    def window_summary(self) -> pd.DataFrame:
        """Tabulate performance across all walk-forward windows."""
        records = []
        for w in self.windows:
            records.append({
                "Window": w.window_id,
                "Train Range": f"[{w.train_start}:{w.train_end}]",
                "Test Range": f"[{w.test_start}:{w.test_end}]",
                "Selected Strategy": w.selected_strategy_name,
                "IS Sharpe": f"{w.is_sharpe:.3f}",
                "OOS Sharpe": f"{w.oos_sharpe:.3f}",
                "OOS Return": f"{float(np.sum(w.test_returns)):.2%}"
            })
        return pd.DataFrame(records)


class WalkForwardValidator:
    """
    Executes walk-forward cross-validation.
    
    Parameters
    ----------
    train_window : int, default 252
        Number of bars in training/optimization window.
    test_window : int, default 63
        Number of bars in forward testing window (e.g. 1 quarter).
    anchored : bool, default False
        If True, training window expands from t=0. If False, rolling window of fixed size.
    risk_free_rate : float, default 0.0
        Annual risk-free rate.
    periods_per_year : int, default 252
        Trading days per year.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        anchored: bool = False,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        if train_window < 20:
            raise ValueError(f"train_window must be >= 20, got {train_window}")
        if test_window < 5:
            raise ValueError(f"test_window must be >= 5, got {test_window}")
            
        self.train_window = train_window
        self.test_window = test_window
        self.anchored = anchored
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def evaluate(
        self,
        returns_matrix: Union[np.ndarray, pd.DataFrame],
        strategy_names: Optional[List[str]] = None
    ) -> WalkForwardResult:
        """
        Execute walk-forward validation across the returns matrix.
        
        Parameters
        ----------
        returns_matrix : np.ndarray or pd.DataFrame, shape (T, N)
            Returns of N candidate strategies across T periods.
        strategy_names : List[str], optional
            Names for the N candidate strategies.
        """
        if isinstance(returns_matrix, pd.DataFrame):
            if strategy_names is None:
                strategy_names = list(returns_matrix.columns)
            matrix = returns_matrix.values.astype(float)
        else:
            matrix = np.asarray(returns_matrix, dtype=float)

        n_obs, n_strategies = matrix.shape
        if strategy_names is None:
            strategy_names = [f"Strategy_{i+1}" for i in range(n_strategies)]

        min_required = self.train_window + self.test_window
        if n_obs < min_required:
            raise ValueError(f"Observations ({n_obs}) < train_window + test_window ({min_required})")

        windows: List[WalkForwardWindow] = []
        stitched_slices: List[np.ndarray] = []
        current_train_start = 0
        current_train_end = self.train_window
        window_id = 1
        previous_selected = None
        switch_count = 0

        while current_train_end + self.test_window <= n_obs:
            test_start = current_train_end
            test_end = test_start + self.test_window

            # In-Sample slice
            is_slice = matrix[current_train_start:current_train_end]
            # Out-of-Sample slice
            oos_slice = matrix[test_start:test_end]

            # Vectorized Sharpe optimization
            is_sr = vectorized_sharpe(
                is_slice,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year
            )
            best_idx = int(np.argmax(is_sr))
            best_name = strategy_names[best_idx]

            # Evaluate forward on unseen test slice
            test_returns = oos_slice[:, best_idx]
            oos_sr = vectorized_sharpe(
                oos_slice,
                risk_free_rate=self.risk_free_rate,
                periods_per_year=self.periods_per_year
            )[best_idx]

            if previous_selected is not None and best_idx != previous_selected:
                switch_count += 1
            previous_selected = best_idx

            win = WalkForwardWindow(
                window_id=window_id,
                train_start=current_train_start,
                train_end=current_train_end,
                test_start=test_start,
                test_end=test_end,
                selected_strategy_index=best_idx,
                selected_strategy_name=best_name,
                is_sharpe=float(is_sr[best_idx]),
                oos_sharpe=float(oos_sr),
                test_returns=test_returns
            )
            windows.append(win)
            stitched_slices.append(test_returns)

            # Advance window
            if not self.anchored:
                current_train_start += self.test_window
            current_train_end += self.test_window
            window_id += 1

        stitched_returns = np.concatenate(stitched_slices)
        stitched_equity = np.cumprod(1.0 + stitched_returns)
        wf_metrics = calculate_all_metrics(
            stitched_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )

        return WalkForwardResult(
            windows=windows,
            stitched_oos_returns=stitched_returns,
            stitched_oos_equity=stitched_equity,
            walk_forward_metrics=wf_metrics,
            strategy_switch_count=switch_count,
            total_windows=len(windows),
            anchored=self.anchored
        )
