"""
Chronological Out-of-Sample Validation Module
=============================================
Evaluates strategies using strict chronological partitioning:
- In-Sample (IS) window for model/parameter selection
- Out-of-Sample (OOS) holdout window strictly for final verification

Guarantees zero future look-ahead leakage across the time boundary.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quant_lab.engine.metrics import calculate_all_metrics
from quant_lab.validation.pbo import vectorized_sharpe


@dataclass
class ChronologicalOOSResult:
    """Encapsulates chronological out-of-sample holdout findings."""
    split_ratio: float
    split_index: int
    split_date: Optional[pd.Timestamp]
    best_strategy_name: str
    best_strategy_index: int
    is_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    performance_decay: float
    oos_rank: int
    total_strategies: int
    is_returns: np.ndarray
    oos_returns: np.ndarray
    is_equity: np.ndarray
    oos_equity: np.ndarray

    def summary(self) -> pd.DataFrame:
        keys = [
            ("CAGR Return", "annualized_return_cagr", True),
            ("Annual Volatility", "annualized_volatility", True),
            ("Sharpe Ratio", "sharpe_ratio", False),
            ("Sortino Ratio", "sortino_ratio", False),
            ("Max Drawdown", "max_drawdown", True)
        ]
        labels = []
        is_formatted = []
        oos_formatted = []
        for label, k, is_pct in keys:
            labels.append(label)
            is_val = self.is_metrics.get(k, 0.0)
            oos_val = self.oos_metrics.get(k, 0.0)
            if is_pct:
                is_formatted.append(f"{is_val:.2%}")
                oos_formatted.append(f"{oos_val:.2%}")
            else:
                is_formatted.append(f"{is_val:.3f}")
                oos_formatted.append(f"{oos_val:.3f}")
                
        df = pd.DataFrame({
            "Metric": labels,
            "In-Sample (Training)": is_formatted,
            "Out-of-Sample (Holdout)": oos_formatted
        })
        return df


class ChronologicalOOSValidator:
    """
    Chronological Train/Test Holdout Evaluator.
    
    Parameters
    ----------
    train_ratio : float, default 0.70
        Fraction of observations allocated to In-Sample selection.
    risk_free_rate : float, default 0.0
        Annual risk-free rate.
    periods_per_year : int, default 252
        Trading intervals per year.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        if not (0.1 < train_ratio < 0.95):
            raise ValueError(f"train_ratio must be between 0.10 and 0.95, got {train_ratio}")
        self.train_ratio = train_ratio
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def evaluate(
        self,
        returns_matrix: Union[np.ndarray, pd.DataFrame],
        strategy_names: Optional[List[str]] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> ChronologicalOOSResult:
        """
        Execute chronological train/holdout validation.
        
        Parameters
        ----------
        returns_matrix : np.ndarray or pd.DataFrame, shape (T, N)
            Return matrix for N candidate strategies over T periods.
        strategy_names : List[str], optional
            Names for the N strategies.
        dates : pd.DatetimeIndex, optional
            Timestamp index.
        """
        if isinstance(returns_matrix, pd.DataFrame):
            if strategy_names is None:
                strategy_names = list(returns_matrix.columns)
            if dates is None and isinstance(returns_matrix.index, pd.DatetimeIndex):
                dates = returns_matrix.index
            matrix = returns_matrix.values.astype(float)
        else:
            matrix = np.asarray(returns_matrix, dtype=float)

        n_obs, n_strategies = matrix.shape
        if strategy_names is None:
            strategy_names = [f"Strategy_{i+1}" for i in range(n_strategies)]

        split_idx = int(n_obs * self.train_ratio)
        split_date = dates[split_idx] if dates is not None else None

        # 1. Partition along the chronological timeline
        is_matrix = matrix[:split_idx]
        oos_matrix = matrix[split_idx:]

        # 2. Select best strategy based on In-Sample Sharpe
        is_sharpes = vectorized_sharpe(
            is_matrix,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )
        best_idx = int(np.argmax(is_sharpes))
        best_name = strategy_names[best_idx]

        # 3. Evaluate selected strategy out-of-sample
        oos_sharpes = vectorized_sharpe(
            oos_matrix,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )
        
        # OOS rank of selected strategy (1 = highest OOS Sharpe)
        oos_ranks = rankdata(-oos_sharpes, method="average")
        selected_oos_rank = int(round(oos_ranks[best_idx]))

        best_is_returns = is_matrix[:, best_idx]
        best_oos_returns = oos_matrix[:, best_idx]

        is_metrics = calculate_all_metrics(
            best_is_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )
        oos_metrics = calculate_all_metrics(
            best_oos_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year
        )

        is_sr = is_metrics["sharpe_ratio"]
        oos_sr = oos_metrics["sharpe_ratio"]
        decay = (is_sr - oos_sr) / abs(is_sr) if abs(is_sr) > 1e-12 else 0.0

        is_equity = np.cumprod(1.0 + best_is_returns)
        oos_equity = np.cumprod(1.0 + best_oos_returns)

        return ChronologicalOOSResult(
            split_ratio=self.train_ratio,
            split_index=split_idx,
            split_date=split_date,
            best_strategy_name=best_name,
            best_strategy_index=best_idx,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            performance_decay=decay,
            oos_rank=selected_oos_rank,
            total_strategies=n_strategies,
            is_returns=best_is_returns,
            oos_returns=best_oos_returns,
            is_equity=is_equity,
            oos_equity=oos_equity
        )
