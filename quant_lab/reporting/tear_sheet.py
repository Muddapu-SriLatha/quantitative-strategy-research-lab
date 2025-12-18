"""
Comprehensive Quantitative Tear Sheet Module
============================================
Consolidates performance across:
1. Gross vs Net of transaction costs
2. Chronological In-Sample vs Out-of-Sample holdout
3. Walk-Forward rolling validation
4. Combinatorially Symmetric Cross-Validation and PBO diagnostics
"""

from typing import Dict, Optional
import pandas as pd

from quant_lab.engine.backtest import BacktestResult
from quant_lab.validation.chronological_oos import ChronologicalOOSResult
from quant_lab.validation.walk_forward import WalkForwardResult
from quant_lab.validation.pbo import PBOResult


class QuantitativeTearSheet:
    """Consolidated reporting generator for quantitative research."""

    @staticmethod
    def generate_full_report(
        backtest_result: Optional[BacktestResult] = None,
        oos_result: Optional[ChronologicalOOSResult] = None,
        wf_result: Optional[WalkForwardResult] = None,
        pbo_result: Optional[PBOResult] = None,
        title: str = "Quantitative Strategy Research & Validation Report"
    ) -> str:
        """
        Format a unified text/markdown tear sheet.
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"  {title.upper()}")
        lines.append("=" * 80)

        # 1. Transaction Cost Impact
        if backtest_result is not None:
            lines.append("\n[1] TRANSACTION COST & FRICTION ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Strategy Name:         {backtest_result.strategy_name}")
            lines.append(f"Total Trade Turnover:  {backtest_result.total_trades} rebalance events")
            lines.append(f"Total Friction Drag:   {backtest_result.cost_drag_total:.2%} cumulative return lost to costs")
            lines.append(f"Cost Model Specs:      {backtest_result.cost_model.commission_bps:.1f} bps comm + "
                         f"{backtest_result.cost_model.half_spread_bps:.1f} bps half-spread + "
                         f"{backtest_result.cost_model.slippage_bps:.1f} bps slippage")
            lines.append("\nPerformance Comparison (Pre- vs Post-Friction):")
            summary_df = backtest_result.summary()
            lines.append(summary_df.to_string(index=False))

        # 2. Chronological Out-of-Sample Holdout
        if oos_result is not None:
            lines.append("\n\n[2] CHRONOLOGICAL OUT-OF-SAMPLE (OOS) HOLDOUT EVALUATION")
            lines.append("-" * 80)
            lines.append(f"Partition Scheme:      {oos_result.split_ratio:.0%} In-Sample / "
                         f"{1 - oos_result.split_ratio:.0%} Out-of-Sample Holdout")
            lines.append(f"Selected Best IS:      {oos_result.best_strategy_name} (Index {oos_result.best_strategy_index})")
            lines.append(f"Realized OOS Rank:     Rank {oos_result.oos_rank} out of {oos_result.total_strategies} strategies")
            lines.append(f"Sharpe Ratio Decay:    {oos_result.performance_decay:.2%} "
                         f"({oos_result.is_metrics['sharpe_ratio']:.3f} IS -> {oos_result.oos_metrics['sharpe_ratio']:.3f} OOS)")
            lines.append("\nIn-Sample vs Holdout Metrics:")
            lines.append(oos_result.summary().to_string(index=False))

        # 3. Walk-Forward Validation
        if wf_result is not None:
            lines.append("\n\n[3] WALK-FORWARD VALIDATION (ADAPTIVE RE-OPTIMIZATION)")
            lines.append("-" * 80)
            lines.append(f"Total Windows:         {wf_result.total_windows}")
            lines.append(f"Strategy Switches:     {wf_result.strategy_switch_count} changes across windows")
            lines.append(f"Walk-Forward CAGR:     {wf_result.walk_forward_metrics['annualized_return_cagr']:.2%}")
            lines.append(f"Walk-Forward Sharpe:   {wf_result.walk_forward_metrics['sharpe_ratio']:.3f}")
            lines.append(f"Walk-Forward Max DD:   {wf_result.walk_forward_metrics['max_drawdown']:.2%}")
            lines.append("\nWindow Breakdown (First 5 and Last 5):")
            w_df = wf_result.window_summary()
            if len(w_df) > 10:
                display_df = pd.concat([w_df.head(5), w_df.tail(5)])
            else:
                display_df = w_df
            lines.append(display_df.to_string(index=False))

        # 4. CSCV and PBO Diagnostics
        if pbo_result is not None:
            lines.append("\n\n[4] COMBINATORIAL SYMMETRIC CROSS-VALIDATION & PBO DIAGNOSTICS")
            lines.append("-" * 80)
            pbo_sum = pbo_result.summary()
            for k, v in pbo_sum.items():
                lines.append(f"{k:<42} : {v}")

            lines.append("\nEmpirical Risk Assessment:")
            pbo_val = pbo_result.pbo
            if pbo_val < 0.15:
                risk_level = "MINIMAL OVERFITTING RISK - Selection process is robust."
            elif pbo_val < 0.35:
                risk_level = "MODERATE SELECTION BIAS - Standard parameter optimization drag."
            elif pbo_val < 0.50:
                risk_level = "ELEVATED OVERFITTING RISK - Significant probability of underperformance."
            else:
                risk_level = "CRITICAL / SEVERE OVERFITTING - In-sample optimum is driven by random luck."
            lines.append(f"-> Assessment: {risk_level}")

        lines.append("\n" + "=" * 80)
        lines.append("METHODOLOGICAL LIMITATIONS & RISK DISCLOSURE:")
        lines.append("None of walk-forward validation, chronological holdouts, or CSCV/PBO guarantees")
        lines.append("future profitability. Financial markets exhibit non-stationarity, regime shifts,")
        lines.append("and structural breaks that past data validation cannot fully predict.")
        lines.append("=" * 80)

        return "\n".join(lines)
