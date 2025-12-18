"""
End-to-End Quantitative Strategy Research & Validation Demonstration
=====================================================================
Executes the complete research workflow:
1. Simulates market asset price series.
2. Generates a multi-strategy parameter sweep (Moving Average crossovers).
3. Backtests all strategies with realistic transaction costs & slippage.
4. Performs Chronological Out-of-Sample (OOS) holdout validation.
5. Performs Walk-Forward rolling re-optimization validation.
6. Evaluates Combinatorially Symmetric Cross-Validation (CSCV) and PBO.
7. Generates unified tear sheet and publication-quality diagnostic charts.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure quant_lab is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from quant_lab.data.generator import generate_market_data
from quant_lab.engine.transaction_costs import TransactionCostModel
from quant_lab.strategies.universe import StrategyUniverse
from quant_lab.validation.chronological_oos import ChronologicalOOSValidator
from quant_lab.validation.walk_forward import WalkForwardValidator
from quant_lab.validation.pbo import ProbabilityOfBacktestOverfitting
from quant_lab.reporting.tear_sheet import QuantitativeTearSheet
from quant_lab.reporting.visualization import (
    plot_backtest_performance,
    plot_pbo_diagnostics,
    plot_walk_forward_results,
)


def main():
    print("=" * 80)
    print("  QUANTITATIVE STRATEGY RESEARCH & BACKTESTING LAB")
    print("  End-to-End Research, Friction Modeling & Overfitting Validation")
    print("=" * 80)

    reports_dir = project_root / "reports"
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Market Data Generation
    print("\n[Step 1/6] Simulating market data (1,000 trading days ~ 4 years)...")
    prices, dates = generate_market_data(
        n_days=1000,
        start_price=100.0,
        annual_drift=0.08,
        annual_vol=0.18,
        seed=42
    )
    print(f"  Generated {len(prices)} bars: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # 2. Strategy Universe Creation
    print("\n[Step 2/6] Building Strategy Universe: Moving Average Parameter Sweep...")
    # Fast windows: 5, 10, 15, 20
    # Slow windows: 40, 50, 60, 75, 100
    universe = StrategyUniverse.create_moving_average_sweep(
        fast_windows=[5, 10, 15, 20],
        slow_windows=[40, 50, 60, 75, 100],
        kind="ema",
        long_only=True
    )
    print(f"  Total strategy configurations in universe: {len(universe.strategies)}")

    # 3. Backtest Execution with Transaction Costs
    print("\n[Step 3/6] Running backtests with realistic transaction friction...")
    cost_model = TransactionCostModel(
        commission_bps=5.0,     # 5 bps brokerage
        half_spread_bps=2.0,    # 2 bps bid-ask half spread
        slippage_bps=3.0,       # 3 bps linear slippage
        fixed_fee_per_trade=1.0 # $1 fixed ticket fee
    )
    print(f"  Friction Model: {cost_model.total_linear_bps:.1f} bps linear turnover cost + $1 ticket fee")

    # Run universe net of costs
    returns_matrix, strategy_names, results = universe.run_all(
        prices=prices,
        dates=dates,
        cost_model=cost_model,
        net=True
    )
    print(f"  Strategy return matrix shape: {returns_matrix.shape} (Observations x Strategies)")

    # Select representative strategy for detailed reporting
    sample_strat_idx = 0
    sample_result = results[sample_strat_idx]

    # 4. Chronological Out-of-Sample Holdout
    print("\n[Step 4/6] Conducting Chronological Out-of-Sample Holdout (70% IS / 30% OOS)...")
    oos_validator = ChronologicalOOSValidator(train_ratio=0.70)
    oos_result = oos_validator.evaluate(
        returns_matrix=returns_matrix,
        strategy_names=strategy_names,
        dates=dates
    )
    print(f"  Selected IS Best: {oos_result.best_strategy_name}")
    print(f"  IS Sharpe: {oos_result.is_metrics['sharpe_ratio']:.3f} -> OOS Sharpe: {oos_result.oos_metrics['sharpe_ratio']:.3f}")
    print(f"  OOS Rank:  Rank {oos_result.oos_rank} of {oos_result.total_strategies}")

    # 5. Walk-Forward Validation
    print("\n[Step 5/6] Executing Walk-Forward Validation (252-day train, 63-day test)...")
    wf_validator = WalkForwardValidator(train_window=252, test_window=63, anchored=False)
    wf_result = wf_validator.evaluate(returns_matrix=returns_matrix, strategy_names=strategy_names)
    print(f"  Completed {wf_result.total_windows} rolling windows across timeline")
    print(f"  Walk-Forward Stitched CAGR: {wf_result.walk_forward_metrics['annualized_return_cagr']:.2%}")
    print(f"  Walk-Forward Stitched Sharpe: {wf_result.walk_forward_metrics['sharpe_ratio']:.3f}")

    # 6. CSCV and Probability of Backtest Overfitting (PBO)
    print("\n[Step 6/6] Computing CSCV Partitions & Probability of Backtest Overfitting (PBO)...")
    pbo_engine = ProbabilityOfBacktestOverfitting(n_groups=10)
    pbo_result = pbo_engine.fit(returns_matrix=returns_matrix, verbose=True)

    # Generate Tear Sheet Report
    tear_sheet_text = QuantitativeTearSheet.generate_full_report(
        backtest_result=sample_result,
        oos_result=oos_result,
        wf_result=wf_result,
        pbo_result=pbo_result,
        title="Quantitative Strategy Research & Validation Tear Sheet"
    )

    report_file = reports_dir / "lab_validation_report.txt"
    report_file.write_text(tear_sheet_text, encoding="utf-8")
    print(f"\nSaved Comprehensive Report: {report_file}")

    # Generate Diagnostic Visualizations
    print("\nGenerating publication-grade diagnostic plots...")
    plot_backtest_performance(
        sample_result,
        save_path=str(plots_dir / "equity_and_transaction_costs.png")
    )
    plot_pbo_diagnostics(
        pbo_result,
        save_path=str(plots_dir / "pbo_diagnostics.png")
    )
    plot_walk_forward_results(
        wf_result,
        save_path=str(plots_dir / "walk_forward_progression.png")
    )
    print(f"Saved Diagnostic Charts to: {plots_dir}")

    # Print summary to terminal
    print("\n" + tear_sheet_text)


if __name__ == "__main__":
    main()
