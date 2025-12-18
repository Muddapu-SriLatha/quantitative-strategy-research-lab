"""
Visualization and Diagnostics Module
====================================
Generates publication-quality quantitative diagnostic charts:
- Gross vs Net equity curves & transaction friction
- In-sample vs Out-of-sample performance degradation scatter
- CSCV relative rank and logit distributions
- Walk-forward multi-window equity tracking
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from quant_lab.engine.backtest import BacktestResult
from quant_lab.validation.pbo import PBOResult
from quant_lab.validation.walk_forward import WalkForwardResult


def plot_backtest_performance(
    result: BacktestResult,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot equity curves before and after transaction costs alongside drawdowns.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    dates = result.dates
    gross_eq = result.gross_equity
    net_eq = result.net_equity
    
    # Upper plot: Equity curves
    ax1.plot(dates, gross_eq, label="Gross (Pre-Friction)", color="#1f77b4", linewidth=1.8)
    ax1.plot(dates, net_eq, label="Net (Post-Friction)", color="#2ca02c", linewidth=2.0)
    ax1.fill_between(dates, gross_eq, net_eq, color="#d62728", alpha=0.18, label="Transaction Cost Drag")
    ax1.set_title(f"Cumulative Performance: {result.strategy_name}", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Wealth Index ($1 Initial)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Lower plot: Drawdowns
    running_max = np.maximum.accumulate(net_eq)
    drawdowns = (net_eq - running_max) / running_max
    ax2.fill_between(dates, drawdowns, 0, color="#d62728", alpha=0.35, label="Net Drawdown")
    ax2.plot(dates, drawdowns, color="#d62728", linewidth=1.0)
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, facecolor="white")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_pbo_diagnostics(
    pbo_result: PBOResult,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate the 3 canonical PBO diagnostic visualisations:
    1. Performance Degradation Scatter (IS Sharpe vs OOS Sharpe of selected strategy)
    2. Distribution of Relative Ranks with PBO bottom-50% threshold
    3. Distribution of Logits (lambda_c)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    is_sr = pbo_result.is_sharpes_selected
    oos_sr = pbo_result.oos_sharpes_selected
    rel_ranks = pbo_result.relative_ranks_selected
    logits = pbo_result.logits

    # 1. Performance Degradation Scatter
    ax = axes[0]
    min_val = min(np.min(is_sr), np.min(oos_sr), 0.0) - 0.2
    max_val = max(np.max(is_sr), np.max(oos_sr)) + 0.2
    
    ax.scatter(is_sr, oos_sr, alpha=0.6, color="#1f77b4", edgecolors="black", linewidth=0.5, s=35)
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="No Degradation (y = x)")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title(f"Degradation Scatter (P(Loss)={pbo_result.probability_of_loss:.1%})", fontsize=11, fontweight="bold")
    ax.set_xlabel("In-Sample Sharpe (Selected)", fontsize=10)
    ax.set_ylabel("Out-of-Sample Sharpe", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # 2. Relative Rank Histogram (PBO)
    ax = axes[1]
    counts, bins, patches = ax.hist(rel_ranks, bins=15, range=(0.0, 1.0), density=True, color="#4c72b0", edgecolor="white", alpha=0.8)
    # Color bars above 0.5 (bottom half) in red
    for patch, b in zip(patches, bins[:-1]):
        if b >= 0.5:
            patch.set_facecolor("#c44e52")
            
    ax.axvline(0.5, color="black", linestyle="--", linewidth=2.0, label="Median Threshold (0.5)")
    ax.set_title(f"OOS Relative Rank (PBO = {pbo_result.pbo:.1%})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative Rank omega_bar (1=Worst, 0=Best)", fontsize=10)
    ax.set_ylabel("Empirical Density", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    # 3. Logit Distribution
    ax = axes[2]
    finite_logits = logits[np.isfinite(logits)]
    if len(finite_logits) > 0:
        ax.hist(finite_logits, bins=15, density=True, color="#55a868", edgecolor="white", alpha=0.8)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.8, label="lambda = 0 (PBO cut)")
    ax.set_title(f"Distribution of Logits (lambda_c)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Logit lambda_c = ln(omega / (1 - omega))", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_walk_forward_results(
    wf_result: WalkForwardResult,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot stitched walk-forward out-of-sample equity curve and per-window Sharpes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    
    # Upper plot: Stitched equity
    ax1.plot(wf_result.stitched_oos_equity, color="#2b5c8f", linewidth=2.0, label="Stitched OOS Walk-Forward")
    ax1.set_title(f"Walk-Forward Validation Equity (Sharpe = {wf_result.walk_forward_metrics['sharpe_ratio']:.2f})",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Wealth Index", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Lower plot: IS vs OOS Sharpe per window
    w_ids = [w.window_id for w in wf_result.windows]
    is_sr = [w.is_sharpe for w in wf_result.windows]
    oos_sr = [w.oos_sharpe for w in wf_result.windows]
    
    width = 0.35
    x = np.arange(len(w_ids))
    ax2.bar(x - width/2, is_sr, width, label="In-Sample Sharpe", color="#4c72b0", alpha=0.85)
    ax2.bar(x + width/2, oos_sr, width, label="Out-of-Sample Sharpe", color="#55a868", alpha=0.85)
    ax2.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Walk-Forward Window Index", fontsize=10)
    ax2.set_ylabel("Sharpe Ratio", fontsize=10)
    ax2.set_xticks(x[::max(1, len(x)//10)])
    ax2.set_xticklabels(w_ids[::max(1, len(x)//10)])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
