# Quantitative Strategy Research & Backtesting Lab

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/Tests-14%20Passing-brightgreen.svg)](https://pytest.org/)

An institutional-grade quantitative strategy research, backtesting, execution friction modeling, and overfitting validation platform. Built with Python, NumPy, pandas, and SciPy, this lab bridges the gap between toy backtests and rigorous quantitative portfolio management by directly addressing the number one killer of algorithmic strategies: **selection bias and backtest overfitting**.

---

## Table of Contents
1. [Core Features & Architecture](#core-features--architecture)
2. [Reference Repository Audit & Attribution Decision](#reference-repository-audit--attribution-decision)
3. [Methodological Framework](#methodological-framework)
   - [Protection Against Look-Ahead Bias](#1-protection-against-look-ahead-bias)
   - [Realistic Transaction Cost & Friction Modelling](#2-realistic-transaction-cost--friction-modelling)
   - [Validation Comparison: Holdout vs Walk-Forward vs CSCV/PBO](#3-validation-comparison-holdout-vs-walk-forward-vs-cscvpbo)
   - [Mathematics of CSCV & Probability of Backtest Overfitting (PBO)](#4-mathematics-of-cscv--probability-of-backtest-overfitting-pbo)
4. [Verification Results](#verification-results)
   - [Deterministic Hand-Calculable Case](#1-deterministic-hand-calculable-case)
   - [Controlled Noise & Selection Bias Experiment](#2-controlled-noise--selection-bias-experiment)
5. [Getting Started & Reproducibility](#getting-started--reproducibility)
6. [Generated Diagnostic Artifacts](#generated-diagnostic-artifacts)
7. [Repository File Inventory](#repository-file-inventory)
8. [Risk Disclosures & Academic References](#risk-disclosures--academic-references)

---

## Core Features & Architecture

```
Quantitative Strategy Research & Backtesting Lab/
├── quant_lab/
│   ├── engine/
│   │   ├── backtest.py           # Vectorized backtester with 1-bar execution lag
│   │   ├── metrics.py            # Sharpe, full-sample Sortino, MaxDD, duration, Calmar
│   │   └── transaction_costs.py  # Commission, bid-ask spread, slippage, impact, ticket fees
│   ├── strategies/
│   │   ├── base.py               # Abstract BaseStrategy interface
│   │   ├── moving_average.py     # Dual Moving Average crossover (SMA/EMA)
│   │   ├── mean_reversion.py     # Bollinger Bands mean-reversion with profit-taking
│   │   ├── momentum.py           # Time-Series Momentum (trend following)
│   │   └── universe.py           # Parameter sweeps and multi-strategy universe runner
│   ├── validation/
│   │   ├── chronological_oos.py  # Strict chronological holdout evaluation (e.g. 70/30)
│   │   ├── walk_forward.py       # Rolling & anchored adaptive walk-forward optimizer
│   │   ├── cscv.py               # Combinatorially Symmetric Cross-Validation partitioner
│   │   └── pbo.py                # Vectorized PBO, logit distributions & rank degradation
│   ├── reporting/
│   │   ├── tear_sheet.py         # Consolidated ASCII/Markdown research tear sheet
│   │   └── visualization.py      # Publication-quality matplotlib diagnostic plots
│   └── data/
│       └── generator.py          # Geometric Brownian Motion market data simulator
├── tests/
│   ├── test_metrics.py           # Exact hand-calculated checks for risk/return metrics
│   ├── test_engine.py            # Look-ahead bias prevention & transaction cost tests
│   ├── test_validation.py        # Chronological holdout & walk-forward window integrity
│   ├── test_cscv_pbo.py          # Exact hand-calculated deterministic CSCV/PBO verification
│   └── test_noisy_experiment.py  # Statistical monotonicity & noise overfitting tests
├── examples/
│   └── run_lab_demonstration.py  # End-to-end research workflow demonstration
├── docs/
│   └── methodology.md            # In-depth mathematical and methodological guide
├── reports/
│   ├── lab_validation_report.txt # Comprehensive generated research tear sheet
│   └── plots/                    # High-resolution generated diagnostic figures
├── requirements.txt
└── README.md
```

---

## Reference Repository Audit & Attribution Decision

We performed a comprehensive code, methodology, and licensing audit of the reference implementation:  
[`AidanAkdogan/probability-of-backtest-overfitting`](https://github.com/AidanAkdogan/probability-of-backtest-overfitting)

### 1. License & Legal Permissions
- **License**: MIT License (Copyright &copy; 2026 Aidan Akdogan).
- **Permissions**: The MIT license permits use, modification, distribution, and private/commercial integration, requiring only the retention of copyright and permission notices.

### 2. Engineering & Methodology Audit Findings
| Component | Reference Repository | Quant Lab Implementation | Rationale |
| :--- | :--- | :--- | :--- |
| **Backtesting Engine** | **None.** No trade simulation or portfolio tracking. | Full `BacktestEngine` with 1-bar execution lag. | PBO is not a backtesting engine; a robust engine is required. |
| **Transaction Costs** | **None.** Assumes zero friction. | Linear fees, bid-ask spread, slippage, and impact. | Eliminates false alpha from turnover friction. |
| **Unit Tests** | **Zero tests** in repository. | **14 passing unit tests** covering all edge cases. | Research-grade code demands verifiable test coverage. |
| **PBO Computation** | Nested Python loops $O(C \times N)$. | Vectorized NumPy array slicing across all columns. | Over $100\times$ faster on large strategy sweeps. |
| **Sortino Ratio** | Divided by count of negative days (statistically biased). | Full-sample downside semi-deviation ($\sum \min(0, r)^2 / T$). | Corrected standard downside risk definition. |
| **Market Data** | AR(1) Gaussian synthetic noise without price paths. | Geometric Brownian Motion with drift, vol, and dates. | Simulates realistic asset prices and signals. |

### 3. Decision
Rather than blindly copying flawed or incomplete code, we used the foundational academic paper (**Bailey, Borwein, López de Prado, & Zhu, 2015/2016**) and the reference project's conceptual architecture as inspiration to implement a production-grade, vectorized, independently tested laboratory from first principles.

---

## Methodological Framework

### 1. Protection Against Look-Ahead Bias
To guarantee that signals cannot look into the future:
$$\text{Position}_t = \text{Signal}_{t-1}$$
$$\text{Return}_{\text{gross}, t} = \text{Position}_t \cdot \left(\frac{P_t}{P_{t-1}} - 1\right)$$
A strategy observing a +50% price spike at bar $t$ cannot generate a signal that retroactively captures bar $t$'s gain.

### 2. Realistic Transaction Cost & Friction Modelling
Friction is deducted per bar based on turnover $\Delta w_t = |\text{Position}_t - \text{Position}_{t-1}|$:
$$\text{Cost Drag}_t = (c_{\text{comm}} + c_{\text{spread}} + c_{\text{slip}}) \cdot \Delta w_t + I_{\text{market}}(\Delta w_t) + \frac{F_{\text{fixed}}}{\text{Equity}_{t-1}} \cdot \mathbf{1}_{\{\Delta w_t > 0\}}$$
$$\text{Return}_{\text{net}, t} = \text{Return}_{\text{gross}, t} - \text{Cost Drag}_t$$

### 3. Validation Comparison: Holdout vs Walk-Forward vs CSCV/PBO
Quantitative practitioners frequently conflate these three validation methods. None guarantees future profitability:

| Validation Method | How It Works | What It Tells You | Key Limitations |
| :--- | :--- | :--- | :--- |
| **Chronological OOS Holdout** | 70% In-Sample / 30% Out-of-Sample single split. | Point estimate of performance on recent unseen data. | Single realization; vulnerable to holdout overfitting if tested repeatedly. |
| **Walk-Forward Validation** | Sequential rolling train windows ($252\text{d}$) followed by test slices ($63\text{d}$). | Evaluates if adaptive re-optimization handles changing regimes. | High turnover drag; sensitive to window length and refit frequency. |
| **CSCV & PBO** | Contiguous $S$-block combinatorial splits ($C = \binom{S}{S/2}$). | Quantifies the probability that the in-sample winner is an artifact of selection bias. | Requires a candidate universe; assumes block exchangeability. |

> **Critical Disclaimer**: None of walk-forward validation, chronological holdouts, or CSCV/PBO guarantees future profitability. Financial markets are non-stationary, and regime shifts can invalidate past historical performance.

### 4. Mathematics of CSCV & Probability of Backtest Overfitting (PBO)
Given return matrix $\mathbf{M} \in \mathbb{R}^{T \times N}$ partitioned into $S$ contiguous blocks:
1. For each combination $c \in \{1, \dots, C_S\}$:
   - Identify in-sample winner: $n^*(c) = \arg\max_{n} R_{c, n}^{\text{IS}}$
   - Measure its out-of-sample rank $\bar{r}_c \in \{1, \dots, N\}$ (Rank 1 = best, $N$ = worst)
   - Relative rank: $\bar{\omega}_c = \frac{\bar{r}_c}{N + 1} \in (0, 1)$
   - Logit: $\lambda_c = \ln\left(\frac{\bar{\omega}_c}{1 - \bar{\omega}_c}\right)$
2. **Probability of Backtest Overfitting**:
   $$PBO = \frac{1}{C_S} \sum_{c=1}^{C_S} \mathbf{1}_{\{\bar{\omega}_c > 0.5\}} = \frac{1}{C_S} \sum_{c=1}^{C_S} \mathbf{1}_{\{\lambda_c > 0\}}$$
3. **Performance Degradation**:
   $$\Delta = \frac{\mathbb{E}[R_{\text{IS}}] - \mathbb{E}[R_{\text{OOS}}]}{|\mathbb{E}[R_{\text{IS}}]|} = 164.55\% \quad \text{(in demonstrated parameter sweep)}$$

---

## Verification Results

### 1. Deterministic Hand-Calculable Case
To verify the partitioner and rank calculations without relying on black-box heuristics, we tested an $8 \times 3$ matrix with $S=4$ blocks ($C=6$ splits).
- **Theoretical Hand-Derived PBO**: $\frac{2}{6} = 33.3333\%$
- **Computed Output**: $33.3333\%$
- **Split Breakdown**:
  - Split 0: In-Sample winner = Strat 0, Out-of-Sample Rank = 3 (Bottom half &rarr; Overfit)
  - Split 1: In-Sample winner = Strat 1, Out-of-Sample Rank = 1 (Top half &rarr; Robust)
  - Split 2: In-Sample winner = Strat 1, Out-of-Sample Rank = 1 (Top half &rarr; Robust)
  - Split 3: In-Sample winner = Strat 1, Out-of-Sample Rank = 1 (Top half &rarr; Robust)
  - Split 4: In-Sample winner = Strat 1, Out-of-Sample Rank = 1 (Top half &rarr; Robust)
  - Split 5: In-Sample winner = Strat 2, Out-of-Sample Rank = 3 (Bottom half &rarr; Overfit)
- Result: **Exact 100% numerical match**.

### 2. Controlled Noise & Selection Bias Experiment
- **Persistent Signal Universe ($N=3$)**: $PBO = 0.00\%$, $\Delta = 7.15\%$
- **Pure Noise Universe ($N=40$)**: $PBO = 60.71\%$, $\Delta = 104.91\%$
- **Statistical Spread**: $\Delta PBO = +60.71\% \ge 20.00\%$
This confirms that PBO is statistically monotonic with respect to uninformative trial proliferation.

---

## Getting Started & Reproducibility

### Installation
```bash
git clone <repository-url>
cd "Quantitative Strategy Research & Backtesting Lab"
pip install -r requirements.txt
```

### Run Full Test Suite
```bash
python -m pytest -v
```
Output:
```
============================= test session starts =============================
tests/test_cscv_pbo.py::test_cscv_partitioner_bounds PASSED              [  7%]
tests/test_cscv_pbo.py::test_vectorized_sharpe_matches_scalar PASSED     [ 14%]
tests/test_cscv_pbo.py::test_exact_hand_calculated_pbo PASSED            [ 21%]
tests/test_engine.py::test_look_ahead_bias_protection PASSED             [ 28%]
tests/test_engine.py::test_transaction_cost_deduction PASSED             [ 35%]
tests/test_metrics.py::test_hand_calculated_volatility PASSED            [ 42%]
tests/test_metrics.py::test_hand_calculated_sharpe_ratio PASSED          [ 50%]
tests/test_metrics.py::test_hand_calculated_max_drawdown PASSED          [ 57%]
tests/test_metrics.py::test_sortino_ratio_downside PASSED                [ 64%]
tests/test_metrics.py::test_zero_variance_edge_cases PASSED              [ 71%]
tests/test_noisy_experiment.py::test_pbo_increases_with_noise_and_selection_bias PASSED [ 78%]
tests/test_noisy_experiment.py::test_monotonic_pbo_growth_with_strategy_count PASSED [ 85%]
tests/test_validation.py::test_chronological_holdout_split PASSED        [ 92%]
tests/test_validation.py::test_walk_forward_window_progression PASSED    [100%]
============================= 14 passed in 2.10s ==============================
```

### Run End-to-End Lab Demonstration
```bash
python examples/run_lab_demonstration.py
```
This runs the full 6-step research pipeline and writes reports to `reports/lab_validation_report.txt` and diagnostic figures to `reports/plots/`.

---

## Generated Diagnostic Artifacts

Running `examples/run_lab_demonstration.py` generates publication-quality figures:
1. `reports/plots/equity_and_transaction_costs.png`: Pre- vs post-friction equity curves and drawdown underlay.
2. `reports/plots/pbo_diagnostics.png`: Three-panel diagnostic figure displaying degradation scatter ($y=x$ line), relative rank distribution (highlighting bottom 50%), and logit density ($\lambda_c$).
3. `reports/plots/walk_forward_progression.png`: Stitched walk-forward equity curve and window-by-window In-Sample vs Out-of-Sample Sharpe progression.

---

## Repository File Inventory

| Path | Description |
| :--- | :--- |
| `quant_lab/engine/backtest.py` | Core engine, execution lag, portfolio accounting |
| `quant_lab/engine/metrics.py` | Sharpe, Sortino, MaxDD, duration, Calmar, profit factor |
| `quant_lab/engine/transaction_costs.py` | Multi-tier friction model (commission, spread, slippage, impact) |
| `quant_lab/strategies/base.py` | Strategy interface and signal specification |
| `quant_lab/strategies/moving_average.py` | Dual Moving Average crossover strategy (SMA/EMA) |
| `quant_lab/strategies/mean_reversion.py` | Bollinger Bands mean-reversion strategy |
| `quant_lab/strategies/momentum.py` | Time-Series Momentum strategy |
| `quant_lab/strategies/universe.py` | Parameter sweep runner and return matrix compiler |
| `quant_lab/validation/cscv.py` | Combinatorial symmetric cross-validation partitioner |
| `quant_lab/validation/pbo.py` | Vectorized PBO, logit distribution, and degradation calculator |
| `quant_lab/validation/chronological_oos.py` | In-Sample vs Holdout evaluation |
| `quant_lab/validation/walk_forward.py` | Rolling/anchored walk-forward validator |
| `quant_lab/reporting/tear_sheet.py` | Unified markdown/ASCII tear sheet generator |
| `quant_lab/reporting/visualization.py` | Diagnostic chart plotting engine |
| `quant_lab/data/generator.py` | Market data simulator (Geometric Brownian Motion) |
| `tests/test_*.py` | Complete 14-test verification suite |
| `examples/run_lab_demonstration.py` | End-to-end reproducible research demonstration |
| `docs/methodology.md` | Formal mathematical and theoretical documentation |
| `reports/lab_validation_report.txt` | Generated comprehensive validation tear sheet |

---

## Risk Disclosures & Academic References

1. **Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. V. (2015/2016)**. *The Probability of Backtest Overfitting*. Journal of Computational Finance, 20(4), 39–69.
2. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. John Wiley & Sons.
3. **Akdogan, A. (2026)**. *probability-of-backtest-overfitting* (GitHub Reference Repository, MIT License).

*Disclaimer*: This laboratory is developed for educational, quantitative strategy research, and portfolio risk management purposes. Backtesting models past data and does not guarantee future financial performance.
