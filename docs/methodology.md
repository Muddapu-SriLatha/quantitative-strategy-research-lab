# Quantitative Strategy Research & Backtesting Lab: Methodology & Validation Guide

## 1. Executive Summary & Foundational Principles
The Quantitative Strategy Research & Backtesting Lab is an institutional-grade research platform built in Python (NumPy, pandas, SciPy, matplotlib) designed to evaluate trading strategies, simulate realistic execution frictions, and quantify selection bias (data snooping / backtest overfitting).

This guide documents the theoretical foundations, implementation architecture, mathematical definitions, and validation protocols implemented in the laboratory.

---

## 2. Core Backtesting Architecture & Look-Ahead Protection

### 2.1 The Mechanism of Look-Ahead Bias
In empirical finance, look-ahead bias occurs whenever an algorithm uses information at bar $t$ that was not available at the execution timestamp. A common flaw in junior backtesters is executing trades at bar $t$'s closing price $P_t$ using a signal $S_t$ derived from the close $P_t$. Because $P_t$ is required to determine $S_t$, an order cannot realistically fill at $P_t$ unless the trader possesses a time machine.

### 2.2 Execution Lag Protocol
Our `BacktestEngine` enforces a strict 1-bar execution lag:
$$\text{Position}_t = \text{Signal}_{t-1}$$
$$\text{Return}_{\text{gross}, t} = \text{Position}_t \cdot \left(\frac{P_t}{P_{t-1}} - 1\right)$$

This mathematical formulation guarantees:
1. Signals computed at the close of bar $t-1$ are executed at the open/effective price of bar $t$.
2. Price shocks occurring at bar $t$ cannot be retroactively captured by signals that trigger upon observing that shock.

---

## 3. Realistic Transaction Cost & Market Friction Modelling

Paper backtests that omit friction invariably produce spurious alpha. The laboratory implements a multi-tier transaction cost model applied to portfolio turnover:

$$\text{Turnover}_t = |\text{Position}_t - \text{Position}_{t-1}|$$

### 3.1 Linear and Non-Linear Cost Components
1. **Brokerage Commission ($c_{\text{comm}}$)**: Proportional broker fee per dollar volume traded (default: 5.0 bps = 0.05%).
2. **Bid-Ask Spread ($c_{\text{spread}}$)**: Half-spread crossed upon entering and exiting positions (default: 2.0 bps = 0.02%).
3. **Execution Slippage ($c_{\text{slip}}$)**: Linear slippage due to latency and order matching queue position (default: 3.0 bps = 0.03%).
4. **Ticket Fee ($F_{\text{fixed}}$)**: Flat dollar fee charged per execution event ($1.00 per trade).
5. **Nonlinear Market Impact ($I_{\text{market}}$)**: Models square-root order book exhaustion for larger sizes:
   $$I_{\text{market}}(\text{Turnover}_t) = \gamma \cdot \left(\text{Turnover}_t\right)^{1.5}$$

### 3.2 Net Return Formula
$$\text{Cost Drag}_t = (c_{\text{comm}} + c_{\text{spread}} + c_{\text{slip}}) \cdot \text{Turnover}_t + I_{\text{market}}(\text{Turnover}_t) + \frac{F_{\text{fixed}}}{\text{Equity}_{t-1}} \cdot \mathbf{1}_{\{\text{Turnover}_t > 0\}}$$
$$\text{Return}_{\text{net}, t} = \text{Return}_{\text{gross}, t} - \text{Cost Drag}_t$$

---

## 4. Comparing the Three Validation Methodologies

A core requirement of quantitative research is understanding what each validation technique measures and what it fails to measure.

| Dimension | Chronological OOS Holdout | Walk-Forward Validation | CSCV / PBO |
| :--- | :--- | :--- | :--- |
| **Data Partitioning** | Fixed chronological split (e.g. 70% In-Sample / 30% Out-of-Sample). | Sequential rolling/expanding train windows followed by forward test slices. | Contiguous block combinatorial partitioning into $S$ groups; all $\binom{S}{S/2}$ splits. |
| **Temporal Structure** | Strictly preserved; one single forward horizon. | Strictly preserved; rolled sequentially over history. | Preserved within blocks; block order is permuted across train/test splits. |
| **Primary Metric** | Point estimate of OOS Sharpe, drawdown, and decay. | Stitched out-of-sample CAGR, rolling Sharpe, re-optimization stability. | Probability of Backtest Overfitting ($PBO$), degradation $\Delta$, logit distribution. |
| **Core Question** | *"How did this parameter set perform on the final holdout period?"* | *"Does adaptive re-optimization succeed across changing market regimes?"* | *"What is the probability the in-sample selected strategy fails out-of-sample due to selection bias?"* |
| **Key Limitation** | Single point estimate; vulnerable to holdout snooping and regime luck. | Sensitive to window length and refit frequency; high turnover drag. | Assumes block exchangeability; does not evaluate single-strategy regime timing. |
| **Profitability Guarantee** | **None.** Future market dynamics can diverge completely from the holdout. | **None.** Market regimes can shift faster than the re-estimation frequency. | **None.** A strategy with $PBO = 0\%$ can still suffer structural losses. |

### 4.1 Detailed Methodological Limitations & Risks

1. **Chronological Holdout**:
   - *Limitation*: Testing multiple times against the "unseen holdout" turns the holdout into an in-sample training set (holdout overfitting).
   - *Risk*: A strategy may fail out-of-sample merely because the holdout happened to coincide with an unusual macro event (e.g., pandemic crash, rate hike cycle), not because the strategy logic is flawed.

2. **Walk-Forward Validation**:
   - *Limitation*: Re-optimizing parameters every quarter or month can induce whipsaws and severe transaction cost friction.
   - *Risk*: If parameter distributions are noisy, the walk-forward optimizer constantly switches between strategies right after they experience short-term peaks (buying the top of parameter performance).

3. **Combinatorial Symmetric Cross-Validation (CSCV) and PBO**:
   - *Limitation*: CSCV tests selection bias among a defined *universe of trials*. It does not evaluate external factors outside the tested universe.
   - *Risk*: A low PBO simply proves that the in-sample winner was genuinely better than the other candidates in the sample. It does **not** guarantee that the entire candidate universe itself has positive economic edge!

---

## 5. Mathematical Formulation of CSCV and PBO (Bailey et al. 2015/2016)

Let $\mathbf{M} \in \mathbb{R}^{T \times N}$ be a matrix of returns for $N$ model/strategy configurations across $T$ observations.

### 5.1 Partitioning
Divide $\mathbf{M}$ row-wise into an even number $S \ge 4$ of contiguous submatrices $M_1, M_2, \dots, M_S$ of length $\approx T/S$.
Contiguous blocking preserves autocorrelation, volatility clustering, and microstructure within each time block.

### 5.2 Combinatorial Splits
Form all $C_S = \binom{S}{S/2}$ unique combinations choosing $S/2$ blocks as the In-Sample (training) set $J_c$, and the remaining $S/2$ blocks as the Out-of-Sample (testing) set $\bar{J}_c$.

For each split $c = 1, \dots, C_S$:
1. **In-Sample Evaluation**: Compute performance metric (annualized Sharpe ratio) across all columns on $J_c$:
   $$R_c = (R_{c, 1}, R_{c, 2}, \dots, R_{c, N})$$
2. **In-Sample Selection**: Select the best strategy:
   $$n^*(c) = \arg\max_{n \in \{1, \dots, N\}} R_{c, n}$$
3. **Out-of-Sample Evaluation**: Compute performance metric across all columns on $\bar{J}_c$:
   $$\bar{R}_c = (\bar{R}_{c, 1}, \bar{R}_{c, 2}, \dots, \bar{R}_{c, N})$$
4. **Out-of-Sample Ranking**: Determine the rank $\bar{r}_{c, n} \in \{1, \dots, N\}$ of each strategy in $\bar{R}_c$ using descending order (Rank 1 = highest OOS Sharpe, Rank $N$ = lowest OOS Sharpe).
5. **Relative Rank**: Compute the normalized relative rank of the selected strategy $n^*(c)$:
   $$\bar{\omega}_c = \frac{\bar{r}_{c, n^*(c)}}{N + 1} \in (0, 1)$$
6. **Logit Transformation**:
   $$\lambda_c = \ln\left(\frac{\bar{\omega}_c}{1 - \bar{\omega}_c}\right)$$

### 5.3 Diagnostic Outputs
- **Probability of Backtest Overfitting ($PBO$)**:
  The empirical probability that the best in-sample strategy performs in the bottom half out-of-sample:
  $$PBO = \frac{1}{C_S} \sum_{c=1}^{C_S} \mathbf{1}_{\{\bar{\omega}_c > 0.5\}} = \frac{1}{C_S} \sum_{c=1}^{C_S} \mathbf{1}_{\{\lambda_c > 0\}}$$
- **Probability of Out-of-Sample Loss ($P(\text{Loss})$)**:
  $$P(\text{Loss}) = \frac{1}{C_S} \sum_{c=1}^{C_S} \mathbf{1}_{\{\bar{R}_{c, n^*(c)} \le 0\}}$$
- **Expected Performance Degradation ($\Delta$)**:
  $$\Delta = \frac{\frac{1}{C_S}\sum_{c=1}^{C_S} R_{c, n^*(c)} - \frac{1}{C_S}\sum_{c=1}^{C_S} \bar{R}_{c, n^*(c)}}{\left|\frac{1}{C_S}\sum_{c=1}^{C_S} R_{c, n^*(c)}\right|}$$

---

## 6. Audit of the Reference Repository

### 6.1 Reference Repository Details
- **Repository**: `AidanAkdogan/probability-of-backtest-overfitting`
- **License**: MIT License (Copyright 2026 Aidan Akdogan).
- **Attribution & Reuse Decision**: The MIT license permits use, modification, and integration with attribution. However, based on our engineering audit, we did not copy or vendor the reference repository because:
  1. The reference repository contains **no backtesting engine**, no portfolio accounting, no execution simulation, and no transaction cost model.
  2. The reference repository contains **zero unit tests** (`test_*.py` absent).
  3. The Sortino ratio calculation in the reference repository contained a mathematical error (dividing sum of squared downside returns by the count of negative days instead of total observations $T$).
  4. The PBO computation in the reference repository used an $O(C_S \times N)$ nested loop in Python rather than vectorized matrix operations, creating significant computational bottlenecks.
  5. The synthetic data generator in the reference repository did not simulate actual price action or strategies, but simply generated AR(1) Gaussian noise.
- **Architectural Implementation**: We utilized the published methodology of Bailey et al. (2015/2016) and built an independent, fully vectorized, high-performance backtesting and PBO engine from first principles with a complete 14-test verification suite.

---

## 7. Verification Results Summary

### 7.1 Deterministic Hand-Calculated Case ($S=4, T=8, N=3$)
- Hand-derived theoretical PBO: $2 / 6 = 33.3333\%$.
- Computed PBO: $33.3333\%$.
- Exact rank match across all 6 splits:
  - Split 0: Best IS is Strat 0, OOS Rank 3 (Overfit).
  - Split 1: Best IS is Strat 1, OOS Rank 1 (Robust).
  - Split 2: Best IS is Strat 1, OOS Rank 1 (Robust).
  - Split 3: Best IS is Strat 1, OOS Rank 1 (Robust).
  - Split 4: Best IS is Strat 1, OOS Rank 1 (Robust).
  - Split 5: Best IS is Strat 2, OOS Rank 3 (Overfit).

### 7.2 Controlled Synthetic Noise Experiment
- **Persistent Signal Universe ($N=3$)**: $PBO = 0.00\%$, $\Delta = 7.15\%$.
- **Pure Noise Universe ($N=40$)**: $PBO = 60.71\%$, $\Delta = 104.91\%$.
- **Spread**: $PBO_{\text{noise}} - PBO_{\text{signal}} = 60.71\% > 20.00\%$, verifying selection bias growth and monotonicity.
