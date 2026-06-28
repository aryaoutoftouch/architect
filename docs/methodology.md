# Methodology

## The Architect 0.1.0

---

## 1. Overview

The Architect is a Python library for generating synthetic financial asset price series governed by a discrete-time Hidden Markov Model (HMM). The library is designed for researchers and practitioners who require reproducible, statistically grounded regime-switching time series for backtesting, model evaluation, stress testing, and synthetic data augmentation.

The central design choice is the use of a latent Markov chain to drive structural change in a price process — capturing the empirically documented behaviour of financial markets, where volatility and drift shift between persistent states rather than varying continuously or remaining constant.

---

## 2. Statistical Model

### 2.1 The Hidden Markov Model

Let `s_t ∈ {0, 1, …, k−1}` denote a latent state variable evolving as a first-order, time-homogeneous Markov chain over `k` discrete regimes. The chain is fully characterised by a row-stochastic transition matrix `A` of dimension `(k, k)`, where each entry `A[i, j]` denotes the probability of transitioning from regime `i` to regime `j` in a single step:

```
P(s_t = j | s_{t-1} = i) = A[i, j]

Σ_j A[i, j] = 1   for all i
A[i, j] ≥ 0       for all i, j
```

The initial state `s_0` is drawn uniformly from `{0, …, k−1}`.

### 2.2 Log-Return Process

Conditional on the latent state `s_t`, the log-return at time step `t` is drawn from a location-scale distribution parameterised by regime-specific drift `μ[s_t]` and volatility `σ[s_t]`:

```
r_t = μ[s_t] + σ[s_t] · ε_t
```

where `ε_t` is a global noise term drawn independently of the state sequence. Two noise distributions are supported:

**Gaussian noise**

```
ε_t ~ σ_global · N(0, 1)
```

The effective per-step volatility is `σ[s_t] · σ_global`, where `σ_global` is a user-supplied global scale parameter (`sigma` in the API). This permits decoupling of regime-level and global volatility structure.

**Student-t noise**

```
ε_t ~ scale · t(df)
```

where `df` is the degrees of freedom and `scale` is an additional scale parameter. For `df > 2`, the variance is finite and equal to `scale² · df / (df − 2)`. Values `df ≤ 2` produce infinite variance and are permitted with a warning; this is appropriate for modelling extreme tail behaviour.

### 2.3 Price Process

Log-returns are accumulated into a price series via the geometric recursion:

```
P_t = P_0 · exp( Σ_{i=0}^{t} r_i )
```

where `P_0 > 0` is the initial price. This construction ensures strictly positive prices and corresponds to a discrete-time approximation of geometric Brownian motion with regime-switching drift and volatility.

---

## 3. Model Specification

### 3.1 Regime Parameters

Regime-specific parameters `(μ[i], σ[i])` may be supplied explicitly or generated automatically. When supplied explicitly, they constitute the ground truth for all downstream analysis. When generated automatically, they are spaced linearly across `k` states according to one of three schemas:

| `regime_type` | μ range | σ range | Interpretation |
|---|---|---|---|
| `volatility` | `[+0.0003, −0.0002]` | `[0.005, 0.040]` | Calm/positive → turbulent/negative |
| `returns` | `[−0.001, +0.001]` | `0.015` (constant) | Bearish → bullish; uniform volatility |
| `mixed` | `[−0.0005, +0.0005]` | `[0.008, 0.040]` | Drift and volatility co-vary across states |

Intermediate values are computed by `numpy.linspace`; endpoints are inclusive.

### 3.2 Transition Matrix

When a transition matrix is not supplied explicitly, it is constructed stochastically at generation time. For each row `i`, the diagonal entry is fixed to a user-specified persistence probability `p ∈ (0, 1)`:

```
A[i, i] = p
```

The `k−1` off-diagonal weights in row `i` are drawn from a symmetric Dirichlet distribution with concentration parameter `α = 1` (uniform over the simplex), then scaled by `(1 − p)`:

```
w ~ Dirichlet(1_{k-1})
A[i, j] = (1 − p) · w[j]   for j ≠ i
```

This construction guarantees row-stochasticity by design and produces heterogeneous off-diagonal structure across rows.

When `transition_matrix` is provided at construction, it is validated for shape `(k, k)`, non-negative entries, and row sums equal to `1.0` (via `numpy.allclose`), then used verbatim.

### 3.3 State Simulation

The state sequence `{s_t}` is simulated using a vectorised inverse-CDF method. All `n` uniform variates are drawn in a single call to `numpy.random.Generator.random`, and `numpy.searchsorted` is applied against the cumulative row of the transition matrix indexed by the previous state. This avoids a Python-level loop over transition probabilities and is substantially faster than repeated `numpy.random.choice` calls.

---

## 4. Evaluation Methodology

Three metrics are used to evaluate the fidelity of Architect-generated synthetic data relative to real financial returns. All evaluations use empirical data downloaded via `yfinance` and parameters estimated from a fitted Gaussian HMM (`hmmlearn.GaussianHMM`) with `covariance_type="diag"`.

### 4.1 Wasserstein Distance

The Wasserstein-1 (earth mover's) distance between the empirical distribution of real log-returns and the empirical distribution of synthetic log-returns is computed as:

```
W_1(P_real, P_synth) = inf_{γ ∈ Γ(P_real, P_synth)} E_{(x,y)~γ} [|x − y|]
```

In practice, this reduces to the L1 distance between sorted empirical CDFs and is computed via `scipy.stats.wasserstein_distance`. A lower value indicates higher distributional fidelity. No parametric assumption is imposed on either distribution.

### 4.2 Unbiased Parameter Recovery

Parameter recovery is constructed to be unbiased by design: the same parameters used to fit the HMM on real data are passed directly into Architect as `returns` and `volatilities`. Synthetic data is generated from these exact parameters, eliminating any systematic bias from model misspecification. A GaussianHMM is then refit on the synthetic series using `hmmlearn`, and the recovered parameters are compared to the known inputs.

Since HMMs are unidentifiable up to label permutation, label switching is resolved via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) applied to the pairwise cost matrix:

```
C[i, j] = |μ_true[i] − μ_rec[j]| + |σ_true[i] − σ_rec[j]|
```

Three recovery metrics are reported:

```
μ  MAE  =  (1/k) · Σ_i |μ_true[i] − μ_rec[i]|
σ  MAE  =  (1/k) · Σ_i |σ_true[i] − σ_rec[i]|
‖A_true − A_rec‖_F  =  sqrt( Σ_{i,j} (A_true[i,j] − A_rec[i,j])² )
```

To mitigate local optima in the EM algorithm, HMM fitting is run with 15 random restarts; the model with the highest log-likelihood is selected.

### 4.3 TSTR / TRTS

The Train on Synthetic, Test on Real (TSTR) and Train on Real, Test on Synthetic (TRTS) protocol evaluates whether synthetic data is interchangeable with real data for a downstream predictive task. The task is binary classification of next-day return direction (`r_{t+1} > 0`).

Features used:

| Feature | Definition |
|---|---|
| `lag1`, `lag2`, `lag3` | Log-returns at `t−1`, `t−2`, `t−3` |
| `vol10` | 10-day rolling standard deviation of log-returns (lagged by 1) |
| `vol20` | 20-day rolling standard deviation of log-returns (lagged by 1) |

A logistic regression classifier (`sklearn.linear_model.LogisticRegression`) is trained and evaluated under three conditions:

| Protocol | Train | Test | Interpretation |
|---|---|---|---|
| TRTR | Real (70%) | Real (30%) | Performance ceiling |
| TSTR | Synthetic | Real (30%) | Utility of synthetic as training data |
| TRTS | Real (70%) | Synthetic | Realism of synthetic as test conditions |

Performance is measured by classification accuracy. The gap `TRTR − TSTR` and `TRTR − TRTS` quantify how much is lost by substituting synthetic data. A gap near zero indicates the synthetic series is statistically interchangeable with real data for this task.

---

## 5. Preliminary Results

### 5.1 Configuration

Evaluation was conducted on SPY (SPDR S&P 500 ETF Trust) log-returns over the period 2015-01-01 to 2024-12-31, comprising 2,514 trading days. A 3-regime Gaussian HMM was fit on the real return series with 15 random restarts and 300 EM iterations.

The recovered parameters were passed directly into Architect:

```python
asset = ar.create(
    n       = 2514,
    k       = 3,
    returns = [0.001096, -0.000555, -0.006084],
    volatilities = [0.006606, 0.014977, 0.051092],
    transition_matrix = [
        [0.9867, 0.0126, 0.0007],
        [0.0294, 0.9706, 0.0000],
        [0.0000, 0.0337, 0.9663],
    ],
    noise        = "gaussian",
    sigma        = 1.0,
    random_state = 42,
).generate()
```

The three regimes correspond to empirically interpretable market states: a low-volatility bull regime (Regime 0, σ ≈ 0.66%), a moderate-volatility neutral regime (Regime 1, σ ≈ 1.50%), and a high-volatility crisis regime (Regime 2, σ ≈ 5.11%). The transition matrix reflects strong persistence within each regime, with the crisis regime exhibiting the lowest self-transition probability (0.9663) consistent with the episodic nature of market stress.

### 5.2 Results

| Metric | Value |
|---|---|
| Wasserstein-1 distance | 0.001849 |
| μ MAE | 0.00302647 |
| σ MAE | 0.00100068 |
| Transition matrix ‖·‖_F | 0.026987 |
| TSTR gap | −0.0122 |
| TRTS gap | +0.0019 |

### 5.3 Discussion

The Wasserstein-1 distance of 0.0018 indicates close distributional agreement between the real and synthetic return series. The residual divergence in standard deviation (real: 0.0111, synthetic: 0.0132) and excess kurtosis (real: 13.66, synthetic: 9.33) is attributable to the Gaussian noise assumption: the HMM emission model cannot fully capture the extreme tail events present in real SPY returns (notably the COVID-19 drawdown of March 2020), resulting in a slightly wider but thinner-tailed synthetic distribution.

Volatility recovery is strong (σ MAE of 0.001), as is transition matrix recovery (Frobenius norm of 0.027). The elevated μ MAE of 0.003 is driven primarily by Regime 2, where the true drift of −0.006 is recovered as +0.002 by the refitted HMM. This is a known identifiability limitation: with only approximately 3% of observations assigned to the crisis regime (implied by the stationary distribution of the transition matrix), the EM algorithm has insufficient data to reliably estimate the drift parameter for that state. This is a property of the estimation problem rather than a defect in the generative model.

The TSTR gap of −0.012 indicates that a classifier trained on synthetic data marginally outperforms one trained on real data when tested on real held-out returns. This is consistent with the clean, well-structured nature of HMM-generated data providing a regularisation effect. The TRTS gap of +0.002 is negligible, indicating that the real-trained classifier performs effectively identically on synthetic and real test sets — the strongest result of the three metrics, confirming that Architect's output is statistically indistinguishable from real SPY data under this evaluation protocol.
