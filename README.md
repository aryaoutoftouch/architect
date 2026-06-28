# The Architect 0.1.0

The Architect is a synthetic financial time series generator built on a discrete-time Hidden Markov Model. It gives researchers direct control over regime parameters, transition dynamics, and noise distributions, with exact ground truths available for all generated data. Financial returns exhibit persistent structural shifts, alternating between low-volatility positive-drift periods and high-volatility drawdown regimes. The HMM provides a tractable generative framework for this behaviour, producing realistic price series whose underlying structure is fully specified and recoverable. A latent Markov chain with k states governs the return process at each time step:

```
r_t  = mu[s_t] + sigma[s_t] * eps_t
P_t  = P_0 * exp( sum(r_0, ..., r_t) )
```

The state sequence `{s_t}` evolves under a row-stochastic transition matrix A. The noise term `eps_t` is drawn i.i.d. from a Gaussian or Student-t distribution. Per-regime drift and volatility, the transition matrix, noise distribution, and initial price are all user-specified. Generation is fully reproducible via a random seed.

---

## Dependencies

- Python >= 3.10
- NumPy
- pandas

For export to Parquet: `pyarrow`.

---

## Installation

```bash
git clone https://github.com/aryaoutoftouch/architect.git
cd architect
pip install -r requirements.txt
```

---

## Usage

**Auto-generated regime spacing**

```python
import architect as ar

asset = ar.create(
    n            = 2000,
    k            = 3,
    regime_type  = "volatility",
    persistence  = 0.95,
    random_state = 0,
).generate()
```

`regime_type` controls how mu and sigma are spaced linearly across k states. Options: `"volatility"`, `"returns"`, `"mixed"`. See [`docs/methodology.md`](docs/methodology.md) for the exact parameter ranges.

**Explicit regime parameters**

```python
asset = ar.create(
    n            = 2000,
    k            = 3,
    returns      = [0.001096, -0.000555, -0.006084],
    volatilities = [0.006606,  0.014977,  0.051092],
    random_state = 42,
).generate()
```

**Explicit transition matrix**

```python
asset = ar.create(
    n                 = 2000,
    k                 = 3,
    returns           = [0.001096, -0.000555, -0.006084],
    volatilities      = [0.006606,  0.014977,  0.051092],
    transition_matrix = [
        [0.9867, 0.0126, 0.0007],
        [0.0294, 0.9706, 0.0000],
        [0.0000, 0.0337, 0.9663],
    ],
    random_state = 42,
).generate()
```

**Student-t noise**

```python
asset = ar.create(
    n            = 2000,
    k            = 3,
    noise        = "student_t",
    df           = 5,
    scale        = 0.3,
    random_state = 0,
).generate()
```

For `df <= 2`, variance is infinite. A `UserWarning` is raised and generation proceeds.

**Export**

```python
asset.to_csv("series.csv")
asset.to_parquet("series.parquet")
```

---

## Output

After `.generate()`, the following attributes are populated:

| Attribute | Type | Description |
|---|---|---|
| `series` | `pd.DataFrame` | Columns: `timestamp`, `price`, `returns`, `regime` |
| `true_regimes` | `np.ndarray` | Latent state at each time step |
| `generated_regimes` | `list[dict]` | Per-regime mu and sigma used in generation |
| `transition_matrix` | `np.ndarray` | Row-stochastic (k, k) matrix used in generation |
| `summary` | `dict` | Return statistics and per-regime breakdowns |

---

## Evaluation

Evaluated against SPY log-returns, 2015-01-01 to 2024-12-31 (n = 2,514). Regime parameters and transition matrix were estimated from real data with 15 random restarts, then passed directly into Architect. Three metrics were computed: Wasserstein-1 distance between real and synthetic return distributions, unbiased parameter recovery with Hungarian-aligned HMM refit, and TSTR/TRTS classification utility on next-day return direction.

| Metric | Value |
|---|---|
| Wasserstein-1 | 0.001849 |
| mu MAE | 0.00302647 |
| sigma MAE | 0.00100068 |
| Transition matrix Frobenius norm | 0.026987 |
| TSTR gap | -0.0122 |
| TRTS gap | +0.0019 |

Full methodology, parameter specification, and result discussion in [`docs/methodology.md`](docs/methodology.md).

---

## Documentation

| Document | Description |
|---|---|
| [`docs/documentation.md`](docs/documentation.md) | Technical Documentation |
| [`docs/installation.md`](docs/installation.md) | Installation guide |
| [`docs/methodology.md`](docs/methodology.md) | Statistical model, evaluation methodology, and results |

---

## License

MIT
