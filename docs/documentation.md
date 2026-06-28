# The Architect 0.1.0 Technical Documentation

## `ar.create`

```
ar.create(
    n,
    k=3,
    regime_type="volatility",
    returns=None,
    volatilities=None,
    persistence=0.90,
    transition_matrix=None,
    noise="gaussian",
    sigma=1.0,
    df=5,
    scale=1.0,
    p0=100.0,
    random_state=None,
)
```

Factory function for constructing an `Asset` instance. Preferred over calling
`Asset(...)` directly. All parameters are forwarded verbatim to `Asset.__init__`.

**Returns** → `Asset`

---

## `ar.Asset`

```
ar.Asset(
    n,
    k,
    regime_type,
    returns,
    volatilities,
    persistence,
    transition_matrix,
    noise,
    sigma,
    df,
    scale,
    p0,
    random_state,
)
```

Synthetic financial asset driven by a discrete-time Hidden Markov Model (HMM).

At each time step `t`, a latent Markov chain selects an active regime
`s_t ∈ {0, …, k−1}`. Log-returns are drawn from that regime's distribution and
accumulated into a price series via the geometric recursion:

```
r_t  = μ[s_t] + σ[s_t] · ε_t
P_t  = P_0 · exp( Σ_{i=0}^{t} r_i )
```

where `ε_t` is drawn from the configured global noise distribution. The result is
a price path with time-varying drift and volatility whose structural breakpoints
are governed by the transition matrix.

Use `ar.create(...)` rather than instantiating this class directly.

---

### Parameters

**n** : `int`  
Number of time steps to generate. Must be a positive integer.

**k** : `int`, default=`3`  
Number of hidden regimes. Must be a positive integer.

**regime_type** : `{"volatility", "returns", "mixed"}`, default=`"volatility"`  
Controls how auto-generated regime parameters are spaced when `returns` and
`volatilities` are not provided. See [Auto-Regime Defaults](#auto-regime-defaults)
for the exact values used. Ignored when `returns` and `volatilities` are supplied.

**returns** : `list of float` or `None`, default=`None`  
Per-regime drift `μ`. Must have length `k`. Must be provided together with
`volatilities`; supplying one without the other raises `ValueError`.

**volatilities** : `list of float` or `None`, default=`None`  
Per-regime volatility `σ`. Must have length `k`. All values must be strictly
positive. Must be provided together with `returns`.

**persistence** : `float`, default=`0.90`  
Self-transition probability `P(s_t = i | s_{t-1} = i)`. Must satisfy
`0 < persistence < 1`. Ignored when `transition_matrix` is provided.

**transition_matrix** : `list of list of float` or `None`, default=`None`  
Explicit row-stochastic `(k, k)` transition matrix. When supplied, overrides
`persistence` entirely. Validated to have shape `(k, k)`, non-negative entries,
and rows that each sum to `1.0` (checked with `numpy.allclose`).

**noise** : `{"gaussian", "student_t"}`, default=`"gaussian"`  
Global noise distribution applied to every time step. Gaussian noise uses `sigma`;
Student-t noise uses `df` and `scale`. See [Return Computation](#return-computation).

**sigma** : `float`, default=`1.0`  
Scale of Gaussian noise. Active only when `noise="gaussian"`. Must be `> 0`.

**df** : `float`, default=`5`  
Degrees of freedom for Student-t noise. Active only when `noise="student_t"`.
Must be `> 0`. Values `≤ 2` produce infinite variance; a `UserWarning` is raised
but generation proceeds.

**scale** : `float`, default=`1.0`  
Scale parameter for Student-t noise. Active only when `noise="student_t"`.
Must be `> 0`.

**p0** : `float`, default=`100.0`  
Initial asset price `P_0`. Must be `> 0`.

**random_state** : `int` or `None`, default=`None`  
Seed passed to `numpy.random.default_rng`. Pass an integer for fully reproducible
output. `None` produces a randomly seeded RNG. Must be an `int` or `None`;
any other type raises `TypeError`.

---

### Attributes

The following attributes are set on construction from the supplied parameters:

| Attribute | Type | Description |
|---|---|---|
| `n` | `int` | Number of time steps. |
| `k` | `int` | Number of hidden regimes. |
| `regime_type` | `str` | Auto-regime flavor. |
| `persistence` | `float` | Self-transition probability. |
| `noise` | `str` | Noise distribution identifier. |
| `sigma` | `float` | Gaussian noise scale. |
| `df` | `float` | Student-t degrees of freedom. |
| `scale` | `float` | Student-t scale. |
| `p0` | `float` | Starting price. |
| `random_state` | `int` or `None` | RNG seed. |

The following attributes are `None` until `.generate()` is called:

**series** : `pandas.DataFrame` or `None`  
Generated time series. Columns:

| Column | dtype | Description |
|---|---|---|
| `timestamp` | `int` | Integer time index `0, 1, …, n−1`. |
| `price` | `float` | Asset price `P_t`. |
| `returns` | `float` | Log-return `r_t` at each step. |
| `regime` | `int` | Active regime label `s_t ∈ {0, …, k−1}`. |

**true_regimes** : `numpy.ndarray` of shape `(n,)` or `None`  
Regime label at each time step. Integer dtype (`numpy.intp`). Equivalent to
`asset.series["regime"].values`.

**generated_regimes** : `list of dict` or `None`  
List of `k` dicts, one per regime, each with keys:

| Key | Type | Description |
|---|---|---|
| `"regime"` | `int` | Regime index `i`. |
| `"mu"` | `float` | Per-regime drift used in generation. |
| `"sigma"` | `float` | Per-regime volatility used in generation. |

**summary** : `dict` or `None`  
Descriptive statistics computed after generation. Structure:

```python
{
    "n":                 int,    # number of time steps
    "mean":              float,  # mean of log-returns
    "std":               float,  # std of log-returns (ddof=1, via pandas)
    "variance":          float,  # variance of log-returns (ddof=1, via pandas)
    "skewness":          float,  # Fisher's bias-adjusted skewness (via pandas)
    "excess_kurtosis":   float,  # Fisher's excess kurtosis (via pandas)
    "min":               float,
    "max":               float,
    "median":            float,
    "q1":                float,  # 25th percentile
    "q3":                float,  # 75th percentile

    "regime_frequencies": {
        i: float,  # fraction of steps in regime i
        ...
    },

    "regime_durations": {
        i: {
            "avg": float,  # mean consecutive-run length for regime i
            "max": int,    # longest consecutive run
            "min": int,    # shortest consecutive run
        },
        ...
    },

    "regime_statistics": {
        i: {
            "mean":  float,  # mean log-return while in regime i
            "std":   float,  # std of log-returns while in regime i (numpy, ddof=0)
            "count": int,    # number of steps spent in regime i
        },
        ...
    },
}
```

> **Note:** `summary["std"]` and `summary["variance"]` use `pandas.Series.std()`
> and `.var()` (ddof=1). `regime_statistics[i]["std"]` uses `numpy.std()` (ddof=0).
> These are **not** computed with the same normalisation.

**metadata** : `dict` or `None`  
Snapshot of input configuration recorded at generation time. Structure:

```python
{
    "n":            int,
    "k":            int,
    "regime_type":  str,
    "noise":        str,
    "noise_params": {"sigma": float}                   # when noise="gaussian"
                 | {"df": float, "scale": float},      # when noise="student_t"
    "persistence":  float,
    "p0":           float,
    "random_state": int | None,
}
```

---

### Properties

**transition_matrix** : `numpy.ndarray` of shape `(k, k)` or `None`  
Read-only. Row-stochastic transition matrix actually used during generation.
`None` until `.generate()` is called. When `transition_matrix` was provided at
construction, this is a float64 copy of that input. When it was not provided, this
is the matrix built stochastically from `persistence` (see
[Transition Matrix Construction](#transition-matrix-construction)).

---

### Methods

#### `generate() → Asset`

Run the simulation and populate all output attributes (`series`, `true_regimes`,
`generated_regimes`, `transition_matrix`, `summary`, `metadata`).

Returns `self`, enabling method chaining:

```python
asset = ar.create(n=500, k=3, random_state=0).generate()
```

**Raises**  
`RuntimeError` — if `generate()` has already been called on this instance.
Call `.reset()` first to reuse the same configuration.

---

#### `reset() → Asset`

Clear all generated state, setting `series`, `true_regimes`,
`generated_regimes`, `transition_matrix`, `summary`, and `metadata` back to
`None`. Input parameters are preserved. Returns `self`.

```python
asset.reset().generate()   # run a second simulation with the same config
```

---

#### `to_csv(path, **kwargs) → None`

Write `asset.series` to a CSV file. Calls `pandas.DataFrame.to_csv(path, index=False, **kwargs)`. Any additional keyword arguments are forwarded to pandas.

**Raises**  
`RuntimeError` — if `.generate()` has not been called.

---

#### `to_parquet(path, **kwargs) → None`

Write `asset.series` to a Parquet file. Calls `pandas.DataFrame.to_parquet(path, index=False, **kwargs)`. Any additional keyword arguments are forwarded to pandas.

**Raises**  
`RuntimeError` — if `.generate()` has not been called.

---

### Special Methods

**`__repr__`**  
Returns a concise string identifying key configuration and generation status:

```
Asset(n=500, k=3, regime_type='volatility', noise='gaussian', status=pending)
Asset(n=500, k=3, regime_type='volatility', noise='gaussian', status=generated)
```

**`__eq__`**  
Two `Asset` instances compare equal when all input parameters match. Transition
matrices (if provided) are compared element-wise with `numpy.array_equal`.
`returns` and `volatilities` lists are compared with Python's standard `==`.
Generated state (`series`, `true_regimes`, etc.) is not considered. Comparing
with a non-`Asset` object returns `NotImplemented`.

---

### Notes

#### Auto-Regime Defaults

When `returns` and `volatilities` are not provided, regime parameters are spaced
linearly across `k` states according to `regime_type`:

| `regime_type` | `μ` range | `σ` range | Interpretation |
|---|---|---|---|
| `"volatility"` | `[+0.0003, −0.0002]` | `[0.005, 0.040]` | Low-vol/positive-drift → high-vol/negative-drift |
| `"returns"` | `[−0.001, +0.001]` | `0.015` (constant) | Bearish → bullish; uniform volatility |
| `"mixed"` | `[−0.0005, +0.0005]` | `[0.008, 0.040]` | Both drift and volatility increase across states |

End-points are inclusive; intermediate values are computed with `numpy.linspace`.
For `k=1` all regimes collapse to the single end-point value.

#### Transition Matrix Construction

When `transition_matrix` is not provided:

- If `k=1`, the matrix is `[[1.0]]`.
- Otherwise, for each row `i`, the diagonal is set to `persistence` and the
  `k−1` off-diagonal weights are drawn from a symmetric Dirichlet distribution
  (`α = ones(k−1)`) then scaled by `(1 − persistence)`. Off-diagonal allocations
  therefore differ across rows and depend on `random_state`. The resulting matrix
  is always row-stochastic by construction.

#### Return Computation

The log-return at step `t` is:

```
r_t = μ[s_t] + σ[s_t] · ε_t
```

where `ε_t` is drawn globally (not per-regime) from:

- **Gaussian**: `ε_t ~ sigma · N(0, 1)` — the `sigma` parameter acts as an
  additional global scale on top of per-regime `σ[s_t]`.
- **Student-t**: `ε_t ~ scale · t(df)` — `scale` similarly multiplies the
  per-regime `σ[s_t]`.

Effective volatility at step `t` is therefore `σ[s_t] · sigma` (Gaussian) or
`σ[s_t] · scale` (Student-t).

#### State Simulation

The initial state `s_0` is drawn uniformly from `{0, …, k−1}`. Subsequent
states use a vectorised inverse-CDF approach: all `n` uniform variates are
drawn in one call, and `numpy.searchsorted` is applied against the cumulative
row of the transition matrix indexed by the previous state.

#### Generation Lifecycle

An `Asset` instance is stateful: `generate()` may only be called once. Calling it
a second time raises `RuntimeError`. The intended pattern for repeated simulations
is `.reset()` followed by `.generate()`, or constructing a new instance via
`ar.create(...)`.

---

### Raises (construction)

| Exception | Condition |
|---|---|
| `ValueError` | `n` or `k` is not a positive integer |
| `ValueError` | Exactly one of `returns` / `volatilities` is provided |
| `ValueError` | `len(returns) != k` or `len(volatilities) != k` |
| `ValueError` | Any element of `volatilities` is `≤ 0` |
| `ValueError` | `transition_matrix` shape is not `(k, k)`, contains negative entries, or rows do not sum to `1.0` |
| `ValueError` | `persistence` not in `(0, 1)` |
| `ValueError` | `noise` not in `{"gaussian", "student_t"}` |
| `ValueError` | `regime_type` not in `{"volatility", "returns", "mixed"}` |
| `ValueError` | `p0`, `sigma`, `scale`, or `df` is `≤ 0` |
| `TypeError` | `random_state` is not an `int` or `None` |
| `UserWarning` | `df ≤ 2` (infinite variance in Student-t; construction continues) |

---

### Examples

**Basic usage — default volatility regimes, Gaussian noise**

```python
import architect as ar

asset = ar.create(n=1000, k=3, random_state=42).generate()

print(asset)
# Asset(n=1000, k=3, regime_type='volatility', noise='gaussian', status=generated)

print(asset.series.head())
#    timestamp       price   returns  regime
# 0          0  100.006234  0.006234       0
# ...

print(asset.summary["mean"], asset.summary["std"])
```

**Custom regime parameters**

```python
asset = ar.create(
    n=500,
    k=2,
    returns=[-0.001, 0.002],
    volatilities=[0.01, 0.025],
    persistence=0.95,
    random_state=0,
).generate()

print(asset.generated_regimes)
# [{"regime": 0, "mu": -0.001, "sigma": 0.01},
#  {"regime": 1, "mu":  0.002, "sigma": 0.025}]
```

**Heavy-tailed noise with Student-t**

```python
asset = ar.create(
    n=2000,
    k=3,
    noise="student_t",
    df=3,       # heavy tails; warns that df <= 2 has infinite variance (df=3 is fine)
    scale=0.5,
    random_state=7,
).generate()
```

**Explicit transition matrix**

```python
tm = [
    [0.97, 0.02, 0.01],
    [0.05, 0.90, 0.05],
    [0.01, 0.04, 0.95],
]
asset = ar.create(n=1000, k=3, transition_matrix=tm, random_state=1).generate()
print(asset.transition_matrix)
```

**Resetting and re-running**

```python
asset = ar.create(n=500, k=2, random_state=99)
asset.generate()
asset.reset()
asset.generate()   # second independent run with the same config
```

**Exporting output**

```python
asset = ar.create(n=5000, k=4, random_state=12).generate()
asset.to_csv("prices.csv")
asset.to_parquet("prices.parquet")
```

**Inspecting per-regime statistics**

```python
asset = ar.create(n=2000, k=3, random_state=5).generate()

for i, stats in asset.summary["regime_statistics"].items():
    freq = asset.summary["regime_frequencies"][i]
    dur  = asset.summary["regime_durations"][i]
    print(
        f"Regime {i}: freq={freq:.2%}, "
        f"mean_ret={stats['mean']:.5f}, std={stats['std']:.5f}, "
        f"avg_run={dur['avg']:.1f} steps"
    )
```

---

### See Also

- `numpy.random.default_rng` — RNG used internally; pass `random_state` for seeding.
- `numpy.random.Generator.dirichlet` — used to randomise off-diagonal transition weights.
- `pandas.DataFrame.to_csv`, `pandas.DataFrame.to_parquet` — kwargs forwarded by `to_csv` / `to_parquet`.
