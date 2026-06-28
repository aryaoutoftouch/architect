from __future__ import annotations

import warnings
from itertools import groupby
from typing import Optional

import numpy as np
import pandas as pd


_REGIME_TYPES: tuple[str, ...] = ("volatility", "returns", "mixed")
_NOISE_TYPES:  tuple[str, ...] = ("gaussian", "student_t")


def create(
    n: int,
    k: int = 3,
    regime_type: str = "volatility",
    returns: Optional[list[float]] = None,
    volatilities: Optional[list[float]] = None,
    persistence: float = 0.90,
    transition_matrix: Optional[list[list[float]]] = None,
    noise: str = "gaussian",
    sigma: float = 1.0,
    df: float = 5,
    scale: float = 1.0,
    p0: float = 100.0,
    random_state: Optional[int] = None,
) -> Asset:
    """Create an Asset instance without calling the class constructor directly.

    Parameters
    ----------
    n : int
        Number of time steps to generate.
    k : int, default=3
        Number of hidden regimes.
    regime_type : {"volatility", "returns", "mixed"}, default="volatility"
        How auto-generated regimes differ across states. Ignored when
        `returns` and `volatilities` are provided.
    returns : list of float, optional
        Per-regime drift (mu). Must have length k. Required together with
        `volatilities`; if omitted, values are set automatically.
    volatilities : list of float, optional
        Per-regime volatility (sigma). Must have length k and all values > 0.
    persistence : float, default=0.90
        Probability of staying in the current regime at each step.
        Must be in (0, 1).
    transition_matrix : list of list of float, optional
        Row-stochastic (k, k) matrix. Overrides `persistence` when provided.
    noise : {"gaussian", "student_t"}, default="gaussian"
        Noise distribution applied to returns.
    sigma : float, default=1.0
        Gaussian noise scale. Used only when noise="gaussian".
    df : float, default=5
        Degrees of freedom for Student-t noise. Used only when
        noise="student_t". Values <= 2 produce infinite variance (warning
        is raised).
    scale : float, default=1.0
        Scale parameter for Student-t noise.
    p0 : float, default=100.0
        Initial asset price.
    random_state : int, optional
        Seed for the random number generator. Pass an int for reproducibility.

    Returns
    -------
    Asset
        Configured Asset ready for .generate().
    """
    return Asset(
        n=n,
        k=k,
        regime_type=regime_type,
        returns=returns,
        volatilities=volatilities,
        persistence=persistence,
        transition_matrix=transition_matrix,
        noise=noise,
        sigma=sigma,
        df=df,
        scale=scale,
        p0=p0,
        random_state=random_state,
    )


class Asset:
    """Synthetic financial asset driven by a Hidden Markov Model.

    The asset price follows a regime-switching process: at each time step a
    latent Markov chain selects the active regime, and log-returns are drawn
    from that regime's Gaussian or Student-t distribution. Prices are
    recovered via the cumulative exponential of log-returns.

    Use `architect.create(...)` as a convenience constructor rather than
    instantiating this class directly.

    Attributes
    ----------
    series : pd.DataFrame or None
        Generated time series with columns [timestamp, price, returns, regime].
        None until .generate() is called.
    true_regimes : np.ndarray or None
        Regime label at each time step. None until .generate() is called.
    generated_regimes : list of dict or None
        Per-regime mu and sigma used during generation.
    summary : dict or None
        Descriptive statistics of returns and per-regime breakdowns.
    metadata : dict or None
        Input parameters recorded at generation time.
    """

    def __init__(
        self,
        n: int,
        k: int,
        regime_type: str,
        returns: Optional[list[float]],
        volatilities: Optional[list[float]],
        persistence: float,
        transition_matrix: Optional[list[list[float]]],
        noise: str,
        sigma: float,
        df: float,
        scale: float,
        p0: float,
        random_state: Optional[int],
    ) -> None:
        """
        Parameters
        ----------
        n : int
            Number of time steps.
        k : int
            Number of hidden regimes.
        regime_type : str
            Auto-regime flavor; one of "volatility", "returns", "mixed".
        returns : list of float or None
            Per-regime mu values; length must equal k.
        volatilities : list of float or None
            Per-regime sigma values; length must equal k, all > 0.
        persistence : float
            Self-transition probability in (0, 1).
        transition_matrix : list of list of float or None
            Row-stochastic (k, k) override matrix.
        noise : str
            "gaussian" or "student_t".
        sigma : float
            Gaussian noise scale (noise="gaussian" only).
        df : float
            Student-t degrees of freedom (noise="student_t" only).
        scale : float
            Student-t scale (noise="student_t" only).
        p0 : float
            Starting price; must be > 0.
        random_state : int or None
            RNG seed for reproducibility.
        """
        self._validate(
            n, k, returns, volatilities, persistence,
            transition_matrix, noise, regime_type, p0, sigma, scale, df, random_state,
        )

        self.n = n
        self.k = k
        self.regime_type = regime_type
        self._returns_input = returns
        self._volatilities_input = volatilities
        self.persistence = persistence
        self._transition_matrix_input = transition_matrix
        self.noise = noise
        self.sigma = sigma
        self.df = df
        self.scale = scale
        self.p0 = p0
        self.random_state = random_state

        # Generated state; None until .generate() is called
        self.series:           Optional[pd.DataFrame]  = None
        self.true_regimes:     Optional[np.ndarray]    = None
        self._transition_matrix: Optional[np.ndarray]  = None
        self.generated_regimes: Optional[list[dict]]   = None
        self.summary:          Optional[dict]           = None
        self.metadata:         Optional[dict]           = None

    # Dunder methods

    def __repr__(self) -> str:
        status = "generated" if self.series is not None else "pending"
        return (
            f"Asset(n={self.n}, k={self.k}, regime_type='{self.regime_type}', "
            f"noise='{self.noise}', status={status})"
        )

    def __eq__(self, other: object) -> bool:
        """Two Assets are equal when all their input parameters match."""
        if not isinstance(other, Asset):
            return NotImplemented
        tm_self  = self._transition_matrix_input
        tm_other = other._transition_matrix_input
        tm_equal = (tm_self is None and tm_other is None) or (
            tm_self is not None
            and tm_other is not None
            and np.array_equal(tm_self, tm_other)
        )
        return (
            self.n             == other.n
            and self.k         == other.k
            and self.regime_type == other.regime_type
            and self._returns_input     == other._returns_input
            and self._volatilities_input == other._volatilities_input
            and self.persistence == other.persistence
            and tm_equal
            and self.noise     == other.noise
            and self.sigma     == other.sigma
            and self.df        == other.df
            and self.scale     == other.scale
            and self.p0        == other.p0
            and self.random_state == other.random_state
        )

    # Properties

    @property
    def transition_matrix(self) -> Optional[np.ndarray]:
        """Row-stochastic (k, k) transition matrix used during generation.

        None until .generate() is called.
        """
        return self._transition_matrix

    # Public methods

    def generate(self) -> Asset:
        """Run the simulation and populate all output attributes.

        Raises
        ------
        RuntimeError
            If called on an Asset that has already been generated. Call
            .reset() first to reuse the same configuration.

        Returns
        -------
        self : Asset
        """
        if self.series is not None:
            raise RuntimeError(
                "This Asset has already been generated. "
                "Call .reset() to clear the previous run before generating again."
            )

        rng = np.random.default_rng(self.random_state)

        regimes = self._build_regimes()
        tm      = self._build_transition_matrix(rng)
        states  = self._simulate_states(tm, rng)
        eps     = self._generate_noise(rng)
        ret     = self._compute_returns(states, regimes, eps)
        prices  = self.p0 * np.exp(np.cumsum(ret))

        self.generated_regimes   = regimes
        self._transition_matrix  = tm
        self.true_regimes        = states
        self.series = pd.DataFrame({
            "timestamp": np.arange(self.n),
            "price":     prices,
            "returns":   ret,
            "regime":    states,
        })
        self.metadata = self._build_metadata()
        self.summary  = self._build_summary(ret, states)

        return self

    def reset(self) -> Asset:
        """Clear all generated state so the Asset can be run again.

        Returns
        -------
        self : Asset
        """
        self.series              = None
        self.true_regimes        = None
        self._transition_matrix  = None
        self.generated_regimes   = None
        self.summary             = None
        self.metadata            = None
        return self

    def to_csv(self, path: str, **kwargs) -> None:
        """Write the generated series to a CSV file.

        Parameters
        ----------
        path : str
            Destination file path.
        **kwargs
            Forwarded to pandas.DataFrame.to_csv.

        Raises
        ------
        RuntimeError
            If .generate() has not been called.
        """
        if self.series is None:
            raise RuntimeError("No data to export — call .generate() first.")
        self.series.to_csv(path, index=False, **kwargs)

    def to_parquet(self, path: str, **kwargs) -> None:
        """Write the generated series to a Parquet file.

        Parameters
        ----------
        path : str
            Destination file path.
        **kwargs
            Forwarded to pandas.DataFrame.to_parquet.

        Raises
        ------
        RuntimeError
            If .generate() has not been called.
        """
        if self.series is None:
            raise RuntimeError("No data to export — call .generate() first.")
        self.series.to_parquet(path, index=False, **kwargs)

    # Private helpers

    def _validate(
        self,
        n: int,
        k: int,
        returns,
        volatilities,
        persistence: float,
        transition_matrix,
        noise: str,
        regime_type: str,
        p0: float,
        sigma: float,
        scale: float,
        df: float,
        random_state,
    ) -> None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if (returns is None) != (volatilities is None):
            raise ValueError("returns and volatilities must both be provided or both be None")
        if returns is not None:
            if len(returns) != k:
                raise ValueError(
                    f"len(returns) must equal k ({k}), got {len(returns)}"
                )
            if len(volatilities) != k:
                raise ValueError(
                    f"len(volatilities) must equal k ({k}), got {len(volatilities)}"
                )
            if any(v <= 0 for v in volatilities):
                raise ValueError("all volatilities must be > 0")
        if transition_matrix is not None:
            tm = np.array(transition_matrix)
            if tm.shape != (k, k):
                raise ValueError(
                    f"transition_matrix must have shape ({k}, {k}), got {tm.shape}"
                )
            if not np.all(tm >= 0):
                raise ValueError("transition_matrix entries must be non-negative")
            if not np.allclose(tm.sum(axis=1), 1.0):
                raise ValueError("transition_matrix rows must each sum to 1")
        if not 0 < persistence < 1:
            raise ValueError(f"persistence must be in (0, 1), got {persistence!r}")
        if noise not in _NOISE_TYPES:
            raise ValueError(f"noise must be one of {_NOISE_TYPES}, got {noise!r}")
        if regime_type not in _REGIME_TYPES:
            raise ValueError(
                f"regime_type must be one of {_REGIME_TYPES}, got {regime_type!r}"
            )
        if p0 <= 0:
            raise ValueError(f"p0 must be > 0, got {p0!r}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma!r}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale!r}")
        if df <= 0:
            raise ValueError(f"df must be > 0, got {df!r}")
        if df <= 2:
            warnings.warn(
                f"df={df!r} gives infinite variance in the Student-t distribution "
                "(df > 2 required for finite variance).",
                UserWarning,
                stacklevel=3,
            )
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError(
                f"random_state must be an int or None, got {type(random_state).__name__!r}"
            )

    def _build_regimes(self) -> list[dict]:
        if self._returns_input is not None:
            return [
                {
                    "regime": i,
                    "mu":     float(self._returns_input[i]),
                    "sigma":  float(self._volatilities_input[i]),
                }
                for i in range(self.k)
            ]
        return self._auto_regimes()

    def _auto_regimes(self) -> list[dict]:
        if self.regime_type == "volatility":
            mus    = np.linspace(0.0003, -0.0002, self.k)
            sigmas = np.linspace(0.005, 0.04, self.k)
        elif self.regime_type == "returns":
            mus    = np.linspace(-0.001, 0.001, self.k)
            sigmas = np.full(self.k, 0.015)
        else:  # mixed
            mus    = np.linspace(-0.0005, 0.0005, self.k)
            sigmas = np.linspace(0.008, 0.04, self.k)

        return [
            {"regime": i, "mu": float(mus[i]), "sigma": float(sigmas[i])}
            for i in range(self.k)
        ]

    def _build_transition_matrix(self, rng: np.random.Generator) -> np.ndarray:
        if self._transition_matrix_input is not None:
            return np.array(self._transition_matrix_input, dtype=float)

        if self.k == 1:
            return np.array([[1.0]])

        tm = np.zeros((self.k, self.k))
        for i in range(self.k):
            off     = [j for j in range(self.k) if j != i]
            weights = rng.dirichlet(np.ones(len(off)))
            tm[i, i] = self.persistence
            for j, w in zip(off, weights):
                tm[i, j] = (1 - self.persistence) * w

        return tm

    def _simulate_states(self, tm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Pre-draw all uniforms in one vectorised call; use searchsorted (C-level)
        # against the cumulative transition rows; much faster than rng.choice(n times).
        cum_tm = np.cumsum(tm, axis=1)               # (k, k) cumulative row probabilities
        u      = rng.random(self.n)                  # all uniforms at once
        states = np.empty(self.n, dtype=np.intp)
        states[0] = rng.integers(0, self.k)
        for t in range(1, self.n):
            states[t] = np.searchsorted(cum_tm[states[t - 1]], u[t])
        return states

    def _generate_noise(self, rng: np.random.Generator) -> np.ndarray:
        if self.noise == "gaussian":
            return self.sigma * rng.standard_normal(self.n)
        return self.scale * rng.standard_t(self.df, size=self.n)

    def _compute_returns(
        self,
        states:  np.ndarray,
        regimes: list[dict],
        eps:     np.ndarray,
    ) -> np.ndarray:
        mus    = np.array([r["mu"]    for r in regimes])
        sigmas = np.array([r["sigma"] for r in regimes])
        return mus[states] + sigmas[states] * eps

    def _build_metadata(self) -> dict:
        noise_params = (
            {"df": self.df, "scale": self.scale}
            if self.noise == "student_t"
            else {"sigma": self.sigma}
        )
        return {
            "n":            self.n,
            "k":            self.k,
            "regime_type":  self.regime_type,
            "noise":        self.noise,
            "noise_params": noise_params,
            "persistence":  self.persistence,
            "p0":           self.p0,
            "random_state": self.random_state,
        }

    def _build_summary(self, ret: np.ndarray, states: np.ndarray) -> dict:
        r = pd.Series(ret)

        # Single O(n) pass over all regimes using itertools.groupby
        all_runs: dict[int, list[int]] = {i: [] for i in range(self.k)}
        for regime, grp in groupby(states.tolist()):
            all_runs[regime].append(sum(1 for _ in grp))

        freq:         dict[int, float] = {}
        durations:    dict[int, dict]  = {}
        regime_stats: dict[int, dict]  = {}

        for i in range(self.k):
            mask = states == i
            freq[i] = float(mask.mean())

            runs = all_runs[i]
            durations[i] = {
                "avg": float(np.mean(runs)) if runs else 0.0,
                "max": int(np.max(runs))    if runs else 0,
                "min": int(np.min(runs))    if runs else 0,
            }

            regime_ret = ret[mask]
            regime_stats[i] = {
                "mean":  float(np.mean(regime_ret)) if mask.any() else 0.0,
                "std":   float(np.std(regime_ret))  if mask.any() else 0.0,
                "count": int(mask.sum()),
            }

        return {
            "n":                 self.n,
            "mean":              float(r.mean()),
            "std":               float(r.std()),
            "variance":          float(r.var()),
            "skewness":          float(r.skew()),
            "excess_kurtosis":   float(r.kurtosis()),
            "min":               float(r.min()),
            "max":               float(r.max()),
            "median":            float(r.median()),
            "q1":                float(r.quantile(0.25)),
            "q3":                float(r.quantile(0.75)),
            "regime_frequencies": freq,
            "regime_durations":   durations,
            "regime_statistics":  regime_stats,
        }
    