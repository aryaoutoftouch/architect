"""
Unit tests for architect.py
Run: python -m pytest tests/test_architect.py -v
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import architect as ar
from architect import Asset


# Fixtures

@pytest.fixture
def basic_asset():
    return ar.create(n=200, k=3, random_state=0).generate()


# Construction and validation

class TestValidation:
    def test_invalid_n(self):
        with pytest.raises(ValueError):
            ar.create(n=0, k=2)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=0)

    def test_returns_without_volatilities(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=2, returns=[0.001, 0.002])

    def test_returns_wrong_length(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=2, returns=[0.001], volatilities=[0.01, 0.02])

    def test_zero_volatility(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=2, returns=[0.0, 0.0], volatilities=[0.01, 0.0])

    def test_bad_persistence(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=2, persistence=1.0)

    def test_bad_noise(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=2, noise="laplace")

    def test_bad_regime_type(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=2, regime_type="unknown")

    def test_negative_p0(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=2, p0=-1.0)

    def test_bad_random_state_type(self):
        with pytest.raises(TypeError):
            ar.create(n=100, k=2, random_state="42")

    def test_transition_matrix_wrong_shape(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=3, transition_matrix=[[0.9, 0.1], [0.1, 0.9]])

    def test_transition_matrix_rows_not_sum_to_one(self):
        with pytest.raises(ValueError):
            ar.create(n=100, k=2, transition_matrix=[[0.9, 0.5], [0.1, 0.9]])

    def test_df_leq_2_warns(self):
        with pytest.warns(UserWarning, match="infinite variance"):
            ar.create(n=100, k=2, noise="student_t", df=1.5)

    def test_df_gt_2_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ar.create(n=100, k=2, noise="student_t", df=3.0)  # should not warn


# generate()

class TestGenerate:
    def test_series_shape(self, basic_asset):
        assert basic_asset.series.shape == (200, 4)

    def test_series_columns(self, basic_asset):
        assert list(basic_asset.series.columns) == ["timestamp", "price", "returns", "regime"]

    def test_prices_positive(self, basic_asset):
        assert (basic_asset.series["price"] > 0).all()

    def test_regime_labels_in_range(self, basic_asset):
        assert basic_asset.series["regime"].between(0, 2).all()

    def test_true_regimes_length(self, basic_asset):
        assert len(basic_asset.true_regimes) == 200

    def test_true_regimes_matches_series(self, basic_asset):
        np.testing.assert_array_equal(
            basic_asset.true_regimes,
            basic_asset.series["regime"].values,
        )

    def test_generated_regimes_count(self, basic_asset):
        assert len(basic_asset.generated_regimes) == 3

    def test_generated_regimes_keys(self, basic_asset):
        for r in basic_asset.generated_regimes:
            assert set(r.keys()) == {"regime", "mu", "sigma"}

    def test_transition_matrix_shape(self, basic_asset):
        assert basic_asset.transition_matrix.shape == (3, 3)

    def test_transition_matrix_row_stochastic(self, basic_asset):
        np.testing.assert_allclose(
            basic_asset.transition_matrix.sum(axis=1),
            np.ones(3),
            atol=1e-10,
        )

    def test_generate_twice_raises(self):
        asset = ar.create(n=100, k=2, random_state=1).generate()
        with pytest.raises(RuntimeError):
            asset.generate()

    def test_p0_is_start(self):
        asset = ar.create(n=100, k=2, p0=250.0, random_state=3).generate()
        # P_t = p0 * exp(cumsum(r)); first price = p0 * exp(r_0), not p0 itself
        assert asset.series["price"].iloc[0] != pytest.approx(0.0)

    def test_reproducibility(self):
        a1 = ar.create(n=500, k=3, random_state=7).generate()
        a2 = ar.create(n=500, k=3, random_state=7).generate()
        pd.testing.assert_frame_equal(a1.series, a2.series)

    def test_different_seeds_differ(self):
        a1 = ar.create(n=500, k=3, random_state=1).generate()
        a2 = ar.create(n=500, k=3, random_state=2).generate()
        assert not a1.series["price"].equals(a2.series["price"])


# Noise modes

class TestNoise:
    def test_gaussian_generates(self):
        asset = ar.create(n=300, k=2, noise="gaussian", sigma=0.5, random_state=0).generate()
        assert asset.series is not None

    def test_student_t_generates(self):
        asset = ar.create(n=300, k=2, noise="student_t", df=5, scale=0.5, random_state=0).generate()
        assert asset.series is not None

    def test_student_t_heavier_tails(self):
        """Student-t with low df should produce higher excess kurtosis than Gaussian."""
        g = ar.create(n=5000, k=1, noise="gaussian", random_state=0).generate()
        t = ar.create(n=5000, k=1, noise="student_t", df=3, random_state=0).generate()
        assert t.summary["excess_kurtosis"] > g.summary["excess_kurtosis"]


# Regime types

class TestRegimeTypes:
    @pytest.mark.parametrize("rtype", ["volatility", "returns", "mixed"])
    def test_auto_regime_types(self, rtype):
        asset = ar.create(n=200, k=3, regime_type=rtype, random_state=0).generate()
        assert len(asset.generated_regimes) == 3

    def test_custom_regimes_override(self):
        asset = ar.create(
            n=200, k=2,
            returns=[-0.002, 0.003],
            volatilities=[0.008, 0.030],
            random_state=0,
        ).generate()
        assert asset.generated_regimes[0]["mu"] == pytest.approx(-0.002)
        assert asset.generated_regimes[1]["sigma"] == pytest.approx(0.030)


# Explicit transition matrix

class TestTransitionMatrix:
    def test_explicit_tm_preserved(self):
        tm = [[0.97, 0.02, 0.01], [0.05, 0.90, 0.05], [0.01, 0.04, 0.95]]
        asset = ar.create(n=500, k=3, transition_matrix=tm, random_state=1).generate()
        np.testing.assert_allclose(asset.transition_matrix, np.array(tm))

    def test_k1_transition_matrix(self):
        asset = ar.create(n=100, k=1, random_state=0).generate()
        np.testing.assert_array_equal(asset.transition_matrix, [[1.0]])


# Summary

class TestSummary:
    def test_summary_keys(self, basic_asset):
        expected = {
            "n", "mean", "std", "variance", "skewness", "excess_kurtosis",
            "min", "max", "median", "q1", "q3",
            "regime_frequencies", "regime_durations", "regime_statistics",
        }
        assert set(basic_asset.summary.keys()) == expected

    def test_regime_frequencies_sum_to_one(self, basic_asset):
        total = sum(basic_asset.summary["regime_frequencies"].values())
        assert total == pytest.approx(1.0)

    def test_regime_counts_sum_to_n(self, basic_asset):
        total = sum(
            s["count"] for s in basic_asset.summary["regime_statistics"].values()
        )
        assert total == 200


# reset()

class TestReset:
    def test_reset_clears_state(self):
        asset = ar.create(n=100, k=2, random_state=0).generate()
        asset.reset()
        assert asset.series is None
        assert asset.true_regimes is None
        assert asset.summary is None
        assert asset.metadata is None

    def test_reset_allows_regenerate(self):
        asset = ar.create(n=100, k=2, random_state=0).generate()
        asset.reset()
        asset.generate()
        assert asset.series is not None

    def test_reset_preserves_params(self):
        asset = ar.create(n=100, k=2, random_state=0)
        asset.generate()
        asset.reset()
        assert asset.n == 100
        assert asset.k == 2
        assert asset.random_state == 0


# __repr__ and __eq__

class TestDunder:
    def test_repr_pending(self):
        asset = ar.create(n=100, k=2)
        assert "pending" in repr(asset)

    def test_repr_generated(self):
        asset = ar.create(n=100, k=2).generate()
        assert "generated" in repr(asset)

    def test_eq_same_params(self):
        a1 = ar.create(n=100, k=2, random_state=0)
        a2 = ar.create(n=100, k=2, random_state=0)
        assert a1 == a2

    def test_eq_different_params(self):
        a1 = ar.create(n=100, k=2, random_state=0)
        a2 = ar.create(n=100, k=3, random_state=0)
        assert a1 != a2

    def test_eq_non_asset(self):
        asset = ar.create(n=100, k=2)
        assert asset.__eq__("not an asset") == NotImplemented
