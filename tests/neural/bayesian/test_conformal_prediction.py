"""Tests for Conformal Prediction module.

Tests the existing ConformalPrediction class in calibration_tools.py:
- Calibration from residuals
- Prediction intervals with coverage guarantees
- Empirical coverage computation
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.bayesian.calibration_tools import (
    ConformalPrediction,
)


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


# =========================================================================
# ConformalPrediction Tests
# =========================================================================


class TestConformalPrediction:
    """Test the ConformalPrediction module."""

    def test_init(self, rngs):
        """ConformalPrediction should initialize with default alpha."""
        cp = ConformalPrediction(rngs=rngs)
        assert cp.alpha == 0.1

    def test_init_custom_alpha(self, rngs):
        """Should accept custom miscoverage level."""
        cp = ConformalPrediction(alpha=0.05, rngs=rngs)
        assert cp.alpha == 0.05

    def test_calibrate_sets_quantile(self, rngs):
        """calibrate should compute and store quantile from residuals."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_values = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(predictions, true_values)
        assert float(cp.quantile.value) > 0

    def test_predict_intervals_shape(self, rngs):
        """predict_intervals should return (lower, upper) with same shape."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        # Calibrate first
        cal_pred = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cal_true = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(cal_pred, cal_true)

        # Predict intervals
        test_pred = jnp.array([6.0, 7.0, 8.0])
        lower, upper = cp.predict_intervals(test_pred)
        assert lower.shape == test_pred.shape
        assert upper.shape == test_pred.shape

    def test_intervals_symmetric(self, rngs):
        """Intervals should be symmetric around predictions."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        cal_pred = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cal_true = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(cal_pred, cal_true)

        test_pred = jnp.array([10.0])
        lower, upper = cp.predict_intervals(test_pred)

        # Width above = width below
        width_above = upper - test_pred
        width_below = test_pred - lower
        assert jnp.allclose(width_above, width_below)

    def test_intervals_contain_predictions(self, rngs):
        """Predictions should lie within intervals."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        cal_pred = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cal_true = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(cal_pred, cal_true)

        test_pred = jnp.array([6.0, 7.0, 8.0])
        lower, upper = cp.predict_intervals(test_pred)
        assert jnp.all(test_pred >= lower)
        assert jnp.all(test_pred <= upper)

    def test_coverage_perfect(self, rngs):
        """Coverage should be 1.0 when all true values are within intervals."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        lower = jnp.array([0.0, 1.0, 2.0])
        upper = jnp.array([2.0, 3.0, 4.0])
        true_vals = jnp.array([1.0, 2.0, 3.0])
        coverage = cp.compute_coverage(lower, upper, true_vals)
        assert coverage == 1.0

    def test_coverage_partial(self, rngs):
        """Coverage should reflect fraction of covered points."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        lower = jnp.array([0.0, 1.0, 2.0, 3.0])
        upper = jnp.array([1.0, 2.0, 3.0, 4.0])
        # First two are in [0,1] and [1,2], third is on boundary, fourth outside
        true_vals = jnp.array([0.5, 1.5, 2.5, 5.0])
        coverage = cp.compute_coverage(lower, upper, true_vals)
        assert coverage == 0.75

    def test_coverage_guarantees(self, rngs):
        """With enough calibration data, coverage >= 1-alpha (asymptotic)."""
        cp = ConformalPrediction(alpha=0.2, rngs=rngs)

        # Generate calibration data with known noise
        key = jax.random.PRNGKey(42)
        n_cal = 200
        cal_pred = jax.random.normal(key, (n_cal,))
        noise = 0.5 * jax.random.normal(jax.random.PRNGKey(43), (n_cal,))
        cal_true = cal_pred + noise
        cp.calibrate(cal_pred, cal_true)

        # Test on fresh data from same distribution
        n_test = 100
        test_pred = jax.random.normal(jax.random.PRNGKey(44), (n_test,))
        test_noise = 0.5 * jax.random.normal(jax.random.PRNGKey(45), (n_test,))
        test_true = test_pred + test_noise

        lower, upper = cp.predict_intervals(test_pred)
        coverage = cp.compute_coverage(lower, upper, test_true)
        # Coverage should be at least 1 - alpha - some margin
        assert coverage >= 0.7, f"Coverage {coverage} too low for alpha=0.2"

    def test_wider_interval_more_coverage(self, rngs):
        """Lower alpha (wider intervals) should give higher coverage."""
        key = jax.random.PRNGKey(0)
        n = 100
        pred = jax.random.normal(key, (n,))
        noise = 0.3 * jax.random.normal(jax.random.PRNGKey(1), (n,))
        true = pred + noise

        coverages = []
        for alpha in [0.5, 0.2, 0.05]:
            cp = ConformalPrediction(alpha=alpha, rngs=rngs)
            cp.calibrate(pred, true)
            lower, upper = cp.predict_intervals(pred)
            cov = cp.compute_coverage(lower, upper, true)
            coverages.append(cov)

        # Monotone: wider intervals → higher coverage
        assert coverages[0] <= coverages[1] <= coverages[2]

    def test_quantile_finite(self, rngs):
        """Quantile should be finite after calibration."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        cal_pred = jnp.linspace(0, 10, 50)
        cal_true = cal_pred + 0.1
        cp.calibrate(cal_pred, cal_true)
        assert jnp.isfinite(cp.quantile.value)
