"""Tests for Conformal Prediction modules.

Tests:
- ConformalPrediction class in calibration_tools.py (legacy, operates on raw predictions)
- ConformalPredictor class in conformal.py (model-wrapping split conformal prediction)
- ConformalConfig validation
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.bayesian.calibration_tools import ConformalPrediction
from opifex.neural.bayesian.conformal import ConformalConfig, ConformalPredictor


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Provide shared random number generators."""
    return nnx.Rngs(0)


# ── Helper models for testing ──────────────────────────────────────────


class SimpleMLP(nnx.Module):
    """Minimal MLP for testing conformal prediction wrapping."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(in_features, 32, rngs=rngs)
        self.linear2 = nnx.Linear(32, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through two linear layers with tanh activation."""
        h = jnp.tanh(self.linear1(x))
        return self.linear2(h)


class SimpleOperatorMLP(nnx.Module):
    """Minimal MLP mimicking an operator network for testing."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(in_features, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 32, rngs=rngs)
        self.linear3 = nnx.Linear(32, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through three linear layers with tanh activations."""
        h = jnp.tanh(self.linear1(x))
        h = jnp.tanh(self.linear2(h))
        return self.linear3(h)


# =========================================================================
# ConformalPrediction Tests (legacy class from calibration_tools)
# =========================================================================


class TestConformalPrediction:
    """Test the legacy ConformalPrediction module from calibration_tools."""

    def test_init(self, rngs: nnx.Rngs) -> None:
        """ConformalPrediction should initialize with default alpha."""
        cp = ConformalPrediction(rngs=rngs)
        assert cp.alpha == 0.1

    def test_init_custom_alpha(self, rngs: nnx.Rngs) -> None:
        """Should accept custom miscoverage level."""
        cp = ConformalPrediction(alpha=0.05, rngs=rngs)
        assert cp.alpha == 0.05

    def test_calibrate_sets_quantile(self, rngs: nnx.Rngs) -> None:
        """calibrate should compute and store quantile from residuals."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true_values = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(predictions, true_values)
        assert float(cp.quantile.value) > 0

    def test_predict_intervals_shape(self, rngs: nnx.Rngs) -> None:
        """predict_intervals should return (lower, upper) with same shape."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        cal_pred = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cal_true = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(cal_pred, cal_true)

        test_pred = jnp.array([6.0, 7.0, 8.0])
        lower, upper = cp.predict_intervals(test_pred)
        assert lower.shape == test_pred.shape
        assert upper.shape == test_pred.shape

    def test_intervals_symmetric(self, rngs: nnx.Rngs) -> None:
        """Intervals should be symmetric around predictions."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        cal_pred = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cal_true = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(cal_pred, cal_true)

        test_pred = jnp.array([10.0])
        lower, upper = cp.predict_intervals(test_pred)

        width_above = upper - test_pred
        width_below = test_pred - lower
        assert jnp.allclose(width_above, width_below)

    def test_intervals_contain_predictions(self, rngs: nnx.Rngs) -> None:
        """Predictions should lie within intervals."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        cal_pred = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cal_true = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        cp.calibrate(cal_pred, cal_true)

        test_pred = jnp.array([6.0, 7.0, 8.0])
        lower, upper = cp.predict_intervals(test_pred)
        assert jnp.all(test_pred >= lower)
        assert jnp.all(test_pred <= upper)

    def test_coverage_perfect(self, rngs: nnx.Rngs) -> None:
        """Coverage should be 1.0 when all true values are within intervals."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        lower = jnp.array([0.0, 1.0, 2.0])
        upper = jnp.array([2.0, 3.0, 4.0])
        true_vals = jnp.array([1.0, 2.0, 3.0])
        coverage = cp.compute_coverage(lower, upper, true_vals)
        assert coverage == 1.0

    def test_coverage_partial(self, rngs: nnx.Rngs) -> None:
        """Coverage should reflect fraction of covered points."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        lower = jnp.array([0.0, 1.0, 2.0, 3.0])
        upper = jnp.array([1.0, 2.0, 3.0, 4.0])
        true_vals = jnp.array([0.5, 1.5, 2.5, 5.0])
        coverage = cp.compute_coverage(lower, upper, true_vals)
        assert coverage == 0.75

    def test_coverage_guarantees(self, rngs: nnx.Rngs) -> None:
        """With enough calibration data, coverage >= 1-alpha (asymptotic)."""
        cp = ConformalPrediction(alpha=0.2, rngs=rngs)

        key = jax.random.PRNGKey(42)
        n_cal = 200
        cal_pred = jax.random.normal(key, (n_cal,))
        noise = 0.5 * jax.random.normal(jax.random.PRNGKey(43), (n_cal,))
        cal_true = cal_pred + noise
        cp.calibrate(cal_pred, cal_true)

        n_test = 100
        test_pred = jax.random.normal(jax.random.PRNGKey(44), (n_test,))
        test_noise = 0.5 * jax.random.normal(jax.random.PRNGKey(45), (n_test,))
        test_true = test_pred + test_noise

        lower, upper = cp.predict_intervals(test_pred)
        coverage = cp.compute_coverage(lower, upper, test_true)
        assert coverage >= 0.7, f"Coverage {coverage} too low for alpha=0.2"

    def test_wider_interval_more_coverage(self, rngs: nnx.Rngs) -> None:
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

        assert coverages[0] <= coverages[1] <= coverages[2]

    def test_quantile_finite(self, rngs: nnx.Rngs) -> None:
        """Quantile should be finite after calibration."""
        cp = ConformalPrediction(alpha=0.1, rngs=rngs)
        cal_pred = jnp.linspace(0, 10, 50)
        cal_true = cal_pred + 0.1
        cp.calibrate(cal_pred, cal_true)
        assert jnp.isfinite(cp.quantile.value)


# =========================================================================
# ConformalConfig Tests
# =========================================================================


class TestConformalConfig:
    """Test ConformalConfig validation."""

    def test_default_alpha(self) -> None:
        """Default alpha should be 0.1."""
        config = ConformalConfig()
        assert config.alpha == 0.1

    def test_custom_alpha(self) -> None:
        """Should accept valid alpha values."""
        config = ConformalConfig(alpha=0.05)
        assert config.alpha == 0.05

    def test_conformal_config_validation(self) -> None:
        """Invalid alpha values must be rejected."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalConfig(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalConfig(alpha=1.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalConfig(alpha=-0.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalConfig(alpha=1.5)


# =========================================================================
# ConformalPredictor Tests (model-wrapping split conformal prediction)
# =========================================================================


class TestConformalPredictor:
    """Test the ConformalPredictor that wraps NNX models."""

    @pytest.fixture
    def pinn_model(self) -> SimpleMLP:
        """Create a simple MLP acting as a PINN for testing."""
        return SimpleMLP(in_features=1, out_features=1, rngs=nnx.Rngs(42))

    @pytest.fixture
    def operator_model(self) -> SimpleOperatorMLP:
        """Create a simple operator-style MLP for testing."""
        return SimpleOperatorMLP(in_features=2, out_features=1, rngs=nnx.Rngs(99))

    def test_conformal_calibration_coverage(self, pinn_model: SimpleMLP) -> None:
        """90% nominal coverage should yield ~90% empirical coverage (within 5%)."""
        config = ConformalConfig(alpha=0.1)
        predictor = ConformalPredictor(model=pinn_model, config=config)

        # Generate calibration data: y = sin(x) + small noise
        key = jax.random.PRNGKey(0)
        n_cal = 500
        x_cal = jax.random.uniform(key, (n_cal, 1), minval=-3.0, maxval=3.0)
        y_cal = jnp.sin(x_cal) + 0.1 * jax.random.normal(
            jax.random.PRNGKey(1), (n_cal, 1)
        )

        predictor.calibrate(x_cal, y_cal)

        # Generate test data from the same distribution
        n_test = 300
        x_test = jax.random.uniform(
            jax.random.PRNGKey(2), (n_test, 1), minval=-3.0, maxval=3.0
        )
        y_test = jnp.sin(x_test) + 0.1 * jax.random.normal(
            jax.random.PRNGKey(3), (n_test, 1)
        )

        predictions, lower, upper = predictor.predict_with_intervals(x_test)

        # Compute empirical coverage
        in_interval = (y_test >= lower) & (y_test <= upper)
        empirical_coverage = float(jnp.mean(in_interval))

        # Coverage should be within +/-5% of nominal 90%
        assert empirical_coverage >= 0.85, (
            f"Coverage {empirical_coverage:.3f} below 85% "
            f"(nominal 90% with 5% tolerance)"
        )
        assert empirical_coverage <= 0.95, (
            f"Coverage {empirical_coverage:.3f} above 95% "
            f"(nominal 90% with 5% tolerance)"
        )
        # Predictions should match model output shape
        assert predictions.shape == (n_test, 1)
        assert lower.shape == (n_test, 1)
        assert upper.shape == (n_test, 1)

    def test_conformal_intervals_tighten(self, pinn_model: SimpleMLP) -> None:
        """More calibration data should produce tighter intervals."""
        config = ConformalConfig(alpha=0.1)

        # Generate data: y = model(x) + small noise
        # Use the actual model to generate y so residuals are just noise
        key = jax.random.PRNGKey(10)
        x_all = jax.random.uniform(key, (1000, 1), minval=-2.0, maxval=2.0)
        noise = 0.05 * jax.random.normal(jax.random.PRNGKey(11), (1000, 1))
        y_all = pinn_model(x_all) + noise

        widths = []
        for n_cal in [50, 200, 800]:
            predictor = ConformalPredictor(model=pinn_model, config=config)
            x_subset = x_all[:n_cal]
            y_subset = y_all[:n_cal]
            predictor.calibrate(x_subset, y_subset)

            x_test = jax.random.uniform(
                jax.random.PRNGKey(20), (100, 1), minval=-2.0, maxval=2.0
            )
            _, lower, upper = predictor.predict_with_intervals(x_test)
            mean_width = float(jnp.mean(upper - lower))
            widths.append(mean_width)

        # With more data, the quantile estimate stabilises and intervals
        # should not grow; they should tighten or stay similar.
        # Since the noise is fixed (0.05 scale), with more data the quantile
        # converges to the true value, so width should decrease or stay close.
        assert widths[-1] <= widths[0] * 1.1, (
            f"Intervals did not tighten: widths={widths}. "
            f"Width with 800 points ({widths[-1]:.4f}) should be <= "
            f"width with 50 points ({widths[0]:.4f}) * 1.1"
        )

    def test_conformal_works_with_pinn(self, pinn_model: SimpleMLP) -> None:
        """ConformalPredictor should wrap an NNX model acting as a PINN."""
        predictor = ConformalPredictor(model=pinn_model)

        # Calibrate with some data
        x_cal = jnp.linspace(-1.0, 1.0, 50).reshape(-1, 1)
        y_cal = jnp.sin(x_cal)
        predictor.calibrate(x_cal, y_cal)

        # Predict with intervals
        x_test = jnp.linspace(-0.5, 0.5, 10).reshape(-1, 1)
        predictions, lower, upper = predictor.predict_with_intervals(x_test)

        # Basic structural checks
        assert predictions.shape == (10, 1)
        assert lower.shape == (10, 1)
        assert upper.shape == (10, 1)
        # Intervals must contain the point prediction
        assert jnp.all(predictions >= lower)
        assert jnp.all(predictions <= upper)
        # Intervals should have positive width
        assert jnp.all(upper - lower > 0)

    def test_conformal_works_with_operator(
        self, operator_model: SimpleOperatorMLP
    ) -> None:
        """ConformalPredictor should wrap an NNX model acting as an operator."""
        predictor = ConformalPredictor(model=operator_model)

        # Calibrate with 2D input data
        key = jax.random.PRNGKey(77)
        x_cal = jax.random.normal(key, (100, 2))
        y_cal = jnp.sum(x_cal**2, axis=-1, keepdims=True)
        predictor.calibrate(x_cal, y_cal)

        # Predict with intervals
        x_test = jax.random.normal(jax.random.PRNGKey(78), (20, 2))
        predictions, lower, upper = predictor.predict_with_intervals(x_test)

        # Basic structural checks
        assert predictions.shape == (20, 1)
        assert lower.shape == (20, 1)
        assert upper.shape == (20, 1)
        # Intervals must contain the point prediction
        assert jnp.all(predictions >= lower)
        assert jnp.all(predictions <= upper)
        # Intervals should have positive width
        assert jnp.all(upper - lower > 0)

    def test_predict_before_calibrate_raises(self, pinn_model: SimpleMLP) -> None:
        """Calling predict_with_intervals before calibrate should raise."""
        predictor = ConformalPredictor(model=pinn_model)
        x_test = jnp.ones((5, 1))
        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.predict_with_intervals(x_test)

    def test_default_config(self, pinn_model: SimpleMLP) -> None:
        """ConformalPredictor with no config should use default alpha=0.1."""
        predictor = ConformalPredictor(model=pinn_model)
        assert predictor.config.alpha == 0.1
