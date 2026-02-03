"""Tests for NTK-based training diagnostics.

TDD: These tests define the expected behavior for NTK diagnostics and callbacks.
"""

import jax.numpy as jnp
from flax import nnx


class TestModeWiseErrorDecay:
    """Test mode-wise error decay computation."""

    def test_compute_mode_coefficients(self):
        """Should project residuals onto eigenbasis."""
        from opifex.core.physics.ntk.diagnostics import compute_mode_coefficients

        residuals = jnp.array([1.0, 2.0, 3.0])
        eigenvectors = jnp.eye(3)  # Identity for simplicity

        coeffs = compute_mode_coefficients(residuals, eigenvectors)

        assert coeffs.shape == (3,)
        # With identity eigenvectors, coeffs should equal residuals
        assert jnp.allclose(coeffs, residuals)

    def test_compute_mode_decay_factors(self):
        """Should compute per-mode decay factors."""
        from opifex.core.physics.ntk.diagnostics import compute_mode_decay_factors

        eigenvalues = jnp.array([1.0, 0.5, 0.1])
        learning_rate = 0.1
        iteration = 10

        factors = compute_mode_decay_factors(eigenvalues, learning_rate, iteration)

        assert factors.shape == (3,)
        # (1 - lr * eigenvalue)^iteration
        expected = jnp.array(
            [
                (1 - 0.1 * 1.0) ** 10,
                (1 - 0.1 * 0.5) ** 10,
                (1 - 0.1 * 0.1) ** 10,
            ]
        )
        assert jnp.allclose(factors, expected, atol=1e-5)

    def test_predict_mode_errors(self):
        """Should predict per-mode errors at given iteration."""
        from opifex.core.physics.ntk.diagnostics import predict_mode_errors

        initial_coeffs = jnp.array([1.0, 1.0])
        eigenvalues = jnp.array([0.5, 0.1])
        learning_rate = 0.1
        iteration = 5

        errors = predict_mode_errors(
            initial_coeffs, eigenvalues, learning_rate, iteration
        )

        assert errors.shape == (2,)
        assert jnp.all(jnp.abs(errors) <= jnp.abs(initial_coeffs))  # Should decay


class TestNTKTrainingDiagnostics:
    """Test NTK training diagnostics class."""

    def test_create_diagnostics(self):
        """Should create diagnostics tracker."""
        from opifex.core.physics.ntk.diagnostics import NTKTrainingDiagnostics

        diagnostics = NTKTrainingDiagnostics(track_history=True)
        assert diagnostics is not None

    def test_update_diagnostics(self):
        """Should update diagnostics with new NTK info."""
        from opifex.core.physics.ntk.diagnostics import NTKTrainingDiagnostics

        diagnostics = NTKTrainingDiagnostics(track_history=True)

        # Create simple NTK
        ntk = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        residuals = jnp.array([1.0, 0.5])
        learning_rate = 0.01

        diagnostics.update(ntk, residuals, learning_rate, iteration=0)

        assert diagnostics.eigenvalues is not None
        assert diagnostics.condition_number > 0

    def test_get_convergence_prediction(self):
        """Should predict convergence based on NTK."""
        from opifex.core.physics.ntk.diagnostics import NTKTrainingDiagnostics

        diagnostics = NTKTrainingDiagnostics()

        ntk = jnp.array([[2.0, 0.0], [0.0, 1.0]])
        residuals = jnp.array([1.0, 1.0])
        learning_rate = 0.01

        diagnostics.update(ntk, residuals, learning_rate, iteration=0)

        # Predict errors at future iteration
        predicted = diagnostics.predict_errors_at(iteration=100)

        assert predicted.shape == (2,)
        assert jnp.all(jnp.abs(predicted) < jnp.abs(residuals))  # Should decrease


class TestSpectralBiasDetection:
    """Test spectral bias detection utilities."""

    def test_detect_spectral_bias(self):
        """Should detect spectral bias from eigenvalue distribution."""
        from opifex.core.physics.ntk.diagnostics import detect_spectral_bias

        # Large eigenvalue spread indicates spectral bias
        eigenvalues_biased = jnp.array([100.0, 10.0, 1.0, 0.1, 0.01])
        eigenvalues_balanced = jnp.array([2.0, 1.5, 1.0, 0.8, 0.5])

        bias_biased = detect_spectral_bias(eigenvalues_biased)
        bias_balanced = detect_spectral_bias(eigenvalues_balanced)

        assert bias_biased > bias_balanced

    def test_identify_slow_modes(self):
        """Should identify slow-converging modes."""
        from opifex.core.physics.ntk.diagnostics import identify_slow_modes

        eigenvalues = jnp.array([1.0, 0.5, 0.1, 0.01, 0.001])
        learning_rate = 0.1

        slow_mode_mask = identify_slow_modes(eigenvalues, learning_rate, threshold=0.99)

        assert slow_mode_mask.shape == (5,)
        # Modes with small eigenvalues should be marked as slow
        assert slow_mode_mask[-1]  # Smallest eigenvalue
        assert not slow_mode_mask[0]  # Largest eigenvalue


class TestNTKDiagnosticsCallback:
    """Test NTK diagnostics callback for training."""

    def test_create_callback(self):
        """Should create diagnostics callback."""
        from opifex.core.physics.ntk.diagnostics import NTKDiagnosticsCallback

        callback = NTKDiagnosticsCallback(compute_frequency=10)
        assert callback is not None
        assert callback.frequency == 10

    def test_callback_stores_history(self):
        """Should store diagnostic history."""
        from opifex.core.physics.ntk.diagnostics import NTKDiagnosticsCallback

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(1, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        callback = NTKDiagnosticsCallback(compute_frequency=1)

        x = jnp.array([[0.5], [1.0]])

        # Simulate training iterations
        callback.on_step_end(model, x, step=0)
        callback.on_step_end(model, x, step=1)

        history = callback.get_history()
        assert len(history) >= 1

    def test_callback_condition_number_tracking(self):
        """Should track condition number over training."""
        from opifex.core.physics.ntk.diagnostics import NTKDiagnosticsCallback

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        callback = NTKDiagnosticsCallback(compute_frequency=1)

        x = jnp.array([[0.5, 0.5], [1.0, 0.0]])

        callback.on_step_end(model, x, step=0)

        history = callback.get_history()
        assert "condition_number" in history[0]
        assert history[0]["condition_number"] > 0


class TestConvergenceRateEstimation:
    """Test convergence rate estimation from NTK."""

    def test_estimate_convergence_rate(self):
        """Should estimate convergence rate from eigenvalues."""
        from opifex.core.physics.ntk.diagnostics import estimate_convergence_rate

        eigenvalues = jnp.array([1.0, 0.5, 0.1])
        learning_rate = 0.1

        rate = estimate_convergence_rate(eigenvalues, learning_rate)

        # Rate should be related to smallest eigenvalue
        assert rate > 0
        assert rate < 1  # Should converge (rate < 1)

    def test_estimate_epochs_to_convergence(self):
        """Should estimate epochs needed to reach target error."""
        from opifex.core.physics.ntk.diagnostics import estimate_epochs_to_convergence

        eigenvalues = jnp.array([1.0, 0.5, 0.1])
        learning_rate = 0.1
        target_reduction = 0.01  # Reduce error by 100x

        epochs = estimate_epochs_to_convergence(
            eigenvalues, learning_rate, target_reduction
        )

        assert epochs > 0
        assert jnp.isfinite(epochs)
