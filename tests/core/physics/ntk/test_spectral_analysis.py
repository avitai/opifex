"""Tests for NTK spectral analysis.

TDD: These tests define the expected behavior for NTK eigenvalue analysis
and diagnostics.
"""

import jax.numpy as jnp
from flax import nnx


class TestNTKDiagnostics:
    """Test NTK diagnostics dataclass."""

    def test_create_diagnostics(self):
        """Should create diagnostics with eigenvalue data."""
        from opifex.core.physics.ntk.spectral_analysis import NTKDiagnostics

        eigenvalues = jnp.array([10.0, 5.0, 1.0, 0.1])
        diagnostics = NTKDiagnostics(
            eigenvalues=eigenvalues,
            condition_number=100.0,
            effective_rank=3.0,
            spectral_bias_indicator=0.5,
        )

        assert diagnostics is not None
        assert len(diagnostics.eigenvalues) == 4

    def test_diagnostics_from_ntk(self):
        """Should compute diagnostics from NTK matrix."""
        from opifex.core.physics.ntk.spectral_analysis import NTKDiagnostics

        # Create a positive semi-definite matrix
        ntk = jnp.array([[2.0, 0.5], [0.5, 1.0]])

        diagnostics = NTKDiagnostics.from_ntk(ntk)

        assert diagnostics is not None
        assert len(diagnostics.eigenvalues) == 2
        assert jnp.all(diagnostics.eigenvalues >= 0)


class TestConditionNumber:
    """Test condition number computation."""

    def test_compute_condition_number(self):
        """Should compute condition number from eigenvalues."""
        from opifex.core.physics.ntk.spectral_analysis import compute_condition_number

        eigenvalues = jnp.array([100.0, 10.0, 1.0])
        cond = compute_condition_number(eigenvalues)

        assert jnp.isclose(cond, 100.0)

    def test_condition_number_from_ntk(self):
        """Should compute condition number from NTK matrix."""
        from opifex.core.physics.ntk.spectral_analysis import (
            compute_condition_number_from_ntk,
        )

        # Create a matrix with known condition number
        ntk = jnp.diag(jnp.array([10.0, 1.0]))
        cond = compute_condition_number_from_ntk(ntk)

        assert jnp.isclose(cond, 10.0)

    def test_ill_conditioned_detection(self):
        """Should detect ill-conditioned NTK."""
        from opifex.core.physics.ntk.spectral_analysis import (
            compute_condition_number_from_ntk,
        )

        # Create an ill-conditioned matrix
        ntk = jnp.diag(jnp.array([1000.0, 0.001]))
        cond = compute_condition_number_from_ntk(ntk)

        assert cond > 1e5  # Very ill-conditioned


class TestEffectiveRank:
    """Test effective rank computation."""

    def test_compute_effective_rank(self):
        """Should compute effective rank from eigenvalues."""
        from opifex.core.physics.ntk.spectral_analysis import compute_effective_rank

        # All equal eigenvalues -> effective rank = n
        eigenvalues = jnp.array([1.0, 1.0, 1.0, 1.0])
        rank = compute_effective_rank(eigenvalues)

        assert jnp.isclose(rank, 4.0)

    def test_effective_rank_with_decay(self):
        """Effective rank should be lower with eigenvalue decay."""
        from opifex.core.physics.ntk.spectral_analysis import compute_effective_rank

        # Decaying eigenvalues -> lower effective rank
        eigenvalues = jnp.array([10.0, 1.0, 0.1, 0.01])
        rank = compute_effective_rank(eigenvalues)

        assert rank < 4.0  # Lower than full rank


class TestSpectralBias:
    """Test spectral bias analysis."""

    def test_compute_spectral_bias_indicator(self):
        """Should compute spectral bias indicator."""
        from opifex.core.physics.ntk.spectral_analysis import (
            compute_spectral_bias_indicator,
        )

        eigenvalues = jnp.array([100.0, 10.0, 1.0])
        bias = compute_spectral_bias_indicator(eigenvalues)

        assert jnp.isfinite(bias)
        assert bias >= 0  # Should be non-negative

    def test_spectral_bias_interpretation(self):
        """Higher spectral bias should indicate more spectral decay."""
        from opifex.core.physics.ntk.spectral_analysis import (
            compute_spectral_bias_indicator,
        )

        # Sharp decay
        sharp_decay = jnp.array([1000.0, 10.0, 0.1])
        # Gradual decay
        gradual_decay = jnp.array([10.0, 5.0, 2.0])

        bias_sharp = compute_spectral_bias_indicator(sharp_decay)
        bias_gradual = compute_spectral_bias_indicator(gradual_decay)

        assert bias_sharp > bias_gradual


class TestModeConvergence:
    """Test mode-wise convergence analysis."""

    def test_compute_mode_convergence_rates(self):
        """Should compute convergence rates for each eigenmode."""
        from opifex.core.physics.ntk.spectral_analysis import (
            compute_mode_convergence_rates,
        )

        eigenvalues = jnp.array([10.0, 5.0, 1.0])
        learning_rate = 0.01

        rates = compute_mode_convergence_rates(eigenvalues, learning_rate)

        assert len(rates) == 3
        assert jnp.all(rates >= 0)  # Convergence rates should be non-negative
        assert jnp.all(rates <= 1)  # Should be bounded by 1

    def test_larger_eigenvalue_faster_convergence(self):
        """Modes with larger eigenvalues should converge faster."""
        from opifex.core.physics.ntk.spectral_analysis import (
            compute_mode_convergence_rates,
        )

        eigenvalues = jnp.array([10.0, 1.0])
        learning_rate = 0.01

        rates = compute_mode_convergence_rates(eigenvalues, learning_rate)

        # First mode (larger eigenvalue) should converge faster (smaller rate)
        assert rates[0] < rates[1]


class TestNTKSpectralAnalyzer:
    """Test NTK spectral analyzer class."""

    def test_create_analyzer(self):
        """Should create spectral analyzer for model."""
        from opifex.core.physics.ntk.spectral_analysis import NTKSpectralAnalyzer

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        analyzer = NTKSpectralAnalyzer(model)

        assert analyzer is not None

    def test_analyze_returns_diagnostics(self):
        """Should return diagnostics from analysis."""
        from opifex.core.physics.ntk.spectral_analysis import NTKSpectralAnalyzer

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        analyzer = NTKSpectralAnalyzer(model)

        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        diagnostics = analyzer.analyze(x)

        assert diagnostics is not None
        assert len(diagnostics.eigenvalues) > 0

    def test_track_training_dynamics(self):
        """Should track NTK evolution during training."""
        from opifex.core.physics.ntk.spectral_analysis import NTKSpectralAnalyzer

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.linear = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        analyzer = NTKSpectralAnalyzer(model)

        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])

        # Track condition number over "training" (just repeated analysis here)
        cond_numbers = []
        for _ in range(3):
            diagnostics = analyzer.analyze(x)
            cond_numbers.append(diagnostics.condition_number)

        assert len(cond_numbers) == 3
        assert all(jnp.isfinite(c) for c in cond_numbers)


class TestPDEOrderEstimation:
    """Test PDE order estimation from NTK spectrum."""

    def test_estimate_pde_order(self):
        """Should estimate PDE order from spectral decay."""
        from opifex.core.physics.ntk.spectral_analysis import estimate_pde_order

        # Create eigenvalue spectrum typical of a PDE
        # κ(H) ≈ (ω_max / ω_min)^(2p) for PDE order p
        eigenvalues = jnp.array([100.0, 25.0, 6.25, 1.5625])

        estimated_order = estimate_pde_order(eigenvalues)

        # Should give a reasonable PDE order estimate
        assert estimated_order >= 0
        assert estimated_order <= 10  # Reasonable bound
