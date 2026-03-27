"""Tests for SINDy candidate library.

TDD: These tests define expected behavior for the feature library.
Implementation in src/opifex/discovery/sindy/library.py must pass them.
"""

import jax
import jax.numpy as jnp

from opifex.discovery.sindy.library import CandidateLibrary


class TestPolynomialLibrary:
    """Tests for polynomial basis function generation."""

    def test_degree_1_generates_identity_plus_constant(self):
        """Degree-1 polynomial on 2 features: [1, x0, x1]."""
        lib = CandidateLibrary(polynomial_degree=1)
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        theta = lib.transform(x)
        assert theta.shape == (2, 3)  # 1 + 2 features
        # First column should be constant 1
        assert jnp.allclose(theta[:, 0], jnp.ones(2))

    def test_degree_2_generates_quadratic_terms(self):
        """Degree-2: [1, x0, x1, x0^2, x0*x1, x1^2]."""
        lib = CandidateLibrary(polynomial_degree=2)
        x = jnp.array([[2.0, 3.0]])
        theta = lib.transform(x)
        assert theta.shape[1] == 6  # C(2+2,2) = 6
        # Check x0^2 = 4, x0*x1 = 6, x1^2 = 9 appear
        values = {float(v) for v in theta[0]}
        assert 4.0 in values  # x0^2
        assert 6.0 in values  # x0*x1
        assert 9.0 in values  # x1^2

    def test_degree_0_gives_constant_only(self):
        """Degree-0 polynomial: just the constant term [1]."""
        lib = CandidateLibrary(polynomial_degree=0)
        x = jnp.ones((5, 3))
        theta = lib.transform(x)
        assert theta.shape == (5, 1)
        assert jnp.allclose(theta, jnp.ones((5, 1)))

    def test_feature_names(self):
        """Library generates human-readable feature names."""
        lib = CandidateLibrary(polynomial_degree=2)
        x = jnp.ones((1, 2))
        lib.transform(x)
        names = lib.get_feature_names(["x", "y"])
        assert "1" in names
        assert "x" in names
        assert "y" in names
        assert len(names) == 6


class TestTrigonometricLibrary:
    """Tests for trigonometric basis functions."""

    def test_sin_cos_generation(self):
        """Trig library includes sin and cos of each feature."""
        lib = CandidateLibrary(polynomial_degree=0, include_trig=True, n_frequencies=2)
        x = jnp.array([[jnp.pi / 2]])
        theta = lib.transform(x)
        # Should have: 1 (constant) + 2*2 (sin/cos at 2 freqs) = 5
        assert theta.shape[1] == 5


class TestCustomLibrary:
    """Tests for custom basis functions."""

    def test_custom_functions(self):
        """Custom callable basis functions are applied."""
        lib = CandidateLibrary(
            polynomial_degree=0,
            custom_functions=[lambda x: jnp.exp(x[:, 0:1])],
        )
        x = jnp.array([[0.0, 1.0], [1.0, 2.0]])
        theta = lib.transform(x)
        # 1 (constant) + 1 (custom) = 2
        assert theta.shape[1] == 2
        assert jnp.allclose(theta[:, 1], jnp.exp(x[:, 0]))


class TestJITCompatibility:
    """Tests for JAX JIT compatibility."""

    def test_transform_is_jittable(self):
        """Library transform works under jax.jit."""
        lib = CandidateLibrary(polynomial_degree=2)
        x = jnp.ones((4, 3))
        jitted = jax.jit(lib.transform)
        theta = jitted(x)
        assert theta.shape[0] == 4
