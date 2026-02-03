"""Tests for quantum constraint functions.

Following TDD: These tests are written BEFORE implementation.

Tests cover:
- Wavefunction normalization constraints
- Density positivity enforcement
- Hermiticity verification for operators
- Probability conservation in time evolution
- JAX compatibility (JIT, vmap, grad)
"""

import jax
import jax.numpy as jnp

from opifex.core.physics.quantum_constraints import (
    density_positivity_violation,
    hermiticity_violation,
    probability_conservation,
    wavefunction_normalization,
)


class TestWavefunctionNormalization:
    """Test wavefunction normalization constraint: ∫|ψ|²dx = 1."""

    def test_normalized_wavefunction(self):
        """Perfectly normalized wavefunction should have zero violation."""
        # Create normalized gaussian in 1D
        x = jnp.linspace(0, 1, 100)
        dx = x[1] - x[0]

        # Gaussian wavefunction
        psi = jnp.exp(-10 * (x - 0.5) ** 2)
        # Normalize it
        psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx)

        violation = wavefunction_normalization(psi, dx)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_unnormalized_wavefunction(self):
        """Unnormalized wavefunction should have positive violation."""
        x = jnp.linspace(0, 1, 100)
        dx = x[1] - x[0]

        psi = jnp.ones(100) * 2.0  # Clearly unnormalized (norm = 4)

        violation = wavefunction_normalization(psi, dx)
        assert violation > 0.0
        # Should be |4 - 1| = 3
        assert jnp.isclose(violation, 3.0, atol=0.1)

    def test_complex_wavefunction(self):
        """Should work with complex wavefunctions."""
        x = jnp.linspace(0, 1, 100)
        dx = x[1] - x[0]

        # Complex wavefunction with phase
        psi = jnp.exp(1j * 2 * jnp.pi * x - 10 * (x - 0.5) ** 2)
        # Normalize
        psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx)

        violation = wavefunction_normalization(psi, dx)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_zero_wavefunction(self):
        """Zero wavefunction should have violation of 1 (missing probability)."""
        x = jnp.linspace(0, 1, 100)
        dx = x[1] - x[0]

        psi = jnp.zeros(100)

        violation = wavefunction_normalization(psi, dx)
        assert jnp.isclose(violation, 1.0, atol=1e-6)

    def test_tolerance_parameter(self):
        """Tolerance parameter should filter small violations."""
        x = jnp.linspace(0, 1, 100)
        dx = x[1] - x[0]

        # Slightly unnormalized
        psi = jnp.ones(100)
        psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx)
        psi = psi * 1.00001  # Tiny violation

        # With tight tolerance, should detect
        violation_tight = wavefunction_normalization(psi, dx, tolerance=1e-8)
        assert violation_tight > 0.0

        # With loose tolerance, should ignore
        violation_loose = wavefunction_normalization(psi, dx, tolerance=1e-3)
        assert jnp.isclose(violation_loose, 0.0, atol=1e-10)

    def test_jit_compatibility(self):
        """Normalization check should be JIT-compatible."""

        @jax.jit
        def jitted_check(psi, dx):
            return wavefunction_normalization(psi, dx)

        x = jnp.linspace(0, 1, 50)
        dx = x[1] - x[0]
        psi = jnp.ones(50)

        result = jitted_check(psi, dx)
        assert jnp.isfinite(result)


class TestDensityPositivity:
    """Test density positivity constraint: ρ(x) ≥ 0 everywhere."""

    def test_positive_density(self):
        """Fully positive density should have zero violation."""
        rho = jnp.array([1.0, 2.0, 0.5, 3.0, 0.1])
        violation = density_positivity_violation(rho)
        assert jnp.isclose(violation, 0.0, atol=1e-10)

    def test_negative_density(self):
        """Negative density should have positive violation."""
        rho = jnp.array([1.0, -0.5, 2.0, -1.0])
        violation = density_positivity_violation(rho)
        assert violation > 0.0
        # Should be sum of negative parts: 0.5 + 1.0 = 1.5
        assert jnp.isclose(violation, 1.5, atol=1e-6)

    def test_zero_density(self):
        """Zero density is valid (no violation)."""
        rho = jnp.array([0.0, 0.0, 0.0])
        violation = density_positivity_violation(rho)
        assert jnp.isclose(violation, 0.0, atol=1e-10)

    def test_mixed_positive_negative(self):
        """Mixed positive and negative should sum negative parts."""
        rho = jnp.array([5.0, -2.0, 3.0, -1.0, 0.0])
        violation = density_positivity_violation(rho)
        # Only negative parts: 2.0 + 1.0 = 3.0
        assert jnp.isclose(violation, 3.0, atol=1e-6)

    def test_small_negative_tolerance(self):
        """Small negative values within tolerance should be ignored."""
        rho = jnp.array([1.0, -1e-8, 2.0, -5e-9])
        violation = density_positivity_violation(rho, tolerance=1e-6)
        assert jnp.isclose(violation, 0.0, atol=1e-10)

    def test_large_negative_not_ignored(self):
        """Large negative values should not be ignored even with tolerance."""
        rho = jnp.array([1.0, -0.01, 2.0])
        violation = density_positivity_violation(rho, tolerance=1e-6)
        assert violation > 0.0

    def test_vmap_compatibility(self):
        """Should work with batched densities."""
        rho_batch = jnp.array(
            [
                [1.0, 2.0, 0.5],  # All positive
                [3.0, -1.0, 2.0],  # One negative
                [0.5, 0.5, 0.5],  # All positive
            ]
        )

        violations = jax.vmap(density_positivity_violation)(rho_batch)

        assert violations.shape == (3,)
        assert jnp.isclose(violations[0], 0.0, atol=1e-10)
        assert violations[1] > 0.0
        assert jnp.isclose(violations[2], 0.0, atol=1e-10)

    def test_jit_compatibility(self):
        """Should be JIT-compatible."""

        @jax.jit
        def jitted_check(rho):
            return density_positivity_violation(rho)

        rho = jnp.array([1.0, -0.5, 2.0])
        result = jitted_check(rho)
        assert jnp.isfinite(result)


class TestHermiticityViolation:
    """Test Hermiticity verification: H = H†."""

    def test_hermitian_matrix_real(self):
        """Real symmetric matrix is Hermitian."""
        # Symmetric matrix
        H = jnp.array([[1.0, 2.0], [2.0, 3.0]])
        violation = hermiticity_violation(H)
        assert jnp.isclose(violation, 0.0, atol=1e-10)

    def test_hermitian_matrix_complex(self):
        """Complex Hermitian matrix should have zero violation."""
        # Hermitian matrix: H = H†
        H = jnp.array([[1.0 + 0j, 2.0 + 1j], [2.0 - 1j, 3.0 + 0j]])
        violation = hermiticity_violation(H)
        assert jnp.isclose(violation, 0.0, atol=1e-10)

    def test_pauli_matrices_hermitian(self):
        """Pauli matrices are Hermitian."""
        # Pauli X
        sigma_x = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        assert jnp.isclose(hermiticity_violation(sigma_x), 0.0, atol=1e-10)

        # Pauli Y
        sigma_y = jnp.array([[0.0, -1j], [1j, 0.0]])
        assert jnp.isclose(hermiticity_violation(sigma_y), 0.0, atol=1e-10)

        # Pauli Z
        sigma_z = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        assert jnp.isclose(hermiticity_violation(sigma_z), 0.0, atol=1e-10)

    def test_non_hermitian_matrix(self):
        """Non-Hermitian matrix should have positive violation."""
        H = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        violation = hermiticity_violation(H)
        assert violation > 0.0

    def test_anti_hermitian(self):
        """Anti-Hermitian matrix should have large violation."""
        # H = -H†
        H = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        violation = hermiticity_violation(H)
        assert violation > 0.0

    def test_identity_hermitian(self):
        """Identity matrix is Hermitian."""
        I = jnp.eye(3)
        violation = hermiticity_violation(I)
        assert jnp.isclose(violation, 0.0, atol=1e-10)

    def test_tolerance_parameter(self):
        """Tolerance should filter small violations."""
        # Nearly Hermitian (use perturbation representable in float32)
        H = jnp.array([[1.0, 2.0 + 1e-6], [2.0 - 1e-6, 3.0]])

        # Tight tolerance detects (use tolerance smaller than perturbation)
        violation_tight = hermiticity_violation(H, tolerance=1e-8)
        assert violation_tight > 0.0

        # Loose tolerance ignores
        violation_loose = hermiticity_violation(H, tolerance=1e-4)
        assert jnp.isclose(violation_loose, 0.0, atol=1e-10)

    def test_jit_compatibility(self):
        """Should be JIT-compatible."""

        @jax.jit
        def jitted_check(H):
            return hermiticity_violation(H)

        H = jnp.eye(2)
        result = jitted_check(H)
        assert jnp.isfinite(result)


class TestProbabilityConservation:
    """Test probability conservation: ||ψ(t)||² = ||ψ(0)||²."""

    def test_conserved_probability(self):
        """Total probability conserved should have zero violation."""
        # Two states with same norm
        psi_t0 = jnp.array([1.0, 0.0, 0.0])
        psi_t1 = jnp.array([0.0, 1.0, 0.0])

        violation = probability_conservation(psi_t0, psi_t1)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_probability_decay(self):
        """Probability loss should be detected."""
        psi_t0 = jnp.array([1.0, 0.0])
        psi_t1 = jnp.array([0.5, 0.0])  # Lost 3/4 of probability

        violation = probability_conservation(psi_t0, psi_t1)
        assert violation > 0.0

    def test_probability_increase(self):
        """Probability increase should be detected."""
        psi_t0 = jnp.array([0.5, 0.0])
        psi_t1 = jnp.array([1.0, 0.0])  # Gained probability

        violation = probability_conservation(psi_t0, psi_t1)
        assert violation > 0.0

    def test_complex_evolution(self):
        """Should work with complex wavefunctions."""
        psi_t0 = jnp.array([1.0 + 0j, 0.0 + 0j])
        psi_t1 = jnp.array([0.0 + 1.0j, 0.0 + 0j])  # Phase rotation

        violation = probability_conservation(psi_t0, psi_t1)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_rotation_conserves_probability(self):
        """Unitary rotation should conserve probability."""
        # Initial state
        psi_t0 = jnp.array([1.0, 0.0]) / jnp.sqrt(2)
        psi_t0 = jnp.concatenate([psi_t0, jnp.zeros(2)])

        # After rotation (still normalized)
        psi_t1 = jnp.array([0.0, 1.0]) / jnp.sqrt(2)
        psi_t1 = jnp.concatenate([psi_t1, jnp.zeros(2)])

        violation = probability_conservation(psi_t0, psi_t1)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_tolerance_parameter(self):
        """Tolerance should filter small violations."""
        psi_t0 = jnp.array([1.0, 0.0])
        psi_t1 = jnp.array([1.00001, 0.0])  # Tiny change

        # Tight tolerance detects
        violation_tight = probability_conservation(psi_t0, psi_t1, tolerance=1e-8)
        assert violation_tight > 0.0

        # Loose tolerance ignores
        violation_loose = probability_conservation(psi_t0, psi_t1, tolerance=1e-3)
        assert jnp.isclose(violation_loose, 0.0, atol=1e-10)

    def test_jit_compatibility(self):
        """Should be JIT-compatible."""

        @jax.jit
        def jitted_check(psi0, psi1):
            return probability_conservation(psi0, psi1)

        psi0 = jnp.array([1.0, 0.0])
        psi1 = jnp.array([0.0, 1.0])
        result = jitted_check(psi0, psi1)
        assert jnp.isfinite(result)


class TestJAXCompatibility:
    """Test JAX transformations compatibility (JIT, vmap, grad)."""

    def test_all_functions_jittable(self):
        """All quantum constraint functions should be JIT-compatible."""

        # Normalization
        @jax.jit
        def check_norm(psi, dx):
            return wavefunction_normalization(psi, dx)

        # Positivity
        @jax.jit
        def check_pos(rho):
            return density_positivity_violation(rho)

        # Hermiticity
        @jax.jit
        def check_herm(H):
            return hermiticity_violation(H)

        # Probability conservation
        @jax.jit
        def check_prob(psi0, psi1):
            return probability_conservation(psi0, psi1)

        # Run all
        psi = jnp.ones(10)
        dx = 0.1
        rho = jnp.ones(10)
        H = jnp.eye(2)

        assert jnp.isfinite(check_norm(psi, dx))
        assert jnp.isfinite(check_pos(rho))
        assert jnp.isfinite(check_herm(H))
        assert jnp.isfinite(check_prob(psi, psi))

    def test_gradient_compatibility(self):
        """Should be differentiable for optimization."""

        def loss_with_constraint(psi, dx):
            # Some loss plus normalization penalty
            data_loss = jnp.sum(psi**2)
            norm_penalty = wavefunction_normalization(psi, dx)
            return data_loss + norm_penalty

        psi = jnp.ones(10)
        dx = 0.1

        grad_fn = jax.grad(loss_with_constraint)
        grads = grad_fn(psi, dx)

        assert grads.shape == psi.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_vmap_all_functions(self):
        """All functions should work with vmap for batching."""
        batch_size = 5

        # Test normalization with batched psi
        psi_batch = jnp.ones((batch_size, 10))
        dx = 0.1
        norm_violations = jax.vmap(lambda p: wavefunction_normalization(p, dx))(
            psi_batch
        )
        assert norm_violations.shape == (batch_size,)

        # Test positivity with batched rho
        rho_batch = jnp.ones((batch_size, 10))
        pos_violations = jax.vmap(density_positivity_violation)(rho_batch)
        assert pos_violations.shape == (batch_size,)

        # Test hermiticity with batched matrices
        H_batch = jnp.stack([jnp.eye(3) for _ in range(batch_size)])
        herm_violations = jax.vmap(hermiticity_violation)(H_batch)
        assert herm_violations.shape == (batch_size,)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element_wavefunction(self):
        """Single element wavefunction should work."""
        psi = jnp.array([1.0])
        dx = 1.0
        violation = wavefunction_normalization(psi, dx)
        assert jnp.isclose(violation, 0.0, atol=1e-6)

    def test_single_element_density(self):
        """Single element density should work."""
        rho = jnp.array([1.0])
        violation = density_positivity_violation(rho)
        assert jnp.isclose(violation, 0.0, atol=1e-10)

    def test_1x1_matrix_hermiticity(self):
        """1x1 matrix should be Hermitian if real."""
        H = jnp.array([[1.0]])
        violation = hermiticity_violation(H)
        assert jnp.isclose(violation, 0.0, atol=1e-10)

    def test_large_wavefunction(self):
        """Should handle large wavefunctions efficiently."""
        psi = jnp.ones(10000)
        dx = 0.0001
        psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx)

        violation = wavefunction_normalization(psi, dx)
        assert jnp.isclose(violation, 0.0, atol=1e-5)

    def test_negative_tolerance_raises_or_clips(self):
        """Negative tolerance should be handled (raise or clip to 0)."""
        psi = jnp.ones(10)
        dx = 0.1

        # Should either raise ValueError or clip to 0
        # Implementation choice - test that it doesn't crash
        try:
            result = wavefunction_normalization(psi, dx, tolerance=-1e-6)
            assert jnp.isfinite(result)
        except ValueError:
            pass  # Acceptable to raise error
