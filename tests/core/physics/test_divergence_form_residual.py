"""Tests for the variable-coefficient divergence-form PDE residual.

These tests target the ``homogenization`` residual ``-∇·(a(x)∇u) - f`` registered in
:mod:`opifex.core.physics.pde_registry`. The defining property of a *variable*
coefficient is that the flux divergence expands (product rule) to

    ∇·(a∇u) = ∇a·∇u + a Δu,

so a correct residual MUST retain the ``∇a·∇u`` term. Dropping it (the previous
constant-coefficient ``-a Δu - f`` stub) is wrong whenever ``a`` varies in space.

The suite uses the *method of manufactured solutions* (MMS): pick analytic ``a`` and
``u``, derive ``f = -∇·(a∇u)`` by hand, and assert the residual vanishes. A companion
test proves the implementation differs from the constant-coefficient approximation,
i.e. the ``∇a·∇u`` term is genuinely included.
"""
# ruff: noqa: F821
# F821 disabled: Ruff flags jaxtyping symbolic dimension literals ("batch", "dim") as
# undefined names. They are valid jaxtyping string-literal dimension annotations.

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from opifex.core.physics.autodiff_engine import AutoDiffEngine
from opifex.core.physics.pde_registry import PDEResidualRegistry


# =============================================================================
# Manufactured solution: a(x, y) = 1 + x,  u(x, y) = x^2 + y^2
#
#   ∇u   = [2x, 2y],        Δu = 4
#   ∇a   = [1, 0],          ∇a·∇u = 2x
#   ∇·(a∇u) = ∇a·∇u + aΔu = 2x + (1 + x)·4 = 6x + 4
#   f    = -∇·(a∇u) = -(6x + 4)
#   residual r = -∇·(a∇u) - f = 0   (exactly)
# =============================================================================


def _coeff(x: Float[Array, "batch dim"]) -> Float[Array, "batch"]:
    """Spatially varying coefficient ``a(x, y) = 1 + x``."""
    return 1.0 + x[:, 0]


def _solution(x: Float[Array, "batch dim"]) -> Float[Array, "batch"]:
    """Manufactured solution ``u(x, y) = x^2 + y^2``."""
    return jnp.sum(x**2, axis=-1)


def _manufactured_source(x: Float[Array, "batch dim"]) -> Float[Array, "batch"]:
    """Source ``f = -∇·(a∇u) = -(6x + 4)`` for the manufactured pair above."""
    return -(6.0 * x[:, 0] + 4.0)


@pytest.fixture
def points() -> Float[Array, "batch dim"]:
    """Interior collocation points with non-zero x (so ∇a·∇u ≠ 0)."""
    return jnp.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5], [0.9, 0.1]])


class TestManufacturedSolution:
    """The divergence-form residual must vanish for a manufactured solution."""

    def test_residual_is_zero_for_manufactured_solution(self, points):
        """r = -∇·(a∇u) - f ≈ 0 for a = 1 + x, u = x^2 + y^2, f = -(6x + 4)."""
        pde_fn = PDEResidualRegistry.get("homogenization")
        residual = pde_fn(
            _solution,
            points,
            AutoDiffEngine,
            coefficient_fn=_coeff,
            source_term=_manufactured_source(points),
        )
        assert residual.shape == (points.shape[0],)
        assert jnp.allclose(residual, 0.0, atol=1e-5)


class TestGradCoeffTermInclusion:
    """The implementation must include ∇a·∇u (i.e. differ from -aΔu - f)."""

    def test_differs_from_constant_coefficient_approximation(self, points):
        """Correct residual minus the -aΔu-f stub must equal -(∇a·∇u) = -2x."""
        pde_fn = PDEResidualRegistry.get("homogenization")
        source = _manufactured_source(points)

        correct = pde_fn(
            _solution, points, AutoDiffEngine, coefficient_fn=_coeff, source_term=source
        )

        # Reconstruct the (wrong) constant-coefficient approximation -aΔu - f.
        coeff = _coeff(points)
        laplacian = jnp.real(AutoDiffEngine.compute_laplacian(_solution, points))
        constant_coeff_approx = -coeff * laplacian - source

        # The two must NOT agree where a varies: difference = -(∇a·∇u) = -2x.
        difference = correct - constant_coeff_approx
        expected_grad_coeff_term = -(2.0 * points[:, 0])

        assert not jnp.allclose(correct, constant_coeff_approx, atol=1e-3)
        assert jnp.allclose(difference, expected_grad_coeff_term, atol=1e-5)

    def test_reduces_to_constant_coefficient_when_a_is_constant(self, points):
        """With constant a, ∇a = 0, so residual equals -aΔu - f exactly."""
        pde_fn = PDEResidualRegistry.get("homogenization")

        def constant_coeff(x: Float[Array, "batch dim"]) -> Float[Array, "batch"]:
            return jnp.full(x.shape[0], 2.0)

        source = jnp.full(points.shape[0], -1.0)
        residual = pde_fn(
            _solution, points, AutoDiffEngine, coefficient_fn=constant_coeff, source_term=source
        )

        laplacian = jnp.real(AutoDiffEngine.compute_laplacian(_solution, points))
        expected = -2.0 * laplacian - source
        assert jnp.allclose(residual, expected, atol=1e-5)


class TestTransformCompatibility:
    """jit / grad / vmap smoke tests on the divergence-form residual."""

    def test_jit(self, points):
        """Residual compiles under jit and stays finite."""
        pde_fn = PDEResidualRegistry.get("homogenization")

        @jax.jit
        def compute(x: Float[Array, "batch dim"]) -> Float[Array, "batch"]:
            return pde_fn(_solution, x, AutoDiffEngine, coefficient_fn=_coeff)

        residual = compute(points)
        assert jnp.all(jnp.isfinite(residual))

    def test_grad(self, points):
        """Residual is differentiable w.r.t. collocation points."""
        pde_fn = PDEResidualRegistry.get("homogenization")

        def scalar_loss(x: Float[Array, "batch dim"]) -> Float[Array, ""]:
            return jnp.sum(pde_fn(_solution, x, AutoDiffEngine, coefficient_fn=_coeff) ** 2)

        gradient = jax.grad(scalar_loss)(points)
        assert gradient.shape == points.shape
        assert jnp.all(jnp.isfinite(gradient))

    def test_vmap(self):
        """Residual vmaps over a batch of point sets."""
        pde_fn = PDEResidualRegistry.get("homogenization")
        batch_points = jnp.array(
            [
                [[0.3, 0.7], [0.8, 0.2]],
                [[0.5, 0.5], [0.9, 0.1]],
            ]
        )
        residuals = jax.vmap(lambda x: pde_fn(_solution, x, AutoDiffEngine, coefficient_fn=_coeff))(
            batch_points
        )
        assert residuals.shape == (2, 2)
        assert jnp.all(jnp.isfinite(residuals))
