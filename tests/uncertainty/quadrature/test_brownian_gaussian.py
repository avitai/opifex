r"""Brownian × Gaussian (ambient) BQ kernel — Slice 24 (audit finding #7).

Task 6.3 design notes (``notes/04-task-6.3-expansion-design.md
:321-322``) require the ``QuadratureProductBrownian × {ambient,
Lebesgue}`` pair; opifex already ships the Lebesgue form, this slice
adds the **ambient = standard-normal-half-line** variant.

For Brownian motion ``k(x, x') = σ² min(x, x')`` integrated against
the truncated standard normal ``p(x) = N(0, 1) / Z`` on ``[0, ∞)``
(``Z = ½``):

.. math::

    qK(x') = 2\sigma^{2}\!\left[
        \frac{1}{\sqrt{2\pi}}\!\left(
            1 - \exp(-x'^{2}/2)
        \right)
        + x'\,\Phi^{c}(x')
    \right],

with ``Φ^c`` the standard-normal survival function. ``qKq`` is the
matched double-integral; we pin the symbolic answer against a fine
trapezoidal reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def test_brownian_gaussian_qk_matches_numerical_integration_at_known_point() -> None:
    """``qk_brownian_gaussian`` matches a fine 1-D quadrature reference."""
    from opifex.uncertainty.quadrature.kernels import qk_brownian_gaussian

    points = jnp.array([[0.5], [1.0], [2.0]])
    amplitude = jnp.asarray(1.0)
    qk = qk_brownian_gaussian(points=points, amplitude=amplitude)
    # Reference via trapezoidal quadrature on [0, 30] (effectively infinity for N(0,1)).
    x_grid = jnp.linspace(1e-6, 30.0, 200_001)
    half_normal_pdf = 2.0 * jax.scipy.stats.norm.pdf(x_grid)  # truncated to [0, ∞) with Z=1/2

    def numerical_qk(x_target: jax.Array) -> jax.Array:
        integrand = jnp.minimum(x_grid, x_target) * half_normal_pdf
        return jnp.trapezoid(integrand, x_grid)

    reference = jax.vmap(lambda r: numerical_qk(jnp.squeeze(r)))(points)
    assert jnp.allclose(qk, reference, atol=1e-3)


def test_brownian_gaussian_qkq_matches_numerical_integration() -> None:
    """``qkq_brownian_gaussian`` matches a 2-D fine-grid reference."""
    from opifex.uncertainty.quadrature.kernels import qkq_brownian_gaussian

    amplitude = jnp.asarray(1.0)
    qkq = qkq_brownian_gaussian(amplitude=amplitude)
    # Reference via 2-D trapezoidal quadrature.
    grid = jnp.linspace(1e-6, 20.0, 1001)
    half_normal_pdf = 2.0 * jax.scipy.stats.norm.pdf(grid)
    cross = jnp.minimum(grid[:, None], grid[None, :])
    integrand = cross * half_normal_pdf[:, None] * half_normal_pdf[None, :]
    reference = jnp.trapezoid(jnp.trapezoid(integrand, grid, axis=1), grid)
    assert float(jnp.abs(qkq - reference)) < 1e-2


def test_brownian_gaussian_qk_is_jit_compatible() -> None:
    """``qk_brownian_gaussian`` compiles under ``jax.jit``."""
    from opifex.uncertainty.quadrature.kernels import qk_brownian_gaussian

    @jax.jit
    def call(points: jax.Array) -> jax.Array:
        return qk_brownian_gaussian(points=points, amplitude=jnp.asarray(1.0))

    out = call(jnp.array([[0.3], [1.5]]))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))
