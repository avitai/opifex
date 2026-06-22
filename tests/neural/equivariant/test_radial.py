r"""Tests for radial bases and cutoff envelopes.

Behaviour is specified against the MACE radial module (Batatia et al. 2022,
arXiv:2206.07697; reference ``../mace/mace/modules/radial.py``):

* :class:`BesselBasis` -- equation (7): ``sqrt(2/r_c) * sin(n pi r / r_c) / r``.
* :class:`GaussianBasis` -- Gaussians centred on a ``[0, r_c]`` grid.
* :func:`polynomial_cutoff` -- equation (8) polynomial envelope (``-> 0`` at
  ``r_c`` together with its first ``p`` derivatives), masked to zero beyond
  ``r_c``.
* :func:`cosine_cutoff` -- the Behler cosine envelope.

The load-bearing checks are the closed-form values against the published
formulae, the smooth decay of the envelopes to zero at ``r_c``, and
``jit``/``grad``/``vmap`` compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.neural.equivariant import (
    BesselBasis,
    cosine_cutoff,
    GaussianBasis,
    PiecewiseLinearBasis,
    polynomial_cutoff,
)


class TestPiecewiseLinearBasis:
    """The piecewise-linear hat basis (DISCO filter basis, faithful to torch_harmonics)."""

    def test_shape_and_compact_support(self) -> None:
        """Returns ``(..., num_basis)``; vanishes beyond the cutoff."""
        basis = PiecewiseLinearBasis(num_basis=5, cutoff=1.0)
        values = basis(jnp.array([0.0, 0.3, 0.6, 1.5]))
        assert values.shape == (4, 5)
        assert jnp.all(values[-1] == 0.0)  # r = 1.5 > cutoff -> all zero
        assert jnp.all(values >= 0.0)

    def test_hats_are_bounded_tents_that_overlap(self) -> None:
        """Within the support the hats are bounded by 1 and overlap (tent functions)."""
        basis = PiecewiseLinearBasis(num_basis=6, cutoff=1.0)
        radii = jnp.linspace(0.0, 1.0, 50)
        values = basis(radii)
        assert jnp.all(values <= 1.0 + 1e-6)
        # Somewhere inside the support more than one hat is active (overlapping tents).
        assert jnp.max(jnp.sum(values > 0.0, axis=-1)) >= 2

    def test_is_transform_safe(self) -> None:
        """The basis is jit/grad/vmap clean."""
        basis = PiecewiseLinearBasis(num_basis=4, cutoff=2.0)
        radius = jnp.array([0.5, 1.0])
        assert jnp.all(jnp.isfinite(jax.jit(basis)(radius)))
        grad = jax.grad(lambda r: jnp.sum(basis(r)))(radius)
        assert jnp.all(jnp.isfinite(grad))
        assert jax.vmap(basis)(radius).shape == (2, 4)


class TestBesselBasis:
    def test_output_shape(self) -> None:
        basis = BesselBasis(num_basis=8, cutoff=5.0)
        distances = jnp.linspace(0.1, 4.9, 16)
        result = basis(distances)
        assert result.shape == (16, 8)

    def test_matches_mace_formula(self) -> None:
        r"""``b_n(r) = sqrt(2/r_c) sin(n pi r / r_c) / r`` (MACE eq. 7)."""
        cutoff, num_basis = 5.0, 4
        basis = BesselBasis(num_basis=num_basis, cutoff=cutoff)
        radius = jnp.asarray([1.3, 2.7])
        result = basis(radius)
        index = np.arange(1, num_basis + 1)
        expected = (
            np.sqrt(2.0 / cutoff)
            * np.sin(index[None, :] * np.pi * np.asarray(radius)[:, None] / cutoff)
            / np.asarray(radius)[:, None]
        )
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_is_nnx_module(self) -> None:
        assert isinstance(BesselBasis(num_basis=8, cutoff=5.0), nnx.Module)

    def test_jit_grad_vmap(self) -> None:
        basis = BesselBasis(num_basis=8, cutoff=5.0)
        radius = jnp.asarray([1.0, 2.0, 3.0])
        jitted = jax.jit(basis)
        assert jnp.allclose(jitted(radius), basis(radius), atol=1e-5)

        def loss(r: jax.Array) -> jax.Array:
            return jnp.sum(basis(r) ** 2)

        gradient = jax.grad(loss)(radius)
        assert gradient.shape == radius.shape
        assert jnp.all(jnp.isfinite(gradient))

        batched = jax.vmap(basis)(radius[:, None])
        assert batched.shape == (3, 1, 8)


class TestGaussianBasis:
    def test_output_shape(self) -> None:
        basis = GaussianBasis(num_basis=10, cutoff=6.0)
        distances = jnp.linspace(0.0, 6.0, 12)
        assert basis(distances).shape == (12, 10)

    def test_matches_mace_formula(self) -> None:
        r"""``g_n(r) = exp(coeff (r - mu_n)^2)``, ``coeff = -0.5 / spacing^2``."""
        cutoff, num_basis = 6.0, 5
        basis = GaussianBasis(num_basis=num_basis, cutoff=cutoff)
        radius = jnp.asarray([0.5, 3.0])
        centres = np.linspace(0.0, cutoff, num_basis)
        coeff = -0.5 / (cutoff / (num_basis - 1)) ** 2
        expected = np.exp(coeff * (np.asarray(radius)[:, None] - centres[None, :]) ** 2)
        assert jnp.allclose(basis(radius), expected, atol=1e-5)

    def test_peaks_at_centre(self) -> None:
        basis = GaussianBasis(num_basis=5, cutoff=6.0)
        at_first_centre = basis(jnp.asarray([0.0]))
        assert jnp.isclose(at_first_centre[0, 0], 1.0, atol=1e-6)

    def test_jit_grad_vmap(self) -> None:
        basis = GaussianBasis(num_basis=6, cutoff=4.0)
        radius = jnp.asarray([1.0, 2.0])
        assert jnp.allclose(jax.jit(basis)(radius), basis(radius), atol=1e-5)
        gradient = jax.grad(lambda r: jnp.sum(basis(r) ** 2))(radius)
        assert jnp.all(jnp.isfinite(gradient))
        assert jax.vmap(basis)(radius[:, None]).shape == (2, 1, 6)


class TestPolynomialCutoff:
    def test_is_one_at_origin(self) -> None:
        assert jnp.isclose(polynomial_cutoff(jnp.asarray(0.0), 5.0), 1.0, atol=1e-6)

    def test_is_zero_at_cutoff(self) -> None:
        assert jnp.isclose(polynomial_cutoff(jnp.asarray(5.0), 5.0), 0.0, atol=1e-6)

    def test_is_zero_beyond_cutoff(self) -> None:
        assert jnp.isclose(polynomial_cutoff(jnp.asarray(6.0), 5.0), 0.0, atol=1e-6)

    def test_matches_mace_formula(self) -> None:
        r"""MACE eq. 8 polynomial envelope for ``p = 6``."""
        cutoff, p = 5.0, 6
        radius = jnp.asarray([1.0, 2.5, 4.0])
        ratio = np.asarray(radius) / cutoff
        expected = (
            1.0
            - (p + 1.0) * (p + 2.0) / 2.0 * ratio**p
            + p * (p + 2.0) * ratio ** (p + 1)
            - p * (p + 1.0) / 2.0 * ratio ** (p + 2)
        )
        assert jnp.allclose(polynomial_cutoff(radius, cutoff, p=p), expected, atol=1e-5)

    def test_derivative_vanishes_at_cutoff(self) -> None:
        """Smoothness: ``d/dr`` of the envelope is zero at ``r_c``."""
        derivative = jax.grad(lambda r: polynomial_cutoff(r, 5.0))(jnp.asarray(5.0))
        assert jnp.isclose(derivative, 0.0, atol=1e-5)

    def test_jit_grad_vmap(self) -> None:
        radius = jnp.asarray([1.0, 2.0, 3.0])
        jitted = jax.jit(lambda r: polynomial_cutoff(r, 5.0))
        assert jnp.allclose(jitted(radius), polynomial_cutoff(radius, 5.0), atol=1e-5)
        gradient = jax.grad(lambda r: jnp.sum(polynomial_cutoff(r, 5.0)))(radius)
        assert jnp.all(jnp.isfinite(gradient))
        batched = jax.vmap(lambda r: polynomial_cutoff(r, 5.0))(radius)
        assert batched.shape == (3,)


class TestCosineCutoff:
    def test_is_one_at_origin(self) -> None:
        assert jnp.isclose(cosine_cutoff(jnp.asarray(0.0), 5.0), 1.0, atol=1e-6)

    def test_is_zero_at_cutoff(self) -> None:
        assert jnp.isclose(cosine_cutoff(jnp.asarray(5.0), 5.0), 0.0, atol=1e-6)

    def test_is_zero_beyond_cutoff(self) -> None:
        assert jnp.isclose(cosine_cutoff(jnp.asarray(7.0), 5.0), 0.0, atol=1e-6)

    def test_matches_formula(self) -> None:
        r"""``0.5 (cos(pi r / r_c) + 1)`` inside the cutoff."""
        cutoff = 5.0
        radius = jnp.asarray([1.0, 2.0, 3.0])
        expected = 0.5 * (np.cos(np.pi * np.asarray(radius) / cutoff) + 1.0)
        assert jnp.allclose(cosine_cutoff(radius, cutoff), expected, atol=1e-5)

    def test_jit_grad_vmap(self) -> None:
        radius = jnp.asarray([1.0, 2.0])
        assert jnp.allclose(
            jax.jit(lambda r: cosine_cutoff(r, 5.0))(radius),
            cosine_cutoff(radius, 5.0),
            atol=1e-5,
        )
        gradient = jax.grad(lambda r: jnp.sum(cosine_cutoff(r, 5.0)))(radius)
        assert jnp.all(jnp.isfinite(gradient))
        assert jax.vmap(lambda r: cosine_cutoff(r, 5.0))(radius).shape == (2,)
