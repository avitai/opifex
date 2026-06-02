"""Tests for real spherical harmonics ``Y_l(r)``.

Behaviour is specified against the e3nn / e3nn-jax conventions (Geiger & Smidt
2022, arXiv:2207.09453; reference
``../e3nn-jax/e3nn_jax/_src/spherical_harmonics/``).

The load-bearing check couples the spherical harmonics to the Wigner-D matrices:
they MUST share a convention, so equivariance ``Y_l(R r) = D^l(R) Y_l(r)`` holds
for random rotations.  Base cases (``Y_0`` constant, ``Y_1`` proportional to the
direction), normalization, and jit/grad/vmap compatibility are also checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import IrrepsArray
from opifex.neural.equivariant.spherical_harmonics import spherical_harmonics


_RNG = SO3Group()


def _random_rotation(seed: int) -> jax.Array:
    """Return a uniformly random SO(3) matrix for the given integer seed."""
    return _RNG.random_element(jax.random.PRNGKey(seed))


def _random_unit_vector(seed: int) -> jax.Array:
    """Return a random unit 3-vector for the given integer seed."""
    vector = jax.random.normal(jax.random.PRNGKey(seed), (3,))
    return vector / jnp.linalg.norm(vector)


class TestSphericalHarmonics:
    def test_returns_irreps_array_with_expected_layout(self) -> None:
        result = spherical_harmonics(2, _random_unit_vector(0))
        assert isinstance(result, IrrepsArray)
        assert repr(result.irreps) == "1x0e+1x1o+1x2e"
        assert result.array.shape == (9,)

    def test_y0_is_constant(self) -> None:
        value_a = spherical_harmonics(0, _random_unit_vector(1)).array
        value_b = spherical_harmonics(0, _random_unit_vector(2)).array
        assert value_a.shape == (1,)
        assert jnp.allclose(value_a, value_b)
        # Integral normalization: Y_0 = sqrt(1 / (4 pi)).
        assert jnp.allclose(value_a, np.sqrt(1.0 / (4.0 * np.pi)), atol=1e-5)

    def test_y1_is_proportional_to_direction(self) -> None:
        direction = _random_unit_vector(3)
        # spherical_harmonics(1, ...) is "1x0e+1x1o"; the l=1 chunk is the vector part.
        result = spherical_harmonics(1, direction)
        l1_block = result.chunks[1].reshape(-1)
        assert l1_block.shape == (3,)
        ratios = l1_block / direction
        # All components scaled by the same constant => Y_1 is parallel to r.
        assert jnp.allclose(ratios, ratios[0], atol=1e-5)

    @pytest.mark.parametrize("lmax", [1, 2, 3])
    def test_equivariance_with_wigner_d(self, lmax: int) -> None:
        """``Y_l(R r) = D^l(R) Y_l(r)`` couples SH to Wigner-D (shared convention)."""
        direction = _random_unit_vector(4)
        rotation = _random_rotation(5)
        rotated = spherical_harmonics(lmax, rotation @ direction)
        base = spherical_harmonics(lmax, direction)
        for (_, irrep), rotated_chunk, base_chunk in zip(
            rotated.irreps, rotated.chunks, base.chunks, strict=True
        ):
            transformed = jnp.einsum("ij,...j->...i", wigner_d(irrep.l, rotation), base_chunk)
            assert jnp.allclose(rotated_chunk, transformed, atol=1e-4)

    def test_normalize_projects_onto_sphere(self) -> None:
        scaled = spherical_harmonics(2, 5.0 * _random_unit_vector(6), normalize=True)
        unit = spherical_harmonics(2, _random_unit_vector(6), normalize=True)
        assert jnp.allclose(scaled.array, unit.array, atol=1e-5)

    def test_component_normalization_unit_norm_per_degree(self) -> None:
        """With ``component`` normalization the mean square per degree is 1."""
        direction = _random_unit_vector(7)
        result = spherical_harmonics(3, direction, normalization="component")
        for chunk in result.chunks:
            mean_square = jnp.mean(chunk**2)
            assert jnp.allclose(mean_square, 1.0, atol=1e-4)

    def test_batched_vectors(self) -> None:
        vectors = jnp.stack([_random_unit_vector(s) for s in (8, 9, 10)])
        result = spherical_harmonics(2, vectors)
        assert result.array.shape == (3, 9)

    def test_jit_compatibility(self) -> None:
        direction = _random_unit_vector(11)
        jitted = jax.jit(lambda v: spherical_harmonics(2, v).array)
        assert jnp.allclose(jitted(direction), spherical_harmonics(2, direction).array, atol=1e-5)

    def test_grad_compatibility(self) -> None:
        direction = _random_unit_vector(12)

        def loss(v: jax.Array) -> jax.Array:
            return jnp.sum(spherical_harmonics(3, v).array ** 2)

        gradient = jax.grad(loss)(direction)
        assert gradient.shape == (3,)
        assert jnp.all(jnp.isfinite(gradient))

    def test_vmap_compatibility(self) -> None:
        vectors = jnp.stack([_random_unit_vector(s) for s in (13, 14)])
        batched = jax.vmap(lambda v: spherical_harmonics(2, v).array)(vectors)
        assert batched.shape == (2, 9)
        for index in range(2):
            assert jnp.allclose(
                batched[index], spherical_harmonics(2, vectors[index]).array, atol=1e-5
            )
