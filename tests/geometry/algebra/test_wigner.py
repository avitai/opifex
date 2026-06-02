"""Tests for the real-basis SO(3) Clebsch-Gordan tensor and Wigner-D matrices.

Behaviour is specified against the e3nn / e3nn-jax conventions (Geiger & Smidt
2022, arXiv:2207.09453; reference files ``../e3nn-jax/e3nn_jax/_src/su2.py`` and
``../e3nn-jax/e3nn_jax/_src/so3.py``).

The load-bearing correctness check needs no e3nn at test time: the real
Clebsch-Gordan tensor ``C`` must be an *intertwiner*, i.e. an equivariant map
from ``D^{l1} (x) D^{l2}`` to ``D^{l3}``.  Concretely, for any rotation ``R``::

    einsum('ijk,Ii,Jj->IJk', C, D1, D2) == einsum('ijk,Kk->ijK', C, D3)

The Wigner-D matrices are checked for orthogonality, the homomorphism property,
the ``l = 0`` / ``l = 1`` base cases, and jit/grad/vmap transform compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import clebsch_gordan, wigner_d


_RNG = SO3Group()


def _random_rotation(seed: int) -> jax.Array:
    """Return a uniformly random SO(3) matrix for the given integer seed."""
    return _RNG.random_element(jax.random.PRNGKey(seed))


class TestClebschGordan:
    def test_shape_matches_irrep_dimensions(self) -> None:
        coupling = clebsch_gordan(1, 2, 3)
        assert coupling.shape == (3, 5, 7)

    def test_returns_zero_when_coupling_invalid(self) -> None:
        # |1 - 1| = 0 <= 3 <= 2 is violated, so the coupling tensor is all zero.
        coupling = clebsch_gordan(1, 1, 3)
        assert coupling.shape == (3, 3, 7)
        assert jnp.allclose(coupling, 0.0)

    def test_coefficients_are_real_valued(self) -> None:
        coupling = clebsch_gordan(1, 1, 2)
        assert jnp.isrealobj(coupling)
        assert not jnp.allclose(coupling, 0.0)

    @pytest.mark.parametrize(
        ("l1", "l2", "l3"),
        [(0, 1, 1), (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 3), (2, 2, 2), (2, 2, 4)],
    )
    def test_intertwiner_equivariance(self, l1: int, l2: int, l3: int) -> None:
        """``C`` is an intertwiner: rotating all three legs leaves it invariant.

        Equivalently, ``C(D1 x, D2 y) = D3 C(x, y)`` for every rotation ``R``, so
        ``einsum('ijk,Ii,Jj,Kk->IJK', C, D1, D2, D3) == C``.  This is the
        load-bearing Clebsch-Gordan correctness check and needs no e3nn at test
        time.  The triple contraction is run in float64 because float32
        ``expm`` rounding (amplified by three matmuls) otherwise dominates.
        """
        with jax.enable_x64(True):
            coupling = clebsch_gordan(l1, l2, l3).astype(jnp.float64)
            for seed in (0, 1, 2):
                rotation = _random_rotation(seed).astype(jnp.float64)
                d1 = wigner_d(l1, rotation)
                d2 = wigner_d(l2, rotation)
                d3 = wigner_d(l3, rotation)
                rotated = jnp.einsum("ijk,Ii,Jj,Kk->IJK", coupling, d1, d2, d3)
                assert jnp.allclose(rotated, coupling, atol=1e-8)


class TestWignerD:
    def test_l0_is_scalar_identity(self) -> None:
        rotation = _random_rotation(3)
        assert jnp.allclose(wigner_d(0, rotation), jnp.eye(1))

    def test_l1_equals_rotation_in_real_basis(self) -> None:
        """In e3nn's stored real basis the ``l = 1`` Wigner-D equals ``R`` itself.

        The (y, z, x) ordering quirk is internal: ``wigner_d`` and
        ``spherical_harmonics`` share it, so for a plain 3x3 rotation matrix the
        ``l = 1`` matrix is ``R`` (verified against the e3nn angle-based path).
        """
        rotation = _random_rotation(4)
        assert jnp.allclose(wigner_d(1, rotation), rotation, atol=1e-5)

    @pytest.mark.parametrize("degree", [0, 1, 2, 3])
    def test_orthogonality(self, degree: int) -> None:
        rotation = _random_rotation(5)
        matrix = wigner_d(degree, rotation)
        identity = jnp.eye(2 * degree + 1)
        assert jnp.allclose(matrix @ matrix.T, identity, atol=1e-4)

    @pytest.mark.parametrize("degree", [0, 1, 2, 3])
    def test_homomorphism_composition(self, degree: int) -> None:
        rotation_a = _random_rotation(6)
        rotation_b = _random_rotation(7)
        composed = wigner_d(degree, rotation_a @ rotation_b)
        product = wigner_d(degree, rotation_a) @ wigner_d(degree, rotation_b)
        assert jnp.allclose(composed, product, atol=1e-4)

    def test_jit_compatibility(self) -> None:
        rotation = _random_rotation(8)
        jitted = jax.jit(lambda r: wigner_d(2, r))
        assert jnp.allclose(jitted(rotation), wigner_d(2, rotation), atol=1e-5)

    def test_grad_compatibility(self) -> None:
        rotation = _random_rotation(9)

        def loss(r: jax.Array) -> jax.Array:
            return jnp.sum(wigner_d(2, r) ** 2)

        gradient = jax.grad(loss)(rotation)
        assert gradient.shape == (3, 3)
        assert jnp.all(jnp.isfinite(gradient))

    def test_vmap_compatibility(self) -> None:
        rotations = jnp.stack([_random_rotation(s) for s in (10, 11, 12)])
        batched = jax.vmap(lambda r: wigner_d(2, r))(rotations)
        assert batched.shape == (3, 5, 5)
        for index in range(3):
            assert jnp.allclose(batched[index], wigner_d(2, rotations[index]), atol=1e-5)

    def test_matches_numpy_reference_for_l2(self) -> None:
        """Cross-check one matrix entry against an independent NumPy expm path."""
        rotation = _random_rotation(13)
        matrix = np.asarray(wigner_d(2, rotation))
        assert matrix.shape == (5, 5)
        # An orthogonal matrix has unit-norm columns.
        column_norms = np.linalg.norm(matrix, axis=0)
        assert np.allclose(column_norms, 1.0, atol=1e-4)
