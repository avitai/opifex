"""Tests for the Cartesian-to-spherical AO transform.

The per-``l`` transform matrices are validated against PySCF's
``gto.cart2sph(l)`` (the oracle) and the assembled block transform is exercised
under ``jit`` / ``grad`` / ``vmap`` to guarantee the required JAX-transform
compatibility. PySCF is a test-time oracle only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.quantum._spherical import (
    apply_left,
    apply_matrix,
    build_block_transform,
    cart_count,
    cart_to_spherical_matrix,
    spherical_count,
)


def _double_factorial(value: int) -> float:
    """Double factorial ``value!!`` (with ``(-1)!! = 1``)."""
    result = 1.0
    while value > 1:
        result *= value
        value -= 2
    return result


def _opifex_cart_components(angular_momentum: int) -> list[tuple[int, int, int]]:
    """Cartesian ``(l_x, l_y, l_z)`` powers in opifex / PySCF descending order."""
    components: list[tuple[int, int, int]] = []
    for l_x in range(angular_momentum, -1, -1):
        for l_y in range(angular_momentum - l_x, -1, -1):
            components.append((l_x, l_y, angular_momentum - l_x - l_y))
    return components


def _self_overlap_scale(angular_momentum: int) -> np.ndarray:
    r"""Per-component ``s_l[i]`` linking the opifex transform to ``cart2sph``.

    ``s_l[i] = sqrt((2l-1)!! / ((2l_x-1)!!(2l_y-1)!!(2l_z-1)!!))`` is the ratio
    between the axis-aligned and component self-overlaps under unit primitive
    normalisation (see :mod:`opifex.core.quantum._spherical`).
    """
    total = _double_factorial(2 * angular_momentum - 1)
    return np.array(
        [
            np.sqrt(
                total
                / (
                    _double_factorial(2 * l_x - 1)
                    * _double_factorial(2 * l_y - 1)
                    * _double_factorial(2 * l_z - 1)
                )
            )
            for l_x, l_y, l_z in _opifex_cart_components(angular_momentum)
        ]
    )


@pytest.mark.parametrize(("angular_momentum", "n_cart", "n_sph"), [(0, 1, 1), (1, 3, 3), (2, 6, 5)])
def test_transform_shape(angular_momentum: int, n_cart: int, n_sph: int) -> None:
    """The transform has the Cartesian/spherical component counts of its shell."""
    matrix = cart_to_spherical_matrix(angular_momentum)
    assert matrix.shape == (n_cart, n_sph)
    assert cart_count(angular_momentum) == n_cart
    assert spherical_count(angular_momentum) == n_sph


@pytest.mark.parametrize("angular_momentum", [0, 1, 2])
def test_transform_matches_pyscf_cart2sph(angular_momentum: int) -> None:
    r"""Reconstruct ``pyscf.gto.cart2sph(l)`` exactly from the opifex transform.

    With ``cart2sph(l) = diag(1/s_l^2) @ T_l @ diag(N_l)`` (the opifex transform
    already carries one factor of ``s_l`` per row), the per-column constant
    ``N_l[j]`` is recovered from one non-zero entry and used to rebuild PySCF's
    matrix; agreement to ~1e-12 validates the transform for ``l = 0, 1, 2``.
    """
    gto = pytest.importorskip("pyscf.gto")
    pyscf_matrix = np.asarray(gto.cart2sph(angular_momentum))
    opifex_matrix = np.asarray(cart_to_spherical_matrix(angular_momentum))
    scale = _self_overlap_scale(angular_momentum)

    scaled = pyscf_matrix * (scale[:, None] ** 2)
    column_constants = np.empty(pyscf_matrix.shape[1])
    for column in range(pyscf_matrix.shape[1]):
        nonzero = np.abs(opifex_matrix[:, column]) > 1e-12
        ratios = scaled[nonzero, column] / opifex_matrix[nonzero, column]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
        column_constants[column] = ratios[0]

    reconstructed = (opifex_matrix * column_constants[None, :]) / (scale[:, None] ** 2)
    np.testing.assert_allclose(reconstructed, pyscf_matrix, atol=1e-12)


def test_d_shell_coefficients_are_schlegel_frisch() -> None:
    """The d-block carries the textbook ``-1/2`` / ``sqrt(3)/2`` axis-aligned coeffs.

    The axis-aligned rows (xx, yy, zz) match the textbook real-solid-harmonic
    coefficients; the off-axis rows (xy, xz, yz) are scaled by ``sqrt(3)`` to undo
    the opifex Cartesian under-normalisation (so e.g. ``d_xy`` maps with ``sqrt 3``).
    """
    matrix = np.asarray(cart_to_spherical_matrix(2))
    root3 = np.sqrt(3.0)
    # Column 2 is d_z^2 = zz - 1/2 (xx + yy) (axis-aligned rows, unscaled).
    np.testing.assert_allclose(matrix[:, 2], [-0.5, 0.0, 0.0, -0.5, 0.0, 1.0], atol=1e-12)
    # Column 4 is d_x2-y2 = sqrt(3)/2 (xx - yy) (axis-aligned rows, unscaled).
    np.testing.assert_allclose(
        matrix[:, 4], [root3 / 2.0, 0.0, 0.0, -root3 / 2.0, 0.0, 0.0], atol=1e-12
    )
    # Column 0 is d_xy: the off-axis xy row carries the sqrt(3) scaling.
    np.testing.assert_allclose(matrix[:, 0], [0.0, root3, 0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_build_block_transform_is_block_diagonal() -> None:
    """The block transform places each shell's matrix on the diagonal."""
    transform = np.asarray(build_block_transform((0, 1, 2)))
    assert transform.shape == (1 + 3 + 6, 1 + 3 + 5)
    # s-block top-left, off-diagonal coupling between shells is zero.
    assert transform[0, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(transform[0, 1:], 0.0)
    np.testing.assert_array_equal(transform[1:, 0], 0.0)


def test_unsupported_angular_momentum_raises() -> None:
    """Requesting an ``l`` beyond the tabulated transforms fails fast."""
    with pytest.raises(ValueError, match="No Cartesian->spherical transform for l=3"):
        cart_to_spherical_matrix(3)
    with pytest.raises(ValueError, match="non-negative"):
        cart_to_spherical_matrix(-1)


def test_apply_matrix_congruence() -> None:
    """``apply_matrix`` performs the two-sided congruence ``T^T M T``."""
    transform = build_block_transform((2,))
    cartesian = jnp.asarray(np.eye(6))
    spherical = apply_matrix(transform, cartesian)
    expected = transform.T @ cartesian @ transform
    np.testing.assert_allclose(np.asarray(spherical), np.asarray(expected), atol=1e-12)
    assert spherical.shape == (5, 5)


def test_apply_left_maps_only_first_axis() -> None:
    """``apply_left`` contracts axis 0 and leaves trailing axes intact."""
    transform = build_block_transform((2,))
    tensor = jnp.asarray(np.arange(6 * 4, dtype=np.float64).reshape(6, 4))
    result = apply_left(transform, tensor)
    expected = transform.T @ tensor
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-12)
    assert result.shape == (5, 4)


def test_transform_jit_grad_vmap() -> None:
    """The transform application is ``jit`` / ``grad`` / ``vmap`` compatible."""
    transform = build_block_transform((2,))

    def energy(cartesian: jax.Array) -> jax.Array:
        """Scalar reduction of the spherical-transformed matrix, for grad."""
        return jnp.sum(apply_matrix(transform, cartesian) ** 2)

    cartesian = jnp.asarray(np.eye(6))

    jitted = jax.jit(energy)
    np.testing.assert_allclose(float(jitted(cartesian)), float(energy(cartesian)), rtol=1e-12)

    gradient = jax.grad(energy)(cartesian)
    assert gradient.shape == (6, 6)
    assert jnp.all(jnp.isfinite(gradient))

    batch = jnp.broadcast_to(cartesian, (3, 6, 6))
    batched = jax.vmap(lambda matrix: apply_matrix(transform, matrix))(batch)
    assert batched.shape == (3, 5, 5)
