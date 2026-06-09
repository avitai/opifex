r"""Cartesian-to-spherical atomic-orbital transform for ``l >= 2`` shells.

The McMurchie-Davidson integral engine in :mod:`opifex.core.quantum._flat_mmd`
works entirely in the Cartesian-Gaussian basis: an ``l``-shell contributes
``(l+1)(l+2)/2`` Cartesian components (1 for s, 3 for p, **6** for d, ...).
Standard quantum chemistry instead uses the *spherical* (real solid harmonic)
basis, where a ``d``-shell has only 5 components, a ``f``-shell 7, etc. The two
agree for ``l <= 1`` (s and p have equal Cartesian/spherical counts), so STO-3G
never needed a transform; def2-SVP introduces polarisation ``d``-shells and does.

This module builds the per-``l`` Cartesian->spherical matrices and applies them
block-diagonally to a Cartesian AO matrix (e.g. an overlap/kinetic matrix) or to
the leading axes of higher-rank AO tensors.

Normalisation convention
-------------------------
The transform is defined for AOs in the **opifex Cartesian convention** produced
by :func:`opifex.core.quantum.basis._build_shell_coefficients`: the contraction
coefficients are renormalised so the *axis-aligned* component ``(l, 0, 0)`` (e.g.
``d_{xx}``) has unit self-overlap, and every off-axis component
(``d_{xy}, d_{xz}, d_{yz}``) shares those coefficients and so has self-overlap
``(2l_x-1)!!(2l_y-1)!!(2l_z-1)!! / (2l-1)!!`` of it (``1/3`` for the ``d``
off-diagonals). With this convention the resulting spherical AOs come out
unit-normalised (no extra rescaling), and the per-``l`` matrix is the textbook
real-solid-harmonic transform (Schlegel & Frisch, *Int. J. Quantum Chem.* **54**,
83 (1995), Table 1) with each Cartesian row pre-scaled by
:math:`s_i = \sqrt{(2l-1)!! / ((2l_x-1)!!(2l_y-1)!!(2l_z-1)!!)}` to undo the
opifex off-axis under-normalisation (``s = 1`` on the axis-aligned components,
``\sqrt{3}`` on the ``d`` off-diagonals). For ``d`` the spherical functions are

.. math::
    d_{z^2}   &= z^2 - \tfrac{1}{2}(x^2 + y^2), &
    d_{x^2-y^2} &= \tfrac{\sqrt{3}}{2}(x^2 - y^2),

with ``d_{xy}, d_{yz}, d_{xz}`` carried through (their rows scaled by
:math:`\sqrt{3}`).

Relationship to ``pyscf.gto.cart2sph``
--------------------------------------
PySCF's :func:`pyscf.gto.cart2sph` returns the same transform for its *internal*
Cartesian normalisation, which differs from the opifex convention by an overall
spherical-harmonic constant ``N_l`` (the opifex per-row scaling ``s_i`` is
exactly what relates the two Cartesian conventions). The exact relationship,
validated for ``l = 0, 1, 2`` in :mod:`tests.core.quantum.test_spherical`, is

.. math::
    \mathrm{cart2sph}(l)_{ij}
        = \frac{T_l[i, j]\, N_l[j]}{s_l[i]^2},\qquad
    s_l[i] = \sqrt{\frac{(2l-1)!!}{(2l_x^{(i)}-1)!!(2l_y^{(i)}-1)!!(2l_z^{(i)}-1)!!}},

where :math:`T_l` is the opifex transform returned here (already containing one
factor of :math:`s_l[i]` per row) and :math:`N_l[j]` is the PySCF
spherical-harmonic column constant. Equivalently
:math:`\mathrm{cart2sph}(l) = \mathrm{diag}(1/s_l^2)\,T_l\,\mathrm{diag}(N_l)`.

References
----------
* H. B. Schlegel, M. J. Frisch, *Int. J. Quantum Chem.* **54**, 83 (1995)
  (transformation between Cartesian and pure spherical harmonic Gaussians).
* PySCF, ``pyscf.gto.cart2sph`` (oracle for the transform and AO ordering).
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float  # noqa: TC002


logger = logging.getLogger(__name__)

# Square root of 3, reused in the d-shell real-solid-harmonic coefficients.
_SQRT3 = float(np.sqrt(3.0))

# ---------------------------------------------------------------------------
# Per-l Cartesian->spherical transform matrices (opifex Cartesian normalisation
# convention). Each matrix has shape ``(n_cartesian, n_spherical)`` and maps a
# Cartesian AO row vector ``c`` to a spherical one via ``c @ T_l``. Columns are
# ordered to match PySCF's spherical ordering (``mol.spheric_labels()``):
#   l=2: d_xy, d_yz, d_z^2, d_xz, d_x2-y2.
# Cartesian rows follow ``basis._CART_COMPONENTS[l]``:
#   l=2: xx, xy, xz, yy, yz, zz.
# These coefficients are the Schlegel-Frisch real solid harmonics, each row
# pre-scaled by ``s_i`` (sqrt 3 on the d off-diagonals, 1 on the axis-aligned
# rows) to undo the opifex off-axis Cartesian under-normalisation. They are
# validated against ``pyscf.gto.cart2sph(l)`` and against PySCF spherical
# overlap/kinetic integrals to machine precision in the test-suite.
# ---------------------------------------------------------------------------
_CART2SPH: dict[int, np.ndarray] = {
    0: np.array([[1.0]], dtype=np.float64),
    1: np.eye(3, dtype=np.float64),
    2: np.array(
        [
            #   xy     yz     z^2           xz     x2-y2
            [0.0, 0.0, -0.5, 0.0, _SQRT3 / 2.0],  # xx  (s = 1)
            [_SQRT3, 0.0, 0.0, 0.0, 0.0],  # xy  (s = sqrt 3)
            [0.0, 0.0, 0.0, _SQRT3, 0.0],  # xz  (s = sqrt 3)
            [0.0, 0.0, -0.5, 0.0, -_SQRT3 / 2.0],  # yy  (s = 1)
            [0.0, _SQRT3, 0.0, 0.0, 0.0],  # yz  (s = sqrt 3)
            [0.0, 0.0, 1.0, 0.0, 0.0],  # zz  (s = 1)
        ],
        dtype=np.float64,
    ),
}

# Highest angular momentum with a tabulated transform.
_MAX_SUPPORTED_L: int = max(_CART2SPH)


def cart_count(angular_momentum: int) -> int:
    """Number of Cartesian components of an ``l``-shell: ``(l+1)(l+2)/2``."""
    return (angular_momentum + 1) * (angular_momentum + 2) // 2


def spherical_count(angular_momentum: int) -> int:
    """Number of spherical components of an ``l``-shell: ``2l + 1``."""
    return 2 * angular_momentum + 1


def cart_to_spherical_matrix(angular_momentum: int) -> Float[Array, "n_cart n_sph"]:
    r"""Return the Cartesian->spherical transform matrix for one ``l``-shell.

    The matrix :math:`T_l` has shape ``(n_cart, n_sph)`` with ``n_cart =
    (l+1)(l+2)/2`` and ``n_sph = 2l+1`` and maps a Cartesian AO (row) coefficient
    vector ``c`` to its spherical counterpart via ``c @ T_l``. Rows are ordered
    as :data:`opifex.core.quantum.basis._CART_COMPONENTS` and columns as PySCF's
    spherical ordering (see module docstring). The coefficients are the
    Schlegel-Frisch real solid harmonics, row-scaled for the opifex Cartesian
    normalisation convention (see module docstring).

    Args:
        angular_momentum: The shell angular momentum ``l`` (0 = s, 1 = p, 2 = d).

    Returns:
        The transform matrix as a JAX array [Shape: (n_cart, n_sph)].

    Raises:
        ValueError: If ``angular_momentum`` is negative or exceeds the highest
            tabulated shell (currently ``l = 2``).
    """
    if angular_momentum < 0:
        raise ValueError(f"angular_momentum must be non-negative, got {angular_momentum}")
    matrix = _CART2SPH.get(angular_momentum)
    if matrix is None:
        raise ValueError(
            f"No Cartesian->spherical transform for l={angular_momentum};"
            f" tabulated up to l={_MAX_SUPPORTED_L}"
        )
    return jnp.asarray(matrix)


def build_block_transform(
    angular_momenta: tuple[int, ...],
) -> Float[Array, "n_cart n_sph"]:
    r"""Assemble the full block-diagonal Cartesian->spherical transform.

    Concatenates the per-shell :func:`cart_to_spherical_matrix` blocks along the
    diagonal so the returned matrix maps a flat Cartesian AO axis to the flat
    spherical AO axis in shell-major order. The block structure (a NumPy
    scatter of static per-``l`` matrices) carries no tracers, so the result is a
    plain constant array safe to close over inside ``jit``.

    Args:
        angular_momenta: The angular momentum ``l`` of each shell, in AO order.

    Returns:
        The block-diagonal transform [Shape: (n_cart_total, n_sph_total)].
    """
    n_cart_total = sum(cart_count(l) for l in angular_momenta)
    n_sph_total = sum(spherical_count(l) for l in angular_momenta)
    transform = np.zeros((n_cart_total, n_sph_total), dtype=np.float64)
    cart_offset = 0
    sph_offset = 0
    for angular_momentum in angular_momenta:
        block = np.asarray(_CART2SPH[angular_momentum])
        n_cart, n_sph = block.shape
        transform[cart_offset : cart_offset + n_cart, sph_offset : sph_offset + n_sph] = block
        cart_offset += n_cart
        sph_offset += n_sph
    logger.debug(
        "Built Cartesian->spherical transform: %d cart -> %d sph over %d shells",
        n_cart_total,
        n_sph_total,
        len(angular_momenta),
    )
    return jnp.asarray(transform)


def apply_matrix(
    transform: Float[Array, "n_cart n_sph"],
    cartesian_matrix: Float[Array, "n_cart n_cart"],
) -> Float[Array, "n_sph n_sph"]:
    r"""Transform a Cartesian AO matrix to the spherical basis.

    Applies the congruence :math:`M_\text{sph} = T^\top M_\text{cart} T` to a
    two-index Cartesian AO operator (overlap, kinetic, nuclear, Fock, ...). Both
    indices are mapped, contracting the Cartesian axes against ``transform``.

    Args:
        transform: Block-diagonal transform [Shape: (n_cart, n_sph)] from
            :func:`build_block_transform`.
        cartesian_matrix: The Cartesian AO matrix [Shape: (n_cart, n_cart)].

    Returns:
        The spherical AO matrix [Shape: (n_sph, n_sph)].
    """
    return transform.T @ cartesian_matrix @ transform


def apply_left(
    transform: Float[Array, "n_cart n_sph"],
    cartesian: Float[Array, "n_cart ..."],
) -> Float[Array, "n_sph ..."]:
    r"""Transform only the leading (first) Cartesian AO axis to spherical.

    Contracts ``transform`` against axis 0 of ``cartesian`` (``einsum`` ``cs,c...
    -> s...``), leaving every trailing axis untouched. Useful for tensors where
    only one index is an AO axis (e.g. AO values on a grid, or one leg of an
    ERI tensor handled axis-by-axis).

    Args:
        transform: Block-diagonal transform [Shape: (n_cart, n_sph)].
        cartesian: A tensor whose first axis is the Cartesian AO axis
            [Shape: (n_cart, ...)].

    Returns:
        The tensor with its first axis mapped to spherical [Shape: (n_sph, ...)].
    """
    return jnp.tensordot(transform.T, cartesian, axes=([1], [0]))


__all__ = [
    "apply_left",
    "apply_matrix",
    "build_block_transform",
    "cart_count",
    "cart_to_spherical_matrix",
    "spherical_count",
]
