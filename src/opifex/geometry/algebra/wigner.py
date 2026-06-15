r"""Real-basis SO(3) Clebsch-Gordan tensors and Wigner-D matrices.

A native, dependency-free port of the angular algebra used by ``e3nn-jax``
(Geiger & Smidt 2022, arXiv:2207.09453).  Two objects are exposed:

* :func:`clebsch_gordan` -- the real-basis SO(3) Clebsch-Gordan / Wigner-3j
  coupling tensor ``C[l1, l2, l3]`` of shape ``(2l1+1, 2l2+1, 2l3+1)``.  It is
  the unique (up to scale) *intertwiner* between ``D^{l1} (x) D^{l2}`` and
  ``D^{l3}``.
* :func:`wigner_d` -- the real Wigner-D matrix ``D^l(R)`` of degree ``l`` for a
  ``3x3`` rotation matrix ``R``, in the same real-spherical-harmonic basis as
  :func:`clebsch_gordan` and
  :func:`opifex.neural.equivariant.spherical_harmonics.spherical_harmonics`.

The SU(2) Clebsch-Gordan coefficients use the exact Racah formula with
``fractions.Fraction`` arithmetic, then are rotated into the real basis by the
``Q_l = (-i)^l q`` change-of-basis (Wikipedia, "Spherical harmonics, Real
form").  Everything static (the CG tables, the so(3) generators) is computed in
NumPy and cached with :func:`functools.cache`, then converted to ``jax`` arrays
on demand so the public functions are ``jit``/``grad``/``vmap`` clean.

References:
    * ``../e3nn-jax/e3nn_jax/_src/su2.py`` -- :func:`su2_clebsch_gordan`,
      :func:`_su2_cg` (Racah formula), :func:`su2_generators`.
    * ``../e3nn-jax/e3nn_jax/_src/so3.py`` -- :func:`clebsch_gordan`,
      :func:`change_basis_real_to_complex`, :func:`generators`.
    * ``../e3nn-jax/e3nn_jax/_src/irreps.py`` --
      :func:`_wigner_D_from_log_coordinates` (the matrix-exponential path,
      ``D = expm(sum_a w_a X_a)``).
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float  # noqa: TC002

from opifex.geometry.algebra._wigner_j import WIGNER_J


def _su2_clebsch_gordan_coefficient(
    j1: float, m1: float, j2: float, m2: float, j3: float, m3: float
) -> Fraction | float:
    r"""Single SU(2) Clebsch-Gordan coefficient via the exact Racah formula.

    Ported from ``../e3nn-jax/e3nn_jax/_src/su2.py::_su2_cg`` (itself copied from
    QuTiP's ``clebsch``).  Exact :class:`~fractions.Fraction` arithmetic is used
    throughout the sum so the resulting real-basis tensor is numerically clean.

    Args:
        j1: Angular momentum of the first irrep.
        m1: Magnetic quantum number of the first irrep.
        j2: Angular momentum of the second irrep.
        m2: Magnetic quantum number of the second irrep.
        j3: Total angular momentum.
        m3: Total magnetic quantum number.

    Returns:
        The Clebsch-Gordan coefficient ``<j1 m1; j2 m2 | j3 m3>``.
    """
    if m3 != m1 + m2:
        return 0

    def factorial(value: float) -> int:
        """Return the factorial of the nearest integer to ``value``."""
        return math.factorial(round(value))

    v_min = int(max(-j1 + j2 + m3, -j1 + m1, 0))
    v_max = int(min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3))

    prefactor = (
        (2.0 * j3 + 1.0)
        * Fraction(
            factorial(j3 + j1 - j2)
            * factorial(j3 - j1 + j2)
            * factorial(j1 + j2 - j3)
            * factorial(j3 + m3)
            * factorial(j3 - m3),
            factorial(j1 + j2 + j3 + 1)
            * factorial(j1 - m1)
            * factorial(j1 + m1)
            * factorial(j2 - m2)
            * factorial(j2 + m2),
        )
    ) ** 0.5

    summation = Fraction(0)
    for v in range(v_min, v_max + 1):
        summation += (-1.0) ** (v + j2 + m2) * Fraction(
            factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v),
            factorial(v)
            * factorial(j3 - j1 + j2 - v)
            * factorial(j3 + m3 - v)
            * factorial(v + j1 - j2 - m3),
        )
    return prefactor * summation


def _su2_clebsch_gordan(j1: float, j2: float, j3: float) -> np.ndarray:
    r"""SU(2) Clebsch-Gordan matrix of shape ``(2j1+1, 2j2+1, 2j3+1)``.

    Ported from ``../e3nn-jax/e3nn_jax/_src/su2.py::su2_clebsch_gordan``.

    Args:
        j1: Angular momentum of the first irrep.
        j2: Angular momentum of the second irrep.
        j3: Total angular momentum.

    Returns:
        The (real) SU(2) Clebsch-Gordan matrix.
    """
    matrix = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    matrix[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = float(
                        _su2_clebsch_gordan_coefficient(j1, m1, j2, m2, j3, m1 + m2)
                    )
    return matrix / math.sqrt(2 * j3 + 1)


def _su2_generators(j: float) -> np.ndarray:
    r"""Generators of the ``2j+1``-dimensional SU(2) representation.

    Ported from ``../e3nn-jax/e3nn_jax/_src/su2.py::su2_generators``.

    Args:
        j: Angular momentum of the irrep.

    Returns:
        Array of shape ``(3, 2j+1, 2j+1)`` of complex generators.
    """
    m = np.arange(-j, j)
    raising = np.diag(-np.sqrt(j * (j + 1) - m * (m + 1)), k=-1)

    m = np.arange(-j + 1, j + 1)
    lowering = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), k=1)

    m = np.arange(-j, j + 1)
    return np.stack(
        [
            0.5 * (raising + lowering),
            np.diag(1j * m),
            -0.5j * (raising - lowering),
        ],
        axis=0,
    )


def _change_basis_real_to_complex(degree: int) -> np.ndarray:
    r"""Change of basis ``Q_l`` from real to complex spherical harmonics.

    Ported from ``../e3nn-jax/e3nn_jax/_src/so3.py::change_basis_real_to_complex``
    (Wikipedia, "Spherical harmonics, Real form").  The extra ``(-i)^l`` factor
    makes the resulting real Clebsch-Gordan coefficients real-valued.

    Args:
        degree: The representation degree ``l``.

    Returns:
        Complex matrix of shape ``(2l+1, 2l+1)``.
    """
    matrix = np.zeros((2 * degree + 1, 2 * degree + 1), dtype=np.complex128)
    for m in range(-degree, 0):
        matrix[degree + m, degree + abs(m)] = 1 / np.sqrt(2)
        matrix[degree + m, degree - abs(m)] = -1j / np.sqrt(2)
    matrix[degree, degree] = 1
    for m in range(1, degree + 1):
        matrix[degree + m, degree + abs(m)] = (-1) ** m / np.sqrt(2)
        matrix[degree + m, degree - abs(m)] = 1j * (-1) ** m / np.sqrt(2)
    return (-1j) ** degree * matrix


@functools.cache
def clebsch_gordan_numpy(l1: int, l2: int, l3: int) -> np.ndarray:
    r"""Cached real-basis SO(3) Clebsch-Gordan tensor as a concrete NumPy array.

    Ported from ``../e3nn-jax/e3nn_jax/_src/so3.py::_clebsch_gordan``: rotate the
    SU(2) coupling into the real basis via the ``Q_l`` change of basis.

    Use this (rather than :func:`clebsch_gordan`) for *static* / compile-time
    computations -- e.g. spherical-harmonic normalization constants -- because it
    returns a concrete ``numpy.ndarray`` that is safe under ``np.asarray`` even
    when first evaluated inside a ``jax`` trace. The ``jax``-array wrapper
    :func:`clebsch_gordan` may stage as a (constant) tracer inside ``jit`` and so
    must not be passed to ``np.asarray``.

    Args:
        l1: Degree of the first irrep.
        l2: Degree of the second irrep.
        l3: Degree of the third irrep.

    Returns:
        Real tensor of shape ``(2l1+1, 2l2+1, 2l3+1)``.
    """
    coupling = _su2_clebsch_gordan(l1, l2, l3)
    q1 = _change_basis_real_to_complex(l1)
    q2 = _change_basis_real_to_complex(l2)
    q3 = _change_basis_real_to_complex(l3)
    real_coupling = np.einsum("ij,kl,mn,ikn->jlm", q1, q2, np.conj(q3.T), coupling)
    imaginary_norm = np.abs(np.imag(real_coupling)).max(initial=0.0)
    if imaginary_norm > 1e-5:
        raise ValueError(
            f"Real Clebsch-Gordan tensor for (l1={l1}, l2={l2}, l3={l3}) is not real "
            f"(max |Im| = {imaginary_norm:.3e}); the change of basis is inconsistent."
        )
    return np.ascontiguousarray(np.real(real_coupling))


@functools.cache
def _generators_numpy(degree: int) -> np.ndarray:
    r"""Cached real so(3) generators of degree ``l`` as a NumPy array.

    Ported from ``../e3nn-jax/e3nn_jax/_src/so3.py::generators``: conjugate the
    SU(2) generators by the real-to-complex change of basis.  For ``l = 1`` these
    are exactly the standard so(3) generators ``(L_x, L_y, L_z)``, so the Wigner-D
    of degree ``1`` of a rotation matrix is the rotation matrix itself.

    Args:
        degree: The representation degree ``l``.

    Returns:
        Real array of shape ``(3, 2l+1, 2l+1)``.
    """
    su2 = _su2_generators(degree)
    change = _change_basis_real_to_complex(degree)
    real_generators = np.conj(change.T) @ su2 @ change
    imaginary_norm = np.abs(np.imag(real_generators)).max(initial=0.0)
    if imaginary_norm > 1e-5:
        raise ValueError(
            f"Real so(3) generators of degree {degree} are not real "
            f"(max |Im| = {imaginary_norm:.3e}); the change of basis is inconsistent."
        )
    return np.ascontiguousarray(np.real(real_generators))


def clebsch_gordan(l1: int, l2: int, l3: int) -> Float[Array, "d1 d2 d3"]:
    r"""Real-basis SO(3) Clebsch-Gordan coupling tensor.

    The tensor ``C[l1, l2, l3]`` of shape ``(2l1+1, 2l2+1, 2l3+1)`` is the
    (normalized) intertwiner between ``D^{l1} (x) D^{l2}`` and ``D^{l3}``; it
    satisfies, for any rotation ``R`` (see :func:`wigner_d`)::

        einsum('ijk,Ii,Jj->IJk', C, D1, D2) == einsum('ijk,Kk->ijK', C, D3)

    It is identically zero unless the triangle rule ``|l1-l2| <= l3 <= l1+l2``
    holds.

    Ported from ``../e3nn-jax/e3nn_jax/_src/so3.py::clebsch_gordan``.

    Args:
        l1: Degree of the first irrep (non-negative integer).
        l2: Degree of the second irrep (non-negative integer).
        l3: Degree of the third irrep (non-negative integer).

    Returns:
        Real ``jax`` array of shape ``(2l1+1, 2l2+1, 2l3+1)``.
    """
    if min(l1, l2, l3) < 0:
        raise ValueError(f"Degrees must be non-negative, got (l1={l1}, l2={l2}, l3={l3})")
    return jnp.asarray(clebsch_gordan_numpy(l1, l2, l3))


def _log_coordinates_from_matrix(rotation: Float[Array, "3 3"]) -> Float[Array, 3]:
    r"""Standard so(3) log coordinates ``angle * axis`` of a rotation matrix.

    Differentiable everywhere except at ``angle = pi`` (the antipodal locus,
    measure zero).  The same axis-angle convention as
    :meth:`opifex.geometry.algebra.SO3Group.matrix_to_axis_angle`.

    Args:
        rotation: A ``3x3`` rotation matrix.

    Returns:
        The ``3``-vector ``angle * axis``.
    """
    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    # Clip the arccos argument strictly inside (-1, 1): at +-1 (angle 0 / pi) the
    # arccos gradient is infinite and would propagate NaN for axis-aligned
    # rotations (e.g. an edge along the eSCN quantisation axis). The value error
    # is O(1e-3 rad) only at the singular poles, which are measure zero.
    cos_angle = jnp.clip((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = jnp.arccos(cos_angle)
    axis = jnp.stack(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    # Double-where so the sqrt never sees 0 (angle -> 0 => axis -> 0): this guards
    # the *backward* pass, which a forward-only ``where(norm > 0, ...)`` does not.
    axis_sq = jnp.sum(axis**2)
    is_zero = axis_sq <= 1e-12
    axis_safe = jnp.where(is_zero, jnp.ones_like(axis_sq), axis_sq)
    reference = jnp.array([1.0, 0.0, 0.0], dtype=rotation.dtype)
    axis = jnp.where(is_zero, reference, axis / jnp.sqrt(axis_safe))
    return angle * axis


def wigner_d(degree: int, rotation: Float[Array, "3 3"]) -> Float[Array, "d d"]:
    r"""Real Wigner-D matrix ``D^l(R)`` of degree ``l`` for a rotation matrix.

    Computed via the matrix exponential of the real so(3) generators,
    ``D = expm(sum_a w_a X_a)`` where ``w`` are the standard log coordinates of
    ``R`` (``angle * axis``) and ``X_a = generators(l)`` (Ported from
    ``../e3nn-jax/e3nn_jax/_src/irreps.py::_wigner_D_from_log_coordinates``).
    This path is fully ``jit``/``grad``/``vmap`` compatible.

    For ``l = 0`` the result is ``[[1]]``; for ``l = 1`` it equals ``R`` in the
    e3nn real basis (the ``(y, z, x)`` ordering quirk is internal and shared with
    :func:`opifex.neural.equivariant.spherical_harmonics.spherical_harmonics`, so
    equivariance ``Y_l(R r) = D^l(R) Y_l(r)`` holds).

    Args:
        degree: The representation degree ``l`` (non-negative integer).
        rotation: A ``3x3`` rotation matrix.

    Returns:
        Real ``jax`` array of shape ``(2l+1, 2l+1)``.
    """
    if degree < 0:
        raise ValueError(f"Degree must be non-negative, got {degree}")
    if degree == 0:
        return jnp.ones((1, 1), dtype=rotation.dtype)
    generators = jnp.asarray(_generators_numpy(degree), dtype=rotation.dtype)
    log_coordinates = _log_coordinates_from_matrix(rotation)
    algebra_element = jnp.einsum("a,aij->ij", log_coordinates, generators)
    return jax.scipy.linalg.expm(algebra_element)


_POLE_EPSILON = 1e-12
"""Guard for the ``beta = 0 / pi`` pole where the ``(alpha, gamma)`` split degenerates."""


@jax.custom_jvp
def _safe_arccos(value: Float[Array, ""]) -> Float[Array, ""]:
    """``arccos`` with an exact forward value and a finite gradient at ``+-1``.

    The forward pass is the plain ``arccos`` (so ``beta = 0`` is recovered exactly
    on the quantisation pole, unlike a clipped argument); only the derivative
    ``-1 / sqrt(1 - x^2)`` is evaluated at a clamped argument so it stays finite at
    ``x = +-1``. This is fairchem's ``Safeacos``
    (``../fairchem/src/fairchem/core/models/uma/common/rotation.py``).
    """
    return jnp.arccos(value)


@_safe_arccos.defjvp
def _safe_arccos_jvp(  # pyright: ignore[reportUnusedFunction]  # registered as the JVP rule
    primals: tuple[Float[Array, ""]], tangents: tuple[Float[Array, ""]]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """JVP for :func:`_safe_arccos`: exact value, derivative at a clamped argument."""
    (value,) = primals
    (tangent,) = tangents
    clamped = jnp.clip(value, -1.0 + 1e-7, 1.0 - 1e-7)
    derivative = -1.0 / jnp.sqrt(1.0 - clamped * clamped)
    return jnp.arccos(value), derivative * tangent


def _y_rotation(degree: int, angle: Float[Array, ""]) -> Float[Array, "d d"]:
    r"""Real Wigner-D of a rotation about ``+y`` -- the banded ``Z_l`` matrix.

    In opifex's real-spherical-harmonic basis (shared with
    :func:`spherical_harmonics`) a rotation about the ``+y`` quantisation axis acts
    on each order ``m`` as a 2D rotation by ``m * angle``, giving the sparse
    structure ``cos(m theta)`` on the diagonal and ``sin(m theta)`` on the
    anti-diagonal (e3nn 0.4.0 ``o3/_wigner.py::_z_rot_mat``;
    ``../fairchem/src/fairchem/core/models/uma/common/rotation.py``). It equals
    :func:`wigner_d` of the corresponding ``3x3`` ``y``-rotation matrix exactly.

    Args:
        degree: The representation degree ``l`` (non-negative integer).
        angle: The rotation angle about ``+y`` (a scalar array).

    Returns:
        Real array of shape ``(2l+1, 2l+1)``.
    """
    dim = 2 * degree + 1
    indices = jnp.arange(dim)
    reversed_indices = indices[::-1]
    frequencies = jnp.arange(degree, -degree - 1, -1.0, dtype=angle.dtype)
    matrix = jnp.zeros((dim, dim), dtype=angle.dtype)
    matrix = matrix.at[indices, reversed_indices].set(jnp.sin(frequencies * angle))
    return matrix.at[indices, indices].set(jnp.cos(frequencies * angle))


def _matrix_to_euler(
    rotation: Float[Array, "3 3"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    r"""Grad-safe ZYZ-about-``y`` Euler angles of a rotation matrix.

    Returns ``(alpha, beta, gamma)`` with
    ``R = matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)``, the e3nn
    convention (``../e3nn-jax/e3nn_jax/_src/rotation.py::matrix_to_angles`` /
    ``xyz_to_angles``). ``beta = arccos(R[:, 1]_y)`` and ``alpha = atan2(x, z)``
    of the rotated ``+y`` axis; ``gamma`` is recovered from the residual rotation.

    The ``arccos`` argument is clipped strictly inside ``(-1, 1)`` and the two
    ``atan2`` calls use double-``where`` guards so both the forward value and the
    backward gradient stay finite at the ``beta = 0 / pi`` pole (an edge along the
    quantisation axis), mirroring fairchem's ``Safeacos`` / ``Safeatan2`` and the
    guard in :func:`_log_coordinates_from_matrix`.

    Args:
        rotation: A ``3x3`` rotation matrix.

    Returns:
        The Euler angles ``(alpha, beta, gamma)`` as scalar arrays.
    """
    axis = rotation @ jnp.array([0.0, 1.0, 0.0], dtype=rotation.dtype)
    x, y, z = axis[0], axis[1], axis[2]
    beta = _safe_arccos(y)
    # At the pole (x = z = 0) atan2 is ill-defined and its gradient blows up; pin
    # alpha = 0 there (gamma then absorbs the residual y-rotation, which is exact
    # because J @ Z_l(0) @ J = I) and feed atan2 a safe (0, 1) so grad is finite.
    on_pole = (x * x + z * z) <= _POLE_EPSILON
    safe_x = jnp.where(on_pole, 0.0, x)
    safe_z = jnp.where(on_pole, 1.0, z)
    alpha = jnp.arctan2(safe_x, safe_z)

    # Residual rotation R' = matrix_y(alpha)^T matrix_x(beta)^T R = matrix_y(gamma).
    residual = _matrix_x(beta).T @ _matrix_y(alpha).T @ rotation
    r02, r00 = residual[0, 2], residual[0, 0]
    on_gamma_pole = (r02 * r02 + r00 * r00) <= _POLE_EPSILON
    safe_r02 = jnp.where(on_gamma_pole, 0.0, r02)
    safe_r00 = jnp.where(on_gamma_pole, 1.0, r00)
    gamma = jnp.arctan2(safe_r02, safe_r00)
    return alpha, beta, gamma


def _matrix_y(angle: Float[Array, ""]) -> Float[Array, "3 3"]:
    """Return the ``3x3`` rotation matrix about the ``+y`` axis."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    zero, one = jnp.zeros_like(angle), jnp.ones_like(angle)
    return jnp.stack(
        [
            jnp.stack([c, zero, s]),
            jnp.stack([zero, one, zero]),
            jnp.stack([-s, zero, c]),
        ]
    )


def _matrix_x(angle: Float[Array, ""]) -> Float[Array, "3 3"]:
    """Return the ``3x3`` rotation matrix about the ``+x`` axis."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    zero, one = jnp.zeros_like(angle), jnp.ones_like(angle)
    return jnp.stack(
        [
            jnp.stack([one, zero, zero]),
            jnp.stack([zero, c, -s]),
            jnp.stack([zero, s, c]),
        ]
    )


def _wigner_d_from_euler(
    degree: int,
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
) -> Float[Array, "d d"]:
    r"""Assemble the real Wigner-D from Euler angles via the constant ``J_l``.

    ``D^l = Z_l(alpha) @ J_l @ Z_l(beta) @ J_l @ Z_l(gamma)`` (e3nn 0.4.0
    ``o3/_wigner.py``; fairchem ``wigner_D``). For degrees beyond the ported
    ``J_l`` table the result falls back to the exponential path via
    :func:`wigner_d` of the reconstructed rotation matrix.

    Args:
        degree: The representation degree ``l`` (non-negative integer).
        alpha: First Euler angle (rotation about ``+y``).
        beta: Second Euler angle (rotation about ``+x``).
        gamma: Third Euler angle (rotation about ``+y``).

    Returns:
        Real array of shape ``(2l+1, 2l+1)``.
    """
    if degree >= len(WIGNER_J):
        rotation = _matrix_y(alpha) @ _matrix_x(beta) @ _matrix_y(gamma)
        return wigner_d(degree, rotation)
    matrix_j = jnp.asarray(WIGNER_J[degree], dtype=alpha.dtype)
    z_alpha = _y_rotation(degree, alpha)
    z_beta = _y_rotation(degree, beta)
    z_gamma = _y_rotation(degree, gamma)
    return z_alpha @ matrix_j @ z_beta @ matrix_j @ z_gamma


def wigner_d_fast(degree: int, rotation: Float[Array, "3 3"]) -> Float[Array, "d d"]:
    r"""Real Wigner-D ``D^l(R)`` via the fast Euler-angle / ``J_l`` factorisation.

    Numerically identical to :func:`wigner_d` (the trusted matrix-exponential
    reference) but replaces the per-call ``jax.scipy.linalg.expm`` with the cheap
    ``Z_l(alpha) @ J_l @ Z_l(beta) @ J_l @ Z_l(gamma)`` product (e3nn 0.4.0
    ``o3/_wigner.py``; fairchem ``wigner_D``; QHNetV2 eSCN, arXiv:2506.09398).
    This is the rotation primitive of the SO(2)-frame edge convolution; the
    parity with :func:`wigner_d` is asserted in
    ``tests/geometry/algebra/test_wigner.py``.

    Fully ``jit``/``grad``/``vmap`` compatible, including at the ``beta = 0``
    quantisation pole (an edge on the ``+y`` axis), via the guards in
    :func:`_matrix_to_euler`.

    Args:
        degree: The representation degree ``l`` (non-negative integer).
        rotation: A ``3x3`` rotation matrix.

    Returns:
        Real ``jax`` array of shape ``(2l+1, 2l+1)``.
    """
    if degree < 0:
        raise ValueError(f"Degree must be non-negative, got {degree}")
    if degree == 0:
        return jnp.ones((1, 1), dtype=rotation.dtype)
    alpha, beta, gamma = _matrix_to_euler(rotation)
    return _wigner_d_from_euler(degree, alpha, beta, gamma)
