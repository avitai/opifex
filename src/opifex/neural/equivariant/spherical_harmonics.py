r"""Real spherical harmonics ``Y_l(r)`` for E(3)-equivariant networks.

A native, dependency-free port of the recursive spherical-harmonics algorithm of
``e3nn-jax`` (Geiger & Smidt 2022, arXiv:2207.09453; reference
``../e3nn-jax/e3nn_jax/_src/spherical_harmonics/recursive.py``).  Higher degrees
are built from lower ones by contracting two spherical-harmonic blocks with the
real Clebsch-Gordan tensor::

    Y_l = cste(l) / norm(l) * einsum('...i,...j,ijk->...k', Y_l1, Y_l2, C)

where ``l1 = l - 2**floor(log2(l-1))`` and ``l2 = l - l1``.  The per-degree
normalization constant ``norm(l)`` is computed numerically from the value of the
contraction at the "north pole" index (rather than symbolically via ``sympy`` as
in the reference), which is exact for the integer Clebsch-Gordan tables.

The output uses the same real-spherical-harmonic basis as
:func:`opifex.geometry.algebra.wigner.wigner_d`, so equivariance
``Y_l(R r) = D^l(R) Y_l(r)`` holds.  ``Y_0`` is constant and ``Y_1`` is
proportional to the (normalized) direction.  The implementation is
``jit``/``grad``/``vmap`` clean and handles batched inputs.

References:
    * ``../e3nn-jax/e3nn_jax/_src/spherical_harmonics/recursive.py`` -- the
      recurrence, the ``l1/l2`` split, and the ``integral`` / ``component``
      normalization constants.
    * ``../e3nn-jax/e3nn_jax/_src/spherical_harmonics/__init__.py`` -- the
      ``normalize`` (project onto the sphere) and ``normalization`` options.
"""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float  # noqa: TC002

from opifex.geometry.algebra.wigner import clebsch_gordan_numpy
from opifex.neural.equivariant.irreps import Irreps, IrrepsArray


_NORMALIZATIONS = ("integral", "component", "norm")


def _biggest_power_of_two(value: int) -> int:
    """Return the largest power of two less than or equal to ``value``."""
    return 2 ** (value.bit_length() - 1)


def _split_degree(degree: int) -> tuple[int, int]:
    """Return the ``(l1, l2)`` split used by the recursion for degree ``l``."""
    l2 = _biggest_power_of_two(degree - 1)
    return degree - l2, l2


@functools.cache
def _normalized_coupling(degree: int, normalization: str) -> np.ndarray:
    r"""Return the scaled Clebsch-Gordan tensor for the degree-``l`` recursion step.

    The scale folds in the per-degree normalization constant ``cste(l)`` and the
    numerically computed ``norm(l)`` so that the recursion produces correctly
    normalized real spherical harmonics.  Ported from
    ``../e3nn-jax/e3nn_jax/_src/spherical_harmonics/recursive.py`` (the ``norm``
    is the L2 norm of the contraction evaluated at the north-pole indices).

    Args:
        degree: The target degree ``l`` (``>= 2``).
        normalization: One of ``"integral"``, ``"component"``, ``"norm"``.

    Returns:
        Scaled real Clebsch-Gordan tensor of shape ``(2l1+1, 2l2+1, 2l+1)``.
    """
    l1, l2 = _split_degree(degree)
    coupling = clebsch_gordan_numpy(l1, l2, degree)
    north_pole = coupling[l1, l2, :]
    norm = math.sqrt(float(np.sum(north_pole**2)))
    four_pi = 4 * math.pi
    if normalization == "integral":
        constant = math.sqrt((2 * degree + 1) / four_pi) / (
            math.sqrt((2 * l1 + 1) / four_pi) * math.sqrt((2 * l2 + 1) / four_pi)
        )
    elif normalization == "component":
        constant = math.sqrt((2 * degree + 1) / ((2 * l1 + 1) * (2 * l2 + 1)))
    else:  # "norm"
        constant = 1.0
    return (constant / norm) * coupling


def _base_constants(normalization: str) -> tuple[float, float]:
    r"""Return the ``(l=0, l=1)`` prefactors for the chosen normalization.

    Ported from ``../e3nn-jax/e3nn_jax/_src/spherical_harmonics/recursive.py``.

    Args:
        normalization: One of ``"integral"``, ``"component"``, ``"norm"``.

    Returns:
        The pair ``(c0, c1)`` such that ``Y_0 = c0`` and ``Y_1 = c1 * r``.
    """
    if normalization == "integral":
        return math.sqrt(1.0 / (4.0 * math.pi)), math.sqrt(3.0 / (4.0 * math.pi))
    if normalization == "component":
        return 1.0, math.sqrt(3.0)
    return 1.0, 1.0  # "norm"


def _spherical_harmonics_blocks(
    lmax: int, vectors: Float[Array, "... 3"], normalization: str
) -> list[Float[Array, "... dim_l"]]:
    """Return the list of per-degree spherical-harmonic blocks for ``l = 0..lmax``."""
    constant_0, constant_1 = _base_constants(normalization)
    blocks: list[Float[Array, "... dim_l"]] = [
        constant_0 * jnp.ones_like(vectors[..., :1]),
    ]
    if lmax >= 1:
        blocks.append(constant_1 * vectors)
    for degree in range(2, lmax + 1):
        l1, l2 = _split_degree(degree)
        coupling = jnp.asarray(_normalized_coupling(degree, normalization), dtype=vectors.dtype)
        blocks.append(jnp.einsum("...i,...j,ijk->...k", blocks[l1], blocks[l2], coupling))
    return blocks


def spherical_harmonics(
    out: Irreps | int,
    vectors: Float[Array, "... 3"],
    *,
    normalize: bool = True,
    normalization: str = "integral",
) -> IrrepsArray:
    r"""Real spherical harmonics ``Y_l(r)`` for degrees ``l = 0..lmax``.

    The harmonics share the real basis of
    :func:`opifex.geometry.algebra.wigner.wigner_d`, so they are equivariant:
    ``Y_l(R r) = D^l(R) Y_l(r)``.

    Args:
        out: Either an integer ``lmax`` (producing irreps
            ``1x0e + 1x1o + ... + 1x{lmax}{parity}`` where each degree has the
            parity ``(-1)^l`` of a spherical harmonic) or an explicit
            :class:`~opifex.neural.equivariant.Irreps` layout.
        vectors: Cartesian coordinates of shape ``(..., 3)``.
        normalize: If ``True`` (default), the input is projected onto the unit
            sphere before evaluating the polynomials.
        normalization: One of ``"integral"`` (default), ``"component"`` or
            ``"norm"`` -- the convention for the per-degree scale.

    Returns:
        An :class:`~opifex.neural.equivariant.IrrepsArray` with the requested
        irreps and an array of shape ``(..., irreps.dim)``.

    Raises:
        ValueError: If ``normalization`` is unknown, ``vectors`` is not
            3-dimensional in its last axis, or an explicit ``out`` layout is not
            a valid spherical-harmonic layout (multiplicity one, parity
            ``(-1)^l``, ascending degrees).
    """
    if normalization not in _NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {_NORMALIZATIONS}, got {normalization!r}")
    if vectors.shape[-1] != 3:
        raise ValueError(f"vectors must have last dimension 3, got shape {vectors.shape}")

    if isinstance(out, int):
        if out < 0:
            raise ValueError(f"lmax must be non-negative, got {out}")
        terms = (f"1x{degree}{'e' if degree % 2 == 0 else 'o'}" for degree in range(out + 1))
        irreps = Irreps("+".join(terms))
    else:
        irreps = Irreps(out)
    degrees = [irrep.l for _, irrep in irreps]
    _validate_layout(irreps, degrees)

    if normalize:
        squared_norm = jnp.sum(vectors**2, axis=-1, keepdims=True)
        safe_squared_norm = jnp.where(squared_norm == 0.0, 1.0, squared_norm)
        vectors = vectors / jnp.sqrt(safe_squared_norm)

    blocks = _spherical_harmonics_blocks(max(degrees), vectors, normalization)
    array = jnp.concatenate([blocks[degree] for degree in degrees], axis=-1)
    return IrrepsArray(irreps, array)


def _validate_layout(irreps: Irreps, degrees: list[int]) -> None:
    """Raise if ``irreps`` is not a valid (ascending, parity ``(-1)^l``) SH layout."""
    for mul, irrep in irreps:
        if mul != 1:
            raise ValueError(
                f"spherical_harmonics requires multiplicity 1 per degree, got {irreps!r}"
            )
        if irrep.p != (-1) ** irrep.l:
            raise ValueError(
                f"spherical_harmonics requires parity (-1)^l per degree, got {irrep!r}"
            )
    if degrees != sorted(degrees):
        raise ValueError(f"spherical_harmonics requires ascending degrees, got {irreps!r}")
