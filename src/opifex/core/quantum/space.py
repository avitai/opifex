r"""Boundary-condition spaces: free and periodic ``(displacement, shift)`` pairs.

A native, ``jax``/``vmap``-clean reimplementation of the space abstraction in
``../jax-md/jax_md/space.py`` (Schoenholz & Cubuk 2020, "JAX-MD"). A *space* injects
boundary conditions into geometry calculations through two functions:

* ``displacement(ra, rb)`` -- the separation between two points (the
  minimum-image separation under periodic boundary conditions);
* ``shift(position, delta)`` -- advance a position by a displacement (wrapping it
  back into the cell when periodic).

Both implementations are exposed as frozen dataclasses satisfying the
:class:`opifex.core.quantum.protocols.Space` protocol, so models depend on the
abstraction rather than on ``jax-md`` directly (dependency inversion).

Cell convention: ``cell`` is a ``(3, 3)`` matrix whose **rows** are the lattice
vectors (the ASE / MACE / :class:`MolecularSystem` convention). A Cartesian
displacement ``dR`` has fractional coordinates ``f = (cell^T)^{-1} dR``. Rounding
``f`` to the nearest integer image and subtracting (``dR - cell^T round(f)``)
gives the minimum image for orthorhombic cells, but for strongly skewed
(triclinic) cells the nearest-fractional image is not always the shortest
(Allen & Tildesley, "Computer Simulation of Liquids", App. B). To stay correct
for *any* cell we additionally search the 27 neighbouring integer images of the
rounded displacement and keep the shortest -- a fixed-size enumeration that
remains ``jit``/``vmap`` clean.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float  # noqa: TC002


# The 27 integer image offsets within one cell of the nearest fractional image.
_IMAGE_OFFSETS: Float[Array, "27 3"] = jnp.asarray(
    list(itertools.product((-1, 0, 1), repeat=3)), dtype=float
)


@dataclass(frozen=True, slots=True)
class FreeSpace:
    """Free (open) boundary conditions: raw differences, additive shifts."""

    def displacement(self, ra: Float[Array, 3], rb: Float[Array, 3]) -> Float[Array, 3]:
        """Return the raw separation ``ra - rb`` (no wrapping)."""
        return ra - rb

    def shift(self, position: Float[Array, 3], delta: Float[Array, 3]) -> Float[Array, 3]:
        """Return ``position + delta`` (no wrapping)."""
        return position + delta


@dataclass(frozen=True, slots=True)
class PeriodicSpace:
    """Periodic boundary conditions on a (possibly triclinic) lattice.

    Attributes:
        cell: ``(3, 3)`` lattice matrix whose rows are the cell vectors.
    """

    cell: Float[Array, "3 3"]

    def displacement(self, ra: Float[Array, 3], rb: Float[Array, 3]) -> Float[Array, 3]:
        """Return the minimum-image separation between ``ra`` and ``rb``.

        The raw Cartesian separation is mapped to fractional coordinates and
        rounded to the nearest integer image. The 27 neighbouring images of that
        candidate are then enumerated and the shortest is returned, giving the
        true minimum image even for strongly skewed (triclinic) cells.
        """
        raw = ra - rb
        cell_transpose = self.cell.T
        fractional = jnp.linalg.solve(cell_transpose, raw)
        nearest = raw - cell_transpose @ jnp.round(fractional)
        # Shape (27, 3): each neighbouring image of the nearest candidate.
        candidates = nearest[None, :] - _IMAGE_OFFSETS @ cell_transpose.T
        squared_lengths = jnp.sum(candidates**2, axis=-1)
        return candidates[jnp.argmin(squared_lengths)]

    def shift(self, position: Float[Array, 3], delta: Float[Array, 3]) -> Float[Array, 3]:
        """Advance ``position`` by ``delta`` and wrap back into the cell.

        The updated position is expressed in fractional coordinates, taken modulo
        one, and mapped back to Cartesian space, keeping it inside the primitive
        cell.
        """
        cell_transpose = self.cell.T
        fractional = jnp.linalg.solve(cell_transpose, position + delta)
        return cell_transpose @ jnp.mod(fractional, 1.0)


def free() -> FreeSpace:
    """Return a :class:`FreeSpace` (open boundary conditions).

    Returns:
        A space whose ``displacement`` is the raw difference and whose ``shift``
        is plain addition.
    """
    return FreeSpace()


def periodic(cell: Float[Array, "3 3"]) -> PeriodicSpace:
    """Return a :class:`PeriodicSpace` for the given lattice.

    Args:
        cell: ``(3, 3)`` lattice matrix whose rows are the cell vectors.

    Returns:
        A space whose ``displacement`` applies the minimum-image convention and
        whose ``shift`` wraps positions back into the cell.

    Raises:
        ValueError: If ``cell`` is not a ``(3, 3)`` matrix.
    """
    if cell.shape != (3, 3):
        raise ValueError(f"Periodic cell must be a (3, 3) matrix, got shape {cell.shape}")
    return PeriodicSpace(cell=cell)


__all__ = ["FreeSpace", "PeriodicSpace", "free", "periodic"]
