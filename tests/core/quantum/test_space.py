r"""Tests for boundary-condition spaces (free / periodic).

The ``Space`` abstraction returns a ``(displacement_fn, shift_fn)`` pair that
injects boundary conditions into geometry calculations, mirroring
``../jax-md/jax_md/space.py`` (Schoenholz & Cubuk 2020, JAX-MD). The load-bearing
checks are: free displacement is the raw difference, periodic displacement obeys
the minimum-image convention for an arbitrary (possibly triclinic) cell, ``shift``
wraps positions back into the cell, and both are ``jit``/``vmap`` clean.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.core.quantum.protocols import Space
from opifex.core.quantum.space import free, periodic


class TestFree:
    def test_returns_space_protocol(self) -> None:
        """``free()`` returns an object satisfying the ``Space`` protocol."""
        space = free()
        assert isinstance(space, Space)

    def test_displacement_is_raw_difference(self) -> None:
        """Free displacement ``d(a, b) = a - b`` with no wrapping."""
        space = free()
        ra = jnp.asarray([1.0, 2.0, 3.0])
        rb = jnp.asarray([0.5, 0.0, -1.0])
        assert jnp.allclose(space.displacement(ra, rb), ra - rb)

    def test_shift_adds_displacement(self) -> None:
        """Free shift ``shift(R, dR) = R + dR``."""
        space = free()
        position = jnp.asarray([1.0, 1.0, 1.0])
        delta = jnp.asarray([0.5, -0.5, 2.0])
        assert jnp.allclose(space.shift(position, delta), position + delta)

    def test_displacement_jit_vmap(self) -> None:
        """Free displacement is ``jit``- and ``vmap``-clean over batched inputs."""
        space = free()
        ra = jnp.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        rb = jnp.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        batched = jax.jit(jax.vmap(space.displacement))(ra, rb)
        assert jnp.allclose(batched, ra - rb)


class TestPeriodic:
    def test_returns_space_protocol(self) -> None:
        """``periodic(cell)`` returns an object satisfying the ``Space`` protocol."""
        cell = jnp.eye(3) * 10.0
        space = periodic(cell)
        assert isinstance(space, Space)

    def test_minimum_image_orthorhombic(self) -> None:
        """A separation spanning more than half the box wraps to the short image."""
        cell = jnp.eye(3) * 10.0
        space = periodic(cell)
        ra = jnp.asarray([9.0, 0.0, 0.0])
        rb = jnp.asarray([0.0, 0.0, 0.0])
        # Raw difference is +9; minimum image is -1 (the periodic neighbour).
        assert jnp.allclose(space.displacement(ra, rb), jnp.asarray([-1.0, 0.0, 0.0]))

    def test_minimum_image_short_separation_unchanged(self) -> None:
        """A separation under half the box length is returned unchanged."""
        cell = jnp.eye(3) * 10.0
        space = periodic(cell)
        ra = jnp.asarray([3.0, 1.0, 0.0])
        rb = jnp.asarray([1.0, 0.0, 0.0])
        assert jnp.allclose(space.displacement(ra, rb), jnp.asarray([2.0, 1.0, 0.0]))

    def test_minimum_image_triclinic(self) -> None:
        """Minimum image is correct for a non-orthogonal (triclinic) cell."""
        cell = jnp.asarray([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        space = periodic(cell)
        ra = jnp.asarray([11.5, 9.5, 0.0])
        rb = jnp.asarray([0.0, 0.0, 0.0])
        displacement = space.displacement(ra, rb)
        # The minimum image is at most half a cell vector away, so its fractional
        # coordinates lie in [-0.5, 0.5].
        inverse = jnp.linalg.inv(cell.T)
        fractional = inverse @ displacement
        assert bool(jnp.all(jnp.abs(fractional) <= 0.5 + 1e-6))

    def test_minimum_image_is_shortest(self) -> None:
        """The minimum image is the shortest among all 27 nearest images."""
        cell = jnp.asarray([[10.0, 0.0, 0.0], [1.0, 9.0, 0.0], [0.0, 2.0, 8.0]])
        space = periodic(cell)
        ra = jnp.asarray([7.0, 6.0, 5.0])
        rb = jnp.asarray([0.0, 0.0, 0.0])
        displacement = space.displacement(ra, rb)
        raw = ra - rb
        candidates = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    shift = i * cell[0] + j * cell[1] + k * cell[2]
                    candidates.append(raw - shift)
        candidate_norms = jnp.asarray([float(jnp.linalg.norm(c)) for c in candidates])
        assert jnp.isclose(jnp.linalg.norm(displacement), jnp.min(candidate_norms), atol=1e-5)

    def test_shift_wraps_into_cell(self) -> None:
        """Shifting beyond the cell wraps the position back inside it."""
        cell = jnp.eye(3) * 10.0
        space = periodic(cell)
        position = jnp.asarray([9.0, 0.0, 0.0])
        shifted = space.shift(position, jnp.asarray([2.0, 0.0, 0.0]))
        # 9 + 2 = 11 wraps to 1 inside the [0, 10) box.
        assert jnp.allclose(shifted, jnp.asarray([1.0, 0.0, 0.0]))

    def test_displacement_jit_vmap(self) -> None:
        """Periodic displacement is ``jit``- and ``vmap``-clean over batches."""
        cell = jnp.eye(3) * 10.0
        space = periodic(cell)
        ra = jnp.asarray([[9.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        rb = jnp.zeros((2, 3))
        batched = jax.jit(jax.vmap(space.displacement))(ra, rb)
        assert jnp.allclose(batched[0], jnp.asarray([-1.0, 0.0, 0.0]))
        assert jnp.allclose(batched[1], jnp.asarray([3.0, 0.0, 0.0]))
