r"""Tests for the :class:`DirectForcesHead` equivariant direct-force readout.

Load-bearing physics contracts (the direct-force strategy of Orb / fairchem,
Neumann et al. 2024):

* the head emits ``{"forces"}`` of shape ``(n_atoms, 3)``;
* forces are a **direct** ``l = 1`` readout of per-atom equivariant vector
  features -- the head never differentiates an energy (no ``jax.grad`` on an
  energy closure, no :data:`ENERGY_FN_KEY` dependency);
* the readout is **exactly rotationally equivariant**: rotating the geometry by
  ``R`` rotates every per-atom force by ``R`` (forces(R.r) == forces(r).R^T),
  because forces are an invariant-scalar gate times the equivariant vector
  channel;
* the readout is **translation invariant** (vector features are built from
  relative geometry, so a rigid shift leaves the injected vectors -- and hence
  the forces -- unchanged);
* the forward pass is ``jit``/``grad``/``vmap`` clean.

Tight numerical assertions run under ``jax.enable_x64(True)`` (the conftest
defaults ``jax_enable_x64`` to ``False``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.registry import PropertyHeadRegistry
from opifex.geometry.algebra import SO3Group
from opifex.neural.atomistic.heads.direct_forces import (
    DirectForcesHead,
    VECTOR_FEATURES_KEY,
)


_FEATURE_DIM = 8
_N_ATOMS = 3
_GRAPH: tuple[Array, Array] = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))
_VECTOR_KEY = jax.random.PRNGKey(7)
_FEATURE_KEY = jax.random.PRNGKey(3)
_ROTATION_KEY = jax.random.PRNGKey(11)


def _positions(dtype: jnp.dtype = jnp.float32) -> Array:
    """Return a non-symmetric 3-atom geometry."""
    return jnp.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=dtype)


def _system(positions: Array | None = None) -> MolecularSystem:
    """Return a 3-atom water-like system."""
    return MolecularSystem(
        atomic_numbers=jnp.asarray([8, 1, 1]),
        positions=_positions() if positions is None else positions,
    )


def _equivariant_vectors(positions: Array) -> Array:
    """Build per-atom equivariant vector features ``(n_atoms, 3, feature_dim)``.

    Mimics what an equivariant backbone (PaiNN) injects: each channel is a
    learned-but-fixed linear mix of the relative position vectors, so the block
    rotates with the geometry and is invariant to rigid translation.
    """
    centered = positions - jnp.mean(positions, axis=0, keepdims=True)
    mix = jax.random.normal(_VECTOR_KEY, (_N_ATOMS, _FEATURE_DIM), dtype=positions.dtype)
    return centered[:, :, None] * mix[:, None, :]


def _embeddings(positions: Array | None = None) -> dict[str, Array]:
    """Return invariant scalars plus the injected equivariant vector channel."""
    pos = _positions() if positions is None else positions
    node_features = jax.random.normal(_FEATURE_KEY, (_N_ATOMS, _FEATURE_DIM), dtype=pos.dtype)
    return {
        "node_features": node_features,
        VECTOR_FEATURES_KEY: _equivariant_vectors(pos),
    }


class TestDirectForcesHead:
    def test_implemented_properties(self) -> None:
        head = DirectForcesHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        assert head.implemented_properties == ("forces",)

    def test_forces_shape(self) -> None:
        head = DirectForcesHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        forces = head(_system(), _GRAPH, _embeddings())["forces"]
        assert forces.shape == (_N_ATOMS, 3)

    def test_registered_under_name(self) -> None:
        assert PropertyHeadRegistry().require("direct_forces") is DirectForcesHead

    def test_rotational_equivariance(self) -> None:
        with jax.enable_x64(True):
            head = DirectForcesHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            positions = _positions(jnp.float64)
            rotation = SO3Group().random_element(_ROTATION_KEY).astype(jnp.float64)
            rotated_positions = positions @ rotation.T

            base = head(_system(positions), _GRAPH, _embeddings(positions))["forces"]
            moved = head(_system(rotated_positions), _GRAPH, _embeddings(rotated_positions))[
                "forces"
            ]
            assert jnp.allclose(moved, base @ rotation.T, atol=1e-5)

    def test_translation_invariance(self) -> None:
        with jax.enable_x64(True):
            head = DirectForcesHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            positions = _positions(jnp.float64)
            shift = jnp.asarray([1.3, -2.1, 0.7], dtype=jnp.float64)
            shifted = positions + shift

            base = head(_system(positions), _GRAPH, _embeddings(positions))["forces"]
            moved = head(_system(shifted), _GRAPH, _embeddings(shifted))["forces"]
            assert jnp.allclose(moved, base, atol=1e-5)

    def test_does_not_consume_energy_closure(self) -> None:
        """A direct readout works with no position->energy closure injected.

        The conservative :class:`ForcesHead` raises ``KeyError`` without the
        ``_energy_fn`` closure; the direct head never references it, so a forward
        pass with only the vector channel present must succeed.
        """
        head = DirectForcesHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        embeddings = _embeddings()
        assert "_energy_fn" not in embeddings
        forces = head(_system(), _GRAPH, embeddings)["forces"]
        assert bool(jnp.all(jnp.isfinite(forces)))

    def test_jit_grad_vmap_smoke(self) -> None:
        head = DirectForcesHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(head)
        scalars = jax.random.normal(_FEATURE_KEY, (_N_ATOMS, _FEATURE_DIM))

        def forces_for(positions: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            system = MolecularSystem(atomic_numbers=jnp.asarray([8, 1, 1]), positions=positions)
            embeddings = {
                "node_features": scalars,
                VECTOR_FEATURES_KEY: _equivariant_vectors(positions),
            }
            return rebuilt(system, _GRAPH, embeddings)["forces"]

        jitted = jax.jit(forces_for)
        forces = jitted(_positions())
        assert forces.shape == (_N_ATOMS, 3)
        assert bool(jnp.all(jnp.isfinite(forces)))

        gradient = jax.grad(lambda p: jnp.sum(forces_for(p) ** 2))(_positions())
        assert gradient.shape == _positions().shape
        assert bool(jnp.all(jnp.isfinite(gradient)))

        batch = jnp.stack([_positions(), _positions() + 0.1])
        batched = jax.vmap(forces_for)(batch)
        assert batched.shape == (2, _N_ATOMS, 3)
        assert bool(jnp.all(jnp.isfinite(batched)))
