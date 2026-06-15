r"""Tests for the :class:`DipoleHead` molecular-dipole readout.

Load-bearing physics contracts:

* the head emits ``{"dipole"}`` of shape ``(3,)``;
* the molecular dipole is built from total-charge-conserving partial charges as
  :math:`\boldsymbol{\mu} = \sum_i q_i \mathbf{r}_i` (PaiNN, Schuett et al. 2021;
  MACE ``compute_total_charge_dipole_permuted``, utils.py);
* the dipole is **rotationally equivariant** for a neutral system:
  rotating the positions by ``R`` rotates the dipole by ``R``
  (origin-independent only when ``sum(q) == 0``);
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
from opifex.neural.atomistic.heads.dipole import DipoleHead


_FEATURE_DIM = 8
_GRAPH: tuple[Array, Array] = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))
_ROTATION_KEY = jax.random.PRNGKey(11)


def _positions() -> Array:
    """Return a non-symmetric 3-atom geometry (float64)."""
    return jnp.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=jnp.float64)


def _system(charge: int = 0, positions: Array | None = None) -> MolecularSystem:
    """Return a 3-atom system with the given net charge."""
    return MolecularSystem(
        atomic_numbers=jnp.asarray([8, 1, 1]),
        positions=_positions() if positions is None else positions,
        charge=charge,
    )


def _embeddings(seed: int = 0) -> dict[str, Array]:
    """Return per-atom invariant ``node_features`` for a 3-atom system."""
    key = jax.random.PRNGKey(seed)
    return {"node_features": jax.random.normal(key, (3, _FEATURE_DIM), dtype=jnp.float64)}


class TestDipoleHead:
    def test_implemented_properties(self) -> None:
        head = DipoleHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        assert head.implemented_properties == ("dipole",)

    def test_dipole_shape(self) -> None:
        head = DipoleHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        dipole = head(_system(), _GRAPH, _embeddings())["dipole"]
        assert dipole.shape == (3,)

    def test_dipole_rotational_equivariance(self) -> None:
        with jax.enable_x64(True):
            head = DipoleHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            embeddings = _embeddings()
            rotation = SO3Group().random_element(_ROTATION_KEY).astype(jnp.float64)
            base = head(_system(charge=0), _GRAPH, embeddings)["dipole"]
            rotated_positions = _positions() @ rotation.T
            moved = head(_system(charge=0, positions=rotated_positions), _GRAPH, embeddings)[
                "dipole"
            ]
            assert jnp.allclose(moved, base @ rotation.T, atol=1e-9)

    def test_registered_under_name(self) -> None:
        assert PropertyHeadRegistry().require("dipole") is DipoleHead

    def test_jit_grad_vmap_smoke(self) -> None:
        head = DipoleHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(head)
        node_features = _embeddings()["node_features"]

        def dipole_for(positions: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            system = MolecularSystem(atomic_numbers=jnp.asarray([8, 1, 1]), positions=positions)
            return rebuilt(system, _GRAPH, {"node_features": node_features})["dipole"]

        jitted = jax.jit(dipole_for)
        dipole = jitted(_positions())
        assert dipole.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(dipole)))

        gradient = jax.grad(lambda p: jnp.sum(dipole_for(p) ** 2))(_positions())
        assert gradient.shape == _positions().shape
        assert bool(jnp.all(jnp.isfinite(gradient)))

        batch = jnp.stack([_positions(), _positions() + 0.1])
        batched = jax.vmap(dipole_for)(batch)
        assert batched.shape == (2, 3)
        assert bool(jnp.all(jnp.isfinite(batched)))
