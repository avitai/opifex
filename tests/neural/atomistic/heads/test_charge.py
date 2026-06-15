r"""Tests for the :class:`ChargeHead` total-charge-conserving partial-charge readout.

Load-bearing physics contracts:

* the head emits ``{"charges"}`` of shape ``(n_atoms,)``;
* the per-atom partial charges sum to the system's total charge ``Q`` -- the
  conservation constraint (MACE ``scatter_mean`` excess subtraction, models.py;
  PaiNN partial charges, Schuett et al. 2021);
* the conserved sum holds for a charged system (``Q != 0``);
* the partial charges are rotation- and translation-invariant scalars;
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
from opifex.neural.atomistic.heads.charge import ChargeHead


_FEATURE_DIM = 8
_GRAPH: tuple[Array, Array] = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))


def _system(charge: int = 0, positions: Array | None = None) -> MolecularSystem:
    """Return a 3-atom water-like system with the given net charge."""
    if positions is None:
        positions = jnp.asarray(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=jnp.float64
        )
    return MolecularSystem(
        atomic_numbers=jnp.asarray([8, 1, 1]), positions=positions, charge=charge
    )


def _embeddings(seed: int = 0) -> dict[str, Array]:
    """Return per-atom invariant ``node_features`` for a 3-atom system."""
    key = jax.random.PRNGKey(seed)
    return {"node_features": jax.random.normal(key, (3, _FEATURE_DIM), dtype=jnp.float64)}


class TestChargeHead:
    def test_implemented_properties(self) -> None:
        head = ChargeHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        assert head.implemented_properties == ("charges",)

    def test_charges_shape(self) -> None:
        head = ChargeHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        charges = head(_system(), _GRAPH, _embeddings())["charges"]
        assert charges.shape == (3,)

    def test_neutral_charge_conservation(self) -> None:
        with jax.enable_x64(True):
            head = ChargeHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            charges = head(_system(charge=0), _GRAPH, _embeddings())["charges"]
            assert jnp.allclose(jnp.sum(charges), 0.0, atol=1e-9)

    def test_charged_system_conservation(self) -> None:
        with jax.enable_x64(True):
            head = ChargeHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            charges = head(_system(charge=1), _GRAPH, _embeddings())["charges"]
            assert jnp.allclose(jnp.sum(charges), 1.0, atol=1e-9)

    def test_charges_rotation_invariant(self) -> None:
        with jax.enable_x64(True):
            head = ChargeHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            embeddings = _embeddings()
            base = head(_system(), _GRAPH, embeddings)["charges"]
            # Partial charges depend only on the (rotation-invariant) embeddings,
            # so they must be unchanged by any rigid motion of the geometry.
            rotated = jnp.asarray(
                [[0.0, 0.0, 0.0], [0.0, 0.96, 0.0], [0.0, -0.24, 0.93]], dtype=jnp.float64
            )
            moved = head(_system(positions=rotated), _GRAPH, embeddings)["charges"]
            assert jnp.allclose(base, moved, atol=1e-12)

    def test_registered_under_name(self) -> None:
        assert PropertyHeadRegistry().require("charges") is ChargeHead

    def test_jit_grad_vmap_smoke(self) -> None:
        head = ChargeHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(head)
        system = _system()
        node_features = _embeddings()["node_features"]

        def charges_for(features: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            return rebuilt(system, _GRAPH, {"node_features": features})["charges"]

        jitted = jax.jit(charges_for)
        charges = jitted(node_features)
        assert charges.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(charges)))

        gradient = jax.grad(lambda f: jnp.sum(charges_for(f) ** 2))(node_features)
        assert gradient.shape == node_features.shape
        assert bool(jnp.all(jnp.isfinite(gradient)))

        batch = jnp.stack([node_features, node_features + 0.1])
        batched = jax.vmap(charges_for)(batch)
        assert batched.shape == (2, 3)
        assert bool(jnp.all(jnp.isfinite(batched)))
