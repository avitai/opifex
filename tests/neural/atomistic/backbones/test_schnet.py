r"""Tests for the :class:`SchNet` continuous-filter convolutional backbone.

Load-bearing physics contracts (shared via :mod:`._helpers`): energy invariance
under rotation/translation/permutation, force equivariance, force = -grad(E)
versus finite differences, learning a toy pairwise potential, and jit/grad/vmap
cleanliness. SchNet-specific checks cover the output shape, the shifted-softplus
activation and registry self-registration.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from tests.neural.atomistic.backbones import _helpers

from opifex.core.quantum.registry import BackboneRegistry
from opifex.neural.atomistic.backbones import SchNet, SchNetConfig
from opifex.neural.atomistic.backbones.schnet import shifted_softplus


_FEATURE_DIM = 16


def _make_schnet(rngs: nnx.Rngs) -> SchNet:
    config = SchNetConfig(
        feature_dim=_FEATURE_DIM, num_interactions=2, num_radial_basis=8, cutoff=5.0
    )
    return SchNet(config=config, rngs=rngs)


class TestSchNetContracts:
    def test_energy_invariant(self) -> None:
        _helpers.assert_energy_invariant(_make_schnet, feature_dim=_FEATURE_DIM)

    def test_force_equivariant(self) -> None:
        _helpers.assert_force_equivariant(_make_schnet, feature_dim=_FEATURE_DIM)

    def test_forces_match_finite_difference(self) -> None:
        _helpers.assert_forces_match_finite_difference(_make_schnet, feature_dim=_FEATURE_DIM)

    def test_learns_toy_potential(self) -> None:
        _helpers.assert_learns_toy_potential(_make_schnet, feature_dim=_FEATURE_DIM)

    def test_jit_grad_vmap_smoke(self) -> None:
        _helpers.assert_jit_grad_vmap_smoke(_make_schnet, feature_dim=_FEATURE_DIM)


class TestSchNetSpecifics:
    def test_node_feature_shape(self) -> None:
        backbone = _make_schnet(nnx.Rngs(0))
        system = _helpers.water()
        graph = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))
        features = backbone(system, graph)["node_features"]
        assert features.shape == (system.n_atoms, _FEATURE_DIM)

    def test_shifted_softplus_is_zero_at_origin(self) -> None:
        assert jnp.allclose(shifted_softplus(jnp.zeros(())), 0.0, atol=1e-7)

    def test_registered_under_name(self) -> None:
        assert BackboneRegistry().require("schnet") is SchNet
