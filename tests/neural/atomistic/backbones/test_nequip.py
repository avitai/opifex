r"""Tests for the :class:`NequIP` E(3)-equivariant tensor-product backbone.

Load-bearing physics contracts (shared via :mod:`._helpers`): energy invariance
under rotation/translation/permutation, force equivariance, force = -grad(E)
versus finite differences, learning a toy pairwise potential, and jit/grad/vmap
cleanliness. NequIP-specific checks cover the scalar-readout shape, registry
self-registration and the documented MACE-correlation deferral guard.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from flax import nnx
from tests.neural.atomistic.backbones import _helpers

from opifex.core.quantum.registry import BackboneRegistry
from opifex.neural.atomistic.backbones import NequIP, NequIPConfig


_HIDDEN_IRREPS = "8x0e + 4x1o + 2x2e"
_NUM_SCALARS = 8


def _make_nequip(rngs: nnx.Rngs) -> NequIP:
    config = NequIPConfig(
        hidden_irreps=_HIDDEN_IRREPS,
        sh_lmax=2,
        num_interactions=2,
        num_radial_basis=8,
        cutoff=5.0,
    )
    return NequIP(config=config, rngs=rngs)


class TestNequIPContracts:
    def test_energy_invariant(self) -> None:
        _helpers.assert_energy_invariant(_make_nequip, feature_dim=_NUM_SCALARS)

    def test_force_equivariant(self) -> None:
        _helpers.assert_force_equivariant(_make_nequip, feature_dim=_NUM_SCALARS)

    def test_forces_match_finite_difference(self) -> None:
        _helpers.assert_forces_match_finite_difference(_make_nequip, feature_dim=_NUM_SCALARS)

    def test_learns_toy_potential(self) -> None:
        _helpers.assert_learns_toy_potential(_make_nequip, feature_dim=_NUM_SCALARS)

    def test_jit_grad_vmap_smoke(self) -> None:
        _helpers.assert_jit_grad_vmap_smoke(_make_nequip, feature_dim=_NUM_SCALARS)


class TestNequIPSpecifics:
    def test_node_feature_shape(self) -> None:
        backbone = _make_nequip(nnx.Rngs(0))
        system = _helpers.water()
        graph = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))
        features = backbone(system, graph)["node_features"]
        assert features.shape == (system.n_atoms, _NUM_SCALARS)

    def test_registered_under_name(self) -> None:
        assert BackboneRegistry().require("nequip") is NequIP

    def test_correlation_above_one_is_rejected(self) -> None:
        """The MACE-style higher-correlation upgrade is a documented deferral."""
        with pytest.raises(ValueError, match="symmetric contraction"):
            NequIP(config=NequIPConfig(correlation=2), rngs=nnx.Rngs(0))

    def test_hidden_irreps_without_scalar_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="0e scalar channel"):
            NequIP(config=NequIPConfig(hidden_irreps="4x1o"), rngs=nnx.Rngs(0))
