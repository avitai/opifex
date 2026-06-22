r"""Tests for the :class:`NequIP` E(3)-equivariant tensor-product backbone.

Load-bearing physics contracts (shared via :mod:`._helpers`): energy invariance
under rotation/translation/permutation, force equivariance, force = -grad(E)
versus finite differences, learning a toy pairwise potential, and jit/grad/vmap
cleanliness. NequIP-specific checks cover the scalar-readout shape, registry
self-registration, the config-option wiring, and the higher-body-order
(correlation>1 symmetric-contraction) path and its constraints.
"""

from __future__ import annotations

import jax
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

    def test_correlation_above_one_requires_uniform_multiplicity(self) -> None:
        """The symmetric-contraction (correlation>1) path needs uniform-mul irreps."""
        with pytest.raises(ValueError, match="uniform-multiplicity"):
            NequIP(
                config=NequIPConfig(hidden_irreps="8x0e + 4x1o", correlation=2, species=(1, 8)),
                rngs=nnx.Rngs(0),
            )

    def test_correlation_above_one_requires_species(self) -> None:
        """The symmetric contraction's per-element weights need config.species."""
        with pytest.raises(ValueError, match="species"):
            NequIP(
                config=NequIPConfig(hidden_irreps="8x0e + 8x1o", correlation=2),
                rngs=nnx.Rngs(0),
            )

    def test_hidden_irreps_without_scalar_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="0e scalar channel"):
            NequIP(config=NequIPConfig(hidden_irreps="4x1o"), rngs=nnx.Rngs(0))

    def test_config_defaults_to_nequip_conventions(self) -> None:
        """The conditioning options default to the NequIP-correct values."""
        config = NequIPConfig()
        assert config.sh_normalization == "component"
        assert config.normalize_gate_act is True

    def test_gate_normalization_option_changes_output(self) -> None:
        """``normalize_gate_act`` is wired through to the gate (alters features)."""
        system = _helpers.water()
        graph = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))

        def features(normalize: bool) -> jnp.ndarray:
            config = NequIPConfig(
                hidden_irreps=_HIDDEN_IRREPS,
                sh_lmax=2,
                num_interactions=2,
                num_radial_basis=8,
                cutoff=5.0,
                normalize_gate_act=normalize,
            )
            return NequIP(config=config, rngs=nnx.Rngs(0))(system, graph)["node_features"]

        assert not jnp.allclose(features(normalize=True), features(normalize=False))


class TestNequIPSpeciesSkip:
    """The species-indexed self-connection stays a valid E(3) potential."""

    @staticmethod
    def _make_species_nequip(rngs: nnx.Rngs) -> NequIP:
        # Water is O, H, H -> distinct atomic numbers 1 and 8.
        config = NequIPConfig(
            hidden_irreps=_HIDDEN_IRREPS,
            sh_lmax=2,
            num_interactions=2,
            num_radial_basis=8,
            cutoff=5.0,
            species=(1, 8),
        )
        return NequIP(config=config, rngs=rngs)

    def test_energy_invariant(self) -> None:
        _helpers.assert_energy_invariant(self._make_species_nequip, feature_dim=_NUM_SCALARS)

    def test_force_equivariant(self) -> None:
        _helpers.assert_force_equivariant(self._make_species_nequip, feature_dim=_NUM_SCALARS)

    def test_forces_match_finite_difference(self) -> None:
        _helpers.assert_forces_match_finite_difference(
            self._make_species_nequip, feature_dim=_NUM_SCALARS
        )

    def test_jit_grad_vmap_smoke(self) -> None:
        _helpers.assert_jit_grad_vmap_smoke(self._make_species_nequip, feature_dim=_NUM_SCALARS)

    def test_species_skip_changes_output(self) -> None:
        """The per-element skip differs from the shared-linear self-connection."""
        system = _helpers.water()
        graph = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))
        shared = _make_nequip(nnx.Rngs(0))(system, graph)["node_features"]
        species = self._make_species_nequip(nnx.Rngs(0))(system, graph)["node_features"]
        assert not jnp.allclose(shared, species)


class TestNequIPHigherBodyOrder:
    """A correlation>1 (symmetric-contraction) NequIP stays a valid E(3) potential."""

    @staticmethod
    def _make_corr3(rngs: nnx.Rngs) -> NequIP:
        # Uniform-multiplicity hidden irreps (required by the channel-wise contraction).
        config = NequIPConfig(
            hidden_irreps="8x0e + 8x1o + 8x2e",
            sh_lmax=2,
            num_interactions=2,
            num_radial_basis=8,
            cutoff=5.0,
            correlation=3,
            species=(1, 8),
        )
        return NequIP(config=config, rngs=rngs)

    def test_energy_invariant(self) -> None:
        _helpers.assert_energy_invariant(self._make_corr3, feature_dim=8)

    def test_force_equivariant(self) -> None:
        _helpers.assert_force_equivariant(self._make_corr3, feature_dim=8)

    def test_forces_match_finite_difference(self) -> None:
        _helpers.assert_forces_match_finite_difference(self._make_corr3, feature_dim=8)

    def test_jit_grad_vmap_smoke(self) -> None:
        _helpers.assert_jit_grad_vmap_smoke(self._make_corr3, feature_dim=8)

    def test_higher_body_order_changes_output(self) -> None:
        """correlation=3 differs from the two-body correlation=1 baseline."""
        system = _helpers.water()
        graph = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))
        corr3 = self._make_corr3(nnx.Rngs(0))(system, graph)["node_features"]
        config1 = NequIPConfig(
            hidden_irreps="8x0e + 8x1o + 8x2e",
            sh_lmax=2,
            num_interactions=2,
            num_radial_basis=8,
            cutoff=5.0,
            correlation=1,
            species=(1, 8),
        )
        corr1 = NequIP(config=config1, rngs=nnx.Rngs(0))(system, graph)["node_features"]
        assert corr3.shape == corr1.shape
        assert not jnp.allclose(corr3, corr1)


class TestNequIPDtype:
    """The whole NequIP stack tracks JAX's ``x64`` flag uniformly.

    The equivariant weights already promote under ``x64`` (bare
    ``jax.random.normal``); these tests guard the previously float32-pinned
    ``nnx.Linear`` radial MLP + energy-head readout so that energy and -- via
    ``jax.grad`` -- the forces retain full double precision when requested.
    """

    @staticmethod
    def _param_dtypes(model: object) -> set[jnp.dtype]:
        leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
        return {leaf.dtype for leaf in leaves}

    def test_params_float32_without_x64(self) -> None:
        model = _helpers.build_model(_make_nequip, feature_dim=_NUM_SCALARS)
        assert self._param_dtypes(model) == {jnp.dtype(jnp.float32)}

    def test_params_and_outputs_float64_under_x64(self) -> None:
        with jax.enable_x64(True):
            model = _helpers.build_model(_make_nequip, feature_dim=_NUM_SCALARS)
            assert self._param_dtypes(model) == {jnp.dtype(jnp.float64)}
            outputs = model(_helpers.water())
            assert outputs["energy"].dtype == jnp.dtype(jnp.float64)
            assert outputs["forces"].dtype == jnp.dtype(jnp.float64)
