r"""Tests for the :class:`EvidentialEnergyHead` single-pass UQ energy readout.

Load-bearing contracts:

* the head emits the total ``"energy"`` plus its evidential NIG parameters
  (``energy_nu``, ``energy_alpha``, ``energy_beta``) and the
  uncertainty-variance decomposition
  (``energy_aleatoric_var``, ``energy_epistemic_var``, ``energy_variance``);
* the total energy is the sum of per-atom evidential means (eIP, the
  extensive sum-of-atomic-energies contract);
* aleatoric variance is strictly below the total variance (epistemic > 0);
* the head is registered under ``"evidential_energy"``;
* the head reuses the NIG primitive so its outputs map to a valid
  :class:`PredictiveDistribution`;
* the forward pass is ``jit``/``grad``/``vmap`` clean (REQUIRED).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.registry import PropertyHeadRegistry
from opifex.neural.atomistic.heads.evidential import EvidentialEnergyHead
from opifex.uncertainty.evidential import nig_to_predictive_distribution, NIGParams


_FEATURE_DIM = 6
_N_ATOMS = 4
_GRAPH: tuple[Array, Array] = (jnp.asarray([0, 1, 2, 3]), jnp.asarray([1, 2, 3, 0]))


def _system() -> MolecularSystem:
    """Return a tiny 4-atom system (CH3-like) for plumbing tests."""
    positions = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.9, 0.0],
            [-0.5, -0.9, 0.0],
        ],
        dtype=jnp.float64,
    )
    return MolecularSystem(atomic_numbers=jnp.asarray([6, 1, 1, 1]), positions=positions)


def _embeddings(seed: int = 0) -> dict[str, Array]:
    """Return per-atom invariant ``node_features`` for the 4-atom system."""
    key = jax.random.PRNGKey(seed)
    return {"node_features": jax.random.normal(key, (_N_ATOMS, _FEATURE_DIM), dtype=jnp.float64)}


def _head() -> EvidentialEnergyHead:
    return EvidentialEnergyHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))


class TestEvidentialEnergyHead:
    def test_implemented_properties_include_energy_and_uncertainty(self) -> None:
        head = _head()
        props = head.implemented_properties
        assert "energy" in props
        for key in (
            "energy_nu",
            "energy_alpha",
            "energy_beta",
            "energy_aleatoric_var",
            "energy_epistemic_var",
            "energy_variance",
        ):
            assert key in props

    def test_output_keys_match_implemented_properties(self) -> None:
        head = _head()
        out = head(_system(), _GRAPH, _embeddings())
        assert set(out.keys()) == set(head.implemented_properties)

    def test_energy_is_scalar(self) -> None:
        head = _head()
        out = head(_system(), _GRAPH, _embeddings())
        assert out["energy"].shape == ()

    def test_total_nig_params_are_valid(self) -> None:
        head = _head()
        out = head(_system(), _GRAPH, _embeddings())
        assert float(out["energy_nu"]) > 0.0
        assert float(out["energy_alpha"]) > 1.0
        assert float(out["energy_beta"]) > 0.0

    def test_energy_is_sum_of_per_atom_means(self) -> None:
        with jax.enable_x64(True):
            head = _head()
            embeddings = _embeddings()
            per_atom = head.per_atom_params(embeddings["node_features"])
            total = head(_system(), _GRAPH, embeddings)["energy"]
            assert jnp.allclose(total, jnp.sum(per_atom.gamma), atol=1e-10)

    def test_aleatoric_below_total_variance(self) -> None:
        head = _head()
        out = head(_system(), _GRAPH, _embeddings())
        assert float(out["energy_aleatoric_var"]) < float(out["energy_variance"])

    def test_total_variance_is_aleatoric_plus_epistemic(self) -> None:
        with jax.enable_x64(True):
            head = _head()
            out = head(_system(), _GRAPH, _embeddings())
            expected = out["energy_aleatoric_var"] + out["energy_epistemic_var"]
            assert jnp.allclose(out["energy_variance"], expected, atol=1e-10)

    def test_outputs_map_to_valid_predictive_distribution(self) -> None:
        with jax.enable_x64(True):
            head = _head()
            out = head(_system(), _GRAPH, _embeddings())
            total = NIGParams(
                gamma=out["energy"],
                nu=out["energy_nu"],
                alpha=out["energy_alpha"],
                beta=out["energy_beta"],
            )
            predictive = nig_to_predictive_distribution(total)
            predictive.validate()
            assert jnp.allclose(predictive.mean, out["energy"], atol=1e-10)

    def test_registered_under_name(self) -> None:
        assert PropertyHeadRegistry().require("evidential_energy") is EvidentialEnergyHead

    def test_jit_grad_vmap_smoke(self) -> None:
        head = _head()
        graphdef, state = nnx.split(head)
        system = _system()
        node_features = _embeddings()["node_features"]

        def energy_for(features: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            return rebuilt(system, _GRAPH, {"node_features": features})["energy"]

        jitted = jax.jit(energy_for)
        energy = jitted(node_features)
        assert energy.shape == ()
        assert bool(jnp.isfinite(energy))

        gradient = jax.grad(lambda f: energy_for(f) ** 2)(node_features)
        assert gradient.shape == node_features.shape
        assert bool(jnp.all(jnp.isfinite(gradient)))

        batch = jnp.stack([node_features, node_features + 0.1])
        batched = jax.vmap(energy_for)(batch)
        assert batched.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(batched)))

    def test_variance_outputs_are_jit_clean(self) -> None:
        head = _head()
        graphdef, state = nnx.split(head)
        system = _system()
        node_features = _embeddings()["node_features"]

        def variance_for(features: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            return rebuilt(system, _GRAPH, {"node_features": features})["energy_variance"]

        var = jax.jit(variance_for)(node_features)
        assert var.shape == ()
        assert bool(jnp.isfinite(var)) and float(var) > 0.0
