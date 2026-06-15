r"""Tests for :mod:`opifex.neural.atomistic.conditioning`.

Exercises the UMA/OrbMol charge-spin conditioning contract (arXiv:2506.23971):
distinct charge/multiplicity produce distinct conditioning vectors, the neutral
singlet is a well-defined baseline, the conditioning broadcasts to
``(n_atoms, feature_dim)`` identically for every atom (permutation invariance),
and the module survives ``jit``/``grad``/``vmap`` with the static integer
charge/spin passed as trace-time constants (the repo's numpy-static-metadata
convention).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.atomistic.conditioning import (
    ChargeSpinConditioning,
    ChargeSpinConditioningConfig,
)


_FEATURE_DIM = 16
_N_ATOMS = 4


def _make_module(embedding_type: str = "table") -> ChargeSpinConditioning:
    """Build a conditioning module with a fixed seed for deterministic tests."""
    config = ChargeSpinConditioningConfig(feature_dim=_FEATURE_DIM, embedding_type=embedding_type)
    return ChargeSpinConditioning(config=config, rngs=nnx.Rngs(0))


def _make_system(charge: int = 0, multiplicity: int = 1) -> MolecularSystem:
    """Build a small water-like system with the given charge/multiplicity."""
    return MolecularSystem(
        atomic_numbers=jnp.array([8, 1, 1, 1]),
        positions=jnp.zeros((_N_ATOMS, 3)),
        charge=charge,
        multiplicity=multiplicity,
    )


@pytest.mark.parametrize("embedding_type", ["table", "fourier"])
def test_output_broadcasts_to_n_atoms_feature_dim(embedding_type: str) -> None:
    """The conditioning broadcasts to ``(n_atoms, feature_dim)``."""
    module = _make_module(embedding_type)
    out = module(charge=0, multiplicity=1, n_atoms=_N_ATOMS)
    assert out.shape == (_N_ATOMS, _FEATURE_DIM)


@pytest.mark.parametrize("embedding_type", ["table", "fourier"])
def test_all_atoms_share_the_same_conditioning(embedding_type: str) -> None:
    """Every atom gets the identical global conditioning (permutation invariant)."""
    module = _make_module(embedding_type)
    out = module(charge=1, multiplicity=2, n_atoms=_N_ATOMS)
    for atom_index in range(1, _N_ATOMS):
        np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(out[atom_index]))


@pytest.mark.parametrize("embedding_type", ["table", "fourier"])
def test_different_charge_gives_different_vector(embedding_type: str) -> None:
    """Distinct total charge yields a distinct conditioning vector."""
    module = _make_module(embedding_type)
    neutral = module(charge=0, multiplicity=1, n_atoms=_N_ATOMS)
    cation = module(charge=1, multiplicity=1, n_atoms=_N_ATOMS)
    assert not np.allclose(np.asarray(neutral), np.asarray(cation))


@pytest.mark.parametrize("embedding_type", ["table", "fourier"])
def test_different_multiplicity_gives_different_vector(embedding_type: str) -> None:
    """Distinct spin multiplicity yields a distinct conditioning vector."""
    module = _make_module(embedding_type)
    singlet = module(charge=0, multiplicity=1, n_atoms=_N_ATOMS)
    triplet = module(charge=0, multiplicity=3, n_atoms=_N_ATOMS)
    assert not np.allclose(np.asarray(singlet), np.asarray(triplet))


def test_neutral_singlet_is_well_defined_baseline() -> None:
    """The neutral singlet (charge=0, mult=1) produces a finite baseline vector."""
    module = _make_module()
    out = module(charge=0, multiplicity=1, n_atoms=_N_ATOMS)
    assert out.shape == (_N_ATOMS, _FEATURE_DIM)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_from_system_reads_charge_and_multiplicity() -> None:
    """The system-driven entry point reads charge/multiplicity off MolecularSystem."""
    module = _make_module()
    system = _make_system(charge=-1, multiplicity=2)
    out_from_system = module.from_system(system)
    out_explicit = module(charge=-1, multiplicity=2, n_atoms=system.n_atoms)
    np.testing.assert_array_equal(np.asarray(out_from_system), np.asarray(out_explicit))


def test_out_of_range_charge_raises() -> None:
    """A charge outside the documented bounded range fails fast."""
    module = _make_module("table")
    with pytest.raises(ValueError, match="charge"):
        module(charge=500, multiplicity=1, n_atoms=_N_ATOMS)


def test_invalid_multiplicity_raises() -> None:
    """A non-positive multiplicity fails fast."""
    module = _make_module("table")
    with pytest.raises(ValueError, match="multiplicity"):
        module(charge=0, multiplicity=0, n_atoms=_N_ATOMS)


def test_addition_to_node_features_preserves_shape() -> None:
    """The conditioning adds onto backbone node features without reshaping."""
    module = _make_module()
    node_features = jnp.ones((_N_ATOMS, _FEATURE_DIM))
    conditioned = node_features + module(charge=2, multiplicity=1, n_atoms=_N_ATOMS)
    assert conditioned.shape == node_features.shape


def test_jit_with_static_charge_spin() -> None:
    """``jit`` with charge/spin/n_atoms marked static traces and runs."""
    module = _make_module()
    graphdef, state = nnx.split(module)

    @partial_jit_static
    def apply(state_arg, charge: int, multiplicity: int, n_atoms: int):
        merged = nnx.merge(graphdef, state_arg)
        return merged(charge=charge, multiplicity=multiplicity, n_atoms=n_atoms)

    out = apply(state, 1, 2, _N_ATOMS)
    assert out.shape == (_N_ATOMS, _FEATURE_DIM)


def test_grad_flows_to_embedding_parameters() -> None:
    """``grad`` w.r.t. the learned parameters is finite (charge/spin are static)."""
    module = _make_module()
    graphdef, state = nnx.split(module)

    def loss(state_arg) -> jax.Array:
        merged = nnx.merge(graphdef, state_arg)
        return jnp.sum(merged(charge=1, multiplicity=2, n_atoms=_N_ATOMS) ** 2)

    grads = jax.grad(loss)(state)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_vmap_over_a_batch_of_systems() -> None:
    """``vmap`` over per-system parameter states maps with static charge/spin."""
    module = _make_module()
    graphdef, state = nnx.split(module)
    batch_size = 3
    batched_state = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (batch_size, *leaf.shape)), state
    )

    def apply(state_arg) -> jax.Array:
        merged = nnx.merge(graphdef, state_arg)
        return merged(charge=1, multiplicity=2, n_atoms=_N_ATOMS)

    out = jax.vmap(apply)(batched_state)
    assert out.shape == (batch_size, _N_ATOMS, _FEATURE_DIM)


def partial_jit_static(fn):
    """Wrap ``fn`` in ``jit`` marking the trailing int args static.

    The state is argument 0 (traced); ``charge``, ``multiplicity`` and
    ``n_atoms`` are arguments 1-3 and static, matching the numpy-static-metadata
    convention used across the atomistic stack.
    """
    return jax.jit(fn, static_argnums=(1, 2, 3))
