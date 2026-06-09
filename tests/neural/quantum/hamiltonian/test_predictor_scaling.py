r"""Compile-scaling and behaviour-identity gates for :class:`HamiltonianPredictor`.

The block-assembly stage used to run a Python ``for`` loop over every shell-pair
block (one :func:`jax.lax.dynamic_update_slice` per block). That loop is fully
unrolled at trace time, so XLA compilation grew with the *block count*: a 3-atom
water molecule (~144 blocks) compiles in seconds, but a 12-atom molecule (~2025
blocks) took well over an hour. The assembly is now vectorised per
angular-momentum pair type (one gather + one batched expansion + one scatter per
type), so compilation depends only on the small, fixed number of pair types.

Two gates guard the refactor:

#. **Behaviour identity** -- on water the vectorised assembly must reproduce, to
   ``1e-10``, the result of the old per-block reference (re-implemented inline
   here from the same specs and expansions), proving the rewrite is exact.
#. **Compile scaling** -- a 12-atom def2-SVP molecule's ``__call__`` must compile
   *and* run in well under a minute (it was > 3400 s before).
"""

from __future__ import annotations

import time

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.equivariant import IrrepsArray
from opifex.neural.quantum.hamiltonian import HamiltonianPredictor, HamiltonianPredictorConfig
from opifex.neural.quantum.hamiltonian._expansion import PairExpansion  # noqa: TC001
from opifex.neural.quantum.hamiltonian.predictor import _pair_type_key


def _water() -> MolecularSystem:
    """A small water molecule (Bohr) exercising H/O shells and edges."""
    return MolecularSystem(
        atomic_numbers=jnp.array([8, 1, 1]),
        positions=jnp.array(
            [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]], dtype=jnp.float64
        ),
        basis_set="sto-3g",
    )


def _benzene() -> MolecularSystem:
    """Benzene C6H6 (12 atoms, Bohr) -- the def2-SVP compile-scaling stress case."""
    # Planar D6h benzene; C-C ~ 2.64 Bohr, C-H ~ 2.04 Bohr (standard geometry).
    carbon_radius = 2.64
    hydrogen_radius = 4.68
    angles = np.deg2rad(np.arange(6) * 60.0)
    carbons = np.stack(
        [carbon_radius * np.cos(angles), carbon_radius * np.sin(angles), np.zeros(6)], axis=1
    )
    hydrogens = np.stack(
        [hydrogen_radius * np.cos(angles), hydrogen_radius * np.sin(angles), np.zeros(6)], axis=1
    )
    positions = np.concatenate([carbons, hydrogens], axis=0)
    atomic_numbers = np.array([6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    return MolecularSystem(
        atomic_numbers=jnp.asarray(atomic_numbers),
        positions=jnp.asarray(positions, dtype=jnp.float64),
        basis_set="def2-svp",
    )


def _reference_assemble(
    predictor: HamiltonianPredictor,
    node_features: IrrepsArray,
    edge_features: IrrepsArray,
) -> jax.Array:
    """Old per-block assembly: one ``dynamic_update_slice`` per shell-pair block.

    Reproduces the pre-refactor ``_assemble`` exactly from the still-present
    ``_diagonal_specs`` / ``_off_specs`` and the shared expansions, so the
    vectorised implementation can be checked against the original behaviour.
    """
    dtype = node_features.array.dtype
    matrix = jnp.zeros((predictor._n_ao, predictor._n_ao), dtype=dtype)
    for row_offset, col_offset, l_i, l_j, rank_i, rank_j, atom_i in predictor._diagonal_specs:
        expansion: PairExpansion = predictor.node_expansions[_pair_type_key(l_i, l_j)]
        feature = IrrepsArray(node_features.irreps, node_features.array[atom_i])
        block = expansion(feature)[rank_i, rank_j]
        matrix = jax.lax.dynamic_update_slice(matrix, block, (row_offset, col_offset))
    for row_offset, col_offset, l_i, l_j, rank_i, rank_j, edge_slot in predictor._off_specs:
        expansion = predictor.edge_expansions[_pair_type_key(l_i, l_j)]
        feature = IrrepsArray(edge_features.irreps, edge_features.array[edge_slot])
        block = expansion(feature)[rank_i, rank_j]
        matrix = jax.lax.dynamic_update_slice(matrix, block, (row_offset, col_offset))
    return matrix


def test_vectorized_assemble_matches_per_block_reference() -> None:
    r"""Vectorised ``_assemble`` equals the old per-block result to ``1e-10`` on water.

    Run in float64 so the comparison reflects only the assembly *logic*: the
    batched expansion and the per-block expansion contract the same weights, but
    on GPU float32 their matmuls schedule into different (TF32-class) kernels that
    disagree at ~1e-3. float64 removes that hardware noise, leaving a residual at
    machine epsilon and proving the rewrite is mathematically identical.
    """
    with jax.experimental.enable_x64():
        system = _water()
        basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="sto-3g")
        config = HamiltonianPredictorConfig(
            hidden_irreps="8x0e + 8x1o + 4x2e", num_interactions=2, cutoff=6.0
        )
        predictor = HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(0))

        graph = predictor._complete_graph(system.n_atoms)
        node_features, geometry, edge_sh, radial_envelope = predictor._node_features(system, graph)
        edge_features = predictor._edge_features(node_features, geometry, edge_sh, radial_envelope)

        vectorized = predictor._assemble(node_features, edge_features)
        reference = _reference_assemble(predictor, node_features, edge_features)

    assert vectorized.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(vectorized), np.asarray(reference), atol=1e-10, rtol=0.0)


def test_twelve_atom_predictor_compiles_under_one_minute() -> None:
    """A 12-atom def2-SVP molecule's ``__call__`` compiles + runs well under 60 s."""
    system = _benzene()
    assert system.n_atoms == 12
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="def2-svp")
    config = HamiltonianPredictorConfig(
        hidden_irreps="8x0e + 8x1o + 4x2e", sh_lmax=2, num_interactions=2, cutoff=6.0
    )
    predictor = HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(predictor)

    def predict(state_in: nnx.State, positions: jax.Array) -> jax.Array:
        module = nnx.merge(graphdef, state_in)
        moved = MolecularSystem(
            atomic_numbers=system.atomic_numbers,
            positions=positions,
            basis_set=system.basis_set,
        )
        return module(moved)["hamiltonian"]

    compiled = jax.jit(predict)
    start = time.perf_counter()
    matrix = compiled(state, system.positions)
    matrix.block_until_ready()
    elapsed = time.perf_counter() - start

    assert matrix.shape == (basis.n_atomic_orbitals, basis.n_atomic_orbitals)
    assert jnp.all(jnp.isfinite(matrix))
    assert elapsed < 60.0, f"12-atom compile+run took {elapsed:.1f}s (expected < 60s)"


def test_twelve_atom_predictor_jit_grad_vmap_compatible() -> None:
    """The vectorised predictor stays jit/grad/vmap clean on a 12-atom molecule.

    JAX/NNX transform compatibility is a required exit criterion: the rewrite
    must keep ``__call__`` differentiable (forces) and batchable over geometry
    (the batched QH9 training path), not just compilable.
    """
    system = _benzene()
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="def2-svp")
    config = HamiltonianPredictorConfig(
        hidden_irreps="8x0e + 8x1o + 4x2e", sh_lmax=2, num_interactions=2, cutoff=6.0
    )
    predictor = HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(predictor)

    def predict(positions: jax.Array) -> jax.Array:
        module = nnx.merge(graphdef, state)
        moved = MolecularSystem(
            atomic_numbers=system.atomic_numbers,
            positions=positions,
            basis_set=system.basis_set,
        )
        return module(moved)["hamiltonian"]

    # jit: symmetric Hamiltonian.
    matrix = jax.jit(predict)(system.positions)
    assert jnp.allclose(matrix, matrix.T)

    # grad: a scalar energy of the matrix is differentiable w.r.t. geometry.
    grad = jax.grad(lambda positions: jnp.sum(predict(positions) ** 2))(system.positions)
    assert grad.shape == system.positions.shape
    assert jnp.all(jnp.isfinite(grad))

    # vmap: batched over a stack of geometries (the bucketed-training contract).
    # Assert the transform-correctness properties (shape, finiteness, per-slice
    # symmetry); exact batched-vs-single equality is a float64 concern covered by
    # test_vectorized_assemble_matches_per_block_reference -- on GPU float32 the
    # batched and single matmuls schedule into different TF32-class kernels that
    # differ at ~1e-3, which is a hardware artifact, not a transform issue.
    batch = jnp.broadcast_to(system.positions, (4, *system.positions.shape))
    batched = jax.jit(jax.vmap(predict))(batch)
    assert batched.shape == (4, basis.n_atomic_orbitals, basis.n_atomic_orbitals)
    assert jnp.all(jnp.isfinite(batched))
    for single in batched:
        assert jnp.allclose(single, single.T)


def test_predictor_survives_repeated_nnx_jit_calls() -> None:
    """``nnx.jit`` over the predictor stays valid across repeated calls.

    A training loop calls the jitted step on the same module instance every
    step; on the second call NNX checks the graphdef *metadata* for equality, and
    it rejects array-valued metadata. The static assembly scatter plans must
    therefore be hashable (tuple) fields, not arrays -- this guards that contract
    (the per-step batched QH9 training path) which a single jit call cannot catch.
    """
    system = _water()
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="sto-3g")
    config = HamiltonianPredictorConfig(
        hidden_irreps="8x0e + 8x1o + 4x2e", num_interactions=2, cutoff=6.0
    )
    predictor = HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(0))

    @nnx.jit
    def call(module: HamiltonianPredictor, positions: jax.Array) -> jax.Array:
        moved = MolecularSystem(
            atomic_numbers=system.atomic_numbers,
            positions=positions,
            basis_set=system.basis_set,
        )
        return module(moved)["hamiltonian"]

    first = call(predictor, system.positions)
    # Second call on the same instance triggers the graphdef metadata equality
    # check that previously raised "arrays cannot be passed as metadata fields".
    second = call(predictor, system.positions + 0.01)
    assert first.shape == second.shape == (basis.n_atomic_orbitals,) * 2
    assert jnp.all(jnp.isfinite(first)) and jnp.all(jnp.isfinite(second))
