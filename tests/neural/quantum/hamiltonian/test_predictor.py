r"""Tests for :class:`HamiltonianPredictor` (QHNet-style equivariant H/S prediction).

The central correctness gate is **block-wise equivariance**: under a random
rotation ``R`` of the molecular geometry the predicted dense matrix must transform
as ``H(R r) = D(R) H(r) D(R)^T``, where ``D(R)`` is the block-diagonal Wigner-D
matrix assembled per shell (each shell of degree ``l`` carried by ``wigner_d(l,
R)``). Symmetry (``H == H^T``), the registry wiring, and ``jit``/``grad``/``vmap``
cleanliness are also checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.spatial.transform import Rotation

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.registry import PropertyHeadRegistry
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.quantum.hamiltonian import HamiltonianPredictor, HamiltonianPredictorConfig


def _water() -> MolecularSystem:
    """A small water molecule (Bohr), enough to exercise H + O shells and edges."""
    return MolecularSystem(
        atomic_numbers=jnp.array([8, 1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]),
        basis_set="sto-3g",
    )


def _block_diagonal_wigner(basis: AtomicOrbitalBasis, rotation: jax.Array) -> jax.Array:
    """Assemble the per-shell block-diagonal Wigner-D rotation of the AO basis."""
    blocks = [wigner_d(shell.angular_momentum, rotation) for shell in basis.shells]
    return jax.scipy.linalg.block_diag(*blocks)


def _random_rotation(seed: int) -> jax.Array:
    """Return a random proper rotation matrix seeded reproducibly."""
    return jnp.asarray(Rotation.random(rng=np.random.default_rng(seed)).as_matrix())


def _predictor(system: MolecularSystem, *, seed: int = 0) -> HamiltonianPredictor:
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="sto-3g")
    config = HamiltonianPredictorConfig(
        hidden_irreps="8x0e + 8x1o + 4x2e",
        num_interactions=2,
        cutoff=6.0,
    )
    return HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(seed))


def test_registered_in_property_head_registry() -> None:
    """The predictor self-registers under the ``"hamiltonian"`` head name."""
    assert PropertyHeadRegistry().require("hamiltonian") is HamiltonianPredictor


def test_implemented_properties() -> None:
    """The head declares it emits the ``"hamiltonian"`` matrix."""
    system = _water()
    predictor = _predictor(system)
    assert "hamiltonian" in predictor.implemented_properties


def test_predicted_matrix_shape_and_symmetry() -> None:
    """The output is a square ``(n_ao, n_ao)`` symmetric matrix."""
    system = _water()
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="sto-3g")
    predictor = _predictor(system)
    matrix = predictor(system)["hamiltonian"]
    assert matrix.shape == (basis.n_atomic_orbitals, basis.n_atomic_orbitals)
    np.testing.assert_allclose(np.asarray(matrix), np.asarray(matrix.T), atol=1e-6)


def test_block_wise_equivariance() -> None:
    r"""``H(R r) = D(R) H(r) D(R)^T`` to ``~1e-5`` under random rotations."""
    system = _water()
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="sto-3g")
    predictor = _predictor(system, seed=1)

    matrix = predictor(system)["hamiltonian"]
    for seed in range(3):
        rotation = _random_rotation(seed)
        rotated_system = MolecularSystem(
            atomic_numbers=system.atomic_numbers,
            positions=system.positions @ rotation.T,
            basis_set=system.basis_set,
        )
        rotated_matrix = predictor(rotated_system)["hamiltonian"]
        wigner = _block_diagonal_wigner(basis, rotation)
        expected = wigner @ matrix @ wigner.T
        np.testing.assert_allclose(np.asarray(rotated_matrix), np.asarray(expected), atol=1e-4)


def test_overlap_head_equivariance() -> None:
    """An overlap predictor obeys the same block-wise transformation law."""
    system = _water()
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="sto-3g")
    config = HamiltonianPredictorConfig(
        hidden_irreps="8x0e + 8x1o + 4x2e", num_interactions=2, cutoff=6.0, property_name="overlap"
    )
    predictor = HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(2))
    assert "overlap" in predictor.implemented_properties

    matrix = predictor(system)["overlap"]
    rotation = _random_rotation(5)
    rotated_system = MolecularSystem(
        atomic_numbers=system.atomic_numbers,
        positions=system.positions @ rotation.T,
        basis_set=system.basis_set,
    )
    rotated_matrix = predictor(rotated_system)["overlap"]
    wigner = _block_diagonal_wigner(basis, rotation)
    expected = wigner @ matrix @ wigner.T
    np.testing.assert_allclose(np.asarray(rotated_matrix), np.asarray(expected), atol=1e-4)


def test_translation_invariance() -> None:
    """A rigid translation leaves the predicted matrix unchanged."""
    system = _water()
    predictor = _predictor(system, seed=3)
    matrix = predictor(system)["hamiltonian"]
    shifted = MolecularSystem(
        atomic_numbers=system.atomic_numbers,
        positions=system.positions + jnp.array([1.3, -0.7, 2.0]),
        basis_set=system.basis_set,
    )
    shifted_matrix = predictor(shifted)["hamiltonian"]
    np.testing.assert_allclose(np.asarray(matrix), np.asarray(shifted_matrix), atol=1e-5)


def test_jit_grad_vmap() -> None:
    """The predictor is ``jit``/``grad``/``vmap`` clean over positions."""
    system = _water()
    predictor = _predictor(system, seed=4)
    graphdef, state = nnx.split(predictor)

    def matrix_norm(state_in: nnx.State, positions: jax.Array) -> jax.Array:
        module = nnx.merge(graphdef, state_in)
        moved = MolecularSystem(
            atomic_numbers=system.atomic_numbers,
            positions=positions,
            basis_set=system.basis_set,
        )
        return jnp.sum(module(moved)["hamiltonian"] ** 2)

    positions = system.positions
    value = jax.jit(matrix_norm)(state, positions)
    assert jnp.isfinite(value)

    gradient = jax.grad(matrix_norm, argnums=1)(state, positions)
    assert gradient.shape == positions.shape
    assert jnp.all(jnp.isfinite(gradient))

    batch = jnp.stack([positions, positions + 0.1, positions - 0.2])
    batched = jax.vmap(matrix_norm, in_axes=(None, 0))(state, batch)
    assert batched.shape == (3,)


def test_padding_invariance_changing_atom_count() -> None:
    """The same predictor (shared per-l weights) handles a different molecule."""
    system = _water()
    predictor = _predictor(system, seed=6)

    # A hydrogen molecule: 2 atoms, only s shells. Build a fresh basis/plan but
    # reuse the *same* trunk + per-(l_i,l_j) expansion weights.
    h2 = MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        basis_set="sto-3g",
    )
    h2_basis = AtomicOrbitalBasis.from_molecular_system(h2, basis_name="sto-3g")
    rebound = predictor.rebind(h2_basis)
    matrix = rebound(h2)["hamiltonian"]
    assert matrix.shape == (h2_basis.n_atomic_orbitals, h2_basis.n_atomic_orbitals)
    np.testing.assert_allclose(np.asarray(matrix), np.asarray(matrix.T), atol=1e-6)
