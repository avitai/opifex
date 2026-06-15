r"""Reference-gated fit of :class:`HamiltonianPredictor` to PySCF ground truth.

Generates the converged restricted-Hartree-Fock Fock matrix ``H`` and the AO
overlap matrix ``S`` from PySCF (``cart=True``, which reproduces opifex's exact
shell/AO ordering: atom-major, ``s`` shells then ``p`` in ``(x, y, z)``) for H2
and H2O in the STO-3G basis, then fits the equivariant predictor to each and
reports the element-wise MAE. The predictor is E(3)-equivariant by construction
(see :mod:`test_predictor`), so a single-geometry overfit is a sound smoke test
that the architecture has the capacity to represent a real Fock/overlap matrix in
opifex's AO ordering.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.hamiltonian import HamiltonianPredictor, HamiltonianPredictorConfig


_H2 = (
    jnp.array([1, 1]),
    jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
)
_H2O = (
    jnp.array([8, 1, 1]),
    jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]),
)


def _pyscf_targets(
    atomic_numbers: jax.Array, positions: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    """Return the converged RHF Fock ``H`` and overlap ``S`` from PySCF (cart order)."""
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")
    atoms = [
        (int(z), tuple(float(c) for c in pos))
        for z, pos in zip(np.asarray(atomic_numbers), np.asarray(positions), strict=True)
    ]
    mol = gto.M(atom=atoms, basis="sto-3g", unit="Bohr", cart=True)
    mean_field = scf.RHF(mol)
    mean_field.kernel()
    overlap = np.asarray(mol.intor("int1e_ovlp"))
    fock = np.asarray(mean_field.get_fock())
    return fock, overlap


def _fit_matrix(
    system: MolecularSystem,
    target: np.ndarray,
    *,
    property_name: str,
    steps: int,
    seed: int,
) -> np.ndarray:
    """Overfit the predictor to a single ground-truth matrix; return the prediction."""
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="sto-3g")
    config = HamiltonianPredictorConfig(
        hidden_irreps="32x0e + 24x1o + 16x2e",
        num_interactions=2,
        cutoff=8.0,
        property_name=property_name,
    )
    predictor = HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(predictor, optax.adam(3e-3), wrt=nnx.Param)
    target_array = jnp.asarray(target)

    def loss_fn(module: HamiltonianPredictor) -> jax.Array:
        prediction = module(system)[property_name]
        return jnp.mean((prediction - target_array) ** 2)

    @nnx.jit
    def step(module: HamiltonianPredictor, opt: nnx.Optimizer) -> None:
        grads = nnx.grad(loss_fn)(module)
        opt.update(module, grads)

    for _ in range(steps):
        step(predictor, optimizer)

    return np.asarray(predictor(system)[property_name])


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Element-wise mean absolute error between two matrices."""
    return float(np.mean(np.abs(prediction - target)))


@pytest.mark.parametrize(
    ("name", "atomic_numbers", "positions"),
    [("h2", *_H2), ("h2o", *_H2O)],
)
def test_fit_hamiltonian_to_pyscf(
    name: str, atomic_numbers: jax.Array, positions: jax.Array
) -> None:
    """The predictor can fit the PySCF Fock matrix to a small element-wise MAE."""
    pytest.importorskip("pyscf")
    pytest.importorskip("optax")
    system = MolecularSystem(atomic_numbers=atomic_numbers, positions=positions, basis_set="sto-3g")
    fock, _ = _pyscf_targets(atomic_numbers, positions)
    scale = float(np.abs(fock).max())
    prediction = _fit_matrix(system, fock, property_name="hamiltonian", steps=400, seed=0)
    mae = _mae(prediction, fock)
    # Element-wise MAE well below the matrix scale: the architecture represents H.
    assert mae < 0.05 * scale, f"{name} Fock MAE {mae:.4f} too large (scale {scale:.3f})"


def test_fit_overlap_to_pyscf() -> None:
    """The same head fits the PySCF overlap matrix S for H2O."""
    pytest.importorskip("pyscf")
    pytest.importorskip("optax")
    atomic_numbers, positions = _H2O
    system = MolecularSystem(atomic_numbers=atomic_numbers, positions=positions, basis_set="sto-3g")
    _, overlap = _pyscf_targets(atomic_numbers, positions)
    prediction = _fit_matrix(system, overlap, property_name="overlap", steps=400, seed=1)
    assert _mae(prediction, overlap) < 0.05, "H2O overlap MAE too large"


def test_predicted_hamiltonian_orbital_energies() -> None:
    """Generalised eigenvalues of the *predicted* H (with PySCF S) match PySCF.

    Solving ``H C = S C eps`` for the fitted Hamiltonian and the true overlap
    recovers the converged RHF orbital energies -- a physics-level check that the
    predicted matrix is usable, beyond element-wise MAE.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    pytest.importorskip("optax")
    atomic_numbers, positions = _H2O
    system = MolecularSystem(atomic_numbers=atomic_numbers, positions=positions, basis_set="sto-3g")
    fock, overlap = _pyscf_targets(atomic_numbers, positions)
    reference_energies = np.sort(scipy_linalg.eigh(fock, overlap, eigvals_only=True))

    predicted = _fit_matrix(system, fock, property_name="hamiltonian", steps=600, seed=0)
    predicted_energies = np.sort(scipy_linalg.eigh(predicted, overlap, eigvals_only=True))
    np.testing.assert_allclose(predicted_energies, reference_energies, atol=2e-2)
