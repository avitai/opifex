r"""Tests for the QH9 Fock-prediction training harness.

The harness (``opifex.neural.quantum.hamiltonian.qh9_training``) wires the
committed :class:`HamiltonianPredictor` to QH9 (Yu et al. 2023, arXiv:2306.04922)
def2-SVP Fock targets in spherical AO ordering. The predictor emits a Cartesian
AO matrix; the harness maps it to the spherical basis (``H_sph = T^T H_cart T``)
so prediction and target share QH9's ordering.

These tests use small SYNTHETIC symmetric Fock targets (not real QH9 data) on a
tiny water molecule and assert: the predicted matrix lands in spherical AO
ordering; the masked Fock loss reduces correctly; one train step runs and the
loss decreases on an overfit-one-batch fit; the loss is jit/grad clean.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.hamiltonian import HamiltonianPredictor, HamiltonianPredictorConfig
from opifex.neural.quantum.hamiltonian.qh9_training import (
    batched_fock_loss,
    fit_qh9,
    fit_qh9_bucket,
    fock_loss,
    make_batched_train_step,
    predict_spherical_fock,
    predict_spherical_fock_batch,
    QH9TrainConfig,
    spherical_transform_for,
)


def _water() -> MolecularSystem:
    """A small water molecule (Bohr) exercising H/O def2-SVP shells (incl. d)."""
    return MolecularSystem(
        atomic_numbers=jnp.asarray([8, 1, 1], dtype=jnp.int32),
        positions=jnp.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]], dtype=jnp.float64
        ),
        basis_set="def2-svp",
    )


def _predictor(system: MolecularSystem, *, seed: int = 0) -> HamiltonianPredictor:
    """Build a small def2-SVP-capable predictor (carries the 2e d-shell channel)."""
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="def2-svp")
    config = HamiltonianPredictorConfig(
        hidden_irreps="8x0e + 8x1o + 4x2e",
        sh_lmax=2,
        num_interactions=2,
        cutoff=6.0,
    )
    return HamiltonianPredictor(basis=basis, config=config, rngs=nnx.Rngs(seed))


def _noise_target(system: MolecularSystem, *, seed: int = 3) -> jax.Array:
    """A random symmetric spherical-AO matrix (NOT real QH9 data, NOT realisable).

    Used only for jit/grad and shape smoke-checks where the *value* of the
    target is irrelevant.
    """
    transform = spherical_transform_for(system)
    n_ao = int(transform.shape[1])
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n_ao, n_ao)).astype(np.float64)
    return jnp.asarray(matrix + matrix.T)


def _teacher_target(system: MolecularSystem, *, seed: int = 7) -> jax.Array:
    """A *realisable* spherical-AO Fock target from a teacher predictor.

    A randomly-initialised predictor emits a symmetric, equivariant spherical
    Fock matrix; using it as the target makes the overfit-one-batch check
    meaningful (a different-seed student can in principle reproduce it),
    standing in for a real QH9 Fock without any download.
    """
    teacher = _predictor(system, seed=seed)
    transform = spherical_transform_for(system)
    return jax.lax.stop_gradient(predict_spherical_fock(teacher, system, transform))


# =============================================================================
# Spherical prediction ordering
# =============================================================================


def test_predicted_fock_is_spherical_shaped_and_symmetric() -> None:
    """The predicted Fock is the spherical ``(24, 24)`` symmetric H2O matrix."""
    system = _water()
    predictor = _predictor(system)
    transform = spherical_transform_for(system)
    prediction = predict_spherical_fock(predictor, system, transform)
    assert prediction.shape == (24, 24)
    np.testing.assert_allclose(np.asarray(prediction), np.asarray(prediction).T, atol=1e-5)


def test_spherical_transform_maps_cartesian_to_spherical() -> None:
    """The transform turns the 25-AO Cartesian H2O matrix into 24 spherical AOs."""
    system = _water()
    transform = spherical_transform_for(system)
    assert transform.shape == (25, 24)


# =============================================================================
# Fock loss
# =============================================================================


def test_fock_loss_zero_on_identical_matrices() -> None:
    """The loss vanishes when prediction equals target."""
    matrix = jnp.asarray(np.eye(4))
    assert float(fock_loss(matrix, matrix, kind="mae")) == 0.0
    assert float(fock_loss(matrix, matrix, kind="mse")) == 0.0


def test_fock_loss_mae_value() -> None:
    """MAE reduces to the mean absolute residual over all entries."""
    prediction = jnp.zeros((2, 2))
    target = jnp.asarray([[1.0, -3.0], [2.0, 0.0]])
    assert float(fock_loss(prediction, target, kind="mae")) == 1.5


def test_fock_loss_mask_excludes_padding() -> None:
    """Masked entries do not contribute to the reduced loss."""
    prediction = jnp.zeros((2, 2))
    target = jnp.asarray([[1.0, 100.0], [100.0, 1.0]])
    mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    # Only the two diagonal residuals (each 1.0) count.
    assert float(fock_loss(prediction, target, kind="mae", mask=mask)) == 1.0


# =============================================================================
# jit / grad smoke
# =============================================================================


def test_loss_is_jit_and_grad_clean() -> None:
    """The spherical Fock loss is differentiable through the predictor under jit."""
    system = _water()
    predictor = _predictor(system)
    transform = spherical_transform_for(system)
    target = _noise_target(system)

    def loss_of(module: HamiltonianPredictor) -> jax.Array:
        prediction = predict_spherical_fock(module, system, transform)
        return fock_loss(prediction, target, kind="mse")

    loss, grads = nnx.jit(nnx.value_and_grad(loss_of))(predictor)
    assert jnp.isfinite(loss)
    flat = jax.tree_util.tree_leaves(grads)
    assert flat
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in flat)


# =============================================================================
# Overfit-one-batch: a single train step runs and the loss decreases
# =============================================================================


def test_single_train_step_runs_and_loss_decreases() -> None:
    """Overfitting one molecule's Fock target reduces the loss monotonically-ish."""
    system = _water()
    predictor = _predictor(system, seed=0)
    target = _teacher_target(system, seed=7)

    result = fit_qh9(
        predictor,
        system,
        target,
        config=QH9TrainConfig(learning_rate=1e-2, num_steps=200, loss_kind="mse"),
    )
    history = np.asarray(result.loss_history)
    assert history.shape == (200,)
    assert np.all(np.isfinite(history))
    # The loss decreases essentially monotonically (Adam on a single batch).
    assert result.final_loss < float(history[0])
    assert float(history[-1]) < float(history[len(history) // 2]) < float(history[0])
    # And drops substantially from its first value (overfit-one-batch).
    assert result.final_loss < 0.5 * float(history[0])


# =============================================================================
# Batched bucket training: vmap over a same-signature bucket fills the GPU
# =============================================================================


def _water_atoms() -> jax.Array:
    """The shared atomic-number sequence of a water-signature bucket."""
    return jnp.asarray([8, 1, 1], dtype=jnp.int32)


def _water_positions_batch(batch: int = 4) -> jax.Array:
    """A batch of perturbed water geometries sharing the water Z signature."""
    base = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]], dtype=jnp.float64)
    rng = np.random.default_rng(0)
    deltas = rng.standard_normal((batch, 3, 3)).astype(np.float64) * 0.05
    return jnp.asarray(np.asarray(base)[None] + deltas)


def _teacher_targets_batch(atoms: jax.Array, positions: jax.Array, *, seed: int = 7) -> jax.Array:
    """Realisable per-molecule spherical Fock targets from a teacher predictor."""
    system0 = MolecularSystem(atomic_numbers=atoms, positions=positions[0], basis_set="def2-svp")
    teacher = _predictor(system0, seed=seed)
    transform = spherical_transform_for(system0)
    return jax.lax.stop_gradient(predict_spherical_fock_batch(teacher, atoms, positions, transform))


def test_batched_fock_loss_sums_per_molecule_errors() -> None:
    """The batched loss is the sum of per-molecule masked-mean errors."""
    predictions = jnp.zeros((2, 2, 2))
    targets = jnp.asarray([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]])
    # Unmasked MAE per molecule: 1.0 and 2.0 -> sum 3.0.
    assert float(batched_fock_loss(predictions, targets, kind="mae")) == 3.0
    mask = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    # Masked diagonals only: still 1.0 + 2.0 = 3.0 (means over 2 valid entries).
    assert float(batched_fock_loss(predictions, targets, kind="mae", mask=mask)) == 3.0


def test_batched_prediction_matches_per_molecule_prediction() -> None:
    """vmap over a bucket reproduces the per-molecule spherical Fock predictions."""
    atoms = _water_atoms()
    positions = _water_positions_batch(batch=3)
    system0 = MolecularSystem(atomic_numbers=atoms, positions=positions[0], basis_set="def2-svp")
    predictor = _predictor(system0)
    transform = spherical_transform_for(system0)

    batched = predict_spherical_fock_batch(predictor, atoms, positions, transform)
    assert batched.shape == (3, 24, 24)
    single = predict_spherical_fock(
        predictor,
        MolecularSystem(atomic_numbers=atoms, positions=positions[1], basis_set="def2-svp"),
        transform,
    )
    # The vmap path batches the trunk matmuls, which XLA may dispatch at a
    # slightly different precision than the single-molecule path (GPU float32 /
    # TF32); the predictions agree to a physically-meaningful tolerance.
    np.testing.assert_allclose(np.asarray(batched[1]), np.asarray(single), atol=5e-3)


def test_batched_train_step_is_jit_and_grad_clean() -> None:
    """One batched train step runs under jit with finite loss and gradients."""
    atoms = _water_atoms()
    positions = _water_positions_batch(batch=4)
    system0 = MolecularSystem(atomic_numbers=atoms, positions=positions[0], basis_set="def2-svp")
    predictor = _predictor(system0, seed=0)
    transform = spherical_transform_for(system0)
    targets = _teacher_targets_batch(atoms, positions, seed=7)
    mask = jnp.ones_like(targets)

    optimizer = nnx.Optimizer(predictor, optax.adam(1e-2), wrt=nnx.Param)
    train_step = make_batched_train_step(
        transform, atoms, loss_kind="mse", property_name="hamiltonian"
    )
    loss = train_step(predictor, optimizer, positions, targets, mask)
    assert jnp.isfinite(loss)


def test_batched_fit_decreases_loss_on_overfit_bucket() -> None:
    """A batched fit over a same-signature water bucket reduces the summed loss."""
    atoms = _water_atoms()
    positions = _water_positions_batch(batch=4)
    targets = _teacher_targets_batch(atoms, positions, seed=7)
    mask = jnp.ones_like(targets)

    system0 = MolecularSystem(atomic_numbers=atoms, positions=positions[0], basis_set="def2-svp")
    student = _predictor(system0, seed=0)

    result = fit_qh9_bucket(
        student,
        atoms,
        positions,
        targets,
        mask=mask,
        config=QH9TrainConfig(learning_rate=1e-2, num_steps=150, loss_kind="mse"),
    )
    history = np.asarray(result.loss_history)
    assert history.shape == (150,)
    assert np.all(np.isfinite(history))
    assert result.final_loss < float(history[0])
    assert float(history[-1]) < float(history[len(history) // 2]) < float(history[0])
    assert result.final_loss < 0.5 * float(history[0])
