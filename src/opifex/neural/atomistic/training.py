r"""Energy + forces training objective and step for atomistic models (MLIPs).

The array-based :class:`opifex.core.training.Trainer` consumes plain ``(x, y)``
array batches and a single model output, so it cannot express a machine-learning
interatomic potential: an MLIP maps a :class:`~opifex.core.quantum.\
molecular_system.MolecularSystem` to a *dict* of outputs (``"energy"`` and the
conservative ``"forces"`` = ``-grad(E)``) and is trained against *both*. This
module supplies that capability, reusing the project's optimiser stack
(:func:`opifex.core.training.optimizers.create_optimizer` + ``nnx.Optimizer``)
and the ``nnx.value_and_grad`` + ``optimizer.update`` step idiom from
``examples/quantum-chemistry/neural_xc_functional.py``.

The combined objective is the standard weighted energy + forces loss

.. math:: \mathcal{L} = w_E \, \operatorname{MSE}(E) + w_F \, \operatorname{MSE}(F)

(Batzner et al. 2022, NequIP, arXiv:2101.03164; the ``../mace``
``WeightedEnergyForcesLoss``). Because the forces are themselves a gradient of
the energy, fitting forces trains the model through second-order autodiff
(grad-of-grad), which the backbones are tested ``jit``/``grad``/``vmap``-clean
for.

Batches are :class:`AtomisticBatch` -- a stacked-array JAX PyTree -- so the whole
training step is ``nnx.jit``-compatible (a ``MolecularSystem`` is itself a single
opaque leaf and cannot be batched directly).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: TC003

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Float  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig
from opifex.neural.atomistic.base import AtomisticModel  # noqa: TC001


@struct.dataclass(frozen=True, kw_only=True)
class AtomisticBatch:
    """A stacked-array batch of equal-size configurations with energy+force labels.

    All configurations share one ``atomic_numbers`` vector (same composition and
    atom ordering), so positions, energies and forces stack along a leading batch
    axis and the whole batch is a JAX PyTree traceable under ``jit``/``vmap``.

    Attributes:
        positions: Atomic positions, shape ``(batch, n_atoms, 3)``.
        atomic_numbers: Shared nuclear charges, shape ``(n_atoms,)``.
        energies: Reference total energies, shape ``(batch,)``.
        forces: Reference forces, shape ``(batch, n_atoms, 3)``.
    """

    positions: Float[Array, "batch n_atoms 3"]
    atomic_numbers: Array
    energies: Float[Array, " batch"]
    forces: Float[Array, "batch n_atoms 3"]

    @classmethod
    def from_systems(
        cls,
        systems: Sequence[MolecularSystem],
        energies: Float[Array, " batch"],
        forces: Float[Array, "batch n_atoms 3"],
    ) -> AtomisticBatch:
        """Stack a sequence of same-composition systems into an array batch.

        Args:
            systems: Configurations sharing one composition / atom ordering.
            energies: Reference total energies, shape ``(batch,)``.
            forces: Reference forces, shape ``(batch, n_atoms, 3)``.

        Returns:
            The stacked :class:`AtomisticBatch`.

        Raises:
            ValueError: If ``systems`` is empty.
        """
        if not systems:
            raise ValueError("AtomisticBatch.from_systems requires at least one system.")
        positions = jnp.stack([system.positions for system in systems])
        return cls(
            positions=positions,
            atomic_numbers=systems[0].atomic_numbers,
            energies=jnp.asarray(energies),
            forces=jnp.asarray(forces),
        )


def _batch_loss(
    model: AtomisticModel,
    batch: AtomisticBatch,
    *,
    energy_weight: float,
    force_weight: float,
) -> Array:
    """Weighted energy + forces MSE over a stacked :class:`AtomisticBatch`."""

    def predict(positions: Array) -> tuple[Array, Array]:
        system = MolecularSystem(atomic_numbers=batch.atomic_numbers, positions=positions)
        outputs = model(system)
        return outputs["energy"], outputs["forces"]

    predicted_energies, predicted_forces = jax.vmap(predict)(batch.positions)
    energy_mse = jnp.mean((predicted_energies - batch.energies) ** 2)
    force_mse = jnp.mean((predicted_forces - batch.forces) ** 2)
    return energy_weight * energy_mse + force_weight * force_mse


def energy_forces_loss(
    model: AtomisticModel,
    systems: Sequence[MolecularSystem],
    energies: Float[Array, " batch"],
    forces: Float[Array, "batch n_atoms 3"],
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
) -> Array:
    r"""Weighted energy + forces MSE for a batch of molecular systems.

    Computes ``energy_weight * MSE(E) + force_weight * MSE(F)`` by running the
    model on each configuration (via :func:`jax.vmap`) and comparing both the
    total energy and the conservative forces against the references. The loss is
    differentiable w.r.t. the model parameters (the force term differentiates
    through the :class:`~opifex.neural.atomistic.heads.forces.ForcesHead` autodiff,
    i.e. grad-of-grad).

    Args:
        model: The assembled :class:`AtomisticModel` to evaluate.
        systems: Same-composition configurations to score.
        energies: Reference total energies, shape ``(batch,)``.
        forces: Reference forces, shape ``(batch, n_atoms, 3)``.
        energy_weight: Weight ``w_E`` of the energy MSE term.
        force_weight: Weight ``w_F`` of the forces MSE term (``0`` -> energy-only).

    Returns:
        The scalar weighted loss.
    """
    batch = AtomisticBatch.from_systems(systems, energies, forces)
    return _batch_loss(model, batch, energy_weight=energy_weight, force_weight=force_weight)


def make_atomistic_train_step(
    model: AtomisticModel,
    optimizer: nnx.Optimizer,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
) -> Callable[[AtomisticModel, nnx.Optimizer, AtomisticBatch], Array]:
    """Build a jitted energy+forces training step.

    The returned ``@nnx.jit`` step computes the weighted loss and its gradient
    with :func:`flax.nnx.value_and_grad`, applies the update in place via
    ``optimizer.update`` and returns the (pre-update) loss -- the
    ``examples/quantum-chemistry/neural_xc_functional.py`` idiom.

    Args:
        model: The model to optimise (its graphdef pins the static structure).
        optimizer: An ``nnx.Optimizer`` wrapping ``model`` (``wrt=nnx.Param``).
        energy_weight: Weight of the energy MSE term.
        force_weight: Weight of the forces MSE term.

    Returns:
        A callable ``(model, optimizer, batch) -> loss``.
    """
    del model, optimizer  # captured by the caller; the step is parametric in them.

    def loss_fn(model: AtomisticModel, batch: AtomisticBatch) -> Array:
        return _batch_loss(model, batch, energy_weight=energy_weight, force_weight=force_weight)

    @nnx.jit
    def train_step(model: AtomisticModel, optimizer: nnx.Optimizer, batch: AtomisticBatch) -> Array:
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        optimizer.update(model, grads)
        return loss

    return train_step


def fit_atomistic(
    model: AtomisticModel,
    batches: Sequence[AtomisticBatch],
    optimizer_config: OptimizerConfig,
    num_epochs: int,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
) -> list[float]:
    """Train an atomistic model for ``num_epochs`` over the given batches.

    Builds the optimiser from ``optimizer_config`` with
    :func:`opifex.core.training.optimizers.create_optimizer`, wraps it in an
    ``nnx.Optimizer`` and runs the jitted :func:`make_atomistic_train_step` over
    every batch each epoch. The model is updated in place.

    Args:
        model: The model to train in place.
        batches: The training batches (cycled once per epoch).
        optimizer_config: Optimiser configuration.
        num_epochs: Number of passes over ``batches``.
        energy_weight: Weight of the energy MSE term.
        force_weight: Weight of the forces MSE term.

    Returns:
        The mean training loss per epoch (length ``num_epochs``).
    """
    optimizer = nnx.Optimizer(model, create_optimizer(optimizer_config), wrt=nnx.Param)
    train_step = make_atomistic_train_step(
        model, optimizer, energy_weight=energy_weight, force_weight=force_weight
    )
    history: list[float] = []
    for _ in range(num_epochs):
        epoch_losses = [float(train_step(model, optimizer, batch)) for batch in batches]
        history.append(sum(epoch_losses) / len(epoch_losses))
    return history


__all__ = [
    "AtomisticBatch",
    "energy_forces_loss",
    "fit_atomistic",
    "make_atomistic_train_step",
]
