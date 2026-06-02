r"""Tests for the energy+forces MLIP training objective and step.

The reusable :mod:`opifex.neural.atomistic.training` provides the dict-output /
``MolecularSystem``-input training capability that the array-based
:class:`opifex.core.training.Trainer` cannot express. The combined energy +
forces objective is the standard MLIP loss (Batzner et al. 2022, NequIP,
arXiv:2101.03164; the ``../mace`` weighted-energy-forces loss), and the step
mirrors the ``examples/quantum-chemistry/neural_xc_functional.py``
``nnx.value_and_grad`` + ``optimizer.update`` idiom.

Load-bearing checks:

* the loss is differentiable w.r.t. model params (grad-of-grad through the
  conservative :class:`ForcesHead`);
* a few jitted steps REDUCE the loss on a synthetic pairwise-potential dataset;
* ``force_weight=0`` reduces the objective to energy-only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.protocols import RadiusNeighborList
from opifex.core.training import OptimizerConfig
from opifex.neural.atomistic import AtomisticModel
from opifex.neural.atomistic.heads import EnergyHead, ForcesHead
from opifex.neural.atomistic.training import (
    AtomisticBatch,
    energy_forces_loss,
    fit_atomistic,
    make_atomistic_train_step,
)
from opifex.neural.equivariant import scatter_sum


_ATOMIC_NUMBERS = jnp.asarray([8, 1, 1])


class _ToyBackbone(nnx.Module):
    """Minimal invariant backbone (cutoff-weighted coordination MLP)."""

    def __init__(self, *, hidden: int = 8, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.cutoff = 3.0
        self.linear_in = nnx.Linear(1, hidden, rngs=rngs)
        self.linear_out = nnx.Linear(hidden, hidden, rngs=rngs)

    def __call__(self, system: MolecularSystem, graph: tuple[Array, Array]) -> dict[str, Array]:
        senders, receivers = graph
        valid = senders >= 0
        safe_senders = jnp.where(valid, senders, 0)
        safe_receivers = jnp.where(valid, receivers, 0)
        deltas = system.positions[safe_senders] - system.positions[safe_receivers]
        distances = jnp.linalg.norm(deltas + 1e-12, axis=-1)
        envelope = jnp.where(valid, jnp.clip(1.0 - distances / self.cutoff, 0.0, None), 0.0)
        coordination = scatter_sum(envelope[:, None], safe_receivers, num_segments=system.n_atoms)
        features = nnx.tanh(self.linear_in(coordination))
        return {"node_features": nnx.tanh(self.linear_out(features))}


def _build_model() -> AtomisticModel:
    rngs = nnx.Rngs(0)
    backbone = _ToyBackbone(rngs=rngs)
    heads: dict[str, object] = {
        "energy": EnergyHead(feature_dim=8, rngs=rngs),
        "forces": ForcesHead(),
    }
    return AtomisticModel(
        backbone=backbone,
        heads=heads,  # type: ignore[arg-type]
        neighbor_list=RadiusNeighborList(cutoff=3.0),
        max_edges=32,
    )


def _pairwise_energy(positions: Array) -> Array:
    """Analytic pairwise Morse-like well energy (the synthetic target)."""
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = jnp.sqrt(jnp.sum(deltas**2, axis=-1) + 1e-9)
    upper = jnp.triu(jnp.ones_like(distances), k=1)
    well = (1.0 - jnp.exp(-(distances - 1.0))) ** 2
    return jnp.sum(well * upper)


def _synthetic_dataset(
    num_configs: int = 6,
) -> tuple[list[MolecularSystem], Array, Array]:
    """Configs + analytic energies + analytic forces around a water geometry."""
    base = jnp.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    key = jax.random.PRNGKey(0)
    perturbations = 0.1 * jax.random.normal(key, (num_configs, *base.shape))
    geometries = base[None] + perturbations
    energies = jax.vmap(_pairwise_energy)(geometries)
    forces = jax.vmap(lambda p: -jax.grad(_pairwise_energy)(p))(geometries)
    systems = [
        MolecularSystem(atomic_numbers=_ATOMIC_NUMBERS, positions=geometries[i])
        for i in range(num_configs)
    ]
    return systems, energies, forces


class TestAtomisticBatch:
    def test_from_systems_stacks_positions(self) -> None:
        systems, energies, forces = _synthetic_dataset(num_configs=4)
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        assert batch.positions.shape == (4, 3, 3)
        assert batch.atomic_numbers.shape == (3,)
        assert batch.energies.shape == (4,)
        assert batch.forces.shape == (4, 3, 3)


class TestEnergyForcesLoss:
    def test_loss_is_non_negative_scalar(self) -> None:
        model = _build_model()
        systems, energies, forces = _synthetic_dataset()
        loss = energy_forces_loss(model, systems, energies, forces)
        assert loss.shape == ()
        assert float(loss) >= 0.0

    def test_force_weight_zero_is_energy_only(self) -> None:
        """``force_weight=0`` drops the force term entirely."""
        model = _build_model()
        systems, energies, forces = _synthetic_dataset()
        energy_only = energy_forces_loss(
            model, systems, energies, forces, energy_weight=1.0, force_weight=0.0
        )
        wrong_forces = forces + 100.0
        energy_only_wrong = energy_forces_loss(
            model, systems, energies, wrong_forces, energy_weight=1.0, force_weight=0.0
        )
        assert jnp.allclose(energy_only, energy_only_wrong, atol=1e-6)

    def test_force_term_responds_to_force_targets(self) -> None:
        """With ``force_weight>0`` the loss depends on the force targets."""
        model = _build_model()
        systems, energies, forces = _synthetic_dataset()
        good = energy_forces_loss(model, systems, energies, forces, force_weight=1.0)
        bad = energy_forces_loss(model, systems, energies, forces + 50.0, force_weight=1.0)
        assert float(bad) > float(good)

    def test_loss_is_differentiable_through_forces(self) -> None:
        """Grad-of-grad: params gradient flows through the ForcesHead autodiff."""
        model = _build_model()
        systems, energies, forces = _synthetic_dataset()
        graphdef, state = nnx.split(model)

        def loss_of_state(state: nnx.State) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            return energy_forces_loss(rebuilt, systems, energies, forces, force_weight=1.0)

        grads = jax.grad(loss_of_state)(state)
        leaves = jax.tree.leaves(grads)
        assert leaves
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
        assert any(float(jnp.sum(jnp.abs(leaf))) > 0.0 for leaf in leaves)


class TestMakeAtomisticTrainStep:
    def test_step_reduces_loss(self) -> None:
        model = _build_model()
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        train_step = make_atomistic_train_step(model, optimizer)
        initial = float(train_step(model, optimizer, batch))
        last = initial
        for _ in range(15):
            last = float(train_step(model, optimizer, batch))
        assert last < initial

    def test_step_returns_finite_loss(self) -> None:
        model = _build_model()
        optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
        train_step = make_atomistic_train_step(model, optimizer)
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        loss = train_step(model, optimizer, batch)
        assert jnp.isfinite(loss)


class TestFitAtomistic:
    def test_fit_reduces_loss(self) -> None:
        model = _build_model()
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)
        initial = float(energy_forces_loss(model, systems, energies, forces))
        history = fit_atomistic(model, [batch], config, num_epochs=15)
        final = float(energy_forces_loss(model, systems, energies, forces))
        assert len(history) == 15
        assert final < initial
