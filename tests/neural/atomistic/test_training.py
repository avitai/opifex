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
import pytest
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.protocols import RadiusNeighborList
from opifex.core.training import OptimizerConfig
from opifex.core.training.optimizers import create_optimizer
from opifex.neural.atomistic import AtomisticModel
from opifex.neural.atomistic.heads import EnergyHead, ForcesHead
from opifex.neural.atomistic.training import (
    _ema_blend,
    AtomisticBatch,
    energy_forces_loss,
    fit_atomistic,
    make_atomistic_train_step,
    make_scanned_epoch,
    ParamEMA,
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

    def test_from_arrays_matches_from_systems(self) -> None:
        """The array-batch path mirrors the system path for stacked loader data."""
        systems, energies, forces = _synthetic_dataset(num_configs=4)
        positions = jnp.stack([system.positions for system in systems])
        from_systems = AtomisticBatch.from_systems(systems, energies, forces)
        from_arrays = AtomisticBatch.from_arrays(positions, _ATOMIC_NUMBERS, energies, forces)
        assert jnp.array_equal(from_arrays.positions, from_systems.positions)
        assert jnp.array_equal(from_arrays.atomic_numbers, from_systems.atomic_numbers)
        assert jnp.array_equal(from_arrays.energies, from_systems.energies)
        assert jnp.array_equal(from_arrays.forces, from_systems.forces)

    def test_from_arrays_rejects_unbatched_positions(self) -> None:
        """A missing leading batch axis fails fast."""
        single = jnp.zeros((3, 3))
        with pytest.raises(ValueError, match="batch >= 1"):
            AtomisticBatch.from_arrays(
                single, _ATOMIC_NUMBERS, jnp.zeros((1,)), jnp.zeros((1, 3, 3))
            )


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

    def test_device_accumulation_matches_per_step_host_sync(self) -> None:
        """The once-per-epoch device accumulation equals per-step ``float`` syncing.

        ``fit_atomistic`` keeps the per-step loss on device and syncs to host
        once per epoch (avoiding ~1 ms/step of host blocking under JAX async
        dispatch). This must be numerically identical to the previous behaviour
        of ``float(train_step(...))`` per step, accumulated on the host. Both
        paths run from an identically-seeded model + optimiser over the same
        batches, so their per-epoch loss histories must match to float precision.
        """
        systems, energies, forces = _synthetic_dataset()
        batches = [
            AtomisticBatch.from_systems(systems[:3], energies[:3], forces[:3]),
            AtomisticBatch.from_systems(systems[3:], energies[3:], forces[3:]),
        ]
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)
        num_epochs = 8

        # Library path: device accumulation, one host sync per epoch.
        device_history = fit_atomistic(_build_model(), batches, config, num_epochs=num_epochs)

        # Reference path: per-step host sync (the previous behaviour), same init.
        reference_model = _build_model()
        reference_optimizer = nnx.Optimizer(
            reference_model, create_optimizer(config), wrt=nnx.Param
        )
        reference_step = make_atomistic_train_step(reference_model, reference_optimizer)
        reference_history: list[float] = []
        for _ in range(num_epochs):
            epoch_losses = [
                float(reference_step(reference_model, reference_optimizer, batch))
                for batch in batches
            ]
            reference_history.append(sum(epoch_losses) / len(epoch_losses))

        assert len(device_history) == len(reference_history)
        for device_loss, reference_loss in zip(device_history, reference_history, strict=True):
            assert device_loss == pytest.approx(reference_loss, rel=1e-5, abs=1e-6)


def _param_leaves(model: AtomisticModel) -> list[Array]:
    """Flatten a model's ``nnx.Param`` state into a deterministic leaf list."""
    return jax.tree.leaves(nnx.state(model, nnx.Param))


class TestParamEMA:
    """The reusable EMA-of-parameters primitive (NequIP/MACE eval convention)."""

    def test_rejects_decay_out_of_range(self) -> None:
        """A decay outside ``[0, 1)`` fails fast."""
        model = _build_model()
        with pytest.raises(ValueError, match=r"decay must be in"):
            ParamEMA(model, decay=1.0)

    def test_update_matches_ema_recurrence(self) -> None:
        """``update`` realises ``ema = d*ema + (1-d)*params`` over every leaf.

        After two manual optimiser steps the shadow must equal the hand-computed
        EMA of the three parameter snapshots (initial, post-step-1, post-step-2).
        """
        decay = 0.9
        model = _build_model()
        optimizer = nnx.Optimizer(model, optax.adam(5e-2), wrt=nnx.Param)
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        train_step = make_atomistic_train_step(model, optimizer)

        params0 = _param_leaves(model)
        ema = ParamEMA(model, decay=decay)
        # Shadow is seeded with the initial params.
        for shadow, theta0 in zip(jax.tree.leaves(ema.state), params0, strict=True):
            assert jnp.allclose(shadow, theta0)

        train_step(model, optimizer, batch)
        params1 = _param_leaves(model)
        ema.update(model)
        train_step(model, optimizer, batch)
        params2 = _param_leaves(model)
        ema.update(model)

        # ema1 = d*params0 + (1-d)*params1 ; ema2 = d*ema1 + (1-d)*params2.
        for theta0, theta1, theta2, shadow in zip(
            params0, params1, params2, jax.tree.leaves(ema.state), strict=True
        ):
            expected1 = decay * theta0 + (1.0 - decay) * theta1
            expected2 = decay * expected1 + (1.0 - decay) * theta2
            assert jnp.allclose(shadow, expected2, atol=1e-6)

    def test_copy_to_loads_shadow_into_model(self) -> None:
        """``copy_to`` overwrites the live params with the EMA shadow."""
        decay = 0.8
        model = _build_model()
        optimizer = nnx.Optimizer(model, optax.adam(5e-2), wrt=nnx.Param)
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        train_step = make_atomistic_train_step(model, optimizer)

        ema = ParamEMA(model, decay=decay)
        for _ in range(5):
            train_step(model, optimizer, batch)
            ema.update(model)
        shadow_leaves = jax.tree.leaves(ema.state)

        ema.copy_to(model)
        for live, shadow in zip(_param_leaves(model), shadow_leaves, strict=True):
            assert jnp.allclose(live, shadow)

    def test_swap_in_restores_live_params(self) -> None:
        """``swap_in`` evaluates with EMA weights then restores the live ones."""
        decay = 0.95
        model = _build_model()
        optimizer = nnx.Optimizer(model, optax.adam(5e-2), wrt=nnx.Param)
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        train_step = make_atomistic_train_step(model, optimizer)

        ema = ParamEMA(model, decay=decay)
        for _ in range(5):
            train_step(model, optimizer, batch)
            ema.update(model)
        live_before = _param_leaves(model)
        shadow_leaves = jax.tree.leaves(ema.state)

        with ema.swap_in(model):
            inside = _param_leaves(model)
            for current, shadow in zip(inside, shadow_leaves, strict=True):
                assert jnp.allclose(current, shadow)
        # The live params are restored on exit.
        for after, before in zip(_param_leaves(model), live_before, strict=True):
            assert jnp.allclose(after, before)

    def test_update_uses_jitted_pure_arithmetic(self) -> None:
        """``update`` delegates the EMA blend to the jitted, pure ``_ema_blend``.

        The per-step blend is side-effect-free pytree arithmetic, so the same
        result is reproducible by calling the underlying jitted function directly
        on the shadow + live state -- proving the cheap path is jit-compiled.
        """
        decay = 0.99
        model = _build_model()
        optimizer = nnx.Optimizer(model, optax.adam(5e-2), wrt=nnx.Param)
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        train_step = make_atomistic_train_step(model, optimizer)
        ema = ParamEMA(model, decay=decay)

        shadow_before = ema.state
        train_step(model, optimizer, batch)
        live_state = nnx.state(model, nnx.Param)
        expected = _ema_blend(shadow_before, live_state, jnp.asarray(decay))
        ema.update(model)

        for got, want in zip(jax.tree.leaves(ema.state), jax.tree.leaves(expected), strict=True):
            assert jnp.allclose(got, want, atol=1e-6)
            assert bool(jnp.all(jnp.isfinite(got)))

    def test_eval_under_jit_with_ema_params(self) -> None:
        """A jitted forward pass against the EMA weights gives a finite loss."""
        decay = 0.99
        model = _build_model()
        optimizer = nnx.Optimizer(model, optax.adam(5e-2), wrt=nnx.Param)
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        train_step = make_atomistic_train_step(model, optimizer)
        ema = ParamEMA(model, decay=decay)
        for _ in range(8):
            train_step(model, optimizer, batch)
            ema.update(model)

        @nnx.jit
        def eval_loss(m: AtomisticModel, b: AtomisticBatch) -> Array:
            return energy_forces_loss(m, systems, b.energies, b.forces, force_weight=1.0)

        with ema.swap_in(model):
            ema_loss = float(eval_loss(model, batch))
        assert jnp.isfinite(ema_loss)


class TestFitAtomisticEMA:
    """``fit_atomistic`` EMA integration (opt-in ``ema_decay``)."""

    def test_ema_none_leaves_raw_weights(self) -> None:
        """``ema_decay=None`` reproduces the no-EMA behaviour exactly."""
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)

        baseline = _build_model()
        history = fit_atomistic(baseline, [batch], config, num_epochs=10)

        explicit_none = _build_model()
        history_none = fit_atomistic(explicit_none, [batch], config, num_epochs=10, ema_decay=None)

        assert len(history) == 10
        for raw, none in zip(_param_leaves(baseline), _param_leaves(explicit_none), strict=True):
            assert jnp.allclose(raw, none)
        for loss_a, loss_b in zip(history, history_none, strict=True):
            assert loss_a == pytest.approx(loss_b, rel=1e-6, abs=1e-7)

    def test_ema_path_differs_from_raw_and_is_finite(self) -> None:
        """With ``ema_decay`` set, the model ends on smoothed (different) weights.

        The same seed and batches are trained twice -- once raw, once with EMA --
        so the only difference is that the EMA run loads the averaged weights on
        return; those must differ from the raw last-step weights yet stay finite.
        """
        systems, energies, forces = _synthetic_dataset()
        batch = AtomisticBatch.from_systems(systems, energies, forces)
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)

        raw_model = _build_model()
        fit_atomistic(raw_model, [batch], config, num_epochs=12)

        ema_model = _build_model()
        history = fit_atomistic(ema_model, [batch], config, num_epochs=12, ema_decay=0.9)

        assert len(history) == 12
        raw_leaves = _param_leaves(raw_model)
        ema_leaves = _param_leaves(ema_model)
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in ema_leaves)
        # At least one leaf must differ -- the EMA lags the noisy last-step weights.
        assert any(
            not bool(jnp.allclose(raw, ema, atol=1e-5))
            for raw, ema in zip(raw_leaves, ema_leaves, strict=True)
        )


class TestAtomisticBatchStack:
    """``AtomisticBatch.stack`` builds the leading-step-axis pytree for ``lax.scan``."""

    def test_stack_adds_leading_step_axis(self) -> None:
        systems, energies, forces = _synthetic_dataset(num_configs=6)
        batches = [
            AtomisticBatch.from_systems(systems[:3], energies[:3], forces[:3]),
            AtomisticBatch.from_systems(systems[3:], energies[3:], forces[3:]),
        ]
        stacked = AtomisticBatch.stack(batches)
        # Two steps, each a batch of 3 water configs.
        assert stacked.positions.shape == (2, 3, 3, 3)
        assert stacked.energies.shape == (2, 3)
        assert stacked.forces.shape == (2, 3, 3, 3)
        # ``atomic_numbers`` is shared, not stacked.
        assert stacked.atomic_numbers.shape == (3,)
        assert jnp.array_equal(stacked.atomic_numbers, _ATOMIC_NUMBERS)

    def test_stack_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one batch"):
            AtomisticBatch.stack([])

    def test_stack_rejects_mismatched_composition(self) -> None:
        systems, energies, forces = _synthetic_dataset(num_configs=4)
        good = AtomisticBatch.from_systems(systems[:2], energies[:2], forces[:2])
        other = AtomisticBatch.from_arrays(
            jnp.stack([s.positions for s in systems[2:]]),
            jnp.asarray([7, 1, 1]),  # different composition
            energies[2:],
            forces[2:],
        )
        with pytest.raises(ValueError, match="same composition"):
            AtomisticBatch.stack([good, other])

    def test_stack_rejects_mismatched_shape(self) -> None:
        systems, energies, forces = _synthetic_dataset(num_configs=5)
        a = AtomisticBatch.from_systems(systems[:2], energies[:2], forces[:2])
        b = AtomisticBatch.from_systems(systems[2:], energies[2:], forces[2:])  # batch 3 != 2
        with pytest.raises(ValueError, match="equal-shaped"):
            AtomisticBatch.stack([a, b])


class TestScannedEpochCorrectness:
    """The fused ``lax.scan`` epoch is bit-comparable to the per-step Python loop.

    The fused path (:func:`make_scanned_epoch` / ``fit_atomistic(fused=True)``)
    exists purely to remove per-step host->device dispatch gaps that starve the
    GPU; it must change *no* math. These tests pin that the per-epoch loss
    trajectory, the EMA shadow and the final weights match the explicit per-step
    loop, and that the second-order conservative-force gradient still flows inside
    the scan.
    """

    def _two_batches(self) -> list[AtomisticBatch]:
        systems, energies, forces = _synthetic_dataset(num_configs=6)
        return [
            AtomisticBatch.from_systems(systems[:3], energies[:3], forces[:3]),
            AtomisticBatch.from_systems(systems[3:], energies[3:], forces[3:]),
        ]

    def test_fused_loss_trajectory_matches_per_step(self) -> None:
        """Fused and per-step epoch loss histories match to float precision."""
        batches = self._two_batches()
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)
        num_epochs = 6

        per_step = fit_atomistic(
            _build_model(), batches, config, num_epochs=num_epochs, force_weight=1.0, fused=False
        )
        fused = fit_atomistic(
            _build_model(), batches, config, num_epochs=num_epochs, force_weight=1.0, fused=True
        )
        assert len(fused) == len(per_step)
        for fused_loss, per_step_loss in zip(fused, per_step, strict=True):
            assert fused_loss == pytest.approx(per_step_loss, rel=1e-5, abs=1e-6)

    def test_fused_final_weights_match_per_step(self) -> None:
        """The fused path leaves identical (raw) weights to the per-step loop."""
        batches = self._two_batches()
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)

        per_step_model = _build_model()
        fit_atomistic(per_step_model, batches, config, num_epochs=6, force_weight=1.0, fused=False)
        fused_model = _build_model()
        fit_atomistic(fused_model, batches, config, num_epochs=6, force_weight=1.0, fused=True)

        for per_step_leaf, fused_leaf in zip(
            _param_leaves(per_step_model), _param_leaves(fused_model), strict=True
        ):
            assert jnp.allclose(per_step_leaf, fused_leaf, rtol=1e-5, atol=1e-6)

    def test_fused_ema_weights_match_per_step(self) -> None:
        """The EMA shadow fused inside the scan matches the per-step EMA exactly."""
        batches = self._two_batches()
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)
        decay = 0.9

        per_step_model = _build_model()
        fit_atomistic(
            per_step_model,
            batches,
            config,
            num_epochs=6,
            force_weight=1.0,
            ema_decay=decay,
            fused=False,
        )
        fused_model = _build_model()
        fit_atomistic(
            fused_model,
            batches,
            config,
            num_epochs=6,
            force_weight=1.0,
            ema_decay=decay,
            fused=True,
        )
        for per_step_leaf, fused_leaf in zip(
            _param_leaves(per_step_model), _param_leaves(fused_model), strict=True
        ):
            assert jnp.allclose(per_step_leaf, fused_leaf, rtol=1e-5, atol=1e-6)

    def test_scanned_epoch_force_grad_is_finite_and_reduces(self) -> None:
        """The grad-of-grad force term works inside the scan (finite, decreasing)."""
        batches = self._two_batches()
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)
        history = fit_atomistic(
            _build_model(), batches, config, num_epochs=12, force_weight=1.0, fused=True
        )
        assert all(bool(jnp.isfinite(jnp.asarray(loss))) for loss in history)
        assert history[-1] < history[0]

    def test_scanned_epoch_is_jit_and_returns_per_step_losses(self) -> None:
        """``make_scanned_epoch`` is jitted and returns one loss per step."""
        batches = self._two_batches()
        config = OptimizerConfig(optimizer_type="adam", learning_rate=1e-2)
        model = _build_model()
        optimizer = nnx.Optimizer(model, create_optimizer(config), wrt=nnx.Param)
        scanned = make_scanned_epoch(model, optimizer, force_weight=1.0)
        stacked = AtomisticBatch.stack(batches)

        ema_state, losses = scanned(model, optimizer, stacked, None)
        assert ema_state is None
        assert losses.shape == (len(batches),)
        assert bool(jnp.all(jnp.isfinite(losses)))
