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

from collections.abc import Callable, Iterator, Sequence  # noqa: TC003
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Float  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.training.optimizers import create_optimizer, OptimizerConfig
from opifex.neural.atomistic.base import AtomisticModel  # noqa: TC001


@jax.jit
def _ema_blend(ema_state: nnx.State, live_state: nnx.State, decay: Array) -> nnx.State:
    """Pure ``ema = decay * ema + (1 - decay) * live`` over a Param-state pytree.

    Factored out as a side-effect-free, ``jit``-compiled function so the per-step
    EMA arithmetic is cheap and traces cleanly; :meth:`ParamEMA.update` only does
    the (Python-side) reassignment of the returned shadow.
    """
    return jax.tree.map(lambda ema, live: decay * ema + (1.0 - decay) * live, ema_state, live_state)


class ParamEMA:
    r"""Exponential moving average (EMA) of a model's :class:`~flax.nnx.Param` state.

    Validation and inference for machine-learning interatomic potentials are
    standardly run against an EMA of the weights rather than the noisy last-step
    weights: NequIP exposes an ``ema_decay`` hyper-parameter and MACE wraps the
    model in ``torch_ema.ExponentialMovingAverage`` (``mace/tools/train.py``),
    both defaulting to ``decay = 0.99`` (NequIP configs; the MACE
    ``--ema_decay`` argument, ``mace/tools/arg_parser.py``). The shadow weights
    track

    .. math:: \theta_{\text{ema}} \leftarrow d\,\theta_{\text{ema}}
              + (1 - d)\,\theta

    where :math:`d` is the decay and :math:`\theta` the live parameters.

    The EMA shadow is a plain :class:`flax.nnx.State` pytree of the model's
    ``nnx.Param`` leaves, so :meth:`update` is pure pytree arithmetic and stays
    cheap and ``jit``-friendly when called inside a jitted training loop.

    Attributes:
        decay: The EMA decay :math:`d \in [0, 1)`; higher decays average over a
            longer window. ``0.99``-``0.999`` is the MLIP convention.
    """

    def __init__(self, model: AtomisticModel, *, decay: float) -> None:
        r"""Initialise the EMA shadow from the model's current parameters.

        Args:
            model: The model whose ``nnx.Param`` leaves are tracked. The shadow
                is seeded with a copy of the current parameter values.
            decay: The EMA decay :math:`d \in [0, 1)`.

        Raises:
            ValueError: If ``decay`` is not in ``[0, 1)``.
        """
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ParamEMA decay must be in [0, 1), got {decay}.")
        self.decay = decay
        self._decay_array = jnp.asarray(decay)
        self._ema_state: nnx.State = jax.tree.map(jnp.asarray, nnx.state(model, nnx.Param))

    @property
    def state(self) -> nnx.State:
        """The current EMA shadow :class:`~flax.nnx.State` (the averaged params)."""
        return self._ema_state

    def update(self, model: AtomisticModel) -> None:
        """Blend the model's live parameters into the EMA shadow in place.

        Computes ``ema = decay * ema + (1 - decay) * params`` over every matching
        ``nnx.Param`` leaf via the jitted, side-effect-free :func:`_ema_blend`;
        only the (Python-side) reassignment of the new shadow happens here, so
        the arithmetic is cheap and the method must not itself be wrapped in
        ``jit`` (the attribute write is a side effect).

        Args:
            model: The model whose current parameters update the shadow.
        """
        live_state = nnx.state(model, nnx.Param)
        self._ema_state = _ema_blend(self._ema_state, live_state, self._decay_array)

    def copy_to(self, model: AtomisticModel) -> None:
        """Overwrite the model's parameters with the EMA shadow (in place).

        Args:
            model: The model to load the averaged parameters into.
        """
        nnx.update(model, self._ema_state)

    @contextmanager
    def swap_in(self, model: AtomisticModel) -> Iterator[None]:
        """Temporarily evaluate ``model`` with the EMA params, then restore.

        Mirrors MACE's ``ema.average_parameters()`` context
        (``mace/tools/train.py``): the raw (live) parameters are saved, the EMA
        shadow is loaded for the duration of the ``with`` block, and the live
        parameters are restored on exit so subsequent training is unaffected.

        Args:
            model: The model to evaluate with EMA weights inside the block.

        Yields:
            ``None``; use the block to run evaluation against the EMA weights.
        """
        saved_state = jax.tree.map(jnp.asarray, nnx.state(model, nnx.Param))
        try:
            self.copy_to(model)
            yield
        finally:
            nnx.update(model, saved_state)


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

    @classmethod
    def stack(cls, batches: Sequence[AtomisticBatch]) -> AtomisticBatch:
        """Stack equal-shaped batches along a new leading axis for ``lax.scan``.

        The fused-epoch training path (:func:`make_scanned_epoch`) scans the
        training step over an epoch's batches, so the per-batch
        :class:`AtomisticBatch` leaves must be stacked into one pytree carrying a
        leading ``num_steps`` axis. Every batch must share the same
        ``atomic_numbers`` and per-batch shapes (same composition and batch size),
        which is the invariant the dataset loaders already satisfy.

        Args:
            batches: The per-step batches of one epoch; all of equal shape and
                composition.

        Returns:
            One :class:`AtomisticBatch` whose ``positions``/``energies``/``forces``
            gain a leading ``num_steps`` axis and whose ``atomic_numbers`` is the
            shared (un-stacked) vector.

        Raises:
            ValueError: If ``batches`` is empty or the batches disagree on
                ``atomic_numbers`` or per-batch shape.
        """
        if not batches:
            raise ValueError("AtomisticBatch.stack requires at least one batch.")
        reference = batches[0]
        for batch in batches[1:]:
            if not jnp.array_equal(batch.atomic_numbers, reference.atomic_numbers):
                raise ValueError(
                    "AtomisticBatch.stack requires all batches to share one "
                    "atomic_numbers vector (same composition)."
                )
            if batch.positions.shape != reference.positions.shape:
                raise ValueError(
                    "AtomisticBatch.stack requires equal-shaped batches; got "
                    f"{batch.positions.shape} vs {reference.positions.shape}."
                )
        return cls(
            positions=jnp.stack([batch.positions for batch in batches]),
            atomic_numbers=reference.atomic_numbers,
            energies=jnp.stack([batch.energies for batch in batches]),
            forces=jnp.stack([batch.forces for batch in batches]),
        )

    @classmethod
    def from_arrays(
        cls,
        positions: Float[Array, "batch n_atoms 3"],
        atomic_numbers: Array,
        energies: Float[Array, " batch"],
        forces: Float[Array, "batch n_atoms 3"],
    ) -> AtomisticBatch:
        """Build a batch from already-stacked arrays (the dataset-loader path).

        Loaders such as :func:`opifex.data.loaders.create_rmd17_loader` emit
        configurations as stacked arrays sharing one ``atomic_numbers`` vector, so
        no per-configuration :class:`MolecularSystem` round-trip is needed; this
        is the array-batch complement of :meth:`from_systems`.

        Args:
            positions: Atomic positions, shape ``(batch, n_atoms, 3)``.
            atomic_numbers: Shared nuclear charges, shape ``(n_atoms,)``.
            energies: Reference total energies, shape ``(batch,)``.
            forces: Reference forces, shape ``(batch, n_atoms, 3)``.

        Returns:
            The stacked :class:`AtomisticBatch`.

        Raises:
            ValueError: If ``positions`` has no leading batch axis (is empty).
        """
        positions = jnp.asarray(positions)
        if positions.ndim != 3 or positions.shape[0] == 0:
            raise ValueError(
                "AtomisticBatch.from_arrays requires positions of shape "
                f"(batch, n_atoms, 3) with batch >= 1, got {positions.shape}."
            )
        return cls(
            positions=positions,
            atomic_numbers=jnp.asarray(atomic_numbers),
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


def make_scanned_epoch(
    model: AtomisticModel,
    optimizer: nnx.Optimizer,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    ema_decay: float | None = None,
) -> Callable[
    [AtomisticModel, nnx.Optimizer, AtomisticBatch, nnx.State | None],
    tuple[nnx.State | None, Array],
]:
    r"""Build a jitted fused-epoch step that scans the training step over batches.

    The per-step Python loop in :func:`fit_atomistic` issues one host->device
    dispatch per ``train_step`` (and per ``ema.update``); for the small per-step
    kernels of an MLIP this drains the device queue and starves the GPU between
    steps. This function fuses **a whole epoch's steps into one jitted**
    :func:`jax.lax.scan`, so the device queue stays full and the GPU does not idle
    on dispatch gaps. The math is unchanged: each scan iteration runs the identical
    ``nnx.value_and_grad`` + ``optimizer.update`` + EMA blend as one Python step,
    in the same order, with the optimiser (and EMA) state threaded as the scan
    carry.

    The lifting follows the Flax NNX Functional API (``flax`` ``docs_nnx/
    nnx_basics.md``, "The Flax Functional API"): the ``(model, optimizer)`` graph
    is split into a ``graphdef`` (static) and a ``state`` pytree carry; the scan
    body :func:`~flax.nnx.merge`\ s the carry back into objects, runs the step,
    then re-:func:`~flax.nnx.split`\ s to propagate the updated state -- the
    documented ``nnx.split`` -> JAX-transform -> ``nnx.split`` round-trip. The
    optimiser-state-as-scan-carry idiom is the canonical ``jax.lax.scan`` pattern
    (``jax`` ``docs/pytrees.md`` SGD ``update`` over a pytree; ``docs/
    gpu_performance_tips.md`` ``lax.scan`` over a stepping body). This is a scan
    over training **steps** (each a full fwd+bwd), so the second-order conservative
    -force gradient inside :func:`_batch_loss` is unaffected -- unlike a scan over
    network *layers*.

    Args:
        model: The model to optimise (its graphdef pins the static structure).
        optimizer: An ``nnx.Optimizer`` wrapping ``model`` (``wrt=nnx.Param``).
        energy_weight: Weight of the energy MSE term.
        force_weight: Weight of the forces MSE term.
        ema_decay: If set, the EMA shadow state is threaded through the scan carry
            and blended inside the scan body (``ema = d*ema + (1-d)*params``); the
            returned carry holds the updated shadow. If ``None``, the EMA carry is
            ``None`` and no blend happens.

    Returns:
        A jitted ``(model, optimizer, stacked_epoch, ema_state) ->
        (ema_state, per_step_losses)`` where ``stacked_epoch`` is an
        :class:`AtomisticBatch` with a leading ``num_steps`` axis (see
        :meth:`AtomisticBatch.stack`), ``ema_state`` is the EMA shadow carry (or
        ``None``), and ``per_step_losses`` has shape ``(num_steps,)``. The
        ``model`` and ``optimizer`` passed in are updated in place (NNX propagates
        the in-place state mutation of the jitted arguments back to the caller).
    """
    del model, optimizer  # captured by the caller; the step is parametric in them.

    def loss_fn(model: AtomisticModel, batch: AtomisticBatch) -> Array:
        return _batch_loss(model, batch, energy_weight=energy_weight, force_weight=force_weight)

    decay_array = jnp.asarray(ema_decay) if ema_decay is not None else None

    @nnx.jit
    def scanned_epoch(
        model: AtomisticModel,
        optimizer: nnx.Optimizer,
        stacked_epoch: AtomisticBatch,
        ema_state: nnx.State | None,
    ) -> tuple[nnx.State | None, Array]:
        # Split the mutable variables (model ``nnx.Param`` + optimiser state, all
        # ``nnx.Variable`` subclasses) from the static remainder. The static
        # partition holds the bare-array leaves of *frozen* sub-pytrees -- e.g. the
        # :class:`~opifex.neural.atomistic.scale_shift.AtomicScaleShift` readout and
        # the Bessel ``_frequencies`` -- which are constant during training and live
        # inside ``flax.struct.dataclass(frozen=True)`` nodes that reject in-place
        # writes. Only ``variables`` is threaded through the scan carry and written
        # back; ``static`` is closed over and never updated (the Flax NNX
        # filtered-split idiom, ``flax`` ``docs_nnx/nnx_basics.md`` "Splitting").
        graphdef, variables, static = nnx.split((model, optimizer), nnx.Variable, ...)
        atomic_numbers = stacked_epoch.atomic_numbers
        # Scan only over the per-step arrays; ``atomic_numbers`` is shared and
        # closed over (it has no leading step axis and must not be sliced).
        per_step = (stacked_epoch.positions, stacked_epoch.energies, stacked_epoch.forces)

        def step(
            carry: tuple[nnx.State, nnx.State | None],
            step_arrays: tuple[Array, Array, Array],
        ) -> tuple[tuple[nnx.State, nnx.State | None], Array]:
            variables, ema_state = carry
            step_model, step_optimizer = nnx.merge(graphdef, variables, static)
            positions, energies, forces = step_arrays
            batch = AtomisticBatch(
                positions=positions,
                atomic_numbers=atomic_numbers,
                energies=energies,
                forces=forces,
            )
            loss, grads = nnx.value_and_grad(loss_fn)(step_model, batch)
            step_optimizer.update(step_model, grads)
            if ema_state is not None:
                live_state = nnx.state(step_model, nnx.Param)
                ema_state = _ema_blend(ema_state, live_state, decay_array)
            _, new_variables, _ = nnx.split((step_model, step_optimizer), nnx.Variable, ...)
            return (new_variables, ema_state), loss

        (final_variables, final_ema), losses = jax.lax.scan(step, (variables, ema_state), per_step)
        # Propagate the post-scan mutable variables into the (jitted-argument) live
        # objects; ``nnx.jit`` carries this mutation back out to the caller's
        # references. The static partition is unchanged, so it is not written back.
        nnx.update((model, optimizer), final_variables)
        return final_ema, losses

    return scanned_epoch


def fit_atomistic(
    model: AtomisticModel,
    batches: Sequence[AtomisticBatch],
    optimizer_config: OptimizerConfig,
    num_epochs: int,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    ema_decay: float | None = None,
    fused: bool = True,
) -> list[float]:
    """Train an atomistic model for ``num_epochs`` over the given batches.

    Builds the optimiser from ``optimizer_config`` with
    :func:`opifex.core.training.optimizers.create_optimizer`, wraps it in an
    ``nnx.Optimizer`` and runs the energy+forces step over every batch each epoch.
    The model is updated in place.

    With ``fused=True`` (the default) an epoch's steps are fused into one jitted
    :func:`jax.lax.scan` via :func:`make_scanned_epoch`, eliminating the
    per-step host->device dispatch that otherwise starves the GPU between the
    small MLIP kernels. The fused path is **bit-for-bit the same computation** as
    the per-step loop -- same optimiser updates, same EMA blend, same per-step
    losses, same order -- it only removes the dispatch gaps. Set ``fused=False``
    to use the explicit Python per-step loop (e.g. for step-level debugging).

    When ``ema_decay`` is set, a :class:`ParamEMA` shadow is maintained and
    updated after every training step, and **the model is left holding the EMA
    (averaged) parameters on return** -- the standard NequIP/MACE convention of
    evaluating against smoothed weights rather than the noisy last-step weights
    (NequIP ``ema_decay``; MACE ``ema.average_parameters()``,
    ``mace/tools/train.py``). With ``ema_decay=None`` (the default) the model
    holds the raw last-step weights -- the original behaviour, unchanged.

    Args:
        model: The model to train in place.
        batches: The training batches (cycled once per epoch). The fused path
            requires all batches to share one composition and batch size (see
            :meth:`AtomisticBatch.stack`).
        optimizer_config: Optimiser configuration.
        num_epochs: Number of passes over ``batches``.
        energy_weight: Weight of the energy MSE term.
        force_weight: Weight of the forces MSE term.
        ema_decay: If set (``0.99``-``0.999`` is the MLIP convention), maintain
            an EMA of the weights and load it into ``model`` on return; if
            ``None``, no EMA is kept and the raw weights are retained.
        fused: If ``True`` (default) fuse each epoch's steps into one jitted
            ``lax.scan``; if ``False`` run the explicit per-step Python loop.

    Returns:
        The mean training loss per epoch (length ``num_epochs``).
    """
    optimizer = nnx.Optimizer(model, create_optimizer(optimizer_config), wrt=nnx.Param)
    if fused:
        return _fit_atomistic_fused(
            model,
            optimizer,
            batches,
            num_epochs,
            energy_weight=energy_weight,
            force_weight=force_weight,
            ema_decay=ema_decay,
        )
    return _fit_atomistic_per_step(
        model,
        optimizer,
        batches,
        num_epochs,
        energy_weight=energy_weight,
        force_weight=force_weight,
        ema_decay=ema_decay,
    )


def _fit_atomistic_per_step(
    model: AtomisticModel,
    optimizer: nnx.Optimizer,
    batches: Sequence[AtomisticBatch],
    num_epochs: int,
    *,
    energy_weight: float,
    force_weight: float,
    ema_decay: float | None,
) -> list[float]:
    """Explicit per-step Python training loop (one dispatch per step)."""
    train_step = make_atomistic_train_step(
        model, optimizer, energy_weight=energy_weight, force_weight=force_weight
    )
    ema = ParamEMA(model, decay=ema_decay) if ema_decay is not None else None
    history: list[float] = []
    for _ in range(num_epochs):
        # Keep the per-step loss on device and accumulate it; sync to host once
        # per epoch (the single ``float`` below) instead of once per step. JAX
        # dispatches asynchronously, so ``float(loss)`` per step blocks the host
        # on every step and can serialise the otherwise-overlapped device queue;
        # one host sync per epoch follows the JAX async-dispatch discipline.
        epoch_loss_sum = jnp.zeros(())
        for batch in batches:
            epoch_loss_sum = epoch_loss_sum + train_step(model, optimizer, batch)
            if ema is not None:
                ema.update(model)
        history.append(float(epoch_loss_sum) / len(batches))
    if ema is not None:
        # Leave the model holding the smoothed (EMA) weights for evaluation.
        ema.copy_to(model)
    return history


def _fit_atomistic_fused(
    model: AtomisticModel,
    optimizer: nnx.Optimizer,
    batches: Sequence[AtomisticBatch],
    num_epochs: int,
    *,
    energy_weight: float,
    force_weight: float,
    ema_decay: float | None,
) -> list[float]:
    """Fused-epoch loop: each epoch is one jitted ``lax.scan`` over the steps."""
    scanned_epoch = make_scanned_epoch(
        model,
        optimizer,
        energy_weight=energy_weight,
        force_weight=force_weight,
        ema_decay=ema_decay,
    )
    stacked_epoch = AtomisticBatch.stack(batches)
    ema_state: nnx.State | None = (
        jax.tree.map(jnp.asarray, nnx.state(model, nnx.Param)) if ema_decay is not None else None
    )
    num_steps = len(batches)
    history: list[float] = []
    for _ in range(num_epochs):
        ema_state, losses = scanned_epoch(model, optimizer, stacked_epoch, ema_state)
        history.append(float(jnp.sum(losses)) / num_steps)
    if ema_state is not None:
        # Leave the model holding the smoothed (EMA) weights for evaluation.
        nnx.update(model, ema_state)
    return history


__all__ = [
    "AtomisticBatch",
    "ParamEMA",
    "energy_forces_loss",
    "fit_atomistic",
    "make_atomistic_train_step",
    "make_scanned_epoch",
]
