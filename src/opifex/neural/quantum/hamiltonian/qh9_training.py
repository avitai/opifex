r"""Training harness wiring :class:`HamiltonianPredictor` to QH9 Fock targets.

QH9 (Yu et al. 2023, "QH9: A Quantum Hamiltonian Prediction Benchmark for QM9
Molecules", arXiv:2306.04922; reference ``divelab/AIRS``
``OpenDFT/QHBench/QH9``) supervises the predictor with converged def2-SVP Fock
matrices in PySCF *spherical* AO ordering. The opifex
:class:`~opifex.neural.quantum.hamiltonian.predictor.HamiltonianPredictor`
assembles its dense matrix in the *Cartesian* AO basis (an ``l = 2`` d-shell has
six Cartesian components but only five spherical ones), so before the loss the
predicted Cartesian matrix is mapped to the spherical basis with the
block-diagonal congruence ``H_sph = T^T H_cart T`` from
:mod:`opifex.core.quantum._spherical` -- whose spherical column ordering is
exactly PySCF's ``mol.spheric_labels()`` order, i.e. the QH9 target ordering.
The matched prediction and target therefore live in the identical AO ordering.

The harness is deliberately thin: a Fock-matrix loss (mean absolute / squared
error over the valid AO block, with optional padding mask) and a jitted
``optax``/``flax.nnx`` train step + fit loop reusing the exact
``nnx.Optimizer`` + ``nnx.value_and_grad`` pattern of the committed Hamiltonian
example (``docs/examples/quantum-chemistry/hamiltonian-prediction.md``). No
optimizer loop is reinvented beyond that minimal pattern.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence  # noqa: TC003
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np  # noqa: TC002
import optax
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.core.quantum._spherical import apply_matrix, build_block_transform
from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.neural.quantum.hamiltonian.predictor import HamiltonianPredictor  # noqa: TC001


logger = logging.getLogger(__name__)

LossKind = Literal["mae", "mse"]
"""Fock-matrix loss reduction: mean-absolute or mean-squared error."""


@dataclass(frozen=True, slots=True, kw_only=True)
class QH9TrainConfig:
    """Hyper-parameters of a QH9 Fock-prediction fit.

    Attributes:
        learning_rate: Adam step size.
        num_steps: Number of optimisation steps in :func:`fit_qh9`.
        loss_kind: ``"mae"`` or ``"mse"`` Fock-matrix reduction.
        property_name: The predictor output key to read the matrix from.
    """

    learning_rate: float = 1e-3
    num_steps: int = 200
    loss_kind: LossKind = "mae"
    property_name: str = "hamiltonian"


def spherical_transform_for(system: MolecularSystem) -> Float[Array, "n_cart n_sph"]:
    """Build the Cartesian->spherical def2-SVP transform for ``system``.

    The block-diagonal transform maps the predictor's Cartesian AO axes onto the
    spherical AO axes that QH9 targets are stored in. It is a static constant
    (no tracers), safe to close over inside ``jit``.

    Args:
        system: The molecular system (fixes the per-shell angular momenta).

    Returns:
        The block-diagonal transform, shape ``(n_cart, n_sph)``.
    """
    basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="def2-svp")
    angular_momenta = tuple(shell.angular_momentum for shell in basis.shells)
    return build_block_transform(angular_momenta)


def predict_spherical_fock(
    predictor: HamiltonianPredictor,
    system: MolecularSystem,
    transform: Float[Array, "n_cart n_sph"],
    *,
    property_name: str = "hamiltonian",
) -> Float[Array, "n_sph n_sph"]:
    """Predict the def2-SVP Fock matrix in spherical (QH9) AO ordering.

    Runs the predictor (Cartesian AO matrix) and applies the congruence
    ``H_sph = T^T H_cart T`` to land in QH9's spherical ordering.

    Args:
        predictor: The bound Hamiltonian predictor.
        system: The molecular system to predict for.
        transform: Cartesian->spherical transform from
            :func:`spherical_transform_for`.
        property_name: The predictor output key holding the matrix.

    Returns:
        The predicted spherical Fock matrix, shape ``(n_sph, n_sph)``.
    """
    cartesian = predictor(system)[property_name]
    return apply_matrix(transform, cartesian)


def fock_loss(
    prediction: Float[Array, "n_ao n_ao"],
    target: Float[Array, "n_ao n_ao"],
    *,
    kind: LossKind = "mae",
    mask: Float[Array, "n_ao n_ao"] | None = None,
) -> Float[Array, ""]:
    r"""Masked mean Fock-matrix error between a prediction and a target.

    The error is reduced over the valid AO entries only: when ``mask`` is given
    (1 on real AO blocks, 0 on padding) the mean is taken over the masked
    entries, so padded rows/columns introduced to batch ragged molecules never
    contribute to the gradient.

    Args:
        prediction: Predicted Fock matrix, shape ``(n_ao, n_ao)``.
        target: Reference Fock matrix, shape ``(n_ao, n_ao)``.
        kind: ``"mae"`` (mean ``|.|``) or ``"mse"`` (mean ``(.)^2``).
        mask: Optional ``{0, 1}`` validity mask over AO entries.

    Returns:
        The scalar masked-mean error.

    Raises:
        ValueError: If ``kind`` is neither ``"mae"`` nor ``"mse"``.
    """
    residual = prediction - target
    if kind == "mae":
        elementwise = jnp.abs(residual)
    elif kind == "mse":
        elementwise = residual**2
    else:
        raise ValueError(f"loss kind must be 'mae' or 'mse', got {kind!r}.")

    if mask is None:
        return jnp.mean(elementwise)
    total = jnp.sum(elementwise * mask)
    count = jnp.sum(mask)
    return total / jnp.clip(count, a_min=1.0)


@dataclass(frozen=True, slots=True)
class QH9FitResult:
    """Outcome of a QH9 Fock-prediction fit.

    Attributes:
        loss_history: Per-step training loss.
        final_loss: Loss after the last step.
    """

    loss_history: Float[Array, " num_steps"]
    final_loss: float


def make_train_step(
    transform: Float[Array, "n_cart n_sph"],
    system: MolecularSystem,
    target: Float[Array, "n_ao n_ao"],
    mask: Float[Array, "n_ao n_ao"] | None,
    *,
    loss_kind: LossKind,
    property_name: str,
):
    """Build a jitted single-molecule QH9 Fock train step.

    The returned closure runs one ``nnx.value_and_grad`` + ``optimizer.update``
    on the masked spherical Fock loss. The molecule / target / transform / loss
    settings are *closed over* (not jit arguments): only the predictor and its
    optimizer are traced, mirroring the committed Hamiltonian example's
    ``loss_fn(module)`` closure -- the predictor needs the system's atom count
    as a Python static, so the system must stay outside the traced arguments.

    Args:
        transform: Cartesian->spherical transform for the molecule.
        system: The molecular system to predict for (closed over).
        target: The spherical Fock target, shape ``(n_ao, n_ao)`` (closed over).
        mask: Optional ``{0, 1}`` AO-validity mask (closed over).
        loss_kind: ``"mae"`` or ``"mse"``.
        property_name: The predictor output key holding the matrix.

    Returns:
        A jitted ``(predictor, optimizer) -> loss`` step.
    """

    def loss_fn(module: HamiltonianPredictor) -> Float[Array, ""]:
        prediction = predict_spherical_fock(module, system, transform, property_name=property_name)
        return fock_loss(prediction, target, kind=loss_kind, mask=mask)

    @nnx.jit
    def train_step(
        module: HamiltonianPredictor,
        opt: nnx.Optimizer,
    ) -> Float[Array, ""]:
        loss, grads = nnx.value_and_grad(loss_fn)(module)
        opt.update(module, grads)
        return loss

    return train_step


def fit_qh9(
    predictor: HamiltonianPredictor,
    system: MolecularSystem,
    target: Float[Array, "n_ao n_ao"],
    *,
    config: QH9TrainConfig | None = None,
    mask: Float[Array, "n_ao n_ao"] | None = None,
) -> QH9FitResult:
    """Overfit ``predictor`` to a single molecule's QH9 Fock matrix.

    A thin Adam fit on the masked spherical Fock loss, reusing the committed
    Hamiltonian example's ``nnx.Optimizer`` + jitted ``nnx.value_and_grad``
    pattern. Intended both as the per-molecule inner loop of a full QH9 training
    run and as the overfit-one-batch sanity check.

    Args:
        predictor: The Hamiltonian predictor to fit (modified in place).
        system: The molecular system to fit.
        target: The def2-SVP spherical Fock target, shape ``(n_ao, n_ao)``.
        config: Fit hyper-parameters. Defaults to :class:`QH9TrainConfig`.
        mask: Optional ``{0, 1}`` AO-validity mask.

    Returns:
        A :class:`QH9FitResult` with the per-step loss history.
    """
    resolved = config if config is not None else QH9TrainConfig()
    target_array = jnp.asarray(target)
    mask_array = None if mask is None else jnp.asarray(mask)

    optimizer = nnx.Optimizer(predictor, optax.adam(resolved.learning_rate), wrt=nnx.Param)
    transform = spherical_transform_for(system)
    train_step = make_train_step(
        transform,
        system,
        target_array,
        mask_array,
        loss_kind=resolved.loss_kind,
        property_name=resolved.property_name,
    )

    losses: list[Array] = []
    for step in range(resolved.num_steps):
        loss = train_step(predictor, optimizer)
        losses.append(loss)
        if step % max(1, resolved.num_steps // 10) == 0:
            logger.info("QH9 fit step %d: loss=%.3e", step, float(loss))

    loss_history = jnp.stack(losses) if losses else jnp.zeros((0,))
    final_loss = float(loss_history[-1]) if losses else float("nan")
    return QH9FitResult(loss_history=loss_history, final_loss=final_loss)


def fit_qh9_examples(
    predictor: HamiltonianPredictor,
    examples: Sequence[tuple[MolecularSystem, np.ndarray]],
    *,
    config: QH9TrainConfig | None = None,
) -> QH9FitResult:
    """Fit ``predictor`` across a sequence of QH9 (system, Fock) examples.

    Because QH9 molecules have heterogeneous AO counts, the predictor is rebound
    per molecule (shared weights, fresh static block plan) and each example
    contributes ``config.num_steps`` Adam steps. The returned history
    concatenates every example's per-step loss.

    Args:
        predictor: The Hamiltonian predictor to fit (modified in place).
        examples: Sequence of ``(system, spherical_fock)`` pairs.
        config: Fit hyper-parameters. Defaults to :class:`QH9TrainConfig`.

    Returns:
        A :class:`QH9FitResult` over the concatenated example losses.
    """
    resolved = config if config is not None else QH9TrainConfig()
    histories: list[Array] = []
    bound = predictor
    for system, fock in examples:
        basis = AtomicOrbitalBasis.from_molecular_system(system, basis_name="def2-svp")
        bound = bound.rebind(basis)
        result = fit_qh9(bound, system, jnp.asarray(fock), config=resolved)
        histories.append(result.loss_history)

    loss_history = jnp.concatenate(histories) if histories else jnp.zeros((0,))
    final_loss = float(loss_history[-1]) if histories else float("nan")
    return QH9FitResult(loss_history=loss_history, final_loss=final_loss)


__all__ = [
    "LossKind",
    "QH9FitResult",
    "QH9TrainConfig",
    "fit_qh9",
    "fit_qh9_examples",
    "fock_loss",
    "make_train_step",
    "predict_spherical_fock",
    "spherical_transform_for",
]
