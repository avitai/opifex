"""Uncertainty-guided training utilities.

Sample-selection and uncertainty-propagation helpers used alongside the unified
:class:`opifex.core.training.Trainer` (these are not standalone training loops):

- :class:`UncertaintyGuidedTrainer` — select the most uncertain samples and weight them.
- :class:`MultiFidelityUncertaintyTrainer` — propagate uncertainty across model fidelities.
- :class:`ActiveUncertaintyLearner` — acquire informative samples (delegates to
  :mod:`opifex.uncertainty.active.acquisition`).

The general-purpose training loop lives in :class:`opifex.core.training.Trainer` (NNX-native,
``nnx.Optimizer``); this module no longer hosts a hand-rolled optax trainer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx  # noqa: TC002 — pyproject dep kept eager (project convention)

from opifex.uncertainty.active.acquisition import (
    acquire as _active_acquire,
    AcquisitionStrategy,
)
from opifex.uncertainty.aggregators.basic import UncertaintyQuantifier  # noqa: TC001
from opifex.uncertainty.types import PredictiveDistribution


logger = logging.getLogger(__name__)


def _squeeze_output_axis(arr: jax.Array) -> jax.Array:
    """Drop a trailing singleton output axis if present."""
    if arr.ndim > 1 and arr.shape[-1] == 1:
        return arr.squeeze(-1)
    return arr


def _stochastic_ensemble_from_model(
    model: nnx.Module | Callable[[jax.Array], jax.Array],
    x: jax.Array,
    *,
    num_samples: int,
    rngs: nnx.Rngs,
    noise_scale: float = 1e-2,
) -> jax.Array:
    """Build a ``(num_samples, batch, output)`` ensemble by really calling ``model``.

    The model is invoked exactly **once** per ensemble member. For NNX
    modules with stochastic state (dropout, MC sampling) every member
    yields a distinct output. For deterministic models we add a small
    aleatoric noise floor so the downstream
    :meth:`UncertaintyQuantifier.decompose_uncertainty` does not see a
    rank-deficient batch (whose ``var=0`` would produce degenerate
    acquisition scores). The noise scale is small (1e-2) so the model's
    own output dominates whenever it varies.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples!r}")
    predictions = []
    for _ in range(int(num_samples)):
        # NNX state mutates on each call when the model carries stochastic
        # state; for deterministic models all members start from the same
        # prediction and only differ in the aleatoric-noise overlay below.
        pred = model(x)  # type: ignore[operator]  # nnx.Module is callable
        if pred.ndim == 1:
            pred = pred[:, None]
        predictions.append(pred)
    stacked = jnp.stack(predictions, axis=0)  # (num_samples, batch, output)
    key = extract_rng_key(
        rngs,
        streams=("active_acquire", "default", "params", "sample"),
        context="active-learning ensemble",
    )
    eps = noise_scale * jax.random.normal(key, stacked.shape)
    return stacked + eps


class UncertaintyGuidedTrainer:
    """Uncertainty-guided adaptive training with active learning strategies.

    Invokes ``uncertainty_quantifier`` on model predictions. The helper
    :func:`_stochastic_ensemble_from_model` is the single point where
    ``model(x)`` is called per ensemble member, keeping acquisition
    formulas out of the inline path.
    """

    def __init__(
        self,
        model: nnx.Module | Callable[[jax.Array], jax.Array],
        uncertainty_quantifier: UncertaintyQuantifier,
        rngs: nnx.Rngs,
        uncertainty_threshold: float = 0.1,
        adaptation_strategy: str = "active_learning",
    ) -> None:
        """Initialize uncertainty-guided trainer.

        Args:
            model: Neural network model to train.
            uncertainty_quantifier: Uncertainty quantification module,
                typed to :class:`UncertaintyQuantifier`.
            rngs: Caller-owned :class:`nnx.Rngs` bundle used to materialise
                aleatoric-noise overlays for the ensemble draws.
            uncertainty_threshold: Threshold for high uncertainty detection
            adaptation_strategy: Strategy for adapting training
                (``"active_learning"`` / ``"loss_weighting"`` /
                ``"convergence_monitoring"``).
        """
        self.model = model
        self.uncertainty_quantifier = uncertainty_quantifier
        self.rngs = rngs
        self.uncertainty_threshold = uncertainty_threshold
        self.adaptation_strategy = adaptation_strategy

    def select_uncertain_samples(self, x_pool: jax.Array, num_samples: int = 10) -> list[int]:
        """Select most uncertain samples from pool for active learning."""
        predictions = _stochastic_ensemble_from_model(
            self.model,
            x_pool,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        components = self.uncertainty_quantifier.decompose_uncertainty(predictions)
        total_uncertainty = components.total.squeeze()
        return jnp.argsort(total_uncertainty)[-num_samples:].tolist()

    def compute_uncertainty_weights(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute adaptive loss weights based on uncertainty."""
        del y  # weights depend on input-conditioned uncertainty only.
        predictions = _stochastic_ensemble_from_model(
            self.model,
            x,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        components = self.uncertainty_quantifier.decompose_uncertainty(predictions)
        total_uncertainty = components.total.squeeze()
        max_unc = jnp.max(total_uncertainty)
        # Guard against the all-zero case (deterministic model + zero noise).
        normalised = jnp.where(max_unc > 0.0, total_uncertainty / max_unc, total_uncertainty)
        return jnp.clip(normalised, 0.1, 1.0)

    def monitor_uncertainty_convergence(
        self, x_val: jax.Array, y_val: jax.Array
    ) -> dict[str, float]:
        """Monitor uncertainty convergence during training."""
        del y_val  # ECE is computed downstream; here we expose component magnitudes.
        predictions = _stochastic_ensemble_from_model(
            self.model,
            x_val,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        components = self.uncertainty_quantifier.decompose_uncertainty(predictions)
        calibration_error = jnp.mean(jnp.abs(components.total))
        return {
            "epistemic_uncertainty": float(jnp.mean(components.epistemic)),
            "aleatoric_uncertainty": float(jnp.mean(components.aleatoric)),
            "calibration_error": float(calibration_error),
        }


class MultiFidelityUncertaintyTrainer:
    """Multi-fidelity uncertainty propagation trainer.

    Invokes both ``high_fidelity_model(x)`` and ``low_fidelity_model(x)``
    and combines their uncertainties with the Kennedy-O'Hagan
    ``fidelity_ratio`` weighting between high- and low-fidelity outputs.
    """

    def __init__(
        self,
        high_fidelity_model: nnx.Module | Callable[[jax.Array], jax.Array],
        low_fidelity_model: nnx.Module | Callable[[jax.Array], jax.Array],
        uncertainty_quantifier: UncertaintyQuantifier,
        rngs: nnx.Rngs,
        fidelity_ratio: float = 0.1,
    ) -> None:
        """Initialize multi-fidelity uncertainty trainer.

        Args:
            high_fidelity_model: High fidelity neural network model
            low_fidelity_model: Low fidelity neural network model
            uncertainty_quantifier: Uncertainty quantification module.
            rngs: Caller-owned :class:`nnx.Rngs` bundle for the per-fidelity
                ensemble draws.
            fidelity_ratio: Ratio of high to low fidelity data (Kennedy-O'Hagan
                additive linear weighting).
        """
        self.high_fidelity_model = high_fidelity_model
        self.low_fidelity_model = low_fidelity_model
        self.uncertainty_quantifier = uncertainty_quantifier
        self.rngs = rngs
        self.fidelity_ratio = fidelity_ratio

    def propagate_multi_fidelity_uncertainty(self, x: jax.Array) -> jax.Array:
        """Propagate uncertainty through both fidelity levels."""
        hi_predictions = _stochastic_ensemble_from_model(
            self.high_fidelity_model,
            x,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        lo_predictions = _stochastic_ensemble_from_model(
            self.low_fidelity_model,
            x,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )

        hi_components = self.uncertainty_quantifier.decompose_uncertainty(hi_predictions)
        lo_components = self.uncertainty_quantifier.decompose_uncertainty(lo_predictions)

        propagated = (
            self.fidelity_ratio * hi_components.total
            + (1.0 - self.fidelity_ratio) * lo_components.total
        )
        return propagated.squeeze()


class ActiveUncertaintyLearner:
    """Active learning with uncertainty-based sample acquisition.

    Acquisition is delegated to :func:`opifex.uncertainty.active.acquire`.
    :meth:`acquire_samples` accepts an explicit
    ``acquisition_fn: Callable[[jax.Array, jax.Array], jax.Array]``
    (signature ``(mean, variance) -> per-candidate scores``) in addition
    to the ``sampling_strategy`` string registered with the active-learning
    registry. When ``acquisition_fn`` is provided it overrides the strategy
    string.

    Acquisition formula evaluation goes through the active-learning
    subsystem rather than being inlined in this class.
    """

    def __init__(
        self,
        model: nnx.Module | Callable[[jax.Array], jax.Array],
        uncertainty_quantifier: UncertaintyQuantifier,
        rngs: nnx.Rngs,
        sampling_strategy: str = "max_uncertainty",
        acquisition_size: int = 10,
        diversity_weight: float = 0.0,
        physics_priors: Any | None = None,
    ) -> None:
        """Initialize active uncertainty learner.

        Args:
            model: Neural network model — actually invoked on the pool.
            uncertainty_quantifier: Uncertainty quantification module.
            rngs: Caller-owned :class:`nnx.Rngs` bundle.
            sampling_strategy: Default acquisition strategy when no
                ``acquisition_fn`` is passed to :meth:`acquire_samples`.
                Mapped to :class:`AcquisitionStrategy` values.
            acquisition_size: Number of samples to acquire per round.
            diversity_weight: Optional diversity penalty (currently
                surfaced through the strategy mapping).
            physics_priors: Physics-informed priors (optional, reserved
                for the physics-guided strategy).
        """
        self.model = model
        self.uncertainty_quantifier = uncertainty_quantifier
        self.rngs = rngs
        self.sampling_strategy = sampling_strategy
        self.acquisition_size = acquisition_size
        self.diversity_weight = diversity_weight
        self.physics_priors = physics_priors

    def _predictive_distribution(self, x_pool: jax.Array) -> PredictiveDistribution:
        """Build the :class:`PredictiveDistribution` for ``x_pool``.

        Invokes ``self.model`` ``num_samples`` times via the shared
        :func:`_stochastic_ensemble_from_model` helper, then routes the
        ensemble through ``self.uncertainty_quantifier`` to recover the
        epistemic / aleatoric / total decomposition.
        """
        predictions = _stochastic_ensemble_from_model(
            self.model,
            x_pool,
            num_samples=self.uncertainty_quantifier.num_samples,
            rngs=self.rngs,
        )
        components = self.uncertainty_quantifier.decompose_uncertainty(predictions)
        # Drop the trailing singleton output axis so the PredictiveDistribution's
        # per-candidate shape contract (mean.shape == (batch,) for scalar output)
        # holds for the active-learning acquisition kernels downstream.
        mean_full = jnp.mean(predictions, axis=0)
        mean = mean_full.squeeze(-1) if predictions.shape[-1] == 1 else mean_full
        variance = _squeeze_output_axis(components.total)
        samples = predictions.squeeze(-1) if predictions.shape[-1] == 1 else predictions
        epistemic = _squeeze_output_axis(components.epistemic)
        aleatoric = _squeeze_output_axis(components.aleatoric)
        return PredictiveDistribution(
            mean=mean,
            variance=variance,
            samples=samples,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=variance,
        )

    def acquire_samples(
        self,
        x_pool: jax.Array,
        *,
        acquisition_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ) -> list[int]:
        """Acquire samples from pool by delegating to the active subsystem.

        Args:
            x_pool: Pool of candidate inputs.
            acquisition_fn: Optional callable
                ``(mean, variance) -> per-candidate scores``. When
                supplied, the top-``acquisition_size`` indices by score
                are returned directly. When omitted, dispatch routes
                through :func:`opifex.uncertainty.active.acquire` using
                ``sampling_strategy`` mapped to
                :class:`AcquisitionStrategy`.

        Returns:
            list[int]: indices of the acquired pool elements.
        """
        predictive = self._predictive_distribution(x_pool)

        if acquisition_fn is not None:
            if predictive.variance is None:
                raise RuntimeError("predictive distribution has no variance.")
            scores = acquisition_fn(predictive.mean, predictive.variance)
            top = jnp.argsort(scores)[-self.acquisition_size :]
            return [int(i) for i in top]

        strategy_map = {
            "max_uncertainty": AcquisitionStrategy.MAX_VARIANCE,
            "diverse_uncertainty": AcquisitionStrategy.MAX_VARIANCE,
            "physics_guided_uncertainty": AcquisitionStrategy.MAX_VARIANCE,
            "bald": AcquisitionStrategy.BALD,
            "ucb": AcquisitionStrategy.UCB,
            "lcb": AcquisitionStrategy.LCB,
            "ei": AcquisitionStrategy.EI,
            "log_ei": AcquisitionStrategy.LOG_EI,
            "pi": AcquisitionStrategy.PI,
        }
        strategy = strategy_map.get(self.sampling_strategy, AcquisitionStrategy.MAX_VARIANCE)
        kwargs: dict[str, Any] = {}
        if strategy in (AcquisitionStrategy.EI, AcquisitionStrategy.LOG_EI, AcquisitionStrategy.PI):
            kwargs["best_value"] = float(jnp.min(predictive.mean))
        if strategy in (AcquisitionStrategy.UCB, AcquisitionStrategy.LCB):
            kwargs["beta"] = 1.96
        batch = _active_acquire(
            predictive,
            strategy=strategy,
            batch_size=self.acquisition_size,
            rngs=self.rngs,
            **kwargs,
        )
        return [int(i) for i in batch.indices]
