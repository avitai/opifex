r"""Diffusion tempering schedules (Beck & Tronarp+ 2024, arXiv:2402.12231).

*Diffusion tempering* anneals the **solver noise level** ``σ(epoch)``
from a large initial value (a well-mixed, smooth likelihood that
helps MLE escape local minima) toward a small final value (sharp
posterior) over the course of training. The technique was introduced by
Beck, Bosch, Deistler, Kadhim, Macke, Hennig & Tronarp 2024
(arXiv:2402.12231) for parameter estimation with probabilistic-ODE
integrators, and is the **fifth and final Phase-10 Neural-UQ task**
of the opifex uncertainty-platform plan.

The opifex translation hooks into the existing
:class:`~opifex.uncertainty.objectives.ObjectiveConfig` surface by
rebuilding ``physics_weight`` per epoch from the inverse-noise factor

.. math::

    \text{physics\_weight}(t) = \frac{\text{base\_physics\_weight}}{\sigma(t)^{2}},

so that when ``σ`` is large the physics constraint is soft (matches a
broad likelihood) and tightens monotonically as ``σ`` decays. The
schedule itself is a Pattern-A frozen-slotted-kw-only dataclass; the
``objective_config_at(epoch=, base_config=)`` helper composes with any
``ObjectiveConfig`` consumed by :class:`UQLossComponents` and
:meth:`RobustPINNOptimizer.compute_loss_components`, so no new trainer
class is required (the original
``notes/04-task-6.3-expansion-design.md`` deferral note explicitly
called this out as the intended integration point).

Three schedule shapes are supported:

* :attr:`TemperingScheduleType.LINEAR` — ``σ(t) = σ_0 + t (σ_T - σ_0) / T``.
* :attr:`TemperingScheduleType.EXPONENTIAL` — log-linear interpolation
  between the endpoints (``σ(t) = σ_0 (σ_T / σ_0)^(t/T)``).
* :attr:`TemperingScheduleType.COSINE` — half-cosine smoothing between
  the endpoints (``σ(t) = σ_T + 0.5 (σ_0 - σ_T) (1 + cos(π t / T))``).

References
----------
* Beck, J., Bosch, N., Deistler, M., Kadhim, K. L., Macke, J. H.,
  Hennig, P., Tronarp, F. 2024 — *Diffusion Tempering Improves
  Parameter Estimation with Probabilistic Integrators for Ordinary
  Differential Equations*, arXiv:2402.12231 (PRIMARY).
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum

import jax
import jax.numpy as jnp

from opifex.uncertainty.objectives import (
    ObjectiveConfig,  # noqa: TC001 — kept eager for consistency
)


class TemperingScheduleType(StrEnum):
    """Shape of the noise-level interpolation between the two endpoints."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionTemperingSchedule:
    """Schedule producing an annealed noise level ``σ(epoch)``.

    Attributes:
        initial_noise: ``σ_0`` — the noise level at epoch ``0`` (must be
            strictly positive and strictly larger than ``final_noise``).
        final_noise: ``σ_T`` — the noise level at epoch ``num_epochs``
            (must be strictly positive and strictly smaller than
            ``initial_noise``).
        num_epochs: ``T`` — total annealing horizon. Must be strictly
            positive.
        schedule_type: One of :class:`TemperingScheduleType`.
    """

    initial_noise: float
    final_noise: float
    num_epochs: int
    schedule_type: TemperingScheduleType

    def __post_init__(self) -> None:
        """Validate schedule endpoints + horizon eagerly."""
        if self.initial_noise <= 0.0 or self.final_noise <= 0.0:
            raise ValueError(
                "DiffusionTemperingSchedule requires positive noise levels; "
                f"got initial_noise={self.initial_noise!r}, "
                f"final_noise={self.final_noise!r}."
            )
        if self.final_noise >= self.initial_noise:
            raise ValueError(
                "DiffusionTemperingSchedule requires initial_noise > final_noise "
                "(annealing decreases sigma); got "
                f"initial_noise={self.initial_noise!r}, "
                f"final_noise={self.final_noise!r}."
            )
        if self.num_epochs <= 0:
            raise ValueError(
                "DiffusionTemperingSchedule.num_epochs must be a positive "
                f"integer; got {self.num_epochs!r}."
            )

    def noise_at(self, epoch: int) -> jax.Array:
        """Return ``sigma(epoch)`` clipped to ``[0, num_epochs]``."""
        progress = jnp.clip(jnp.asarray(epoch) / self.num_epochs, 0.0, 1.0)
        if self.schedule_type is TemperingScheduleType.LINEAR:
            return self.initial_noise + progress * (self.final_noise - self.initial_noise)
        if self.schedule_type is TemperingScheduleType.EXPONENTIAL:
            log_initial = jnp.log(self.initial_noise)
            log_final = jnp.log(self.final_noise)
            return jnp.exp(log_initial + progress * (log_final - log_initial))
        # COSINE
        return self.final_noise + 0.5 * (self.initial_noise - self.final_noise) * (
            1.0 + jnp.cos(jnp.pi * progress)
        )

    def objective_config_at(
        self,
        *,
        epoch: int,
        base_config: ObjectiveConfig,
    ) -> ObjectiveConfig:
        r"""Return a new ``ObjectiveConfig`` with ``physics_weight`` rebuilt.

        The new weight is ``base_physics_weight / σ(epoch)²``; every
        other field is forwarded unchanged.

        Args:
            epoch: Current training epoch.
            base_config: The baseline ``ObjectiveConfig`` that supplies
                ``base_physics_weight`` and every other weight.

        Returns:
            A new :class:`ObjectiveConfig` whose ``physics_weight`` is
            the inverse-noise-scaled rebuild.
        """
        sigma = self.noise_at(epoch)
        rebuilt_weight = float(base_config.physics_weight / (sigma**2))
        return dataclasses.replace(base_config, physics_weight=rebuilt_weight)


__all__ = [
    "DiffusionTemperingSchedule",
    "TemperingScheduleType",
]
