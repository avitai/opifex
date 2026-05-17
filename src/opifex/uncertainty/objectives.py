"""Objective and loss-component contracts.

Container patterns:

* :class:`ObjectiveConfig` — plain
  ``@dataclass(frozen=True, slots=True, kw_only=True)`` (Avitai canonical
  config/document pattern). All fields are Python scalars / strings; the
  container is passed as a hashable static argument to jit'd loss kernels and
  has zero per-call pytree overhead. Validation runs in ``__post_init__``.
* :class:`UQLossComponents` —
  ``@flax.struct.dataclass(slots=True, kw_only=True)``. Carries scalar loss
  arrays through ``jit``/``grad``/``vmap``. ``metadata`` is the canonical
  ``tuple[tuple[str, Any], ...]`` aux_data. ``validate()`` is public and is
  NEVER called from ``__post_init__``/``tree_unflatten``.

The optimizer-facing scalar is ``UQLossComponents.total`` — the weighted sum of
every supplied loss component. ``elbo`` is derived from ``negative_elbo``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.types import _metadata_dict, MetadataItems


# StrEnum of supported reductions kept inline (small fixed set; avoids a new
# import for a single field). Promotion to a public StrEnum can happen if a
# caller needs structural dispatch on the value.
_ALLOWED_REDUCTIONS = frozenset({"mean", "sum", "none"})


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ObjectiveConfig:
    """Static weights and dataset metadata for UQ loss aggregation.

    All weights must be non-negative; each multiplies its matching component
    in :meth:`UQLossComponents.from_components`. ``dataset_size`` is ``None``
    when the dataset cardinality is not yet known (e.g., streaming setups);
    :func:`scale_kl` then returns the raw KL.

    Fields:

    * ``kl_weight`` — multiplier on the ``kl`` component (Bayesian /
      variational KL term, typically dataset-scaled via :func:`scale_kl`
      before weighting).
    * ``dataset_size`` — N in the per-example ELBO scaling ``KL / N``;
      ``None`` disables that scaling.
    * ``physics_weight`` — multiplier on ``physics_residual`` (PINN residual
      / PDE-loss term).
    * ``data_weight`` — multiplier applied to both ``data`` and
      ``negative_log_likelihood`` (supervised regression / likelihood term).
    * ``boundary_weight`` — multiplier on ``boundary`` (PINN boundary-condition
      penalty).
    * ``initial_condition_weight`` — multiplier on ``initial_condition``
      (PINN initial-condition penalty).
    * ``regularization_weight`` — multiplier on ``regularization`` (extra
      penalty term: L2 / spectral / etc., separate from the KL term).
    * ``calibration_weight`` — multiplier on ``calibration``
      (calibration-aware training term).
    * ``conformal_weight`` — multiplier on ``conformal``
      (conformal-training term).
    * ``pac_bayes_weight`` — multiplier on ``pac_bayes`` (PAC-Bayes bound
      term).
    * ``reduction`` — how per-example losses are reduced to a scalar; one of
      ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    kl_weight: float
    dataset_size: int | None
    physics_weight: float
    data_weight: float
    boundary_weight: float
    initial_condition_weight: float
    regularization_weight: float
    calibration_weight: float
    conformal_weight: float
    pac_bayes_weight: float
    reduction: str = "mean"

    def __post_init__(self) -> None:
        for name, value in (
            ("kl_weight", self.kl_weight),
            ("physics_weight", self.physics_weight),
            ("data_weight", self.data_weight),
            ("boundary_weight", self.boundary_weight),
            ("initial_condition_weight", self.initial_condition_weight),
            ("regularization_weight", self.regularization_weight),
            ("calibration_weight", self.calibration_weight),
            ("conformal_weight", self.conformal_weight),
            ("pac_bayes_weight", self.pac_bayes_weight),
        ):
            if value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative; got {value!r}. "
                    "Reject negative loss weights at construction time."
                )
        if self.dataset_size is not None and self.dataset_size <= 0:
            raise ValueError(
                f"dataset_size must be positive when provided; got {self.dataset_size!r}."
            )
        if self.reduction not in _ALLOWED_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {sorted(_ALLOWED_REDUCTIONS)}; got {self.reduction!r}."
            )


@struct.dataclass(slots=True, kw_only=True)
class UQLossComponents:
    """Optimizer-facing loss decomposition.

    Every loss-component field is an optional ``jax.Array`` scalar. ``total``
    is the differentiable scalar passed to optimizers; ``from_components``
    computes it from the supplied components using weights from
    :class:`ObjectiveConfig`.

    ``elbo`` is a derived property: ``-negative_elbo`` when ``negative_elbo``
    is available, otherwise ``None``.

    Fields (every loss component is optional and contributes ``0`` to
    ``total`` when ``None``):

    * ``total`` — REQUIRED differentiable scalar (sum of weighted components).
      The single value passed to ``jax.value_and_grad`` / ``optimizer.update``.
    * ``data`` — supervised data-fit term (e.g., MSE on regression targets).
      Weighted by ``data_weight``.
    * ``negative_log_likelihood`` — negative log-likelihood under the model's
      output distribution. Also weighted by ``data_weight``.
    * ``physics_residual`` — PINN PDE-residual penalty
      ``mean(|F[u_theta](x)|^2)``. Weighted by ``physics_weight``.
    * ``boundary`` — PINN boundary-condition penalty. Weighted by
      ``boundary_weight``.
    * ``initial_condition`` — PINN initial-condition penalty. Weighted by
      ``initial_condition_weight``.
    * ``regularization`` — extra regularization term (L2 / spectral / etc.)
      that is NOT the Bayesian KL term. Weighted by ``regularization_weight``.
    * ``kl`` — Bayesian / variational KL divergence between posterior and
      prior. Weighted by ``kl_weight`` AFTER dataset-scaling via
      :func:`scale_kl`.
    * ``negative_elbo`` — pre-computed negative ELBO (for backends that
      provide it directly, e.g., variational samplers). Added to ``total``
      WITHOUT additional weighting. ``elbo`` returns ``-negative_elbo``.
    * ``calibration`` — calibration-aware training term. Weighted by
      ``calibration_weight``.
    * ``conformal`` — conformal-training term. Weighted by
      ``conformal_weight``.
    * ``pac_bayes`` — PAC-Bayes bound term. Weighted by ``pac_bayes_weight``.
      Field is part of the contract from day one so PAC-Bayes consumers do
      not need to widen the schema later.
    * ``metadata`` — canonical ``tuple[tuple[str, Any], ...]`` static aux_data
      (immutable, hashable). Use :meth:`metadata_dict` for ergonomic
      dict-style read.
    """

    total: jax.Array
    data: jax.Array | None = None
    negative_log_likelihood: jax.Array | None = None
    physics_residual: jax.Array | None = None
    boundary: jax.Array | None = None
    initial_condition: jax.Array | None = None
    regularization: jax.Array | None = None
    kl: jax.Array | None = None
    negative_elbo: jax.Array | None = None
    calibration: jax.Array | None = None
    conformal: jax.Array | None = None
    pac_bayes: jax.Array | None = None
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    @property
    def elbo(self) -> jax.Array | None:
        """Return ``-negative_elbo`` when ``negative_elbo`` is supplied."""
        if self.negative_elbo is None:
            return None
        return -self.negative_elbo

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh dict view of ``metadata`` for ergonomic access."""
        return _metadata_dict(self.metadata)

    def validate(self) -> None:
        """Public validation hook; never called from ``__post_init__``/``tree_unflatten``.

        Raises:
            ValueError: If ``total`` is not finite.
        """
        if not bool(jnp.all(jnp.isfinite(self.total))):
            raise ValueError("UQLossComponents.total must be finite.")

    @classmethod
    def from_components(
        cls,
        *,
        config: ObjectiveConfig,
        data: jax.Array | None = None,
        negative_log_likelihood: jax.Array | None = None,
        physics_residual: jax.Array | None = None,
        boundary: jax.Array | None = None,
        initial_condition: jax.Array | None = None,
        regularization: jax.Array | None = None,
        kl: jax.Array | None = None,
        negative_elbo: jax.Array | None = None,
        calibration: jax.Array | None = None,
        conformal: jax.Array | None = None,
        pac_bayes: jax.Array | None = None,
        metadata: MetadataItems = (),
    ) -> UQLossComponents:
        """Compute ``total`` as the weight-driven sum of every non-None component.

        Missing components contribute zero; present components contribute
        ``weight_i * component_i``. The KL term is dataset-scaled via
        :func:`scale_kl` so the ELBO is on the per-example scale that
        Bayesian-NN training expects.
        """
        zero = jnp.array(0.0)
        contributions: list[jax.Array] = []
        if data is not None:
            contributions.append(config.data_weight * data)
        if negative_log_likelihood is not None:
            contributions.append(config.data_weight * negative_log_likelihood)
        if physics_residual is not None:
            contributions.append(config.physics_weight * physics_residual)
        if boundary is not None:
            contributions.append(config.boundary_weight * boundary)
        if initial_condition is not None:
            contributions.append(config.initial_condition_weight * initial_condition)
        if regularization is not None:
            contributions.append(config.regularization_weight * regularization)
        if kl is not None:
            contributions.append(config.kl_weight * scale_kl(kl, config))
        if negative_elbo is not None:
            contributions.append(negative_elbo)
        if calibration is not None:
            contributions.append(config.calibration_weight * calibration)
        if conformal is not None:
            contributions.append(config.conformal_weight * conformal)
        if pac_bayes is not None:
            contributions.append(config.pac_bayes_weight * pac_bayes)

        total = sum(contributions, start=zero) if contributions else zero
        return cls(
            total=total,
            data=data,
            negative_log_likelihood=negative_log_likelihood,
            physics_residual=physics_residual,
            boundary=boundary,
            initial_condition=initial_condition,
            regularization=regularization,
            kl=kl,
            negative_elbo=negative_elbo,
            calibration=calibration,
            conformal=conformal,
            pac_bayes=pac_bayes,
            metadata=metadata,
        )


def scale_kl(kl: jax.Array, config: ObjectiveConfig) -> jax.Array:
    """Divide ``kl`` by ``config.dataset_size`` only when a positive size is provided.

    With ``dataset_size=None`` (streaming or unknown), the raw KL is returned —
    callers can then choose their own per-step scaling.
    """
    if config.dataset_size is None:
        return kl
    return kl / float(config.dataset_size)


__all__ = ["ObjectiveConfig", "UQLossComponents", "scale_kl"]
