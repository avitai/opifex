"""Core UQ value-object contracts.

Implements:

* :class:`PredictiveMode` (StrEnum) — predictive-sampling modes.
* :class:`PredictionInterval` — typed (lower, upper, coverage, method, metadata)
  interval container.
* :class:`PredictionSet` — classification prediction-set container (boolean
  inclusion mask + nonconformity scores).
* :class:`PredictiveDistribution` — the full 11-field predictive-distribution
  container.

**Container pattern.** All three containers use :func:`flax.struct.dataclass`
with ``slots=True, kw_only=True`` — the frozen+slotted+keyword-only data-container
convention. Static metadata fields are marked per-field via
:func:`flax.struct.field(pytree_node=False)`. The ``struct.dataclass`` decorator:

* sets ``frozen=True`` by default (we additionally request slots+kw_only);
* registers the class as a JAX PyTree via
  :func:`jax.tree_util.register_dataclass` under the hood;
* provides a ``.replace(**updates)`` immutable-update method;
* integrates with :mod:`flax.serialization` for checkpoint round-tripping.

Choosing ``struct.dataclass`` over the raw
:func:`jax.tree_util.register_dataclass` decorator keeps the per-field
``pytree_node=False`` static annotation local to each field (more
maintainable than a separate top-level ``meta_fields`` list).

Variance fields (``variance``, ``epistemic``, ``aleatoric``,
``total_uncertainty``) are *variances*, not standard deviations.
:meth:`PredictiveDistribution.validate` enforces variance-additivity
``total_uncertainty == epistemic + aleatoric`` within
``_VARIANCE_RTOL=1e-5``, ``_VARIANCE_ATOL=1e-6`` — the canonical tolerances
reused by :class:`opifex.uncertainty.scientific.solutions.SolutionDistribution`.

``validate()`` is a public method and is **never** called from
``__post_init__`` or the pytree unflatten path. Per
``../jax/docs/custom_pytrees.md``, transformations may reconstruct containers
with placeholder values during tracing, so eager validation on every rebuild
would spuriously fail. Callers explicitly invoke :meth:`validate` after
construction when they want pre-jit safety checks.

The metadata fields use ``tuple[tuple[str, Any], ...]`` because pytree
aux_data must be hashable (it forms part of the JIT cache key).
``MappingProxyType`` is immutable but **not** hashable, so it cannot serve as
aux_data. The tuple-of-pairs form satisfies both immutability and hashability.
"""

from __future__ import annotations

from dataclasses import field
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


# Canonical variance-additivity tolerances. Reused by
# ``SolutionDistribution`` so a round-trip through
# ``SolutionDistribution.as_predictive_distribution(field)`` does not flap
# the additivity test.
_VARIANCE_RTOL: float = 1e-5
_VARIANCE_ATOL: float = 1e-6


# Public type alias for the immutable, hashable metadata container.
MetadataItems = tuple[tuple[str, Any], ...]


def _metadata_dict(items: MetadataItems) -> dict[str, Any]:
    """Convert tuple-of-pairs metadata into a mutable dict for ergonomic access."""
    return dict(items)


class PredictiveMode(StrEnum):
    """Predictive-sampling mode for ``predict_distribution`` callers.

    Members:

    * ``PREDICTIVE`` — the default predictive distribution (posterior predictive
      for Bayesian models, deterministic forward pass for non-Bayesian).
    * ``POSTERIOR_PREDICTIVE`` — explicitly request the posterior predictive
      (raises for models that have no posterior).
    * ``PRIOR_PREDICTIVE`` — request the prior predictive (for diagnostic /
      prior-check workflows).
    * ``MEAN_ONLY`` — return only the predictive mean; skip variance / samples
      / quantiles computation.

    Adding a new member must not break existing callers — every consumer
    matches on a closed subset and either dispatches or raises ``ValueError``
    with an actionable message.
    """

    PREDICTIVE = "predictive"
    POSTERIOR_PREDICTIVE = "posterior_predictive"
    PRIOR_PREDICTIVE = "prior_predictive"
    MEAN_ONLY = "mean_only"


@struct.dataclass(slots=True, kw_only=True)
class PredictionInterval:
    """Prediction interval ``[lower, upper]`` at a given coverage level.

    Shape contract: ``lower.shape == upper.shape == (batch, ...)`` where
    trailing axes are domain-specific (per-output-dim for regression,
    per-grid-point for field outputs).

    ``coverage`` is the nominal coverage probability (``0 < coverage < 1``;
    e.g., ``0.9`` for a 90 % interval).
    """

    lower: jax.Array
    upper: jax.Array
    coverage: float = struct.field(pytree_node=False)
    method: str = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return _metadata_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate the interval. Not called from ``__post_init__``."""
        if not 0.0 < float(self.coverage) < 1.0:
            raise ValueError(f"coverage must be in (0, 1); got {self.coverage!r}")
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                "PredictionInterval lower/upper shape mismatch: "
                f"{self.lower.shape} vs {self.upper.shape}"
            )
        if not self.method:
            raise ValueError("PredictionInterval.method must be a non-empty string.")


@struct.dataclass(slots=True, kw_only=True)
class PredictionSet:
    """Classification prediction set: boolean inclusion mask + scores.

    Shape contract: ``values.shape == scores.shape == (batch, num_classes)``.
    ``values`` is the boolean inclusion mask (per-class True/False); ``scores``
    holds the per-class nonconformity scores that produced the mask.

    Regression-style "interval sets" are represented via :class:`PredictionInterval`,
    not :class:`PredictionSet` (the prediction-set abstraction is for
    classification where set membership is the natural output).

    ``threshold`` is a Python ``float`` (not a 0-d ``jax.Array``) so it can be
    a static pytree aux_data leaf and a JIT-cache key without
    array-equality issues.
    """

    values: jax.Array
    scores: jax.Array
    threshold: float = struct.field(pytree_node=False)
    method: str = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return _metadata_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate the set. Not called from ``__post_init__``."""
        if self.values.shape != self.scores.shape:
            raise ValueError(
                "PredictionSet values/scores shape mismatch: "
                f"{self.values.shape} vs {self.scores.shape}"
            )
        if self.values.dtype != jnp.bool_:
            raise ValueError(f"PredictionSet.values must be boolean; got dtype={self.values.dtype}")
        if not self.method:
            raise ValueError("PredictionSet.method must be a non-empty string.")


@struct.dataclass(slots=True, kw_only=True)
class PredictiveDistribution:
    """Predictive-distribution container.

    Field shapes (all optional except ``mean``):

    * ``mean: (batch, ...)`` — predictive mean.
    * ``samples: (num_samples, batch, ...)`` — first axis is sample count.
    * ``variance: (batch, ...)`` — marginal *variance* (NOT std-dev) matching
      ``mean.shape``.
    * ``covariance: (batch, ..., D)`` flattened-feature covariance, or
      Cholesky/diag factor when ``metadata["covariance_form"]`` is
      ``"cholesky"`` / ``"diag"``.
    * ``epistemic``, ``aleatoric``, ``total_uncertainty: (batch, ...)`` —
      *variances* matching ``mean.shape``. When both ``epistemic`` and
      ``aleatoric`` are supplied, :meth:`validate` enforces
      ``jnp.allclose(total_uncertainty, epistemic + aleatoric,
      rtol=1e-5, atol=1e-6)``.
    * ``quantiles: dict[float, jax.Array]`` — ``alpha → quantile``. JAX's
      built-in dict pytree handler treats sorted keys as treedef structure and
      ``jax.Array`` values as leaves.
    * ``interval: PredictionInterval | None``.
    * ``prediction_set: PredictionSet | None``.
    * ``metadata: tuple[tuple[str, Any], ...]`` — immutable, hashable
      static aux_data. Includes documented keys such as ``"method"``,
      ``"covariance_form"`` (``"full" | "cholesky" | "diag"``), axis names,
      units. Use :meth:`metadata_dict` for dict-style read access.

    Variance fields are variances throughout: there is no std-dev field. Use
    :meth:`std` to derive standard deviations from ``variance``.
    """

    mean: jax.Array
    samples: jax.Array | None = None
    variance: jax.Array | None = None
    covariance: jax.Array | None = None
    epistemic: jax.Array | None = None
    aleatoric: jax.Array | None = None
    total_uncertainty: jax.Array | None = None
    quantiles: dict[float, jax.Array] = field(default_factory=dict)
    interval: PredictionInterval | None = None
    prediction_set: PredictionSet | None = None
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return _metadata_dict(self.metadata)

    def quantile(self, alpha: float) -> jax.Array:
        """Return the supplied quantile at level ``alpha``.

        Raises:
            KeyError: when ``alpha`` is not in :attr:`quantiles`. The error
                message lists the available alpha levels for actionable
                debugging.
        """
        try:
            return self.quantiles[alpha]
        except KeyError as e:
            available = sorted(self.quantiles.keys())
            raise KeyError(
                f"PredictiveDistribution has no quantile for alpha={alpha!r}. "
                f"Available alphas: {available}."
            ) from e

    def std(self) -> jax.Array:
        """Return ``sqrt(variance)`` when variance is supplied.

        Raises:
            ValueError: when :attr:`variance` is ``None``.
        """
        if self.variance is None:
            raise ValueError("PredictiveDistribution.std() requires variance to be set.")
        return jnp.sqrt(self.variance)

    def validate(self) -> None:
        """Eager-validate field-shape and variance-additivity invariants.

        **Never** called from ``__post_init__`` or the pytree unflatten path.
        """
        if self.variance is not None and self.variance.shape != self.mean.shape:
            raise ValueError(
                f"variance shape must match mean shape: {self.variance.shape} vs {self.mean.shape}"
            )
        if self.epistemic is not None and self.epistemic.shape != self.mean.shape:
            raise ValueError(
                "epistemic shape must match mean shape: "
                f"{self.epistemic.shape} vs {self.mean.shape}"
            )
        if self.aleatoric is not None and self.aleatoric.shape != self.mean.shape:
            raise ValueError(
                "aleatoric shape must match mean shape: "
                f"{self.aleatoric.shape} vs {self.mean.shape}"
            )
        if (
            self.epistemic is not None
            and self.aleatoric is not None
            and self.total_uncertainty is not None
        ):
            expected_total = self.epistemic + self.aleatoric
            if not bool(
                jnp.allclose(
                    self.total_uncertainty,
                    expected_total,
                    rtol=_VARIANCE_RTOL,
                    atol=_VARIANCE_ATOL,
                )
            ):
                raise ValueError(
                    "PredictiveDistribution.total_uncertainty must equal "
                    "epistemic + aleatoric within "
                    f"rtol={_VARIANCE_RTOL}, atol={_VARIANCE_ATOL}."
                )
        if self.samples is not None and self.samples.shape[1:] != self.mean.shape:
            raise ValueError(
                "samples must have shape (num_samples, *mean.shape); got "
                f"samples.shape={self.samples.shape}, mean.shape={self.mean.shape}"
            )
