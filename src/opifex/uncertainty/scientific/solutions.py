"""Solver-side predictive distribution for PDE / scientific-computing outputs.

:class:`SolutionDistribution` mirrors :class:`PredictiveDistribution` but
keys every uncertainty leaf by field name (``{"u": ..., "p": ...}``) so a
solver's multi-field output can advertise per-field epistemic / aleatoric
breakdowns alongside conservation diagnostics.

Projection back to the canonical Phase 1 contract is via
:meth:`SolutionDistribution.as_predictive_distribution(field)` — that
return value carries the same eleven fields and the same variance-additivity
tolerance, so downstream calibration / conformal / metrics code consumes
either container without solver-specific awareness.

The container is a :func:`flax.struct.dataclass` so it registers as a JAX
PyTree (per-field array dicts flatten as data leaves; ``metadata`` is
declared ``pytree_node=False`` and stays static aux_data). Use
``solution.replace(field=value)`` for immutable updates.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct

from opifex.core.solver.interface import Solution
from opifex.uncertainty.types import (
    _VARIANCE_ATOL,
    _VARIANCE_RTOL,
    PredictionInterval,
    PredictionSet,
    PredictiveDistribution,
)


if TYPE_CHECKING:
    from opifex.uncertainty.types import MetadataItems


# The six canonical uncertainty-source labels a SolutionDistribution may
# declare in its metadata. Presence of any one source is per-solver; the
# constant pins the vocabulary callers and audits agree on.
UNCERTAINTY_SOURCES: tuple[str, ...] = (
    "numerical",
    "parameter",
    "observation",
    "model_discrepancy",
    "ensemble",
    "calibration",
)


_FieldArrays = dict[str, jax.Array]
_FieldArraysOptional = dict[str, jax.Array] | None
_FieldIntervals = dict[str, PredictionInterval] | None
_FieldSets = dict[str, PredictionSet] | None
_FieldQuantiles = dict[float, dict[str, jax.Array]]


@struct.dataclass(slots=True, kw_only=True)
class SolutionDistribution:
    """Per-field predictive distribution for a PDE solver output.

    Every uncertainty leaf is keyed by field name. The leaf dicts are
    pytree data; ``metadata`` is static aux_data (``pytree_node=False``).
    Reuse :meth:`as_predictive_distribution` to project a single field
    onto the canonical :class:`PredictiveDistribution` contract.
    """

    mean: _FieldArrays
    samples: _FieldArraysOptional = None
    variance: _FieldArraysOptional = None
    covariance: _FieldArraysOptional = None
    epistemic: _FieldArraysOptional = None
    aleatoric: _FieldArraysOptional = None
    total_uncertainty: _FieldArraysOptional = None
    quantiles: _FieldQuantiles = struct.field(pytree_node=False, default_factory=dict)
    interval: _FieldIntervals = None
    prediction_set: _FieldSets = None
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return ``metadata`` as a mutable :class:`dict` for ergonomic read."""
        return dict(self.metadata)

    def validate(self) -> None:
        """Public preflight check — never called from ``__post_init__``.

        Verifies that every supplied uncertainty leaf covers the same
        field keys as :attr:`mean`, and that
        ``total_uncertainty[k] == epistemic[k] + aleatoric[k]`` per
        field within the canonical ``_VARIANCE_RTOL`` / ``_VARIANCE_ATOL``
        tolerances. Mirrors :meth:`PredictiveDistribution.validate`.
        """
        expected_keys = set(self.mean.keys())
        leaves = {
            "samples": self.samples,
            "variance": self.variance,
            "covariance": self.covariance,
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "total_uncertainty": self.total_uncertainty,
        }
        for name, leaf in leaves.items():
            if leaf is not None and set(leaf.keys()) != expected_keys:
                raise ValueError(
                    f"Field keys must agree across mean and {name!r}; "
                    f"got mean keys {sorted(expected_keys)!r}, "
                    f"{name} keys {sorted(leaf.keys())!r}."
                )
        if (
            self.epistemic is not None
            and self.aleatoric is not None
            and self.total_uncertainty is not None
        ):
            for key in expected_keys:
                expected_total = self.epistemic[key] + self.aleatoric[key]
                if not bool(
                    jnp.allclose(
                        self.total_uncertainty[key],
                        expected_total,
                        rtol=_VARIANCE_RTOL,
                        atol=_VARIANCE_ATOL,
                    )
                ):
                    raise ValueError(
                        f"SolutionDistribution variance-additivity violation on field "
                        f"{key!r}: total_uncertainty != epistemic + aleatoric within "
                        f"rtol={_VARIANCE_RTOL}, atol={_VARIANCE_ATOL}."
                    )

    def as_predictive_distribution(self, field: str) -> PredictiveDistribution:
        """Project a single field onto the Phase 1 :class:`PredictiveDistribution`.

        Args:
            field: Field key (e.g. ``"u"``). Must exist in :attr:`mean`.

        Returns:
            :class:`PredictiveDistribution` with the same eleven leaves
            (per-field values pulled from the solver-side dicts).

        Raises:
            KeyError: If ``field`` is not present in :attr:`mean`.
        """
        if field not in self.mean:
            raise KeyError(
                f"as_predictive_distribution called with unknown field {field!r}; "
                f"available fields: {sorted(self.mean.keys())!r}."
            )

        def _project(leaf: _FieldArraysOptional) -> jax.Array | None:
            return None if leaf is None else leaf[field]

        interval = None if self.interval is None else self.interval.get(field)
        prediction_set = None if self.prediction_set is None else self.prediction_set.get(field)
        # Per-field quantiles re-keyed as alpha -> array.
        per_field_quantiles: dict[float, jax.Array] = {
            alpha: leaf[field] for alpha, leaf in self.quantiles.items() if field in leaf
        }
        return PredictiveDistribution(
            mean=self.mean[field],
            samples=_project(self.samples),
            variance=_project(self.variance),
            epistemic=_project(self.epistemic),
            aleatoric=_project(self.aleatoric),
            total_uncertainty=_project(self.total_uncertainty),
            quantiles=per_field_quantiles,
            interval=interval,
            prediction_set=prediction_set,
            metadata=self.metadata,
        )

    def to_solution(
        self,
        *,
        metrics: dict[str, Any] | None = None,
        execution_time: float = 0.0,
        converged: bool = False,
        stats: dict[str, Any] | None = None,
    ) -> Solution:
        """Return a :class:`Solution` carrying mean fields + UQ aux_data.

        Stores this distribution's per-field means in ``Solution.fields``
        and the full UQ payload (samples, variance, epistemic / aleatoric
        leaves, interval / set, metadata) under ``auxiliary_data["uq"]``.
        The source :class:`SolutionDistribution` is **not** mutated; a
        fresh :class:`Solution` is constructed. ``auxiliary_data["uq"]``
        carries the per-field uncertainty leaves and metadata so
        downstream consumers can recover the distribution without a
        backref to the caller's container.
        """
        auxiliary_data: dict[str, Any] = {
            "uq": {
                "samples": self.samples,
                "variance": self.variance,
                "covariance": self.covariance,
                "epistemic": self.epistemic,
                "aleatoric": self.aleatoric,
                "total_uncertainty": self.total_uncertainty,
                "quantiles": dict(self.quantiles),
                "interval": self.interval,
                "prediction_set": self.prediction_set,
                "metadata": self.metadata,
            }
        }
        return Solution(
            fields=dict(self.mean),
            metrics=dict(metrics) if metrics is not None else {},
            execution_time=execution_time,
            auxiliary_data=auxiliary_data,
            converged=converged,
            stats=dict(stats) if stats is not None else {},
        )
