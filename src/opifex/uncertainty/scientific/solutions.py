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

Two utility functions in this module replace the previous solver-side
``*Wrapper`` shim classes:

* :func:`aggregate_solver_solutions` — turns a sequence of
  :class:`Solution` objects (replays of a stochastic solver, ensemble
  outputs, …) into a single :class:`Solution` whose
  ``auxiliary_data["uq"]`` carries the
  :class:`SolutionDistribution`-shaped payload. Supersedes
  ``BayesianWrapper`` / ``ConformalWrapper`` / ``EnsembleWrapper``.
* :func:`summarize_stacked_sample_solution` — handles the *generative*
  case where a single base solver returns one :class:`Solution` whose
  fields already carry a leading sample axis. Supersedes
  ``GenerativeWrapper``.

The previous wrapper classes mostly added 3–5 lines of stacking +
mean/variance on top of the base solver; the function pair replaces
them with no shim layer.
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
    from collections.abc import Sequence

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


def aggregate_solver_solutions(
    solutions: Sequence[Solution],
    *,
    metadata: tuple[tuple[str, object], ...] = (),
    quantiles: tuple[float, ...] = (),
) -> Solution:
    """Aggregate a sequence of :class:`Solution` objects into a single ``Solution``.

    Per-field arrays are stacked along a new leading sample axis and
    summarised with their mean and (unbiased ``ddof=1``) variance. The
    raw stacked samples are preserved under ``auxiliary_data["uq"]``
    along with the metadata tuple, so consumers can recompute alternate
    quantiles or pull individual replays without re-running the solver.

    Replaces the four solver-side ``*Wrapper`` shim classes that
    previously sat under ``opifex.solvers.wrappers``. Callers wanting to
    MC-replay a stochastic solver write the explicit replay loop and
    pass the resulting list of solutions to this function; ensembles of
    independent solvers and replay over a single stochastic solver both
    reduce to the same call. For *generative* solvers that emit a
    single :class:`Solution` whose fields are already pre-stacked
    sample arrays, use :func:`summarize_stacked_sample_solution`
    instead.

    Args:
        solutions: Non-empty sequence of solutions with identical field
            keys.
        metadata: Extra ``(key, value)`` entries appended to the
            :class:`SolutionDistribution.metadata` tuple alongside the
            canonical ``("uncertainty_sources", ("ensemble",))`` entry.
        quantiles: Optional sequence of quantile levels in ``(0, 1)``.
            When non-empty, per-field quantile arrays are stored under
            ``auxiliary_data["uq"]["quantiles"]`` keyed by level.

    Returns:
        A fresh :class:`Solution` whose ``fields`` carry per-field means,
        ``metrics`` carries an ``"ensemble_size"`` entry and the merged
        metrics of the first input solution, and ``auxiliary_data["uq"]``
        carries the full :class:`SolutionDistribution`-shaped payload
        (samples, variance, metadata, quantiles).

    Raises:
        ValueError: When ``solutions`` is empty, field keys disagree,
            or a requested quantile is outside ``(0, 1)``.
    """
    if not solutions:
        raise ValueError("aggregate_solver_solutions requires at least one solution.")
    for q in quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile levels must lie in (0, 1); got {q!r}.")

    base_keys = set(solutions[0].fields.keys())
    for sol in solutions[1:]:
        if set(sol.fields.keys()) != base_keys:
            raise ValueError(
                "Every solution must expose the same field keys; got "
                f"{set(sol.fields.keys())!r} vs {base_keys!r}."
            )

    samples_per_field: dict[str, jax.Array] = {
        key: jnp.stack([s.fields[key] for s in solutions], axis=0) for key in base_keys
    }
    means: dict[str, jax.Array] = {
        key: jnp.mean(samples, axis=0) for key, samples in samples_per_field.items()
    }
    n = len(solutions)
    variances: dict[str, jax.Array] = {
        key: jnp.var(samples, axis=0, ddof=1) if n > 1 else jnp.zeros_like(means[key])
        for key, samples in samples_per_field.items()
    }
    quantile_map: dict[float, dict[str, jax.Array]] = {
        float(q): {
            key: jnp.quantile(samples, q, axis=0) for key, samples in samples_per_field.items()
        }
        for q in quantiles
    }

    distribution = SolutionDistribution(
        mean=means,
        samples=samples_per_field,
        variance=variances,
        quantiles=quantile_map,
        metadata=(
            ("uncertainty_sources", ("ensemble",)),
            ("ensemble_size", int(n)),
            *metadata,
        ),
    )
    merged_metrics = dict(solutions[0].metrics)
    merged_metrics["ensemble_size"] = n
    total_time = sum(s.execution_time for s in solutions)
    return distribution.to_solution(
        metrics=merged_metrics,
        execution_time=total_time,
        converged=all(s.converged for s in solutions),
    )


def summarize_stacked_sample_solution(
    solution: Solution,
    *,
    sample_axis: int = 0,
    metadata: tuple[tuple[str, object], ...] = (),
    quantiles: tuple[float, ...] = (),
) -> Solution:
    """Compute per-field statistics over a Solution whose fields are pre-stacked samples.

    Used for *generative* solvers that emit a single :class:`Solution`
    whose multi-dimensional field arrays already carry a leading sample
    axis (shape ``(num_samples, *field_shape)``). The wrapper computes
    mean / unbiased variance / optional quantile bands along
    ``sample_axis`` and packages everything into a fresh
    :class:`Solution` whose ``auxiliary_data["uq"]`` carries the
    :class:`SolutionDistribution`-shaped payload.

    Scalar fields are passed through unchanged.

    Args:
        solution: Generative-solver output. Non-scalar fields must share
            the same size along ``sample_axis``.
        sample_axis: Axis carrying the sample index. Default ``0``.
        metadata: Extra ``(key, value)`` metadata entries.
        quantiles: Optional quantile levels in ``(0, 1)``.

    Returns:
        A fresh :class:`Solution` with per-field means in ``fields`` and
        per-field samples / variance / quantiles in
        ``auxiliary_data["uq"]``.

    Raises:
        ValueError: When a requested quantile is outside ``(0, 1)`` or
            when two non-scalar fields disagree on the sample-axis
            length.
    """
    for q in quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile levels must lie in (0, 1); got {q!r}.")

    samples_per_field: dict[str, jax.Array] = {}
    scalar_fields: dict[str, jax.Array] = {}
    for key, value in solution.fields.items():
        if value.ndim > 0:
            samples_per_field[key] = jnp.moveaxis(value, sample_axis, 0)
        else:
            scalar_fields[key] = value

    sample_sizes = {key: arr.shape[0] for key, arr in samples_per_field.items()}
    if len(set(sample_sizes.values())) > 1:
        raise ValueError(f"Non-scalar fields must share sample-axis length; got {sample_sizes!r}.")

    means: dict[str, jax.Array] = {
        **scalar_fields,
        **{key: jnp.mean(samples, axis=0) for key, samples in samples_per_field.items()},
    }
    if samples_per_field:
        first = next(iter(samples_per_field.values()))
        ddof = 1 if first.shape[0] > 1 else 0
        variances: dict[str, jax.Array] | None = {
            key: jnp.var(samples, axis=0, ddof=ddof) for key, samples in samples_per_field.items()
        }
        sample_block: dict[str, jax.Array] | None = samples_per_field
    else:
        variances = None
        sample_block = None
    quantile_map: dict[float, dict[str, jax.Array]] = {
        float(q): {
            key: jnp.quantile(samples, q, axis=0) for key, samples in samples_per_field.items()
        }
        for q in quantiles
    }

    distribution = SolutionDistribution(
        mean=means,
        samples=sample_block,
        variance=variances,
        quantiles=quantile_map,
        metadata=(
            ("uncertainty_sources", ("ensemble",)),
            ("method", "generative_sampling"),
            *metadata,
        ),
    )
    merged_metrics = dict(solution.metrics)
    merged_metrics["uq_method"] = "generative_sampling"
    if "log_likelihood" in merged_metrics:
        merged_metrics["mean_log_likelihood"] = merged_metrics["log_likelihood"]
    return distribution.to_solution(
        metrics=merged_metrics,
        execution_time=solution.execution_time,
        converged=solution.converged,
    )
