"""Scientific field-level UQ metrics.

Hosts the canonical :class:`FieldMetadata` schema (Pattern A) and the
field/function-space UQ metrics: spatially aggregated calibration error,
function-space L2 coverage, conservation-law residual summary, and
residual-uncertainty alignment.

All kernels are pure :class:`jax.Array` transformations. The
:class:`FieldMetadata` dataclass is Pattern A (plain frozen dataclass)
since all fields are scalar / string / tuple-of-string with no array
data; it passes through ``jax.jit`` as a static argument.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp


_VALID_NORMS: frozenset[str] = frozenset({"L2", "Linf", "H1"})


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class FieldMetadata:
    """Canonical Pattern-A field metadata schema.

    Fields:
        grid_axes: Named axes that describe the field's grid layout
            (e.g. ``("x", "y")`` or ``("t", "x", "y")``).
        time_axis: Optional named time axis; ``None`` for static fields.
        spatial_axes: Integer axis indices that the field norm reduces
            over.
        norm: Field norm identifier; one of ``"L2"``, ``"Linf"``, ``"H1"``.
        alpha: Miscoverage level in ``(0, 1)``.
        calibration_size: Number of calibration samples used.
        assumption_status: Free-form short tag describing the validity
            assumption (``"exchangeable"``, ``"exchangeability_failed"``,
            ``"weights_required"``, etc.).
    """

    grid_axes: tuple[str, ...]
    time_axis: str | None
    spatial_axes: tuple[int, ...]
    norm: str
    alpha: float
    calibration_size: int
    assumption_status: str

    def __post_init__(self) -> None:
        """Validate ``norm`` against the supported set."""
        if self.norm not in _VALID_NORMS:
            raise ValueError(
                f"Unknown field norm {self.norm!r}; valid choices: {sorted(_VALID_NORMS)}"
            )


# ---------------------------------------------------------------------------
# Spatial calibration error
# ---------------------------------------------------------------------------


def spatial_calibration_error(
    *,
    lower: jax.Array,
    upper: jax.Array,
    targets: jax.Array,
    alpha: float,
    spatial_axes: tuple[int, ...],
) -> jax.Array:
    """Absolute miscalibration between empirical and nominal coverage.

    Computes ``|mean(in_interval) - (1 - alpha)|`` aggregated over both
    the batch and the supplied spatial axes.
    """
    covered = (targets >= lower) & (targets <= upper)
    reduction_axes = (0, *spatial_axes)
    empirical_coverage = jnp.mean(covered.astype(jnp.float32), axis=reduction_axes)
    return jnp.abs(empirical_coverage - (1.0 - alpha))


# ---------------------------------------------------------------------------
# Function-space L2 coverage
# ---------------------------------------------------------------------------


def function_space_l2_coverage(
    *,
    predictions: jax.Array,
    targets: jax.Array,
    threshold: jax.Array,
    spatial_axes: tuple[int, ...],
) -> jax.Array:
    """Fraction of samples whose L2 residual falls below ``threshold``.

    Args:
        predictions: Shape ``(batch, *spatial)``.
        targets: Shape ``(batch, *spatial)``.
        threshold: Scalar L2-norm threshold.
        spatial_axes: Axis indices to reduce over for the L2 norm.

    Returns:
        Scalar coverage fraction in ``[0, 1]``.
    """
    residual = targets - predictions
    l2 = jnp.sqrt(jnp.mean(residual * residual, axis=spatial_axes))
    covered = (l2 <= threshold).astype(jnp.float32)
    return jnp.mean(covered)


# ---------------------------------------------------------------------------
# Conservation-law residual summary
# ---------------------------------------------------------------------------


def conservation_law_residual_summary(
    *,
    residual_field: jax.Array,
    spatial_axes: tuple[int, ...],
) -> jax.Array:
    """L2 norm of a conservation-law residual field, per sample.

    Args:
        residual_field: Shape ``(batch, *spatial)``. Typically the result
            of evaluating a divergence operator on a learned solution.
        spatial_axes: Axis indices to reduce over.

    Returns:
        Per-sample L2 norms of shape ``(batch,)``.
    """
    return jnp.sqrt(jnp.mean(residual_field * residual_field, axis=spatial_axes))


# ---------------------------------------------------------------------------
# Residual-uncertainty alignment
# ---------------------------------------------------------------------------


def residual_uncertainty_alignment(
    *,
    predicted_uncertainty: jax.Array,
    observed_residual_magnitude: jax.Array,
) -> jax.Array:
    """Pearson correlation between predicted uncertainty and observed residual.

    Higher is better. ``1.0`` indicates the uncertainty perfectly tracks
    the residual magnitude; ``0`` indicates no alignment.

    Args:
        predicted_uncertainty: 1-D array of per-sample predicted
            uncertainty (e.g., std-dev or interval width).
        observed_residual_magnitude: 1-D array of per-sample residual
            magnitudes (e.g., ``|y_true - y_pred|``).

    Returns:
        Scalar Pearson correlation in ``[-1, 1]``.
    """
    x = predicted_uncertainty - jnp.mean(predicted_uncertainty)
    y = observed_residual_magnitude - jnp.mean(observed_residual_magnitude)
    numerator = jnp.sum(x * y)
    denominator = jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y)) + 1e-12
    return numerator / denominator
