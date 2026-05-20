"""Field / function-space UQ metric contracts.

Tests pin:

* :class:`opifex.uncertainty.scientific.fields.FieldMetadata` is the
  canonical Pattern-A frozen dataclass (migrated from the Phase 4
  temporary ``_FieldConformalMetadata``).
* Spatially aggregated calibration error reports per-spatial-axis
  miscalibration.
* Function-space L2 coverage measures the fraction of test samples whose
  L2 residual lies inside the prediction interval.
* Conservation-law residual summary aggregates a divergence/residual
  field across spatial axes.
* Residual-uncertainty alignment is higher when uncertainty tracks the
  residual magnitude.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_scientific():
    from opifex.uncertainty.scientific import fields

    return fields


# ---------------------------------------------------------------------------
# FieldMetadata
# ---------------------------------------------------------------------------


def test_field_metadata_is_frozen_pattern_a_dataclass() -> None:
    fields = _import_scientific()
    md = fields.FieldMetadata(
        grid_axes=("x", "y"),
        time_axis=None,
        spatial_axes=(-2, -1),
        norm="L2",
        alpha=0.1,
        calibration_size=256,
        assumption_status="exchangeable",
    )
    assert md.norm == "L2"
    assert md.grid_axes == ("x", "y")
    with pytest.raises(dc.FrozenInstanceError):
        md.norm = "Linf"  # type: ignore[misc]


def test_field_metadata_accepts_optional_time_axis() -> None:
    fields = _import_scientific()
    md = fields.FieldMetadata(
        grid_axes=("t", "x"),
        time_axis="t",
        spatial_axes=(-1,),
        norm="Linf",
        alpha=0.05,
        calibration_size=128,
        assumption_status="exchangeable_assumed",
    )
    assert md.time_axis == "t"


def test_field_metadata_rejects_invalid_norm_string() -> None:
    fields = _import_scientific()
    with pytest.raises(ValueError, match=r"(?i)norm"):
        fields.FieldMetadata(
            grid_axes=("x",),
            time_axis=None,
            spatial_axes=(-1,),
            norm="invalid",
            alpha=0.1,
            calibration_size=10,
            assumption_status="exchangeable",
        )


# ---------------------------------------------------------------------------
# Spatially aggregated calibration error
# ---------------------------------------------------------------------------


def test_spatial_calibration_error_per_axis_matches_per_pixel_picp() -> None:
    """Mean miscalibration over spatial axes equals
    ``|mean(in_interval) - (1 - alpha)|``."""
    fields = _import_scientific()
    rng = np.random.default_rng(0)
    n_calib = 64
    spatial = (8, 8)
    targets = jnp.asarray(rng.standard_normal((n_calib, *spatial)))
    lower = targets - 0.5
    upper = targets + 0.5  # 100% coverage by construction
    err = float(
        fields.spatial_calibration_error(
            lower=lower, upper=upper, targets=targets, alpha=0.1, spatial_axes=(-2, -1)
        )
    )
    assert err == pytest.approx(0.1, abs=0.05)  # |1.0 - 0.9| = 0.1


# ---------------------------------------------------------------------------
# Function-space L2 coverage
# ---------------------------------------------------------------------------


def test_function_space_l2_coverage_counts_samples_within_threshold() -> None:
    """L2(residual) = sqrt(mean(residual^2)) over spatial axes.

    For 4x4 residual field with constant value ``c``, L2 = ``|c|`` (since
    mean(c²) = c² → sqrt = |c|). Set targets so per-sample L2 lies on
    either side of threshold=1.0:
        sample 0: c=0.0 → L2=0.0  < 1.0  → covered
        sample 1: c=2.0 → L2=2.0  > 1.0  → not covered
        sample 2: c=0.5 → L2=0.5  < 1.0  → covered
        sample 3: c=1.5 → L2=1.5  > 1.0  → not covered
    """
    fields = _import_scientific()
    predictions = jnp.zeros((4, 4, 4))
    targets = jnp.stack(
        [
            jnp.zeros((4, 4)),
            jnp.full((4, 4), 2.0),
            jnp.full((4, 4), 0.5),
            jnp.full((4, 4), 1.5),
        ]
    )
    coverage = float(
        fields.function_space_l2_coverage(
            predictions=predictions,
            targets=targets,
            threshold=jnp.asarray(1.0),
            spatial_axes=(-2, -1),
        )
    )
    assert coverage == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Conservation-law residual summary
# ---------------------------------------------------------------------------


def test_conservation_law_residual_summary_aggregates_field_norm() -> None:
    """For divergence field with known L2, the summary equals that norm."""
    fields = _import_scientific()
    # 2 samples, 4x4 spatial. Sample 0: zeros. Sample 1: ones.
    residual_field = jnp.stack([jnp.zeros((4, 4)), jnp.ones((4, 4))])
    out = fields.conservation_law_residual_summary(
        residual_field=residual_field, spatial_axes=(-2, -1)
    )
    assert out.shape == (2,)
    assert float(out[0]) == pytest.approx(0.0, abs=1e-6)
    # L2 norm of all-ones (4x4) = sqrt(mean(1)) = 1.0
    assert float(out[1]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Residual-uncertainty alignment
# ---------------------------------------------------------------------------


def test_residual_uncertainty_alignment_is_high_when_uncertainty_tracks_residual() -> None:
    """Pearson correlation between predicted uncertainty and observed residual
    magnitude is the canonical alignment score; perfect alignment → 1.0."""
    fields = _import_scientific()
    rng = np.random.default_rng(0)
    n = 256
    residual_magnitude = jnp.asarray(rng.uniform(0.1, 1.0, size=(n,)))
    perfectly_aligned_uncertainty = residual_magnitude.copy()
    score_aligned = float(
        fields.residual_uncertainty_alignment(
            predicted_uncertainty=perfectly_aligned_uncertainty,
            observed_residual_magnitude=residual_magnitude,
        )
    )
    misaligned_uncertainty = jnp.asarray(rng.uniform(0.1, 1.0, size=(n,)))
    score_misaligned = float(
        fields.residual_uncertainty_alignment(
            predicted_uncertainty=misaligned_uncertainty,
            observed_residual_magnitude=residual_magnitude,
        )
    )
    assert score_aligned > 0.99
    assert score_aligned > score_misaligned


# ---------------------------------------------------------------------------
# Phase 4 → Phase 5 migration (no remaining TODOs)
# ---------------------------------------------------------------------------


def test_phase4_field_module_no_longer_carries_phase5_todo() -> None:
    """After this task migrates the metadata dataclass, no
    ``TODO(phase5-task-5.2)`` tag may remain in the codebase."""
    import subprocess

    result = subprocess.run(
        [
            "rg",
            "-n",
            r"TODO\(phase5-task-5\.2\)",
            "src/opifex/uncertainty",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    matches = result.stdout.strip()
    assert matches == "", f"Lingering TODO(phase5-task-5.2) markers: {matches!r}"


def test_phase4_temporary_field_conformal_metadata_is_deleted() -> None:
    """The Phase-4 temporary ``_FieldConformalMetadata`` symbol must be gone."""
    import subprocess

    result = subprocess.run(
        [
            "rg",
            "-n",
            r"_FieldConformalMetadata\b",
            "src/opifex/uncertainty",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    matches = result.stdout.strip()
    assert matches == "", f"Lingering _FieldConformalMetadata references: {matches!r}"


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_field_metrics_are_jit_compatible() -> None:
    fields = _import_scientific()
    rng = np.random.default_rng(0)
    predictions = jnp.asarray(rng.standard_normal((4, 8, 8)))
    targets = predictions + 0.1 * jnp.asarray(rng.standard_normal((4, 8, 8)))
    jitted = jax.jit(
        lambda p, t: fields.function_space_l2_coverage(
            predictions=p, targets=t, threshold=jnp.asarray(1.0), spatial_axes=(-2, -1)
        )
    )
    out = jitted(predictions, targets)
    assert bool(jnp.isfinite(out))
