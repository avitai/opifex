# ruff: noqa: UP037
"""Calibration assessment utilities and reliability-binning helpers.

Hosts the JIT-safe ``_bin_calibration_stats`` masked-accumulation helper used
by both the basic ``UncertaintyQuantifier`` and ``CalibrationAssessment``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.aggregators.types import CalibrationMetrics


if TYPE_CHECKING:
    from jaxtyping import Array, Float


def _bin_calibration_stats(
    *,
    confidences: Float[Array, "n_samples"],  # noqa: F821
    accuracies: Float[Array, "n_samples"],  # noqa: F821
    bin_boundaries: Float[Array, "n_bins_plus_1"],  # noqa: F821
) -> tuple[Array, Array, Array]:  # type: ignore[reportUndefinedVariable]
    """Vectorised reliability-bin statistics.

    For each bin ``b`` covering ``[lo_b, hi_b)`` (last bin closed),
    returns the mean confidence, mean accuracy, and sample count using
    pure ``jnp.where`` masked accumulation. No Python branches on traced
    arrays, no boolean fancy-indexing — traces under ``jax.jit``.
    """
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    n_bins = bin_lowers.shape[0]
    # (n_samples, n_bins) bin-membership mask.
    in_bin = (confidences[:, None] >= bin_lowers[None, :]) & (
        confidences[:, None] < bin_uppers[None, :]
    )
    # Close the rightmost bin on the upper edge.
    last_bin = jax.nn.one_hot(n_bins - 1, n_bins, dtype=jnp.bool_)
    closed_right = (confidences[:, None] == bin_uppers[None, :]) & last_bin[None, :]
    in_bin = in_bin | closed_right
    in_bin_f = in_bin.astype(jnp.float32)
    counts = jnp.sum(in_bin_f, axis=0)
    safe_counts = jnp.maximum(counts, 1.0)
    bin_confidences = jnp.sum(in_bin_f * confidences[:, None], axis=0) / safe_counts
    bin_accuracies = jnp.sum(in_bin_f * accuracies[:, None], axis=0) / safe_counts
    # Zero-out bins that had no samples so callers can detect empty bins via counts.
    nonempty = counts > 0
    bin_confidences = jnp.where(nonempty, bin_confidences, 0.0)
    bin_accuracies = jnp.where(nonempty, bin_accuracies, 0.0)
    return bin_confidences, bin_accuracies, counts


class CalibrationAssessment:
    """Enhanced uncertainty calibration assessment tools."""

    @staticmethod
    def expected_calibration_error(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> float:  # type: ignore[reportUndefinedVariable]
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = jnp.linspace(0.0, 1.0, n_bins + 1)
        bin_confidences, bin_accuracies, counts = _bin_calibration_stats(
            confidences=confidences, accuracies=accuracies, bin_boundaries=bin_boundaries
        )
        total = jnp.maximum(jnp.sum(counts), 1.0)
        bin_weights = counts / total
        ece = jnp.sum(bin_weights * jnp.abs(bin_confidences - bin_accuracies))
        return float(ece)

    @staticmethod
    def maximum_calibration_error(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> float:  # type: ignore[reportUndefinedVariable]
        """Compute Maximum Calibration Error (MCE)."""
        bin_boundaries = jnp.linspace(0.0, 1.0, n_bins + 1)
        bin_confidences, bin_accuracies, counts = _bin_calibration_stats(
            confidences=confidences, accuracies=accuracies, bin_boundaries=bin_boundaries
        )
        # Mask empty bins out of the max — set their error to -inf so they
        # never win the argmax / max reduction.
        errors = jnp.abs(bin_confidences - bin_accuracies)
        errors = jnp.where(counts > 0, errors, -jnp.inf)
        return float(jnp.max(errors))

    @staticmethod
    def reliability_diagram_data(
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> dict[str, Array]:  # type: ignore[reportUndefinedVariable]
        """Compute reliability diagram data for visualization."""
        bin_boundaries = jnp.linspace(0.0, 1.0, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2
        bin_confidences, bin_accuracies, counts = _bin_calibration_stats(
            confidences=confidences, accuracies=accuracies, bin_boundaries=bin_boundaries
        )
        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": counts,
        }

    def assess_calibration(
        self,
        confidences: Float[Array, "n_samples"],  # noqa: F821
        accuracies: Float[Array, "n_samples"],  # noqa: F821
        n_bins: int = 10,
    ) -> CalibrationMetrics:  # type: ignore[reportUndefinedVariable]
        """Assess overall calibration with multiple metrics."""
        ece = self.expected_calibration_error(confidences, accuracies, n_bins)
        mce = self.maximum_calibration_error(confidences, accuracies, n_bins)
        rel_data = self.reliability_diagram_data(confidences, accuracies, n_bins)

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            reliability_diagram=rel_data,
            confidence_histogram=rel_data["bin_confidences"],
            accuracy_histogram=rel_data["bin_accuracies"],
        )
