"""Accuracy measures shared by the state-space discretisation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax


def scaled_error(
    estimate: jax.Array | np.ndarray, reference: Sequence[Sequence[float]] | np.ndarray
) -> float:
    """Return ``max |Q_ij - R_ij| / sqrt(R_ii R_jj)``, the per-component relative error."""
    reference_array = np.asarray(reference, dtype=np.float64)
    scale = np.sqrt(np.diag(reference_array))
    difference = np.asarray(estimate, dtype=np.float64) - reference_array
    return float(np.max(np.abs(difference) / np.outer(scale, scale)))


def transition_error(
    estimate: jax.Array | np.ndarray, reference: Sequence[Sequence[float]] | np.ndarray
) -> float:
    """Return ``max |A_ij - R_ij| / max(1, max |R|)``."""
    reference_array = np.asarray(reference, dtype=np.float64)
    difference = np.asarray(estimate, dtype=np.float64) - reference_array
    return float(np.max(np.abs(difference)) / max(1.0, float(np.max(np.abs(reference_array)))))


def relative_frobenius_error(
    estimate: jax.Array | np.ndarray, reference: Sequence[Sequence[float]] | np.ndarray
) -> float:
    """Return ``||Q - R||_F / ||R||_F``."""
    reference_array = np.asarray(reference, dtype=np.float64)
    difference = np.asarray(estimate, dtype=np.float64) - reference_array
    return float(np.linalg.norm(difference) / np.linalg.norm(reference_array))
