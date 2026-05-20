"""Distribution-shift diagnostics for residual streams.

Reuses :func:`opifex.uncertainty.conformal.ks_two_sample_pvalue` and
packages the output as a typed :class:`ShiftReport`. Diagnostics return
explicit status + metadata so downstream consumers cannot silently
overclaim distribution-free coverage when shift is detected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax  # noqa: TC002 — kept eager for consistency with the rest of opifex.uncertainty
from flax import struct

from opifex.uncertainty.conformal.exchangeability import ks_two_sample_pvalue


if TYPE_CHECKING:
    from opifex.uncertainty.types import MetadataItems


@struct.dataclass(slots=True, kw_only=True)
class ShiftReport:
    """Outcome of a residual-stream shift diagnostic."""

    p_value: jax.Array
    passes: bool = struct.field(pytree_node=False)
    method: str = struct.field(pytree_node=False, default="ks_two_sample_residual")
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


def residual_shift_diagnostic(
    *,
    reference_residuals: jax.Array,
    observed_residuals: jax.Array,
    alpha: float = 0.05,
) -> ShiftReport:
    """Compare observed residuals to a reference distribution via two-sample KS.

    Args:
        reference_residuals: 1-D array of historic / calibration-set
            residuals.
        observed_residuals: 1-D array of new evaluation residuals.
        alpha: Significance level; ``passes = p_value > alpha``.

    Returns:
        :class:`ShiftReport` with the p-value, pass flag, and metadata
        recording method + sample sizes + status / assumption flags.
    """
    p_value = ks_two_sample_pvalue(
        calibration_scores=reference_residuals,
        evaluation_scores=observed_residuals,
    )
    passes = bool(float(p_value) > alpha)
    status = "no_shift" if passes else "shift_detected"
    assumption_status = "exchangeable" if passes else "shift_detected"
    metadata: MetadataItems = (
        ("method", "ks_two_sample_residual"),
        ("alpha", float(alpha)),
        ("reference_size", int(reference_residuals.shape[0])),
        ("observed_size", int(observed_residuals.shape[0])),
        ("status", status),
        ("assumption_status", assumption_status),
    )
    return ShiftReport(p_value=p_value, passes=passes, metadata=metadata)
