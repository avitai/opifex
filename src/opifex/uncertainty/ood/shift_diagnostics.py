"""Distribution-shift diagnostics for residual streams.

Reuses :func:`opifex.uncertainty.conformal.ks_two_sample_pvalue` and
packages the output as a typed :class:`ShiftReport`. Diagnostics return
explicit status + metadata so downstream consumers cannot silently
overclaim distribution-free coverage when shift is detected.
"""

from __future__ import annotations

import jax  # noqa: TC002 — kept eager for consistency with the rest of opifex.uncertainty
from flax import struct

from opifex.uncertainty.conformal.exchangeability import ks_two_sample_pvalue
from opifex.uncertainty.types import MetadataItems  # noqa: TC001


@struct.dataclass(slots=True, kw_only=True)
class ShiftReport:
    """Outcome of a residual-stream shift diagnostic."""

    p_value: jax.Array
    passes: jax.Array
    method: str = struct.field(pytree_node=False, default="ks_two_sample_residual")
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


def residual_shift_diagnostic(
    *,
    reference_residuals: jax.Array,
    observed_residuals: jax.Array,
    alpha: float = 0.05,
) -> ShiftReport:
    """Compare observed residuals to a reference distribution via two-sample KS.

    The report is a pytree whose leaves are the p-value and the pass flag, and whose
    static metadata records only what does not depend on the outcome. That keeps one
    structure for every result, so a report may be handed to compiled code without
    forcing a recompilation per outcome. The outcome rendered as text comes from
    :func:`shift_status`, on the host, where strings belong.

    Args:
        reference_residuals: 1-D array of historic / calibration-set
            residuals.
        observed_residuals: 1-D array of new evaluation residuals.
        alpha: Significance level; ``passes = p_value > alpha``.

    Returns:
        :class:`ShiftReport` with the p-value, the pass flag, and metadata recording the
        level and the two sample sizes.

    """
    p_value = ks_two_sample_pvalue(
        calibration_scores=reference_residuals,
        evaluation_scores=observed_residuals,
    )
    metadata: MetadataItems = (
        ("alpha", float(alpha)),
        ("reference_size", int(reference_residuals.shape[0])),
        ("observed_size", int(observed_residuals.shape[0])),
    )
    return ShiftReport(p_value=p_value, passes=p_value > alpha, metadata=metadata)


def shift_status(report: ShiftReport) -> tuple[str, str]:
    """The outcome of a shift diagnostic, rendered for a person.

    Derived rather than stored: a status string held in the report's static metadata would
    differ between a pass and a failure, giving the two outcomes different pytree
    structures and a recompilation apiece.

    Args:
        report: A completed diagnostic.

    Returns:
        ``(status, assumption_status)``.
    """
    if bool(report.passes):
        return "no_shift", "exchangeable"
    return "shift_detected", "shift_detected"
