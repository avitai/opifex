"""Abstention decision: accept / reject masks under a confidence threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct


if TYPE_CHECKING:
    from opifex.uncertainty.types import MetadataItems


@struct.dataclass(slots=True, kw_only=True)
class AbstentionDecision:
    """Outcome of an abstention rule.

    Fields:
        accepted_mask: Boolean array; ``True`` where the sample passes
            the confidence threshold.
        rejected_mask: Boolean array; ``True`` where the sample is
            abstained from.
        threshold: Scalar confidence threshold used for the decision.
        metadata: Static metadata tuple (method, counts, threshold).
    """

    accepted_mask: jax.Array
    rejected_mask: jax.Array
    threshold: float = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


def abstention_decision(
    *,
    confidences: jax.Array,
    threshold: float,
) -> AbstentionDecision:
    """Accept samples whose confidence meets or exceeds ``threshold``.

    Args:
        confidences: 1-D array of per-sample confidence scores.
        threshold: Scalar accept/reject cutoff; samples with
            ``confidence >= threshold`` are accepted.

    Returns:
        :class:`AbstentionDecision` with named ``accepted_mask`` /
        ``rejected_mask`` arrays and metadata recording the threshold
        and per-class counts.
    """
    accepted_mask = confidences >= threshold
    rejected_mask = ~accepted_mask
    num_accepted = int(jnp.sum(accepted_mask.astype(jnp.int32)))
    num_rejected = int(confidences.shape[0] - num_accepted)
    metadata: MetadataItems = (
        ("method", "threshold_abstention"),
        ("threshold", float(threshold)),
        ("num_accepted", num_accepted),
        ("num_rejected", num_rejected),
    )
    return AbstentionDecision(
        accepted_mask=accepted_mask,
        rejected_mask=rejected_mask,
        threshold=float(threshold),
        metadata=metadata,
    )
