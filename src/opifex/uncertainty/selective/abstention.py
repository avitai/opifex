"""Abstention decision: accept / reject masks under a confidence threshold."""

from __future__ import annotations

import dataclasses as dc
from typing import Any

import jax
import jax.numpy as jnp

from opifex.uncertainty.types import MetadataItems  # noqa: TC001


@jax.tree_util.register_pytree_node_class
@dc.dataclass(frozen=True, slots=True, kw_only=True)
class AbstentionDecision:
    """Outcome of an abstention rule.

    Array entries inside ``metadata`` (such as ``threshold``, ``num_accepted``,
    and ``num_rejected``) are flattened as PyTree array leaves while string/scalar
    tags stay in static auxiliary metadata, keeping ``jax.jit`` and eager treedefs
    identical and avoiding downstream recompilations when counts vary.

    Fields:
        accepted_mask: Boolean array; ``True`` where the sample passes
            the confidence threshold.
        rejected_mask: Boolean array; ``True`` where the sample is
            abstained from.
        threshold: Scalar confidence threshold used for the decision.
        metadata: Metadata tuple ``(method, threshold, num_accepted, num_rejected)``.
    """

    accepted_mask: jax.Array
    rejected_mask: jax.Array
    threshold: jax.Array | float
    metadata: MetadataItems = ()

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array | float, tuple[jax.Array, ...]],
        tuple[tuple[str, bool, Any], ...],
    ]:
        """Split array leaves (including array-valued metadata) from static keys."""
        dyn_vals: list[jax.Array] = []
        spec: list[tuple[str, bool, Any]] = []
        for key, val in self.metadata:
            if isinstance(val, jax.Array):
                spec.append((key, True, None))
                dyn_vals.append(val)
            else:
                spec.append((key, False, val))
        children = (self.accepted_mask, self.rejected_mask, self.threshold, tuple(dyn_vals))
        return children, tuple(spec)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[tuple[str, bool, Any], ...],
        children: tuple[jax.Array, jax.Array, jax.Array | float, tuple[jax.Array, ...]],
    ) -> AbstentionDecision:
        """Reconstruct :class:`AbstentionDecision` from PyTree children and static keys."""
        accepted_mask, rejected_mask, threshold, dyn_vals = children
        val_iter = iter(dyn_vals)
        meta: MetadataItems = tuple(
            (key, next(val_iter) if is_dyn else static_val) for key, is_dyn, static_val in aux_data
        )
        return cls(
            accepted_mask=accepted_mask,
            rejected_mask=rejected_mask,
            threshold=threshold,
            metadata=meta,
        )


def abstention_decision(
    *,
    confidences: jax.Array,
    threshold: jax.Array | float,
) -> AbstentionDecision:
    """Accept samples whose confidence meets or exceeds ``threshold``.

    Args:
        confidences: Array of per-sample confidence scores.
        threshold: Scalar accept/reject cutoff; samples with
            ``confidence >= threshold`` are accepted.

    Returns:
        :class:`AbstentionDecision` with named ``accepted_mask`` /
        ``rejected_mask`` arrays and metadata recording the threshold
        and accept/reject counts as traceable ``jax.Array`` leaves.
    """
    dtype = jnp.result_type(confidences, threshold, jnp.float32)
    conf_f = jnp.asarray(confidences, dtype=dtype)
    thresh_arr = jnp.asarray(threshold, dtype=dtype)
    accepted_mask = conf_f >= thresh_arr
    rejected_mask = ~accepted_mask
    num_accepted = jnp.sum(accepted_mask.astype(jnp.int32))
    num_rejected = jnp.sum(rejected_mask.astype(jnp.int32))
    metadata: MetadataItems = (
        ("method", "threshold_abstention"),
        ("threshold", thresh_arr),
        ("num_accepted", num_accepted),
        ("num_rejected", num_rejected),
    )
    return AbstentionDecision(
        accepted_mask=accepted_mask,
        rejected_mask=rejected_mask,
        threshold=thresh_arr,
        metadata=metadata,
    )
