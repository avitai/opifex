"""Conformal score container and shared helpers.

The fitted-state convention for conformal calibrators (Pattern B,
``@flax.struct.dataclass(slots=True, kw_only=True)``) lives next to the
concrete regressors in :mod:`opifex.uncertainty.conformal.regression`. This
module provides the generic :class:`ConformalScore` value object used by
diagnostic and score-routing helpers; concrete regressor states (e.g.
:class:`SplitConformalState`) live next to the regressor that consumes
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import struct


if TYPE_CHECKING:
    import jax

    from opifex.uncertainty.types import MetadataItems


@struct.dataclass(slots=True, kw_only=True)
class ConformalScore:
    """Generic nonconformity-score container.

    ``scores`` carries the raw per-sample nonconformity values; ``score_type``
    is the canonical string identifier
    (``"absolute_residual"``, ``"cqr"``, etc.) that a downstream consumer can
    branch on without inspecting numerical content.
    """

    scores: jax.Array
    score_type: str = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())
