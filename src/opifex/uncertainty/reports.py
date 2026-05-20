"""Aggregated UQ reliability report.

:class:`UQReliabilityReport` is a Pattern-B
``flax.struct.dataclass(slots=True, kw_only=True)`` that collects
already-computed metric values from the calibration / conformal /
forecasting / field / OOD / selective subsystems into a single
serializable container with explicit provenance metadata.

The report is a data class + serializer — never an evaluator. It does
not recompute metrics from raw predictions; callers compute metrics
first (via the canonical subsystem kernels) and pass the results in.

``validate()`` is the public preflight check that ensures at least one
metric is populated. Per the project pytree convention, it is NOT called
from ``__post_init__`` or ``tree_unflatten``.

Convention: failed exchangeability / shift / OOD diagnostics from
Phase 4 (`opifex.uncertainty.conformal`) and Task 5.3
(`opifex.uncertainty.ood`) propagate verbatim into the report metadata
under the ``assumption_warnings`` and ``assumption_status`` keys so
downstream consumers cannot silently convert failed shift checks into
green coverage claims.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax  # noqa: TC002 — kept eager for consistency with the rest of opifex.uncertainty
from flax import struct


if TYPE_CHECKING:
    from opifex.uncertainty.types import MetadataItems


@struct.dataclass(slots=True, kw_only=True)
class UQReliabilityReport:
    """Aggregated UQ reliability report.

    All metric leaves are optional; callers populate the subset they
    have computed. ``metadata`` carries static provenance + assumption
    warnings.
    """

    calibration_ece: jax.Array | None = None
    calibration_nll: jax.Array | None = None
    brier_score: jax.Array | None = None
    empirical_coverage: jax.Array | None = None
    mean_interval_width: jax.Array | None = None
    interval_score: jax.Array | None = None
    crps: jax.Array | None = None
    energy_score: jax.Array | None = None
    spread_skill_ratio: jax.Array | None = None
    ood_auroc: jax.Array | None = None
    ood_auprc: jax.Array | None = None
    ood_fpr95: jax.Array | None = None
    aurc: jax.Array | None = None
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    _METRIC_FIELDS = (
        "calibration_ece",
        "calibration_nll",
        "brier_score",
        "empirical_coverage",
        "mean_interval_width",
        "interval_score",
        "crps",
        "energy_score",
        "spread_skill_ratio",
        "ood_auroc",
        "ood_auprc",
        "ood_fpr95",
        "aurc",
    )

    def validate(self) -> None:
        """Raise ``ValueError`` when no metric leaf is populated.

        Public preflight check — NOT called from ``__post_init__`` or
        the pytree unflatten path so transforms that reconstruct the
        container with placeholder values do not spuriously fail.
        """
        populated = [name for name in self._METRIC_FIELDS if getattr(self, name) is not None]
        if not populated:
            raise ValueError(
                "UQReliabilityReport requires at least one populated metric; "
                "all metric fields are None. Populate via fit / predict outputs "
                "from opifex.uncertainty.{calibration, conformal, forecasting_metrics, "
                "ood, selective}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible mapping.

        Scalar arrays are converted to Python ``float``s; nested
        metadata tuples become a dict. Lists are preserved when a value
        is sequence-like.
        """
        payload: dict[str, Any] = {}
        for name in self._METRIC_FIELDS:
            value = getattr(self, name)
            if value is not None:
                payload[name] = float(value)
        payload["metadata"] = _metadata_to_jsonable(self.metadata)
        return payload


def _metadata_to_jsonable(metadata: MetadataItems) -> dict[str, Any]:
    """Convert the tuple-of-pairs metadata into a JSON-compatible dict.

    Tuples are converted to lists; everything else passes through as-is.
    """
    out: dict[str, Any] = {}
    for key, value in metadata:
        if isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out
