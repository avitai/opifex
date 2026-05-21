"""Shared base for inference-backend stubs awaiting concrete algorithms.

Three peer samplers (Pathfinder / SVGD / ADVI) share an identical
protocol-conforming surface — only the ``name``, ``method_names``, and
``notes`` differ. Extracting the shared shape into
:class:`_DeferredInferenceBackend` eliminates the per-class duplication
of ``fit`` / ``predict_distribution`` / ``posterior_predictive``
``NotImplementedError`` bodies.

Mirrors the :class:`_DeferredAdapterSpec` pattern from
:mod:`opifex.uncertainty.adapters._specs`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax  # noqa: TC002 — kept eager for consistency with the rest of opifex.uncertainty
from flax import nnx  # noqa: TC002

from opifex.uncertainty.inference_backends.base import BackendResult
from opifex.uncertainty.types import PredictiveDistribution


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _DeferredInferenceBackend:
    """Pattern-A base for deferred inference backends.

    Attributes:
        name: Backend identifier used by the router.
        source_package: Backend implementation lives in opifex.
        method_names: Sampler-method identifiers handled by this backend.
        notes: Free-text rationale shown in capability reports.
    """

    name: str = "deferred"
    source_package: str = "opifex"
    method_names: tuple[str, ...] = ()
    notes: str = ""

    def _raise(self, hook: str) -> None:
        """Raise an actionable ``NotImplementedError`` naming the backend + hook."""
        raise NotImplementedError(
            f"{self.name!r} backend hook {hook!r} is not yet wired. "
            f"Notes: {self.notes!r}. Implementation lands in a follow-up "
            f"slice."
        )

    def fit(
        self, target_log_prob: Callable[[jax.Array], jax.Array], *, rngs: nnx.Rngs
    ) -> BackendResult:
        """Run inference; raises until the concrete algorithm lands."""
        del target_log_prob, rngs
        self._raise("fit")
        raise AssertionError("unreachable")  # pragma: no cover

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        """Return a predictive distribution for inputs ``x``."""
        del x, rngs
        self._raise("predict_distribution")
        raise AssertionError("unreachable")  # pragma: no cover

    def posterior_predictive(self, rngs: nnx.Rngs, x: jax.Array) -> PredictiveDistribution:
        """Return a posterior-predictive distribution sample for ``x``."""
        del rngs, x
        self._raise("posterior_predictive")
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["_DeferredInferenceBackend"]
