"""Shared predictive-hook logic for the peer VI/particle backends.

The three peer backends (:class:`~opifex.uncertainty.inference_backends.advi.ADVIBackend`,
:class:`~opifex.uncertainty.inference_backends.svgd.SVGDBackend`,
:class:`~opifex.uncertainty.inference_backends.pathfinder.PathfinderBackend`)
all expose parameter-space draws through ``fit`` and turn those draws into a
:class:`~opifex.uncertainty.types.PredictiveDistribution` with identical logic:
re-fit from the stored ``target_log_prob``, take ``result.sampler_state``, and
route through :func:`opifex.uncertainty._predictive.predictive_from_parameter_samples`.

Extracting that shared body here removes the per-class duplication (Rule 1 —
DRY) and gives every peer backend the same actionable error when no
``target_log_prob`` was stored at construction.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

from opifex.uncertainty._predictive import predictive_from_parameter_samples


if TYPE_CHECKING:
    from collections.abc import Callable

    import jax
    from flax import nnx

    from opifex.uncertainty.inference_backends.base import BackendResult
    from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


@runtime_checkable
class _RefittableBackend(Protocol):
    """Minimal surface a peer backend must expose to share the predictive path.

    The backend must carry a ``name`` and an optional stored
    ``target_log_prob`` and provide a ``fit`` re-running inference. The two
    attribute members are declared read-only (via ``@property``) so the
    frozen-dataclass backends — whose fields are immutable — structurally
    satisfy the protocol.
    """

    @property
    def name(self) -> str:
        """Backend identifier used in actionable error messages."""
        ...

    @property
    def target_log_prob(self) -> Callable[[jax.Array], jax.Array] | None:
        """Stored log-density callable, or ``None`` when unset at construction."""
        ...

    def fit(
        self, target_log_prob: Callable[[jax.Array], jax.Array], *, rngs: nnx.Rngs
    ) -> BackendResult:
        """Run inference and return the backend result with ``sampler_state``."""
        ...


def resolve_target_log_prob(
    backend: _RefittableBackend,
) -> Callable[[jax.Array], jax.Array]:
    """Return the backend's stored ``target_log_prob`` or raise an actionable error.

    The predictive hooks re-fit the backend, which requires a log density. A
    backend constructed without one cannot produce a predictive distribution.
    """
    if backend.target_log_prob is None:
        raise ValueError(
            f"{backend.name!r} backend has no stored 'target_log_prob'; "
            "predict_distribution / posterior_predictive re-fit the backend and "
            "therefore need a log density. Construct the backend with "
            "target_log_prob=<callable> (or thread it through fit) before calling "
            "the predictive hooks."
        )
    return backend.target_log_prob


def peer_predictive_from_refit(
    backend: _RefittableBackend,
    x: jax.Array,
    *,
    rngs: nnx.Rngs,
    predict_fn: Callable[[jax.Array, jax.Array], jax.Array] | None,
    metadata: MetadataItems,
) -> PredictiveDistribution:
    """Re-fit ``backend`` and map its parameter draws to a predictive at ``x``.

    Shared body for every peer backend's ``predict_distribution`` /
    ``posterior_predictive``. Re-runs ``fit`` from the stored
    ``target_log_prob`` and routes the resulting parameter-space draws through
    :func:`predictive_from_parameter_samples` (model-aware when ``predict_fn`` is
    supplied, lightweight otherwise).

    Args:
        backend: The peer backend to re-fit (provides ``fit`` + stored target).
        x: Inputs at which to evaluate the predictive.
        rngs: Caller-owned RNG streams threaded into ``fit``.
        predict_fn: Optional ``predict_fn(params_vector, x) -> predictions``
            forward model selecting the model-aware path.
        metadata: Backend-identifying provenance tuple stamped on the result.

    Returns:
        The :class:`PredictiveDistribution` produced by the adapter.
    """
    target_log_prob = resolve_target_log_prob(backend)
    result = backend.fit(target_log_prob, rngs=rngs)
    return predictive_from_parameter_samples(
        result.sampler_state,
        x,
        predict_fn=predict_fn,
        metadata=metadata,
    )


__all__ = ["peer_predictive_from_refit", "resolve_target_log_prob"]
