r"""Shared :class:`PredictiveDistribution` construction factory.

Canonical constructors that every Gaussian-process / state-space estimator
uses to assemble a :class:`opifex.uncertainty.types.PredictiveDistribution`
and to refresh its provenance metadata. Centralising construction here keeps
the field set and metadata shape in a single place (Rule 1 — DRY) so that
adding or renaming a field touches exactly one site.

Two functions:

* :func:`gaussian_process_predictive` — the canonical constructor for a
  Gaussian-process predictive carrying ``mean`` / ``variance`` plus optional
  ``epistemic`` / ``total_uncertainty`` variance decompositions and metadata.
* :func:`replace_predictive_metadata` — immutable metadata refresh that keeps
  every array field untouched and only re-stamps the
  ``compose_method_metadata`` provenance tuple via :func:`dataclasses.replace`.

Both reproduce, byte-for-byte, the objects the hand-written estimator sites
build today (the markov ``predict_*`` paths and the per-likelihood response
wrappers).
"""

from __future__ import annotations

import dataclasses

import jax  # noqa: TC002 — pyproject dep, kept eager (annotations only under future import)

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import MetadataItems, PredictiveDistribution


def gaussian_process_predictive(
    mean: jax.Array,
    variance: jax.Array,
    *,
    epistemic: jax.Array | None = None,
    total_uncertainty: jax.Array | None = None,
    metadata: MetadataItems = (),
) -> PredictiveDistribution:
    """Assemble a Gaussian-process :class:`PredictiveDistribution`.

    Args:
        mean: Predictive mean, shape ``(batch, ...)``.
        variance: Marginal predictive *variance* (not std-dev) matching
            ``mean.shape``.
        epistemic: Optional epistemic-variance component. Defaults to ``None``.
        total_uncertainty: Optional total-variance component. Defaults to
            ``None``.
        metadata: Immutable, hashable provenance tuple. Defaults to ``()``.

    Returns:
        A :class:`PredictiveDistribution` carrying the supplied moments and
        metadata; all other container fields take their dataclass defaults.
    """
    return PredictiveDistribution(
        mean=mean,
        variance=variance,
        epistemic=epistemic,
        total_uncertainty=total_uncertainty,
        metadata=metadata,
    )


def replace_predictive_metadata(
    predictive: PredictiveDistribution,
    *,
    estimator: str,
    likelihood: str,
    link: str,
    paper: str | None = None,
    source: str = "opifex.uncertainty.markov",
) -> PredictiveDistribution:
    """Return ``predictive`` with refreshed provenance metadata.

    Keeps every array field of ``predictive`` untouched and re-stamps only the
    static metadata tuple, advertising ``estimator`` / ``likelihood`` /
    ``link`` (and an optional ``paper`` citation) for the
    :data:`DefaultStrategy.GAUSSIAN_PROCESS` method.

    Args:
        predictive: The predictive whose metadata is refreshed.
        estimator: Estimator identifier (e.g. ``"bernoulli_markov_vi_gp"``).
        likelihood: Likelihood name (e.g. ``"bernoulli"``).
        link: Response-link name (e.g. ``"logit"``).
        paper: Optional citation string; omitted from metadata when ``None``.
        source: ``source_package`` provenance tag. Defaults to the markov
            inference package.

    Returns:
        A metadata-refreshed copy of ``predictive`` via
        :func:`dataclasses.replace` (the typed equivalent of the flax-struct
        ``.replace`` immutable update).
    """
    extra: list[tuple[str, str]] = [("estimator", estimator)]
    if paper is not None:
        extra.append(("paper", paper))
    extra.append(("likelihood", likelihood))
    extra.append(("link", link))
    # ``dataclasses.replace`` is the typed equivalent of the flax-struct
    # ``predictive.replace`` immutable update (the struct method delegates to
    # it); it copies every array field unchanged and re-stamps only metadata.
    return dataclasses.replace(
        predictive,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=source,
            extra=tuple(extra),
        ),
    )


__all__ = [
    "gaussian_process_predictive",
    "replace_predictive_metadata",
]
