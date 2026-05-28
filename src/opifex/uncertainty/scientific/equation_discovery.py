"""Step-10 stubs for equation-discovery UQ surfaces.

Audit Migration Step 10 requires named interfaces so the Phase 7
capability registries can advertise equation-discovery UQ truthfully
without overclaiming behaviour. Every operational method raises
``NotImplementedError`` per Rule 6 (fail-fast). Constructors only
validate argument types / ranges.

Future implementers: see the audit's Migration Step 10 description
for the contract each stub captures (Bayesian SINDy posterior over
library terms, per-term inclusion probabilities, and per-coefficient
posterior intervals).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from flax import nnx  # noqa: TC002 — kept eager for opifex convention

from opifex.uncertainty.types import PredictionInterval  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import jax


_CANONICAL_MESSAGE = "Step 10 stub: see audit Migration Step 10"


class PosteriorOverTerms:
    """Placeholder for the posterior-over-library-terms return type.

    Phase 8 only ships the type name so the canonical
    :meth:`BayesianSINDyStub.fit` signature can reference it. The
    concrete posterior representation lands when Step 10 is real.
    """


class BayesianSINDyStub:
    """Stub: Bayesian SINDy with term-inclusion probabilities.

    Audit contract: ``fit(x, x_dot, *, rngs)`` returns a posterior over
    candidate library terms; ``term_inclusion_probabilities()`` reads
    per-term marginal probabilities off that posterior.

    Constructor validates ``sparsity_threshold > 0`` and a non-empty
    ``library`` sequence; everything else raises ``NotImplementedError``
    until Step 10 lands.
    """

    def __init__(
        self,
        *,
        library: Sequence[Callable[[jax.Array], jax.Array]],
        sparsity_threshold: float,
        metadata: tuple[tuple[str, Any], ...] = (),
    ) -> None:
        if len(library) == 0:
            raise ValueError("library must contain at least one candidate term.")
        if sparsity_threshold <= 0.0:
            raise ValueError(
                f"sparsity_threshold must be > 0; got {sparsity_threshold}."
            )
        self.library = tuple(library)
        self.sparsity_threshold = sparsity_threshold
        self.metadata = metadata

    def fit(
        self, x: jax.Array, x_dot: jax.Array, *, rngs: nnx.Rngs
    ) -> PosteriorOverTerms:
        """Fit Bayesian SINDy over the library terms; not yet implemented."""
        del x, x_dot, rngs
        raise NotImplementedError(_CANONICAL_MESSAGE)

    def term_inclusion_probabilities(self) -> dict[str, float]:
        """Per-term marginal inclusion probabilities; not yet implemented."""
        raise NotImplementedError(_CANONICAL_MESSAGE)


class TermInclusionProbabilityStub:
    """Stub: extract per-term inclusion probabilities from posterior samples.

    Audit contract: callable that takes ``(num_samples, num_terms)``
    posterior samples and returns a ``dict[term_name, probability]``.
    """

    def __init__(self, *, term_names: tuple[str, ...]) -> None:
        if len(term_names) == 0:
            raise ValueError("term_names must contain at least one entry.")
        self.term_names = term_names

    def __call__(self, posterior_samples: jax.Array) -> dict[str, float]:
        """Compute inclusion probabilities; not yet implemented."""
        del posterior_samples
        raise NotImplementedError(_CANONICAL_MESSAGE)


class CoefficientPosteriorIntervalStub:
    """Stub: per-coefficient posterior credible intervals.

    Audit contract: callable that takes posterior samples and returns
    a :class:`PredictionInterval` at level ``1 - alpha``.
    """

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"alpha must lie strictly in (0, 1); got {alpha}."
            )
        self.alpha = alpha

    def __call__(self, posterior_samples: jax.Array) -> PredictionInterval:
        """Compute the credible interval; not yet implemented."""
        del posterior_samples
        raise NotImplementedError(_CANONICAL_MESSAGE)


__all__ = [
    "BayesianSINDyStub",
    "CoefficientPosteriorIntervalStub",
    "PosteriorOverTerms",
    "TermInclusionProbabilityStub",
]
