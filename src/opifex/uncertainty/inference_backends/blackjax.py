"""BlackJAX inference backend.

Adapts :func:`artifex.generative_models.core.sampling.blackjax_samplers.hmc_sampling`
/ ``nuts_sampling`` / ``mala_sampling`` to the Phase 1
:class:`opifex.uncertainty.inference_backends.InferenceBackendProtocol`.

The module deliberately imports the BlackJAX samplers through Artifex — the
only legitimate direct ``blackjax`` import in the Avitai ecosystem lives in
``../artifex/src/artifex/generative_models/core/sampling/blackjax_samplers.py``.
RNG ownership is enforced through Artifex's :func:`extract_rng_key` helper
(named-stream order ``sample`` → ``default``).

Supported sampler methods (delegated to Artifex):

* ``"hmc"`` — Hamiltonian Monte Carlo.
* ``"nuts"`` — No-U-Turn Sampler.
* ``"mala"`` — Metropolis-Adjusted Langevin Algorithm.

Unsupported sampler families (SGLD, SGHMC, SMC, ADVI, Pathfinder) raise
:class:`UnsupportedBackendError` until Artifex grows a wrapper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from artifex.generative_models.core.sampling.blackjax_samplers import (
    hmc_sampling,
    mala_sampling,
    nuts_sampling,
)

from opifex.uncertainty.inference_backends.base import (
    BackendDiagnostics,
    BackendResult,
    UnsupportedBackendError,
)
from opifex.uncertainty.types import PredictiveDistribution


if TYPE_CHECKING:
    from flax import nnx

_SUPPORTED_METHODS: tuple[str, ...] = ("hmc", "nuts", "mala")
_UNSUPPORTED_METHODS: tuple[str, ...] = ("sgld", "sghmc", "smc", "advi", "pathfinder")
_POSTERIOR_STREAMS: tuple[str, ...] = ("sample", "default")


class BlackJAXBackend:
    """BlackJAX backend adapter conforming to :class:`InferenceBackendProtocol`.

    Wraps Artifex's BlackJAX sampler functions and exposes ``fit`` /
    ``predict_distribution`` / ``posterior_predictive`` with Phase 1
    container types.

    Args:
        target_log_prob: Scalar-valued log-density callable evaluated at a
            single state.
        init_state: Initial sampler position.
        n_samples: Number of posterior samples to retain.
        n_burnin: Warmup samples to discard.
        method: One of ``"hmc"`` / ``"nuts"`` / ``"mala"``. Unsupported
            families raise :class:`UnsupportedBackendError` at construction.

    Raises:
        ValueError: When ``n_samples`` is non-positive or ``n_burnin`` is
            negative.
        UnsupportedBackendError: When ``method`` is in the unsupported list
            or is otherwise unknown.
    """

    def __init__(
        self,
        target_log_prob: Callable[[jax.Array], jax.Array],
        init_state: jax.Array,
        *,
        n_samples: int,
        n_burnin: int = 100,
        method: str = "nuts",
        step_size: float = 0.1,
    ) -> None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive; got {n_samples!r}.")
        if n_burnin < 0:
            raise ValueError(f"n_burnin must be non-negative; got {n_burnin!r}.")
        if method in _UNSUPPORTED_METHODS:
            raise UnsupportedBackendError(
                "blackjax",
                reason=(
                    f"{method!r} is not yet wrapped by Artifex's BlackJAX adapter. "
                    "If Artifex grows a wrapper, BlackJAXBackend will delegate to it; "
                    "until then this method is unsupported."
                ),
            )
        if method not in _SUPPORTED_METHODS:
            raise UnsupportedBackendError(
                "blackjax",
                reason=(
                    f"Unknown sampler method {method!r}. Supported methods: "
                    f"{_SUPPORTED_METHODS!r}; explicitly-unsupported: "
                    f"{_UNSUPPORTED_METHODS!r}."
                ),
            )
        self.target_log_prob = target_log_prob
        self.init_state = init_state
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.method = method
        self.step_size = step_size

    def fit(self, target_log_prob: Any, *, rngs: nnx.Rngs) -> BackendResult:
        """Run the configured sampler and return raw posterior samples.

        ``target_log_prob`` overrides the constructor-time function when
        supplied (Phase 1 protocol allows callers to thread the log density
        through ``fit``).
        """
        log_prob_fn = target_log_prob if target_log_prob is not None else self.target_log_prob
        # Route the RNG through Artifex's canonical helper. The Artifex sampler
        # function itself also calls ``extract_rng_key`` internally; calling it
        # once here gives us a concrete key with a context label and ensures
        # the spy in tests sees Opifex-side invocations.
        key = extract_rng_key(
            rngs,
            streams=_POSTERIOR_STREAMS,
            context="BlackJAXBackend sampling",
        )
        samples = self._dispatch(log_prob_fn, key)
        return BackendResult(sampler_state=samples, diagnostics=BackendDiagnostics())

    def _dispatch(
        self,
        log_prob_fn: Callable[[jax.Array], jax.Array],
        key: jax.Array,
    ) -> jax.Array:
        if self.method == "nuts":
            return nuts_sampling(
                log_prob_fn,
                self.init_state,
                key,
                n_samples=self.n_samples,
                n_burnin=self.n_burnin,
                step_size=self.step_size,
            )
        if self.method == "hmc":
            return hmc_sampling(
                log_prob_fn,
                self.init_state,
                key,
                n_samples=self.n_samples,
                n_burnin=self.n_burnin,
                step_size=self.step_size,
            )
        if self.method == "mala":
            return mala_sampling(
                log_prob_fn,
                self.init_state,
                key,
                n_samples=self.n_samples,
                n_burnin=self.n_burnin,
                step_size=self.step_size,
            )
        # ``__init__`` already gates the supported set; this branch is
        # defensive against future code reaching here with a method that
        # passes the gate but lacks a dispatch entry.
        raise UnsupportedBackendError("blackjax", reason=f"No dispatch for method {self.method!r}.")

    def predict_distribution(self, x: jax.Array, *, rngs: nnx.Rngs) -> PredictiveDistribution:
        """Posterior mean prediction at the input shape, with sample variance."""
        result = self.fit(self.target_log_prob, rngs=rngs)
        samples = result.sampler_state
        mean = jnp.broadcast_to(jnp.mean(samples, axis=0), x.shape)
        variance = jnp.broadcast_to(jnp.var(samples, axis=0), x.shape)
        return PredictiveDistribution(
            mean=mean,
            variance=variance,
            samples=jnp.broadcast_to(samples[:, None, :], (samples.shape[0], *x.shape))
            if x.shape and samples.ndim == 2
            else samples,
            metadata=(
                ("method", self.method),
                ("backend", "blackjax"),
                ("n_samples", self.n_samples),
                ("n_burnin", self.n_burnin),
            ),
        )

    def posterior_predictive(self, rngs: nnx.Rngs, x: jax.Array) -> PredictiveDistribution:
        """Posterior predictive samples evaluated at ``x``.

        For the lightweight case where the target log-density does not
        depend on a separate predictive model, the posterior predictive is
        the same shape contract as :meth:`predict_distribution` with a
        ``posterior_predictive`` method tag.
        """
        out = self.predict_distribution(x, rngs=rngs)
        return PredictiveDistribution(
            mean=out.mean,
            variance=out.variance,
            samples=out.samples,
            metadata=(
                ("method", "posterior_predictive"),
                ("backend", "blackjax"),
                ("sampler", self.method),
                ("n_samples", self.n_samples),
                ("n_burnin", self.n_burnin),
            ),
        )


__all__ = ["BlackJAXBackend"]
