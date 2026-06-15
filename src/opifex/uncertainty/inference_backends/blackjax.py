"""BlackJAX inference backend.

Adapts :func:`artifex.generative_models.core.sampling.blackjax_samplers.hmc_sampling`
/ ``nuts_sampling`` / ``mala_sampling`` to the
:class:`opifex.uncertainty.inference_backends.InferenceBackendProtocol`.

The module deliberately imports the BlackJAX samplers through Artifex — the
direct ``blackjax`` import lives in
``artifex/generative_models/core/sampling/blackjax_samplers.py``.
RNG ownership is enforced through Artifex's :func:`extract_rng_key` helper
(named-stream order ``sample`` → ``default``).

Supported sampler methods (delegated to Artifex):

* ``"hmc"`` — Hamiltonian Monte Carlo.
* ``"nuts"`` — No-U-Turn Sampler.
* ``"mala"`` — Metropolis-Adjusted Langevin Algorithm.

Unsupported sampler families (SGLD, SGHMC, SMC, ADVI, Pathfinder) raise
:class:`UnsupportedBackendError` until Artifex grows a wrapper.

Backend capability metadata is exposed via :data:`BLACKJAX_BACKEND_SPEC`.

**Diagnostic surface limitation.** Artifex's sampler wrappers return only
the posterior-samples array; the per-step BlackJAX ``info`` dict (which
holds acceptance rate, divergences, step size, tree depth) is discarded
internally. :meth:`BlackJAXBackend.fit` therefore returns an empty
:class:`BackendDiagnostics`. Surfacing ESS / R-hat / acceptance from the
sample array post-hoc would require an Opifex-local diagnostics module or
an upstream Artifex change; this is tracked for a later task.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from artifex.generative_models.core.sampling.base import (
    SamplingAlgorithm,
)
from artifex.generative_models.core.sampling.blackjax_samplers import (
    BlackJAXHMC,
    BlackJAXMALA,
    BlackJAXNUTS,
    BlackJAXSamplerState,
    hmc_sampling,
    mala_sampling,
    nuts_sampling,
)
from flax import nnx  # noqa: TC002

from opifex.uncertainty._predictive import predictive_from_parameter_samples
from opifex.uncertainty.inference_backends.base import (
    BackendDiagnostics,
    BackendResult,
    InferenceBackendSpec,
    UnsupportedBackendError,
)
from opifex.uncertainty.types import (
    PredictiveDistribution,  # noqa: TC001 — kept eager per opifex convention
)


if TYPE_CHECKING:
    from collections.abc import Callable

_SUPPORTED_METHODS: tuple[str, ...] = ("hmc", "nuts", "mala")
_UNSUPPORTED_METHODS: tuple[str, ...] = ("sgld", "sghmc", "smc")
_POSTERIOR_STREAMS: tuple[str, ...] = ("sample", "default")


class BlackJAXBackend:
    """BlackJAX backend adapter conforming to :class:`InferenceBackendProtocol`.

    Wraps Artifex's BlackJAX sampler functions and exposes ``fit`` /
    ``predict_distribution`` / ``posterior_predictive`` with the platform's
    canonical container types.

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
        """Validate sampler configuration and store fields for ``sample``."""
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
        supplied (the inference backend protocol allows callers to thread
        the log density through ``fit``).
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
        return BackendResult(
            sampler_state=samples,
            diagnostics=BackendDiagnostics(ess=_compute_ess(samples)),
        )

    def _dispatch(
        self,
        log_prob_fn: Callable[[jax.Array], jax.Array],
        key: jax.Array,
    ) -> jax.Array:
        """Run the configured BlackJAX sampler and return the posterior draws."""
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

    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
        predict_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ) -> PredictiveDistribution:
        """Posterior predictive at ``x`` from re-fitted posterior samples.

        Routes the posterior draws through the shared
        :func:`opifex.uncertainty._predictive.predictive_from_parameter_samples`
        adapter (Rule 1 — DRY): model-aware when ``predict_fn`` is supplied,
        otherwise the lightweight parameter-moment broadcast to ``x.shape``
        (the case where the target log-density does not depend on a separate
        predictive model).
        """
        result = self.fit(self.target_log_prob, rngs=rngs)
        return predictive_from_parameter_samples(
            result.sampler_state,
            x,
            predict_fn=predict_fn,
            metadata=(
                ("method", self.method),
                ("backend", "blackjax"),
                ("n_samples", self.n_samples),
                ("n_burnin", self.n_burnin),
            ),
        )

    def posterior_predictive(
        self,
        rngs: nnx.Rngs,
        x: jax.Array,
        predict_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    ) -> PredictiveDistribution:
        """Posterior predictive samples evaluated at ``x``.

        Re-fits and routes through the shared adapter with a
        ``posterior_predictive`` method tag. For the lightweight case where the
        target log-density does not depend on a separate predictive model, this
        is the same shape contract as :meth:`predict_distribution`.
        """
        result = self.fit(self.target_log_prob, rngs=rngs)
        return predictive_from_parameter_samples(
            result.sampler_state,
            x,
            predict_fn=predict_fn,
            metadata=(
                ("method", "posterior_predictive"),
                ("backend", "blackjax"),
                ("sampler", self.method),
                ("n_samples", self.n_samples),
                ("n_burnin", self.n_burnin),
            ),
        )


def _compute_ess(samples: jax.Array) -> jax.Array:
    """Effective sample size per parameter from a single MCMC chain.

    Implements the standard autocorrelation-based estimator
    ``ESS = N / (1 + 2 * sum_{k=1..K} rho_k)`` where ``rho_k`` is the
    sample autocorrelation at lag ``k`` and ``K`` is the smallest lag where
    autocorrelation crosses zero (Geyer's initial positive sequence cutoff,
    see Geyer 1992 "Practical Markov Chain Monte Carlo").

    Args:
        samples: Posterior samples of shape ``(n_samples, ...)``. Per-element
            ESS is computed independently along the leading sample axis.

    Returns:
        Array of ESS values with shape ``samples.shape[1:]``.

    """
    n = samples.shape[0]
    if n < 4:
        return jnp.full(samples.shape[1:], float(n))

    centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    var = jnp.mean(centered * centered, axis=0)
    var = jnp.where(var == 0, 1.0, var)

    # Compute autocorrelations up to max_lag via direct sum. Bounded at
    # n // 2 to limit cost on long chains.
    max_lag = min(n // 2, 64)
    rho_sum = jnp.zeros(samples.shape[1:], dtype=samples.dtype)
    # Track whether each parameter has hit its first-negative-rho cutoff.
    cutoff_hit = jnp.zeros(samples.shape[1:], dtype=jnp.bool_)
    for lag in range(1, max_lag + 1):
        rho_k = jnp.mean(centered[lag:] * centered[:-lag], axis=0) / var
        # Geyer's initial positive sequence: stop adding once rho_k becomes
        # non-positive (per parameter independently).
        active = jnp.logical_and(~cutoff_hit, rho_k > 0)
        rho_sum = rho_sum + jnp.where(active, rho_k, 0.0)
        cutoff_hit = jnp.logical_or(cutoff_hit, ~active)

    iat = 1.0 + 2.0 * rho_sum
    ess = jnp.asarray(n) / iat
    return jnp.clip(ess, 1.0, float(n))


BLACKJAX_BACKEND_SPEC = InferenceBackendSpec(
    name="blackjax",
    family="MCMC",
    sampler_names=(
        *_SUPPORTED_METHODS,
        *(f"unsupported:{m}" for m in _UNSUPPORTED_METHODS),
    ),
    source_package="artifex",
)


__all__ = [
    "BLACKJAX_BACKEND_SPEC",
    "BlackJAXBackend",
    "BlackJAXHMC",
    "BlackJAXMALA",
    "BlackJAXNUTS",
    "BlackJAXSamplerState",
    "SamplingAlgorithm",
]
