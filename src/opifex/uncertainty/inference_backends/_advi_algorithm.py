"""Mean-field Automatic Differentiation Variational Inference (ADVI).

Line-by-line port of the mean-field VI primitives from
``../blackjax/blackjax/vi/meanfield_vi.py`` and the shared Gaussian-VI
ELBO step from ``../blackjax/blackjax/vi/_gaussian_vi.py``. The only
adaptations are typing-modernisation (PEP 604 unions, ``frozen``
dataclasses with ``slots`` + ``kw_only``) and the addition of an
``approximate`` driver that runs ``num_iterations`` Adam steps so the
``ADVIBackend.fit`` wrapper can call a single function — mirroring the
shape of :mod:`_pathfinder_algorithm` and :mod:`_svgd_algorithm`.

Canonical reference:
* Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., Blei, D. M. 2017
  — *Automatic Differentiation Variational Inference*, JMLR 18(14).
* Roeder, G., Wu, Y., Duvenaud, D. 2017 — *Sticking the landing:
  simple, lower-variance gradient estimators for variational inference*,
  NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, TYPE_CHECKING

import jax
import jax.flatten_util  # needed for ``jax.flatten_util.ravel_pytree``
import jax.numpy as jnp
import jax.scipy as jsp
import optax


if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Objective hierarchy (KL or Rényi alpha)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KL:
    """Standard reverse-KL variational objective."""


@dataclass(frozen=True, slots=True)
class RenyiAlpha:
    """Rényi-alpha variational objective.

    A smooth interpolation between the evidence lower bound and the
    log-marginal likelihood controlled by ``alpha``. ``alpha=1.0``
    recovers reverse-KL.
    """

    alpha: float


Objective = KL | RenyiAlpha


def _objective_value_from_log_ratio(
    log_ratio: jax.Array,
    objective: Objective,
) -> jax.Array:
    """Return the scalar variational loss for a Monte Carlo log-ratio sample.

    ``log_ratio[i] = log q(z_i) - log p(z_i, x)`` for sample ``z_i``.
    """
    if isinstance(objective, KL):
        return jnp.mean(log_ratio)
    if isinstance(objective, RenyiAlpha):
        alpha = objective.alpha
        if alpha == 1.0:
            return jnp.mean(log_ratio)
        scaled = (alpha - 1.0) * log_ratio
        return (jsp.special.logsumexp(scaled) - jnp.log(log_ratio.shape[0])) / (alpha - 1.0)
    raise TypeError(f"Unsupported objective type: {type(objective)!r}")


# ---------------------------------------------------------------------------
# Mean-field VI state and step
# ---------------------------------------------------------------------------


class MFVIState(NamedTuple):
    """Mean-field VI optimisation state."""

    mu: Any  # mean PyTree (variational mean)
    rho: Any  # log-scale PyTree (variational log-std)
    opt_state: optax.OptState


class MFVIInfo(NamedTuple):
    """Per-step diagnostics returned by :func:`step`."""

    elbo: jax.Array


def init(
    position: Any,
    optimizer: optax.GradientTransformation,
) -> MFVIState:
    """Initialise the mean-field state.

    ``position`` provides the PyTree skeleton: ``mu`` is initialised to
    zeros and ``rho`` (log scale) to ``-2`` (i.e. ``sigma ≈ 0.135``).
    """
    mu = jax.tree.map(jnp.zeros_like, position)
    rho = jax.tree.map(lambda x: -2.0 * jnp.ones_like(x), position)
    opt_state = optimizer.init((mu, rho))
    return MFVIState(mu, rho, opt_state)


def _sample(
    rng_key: jax.Array,
    mu: Any,
    rho: Any,
    num_samples: int,
) -> Any:
    """Draw ``num_samples`` reparametrised mean-field samples."""
    sigma = jax.tree.map(jnp.exp, rho)
    mu_flat, unravel_fn = jax.flatten_util.ravel_pytree(mu)
    sigma_flat, _ = jax.flatten_util.ravel_pytree(sigma)
    flat_samples = jax.random.normal(rng_key, (num_samples, *mu_flat.shape)) * sigma_flat + mu_flat
    return jax.vmap(unravel_fn)(flat_samples)


def generate_meanfield_logdensity(mu: Any, rho: Any) -> Callable[[Any], jax.Array]:
    """Return ``log q(z)`` for the current mean-field parameters."""
    sigma = jax.tree.map(jnp.exp, rho)

    def meanfield_logdensity(position: Any) -> jax.Array:
        logq_pytree = jax.tree.map(jsp.stats.norm.logpdf, position, mu, sigma)
        logq = jax.tree.map(jnp.sum, logq_pytree)
        return jax.tree.reduce(jnp.add, logq)

    return meanfield_logdensity


def _elbo_step(
    rng_key: jax.Array,
    parameters: tuple[Any, Any],
    opt_state: optax.OptState,
    logdensity_fn: Callable[[Any], jax.Array],
    optimizer: optax.GradientTransformation,
    sample_fn: Callable[[jax.Array, tuple[Any, Any], int], Any],
    logq_fn: Callable[[tuple[Any, Any]], Callable[[Any], jax.Array]],
    num_samples: int,
    objective: Objective,
    stl_estimator: bool,
) -> tuple[tuple[Any, Any], optax.OptState, jax.Array]:
    """One ELBO optimisation step shared by mean-field and full-rank ADVI."""
    if stl_estimator and isinstance(objective, RenyiAlpha) and objective.alpha != 1.0:
        raise ValueError(
            "stl_estimator is only supported with KL() or RenyiAlpha(alpha=1.0); "
            "use stl_estimator=False for RenyiAlpha(alpha != 1.0)."
        )

    def objective_fn(params: tuple[Any, Any]) -> jax.Array:
        z = sample_fn(rng_key, params, num_samples)
        logq_params = jax.lax.stop_gradient(params) if stl_estimator else params
        logq = jax.vmap(logq_fn(logq_params))(z)
        logp = jax.vmap(logdensity_fn)(z)
        log_ratio = logq - logp
        return _objective_value_from_log_ratio(log_ratio, objective)

    objective_value, objective_grad = jax.value_and_grad(objective_fn)(parameters)
    updates, new_opt_state = optimizer.update(objective_grad, opt_state, parameters)
    new_parameters = jax.tree.map(lambda p, u: p + u, parameters, updates)
    return new_parameters, new_opt_state, objective_value


def step(
    rng_key: jax.Array,
    state: MFVIState,
    logdensity_fn: Callable[[Any], jax.Array],
    optimizer: optax.GradientTransformation,
    num_samples: int = 5,
    objective: Objective | None = None,
    stl_estimator: bool = True,
) -> tuple[MFVIState, MFVIInfo]:
    """Apply one stochastic-gradient ELBO step to the mean-field state."""
    objective = KL() if objective is None else objective
    parameters = (state.mu, state.rho)

    def sample_fn(key: jax.Array, params: tuple[Any, Any], n: int) -> Any:
        return _sample(key, params[0], params[1], n)

    def logq_fn(params: tuple[Any, Any]) -> Callable[[Any], jax.Array]:
        return generate_meanfield_logdensity(params[0], params[1])

    new_parameters, new_opt_state, elbo = _elbo_step(
        rng_key,
        parameters,
        state.opt_state,
        logdensity_fn,
        optimizer,
        sample_fn,
        logq_fn,
        num_samples,
        objective=objective,
        stl_estimator=stl_estimator,
    )
    new_state = MFVIState(new_parameters[0], new_parameters[1], new_opt_state)
    return new_state, MFVIInfo(elbo)


def sample(rng_key: jax.Array, state: MFVIState, num_samples: int = 1) -> Any:
    """Draw ``num_samples`` from the current mean-field approximation."""
    return _sample(rng_key, state.mu, state.rho, num_samples)


# ---------------------------------------------------------------------------
# Backend-facing driver (``approximate`` + ``draw``) — used by ADVIBackend.fit
# ---------------------------------------------------------------------------


def approximate(
    rng_key: jax.Array,
    log_density_fn: Callable[[Any], jax.Array],
    initial_position: Any,
    *,
    num_iterations: int,
    num_mc_samples: int,
    learning_rate: float,
    objective: Objective | None = None,
    stl_estimator: bool = True,
) -> MFVIState:
    """Run ``num_iterations`` Adam steps minimising the ELBO.

    Wraps :func:`init` and :func:`step` into a single ``fit``-style entry
    point that mirrors the shape of :func:`_pathfinder_algorithm.pathfinder_approximate`.
    Returns the final ``MFVIState`` carrying the fitted ``(mu, rho)``.
    """
    objective_value: Objective = KL() if objective is None else objective
    optimizer = optax.adam(learning_rate)
    initial_state = init(initial_position, optimizer)

    def scan_step(state: MFVIState, key: jax.Array) -> tuple[MFVIState, jax.Array]:
        new_state, info = step(
            key,
            state,
            log_density_fn,
            optimizer,
            num_samples=num_mc_samples,
            objective=objective_value,
            stl_estimator=stl_estimator,
        )
        return new_state, info.elbo

    keys = jax.random.split(rng_key, num_iterations)
    final_state, _elbo_trace = jax.lax.scan(scan_step, initial_state, keys)
    return final_state


def draw(rng_key: jax.Array, state: MFVIState, *, num_samples: int) -> jax.Array:
    """Draw ``num_samples`` from the fitted Gaussian, return as ``(N, d)``."""
    samples_pytree = sample(rng_key, state, num_samples)
    return jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(samples_pytree)


__all__ = [
    "KL",
    "MFVIInfo",
    "MFVIState",
    "Objective",
    "RenyiAlpha",
    "approximate",
    "draw",
    "generate_meanfield_logdensity",
    "init",
    "sample",
    "step",
]
