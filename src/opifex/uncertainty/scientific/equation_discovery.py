"""Bayesian SINDy: regularized-horseshoe posterior over SINDy library terms.

Recovers governing equations from data with calibrated uncertainty: a
regularized-horseshoe prior over the SINDy candidate-library coefficients
yields per-term inclusion probabilities and per-coefficient credible
intervals. The model is a faithful port of pysindy's ``SBR`` optimizer
(``/mnt/ssd2/Works/pysindy/pysindy/optimizers/sbr.py``) onto a BlackJAX
NUTS sampler -- opifex's MCMC dependency is ``blackjax`` (numpyro is not a
dependency), so the numpyro probabilistic model is re-expressed as an
explicit JAX log-density sampled in unconstrained space.

Probabilistic model (mirrors ``SBR._numpyro_model`` /
``_sample_reg_horseshoe`` at ``sbr.py:151`` and ``sbr.py:192``). For library
matrix ``Theta`` of shape ``(n_samples, n_terms)`` and targets
``y = x_dot`` of shape ``(n_samples, n_targets)``::

    tau          ~ HalfCauchy(tau0)                                  # sbr.py:157
    c_sq         ~ InverseGamma(slab_nu / 2, slab_nu / 2 * slab_s**2)  # sbr.py:158
    lambda       ~ HalfCauchy(1.0)            shape (n_targets, n_terms)  # sbr.py:206
    lambda_tilde =  sqrt(c_sq) * lambda / sqrt(c_sq + tau**2 * lambda**2)  # sbr.py:207
    beta         ~ Normal(0, lambda_tilde * tau)  shape (n_targets, n_terms)  # sbr.py:208
    sigma        ~ Exponential(noise_lambda)                         # sbr.py:170
    y            ~ Normal(Theta @ beta.T, sigma)                     # sbr.py:171

BlackJAX samples the positive parameters (``tau``, ``c_sq``, ``lambda``,
``sigma``) in unconstrained space via a log transform (``exp`` to constrain)
and adds the change-of-variables log-Jacobian (``log theta`` for each
positive ``theta``); ``beta`` is already unconstrained. The joint
unconstrained log-density is the sum of the constrained-space prior
log-pdfs, the log-Jacobians, and the Gaussian likelihood -- and is fully
jittable. Warmup uses :func:`blackjax.window_adaptation` (Stan-style dual
averaging + diagonal mass-matrix estimation); sampling uses a
``jax.lax.scan`` over :func:`blackjax.nuts`.

**Honest scope.** Like pysindy's ``SBR`` (``sbr.py:32``), the error model is
imposed directly on the derivatives ``x_dot`` rather than on the integrated
states: there is no ODE integration, so ``sigma`` is the noise scale of the
derivatives and should not be interpreted as a state-noise level. The full
data-generating model of Hirsh et al. (2021) eq. 2.4 -- which integrates the
discovered equation -- is a known upstream TODO in pysindy
(see https://github.com/dynamicslab/pysindy/pull/440) and is deliberately
not fabricated here.

References:
    Hirsh, S. M., Barajas-Solano, D. A., & Kutz, J. N. (2021). Sparsifying
    Priors for Bayesian Uncertainty Quantification in Model Discovery
    (arXiv:2107.02107). http://arxiv.org/abs/2107.02107

    Piironen, J., & Vehtari, A. (2017). Sparsity Information and
    Regularization in the Horseshoe and Other Shrinkage Priors. Electronic
    Journal of Statistics, 11, 5018-5051. https://doi.org/10.1214/17-EJS1337SI
"""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

import blackjax
import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from blackjax.mcmc.hmc import (
    HMCState,  # noqa: TC002 — kept eager (pyproject dep) per opifex convention
)
from flax import struct
from jax.scipy.special import gammaln
from jax.scipy.stats import norm as _norm

from opifex.uncertainty.types import PredictionInterval


if TYPE_CHECKING:
    from collections.abc import Callable

    from flax import nnx

    from opifex.discovery.sindy.library import CandidateLibrary

# RNG streams consulted (in order) when a caller passes ``nnx.Rngs``; mirrors
# the platform-wide convention used by the inference backends.
_POSTERIOR_STREAMS: tuple[str, ...] = ("sindy", "sample", "default")

# Inclusion-probability threshold definition (see ``term_inclusion_probabilities``).
# The horseshoe is a continuous shrinkage prior: posterior draws of ``beta`` are
# never exactly zero, so a "term is active" event requires an explicit
# magnitude threshold. We use a threshold proportional to the largest
# posterior-mean coefficient magnitude per target (scale-relative), so the
# definition is invariant to the overall scale of the system.
_INCLUSION_RELATIVE_THRESHOLD: float = 0.1
# Floor protects the relative threshold when every coefficient is shrunk to ~0
# (an all-inactive target), avoiding a degenerate zero threshold.
_INCLUSION_ABSOLUTE_FLOOR: float = 1e-3


@struct.dataclass(slots=True, kw_only=True)
class PosteriorOverTerms:
    """Posterior over SINDy library-term coefficients from Bayesian SINDy.

    Holds the BlackJAX NUTS draws of the regularized-horseshoe coefficients
    plus the hyper-parameter draws that scale them. ``feature_names`` is the
    static, JIT-safe column labelling produced by the candidate library.

    Shape contract:
        * ``beta``  -- ``(num_samples, n_targets, n_terms)`` coefficient draws.
        * ``tau``   -- ``(num_samples,)`` global-shrinkage draws.
        * ``sigma`` -- ``(num_samples,)`` derivative-noise-scale draws.
    """

    beta: jax.Array
    tau: jax.Array
    sigma: jax.Array
    feature_names: tuple[str, ...] = struct.field(pytree_node=False)


class BayesianSINDy:
    """Bayesian SINDy with a regularized-horseshoe posterior over terms.

    Ports pysindy's ``SBR`` regularized-horseshoe model
    (``/mnt/ssd2/Works/pysindy/pysindy/optimizers/sbr.py``) onto BlackJAX
    NUTS. After :meth:`fit`, exposes posterior-mean coefficients, per-term
    inclusion probabilities, and per-coefficient credible intervals.

    Args:
        library: Candidate function library producing the SINDy design
            matrix ``Theta`` and the human-readable feature names. Must
            expose ``transform(x) -> Theta`` and
            ``get_feature_names() -> list[str]`` (see
            :class:`opifex.discovery.sindy.library.CandidateLibrary`).
        tau0: Global-scale hyper-prior for ``tau ~ HalfCauchy(tau0)``. Lower
            values increase sparsity. (``SBR.sparsity_coef_tau0``.)
        slab_nu: Degrees of freedom of the Student-t slab,
            ``c_sq ~ InverseGamma(slab_nu / 2, slab_nu / 2 * slab_s**2)``.
            (``SBR.slab_shape_nu``.)
        slab_s: Scale of the Student-t slab. (``SBR.slab_shape_s``.)
        noise_lambda: Rate of the exponential prior on the derivative-noise
            scale ``sigma ~ Exponential(noise_lambda)``.
            (``SBR.noise_hyper_lambda``.)
        num_warmup: BlackJAX window-adaptation warmup steps (discarded).
        num_samples: Retained NUTS posterior draws.

    Raises:
        ValueError: If any hyper-parameter is non-positive, or if
            ``num_warmup`` / ``num_samples`` are not valid sample counts.
    """

    def __init__(
        self,
        library: CandidateLibrary,
        *,
        tau0: float = 0.1,
        slab_nu: float = 4,
        slab_s: float = 2,
        noise_lambda: float = 1.0,
        num_warmup: int = 200,
        num_samples: int = 400,
    ) -> None:
        """Validate hyper-parameters and store sampler configuration."""
        if tau0 <= 0.0:
            raise ValueError(f"tau0 must be positive; got {tau0!r}.")
        if slab_nu <= 0.0:
            raise ValueError(f"slab_nu must be positive; got {slab_nu!r}.")
        if slab_s <= 0.0:
            raise ValueError(f"slab_s must be positive; got {slab_s!r}.")
        if noise_lambda <= 0.0:
            raise ValueError(f"noise_lambda must be positive; got {noise_lambda!r}.")
        if not isinstance(num_warmup, int) or num_warmup < 0:
            raise ValueError(f"num_warmup must be a non-negative integer; got {num_warmup!r}.")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer; got {num_samples!r}.")

        self.library = library
        self.tau0 = tau0
        self.slab_nu = slab_nu
        self.slab_s = slab_s
        self.noise_lambda = noise_lambda
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self._posterior: PosteriorOverTerms | None = None

    # ------------------------------------------------------------------ #
    # Log-density (jittable change-of-variables joint)
    # ------------------------------------------------------------------ #
    def build_log_density(
        self, x: jax.Array, x_dot: jax.Array
    ) -> tuple[Callable[[dict[str, jax.Array]], jax.Array], dict[str, jax.Array]]:
        """Build the joint unconstrained log-density and an initial position.

        Returns a ``(log_density, init_position)`` pair. ``log_density`` maps
        a parameter dict with unconstrained leaves ``log_tau`` (scalar),
        ``log_c_sq`` (scalar), ``log_lambda`` ``(n_targets, n_terms)``,
        ``beta`` ``(n_targets, n_terms)`` and ``log_sigma`` (scalar) to the
        scalar joint log-density (constrained-space prior log-pdfs +
        change-of-variables log-Jacobians + Gaussian likelihood). The
        callable is pure and jittable.
        """
        theta = self.library.transform(x)
        n_terms = theta.shape[1]
        n_targets = x_dot.shape[1]

        slab_a = self.slab_nu / 2.0
        slab_b = self.slab_nu / 2.0 * self.slab_s**2

        def log_density(params: dict[str, jax.Array]) -> jax.Array:
            # Constrain the positive parameters; the log transform's Jacobian
            # contributes ``log theta`` per parameter (d exp(u)/du = exp(u)).
            tau = jnp.exp(params["log_tau"])
            c_sq = jnp.exp(params["log_c_sq"])
            lam = jnp.exp(params["log_lambda"])
            sigma = jnp.exp(params["log_sigma"])
            beta = params["beta"]

            log_jacobian = (
                params["log_tau"]
                + params["log_c_sq"]
                + jnp.sum(params["log_lambda"])
                + params["log_sigma"]
            )

            # Priors in constrained space (closed-form log-pdfs verified vs scipy).
            lp_tau = _half_cauchy_logpdf(tau, self.tau0)
            lp_c_sq = _inverse_gamma_logpdf(c_sq, slab_a, slab_b)
            lp_lambda = jnp.sum(_half_cauchy_logpdf(lam, 1.0))

            lambda_tilde = jnp.sqrt(c_sq) * lam / jnp.sqrt(c_sq + tau**2 * lam**2)
            beta_scale = lambda_tilde * tau
            lp_beta = jnp.sum(_norm.logpdf(beta, 0.0, beta_scale))

            lp_sigma = _exponential_logpdf(sigma, self.noise_lambda)

            # Likelihood: y ~ Normal(Theta @ beta.T, sigma) (sbr.py:167,171).
            mu = theta @ beta.T
            log_likelihood = jnp.sum(_norm.logpdf(x_dot, mu, sigma))

            return lp_tau + lp_c_sq + lp_lambda + lp_beta + lp_sigma + log_likelihood + log_jacobian

        init_position: dict[str, jax.Array] = {
            "log_tau": jnp.array(0.0),
            "log_c_sq": jnp.array(0.0),
            "log_lambda": jnp.zeros((n_targets, n_terms)),
            "beta": jnp.zeros((n_targets, n_terms)),
            "log_sigma": jnp.array(0.0),
        }
        return log_density, init_position

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def fit(self, x: jax.Array, x_dot: jax.Array, *, rngs: nnx.Rngs) -> PosteriorOverTerms:
        """Run BlackJAX NUTS over the regularized-horseshoe model.

        Args:
            x: State data of shape ``(n_samples, n_features)``.
            x_dot: Target derivatives of shape ``(n_samples, n_targets)``.
            rngs: Explicit RNG owner; a key is derived from the first present
                of the ``sindy`` / ``sample`` / ``default`` streams (caller
                owns the seed, so identical seeds reproduce identical draws).

        Returns:
            The fitted :class:`PosteriorOverTerms`.
        """
        log_density, init_position = self.build_log_density(x, x_dot)
        feature_names = tuple(self.library.get_feature_names())

        key = extract_rng_key(rngs, streams=_POSTERIOR_STREAMS, context="BayesianSINDy.fit")
        warmup_key, sample_key = jax.random.split(key)

        warmup = blackjax.window_adaptation(blackjax.nuts, log_density)
        (initial_state, tuned_parameters), _ = warmup.run(
            warmup_key, init_position, num_steps=self.num_warmup
        )

        states = _nuts_inference_loop(
            sample_key, log_density, initial_state, tuned_parameters, self.num_samples
        )

        # ``states.position`` is the stacked unconstrained-parameter pytree;
        # it has the same dict structure as ``init_position`` (BlackJAX's
        # loosely-typed ``ArrayLikeTree`` stub is narrowed here). The local
        # ``log_lambda`` scales are reconstructable from the draws and are not
        # retained on the posterior container.
        positions = cast("dict[str, jax.Array]", states.position)
        posterior = PosteriorOverTerms(
            beta=positions["beta"],
            tau=jnp.exp(positions["log_tau"]),
            sigma=jnp.exp(positions["log_sigma"]),
            feature_names=feature_names,
        )
        self._posterior = posterior
        return posterior

    # ------------------------------------------------------------------ #
    # Posterior summaries
    # ------------------------------------------------------------------ #
    def coefficients(self) -> jax.Array:
        """Posterior-mean coefficients of shape ``(n_targets, n_terms)``."""
        return jnp.mean(self._require_posterior().beta, axis=0)

    def term_inclusion_probabilities(self) -> dict[str, float]:
        """Per-term posterior inclusion probability ``P(|beta_j| > threshold)``.

        The regularized horseshoe is a *continuous* shrinkage prior: posterior
        draws of each coefficient are never exactly zero, so "inclusion" of a
        term is not a discrete latent and must be defined through a magnitude
        threshold. The chosen definition (stated honestly per Rule 4) is the
        posterior fraction of draws whose magnitude exceeds a *scale-relative*
        threshold::

            threshold_j(target) = max(
                _INCLUSION_RELATIVE_THRESHOLD * max_k |mean(beta[:, target, k])|,
                _INCLUSION_ABSOLUTE_FLOOR,
            )

        i.e. a term counts as active in a draw when its magnitude exceeds 10%
        of the largest posterior-mean coefficient magnitude for that target
        (floored at ``1e-3`` so an all-inactive target does not yield a
        degenerate zero threshold). The relative form makes the probability
        invariant to the overall scale of the system; the per-target
        normalisation matches multi-output SINDy where different equations
        carry different coefficient magnitudes. Probabilities are averaged
        across targets to produce one value per library term.

        Returns:
            ``dict`` mapping each feature name to a probability in ``[0, 1]``.
        """
        posterior = self._require_posterior()
        beta = posterior.beta  # (num_samples, n_targets, n_terms)
        mean_magnitude = jnp.abs(jnp.mean(beta, axis=0))  # (n_targets, n_terms)
        per_target_scale = jnp.max(mean_magnitude, axis=1, keepdims=True)  # (n_targets, 1)
        threshold = jnp.maximum(
            _INCLUSION_RELATIVE_THRESHOLD * per_target_scale, _INCLUSION_ABSOLUTE_FLOOR
        )
        active = jnp.abs(beta) > threshold[None, :, :]  # (num_samples, n_targets, n_terms)
        per_term = jnp.mean(active.astype(jnp.float32), axis=(0, 1))  # (n_terms,)
        return {name: float(per_term[idx]) for idx, name in enumerate(posterior.feature_names)}

    def coefficient_posterior_intervals(self, level: float = 0.95) -> PredictionInterval:
        """Per-coefficient equal-tailed credible interval at coverage ``level``.

        Args:
            level: Nominal coverage in ``(0, 1)``; ``0.95`` gives the central
                95% interval from the posterior quantiles.

        Returns:
            A :class:`PredictionInterval` whose ``lower`` / ``upper`` arrays
            have shape ``(n_targets, n_terms)``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie strictly in (0, 1); got {level!r}.")
        beta = self._require_posterior().beta
        tail = (1.0 - level) / 2.0
        lower = jnp.quantile(beta, tail, axis=0)
        upper = jnp.quantile(beta, 1.0 - tail, axis=0)
        return PredictionInterval(
            lower=lower,
            upper=upper,
            coverage=level,
            method="bayesian_sindy_horseshoe",
        )

    def _require_posterior(self) -> PosteriorOverTerms:
        if self._posterior is None:
            raise RuntimeError("BayesianSINDy.fit must be called before posterior summaries.")
        return self._posterior


# --------------------------------------------------------------------------- #
# BlackJAX inference loop
# --------------------------------------------------------------------------- #
def _nuts_inference_loop(
    rng_key: jax.Array,
    log_density: Callable[[dict[str, jax.Array]], jax.Array],
    initial_state: HMCState,
    tuned_parameters: dict[str, jax.Array],
    num_samples: int,
) -> HMCState:
    """Run a single-chain NUTS ``lax.scan`` and return the stacked states.

    Mirrors the canonical BlackJAX inference loop (``blackjax`` quickstart):
    build the kernel from the window-adaptation-tuned step size / inverse mass
    matrix and scan it over per-step keys. NUTS shares HMC's ``HMCState``
    container (``blackjax.nuts(...).init`` returns an ``HMCState``).
    """
    kernel = blackjax.nuts(log_density, **tuned_parameters).step

    @jax.jit
    def one_step(state: HMCState, key: jax.Array) -> tuple[HMCState, HMCState]:
        state, _info = kernel(key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)
    return states


# --------------------------------------------------------------------------- #
# Closed-form log-pdfs (jax.scipy.stats lacks half-Cauchy / inverse-gamma)
# --------------------------------------------------------------------------- #
def _half_cauchy_logpdf(x: jax.Array, scale: float) -> jax.Array:
    """Log-pdf of ``HalfCauchy(scale)`` on ``x >= 0``.

    ``log 2 - log(pi * scale) - log(1 + (x / scale)**2)``.
    """
    return jnp.log(2.0) - jnp.log(jnp.pi * scale) - jnp.log1p((x / scale) ** 2)


def _inverse_gamma_logpdf(x: jax.Array, concentration: float, rate: float) -> jax.Array:
    """Log-pdf of ``InverseGamma(concentration, rate)`` (shape-rate form).

    ``a log b - lgamma(a) - (a + 1) log x - b / x`` for ``a = concentration``,
    ``b = rate`` -- matching ``numpyro.distributions.InverseGamma`` used by
    ``SBR`` (``sbr.py:158``).
    """
    return (
        concentration * jnp.log(rate)
        - gammaln(concentration)
        - (concentration + 1.0) * jnp.log(x)
        - rate / x
    )


def _exponential_logpdf(x: jax.Array, rate: float) -> jax.Array:
    """Log-pdf of ``Exponential(rate)`` on ``x >= 0``: ``log(rate) - rate * x``."""
    return jnp.log(rate) - rate * x


__all__ = ["BayesianSINDy", "PosteriorOverTerms"]
