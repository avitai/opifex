r"""MUMBO multi-fidelity Bayesian-optimisation acquisition (Moss+ 2020).

Implements the Multi-task Max-value Bayesian Optimisation acquisition
of Moss, Leslie, Rayson (2020) on top of opifex's linear
multi-fidelity GP. Each candidate is a ``(x, fidelity_level)`` pair;
the score is the **mutual information** between the maximum value
``g*`` of the target (highest-fidelity) level and a noisy observation
of the candidate (eq. 4 of the paper).

Algorithm
---------

1. **Samples of the maximum.** Approximate ``Pr(g* < y)`` on a random
   grid of target-level inputs by the product of the marginal CDFs, fit
   a Gumbel to its quartiles and draw ``num_gumbel_samples`` samples
   (Wang & Jegelka 2017, §3.1).
2. **For each candidate** ``(x_i, level_i)``, the bivariate predictive
   of Appendix A:

   a. the target level's latent mean and standard deviation
      ``(mu_g, sigma_g)`` at ``x_i``;
   b. the candidate's latent variance ``sigma_f^2`` and its posterior
      covariance ``Sigma`` with the target at ``x_i``;
   c. ``gamma = (g* - mu_g) / sigma_g`` and
      ``rho = Sigma / (sigma_g sqrt(sigma_f^2 + sigma^2))``, where
      ``sigma^2`` is the observation-noise variance.

3. **Eq. 5** for each sample of ``g*``, whose expectation over the
   extended skew Gaussian ``Z | g < g*`` is evaluated by Simpson's rule
   over eight standard deviations about its mean, averaged over the
   samples.

Cost weighting (dividing by per-level query cost) is left to the
caller — different BO loops apply it differently.

References:
----------
* Moss, Leslie, Rayson 2020 — *MUMBO: MUlti-task Max-value Bayesian
  Optimisation*, ECML-PKDD, arXiv:2006.12093.
* Wang, Jegelka 2017 — *Max-value Entropy Search for Efficient
  Bayesian Optimization*, ICML (single-fidelity MES baseline).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.multi_fidelity.linear import (
    _augment_with_level,
    linear_multi_fidelity_kernel,
    LinearMultiFidelityGPState,
    predict_linear_multi_fidelity_gp,
)


_PSEUDO_NOISE_FLOOR: float = 1e-10


def _gumbel_fit_and_sample(
    *,
    grid_means: jax.Array,
    grid_stds: jax.Array,
    num_samples: int,
    rng_key: jax.Array,
) -> jax.Array:
    r"""Fit a Gumbel approximation to the maximum of the target level and sample it.

    Wang & Jegelka (2017), §3.1: ``Pr(y* < y)`` is approximated by
    ``prod_i Phi((y - mu_i) / sigma_i)`` and matched by the Gumbel
    ``exp(-exp(-(y - a) / b))`` at its quartiles, ``a - b log(-log r) = y_r`` for
    ``r = 0.25`` and ``r = 0.75``:

        b   = (y_0.75 - y_0.25) / log(log(4) / log(4/3)),
        a   = y_0.25 + b log(log(4)),
        y*  = a - b log(-log(r)),  r ~ Uniform(0, 1).

    The quartiles are interpolated on 200 levels spanning five standard
    deviations beyond the grid.
    """
    y_low = jnp.min(grid_means - 5.0 * grid_stds)
    y_high = jnp.max(grid_means + 5.0 * grid_stds)
    levels = jnp.linspace(y_low, y_high, 200)

    def cdf_at(level: jax.Array) -> jax.Array:
        """Return the probability that every grid maximum lies at or below ``level``."""
        normalised = (level - grid_means) / jnp.maximum(grid_stds, _PSEUDO_NOISE_FLOOR)
        # Probability that ALL grid maxima are <= level.
        return jnp.prod(jax.scipy.stats.norm.cdf(normalised))

    cdfs = jax.vmap(cdf_at)(levels)
    quantile_25 = jnp.interp(0.25, cdfs, levels)
    quantile_75 = jnp.interp(0.75, cdfs, levels)
    scale = (quantile_75 - quantile_25) / jnp.log(jnp.log(4.0) / jnp.log(4.0 / 3.0))
    location = quantile_25 + scale * jnp.log(jnp.log(4.0))
    uniform_samples = jax.random.uniform(rng_key, (num_samples,), minval=1e-6, maxval=1.0 - 1e-6)
    return location - scale * jnp.log(-jnp.log(uniform_samples))


def _simpson_weights(num_points: int) -> jax.Array:
    """Composite Simpson weights ``1, 4, 2, ..., 2, 4, 1`` for an odd number of points."""
    weights = jnp.ones(num_points)
    weights = weights.at[1:-1:2].set(4.0)
    return weights.at[2:-1:2].set(2.0)


def _mumbo_information_gain(
    *,
    correlation: jax.Array,
    gamma: jax.Array,
    num_quadrature_points: int,
) -> jax.Array:
    r"""Eq. 5 of Moss, Leslie & Rayson (2020) for one sample ``g*`` of the maximum.

    .. math::

        \rho^2 \frac{\gamma \phi(\gamma)}{2 \Phi(\gamma)} - \log \Phi(\gamma)
        + \mathbb{E}_{\theta \sim Z}\Big[
            \log \Phi\Big(\frac{\gamma - \rho \theta}{\sqrt{1 - \rho^2}}\Big)\Big],

    where ``Z = (y - mu_f) / sqrt(sigma_f^2 + sigma^2) | g < g*`` has the extended skew
    Gaussian density ``phi(theta) Phi((gamma - rho theta) / sqrt(1 - rho^2)) / Phi(gamma)`` of
    Appendix A.1. The expectation is evaluated by composite Simpson's rule over eight standard
    deviations about the mean of ``Z``, as in Appendix A.1. With ``r = phi(gamma) / Phi(gamma)``,
    conditioning on ``g < g*`` gives ``E[Z] = -rho r``; eq. 7 of the paper prints ``+rho r``, which
    centres the range on the wrong side of zero. ``Var[Z] = 1 - rho^2 r (gamma + r)``. At
    ``|rho| = 1`` the expectation vanishes and the score is MES (Wang & Jegelka 2017, eq. 6), as
    §3.2 of the paper notes.

    Two errors bound the float32 result. Rounding is a few float32 units in the last place of
    ``1 + |log Phi(gamma)| + rho^2 |gamma| r / 2``, largest for ``gamma << 0`` where the expectation
    nearly cancels ``-log Phi(gamma)``. When ``sqrt(1 - rho^2)`` is narrower than one Simpson panel
    ``2h``, the skew factor is a step at ``theta = gamma / rho`` that the rule does not resolve, and
    the quadrature error is first order in the node spacing ``h``: at most
    ``phi(gamma / rho) / Phi(gamma) (4h / (3e) + C sqrt(1 - rho^2) / |rho|)`` with
    ``C = int |Phi(u) log Phi(u)| du ~ 0.903``.

    Args:
        correlation: Correlation ``rho`` between the noisy observation and the target level.
        gamma: ``(g* - mu_g) / sigma_g``.
        num_quadrature_points: Odd number of Simpson points.

    Returns:
        Information gain in nats.

    Raises:
        ValueError: If ``num_quadrature_points`` is even or smaller than three.
    """
    if num_quadrature_points < 3 or num_quadrature_points % 2 == 0:
        raise ValueError(
            f"num_quadrature_points must be odd and at least 3; got {num_quadrature_points}."
        )
    rho = jnp.clip(correlation, -1.0, 1.0)
    log_cdf_gamma = jax.scipy.stats.norm.logcdf(gamma)
    ratio = jnp.exp(jax.scipy.stats.norm.logpdf(gamma) - log_cdf_gamma)
    mean = -rho * ratio
    std = jnp.sqrt(jnp.maximum(1.0 - rho**2 * ratio * (gamma + ratio), _PSEUDO_NOISE_FLOOR))
    lower = mean - 8.0 * std
    upper = mean + 8.0 * std
    grid = jnp.linspace(lower, upper, num_quadrature_points)
    skew = jnp.sqrt(jnp.maximum(1.0 - rho**2, _PSEUDO_NOISE_FLOOR))
    log_skew_cdf = jax.scipy.stats.norm.logcdf((gamma - rho * grid) / skew)
    density = jnp.exp(jax.scipy.stats.norm.logpdf(grid) + log_skew_cdf - log_cdf_gamma)
    step = (upper - lower) / (num_quadrature_points - 1)
    expectation = (
        step * jnp.sum(_simpson_weights(num_quadrature_points) * density * log_skew_cdf) / 3.0
    )
    return 0.5 * rho**2 * gamma * ratio - log_cdf_gamma + expectation


def mumbo_acquisition(
    *,
    state: LinearMultiFidelityGPState,
    x_candidates: jax.Array,
    candidate_levels: jax.Array,
    target_level: int,
    rng_key: jax.Array,
    grid_size: int = 1000,
    num_gumbel_samples: int = 10,
    num_quadrature_points: int = 5001,
) -> jax.Array:
    r"""MUMBO multi-fidelity acquisition score per candidate.

    Args:
        state: Fitted linear-MF GP state.
        x_candidates: ``(m, d)`` candidate inputs (without level column).
        candidate_levels: ``(m,)`` integer fidelity levels.
        target_level: Highest-fidelity (objective) level index.
        rng_key: JAX PRNG key for Gumbel sampling.
        grid_size: Random grid size used to fit the Gumbel
            approximation to the target-level GP marginal.
        num_gumbel_samples: Number of Monte-Carlo samples drawn from
            the fitted Gumbel for the outer expectation.
        num_quadrature_points: Odd number of Simpson-rule points for
            the expectation in eq. 5.

    Returns:
        ``(m,)`` acquisition scores (information gain in nats).
        Higher = more informative candidate.
    """
    num_candidates = x_candidates.shape[0]
    grid_key, gumbel_key = jax.random.split(rng_key)
    domain_low = jnp.min(state.x_augmented[:, :-1], axis=0)
    domain_high = jnp.max(state.x_augmented[:, :-1], axis=0)
    grid_inputs = (
        jax.random.uniform(grid_key, (grid_size, x_candidates.shape[1]))
        * (domain_high - domain_low)
        + domain_low
    )
    target_grid_predictive = predict_linear_multi_fidelity_gp(
        state=state, x_test=grid_inputs, target_level=target_level
    )
    if target_grid_predictive.variance is None:
        raise RuntimeError("Target-level grid predictive missing variance.")
    gumbel_samples = _gumbel_fit_and_sample(
        grid_means=target_grid_predictive.mean,
        grid_stds=jnp.sqrt(target_grid_predictive.variance),
        num_samples=num_gumbel_samples,
        rng_key=gumbel_key,
    )

    target_predictive = predict_linear_multi_fidelity_gp(
        state=state, x_test=x_candidates, target_level=target_level
    )
    if target_predictive.variance is None:
        raise RuntimeError("Target-level candidate predictive missing variance.")
    target_means = target_predictive.mean
    target_vars = jnp.clip(target_predictive.variance, min=_PSEUDO_NOISE_FLOOR)
    target_stds = jnp.sqrt(target_vars)
    noise_variance = state.noise_std**2

    def per_candidate_acquisition(
        candidate_x: jax.Array,
        candidate_level: jax.Array,
        target_mean: jax.Array,
        target_std: jax.Array,
    ) -> jax.Array:
        """Compute the information-gain acquisition for one (input, fidelity) candidate."""
        candidate_augmented = jnp.concatenate(
            [candidate_x, candidate_level.reshape(1).astype(candidate_x.dtype)]
        ).reshape(1, -1)
        # Candidate latent predictive variance via the kernel diagonal.
        k_cc = linear_multi_fidelity_kernel(
            candidate_augmented,
            candidate_augmented,
            lengthscales=state.lengthscales,
            output_scales=state.output_scales,
            scaling_factors=state.scaling_factors,
            base_kernel_fn=state.base_kernel_fn,
        )
        k_train_c = linear_multi_fidelity_kernel(
            candidate_augmented,
            state.x_augmented,
            lengthscales=state.lengthscales,
            output_scales=state.output_scales,
            scaling_factors=state.scaling_factors,
            base_kernel_fn=state.base_kernel_fn,
        )
        v_solve = jax.scipy.linalg.solve_triangular(state.cholesky, k_train_c.T, lower=True)
        candidate_var = jnp.maximum(
            k_cc.squeeze() - jnp.sum(v_solve**2, axis=0).squeeze(),
            _PSEUDO_NOISE_FLOOR,
        )
        # Joint cross-covariance between candidate and same-x target-level
        # posterior. Uses the closed-form posterior covariance:
        #   K_post(a, b) = K_prior(a, b) - K(a, X) (K + σ² I)^-1 K(X, b).
        target_augmented = _augment_with_level(candidate_x.reshape(1, -1), target_level)
        k_ct_prior = linear_multi_fidelity_kernel(
            candidate_augmented,
            target_augmented,
            lengthscales=state.lengthscales,
            output_scales=state.output_scales,
            scaling_factors=state.scaling_factors,
            base_kernel_fn=state.base_kernel_fn,
        )
        k_train_t = linear_multi_fidelity_kernel(
            target_augmented,
            state.x_augmented,
            lengthscales=state.lengthscales,
            output_scales=state.output_scales,
            scaling_factors=state.scaling_factors,
            base_kernel_fn=state.base_kernel_fn,
        )
        v_solve_target = jax.scipy.linalg.solve_triangular(state.cholesky, k_train_t.T, lower=True)
        k_ct_post = k_ct_prior.squeeze() - jnp.sum(v_solve * v_solve_target, axis=0).squeeze()
        # Correlation between the noisy observation y and the target g (Appendix A.1).
        correlation = k_ct_post / (target_std * jnp.sqrt(candidate_var + noise_variance))
        gammas = (gumbel_samples - target_mean) / target_std

        def per_sample_gain(gamma: jax.Array) -> jax.Array:
            """Return eq. 5 for one sample of the maximum."""
            return _mumbo_information_gain(
                correlation=correlation,
                gamma=gamma,
                num_quadrature_points=num_quadrature_points,
            )

        return jnp.mean(jax.vmap(per_sample_gain)(gammas))

    candidate_indices = jnp.arange(num_candidates)

    def scan_step(_: jax.Array, idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Evaluate the acquisition for the candidate at index ``idx``."""
        score = per_candidate_acquisition(
            x_candidates[idx],
            candidate_levels[idx],
            target_means[idx],
            target_stds[idx],
        )
        return _, score

    _, scores = jax.lax.scan(scan_step, jnp.asarray(0.0), candidate_indices)
    return scores


__all__ = ["mumbo_acquisition"]
