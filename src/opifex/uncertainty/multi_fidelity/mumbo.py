r"""MUMBO multi-fidelity Bayesian-optimisation acquisition (Moss+ 2020).

Implements the Multi-task Max-value Bayesian Optimisation acquisition
of Moss, Leslie, Rayson (2020) on top of opifex's linear
multi-fidelity GP. Each candidate is a ``(x, fidelity_level)`` pair;
the score is the **mutual information** between the (latent) maximum
value at the target (highest-fidelity) level and observing the
candidate.

Algorithm
---------

1. **Gumbel sampling on the target fidelity.** Fit a Gumbel
   approximation to the target-level GP marginal across a random grid
   of inputs, then draw ``num_gumbel_samples`` plausible objective
   maxima ``y_max``.
2. **For each candidate** ``(x_i, level_i)``:

   a. Predict GP posterior ``(fmean, fvar)`` at the candidate.
   b. Predict GP posterior ``(gmean, gvar)`` at ``(x_i,
      target_level)``.
   c. Compute the cross-covariance ``cov = K_joint((x_i, level_i),
      (x_i, target_level))`` from the linear-MF kernel.
   d. Correlation ``rho = cov / (sigma_f sigma_g)``.
   e. **Extended Skew Gaussian (ESG)** parameters of the conditional
      distribution ``f | g > y_max`` are obtained in closed form from
      ``(rho, fmean, fvar, gmean, gvar, y_max)`` (Moss+ 2020 §3,
      following Owen 1956 ESG identities).
   f. Numerical entropy of the ESG via Simpson-rule integration.

3. **MC-average** the per-Gumbel entropies and convert to information
   gain: ``acquisition = 0.5 log(2 pi e) - H_avg``.

Cost weighting (dividing by per-level query cost) is left to the
caller — different BO loops apply it differently.

References
----------
* Moss, Leslie, Rayson 2020 — *MUMBO: MUlti-task Max-value Bayesian
  Optimisation*, ECML-PKDD.
* Wang, Jegelka 2017 — *Max-value Entropy Search for Efficient
  Bayesian Optimization*, ICML (single-fidelity MES baseline).
* ``emukit.bayesian_optimization.acquisitions.max_value_entropy_search.MUMBO``
  (PRIMARY).
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
    r"""Fit a Gumbel approximation to the target-level GP marginal and sample.

    Uses the canonical ``binary-search-on-the-CDF`` recipe from
    Wang & Jegelka 2017 (and emukit's ``_fit_gumbel``):

        cdf_F(y) = prod_i Phi((y - mu_i) / sigma_i),
        a       = y at cdf = 0.25,
        b       = y at cdf = 0.75 - a / log(log(4) / log(4/3)).

    Then ``y_max = a - b log(-log(uniform_sample))``.
    """
    # Build search grid by spanning candidate maxima across +/- 5 sigma.
    y_low = jnp.min(grid_means - 5.0 * grid_stds)
    y_high = jnp.max(grid_means + 5.0 * grid_stds)
    levels = jnp.linspace(y_low, y_high, 200)

    def cdf_at(level: jax.Array) -> jax.Array:
        normalised = (level - grid_means) / jnp.maximum(grid_stds, _PSEUDO_NOISE_FLOOR)
        # Probability that ALL grid maxima are <= level.
        return jnp.prod(jax.scipy.stats.norm.cdf(normalised))

    cdfs = jax.vmap(cdf_at)(levels)
    # Interpolate quantile points 0.25 and 0.75.
    quantile_25 = jnp.interp(0.25, cdfs, levels)
    quantile_75 = jnp.interp(0.75, cdfs, levels)
    scale = (quantile_25 - quantile_75) / jnp.log(jnp.log(4.0) / jnp.log(4.0 / 3.0))
    location = quantile_25 + scale * jnp.log(jnp.log(4.0))
    uniform_samples = jax.random.uniform(rng_key, (num_samples,), minval=1e-6, maxval=1.0 - 1e-6)
    return location - scale * jnp.log(-jnp.log(uniform_samples))


def _esg_entropy(
    *,
    correlation: jax.Array,
    gamma: jax.Array,
    num_quadrature_points: int = 1000,
) -> jax.Array:
    r"""Differential entropy of the Extended Skew Gaussian (ESG).

    Closed-form formula for the ESG conditional density of ``f | g >
    threshold`` derived in Moss+ 2020 §3 (Owen 1956 identities). The
    entropy is computed by 1D Simpson-rule integration over +/- 8
    standard deviations of the ESG.
    """
    safe_correlation = jnp.clip(correlation, a_min=-1.0 + 1e-6, a_max=1.0 - 1e-6)
    minus_cdf = jnp.maximum(1.0 - jax.scipy.stats.norm.cdf(gamma), _PSEUDO_NOISE_FLOOR)
    pdf_gamma = jax.scipy.stats.norm.pdf(gamma)
    esg_mean = safe_correlation * pdf_gamma / minus_cdf
    esg_var = jnp.maximum(
        1.0 + safe_correlation * esg_mean * (gamma - pdf_gamma / minus_cdf),
        _PSEUDO_NOISE_FLOOR,
    )
    esg_std = jnp.sqrt(esg_var)
    lower = esg_mean - 8.0 * esg_std
    upper = esg_mean + 8.0 * esg_std
    grid = jnp.linspace(lower, upper, num_quadrature_points)
    minus_corr_sq = jnp.sqrt(jnp.maximum(1.0 - safe_correlation**2, _PSEUDO_NOISE_FLOOR))
    density = (
        jax.scipy.stats.norm.pdf(grid)
        * (1.0 - jax.scipy.stats.norm.cdf((gamma - safe_correlation * grid) / minus_corr_sq))
        / minus_cdf
    )
    safe_density = jnp.maximum(density, _PSEUDO_NOISE_FLOOR)
    entropy_terms = -density * jnp.log(safe_density)
    # Simpson-rule weights for ``num_quadrature_points`` evenly-spaced points.
    step = (upper - lower) / (num_quadrature_points - 1)
    weights = jnp.ones(num_quadrature_points)
    weights = weights.at[1:-1:2].set(4.0)
    weights = weights.at[2:-1:2].set(2.0)
    return step * jnp.sum(weights * entropy_terms) / 3.0


def mumbo_acquisition(
    *,
    state: LinearMultiFidelityGPState,
    x_candidates: jax.Array,
    candidate_levels: jax.Array,
    target_level: int,
    rng_key: jax.Array,
    grid_size: int = 1000,
    num_gumbel_samples: int = 10,
    num_quadrature_points: int = 1000,
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
        num_quadrature_points: Number of Simpson-rule points used for
            the ESG entropy integral.

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
    target_vars = jnp.clip(target_predictive.variance, a_min=_PSEUDO_NOISE_FLOOR)
    target_stds = jnp.sqrt(target_vars)

    def per_candidate_acquisition(
        candidate_x: jax.Array,
        candidate_level: jax.Array,
        target_std: jax.Array,
    ) -> jax.Array:
        candidate_augmented = jnp.concatenate(
            [candidate_x, candidate_level.reshape(1).astype(candidate_x.dtype)]
        ).reshape(1, -1)
        # Candidate predictive variance via the kernel diagonal.
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
        candidate_mean = (k_train_c @ state.alpha).squeeze()
        v_solve = jax.scipy.linalg.solve_triangular(state.cholesky, k_train_c.T, lower=True)
        candidate_var = jnp.maximum(
            k_cc.squeeze() - jnp.sum(v_solve**2, axis=0).squeeze(),
            _PSEUDO_NOISE_FLOOR,
        )
        candidate_std = jnp.sqrt(candidate_var)
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
        correlation = k_ct_post / (candidate_std * target_std)
        gammas = (gumbel_samples - candidate_mean) / candidate_std

        def per_gumbel_entropy(gamma: jax.Array) -> jax.Array:
            return _esg_entropy(
                correlation=correlation,
                gamma=gamma,
                num_quadrature_points=num_quadrature_points,
            )

        entropies = jax.vmap(per_gumbel_entropy)(gammas)
        mean_entropy = jnp.mean(entropies)
        return 0.5 * jnp.log(2.0 * jnp.pi * jnp.e) - mean_entropy

    candidate_indices = jnp.arange(num_candidates)

    def scan_step(_: jax.Array, idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        score = per_candidate_acquisition(
            x_candidates[idx],
            candidate_levels[idx],
            target_stds[idx],
        )
        return _, score

    _, scores = jax.lax.scan(scan_step, jnp.asarray(0.0), candidate_indices)
    return scores


__all__ = ["mumbo_acquisition"]
