r"""Bayesian-quadrature acquisitions + experimental-design loop driver.

Five acquisition functions and a greedy loop driver — JAX-native ports
of the emukit reference implementations. The acquisitions are pure
functions that score candidate points; the loop iteratively appends
the arg-max candidate to the training set and evaluates the target
function.

Sibling references (READ-ONLY ports — never imported at runtime):

* :func:`uncertainty_sampling` — ``emukit/quadrature/acquisitions/
  uncertainty_sampling.py:UncertaintySampling``. Posterior variance
  weighted by the measure density: ``a(x) = var(f(x)) · p(x)^q``.

* :func:`model_variance` — ``emukit/experimental_design/acquisitions/
  model_variance.py:ModelVariance``. Pure posterior variance with no
  measure weighting.

* :func:`integral_variance_reduction` —
  ``emukit/quadrature/acquisitions/squared_correlation.py:
  SquaredCorrelation``. Squared correlation between the integral
  value and the integrand evaluation:
  ``a(x) = (qKx - qKX · K_XX⁻¹ · K_Xx)² / (integral_var · y_var)``
  where ``integral_var`` is the current posterior integral variance
  ``qKq - qKX · K_XX⁻¹ · qKX`` and ``y_var`` is the GP predictive
  variance at ``x`` plus the observation noise. Equivalent (up to a
  global normalising constant) to the integral-variance-reduction
  acquisition under a Gaussian-process model.

* :func:`mutual_information` — ``emukit/quadrature/acquisitions/
  mutual_information.py:MutualInformation``. Monotonic transform of
  the squared correlation: ``a(x) = -½ log(1 - ρ²(x))``.

* :func:`integrated_variance_reduction` — ``emukit/
  experimental_design/acquisitions/integrated_variance.py:
  IntegratedVarianceReduction``. Monte-Carlo estimator of the
  expected variance reduction at a held-out reference set if the
  candidate point were added to the training set.

* :func:`experimental_design_loop` — ``emukit/experimental_design/
  experimental_design_loop.py:ExperimentalDesignLoop``. Greedy
  loop that adds the acquisition-maximising candidate at each
  iteration and evaluates the target function there.

This module is the Phase 8.3 active-learning prerequisite per the
design notes.

References
----------
* Osborne, M. A. et al. 2012 — *Active Learning of Model Evidence
  Using Bayesian Quadrature*, NeurIPS 25.
* Briol, F.-X. et al. 2019 — *Probabilistic Integration*, Statistical
  Science 34(1).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager per opifex convention

import jax
import jax.numpy as jnp


def _posterior_variance_at_points(
    points: jax.Array,
    train_points: jax.Array,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    noise_variance: jax.Array,
) -> jax.Array:
    r"""GP posterior variance ``k(x, x) - k(x, X) K_XX⁻¹ k(X, x)`` at each point."""
    num_train = train_points.shape[0]
    gram_matrix = kernel_fn(train_points, train_points)
    gram_regularised = gram_matrix + noise_variance * jnp.eye(num_train)
    cross_kernel = kernel_fn(train_points, points)
    solved = jnp.linalg.solve(gram_regularised, cross_kernel)
    self_kernel = jnp.diagonal(kernel_fn(points, points))
    explained = jnp.sum(cross_kernel * solved, axis=0)
    return self_kernel - explained


def uncertainty_sampling(
    *,
    points: jax.Array,
    train_points: jax.Array,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    noise_variance: jax.Array,
    measure_density_fn: Callable[[jax.Array], jax.Array],
    measure_power: jax.Array,
) -> jax.Array:
    r"""WSABI-L-style uncertainty sampling: ``a(x) = var(f(x)) · p(x)^q``.

    Sibling reference: ``emukit/quadrature/acquisitions/
    uncertainty_sampling.py:UncertaintySampling``.

    Args:
        points: Candidate points ``(n, d)``.
        train_points: Current observation inputs ``(N, d)``.
        kernel_fn: ``(X, Y) -> K`` kernel function.
        noise_variance: Scalar observation noise variance.
        measure_density_fn: Function ``p(x)`` returning the integration
            measure density at each row of ``points``; expected to map
            ``(n, d) -> (n,)``.
        measure_power: Exponent ``q`` on the measure density.

    Returns:
        Acquisition values, shape ``(n,)``.
    """
    variance = _posterior_variance_at_points(
        points, train_points, kernel_fn, noise_variance
    )
    density = measure_density_fn(points)
    return variance * density**measure_power


def model_variance(
    *,
    points: jax.Array,
    train_points: jax.Array,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    noise_variance: jax.Array,
) -> jax.Array:
    r"""Raw GP posterior variance at each candidate point.

    Sibling reference: ``emukit/experimental_design/acquisitions/
    model_variance.py:ModelVariance``.

    Args:
        points: Candidate points ``(n, d)``.
        train_points: Current observation inputs ``(N, d)``.
        kernel_fn: ``(X, Y) -> K`` kernel function.
        noise_variance: Scalar observation noise variance.

    Returns:
        Posterior variance at each candidate, shape ``(n,)``.
    """
    return _posterior_variance_at_points(points, train_points, kernel_fn, noise_variance)


def _squared_correlation_terms(
    points: jax.Array,
    train_points: jax.Array,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    kernel_mean_fn: Callable[[jax.Array], jax.Array],
    double_kernel_mean: jax.Array,
    noise_variance: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Shared terms for squared correlation and mutual information."""
    num_train = train_points.shape[0]
    gram_matrix = kernel_fn(train_points, train_points)
    gram_regularised = gram_matrix + noise_variance * jnp.eye(num_train)
    qkx = kernel_mean_fn(points)  # (n,)
    qkx_train = kernel_mean_fn(train_points)  # (N,)
    cross_kernel = kernel_fn(train_points, points)  # (N, n)

    solved_train_train = jnp.linalg.solve(gram_regularised, qkx_train)  # (N,)
    integral_current_var = double_kernel_mean - qkx_train @ solved_train_train

    solved_train_points = jnp.linalg.solve(gram_regularised, cross_kernel)  # (N, n)
    predictive_cov = qkx - qkx_train @ solved_train_points  # (n,)

    self_kernel = jnp.diagonal(kernel_fn(points, points))
    explained = jnp.sum(cross_kernel * solved_train_points, axis=0)
    y_predictive_var = self_kernel - explained + noise_variance

    return integral_current_var, y_predictive_var, predictive_cov


def integral_variance_reduction(
    *,
    points: jax.Array,
    train_points: jax.Array,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    kernel_mean_fn: Callable[[jax.Array], jax.Array],
    double_kernel_mean: jax.Array,
    noise_variance: jax.Array,
) -> jax.Array:
    r"""Squared correlation between integral value and integrand evaluation.

    ``a(x) = (qKx - qKX · K_XX⁻¹ · K_Xx)² / (integral_var · y_var)``.

    Sibling reference: ``emukit/quadrature/acquisitions/
    squared_correlation.py:SquaredCorrelation``.

    Args:
        points: Candidate points ``(n, d)``.
        train_points: Current observation inputs ``(N, d)``.
        kernel_fn: ``(X, Y) -> K`` kernel function.
        kernel_mean_fn: ``x -> ∫ k(x', x) p(x') dx'`` — closed-form
            kernel mean embedding for the integration measure.
            Mapping ``(n, d) -> (n,)``.
        double_kernel_mean: Scalar ``∫∫ k(x, x') p(x) p(x') dx dx'``.
        noise_variance: Scalar observation noise variance.

    Returns:
        Squared-correlation acquisition value ∈ [0, 1] at each
        candidate, shape ``(n,)``.
    """
    integral_var, y_var, predictive_cov = _squared_correlation_terms(
        points,
        train_points,
        kernel_fn,
        kernel_mean_fn,
        double_kernel_mean,
        noise_variance,
    )
    return predictive_cov**2 / (integral_var * y_var)


def mutual_information(
    *,
    points: jax.Array,
    train_points: jax.Array,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    kernel_mean_fn: Callable[[jax.Array], jax.Array],
    double_kernel_mean: jax.Array,
    noise_variance: jax.Array,
) -> jax.Array:
    r"""Mutual information between integral value and integrand evaluation.

    ``a(x) = -½ log(1 - ρ²(x))`` where ``ρ²`` is the squared
    correlation. Monotonic transform of
    :func:`integral_variance_reduction` so it yields the same
    acquisition argmax.

    Sibling reference: ``emukit/quadrature/acquisitions/
    mutual_information.py:MutualInformation``.
    """
    squared_correlation = integral_variance_reduction(
        points=points,
        train_points=train_points,
        kernel_fn=kernel_fn,
        kernel_mean_fn=kernel_mean_fn,
        double_kernel_mean=double_kernel_mean,
        noise_variance=noise_variance,
    )
    return -0.5 * jnp.log1p(-squared_correlation)


def integrated_variance_reduction(
    *,
    points: jax.Array,
    train_points: jax.Array,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    noise_variance: jax.Array,
    monte_carlo_points: jax.Array,
) -> jax.Array:
    r"""Monte-Carlo estimate of expected posterior-variance reduction.

    For each candidate ``x``, computes
    ``E_{x' ~ p}[var_X(x') - var_{X ∪ {x}}(x')]``
    via a Monte-Carlo average over ``monte_carlo_points``. The closed
    form for the variance reduction at ``x'`` is

    ``Δvar(x') = k_N(x', x)² / (k_N(x, x) + σ²)``,

    where ``k_N`` is the posterior kernel given the current training
    set. This avoids re-solving the gram matrix for each candidate.

    Sibling reference: ``emukit/experimental_design/acquisitions/
    integrated_variance.py:IntegratedVarianceReduction``.
    """
    num_train = train_points.shape[0]
    gram_matrix = kernel_fn(train_points, train_points)
    gram_regularised = gram_matrix + noise_variance * jnp.eye(num_train)

    train_to_candidate = kernel_fn(train_points, points)  # (N, n_candidates)
    train_to_mc = kernel_fn(train_points, monte_carlo_points)  # (N, n_mc)

    solved_candidate = jnp.linalg.solve(gram_regularised, train_to_candidate)  # (N, n_candidates)

    candidate_self = jnp.diagonal(kernel_fn(points, points))
    candidate_var = candidate_self - jnp.sum(train_to_candidate * solved_candidate, axis=0)
    candidate_var_with_noise = candidate_var + noise_variance

    candidate_mc_kernel = kernel_fn(points, monte_carlo_points)  # (n_candidates, n_mc)
    posterior_cross = candidate_mc_kernel - solved_candidate.T @ train_to_mc  # (n_candidates, n_mc)

    variance_reduction = posterior_cross**2 / candidate_var_with_noise[:, None]
    return jnp.mean(variance_reduction, axis=1)


def experimental_design_loop(
    *,
    initial_points: jax.Array,
    initial_values: jax.Array,
    target_fn: Callable[[jax.Array], jax.Array],
    acquisition_fn: Callable[[jax.Array, jax.Array], jax.Array],
    candidate_pool: jax.Array,
    num_iterations: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Greedy experimental design — append acquisition-max candidate each step.

    Sibling reference: ``emukit/experimental_design/
    experimental_design_loop.py:ExperimentalDesignLoop``.

    Args:
        initial_points: Starting training inputs ``(N0, d)``.
        initial_values: Starting target values ``(N0,)``.
        target_fn: Maps a single input ``x`` of shape ``(d,)`` to a
            scalar target value.
        acquisition_fn: ``(candidate_pool, train_points) -> (M,)`` —
            scores each row of ``candidate_pool`` given the current
            training set.
        candidate_pool: Fixed discrete candidate set ``(M, d)``. The
            loop selects the arg-max acquisition row each iteration.
        num_iterations: Number of points to append. Static.

    Returns:
        ``(final_points, final_values)`` — the augmented training set
        with shape ``(N0 + num_iterations, d)`` and ``(N0 + num_iterations,)``.
    """
    train_points = initial_points
    train_values = initial_values
    for _ in range(num_iterations):
        scores = acquisition_fn(candidate_pool, train_points)
        best_index = jnp.argmax(scores)
        new_point = candidate_pool[best_index]
        new_value = target_fn(new_point)
        train_points = jnp.concatenate([train_points, new_point[None, :]], axis=0)
        train_values = jnp.concatenate([train_values, jnp.atleast_1d(new_value)], axis=0)
    return train_points, train_values
