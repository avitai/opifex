r"""JAX-native ports of specialised probabilistic-numerics algorithms.

Three primitives referenced by the probabilistic-numerics adapter
catalogue:

* :func:`manifold_update` — iterated extended Kalman filter (IEKF)
  update enforcing the manifold constraint :math:`g(x) = 0`. The
  residual Jacobian is computed on-the-fly with :func:`jax.jacrev`.

  Sibling reference (READ-ONLY port — never imported at runtime):
  ``ProbNumDiffEq.jl/src/callbacks/manifoldupdate.jl``.

* :func:`dense_output_sample` — single-draw multivariate Gaussian
  sampler used for joint posterior samples at arbitrary density.
  Cholesky-factor based to match the existing covariance square-root
  convention. Caller supplies the smoother posterior at the desired
  (possibly off-grid) time; use :func:`jax.vmap` over the ``key``
  argument for a batch of samples.

  Sibling reference (READ-ONLY port — never imported at runtime):
  ``ProbNumDiffEq.jl/src/solution_sampling.jl`` (``_rand`` plus
  ``sample_states`` / ``dense_sample_states`` lines 64-100).

* :func:`apply_diffusion` — scalar or per-dimension diffusion scaling
  applied to a positive-semidefinite process-noise covariance.
  Equivalent to :math:`Q' = (I_{q+1} \otimes \sqrt{\mathrm{diag}(d)})
  Q (I_{q+1} \otimes \sqrt{\mathrm{diag}(d)})` for the vector branch,
  and :math:`Q' = d \, Q` for the scalar branch. Valid only in
  combination with EK0 / blockdiag DiagonalEK1 priors per the Julia
  algorithm-validation routine (``algorithms.jl:108-129``).

  Sibling reference (READ-ONLY port — never imported at runtime):
  ``ProbNumDiffEq.jl/src/diffusions/apply_diffusion.jl``.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

import jax
import jax.numpy as jnp


def manifold_update(
    *,
    mean: jax.Array,
    cov: jax.Array,
    residual_fn: Callable[[jax.Array], jax.Array],
    observation_matrix: jax.Array,
    max_iters: int = 100,
) -> tuple[jax.Array, jax.Array]:
    r"""Iterated extended Kalman filter update for the constraint :math:`g(u) = 0`.

    Given a residual function ``residual_fn`` mapping a projected state
    vector to a residual vector (zero on the constraint manifold) and a
    projection matrix ``observation_matrix`` mapping the full state to
    that observable subspace, this routine performs ``max_iters`` IEKF
    iterations to drive the state mean toward the manifold while
    propagating the posterior covariance.

    For a linear residual the IEKF reduces to a single EKF step after
    one iteration. For nonlinear residuals additional iterations
    refine the linearisation point. The implementation uses a
    fixed-length :func:`jax.lax.scan` so the routine is JIT-friendly;
    early termination on a convergence tolerance would require dynamic
    control flow incompatible with tracing.

    Sibling reference (READ-ONLY port — no runtime import):
    ``ProbNumDiffEq.jl/src/callbacks/manifoldupdate.jl``. The Julia
    reference uses the Joseph-form covariance update
    ``(I - K H) C (I - K H)^T`` for numerical stability with the
    implicit ``R = 0`` observation noise; here we use the standard
    form ``C - K H C`` to stay consistent with
    :func:`opifex.uncertainty.statespace.kalman.kalman_update`. Both
    forms are mathematically equivalent under the optimal gain.

    Args:
        mean: Prior state mean, shape ``(state_dim,)``.
        cov: Prior state covariance, shape ``(state_dim, state_dim)``.
        residual_fn: Vector-valued residual function
            ``residual_fn(u: (observation_dim,)) -> (residual_dim,)``.
            Must be JAX-traceable so :func:`jax.jacrev` can compute its
            Jacobian.
        observation_matrix: Linear projection from state to the
            residual-domain observable, shape
            ``(observation_dim, state_dim)``.
        max_iters: Number of fixed-length IEKF iterations.

    Returns:
        ``(updated_mean, updated_cov)`` — the post-constraint state.
    """
    jacobian_fn = jax.jacrev(residual_fn)

    def step(
        carry: tuple[jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        current_mean, _current_cov = carry
        observation_value = observation_matrix @ current_mean
        residual = residual_fn(observation_value)
        jacobian = jacobian_fn(observation_value)
        residual_to_state = jacobian @ observation_matrix
        innovation_cov = residual_to_state @ cov @ residual_to_state.T
        gain = jnp.linalg.solve(innovation_cov, residual_to_state @ cov).T
        innovation = residual_to_state @ (current_mean - mean) - residual
        new_mean = mean + gain @ innovation
        new_cov = cov - gain @ residual_to_state @ cov
        return (new_mean, new_cov), None

    (final_mean, final_cov), _ = jax.lax.scan(step, (mean, cov), None, length=max_iters)
    return final_mean, final_cov


def dense_output_sample(
    *,
    mean: jax.Array,
    cov: jax.Array,
    key: jax.Array,
) -> jax.Array:
    r"""Draw a single sample from a multivariate Gaussian via a PSD square root.

    Used for joint posterior sampling at arbitrary density: the caller
    interpolates the Kalman smoother posterior to the desired
    (possibly off-grid) time and calls this routine to draw a single
    sample. Batched sampling is obtained by mapping over the ``key``
    argument with :func:`jax.vmap`.

    The square-root factor is computed via symmetric eigendecomposition
    so the routine is robust to singular covariances (zero variance
    collapses to a Dirac at ``mean``), mirroring the PSD-square-root
    convention of the Julia ``_rand`` helper.

    Sibling reference (READ-ONLY port — no runtime import):
    ``ProbNumDiffEq.jl/src/solution_sampling.jl`` (the ``_rand``
    helper and the ``dense_sample_states`` driver at lines 64-100).

    Args:
        mean: Posterior mean, shape ``(state_dim,)``.
        cov: Posterior covariance, shape ``(state_dim, state_dim)``.
            Must be positive semidefinite.
        key: ``jax.random.PRNGKey``.

    Returns:
        A single sample, shape ``(state_dim,)``.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    square_root = eigenvectors * jnp.sqrt(jnp.clip(eigenvalues, min=0.0))
    perturbation = jax.random.normal(key, mean.shape)
    return mean + square_root @ perturbation


def apply_diffusion(
    *,
    process_noise_cov: jax.Array,
    diffusion: jax.Array,
    num_derivatives: int,
) -> jax.Array:
    r"""Scale a process-noise covariance by a scalar or per-dimension diffusion.

    For a scalar ``diffusion``: returns ``diffusion * Q``.
    For a vector ``diffusion`` of length ``wiener_process_dimension``:
    returns ``S Q S`` with
    ``S = \mathrm{diag}(\sqrt{d}) \otimes I_{q+1}``. Because opifex
    builds the prior SDE with :func:`opifex.uncertainty.scientific.
    _priors_sde.iwp_sde` etc. using ``kron(I_d, M_per_dim)``
    state-major ordering, the per-dimension scaling kronecker factor
    has the diagonal on the **outer** index and the identity on the
    inner derivative index — opposite to Julia's derivative-major
    convention.

    Equivalent reformulation: each ``(q+1) x (q+1)`` per-state-dim
    diagonal block of ``Q`` is scaled by ``d_i`` for state dimension
    ``i``; off-diagonal cross-state blocks are scaled by
    ``sqrt(d_i d_j)``.

    The vector branch is valid only with the EK0 or DiagonalEK1
    correction together with a blockdiag covariance factorisation per
    Julia ``algorithms.jl:108-129`` — EK1 with multivariate diffusion
    requires manual calibration.

    Sibling reference (READ-ONLY port — no runtime import):
    ``ProbNumDiffEq.jl/src/diffusions/apply_diffusion.jl``.

    Args:
        process_noise_cov: Symmetric PSD ``Q`` of shape
            ``(d * (q+1), d * (q+1))`` matching opifex state-major
            ordering.
        diffusion: Scalar (0-d) or vector (1-d, length ``d``).
        num_derivatives: ``q`` from the prior — the number of
            integrated-Wiener derivatives. Static under :func:`jax.jit`.

    Returns:
        Scaled covariance of the same shape as ``process_noise_cov``.
    """
    diffusion_array = jnp.asarray(diffusion)
    if diffusion_array.ndim == 0:
        return diffusion_array * process_noise_cov
    sqrt_diagonal = jnp.sqrt(diffusion_array)
    scale_matrix = jnp.kron(jnp.diag(sqrt_diagonal), jnp.eye(num_derivatives + 1))
    return scale_matrix @ process_noise_cov @ scale_matrix
