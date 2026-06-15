r"""JAX-native ports of specialised probabilistic-numerics algorithms.

Four primitives referenced by the probabilistic-numerics adapter
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

* :func:`perturbed_step_solve` — Conrad+ 2017 perturbed-step
  probabilistic ODE solver. A deterministic one-step integrator
  :math:`\Psi_h` of order :math:`p` (Euler, :math:`p = 1`; classical
  RK4, :math:`p = 4`) is run on the grid and, at every step, a
  calibrated mean-zero Gaussian state perturbation
  :math:`\xi_k \sim \mathcal{N}(0, \sigma^2 h^{2p+1} I)` is added. The
  :math:`h^{2p+1}` covariance scaling (per-step std
  :math:`\propto h^{p+1/2}`) is the order-consistent choice that keeps
  the randomised method at the base integrator's order :math:`p` in the
  mean while the ensemble spread quantifies discretisation uncertainty.

  Reference (paper, not a code port): Conrad, Girolami, Särkkä,
  Stuart, Zygalakis 2017 — *Statistical analysis of differential
  equations: introducing probability measures on numerical solutions*,
  Statistics and Computing 27, 1065-1082 (arXiv:1506.04592),
  Assumption 1 / Theorem 2.2.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Literal

import jax
import jax.numpy as jnp


# Global convergence order ``p`` of the supported deterministic one-step
# integrators, keyed by method name. Used to set the Conrad+ 2017
# order-consistent perturbation covariance ``sigma^2 h^(2p+1)``.
_SOLVER_ORDERS: dict[str, int] = {"euler": 1, "rk4": 4}


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


def _euler_step(
    vector_field: Callable[[jax.Array, jax.Array], jax.Array],
    time: jax.Array,
    state: jax.Array,
    step_size: jax.Array,
) -> jax.Array:
    r"""Single explicit-Euler step (order :math:`p = 1`)."""
    return state + step_size * vector_field(time, state)


def _rk4_step(
    vector_field: Callable[[jax.Array, jax.Array], jax.Array],
    time: jax.Array,
    state: jax.Array,
    step_size: jax.Array,
) -> jax.Array:
    r"""Single classical Runge-Kutta-4 step (order :math:`p = 4`)."""
    half_step = 0.5 * step_size
    k1 = vector_field(time, state)
    k2 = vector_field(time + half_step, state + half_step * k1)
    k3 = vector_field(time + half_step, state + half_step * k2)
    k4 = vector_field(time + step_size, state + step_size * k3)
    return state + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_DETERMINISTIC_STEPS: dict[
    str,
    Callable[
        [Callable[[jax.Array, jax.Array], jax.Array], jax.Array, jax.Array, jax.Array], jax.Array
    ],
] = {"euler": _euler_step, "rk4": _rk4_step}


def perturbed_step_solve(
    *,
    vector_field: Callable[[jax.Array, jax.Array], jax.Array],
    initial_state: jax.Array,
    t0: float,
    t1: float,
    num_steps: int,
    noise_scale: float,
    num_samples: int,
    key: jax.Array,
    method: Literal["euler", "rk4"] = "rk4",
) -> jax.Array:
    r"""Conrad+ 2017 perturbed-step probabilistic ODE solver.

    Integrates the initial-value problem :math:`\dot y = f(t, y)`,
    :math:`y(t_0) = y_0` on a uniform grid of ``num_steps`` steps with a
    deterministic one-step integrator :math:`\Psi_h` of order :math:`p`
    and, after every step, adds a calibrated mean-zero Gaussian state
    perturbation

    .. math::

        y_{k+1} = \Psi_h(t_k, y_k) + \xi_k, \qquad
        \xi_k \sim \mathcal{N}\!\bigl(0,\; \sigma^2 h^{2p+1} I\bigr).

    The per-step standard deviation therefore scales as
    :math:`\sigma h^{p+1/2}`. This is the order-consistent calibration
    of Conrad+ 2017 (Assumption 1): the random perturbation is small
    enough that the randomised method retains the base integrator's
    order-:math:`p` convergence in the mean (Theorem 2.2), while the
    spread across the returned ensemble quantifies the discretisation
    uncertainty. With ``noise_scale = 0`` every ensemble member equals
    the deterministic trajectory exactly.

    The trajectory of each ensemble member is advanced with a
    fixed-length :func:`jax.lax.scan`; the ensemble is produced by a
    :func:`jax.vmap` over per-member PRNG keys so the routine composes
    with :func:`jax.jit`, :func:`jax.grad`, and an outer
    :func:`jax.vmap` over the seed argument.

    Reference (paper, not a code port): Conrad, Girolami, Särkkä,
    Stuart, Zygalakis 2017 — *Statistical analysis of differential
    equations: introducing probability measures on numerical
    solutions*, Statistics and Computing 27, 1065-1082
    (arXiv:1506.04592).

    Args:
        vector_field: Right-hand side ``f(t, y)`` mapping a scalar time
            and a state vector of shape ``(state_dim,)`` to a
            derivative of the same shape. Must be JAX-traceable.
        initial_state: Initial condition :math:`y_0`, shape
            ``(state_dim,)``. Shared (unperturbed) across the ensemble.
        t0: Start time of the integration interval.
        t1: End time of the integration interval (``t1 > t0``).
        num_steps: Number of uniform steps; the step size is
            ``h = (t1 - t0) / num_steps``. Static under :func:`jax.jit`.
        noise_scale: Perturbation scale :math:`\sigma \geq 0`. Zero
            recovers the deterministic solver exactly.
        num_samples: Ensemble size. Static under :func:`jax.jit`.
        key: ``jax.random.PRNGKey`` seeding the ensemble perturbations.
        method: Deterministic base integrator: ``"euler"`` (order 1) or
            ``"rk4"`` (order 4). Static under :func:`jax.jit`.

    Returns:
        The ensemble of trajectories, shape
        ``(num_samples, num_steps + 1, state_dim)``. The first time
        slice is the shared, unperturbed initial condition.

    Raises:
        ValueError: If ``method`` is not a supported integrator or the
            time interval is non-positive.
    """
    if method not in _DETERMINISTIC_STEPS:
        raise ValueError(
            f"Unsupported integrator {method!r}; choose from {tuple(_DETERMINISTIC_STEPS)}."
        )
    # Validate the interval eagerly only when ``t0``/``t1`` are concrete
    # Python scalars; under :func:`jax.jit` they may be tracers, for which
    # boolean comparison is undefined.
    if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and t1 <= t0:
        raise ValueError(f"Require t1 > t0, got t0={t0}, t1={t1}.")

    deterministic_step = _DETERMINISTIC_STEPS[method]
    solver_order = _SOLVER_ORDERS[method]
    initial_state = jnp.asarray(initial_state)
    state_dtype = initial_state.dtype
    start_time = jnp.asarray(t0, dtype=state_dtype)
    step_size = jnp.asarray((t1 - t0) / num_steps, dtype=state_dtype)
    # Conrad+ 2017 order-consistent per-step perturbation std:
    # sqrt(sigma^2 h^(2p+1)) = sigma * h^(p + 1/2).
    perturbation_std = jnp.asarray(noise_scale, dtype=state_dtype) * step_size ** (
        solver_order + 0.5
    )

    def single_trajectory(member_key: jax.Array) -> jax.Array:
        def step(
            carry: tuple[jax.Array, jax.Array], step_index: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            time, state = carry
            propagated = deterministic_step(vector_field, time, state, step_size)
            step_key = jax.random.fold_in(member_key, step_index)
            perturbation = perturbation_std * jax.random.normal(
                step_key, state.shape, dtype=state_dtype
            )
            new_state = propagated + perturbation
            return (time + step_size, new_state), new_state

        step_indices = jnp.arange(num_steps)
        _, propagated_states = jax.lax.scan(step, (start_time, initial_state), step_indices)
        return jnp.concatenate([initial_state[None, :], propagated_states], axis=0)

    member_keys = jax.random.split(key, num_samples)
    return jax.vmap(single_trajectory)(member_keys)
