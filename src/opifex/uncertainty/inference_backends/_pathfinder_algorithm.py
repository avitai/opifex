r"""Pathfinder variational inference (Zhang+ 2022) JAX-native primitives.

Line-by-line port of the Pathfinder reference at
``../blackjax/blackjax/vi/pathfinder.py`` plus its L-BFGS helpers at
``../blackjax/blackjax/optimizers/lbfgs.py``. The algorithm
(Zhang et al, "Pathfinder: Parallel quasi-Newton variational
inference", JMLR 23(306), arXiv:2108.03782) locates normal
approximations to the target density along a quasi-Newton optimization
path, with local covariance estimated by the inverse-Hessian factors
produced by L-BFGS, and returns the iteration with the highest ELBO.

The vendored primitives:

* :func:`lbfgs_recover_alpha` — diagonal inverse-Hessian update,
  Algorithm 3 inner loop (blackjax ``lbfgs_recover_alpha`` line 278).
* :func:`lbfgs_inverse_hessian_factors` — formula II.2 ``(beta, gamma)``
  factors (blackjax ``lbfgs_inverse_hessian_factors`` line 327).
* :func:`bfgs_sample` — Algorithm 4 sampler given factored inverse
  Hessian (blackjax ``bfgs_sample`` line 379).
* :func:`pathfinder_approximate` — top-level entry that runs L-BFGS
  via optax, builds the per-step Gaussians, picks the argmax-ELBO
  state (blackjax ``approximate`` line 70).
* :func:`pathfinder_sample` — draws from the selected Gaussian
  (blackjax ``sample`` line 200).

References
----------
* Zhang, L., Carpenter, B., Gelman, A., Vehtari, A. 2022 —
  *Pathfinder: Parallel quasi-Newton variational inference*, JMLR
  23(306). arXiv:2108.03782.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager per opifex convention
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import lax


class PathfinderState(NamedTuple):
    """Selected (highest-ELBO) Pathfinder iteration.

    Sibling reference: ``blackjax/vi/pathfinder.py:PathfinderState``
    (line 36).
    """

    elbo: jax.Array
    position: jax.Array
    grad_position: jax.Array
    alpha: jax.Array
    beta: jax.Array
    gamma: jax.Array


class _LBFGSHistory(NamedTuple):
    """Per-iteration L-BFGS history slice.

    Mirrors ``blackjax/optimizers/lbfgs.py:LBFGSHistory`` (line 37).
    """

    x: jax.Array
    f: jax.Array
    g: jax.Array
    alpha: jax.Array
    update_mask: jax.Array


def lbfgs_recover_alpha(
    alpha_previous: jax.Array,
    s_step: jax.Array,
    z_step: jax.Array,
    epsilon: float = 1e-12,
) -> tuple[jax.Array, jax.Array]:
    r"""Diagonal inverse-Hessian update for one L-BFGS step.

    Implements the Algorithm 3 inner loop of Zhang+ 2022. The update
    only fires when the curvature predicate ``s·z > eps · ||z||``
    holds; otherwise ``alpha`` is kept unchanged.

    Sibling reference: ``blackjax/optimizers/lbfgs.py:lbfgs_recover_alpha``
    (line 278).

    Args:
        alpha_previous: Diagonal inverse-Hessian ``alpha_{l-1}``.
        s_step: Position increment ``s_l = x_l - x_{l-1}``.
        z_step: Gradient increment ``z_l = g_l - g_{l-1}``.
        epsilon: Curvature-predicate slack.

    Returns:
        ``(alpha_l, update_mask)`` — the new diagonal Hessian factor
        and a boolean array (same shape as ``alpha_l``) flagging
        whether the update fired.
    """

    def _compute_next_alpha(
        s_step_inner: jax.Array, z_step_inner: jax.Array, alpha_inner: jax.Array
    ) -> jax.Array:
        """Update the diagonal inverse-Hessian estimate from one L-BFGS pair."""
        a = z_step_inner.T @ jnp.diag(alpha_inner) @ z_step_inner
        b = z_step_inner.T @ s_step_inner
        c = s_step_inner.T @ jnp.diag(1.0 / alpha_inner) @ s_step_inner
        inv_alpha_l = (
            a / (b * alpha_inner)
            + z_step_inner**2 / b
            - (a * s_step_inner**2) / (b * c * alpha_inner**2)
        )
        return 1.0 / inv_alpha_l

    secant_predicate = s_step.T @ z_step > (epsilon * jnp.linalg.norm(z_step, 2))
    alpha_new = lax.cond(
        secant_predicate,
        _compute_next_alpha,
        lambda *_: alpha_previous,
        s_step,
        z_step,
        alpha_previous,
    )
    update_mask = jnp.where(
        secant_predicate,
        jnp.ones_like(alpha_previous, dtype=bool),
        jnp.zeros_like(alpha_previous, dtype=bool),
    )
    return alpha_new, update_mask


def lbfgs_inverse_hessian_factors(
    history_S: jax.Array, history_Z: jax.Array, alpha: jax.Array
) -> tuple[jax.Array, jax.Array]:
    r"""Factored representation of the inverse Hessian (formula II.2).

    ``H = diag(alpha) + beta · gamma · betaᵀ`` with the factored
    ``(beta, gamma)`` returned here. ``S``, ``Z`` are matrices of
    column-stacked position / gradient increments of shape
    ``(d, maxcor)``.

    Sibling reference: ``blackjax/optimizers/lbfgs.py:
    lbfgs_inverse_hessian_factors`` (line 327).
    """
    param_dim = history_S.shape[-1]
    StZ = history_S.T @ history_Z
    upper_triangular = jnp.triu(StZ) + jnp.eye(param_dim) * jnp.finfo(history_S.dtype).eps
    eta = jnp.diag(StZ)
    beta = jnp.hstack([jnp.diag(alpha) @ history_Z, history_S])
    minus_inverse = -jnp.linalg.inv(upper_triangular)
    alpha_z = jnp.diag(jnp.sqrt(alpha)) @ history_Z
    block_dd = minus_inverse.T @ (alpha_z.T @ alpha_z + jnp.diag(eta)) @ minus_inverse
    gamma = jnp.block(
        [
            [jnp.zeros((param_dim, param_dim)), minus_inverse],
            [minus_inverse.T, block_dd],
        ]
    )
    return beta, gamma


def bfgs_sample(
    *,
    rng_key: jax.Array,
    num_samples: int,
    position: jax.Array,
    grad_position: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Draw samples from a Pathfinder Gaussian (Algorithm 4).

    Returns ``(samples, log_q)`` where ``log_q`` is the per-sample log
    density of the variational approximation.

    Sibling reference: ``blackjax/optimizers/lbfgs.py:bfgs_sample``
    (line 379).
    """
    Q_matrix, R_matrix = jnp.linalg.qr(jnp.diag(jnp.sqrt(1.0 / alpha)) @ beta)
    param_dim = beta.shape[0]
    identity = jnp.identity(R_matrix.shape[0])
    cholesky_lower = jnp.linalg.cholesky(identity + R_matrix @ gamma @ R_matrix.T)

    log_det = jnp.log(jnp.prod(alpha)) + 2.0 * jnp.log(jnp.linalg.det(cholesky_lower))
    mean = position + jnp.diag(alpha) @ grad_position + beta @ gamma @ beta.T @ grad_position
    standard_noise = jax.random.normal(rng_key, (num_samples, param_dim, 1))
    transformed = mean[..., None] + jnp.diag(jnp.sqrt(alpha)) @ (
        Q_matrix @ (cholesky_lower - identity) @ (Q_matrix.T @ standard_noise) + standard_noise
    )
    log_density = -0.5 * (
        log_det
        + jnp.einsum("...ji,...ji->...", standard_noise, standard_noise)
        + param_dim * jnp.log(2.0 * jnp.pi)
    )
    return transformed[..., 0], log_density


def _run_lbfgs_with_history(
    objective_fn: Callable[[jax.Array], jax.Array],
    initial_position: jax.Array,
    maxiter: int,
    maxcor: int,
    gtol: float,
    ftol: float,
    maxls: int,
) -> _LBFGSHistory:
    """Run optax L-BFGS and record per-step history (positions, gradients, alpha).

    Compresses the relevant pieces of
    ``blackjax/optimizers/lbfgs.py:_minimize_lbfgs`` (line 165) into
    the history that Pathfinder needs.
    """
    linesearch = optax.scale_by_zoom_linesearch(max_linesearch_steps=maxls)
    solver = optax.lbfgs(memory_size=maxcor, linesearch=linesearch)
    value_and_grad_fn = optax.value_and_grad_from_state(objective_fn)

    opt_state = solver.init(initial_position)
    value_0, grad_0 = jax.value_and_grad(objective_fn)(initial_position)

    initial_history = _LBFGSHistory(
        x=initial_position,
        f=value_0,
        g=grad_0,
        alpha=jnp.ones_like(initial_position),
        update_mask=jnp.zeros_like(initial_position, dtype=bool),
    )

    def step(
        carry: tuple[tuple[jax.Array, optax.OptState], _LBFGSHistory], iter_index: jax.Array
    ) -> tuple[tuple[tuple[jax.Array, optax.OptState], _LBFGSHistory], _LBFGSHistory]:
        """Run one L-BFGS iteration and append its position/gradient to the history."""
        (params, state), previous_history = carry
        value, gradient = value_and_grad_fn(params, state=state)
        updates, new_state = solver.update(
            gradient, state, params, value=value, grad=gradient, value_fn=objective_fn
        )
        new_params = jnp.asarray(optax.apply_updates(params, updates))
        new_value, new_grad = jax.value_and_grad(objective_fn)(new_params)
        s_step = new_params - params
        z_step = new_grad - gradient

        alpha_new, update_mask = lbfgs_recover_alpha(previous_history.alpha, s_step, z_step)
        new_history = _LBFGSHistory(
            x=new_params,
            f=new_value,
            g=new_grad,
            alpha=alpha_new,
            update_mask=update_mask,
        )
        f_delta = (
            jnp.abs(value - new_value)
            / jnp.asarray([jnp.abs(value), jnp.abs(new_value), 1.0]).max()
        )
        not_converged = (
            (jnp.linalg.norm(gradient) > gtol) & (f_delta > ftol) & (iter_index < maxiter)
        )

        def _take_step(carry_in: tuple[tuple[jax.Array, optax.OptState], _LBFGSHistory]):
            """Accept the new L-BFGS state when the step is valid."""
            del carry_in
            return (new_params, new_state), new_history

        def _stay(carry_in: tuple[tuple[jax.Array, optax.OptState], _LBFGSHistory]):
            """Keep the previous L-BFGS state when the step is rejected."""
            existing_inner, existing_history = carry_in
            return existing_inner, existing_history

        next_carry = lax.cond(not_converged, _take_step, _stay, ((params, state), new_history))
        # Always record the history slot — masked-False steps repeat the previous params.
        return ((next_carry[0]), next_carry[1]), next_carry[1]

    init_carry = ((initial_position, opt_state), initial_history)
    _, history = lax.scan(step, init_carry, jnp.arange(maxiter))

    return _LBFGSHistory(
        x=jnp.concatenate([initial_history.x[None, ...], history.x], axis=0),
        f=jnp.concatenate([initial_history.f[None, ...], history.f], axis=0),
        g=jnp.concatenate([initial_history.g[None, ...], history.g], axis=0),
        alpha=jnp.concatenate([initial_history.alpha[None, ...], history.alpha], axis=0),
        update_mask=jnp.concatenate(
            [initial_history.update_mask[None, ...], history.update_mask], axis=0
        ),
    )


def pathfinder_approximate(
    *,
    rng_key: jax.Array,
    log_density_fn: Callable[[jax.Array], jax.Array],
    initial_position: jax.Array,
    num_samples: int = 64,
    maxiter: int = 30,
    maxcor: int = 6,
    ftol: float = 1e-5,
    gtol: float = 1e-8,
    maxls: int = 1000,
) -> PathfinderState:
    r"""Run Pathfinder and return the highest-ELBO Gaussian approximation.

    Sibling reference: ``blackjax/vi/pathfinder.py:approximate``
    (line 70). Operates on flat arrays — the caller is responsible
    for raveling / unraveling pytrees.

    Args:
        rng_key: PRNG key for ELBO-estimation samples.
        log_density_fn: ``log p(x)`` mapping flat ``(d,) -> scalar``.
        initial_position: L-BFGS starting point, shape ``(d,)``.
        num_samples: Number of Gaussian draws per iteration used to
            estimate the ELBO.
        maxiter: Maximum L-BFGS iterations.
        maxcor: Memory size (history length) of L-BFGS.
        ftol: L-BFGS function-tolerance stopping criterion.
        gtol: L-BFGS gradient-norm stopping criterion.
        maxls: Line-search iteration cap.

    Returns:
        :class:`PathfinderState` for the iteration that maximised
        the ELBO.
    """

    def objective_fn(position: jax.Array) -> jax.Array:
        """Return the negative log-density, the quantity L-BFGS minimises."""
        return -log_density_fn(position)

    history = _run_lbfgs_with_history(
        objective_fn, initial_position, maxiter, maxcor, gtol, ftol, maxls
    )

    position = history.x
    grad_position = history.g
    alpha = history.alpha
    update_mask = history.update_mask[1:]
    s_increments = jnp.diff(position, axis=0)
    z_increments = jnp.diff(grad_position, axis=0)
    s_masked = jnp.where(update_mask, s_increments, jnp.zeros_like(s_increments))
    z_masked = jnp.where(update_mask, z_increments, jnp.zeros_like(z_increments))
    s_padded = jnp.pad(s_masked, ((maxcor, 0), (0, 0)), mode="constant")
    z_padded = jnp.pad(z_masked, ((maxcor, 0), (0, 0)), mode="constant")

    def _per_iteration(
        per_key: jax.Array,
        history_S: jax.Array,
        history_Z: jax.Array,
        alpha_l: jax.Array,
        theta: jax.Array,
        theta_grad: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Build the Gaussian approximation at one L-BFGS iterate and evaluate its ELBO."""
        beta, gamma = lbfgs_inverse_hessian_factors(history_S.T, history_Z.T, alpha_l)
        phi, log_q = bfgs_sample(
            rng_key=per_key,
            num_samples=num_samples,
            position=theta,
            grad_position=theta_grad,
            alpha=alpha_l,
            beta=beta,
            gamma=gamma,
        )
        log_p = -jax.vmap(objective_fn)(phi)
        elbo = (log_p - log_q).mean()
        return elbo, beta, gamma

    path_size = maxiter + 1
    window_index = jnp.arange(path_size)[:, None] + jnp.arange(maxcor)[None, :]
    history_S = s_padded[window_index.reshape(path_size, maxcor)].reshape(path_size, maxcor, -1)
    history_Z = z_padded[window_index.reshape(path_size, maxcor)].reshape(path_size, maxcor, -1)
    per_step_keys = jax.random.split(rng_key, path_size)
    elbo, beta, gamma = jax.vmap(_per_iteration)(
        per_step_keys, history_S, history_Z, alpha, position, grad_position
    )
    elbo = jnp.where(jnp.isfinite(elbo), elbo, -jnp.inf)
    best_index = jnp.argmax(elbo)
    return PathfinderState(
        elbo=elbo[best_index],
        position=position[best_index],
        grad_position=grad_position[best_index],
        alpha=alpha[best_index],
        beta=beta[best_index],
        gamma=gamma[best_index],
    )


def pathfinder_sample(
    *,
    rng_key: jax.Array,
    state: PathfinderState,
    num_samples: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Draw ``num_samples`` samples from the selected Pathfinder Gaussian.

    Sibling reference: ``blackjax/vi/pathfinder.py:sample`` (line 200).
    """
    return bfgs_sample(
        rng_key=rng_key,
        num_samples=num_samples,
        position=state.position,
        grad_position=state.grad_position,
        alpha=state.alpha,
        beta=state.beta,
        gamma=state.gamma,
    )
