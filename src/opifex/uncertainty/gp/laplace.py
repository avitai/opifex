r"""Generic Laplace-approximated GP for any factorising likelihood (RW06 §3.4).

For a likelihood ``p(y | f) = Π_i p(y_i | f_i)`` whose
``(log_lik, ∇log_lik, W = -∇²log_lik)`` triple is available in closed
form, the Laplace approximation (Williams & Barber 1998; Rasmussen &
Williams 2006 §3.4, Algorithm 3.1) replaces the non-Gaussian posterior
``p(f | X, y)`` by a Gaussian centred at the mode ``f̂`` with covariance
``(K^{-1} + W)^{-1}``.

The Newton step (RW06 eq. 3.18) is implemented via the
``B = I + W^{1/2} K W^{1/2}`` Cholesky to keep every intermediate matrix
PSD — the canonical numerically-stable form recommended by RW06.

This module ships the **likelihood-agnostic core** (state container,
Newton scan, log-marginal computation, latent-moment predict). Per-
likelihood wrappers — Bernoulli (binary classification), Poisson
(counts), Student-t (robust regression), Beta (proportions) — live in
:mod:`opifex.uncertainty.gp.laplace_classification` and
:mod:`opifex.uncertainty.gp.laplace_likelihoods` and supply only the
``(log_lik, grad, W, sqrt W)`` quadruple.

Implementation notes — **jittability**
--------------------------------------

* Newton's loop runs through ``jax.lax.scan`` for a static
  ``num_newton_iterations``, so the whole fit compiles end-to-end
  under ``jax.jit``.
* ``predict_laplace_latent_moments`` uses ``cho_solve`` /
  ``solve_triangular`` on the ``B``-Cholesky and avoids any data-
  dependent control flow.
* **No equinox dependency**: the fitted state is a plain
  ``@dataclass(frozen=True, slots=True, kw_only=True)`` carrying
  ``jax.Array`` leaves.

References
----------
* Williams, C. K. I., Barber, D. 1998 — *Bayesian Classification with
  Gaussian Processes*, IEEE TPAMI 20(12).
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes
  for Machine Learning*, MIT Press; §3.4 Algorithm 3.1 (PRIMARY).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp.exact import rbf_kernel


LikelihoodComponentsFn = Callable[
    [jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]
"""Type alias: ``(f, y) -> (log_lik, ∇log_lik, W, √W)``.

``log_lik`` is a scalar ``Σ_i log p(y_i | f_i)``; the remaining
quantities are ``(n,)`` per-observation arrays. ``W = -∇² log p(y | f̂)``
is a diagonal in the Laplace approximation; each per-likelihood wrapper
is responsible for clipping ``W`` to be non-negative so the
``B = I + W^{1/2} K W^{1/2}`` Cholesky stays PSD.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class LaplaceGPState:
    """Laplace-approximated GP posterior state, likelihood-agnostic.

    Attributes:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` training targets in the likelihood's support
            (``{-1, +1}`` for Bernoulli, non-negative integers for
            Poisson, reals for Student-t, ``(0, 1)`` for Beta, ...).
        f_mode: Posterior mode ``f̂`` of shape ``(n,)``.
        sqrt_w: Diagonal of ``W^{1/2}`` at ``f̂`` (shape ``(n,)``).
        cholesky_b: Lower-triangular Cholesky factor of
            ``B = I + W^{1/2} K W^{1/2}``, shape ``(n, n)``.
        gradient_log_likelihood_at_mode: ``∇ log p(y | f̂)``, cached
            for use in the predictive-mean computation.
        log_marginal_likelihood: Laplace-approximated log marginal
            (scalar).
        lengthscale: Kernel length-scale used at fit time.
        output_scale: Kernel output-scale used at fit time.
        kernel_fn: Kernel callable.
    """

    x_train: jax.Array
    y_train: jax.Array
    f_mode: jax.Array
    sqrt_w: jax.Array
    cholesky_b: jax.Array
    gradient_log_likelihood_at_mode: jax.Array
    log_marginal_likelihood: jax.Array
    lengthscale: float
    output_scale: float
    kernel_fn: Callable[..., jax.Array] = field(default=rbf_kernel)


def _newton_step(
    carry: tuple[jax.Array, jax.Array],
    _scan_dummy: jax.Array,
    *,
    k_train: jax.Array,
    y_train: jax.Array,
    n: int,
    components_fn: LikelihoodComponentsFn,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """One RW06 Algorithm 3.1 Newton iteration (likelihood-agnostic)."""
    f, _objective = carry
    _, grad, w_diag, sqrt_w = components_fn(f, y_train)
    b_matrix = jnp.eye(n) + sqrt_w[:, None] * k_train * sqrt_w[None, :]
    cholesky_b = jnp.linalg.cholesky(b_matrix)
    b_vec = w_diag * f + grad
    # a = b - W^{1/2} L^{-T} L^{-1} (W^{1/2} K b)
    intermediate = jax.scipy.linalg.cho_solve((cholesky_b, True), sqrt_w * (k_train @ b_vec))
    a = b_vec - sqrt_w * intermediate
    f_new = k_train @ a
    log_lik_new, _, _, _ = components_fn(f_new, y_train)
    objective_new = -0.5 * jnp.dot(a, f_new) + log_lik_new
    return (f_new, objective_new), objective_new


def fit_laplace_gp(
    *,
    log_likelihood_components_fn: LikelihoodComponentsFn,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    num_newton_iterations: int = 50,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
) -> LaplaceGPState:
    r"""Fit a Laplace-approximated GP for any factorising likelihood.

    Runs ``num_newton_iterations`` of RW06 Algorithm 3.1's Newton
    update (factorised through the ``B = I + W^{1/2} K W^{1/2}``
    Cholesky for PSD stability) and returns a :class:`LaplaceGPState`
    consumable by :func:`predict_laplace_latent_moments`.

    Args:
        log_likelihood_components_fn: Callable
            ``(f, y) -> (log_lik, ∇log_lik, W, √W)`` for the chosen
            likelihood. ``log_lik`` is a scalar; the rest are ``(n,)``
            arrays. ``W`` must be non-negative element-wise; each
            wrapper clips internally before returning.
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` training targets in the likelihood's support.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        num_newton_iterations: Fixed Newton-loop count. Static under
            ``jax.jit`` so the ``jax.lax.scan`` shape is known at
            trace time. Defaults to ``50``.
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.

    Returns:
        :class:`LaplaceGPState` carrying ``f_mode``, the Cholesky
        factor of ``B`` at the mode, and the Laplace-approximated log
        marginal likelihood (RW06 eq. 3.32).
    """
    n = y_train.shape[0]
    k_train = kernel_fn(x_train, x_train, lengthscale=lengthscale, output_scale=output_scale)
    initial_f = jnp.zeros((n,), dtype=jnp.float32)
    initial_objective = jnp.asarray(-jnp.inf, dtype=jnp.float32)
    (f_final, _), _ = jax.lax.scan(
        lambda carry, scan_dummy: _newton_step(
            carry,
            scan_dummy,
            k_train=k_train,
            y_train=y_train,
            n=n,
            components_fn=log_likelihood_components_fn,
        ),
        (initial_f, initial_objective),
        jnp.arange(num_newton_iterations),
    )
    log_lik_final, grad_final, _w_final, sqrt_w_final = log_likelihood_components_fn(
        f_final, y_train
    )
    b_matrix = jnp.eye(n) + sqrt_w_final[:, None] * k_train * sqrt_w_final[None, :]
    cholesky_b = jnp.linalg.cholesky(b_matrix)
    # RW06 eq. 3.32 marginal: -0.5 * a^T f - 0.5 * log |B| + log p(y | f̂).
    # At the mode ``a = ∇ log p(y | f̂)`` (the Newton fixed-point identity).
    log_marginal = (
        -0.5 * jnp.dot(grad_final, f_final) + log_lik_final - jnp.sum(jnp.log(jnp.diag(cholesky_b)))
    )
    return LaplaceGPState(
        x_train=x_train,
        y_train=y_train,
        f_mode=f_final,
        sqrt_w=sqrt_w_final,
        cholesky_b=cholesky_b,
        gradient_log_likelihood_at_mode=grad_final,
        log_marginal_likelihood=log_marginal,
        lengthscale=lengthscale,
        output_scale=output_scale,
        kernel_fn=kernel_fn,
    )


def predict_laplace_latent_moments(
    *,
    state: LaplaceGPState,
    x_test: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Compute latent Gaussian-posterior moments at ``x_test``.

    RW06 Algorithm 3.2:

    .. math::

        \bar{f}_*       &= k_*^{T}\,\nabla \log p(y \mid \hat{f}), \\
        \mathrm{Var}(f_*) &= k(x_*, x_*)
            - (W^{1/2} k_*)^{T}\,B^{-1}\,(W^{1/2} k_*).

    The per-likelihood wrappers feed these moments through the
    appropriate response-mean / response-variance map (MacKay probit
    for Bernoulli, ``exp(μ + V/2)`` for Poisson with log link, ...).

    Args:
        state: Fitted :class:`LaplaceGPState`.
        x_test: ``(m, d)`` test inputs.

    Returns:
        ``(latent_mean, latent_variance)`` each of shape ``(m,)``; the
        variance is clipped to ``≥ 1e-12`` for numerical safety.
    """
    k_cross = state.kernel_fn(
        state.x_train,
        x_test,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
    )  # shape (n, m)
    latent_mean = k_cross.T @ state.gradient_log_likelihood_at_mode
    v = jax.scipy.linalg.solve_triangular(
        state.cholesky_b, state.sqrt_w[:, None] * k_cross, lower=True
    )
    latent_variance = state.output_scale**2 - jnp.sum(v * v, axis=0)
    latent_variance = jnp.clip(latent_variance, a_min=1e-12)
    return latent_mean, latent_variance


__all__ = [
    "LaplaceGPState",
    "LikelihoodComponentsFn",
    "fit_laplace_gp",
    "predict_laplace_latent_moments",
]
