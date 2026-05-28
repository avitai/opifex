r"""Binary GP classification via the Laplace approximation (RW06 §3.4).

For a binary-Bernoulli likelihood ``p(y_i | f_i) = σ(y_i f_i)`` with
``y_i ∈ {-1, +1}`` and a zero-mean GP prior ``f ~ GP(0, K)``, the
posterior ``p(f | X, y)`` is non-Gaussian. The Laplace approximation
(Williams & Barber 1998; Rasmussen & Williams 2006, §3.4, Algorithm
3.1) replaces it with the Gaussian centred at the posterior mode ``f̂``
with covariance ``(K^{-1} + W)^{-1}`` where ``W = -∇² log p(y | f̂)``.

For the Bernoulli likelihood:

.. math::

    \log p(y \mid f)        &= -\log(1 + e^{-y f}), \\
    \nabla \log p(y \mid f) &= \tfrac{y + 1}{2} - \sigma(f)
                             = t - \sigma(f), \\
    W_{ii}                 &= \sigma(f_i)\,\bigl(1 - \sigma(f_i)\bigr).

The Newton step (RW06 eq. 3.18) is implemented through the
``B = I + W^{1/2} K W^{1/2}`` Cholesky to keep every intermediate
matrix PSD — this is the canonical numerically-stable form recommended
by RW06.

The predictive class probability ``p(y_* = +1 \mid x_*)`` is approximated
by **MacKay's probit-approximation** (RW06 eq. 3.25):

.. math::

    p(y_* = +1 \mid x_*) \approx \sigma\!\left(
        \frac{\bar{f}_*}{\sqrt{1 + \pi\,\mathrm{Var}(f_*) / 8}}
    \right).

Implementation notes — **jittability**
--------------------------------------

* Newton's loop runs through ``jax.lax.scan`` for a static
  ``num_newton_iterations``, so the whole fit compiles end-to-end
  under ``jax.jit``.
* ``predict`` uses ``cho_solve`` / ``solve_triangular`` on the
  ``B``-Cholesky and avoids any data-dependent control flow.
* **No equinox dependency**: the fitted state is a plain
  ``@dataclass(frozen=True, slots=True, kw_only=True)`` carrying
  ``jax.Array`` leaves. No tinygp / gpjax import.

Reference implementations consulted (READ-ONLY)
-----------------------------------------------

* ``../GPJax/examples/classification.py`` — illustrates the same
  Laplace-MAP flow; gpjax uses Adam-style optimisation rather than
  the closed-form Newton step. The Newton-based recipe used here
  follows RW06 Algorithm 3.1 directly.

References
----------
* Williams, C. K. I., Barber, D. 1998 — *Bayesian Classification with
  Gaussian Processes*, IEEE TPAMI 20(12).
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes
  for Machine Learning*, MIT Press; §3.4 Algorithm 3.1 (PRIMARY).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.exact import rbf_kernel
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution


_LAPLACE_GP_SOURCE_PACKAGE = "opifex.uncertainty.gp"


@dataclass(frozen=True, slots=True, kw_only=True)
class BernoulliLaplaceGPState:
    """Laplace-approximated posterior state for binary GP classification.

    Attributes:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` training targets in ``{-1, +1}``.
        f_mode: Posterior mode ``f̂`` of shape ``(n,)``.
        sqrt_w: Diagonal of ``W^{1/2}`` at ``f̂`` (shape ``(n,)``).
        cholesky_b: Lower-triangular Cholesky factor of
            ``B = I + W^{1/2} K W^{1/2}``, shape ``(n, n)``.
        gradient_log_likelihood_at_mode: ``∇ log p(y | f̂)``,
            cached for use in the predictive-mean computation.
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


def _bernoulli_log_likelihood_components(
    f: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Return ``(log_lik, ∇log_lik, W = -∇²log_lik, sqrt(W))`` for Bernoulli."""
    # ``jax.nn.log_sigmoid`` is numerically stable (no overflow / underflow).
    log_lik = jnp.sum(jax.nn.log_sigmoid(y * f))
    sigma_f = jax.nn.sigmoid(f)
    t = 0.5 * (y + 1.0)  # 1 when y=+1, 0 when y=-1
    grad = t - sigma_f
    w_diag = sigma_f * (1.0 - sigma_f)
    return log_lik, grad, w_diag, jnp.sqrt(w_diag)


def _newton_step(
    carry: tuple[jax.Array, jax.Array],
    _scan_dummy: jax.Array,
    *,
    k_train: jax.Array,
    y: jax.Array,
    n: int,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """One RW06 Algorithm 3.1 Newton iteration."""
    f, _objective = carry
    _, grad, w_diag, sqrt_w = _bernoulli_log_likelihood_components(f, y)
    b_matrix = jnp.eye(n) + sqrt_w[:, None] * k_train * sqrt_w[None, :]
    cholesky_b = jnp.linalg.cholesky(b_matrix)
    b_vec = w_diag * f + grad
    # a = b - W^{1/2} L^{-T} L^{-1} (W^{1/2} K b)
    intermediate = jax.scipy.linalg.cho_solve(
        (cholesky_b, True), sqrt_w * (k_train @ b_vec)
    )
    a = b_vec - sqrt_w * intermediate
    f_new = k_train @ a
    log_lik_new, _, _, _ = _bernoulli_log_likelihood_components(f_new, y)
    objective_new = -0.5 * jnp.dot(a, f_new) + log_lik_new
    return (f_new, objective_new), objective_new


def fit_bernoulli_laplace_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    num_newton_iterations: int = 50,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
) -> BernoulliLaplaceGPState:
    r"""Fit a binary GP classifier via the Laplace approximation.

    Runs ``num_newton_iterations`` of RW06 Algorithm 3.1's Newton
    update (factorised through the
    ``B = I + W^{1/2} K W^{1/2}`` Cholesky for PSD stability) and
    returns a :class:`BernoulliLaplaceGPState` that
    :func:`predict_bernoulli_laplace_gp` consumes.

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` targets in ``{-1, +1}``.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        num_newton_iterations: Fixed Newton-loop count. Defaults to
            ``50``. Static under ``jax.jit`` so the
            ``jax.lax.scan`` shape is known at trace time.
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.

    Returns:
        :class:`BernoulliLaplaceGPState` carrying ``f_mode``, the
        Cholesky factor of ``B`` at the mode, and the Laplace-
        approximated log marginal likelihood.
    """
    n = y_train.shape[0]
    k_train = kernel_fn(x_train, x_train, lengthscale=lengthscale, output_scale=output_scale)
    initial_f = jnp.zeros_like(y_train, dtype=jnp.float32)
    initial_objective = jnp.asarray(-jnp.inf, dtype=jnp.float32)
    (f_final, _), _ = jax.lax.scan(
        lambda carry, scan_dummy: _newton_step(
            carry, scan_dummy, k_train=k_train, y=y_train, n=n
        ),
        (initial_f, initial_objective),
        jnp.arange(num_newton_iterations),
    )
    log_lik_final, grad_final, _w_final, sqrt_w_final = (
        _bernoulli_log_likelihood_components(f_final, y_train)
    )
    b_matrix = jnp.eye(n) + sqrt_w_final[:, None] * k_train * sqrt_w_final[None, :]
    cholesky_b = jnp.linalg.cholesky(b_matrix)
    # RW06 eq. 3.32 marginal: -0.5 * a^T f - 0.5 * log |B| + log p(y|f̂).
    a_final = grad_final  # at the mode, ∇log p(y|f̂) equals the implicit a
    log_marginal = (
        -0.5 * jnp.dot(a_final, f_final)
        + log_lik_final
        - jnp.sum(jnp.log(jnp.diag(cholesky_b)))
    )
    return BernoulliLaplaceGPState(
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


def predict_bernoulli_laplace_gp(
    *,
    state: BernoulliLaplaceGPState,
    x_test: jax.Array,
) -> PredictiveDistribution:
    r"""Predict class probabilities at ``x_test`` via MacKay's approximation.

    RW06 Algorithm 3.2 gives the latent predictive moments

    .. math::

        \bar{f}_*       &= k_*^{T}\,\nabla \log p(y \mid \hat{f}), \\
        \mathrm{Var}(f_*) &= k(x_*, x_*)
            - (W^{1/2} k_*)^{T}\,B^{-1}\,(W^{1/2} k_*).

    The class probability uses **MacKay's probit approximation**
    (RW06 eq. 3.25):

    .. math::

        p(y_* = +1 \mid x_*) \approx \sigma\!\left(
            \frac{\bar{f}_*}{\sqrt{1 + \pi\,\mathrm{Var}(f_*) / 8}}
        \right).

    Args:
        state: Fitted :class:`BernoulliLaplaceGPState`.
        x_test: ``(m, d)`` test inputs.

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` is
        ``p(y_*=+1 | x_*)`` (in ``[0, 1]``) and whose ``variance`` is
        the latent ``Var(f_*)`` (a useful epistemic-uncertainty
        proxy). Metadata records the source paper.
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
    # MacKay's probit approximation for the class probability.
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent_variance / 8.0)
    class_probability = jax.nn.sigmoid(kappa * latent_mean)
    return PredictiveDistribution(
        mean=class_probability,
        variance=latent_variance,
        epistemic=latent_variance,
        total_uncertainty=latent_variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_LAPLACE_GP_SOURCE_PACKAGE,
            extra=(
                ("estimator", "bernoulli_laplace_gp"),
                ("paper", "Rasmussen & Williams 2006 §3.4 Alg. 3.1"),
            ),
        ),
    )


__all__ = [
    "BernoulliLaplaceGPState",
    "fit_bernoulli_laplace_gp",
    "predict_bernoulli_laplace_gp",
]
