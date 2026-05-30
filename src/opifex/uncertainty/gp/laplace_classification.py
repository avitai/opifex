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

The Newton step (RW06 eq. 3.18) lives in the likelihood-agnostic core
:mod:`opifex.uncertainty.gp.laplace`. This module supplies only the
Bernoulli ``(log_lik, ∇log_lik, W)`` triple and the **MacKay probit-
approximated** class probability (RW06 eq. 3.25):

.. math::

    p(y_* = +1 \mid x_*) \approx \sigma\!\left(
        \frac{\bar{f}_*}{\sqrt{1 + \pi\,\mathrm{Var}(f_*) / 8}}
    \right).

References
----------
* Williams, C. K. I., Barber, D. 1998 — *Bayesian Classification with
  Gaussian Processes*, IEEE TPAMI 20(12).
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes
  for Machine Learning*, MIT Press; §3.4 Algorithm 3.1 (PRIMARY).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.exact import rbf_kernel
from opifex.uncertainty.gp.laplace import (
    fit_laplace_gp,
    LaplaceGPState,
    predict_laplace_latent_moments,
)
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_LAPLACE_GP_SOURCE_PACKAGE = "opifex.uncertainty.gp"


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


def fit_bernoulli_laplace_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    num_newton_iterations: int = 50,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
) -> LaplaceGPState:
    r"""Fit a binary GP classifier via the Laplace approximation.

    Thin Bernoulli-specific wrapper around
    :func:`opifex.uncertainty.gp.laplace.fit_laplace_gp` that supplies
    the ``(log_lik, ∇log_lik, W = σ(f)(1 - σ(f)))`` triple from RW06
    §3.4.

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` targets in ``{-1, +1}``.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        num_newton_iterations: Fixed Newton-loop count. Defaults to
            ``50``. Static under ``jax.jit``.
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.

    Returns:
        :class:`LaplaceGPState` carrying ``f_mode``, the Cholesky
        factor of ``B`` at the mode, and the Laplace-approximated log
        marginal likelihood.
    """
    return fit_laplace_gp(
        log_likelihood_components_fn=_bernoulli_log_likelihood_components,
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        num_newton_iterations=num_newton_iterations,
        kernel_fn=kernel_fn,
    )


def predict_bernoulli_laplace_gp(
    *,
    state: LaplaceGPState,
    x_test: jax.Array,
) -> PredictiveDistribution:
    r"""Predict class probabilities at ``x_test`` via MacKay's approximation.

    RW06 Algorithm 3.2 gives the latent predictive moments

    .. math::

        \bar{f}_*       &= k_*^{T}\,\nabla \log p(y \mid \hat{f}), \\
        \mathrm{Var}(f_*) &= k(x_*, x_*)
            - (W^{1/2} k_*)^{T}\,B^{-1}\,(W^{1/2} k_*),

    obtained via :func:`predict_laplace_latent_moments`. The class
    probability uses **MacKay's probit approximation** (RW06 eq. 3.25):

    .. math::

        p(y_* = +1 \mid x_*) \approx \sigma\!\left(
            \frac{\bar{f}_*}{\sqrt{1 + \pi\,\mathrm{Var}(f_*) / 8}}
        \right).

    Args:
        state: Fitted :class:`LaplaceGPState`.
        x_test: ``(m, d)`` test inputs.

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` is
        ``p(y_*=+1 | x_*)`` (in ``[0, 1]``) and whose ``variance`` is
        the latent ``Var(f_*)`` (a useful epistemic-uncertainty
        proxy). Metadata records the source paper.
    """
    latent_mean, latent_variance = predict_laplace_latent_moments(state=state, x_test=x_test)
    # MacKay's probit approximation for the class probability.
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent_variance / 8.0)
    class_probability = jax.nn.sigmoid(kappa * latent_mean)
    return gaussian_process_predictive(
        class_probability,
        latent_variance,
        epistemic=latent_variance,
        total_uncertainty=latent_variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_LAPLACE_GP_SOURCE_PACKAGE,
            extra=(
                ("estimator", "bernoulli_laplace_gp"),
                ("paper", "Rasmussen & Williams 2006 §3.4 Alg. 3.1"),
                ("likelihood", "bernoulli"),
                ("link", "logit"),
            ),
        ),
    )


__all__ = [
    "fit_bernoulli_laplace_gp",
    "predict_bernoulli_laplace_gp",
]
