r"""Non-conjugate Laplace likelihoods: Poisson, Student-t, Beta (Task 11.1 D5).

Per-likelihood wrappers on top of the generic Laplace core
:mod:`opifex.uncertainty.gp.laplace`. Each wrapper supplies a closed-
form ``(log_lik, ∇log_lik, W, √W)`` quadruple and a response-mean /
response-variance map for the predict path.

**Poisson** (count regression, ``exp`` link) — RW06 §3.4 Alg. 3.1 with
``log p(y|f) = y f - exp(f) - log y!``. Predict returns
``E[λ(x*)] = exp(μ_f + ½ Var f)`` (log-normal expectation under the
latent Gaussian) and ``Var[λ(x*)]`` from log-normal moments.

**Student-t** (robust regression, identity link, df ν, scale σ) — heavy-
tailed likelihood with closed-form ``(log_lik, grad, W)``. The
observed Hessian can be negative in the tails (``|y - f| > σ √ν``);
following the standard Laplace safeguard we clip ``W`` to be
non-negative so the ``B``-Cholesky stays PSD.

**Beta** (proportion regression, logit link, scale ``s``) — gpflow-
style reparameterisation ``α = s m``, ``β = s (1 - m)`` with
``m = σ(f)``. Newton uses the **Fisher information** ``W`` (always
non-negative) rather than the observed Hessian (which can be
indefinite). This matches bayesnewton's
``Beta(GeneralisedGaussNewtonMixin)`` convention.

Reference implementations consulted (READ-ONLY)
-----------------------------------------------

* ``../bayesnewton/bayesnewton/likelihoods.py:Poisson`` (line 891).
* ``../bayesnewton/bayesnewton/likelihoods.py:StudentsT`` (line 1011).
* ``../bayesnewton/bayesnewton/likelihoods.py:Beta`` (line 1047).

References
----------
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; §3.4 Algorithm 3.1 (PRIMARY).
* Wilkinson, W., Solin, A., Adam, V. 2020+ — *bayesnewton*; per-
  likelihood closed forms (cross-checked).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.exact import rbf_kernel
from opifex.uncertainty.gp.laplace import (
    fit_laplace_gp,
    LaplaceGPState,
    LikelihoodComponentsFn,
    predict_laplace_latent_moments,
)
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution


_LAPLACE_LIK_SOURCE_PACKAGE = "opifex.uncertainty.gp"
_LAPLACE_W_FLOOR = 1e-6  # clip for non-convex likelihood Hessians


# -----------------------------------------------------------------------------
# Poisson (exp link)
# -----------------------------------------------------------------------------


def _poisson_log_likelihood_components(
    f: jax.Array, y: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Return ``(log_lik, ∇log_lik, W, √W)`` for Poisson with ``exp`` link.

    For ``μ(f) = exp(f)``:

    .. math::

        \log p(y \mid f) &= y\,f - \mathrm{e}^{f} - \log\Gamma(y + 1), \\
        \nabla \log p(y \mid f) &= y - \mathrm{e}^{f}, \\
        W                       &= \mathrm{e}^{f} \geq 0.
    """
    rate = jnp.exp(f)
    log_lik = jnp.sum(y * f - rate - jax.scipy.special.gammaln(y + 1.0))
    grad = y - rate
    w_diag = rate
    return log_lik, grad, w_diag, jnp.sqrt(w_diag)


def fit_poisson_laplace_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    num_newton_iterations: int = 50,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
) -> LaplaceGPState:
    r"""Fit a Poisson GP regressor (count data, ``exp`` link).

    Thin wrapper around
    :func:`opifex.uncertainty.gp.laplace.fit_laplace_gp` that supplies
    the Poisson ``(log_lik, ∇log_lik, W = exp(f))`` triple.

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` non-negative count observations.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        num_newton_iterations: Fixed Newton-loop count. Defaults to
            ``50``. Static under ``jax.jit``.
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.

    Returns:
        :class:`LaplaceGPState` ready for
        :func:`predict_poisson_laplace_gp`.
    """
    return fit_laplace_gp(
        log_likelihood_components_fn=_poisson_log_likelihood_components,
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        num_newton_iterations=num_newton_iterations,
        kernel_fn=kernel_fn,
    )


def predict_poisson_laplace_gp(
    *,
    state: LaplaceGPState,
    x_test: jax.Array,
) -> PredictiveDistribution:
    r"""Predict Poisson intensity moments at ``x_test``.

    Under the latent Gaussian posterior ``f* ~ N(μ, V)`` and the
    ``exp`` link, the predicted rate ``λ* = exp(f*)`` is log-normal:

    .. math::

        \mathbb{E}[λ_*] &= \exp\!\left(\mu + \tfrac{1}{2} V\right), \\
        \mathrm{Var}[λ_*] &= \mathbb{E}[λ_*]^{2}\,
            \bigl(\mathrm{e}^{V} - 1\bigr).

    Args:
        state: Fitted :class:`LaplaceGPState`.
        x_test: ``(m, d)`` test inputs.

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` is the
        predicted Poisson intensity ``E[λ*]`` and whose ``variance``
        is the rate variance. ``epistemic`` carries the latent
        ``Var f_*``.
    """
    latent_mean, latent_variance = predict_laplace_latent_moments(state=state, x_test=x_test)
    intensity_mean = jnp.exp(latent_mean + 0.5 * latent_variance)
    intensity_variance = (intensity_mean**2) * (jnp.expm1(latent_variance))
    return PredictiveDistribution(
        mean=intensity_mean,
        variance=intensity_variance,
        epistemic=latent_variance,
        total_uncertainty=intensity_variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_LAPLACE_LIK_SOURCE_PACKAGE,
            extra=(
                ("estimator", "poisson_laplace_gp"),
                ("paper", "Rasmussen & Williams 2006 §3.4 Alg. 3.1"),
                ("likelihood", "poisson"),
                ("link", "exp"),
            ),
        ),
    )


# -----------------------------------------------------------------------------
# Student-t (identity link, df nu, scale sigma)
# -----------------------------------------------------------------------------


def _studentst_components_factory(*, df: float, scale: float) -> LikelihoodComponentsFn:
    r"""Build the Student-t ``(log_lik, ∇log_lik, W, √W)`` closure.

    For ``r = y - f``, ``u = 1 + r²/(ν σ²)``:

    .. math::

        \log p(y \mid f) &= C - \tfrac{1}{2}(\nu + 1)\,\log u, \\
        \nabla \log p(y \mid f) &= \frac{(\nu + 1)\,r}{\nu\,\sigma^{2}\,u}.

    The observed Hessian
    ``-∇² log p = (ν+1)(νσ² - r²)/(νσ² + r²)²`` can go negative when
    ``|r| > σ√ν``, breaking Newton stability. Following bayesnewton's
    ``StudentsT(GeneralisedGaussNewtonMixin)`` we use the **Fisher
    information** ``W = (ν+1) / ((ν+3) σ²)`` (a positive constant)
    instead — this gives a Gauss-Newton update that is stable for
    arbitrary residuals and matches the canonical scaled-F-distribution
    derivation of the Fisher info for Student-t.
    """
    df_arr = jnp.asarray(df)
    scale_sq = jnp.asarray(scale) ** 2
    df_times_scale_sq = df_arr * scale_sq
    log_const = (
        jax.scipy.special.gammaln(0.5 * (df_arr + 1.0))
        - jax.scipy.special.gammaln(0.5 * df_arr)
        - 0.5 * (jnp.log(scale_sq) + jnp.log(df_arr) + jnp.log(jnp.pi))
    )
    fisher_info = (df_arr + 1.0) / ((df_arr + 3.0) * scale_sq)

    def _components(
        f: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        residual = y - f
        residual_sq = residual * residual
        u_form = 1.0 + residual_sq / df_times_scale_sq
        log_lik = jnp.sum(log_const - 0.5 * (df_arr + 1.0) * jnp.log(u_form))
        grad = (df_arr + 1.0) * residual / (df_times_scale_sq * u_form)
        w_diag = jnp.full_like(f, fisher_info)
        return log_lik, grad, w_diag, jnp.sqrt(w_diag)

    return _components


def fit_studentst_laplace_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    df: float = 4.0,
    scale: float = 1.0,
    num_newton_iterations: int = 50,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
) -> LaplaceGPState:
    r"""Fit a Student-t robust GP regressor (identity link).

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` real-valued observations.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        df: Student-t degrees of freedom ``ν > 0``. Smaller values
            (e.g. ``ν = 3 - 5``) yield heavier tails and more
            outlier robustness. Defaults to ``4.0``.
        scale: Student-t scale ``σ > 0``. Defaults to ``1.0``.
        num_newton_iterations: Fixed Newton-loop count. Defaults to
            ``50``. Static under ``jax.jit``.
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.

    Returns:
        :class:`LaplaceGPState` ready for
        :func:`predict_studentst_laplace_gp`.
    """
    return fit_laplace_gp(
        log_likelihood_components_fn=_studentst_components_factory(df=df, scale=scale),
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        num_newton_iterations=num_newton_iterations,
        kernel_fn=kernel_fn,
    )


def predict_studentst_laplace_gp(
    *,
    state: LaplaceGPState,
    x_test: jax.Array,
    df: float = 4.0,
    scale: float = 1.0,
) -> PredictiveDistribution:
    r"""Predict Student-t response moments at ``x_test``.

    Under the latent Gaussian ``f* ~ N(μ, V)`` and a Student-t
    likelihood with location ``f*`` and scale ``σ``, the conditional
    mean is ``f*`` (for ``ν > 1``) and the conditional variance is
    ``σ² ν / (ν - 2)`` (for ``ν > 2``). By the law of total
    variance:

    .. math::

        \mathbb{E}[y_*]   &= \mu, \\
        \mathrm{Var}[y_*] &= V + \frac{\sigma^{2}\,\nu}{\nu - 2}.

    Args:
        state: Fitted :class:`LaplaceGPState`.
        x_test: ``(m, d)`` test inputs.
        df: Student-t degrees of freedom used at fit time
            (``ν > 2`` required for finite predictive variance).
            Defaults to ``4.0``.
        scale: Student-t scale used at fit time. Defaults to ``1.0``.

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` is the
        predictive ``E[y*]`` and whose ``variance`` is the total
        predictive variance. ``epistemic`` carries the latent
        ``Var f_*``.
    """
    latent_mean, latent_variance = predict_laplace_latent_moments(state=state, x_test=x_test)
    df_arr = jnp.asarray(df)
    scale_sq = jnp.asarray(scale) ** 2
    response_variance = latent_variance + scale_sq * df_arr / (df_arr - 2.0)
    return PredictiveDistribution(
        mean=latent_mean,
        variance=response_variance,
        epistemic=latent_variance,
        total_uncertainty=response_variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_LAPLACE_LIK_SOURCE_PACKAGE,
            extra=(
                ("estimator", "studentst_laplace_gp"),
                ("paper", "Rasmussen & Williams 2006 §3.4 Alg. 3.1"),
                ("likelihood", "students_t"),
                ("link", "identity"),
            ),
        ),
    )


# -----------------------------------------------------------------------------
# Beta (logit link, scale s)
# -----------------------------------------------------------------------------


def _beta_components_factory(*, scale: float) -> LikelihoodComponentsFn:
    r"""Build the Beta ``(log_lik, ∇log_lik, W, √W)`` closure (logit link).

    With ``m(f) = σ(f)``, ``α = s m``, ``β = s (1 - m)`` (so
    ``α + β = s`` is constant):

    .. math::

        \log p(y \mid f) &= (α - 1)\log y + (β - 1)\log(1 - y) \\
                         &\quad + \log Γ(s) - \log Γ(α) - \log Γ(β), \\
        \nabla \log p(y \mid f) &= s\,m\,(1 - m)\,\bigl[
            \log\tfrac{y}{1 - y} - \psi(α) + \psi(β)
        \bigr], \\
        W &\approx s^{2}\,\bigl(m(1-m)\bigr)^{2}\,\bigl[
            \psi'(α) + \psi'(β)
        \bigr]   \quad \text{(Fisher info, always } \geq 0\text{).}

    The Fisher information (used here in place of the observed
    Hessian) matches bayesnewton's
    ``Beta(GeneralisedGaussNewtonMixin)`` convention and keeps the
    ``B``-Cholesky in the Newton step strictly PSD.
    """
    scale_arr = jnp.asarray(scale)

    def _components(
        f: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        mean = jax.nn.sigmoid(f)
        alpha = mean * scale_arr
        beta = scale_arr - alpha
        # Numerical safety for y at the unit-interval boundaries.
        y_clipped = jnp.clip(y, a_min=1e-6, a_max=1.0 - 1e-6)
        log_y = jnp.log(y_clipped)
        log_one_minus_y = jnp.log(1.0 - y_clipped)
        log_lik = jnp.sum(
            (alpha - 1.0) * log_y
            + (beta - 1.0) * log_one_minus_y
            + jax.scipy.special.gammaln(scale_arr)
            - jax.scipy.special.gammaln(alpha)
            - jax.scipy.special.gammaln(beta)
        )
        sigmoid_derivative = mean * (1.0 - mean)
        digamma_alpha = jax.scipy.special.digamma(alpha)
        digamma_beta = jax.scipy.special.digamma(beta)
        bracket = log_y - log_one_minus_y - digamma_alpha + digamma_beta
        grad = scale_arr * sigmoid_derivative * bracket
        polygamma_alpha = jax.scipy.special.polygamma(1, alpha)
        polygamma_beta = jax.scipy.special.polygamma(1, beta)
        w_diag = (scale_arr**2) * (sigmoid_derivative**2) * (polygamma_alpha + polygamma_beta)
        w_diag_clipped = jnp.clip(w_diag, a_min=_LAPLACE_W_FLOOR)
        return log_lik, grad, w_diag_clipped, jnp.sqrt(w_diag_clipped)

    return _components


def fit_beta_laplace_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    scale: float = 10.0,
    num_newton_iterations: int = 50,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
) -> LaplaceGPState:
    r"""Fit a Beta GP regressor for proportion data (logit link).

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` observations in ``(0, 1)``.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        scale: Beta precision ``s = α + β > 0`` (gpflow / bayesnewton
            convention). Larger ``s`` ⇒ narrower per-point Beta.
            Defaults to ``10.0``.
        num_newton_iterations: Fixed Newton-loop count. Defaults to
            ``50``. Static under ``jax.jit``.
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.

    Returns:
        :class:`LaplaceGPState` ready for
        :func:`predict_beta_laplace_gp`.
    """
    return fit_laplace_gp(
        log_likelihood_components_fn=_beta_components_factory(scale=scale),
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        num_newton_iterations=num_newton_iterations,
        kernel_fn=kernel_fn,
    )


def predict_beta_laplace_gp(
    *,
    state: LaplaceGPState,
    x_test: jax.Array,
    scale: float = 10.0,
) -> PredictiveDistribution:
    r"""Predict Beta response moments at ``x_test`` (logit link).

    Approximates ``E[σ(f*)]`` by **MacKay's probit-style sigmoid
    approximation**:

    .. math::

        \mathbb{E}[σ(f_*)] \approx σ\!\left(
            \frac{\mu}{\sqrt{1 + π\,V / 8}}
        \right),

    then returns the implied Beta variance
    ``Var[y*] = m̂ (1 - m̂) / (s + 1)`` at that mean.

    Args:
        state: Fitted :class:`LaplaceGPState`.
        x_test: ``(m, d)`` test inputs.
        scale: Beta precision used at fit time. Defaults to ``10.0``.

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` is
        ``E[y* | x*] ∈ (0, 1)`` and whose ``variance`` is the implied
        Beta variance. ``epistemic`` carries the latent ``Var f_*``.
    """
    latent_mean, latent_variance = predict_laplace_latent_moments(state=state, x_test=x_test)
    kappa = 1.0 / jnp.sqrt(1.0 + jnp.pi * latent_variance / 8.0)
    response_mean = jax.nn.sigmoid(kappa * latent_mean)
    response_variance = response_mean * (1.0 - response_mean) / (scale + 1.0)
    return PredictiveDistribution(
        mean=response_mean,
        variance=response_variance,
        epistemic=latent_variance,
        total_uncertainty=response_variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_LAPLACE_LIK_SOURCE_PACKAGE,
            extra=(
                ("estimator", "beta_laplace_gp"),
                ("paper", "Rasmussen & Williams 2006 §3.4 Alg. 3.1"),
                ("likelihood", "beta"),
                ("link", "logit"),
            ),
        ),
    )


__all__ = [
    "fit_beta_laplace_gp",
    "fit_poisson_laplace_gp",
    "fit_studentst_laplace_gp",
    "predict_beta_laplace_gp",
    "predict_poisson_laplace_gp",
    "predict_studentst_laplace_gp",
]
