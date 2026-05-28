r"""Exact conjugate-Gaussian GP regression — Rasmussen & Williams 2006 Alg. 2.1.

For training inputs ``X``, observations ``y = f(X) + ε`` with
``ε ~ N(0, σ² I)``, and a zero-mean Gaussian process prior
``f ~ GP(0, k)``, the posterior at any test set ``X*`` is

.. math::

    \alpha &= (K + \sigma^{2} I)^{-1}\,y, \\
    \text{mean}(X^{*}) &= K(X^{*}, X)\,\alpha, \\
    \text{cov}(X^{*}) &= K(X^{*}, X^{*})
                       - K(X^{*}, X)\,(K + \sigma^{2} I)^{-1}\,K(X, X^{*}).

Algorithm 2.1 of Rasmussen & Williams 2006 (§2.2) computes both
quantities through a single Cholesky factorisation
``L = chol(K + σ² I)``:

.. code-block:: text

    α = L^T \ (L \ y)            # back-substitutions
    v = L \ K(X, X*)             # back-substitution
    mean = K(X*, X) α            # one matvec
    var  = K(X*, X*) - v^T v     # per-point reduction

The opifex implementation ports the algorithm directly to pure JAX so
the full pipeline (kernel + fit + predict) compiles under
``jax.jit``. The RBF / squared-exponential kernel is included as the
default; the same surface accepts any callable ``kernel_fn(x1, x2,
lengthscale, output_scale) -> jax.Array`` via the
``kernel_fn`` parameter.

Reference implementations consulted (READ-ONLY — never imported):

* ``../tinygp/src/tinygp/gp.py:GaussianProcess.{condition,predict}`` —
  identical Algorithm-2.1 Cholesky pattern, wrapped in ``eqx.Module``.
* GPJax ``gpjax/gps.py:ConjugatePosterior`` — same algebra.

References
----------
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; Algorithm 2.1 §2.2 (PRIMARY).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution


_EXACT_GP_SOURCE_PACKAGE = "opifex.uncertainty.gp"
_EXACT_GP_METHOD = DefaultStrategy.GAUSSIAN_PROCESS.value


def rbf_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    lengthscale: float,
    output_scale: float,
) -> jax.Array:
    r"""Squared-exponential (RBF) kernel.

    .. math::

        k(x, x') = \sigma_{f}^{2} \exp\!\left(-\tfrac{1}{2}\,
            \frac{\lVert x - x' \rVert_{2}^{2}}{\ell^{2}}\right).

    Args:
        x1: Inputs of shape ``(n, d)``.
        x2: Inputs of shape ``(m, d)``.
        lengthscale: ``ℓ`` (must be strictly positive).
        output_scale: ``σ_f`` (must be strictly positive).

    Returns:
        ``(n, m)`` kernel matrix.

    Raises:
        ValueError: If ``lengthscale`` or ``output_scale`` is non-positive.
    """
    if lengthscale <= 0.0:
        raise ValueError(f"lengthscale must be strictly positive; got {lengthscale!r}.")
    if output_scale <= 0.0:
        raise ValueError(f"output_scale must be strictly positive; got {output_scale!r}.")
    diff = x1[:, None, :] - x2[None, :, :]
    sq_distance = jnp.sum(diff**2, axis=-1)
    return (output_scale**2) * jnp.exp(-0.5 * sq_distance / (lengthscale**2))


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactGPState:
    """Fitted state for an exact conjugate-Gaussian GP.

    Attributes:
        x_train: Training inputs ``(n, d)``.
        y_train: Training targets ``(n,)``.
        cholesky: Lower-triangular Cholesky factor of ``K + σ² I``.
        alpha: ``α = (K + σ² I)^{-1} y`` — pre-solved coefficient
            vector used at predict time.
        lengthscale: Kernel length-scale used at fit time (forwarded
            into ``predict_exact_gp`` for kernel evaluations at test
            points).
        output_scale: Kernel output-scale used at fit time.
        noise_std: Observation noise scale ``σ`` used at fit time.
        kernel_fn: Callable kernel used during fit. Defaults to
            :func:`rbf_kernel`.
    """

    x_train: jax.Array
    y_train: jax.Array
    cholesky: jax.Array
    alpha: jax.Array
    lengthscale: float
    output_scale: float
    noise_std: float
    kernel_fn: Callable[..., jax.Array] = field(default=rbf_kernel)


def fit_exact_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    noise_std: float,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
) -> ExactGPState:
    r"""Fit an exact conjugate-Gaussian GP (RW06 Algorithm 2.1 prelude).

    Computes the Cholesky factor of ``K + σ² I`` and the pre-solved
    ``α = (K + σ² I)^{-1} y`` for downstream :func:`predict_exact_gp`
    calls.

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` training targets.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        noise_std: Observation noise scale ``σ`` (strictly positive —
            also acts as the jitter that keeps ``K + σ² I`` positive
            definite).
        kernel_fn: Optional kernel callable. Defaults to
            :func:`rbf_kernel`.

    Returns:
        An :class:`ExactGPState` carrying the Cholesky factor + ``α``.

    Raises:
        ValueError: If ``noise_std`` is non-positive.
    """
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be strictly positive; got {noise_std!r}.")
    k_train = kernel_fn(x_train, x_train, lengthscale=lengthscale, output_scale=output_scale)
    n = k_train.shape[0]
    gram = k_train + (noise_std**2) * jnp.eye(n)
    cholesky = jnp.linalg.cholesky(gram)
    # alpha = L^T \ (L \ y)  via cho_solve, equivalent to
    # ``jsp.linalg.cho_solve((L, True), y)``.
    alpha = jax.scipy.linalg.cho_solve((cholesky, True), y_train)
    return ExactGPState(
        x_train=x_train,
        y_train=y_train,
        cholesky=cholesky,
        alpha=alpha,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        kernel_fn=kernel_fn,
    )


def predict_exact_gp(
    *,
    state: ExactGPState,
    x_test: jax.Array,
) -> PredictiveDistribution:
    r"""Predict at ``x_test`` using a fitted :class:`ExactGPState`.

    Implements the Algorithm-2.1 *Cholesky-back-substitution* recipe:

    .. code-block:: text

        v    = L \ K(X, X*)
        mean = K(X*, X) @ α
        var  = K(X*, X*) - v^T v   (per test point)

    Args:
        state: Fitted :class:`ExactGPState`.
        x_test: ``(m, d)`` test inputs.

    Returns:
        A :class:`PredictiveDistribution` carrying the predictive
        ``mean`` ``(m,)`` and the marginal ``variance`` ``(m,)`` plus
        metadata advertising ``method=gaussian_process`` and the
        opifex source-package tag.
    """
    k_cross = state.kernel_fn(
        state.x_train,
        x_test,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
    )
    mean = k_cross.T @ state.alpha
    # ``v`` has shape ``(n, m)``; the per-test marginal variance is the
    # the column-wise dot product with itself.
    v = jax.scipy.linalg.solve_triangular(state.cholesky, k_cross, lower=True)
    k_diag = jnp.full((x_test.shape[0],), state.output_scale**2)
    variance = k_diag - jnp.sum(v * v, axis=0)
    return PredictiveDistribution(
        mean=mean,
        variance=variance,
        epistemic=variance,
        total_uncertainty=variance,
        metadata=compose_method_metadata(
            method=_EXACT_GP_METHOD,
            source_package=_EXACT_GP_SOURCE_PACKAGE,
            extra=(
                ("estimator", "exact_conjugate_gp"),
                ("paper", "Rasmussen & Williams 2006 Algorithm 2.1"),
            ),
        ),
    )


def fit_heteroscedastic_exact_gp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    lengthscale: float,
    output_scale: float,
    noise_std: jax.Array,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
) -> ExactGPState:
    r"""Fit an exact GP with per-observation Gaussian noise.

    Generalises :func:`fit_exact_gp` from a scalar noise level to a
    per-observation vector ``noise_std[i] = σ_i``. The Gram matrix
    becomes ``K + diag(σ_i²)``; the rest of the algorithm is
    unchanged so the returned :class:`ExactGPState` is consumable by
    the existing :func:`predict_exact_gp` driver. Setting every
    ``σ_i = σ`` reproduces :func:`fit_exact_gp` exactly (verified by
    the test suite).

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` training targets.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        noise_std: ``(n,)`` per-observation noise scales (every entry
            must be strictly positive).
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.

    Returns:
        :class:`ExactGPState` carrying the Cholesky factor + ``α`` and
        a ``noise_std`` field set to ``-1.0`` so callers can detect
        that the fitted state used the heteroscedastic path (the per-
        observation noises live inside the Cholesky factor by then).

    Raises:
        ValueError: If ``noise_std`` has a non-positive entry or a
            length not matching ``y_train``.
    """
    if noise_std.shape != y_train.shape:
        raise ValueError(
            f"noise_std must have shape matching y_train; got {noise_std.shape} vs {y_train.shape}."
        )
    # Value check (positivity of every entry) requires concrete values
    # that are unavailable under ``jax.jit``. ``jnp.linalg.cholesky``
    # below is the JAX-native PD check: a non-positive entry produces a
    # non-PD Gram and the Cholesky returns NaNs (surfacing downstream as
    # ``jnp.isfinite`` violations). We perform the eager Python-side
    # check only when the input is concrete; under jit we trust Cholesky.
    if not isinstance(noise_std, jax.core.Tracer) and float(jnp.min(noise_std)) <= 0.0:  # type: ignore[attr-defined]
        raise ValueError(
            "All noise_std entries must be strictly positive; got "
            f"min(noise_std)={float(jnp.min(noise_std))!r}."
        )
    k_train = kernel_fn(x_train, x_train, lengthscale=lengthscale, output_scale=output_scale)
    gram = k_train + jnp.diag(noise_std**2)
    cholesky = jnp.linalg.cholesky(gram)
    alpha = jax.scipy.linalg.cho_solve((cholesky, True), y_train)
    return ExactGPState(
        x_train=x_train,
        y_train=y_train,
        cholesky=cholesky,
        alpha=alpha,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=-1.0,  # sentinel: heteroscedastic path used; per-obs noise lives in cholesky
        kernel_fn=kernel_fn,
    )


def exact_gp_loocv_log_predictive(*, state: ExactGPState) -> jax.Array:
    r"""Leave-one-out predictive log-likelihood for a fitted exact GP.

    Rasmussen & Williams 2006 §5.4.2 derives the closed form

    .. math::

        \mu_{-i} &= y_{i} - \frac{\alpha_{i}}{[K^{-1}]_{ii}}, \\
        \sigma_{-i}^{2} &= \frac{1}{[K^{-1}]_{ii}}, \\
        \log p(y_{i} \mid X, y_{-i})
        &= -\tfrac{1}{2}\!\left[\log(2\pi)
            + \log \sigma_{-i}^{2}
            + \frac{(y_{i} - \mu_{-i})^{2}}{\sigma_{-i}^{2}}\right],

    so all ``n`` leave-one-out terms are recovered from a single
    Cholesky factorisation (no refitting required).

    Args:
        state: Fitted :class:`ExactGPState`.

    Returns:
        Scalar LOOCV log-predictive ``Σ_i log p(y_i | X, y_{-i})``.
    """
    # K_inv from the Cholesky factor: K_inv = cho_solve((L, True), I).
    n = state.alpha.shape[0]
    identity = jnp.eye(n)
    k_inv = jax.scipy.linalg.cho_solve((state.cholesky, True), identity)
    k_inv_diag = jnp.diag(k_inv)
    loo_variance = 1.0 / k_inv_diag
    loo_mean = state.y_train - state.alpha / k_inv_diag
    log_terms = -0.5 * (
        jnp.log(2.0 * jnp.pi)
        + jnp.log(loo_variance)
        + (state.y_train - loo_mean) ** 2 / loo_variance
    )
    return jnp.sum(log_terms)


__all__ = [
    "ExactGPState",
    "exact_gp_loocv_log_predictive",
    "fit_exact_gp",
    "fit_heteroscedastic_exact_gp",
    "predict_exact_gp",
    "rbf_kernel",
]
