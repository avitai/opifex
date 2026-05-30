r"""Titsias-collapsed sparse variational Gaussian process (SVGP).

For a Gaussian likelihood with fixed kernel hyperparameters, Titsias
2009 (*Variational Learning of Inducing Variables in Sparse Gaussian
Processes*, AISTATS) derives a **closed-form** optimal variational
posterior ``q*(u) = N(μ*, S*)`` over the inducing values
``u = f(Z)`` at ``M ≪ N`` inducing inputs ``Z``. The opifex
implementation fits this in ``O(n m² + m³)`` time through the standard
``A / B / L_B`` Cholesky factorisation (GPJax
``CollapsedVariationalGaussian`` /
``objectives.collapsed_elbo`` reference).

Key identities (Titsias 2009 / RW06 §8.4):

    A         = L_z^{-1} K_zx / σ,
    B         = I + A A^T,
    L_B       = chol(B),
    μ(x*)     = (L_z^{-1} K_zt)^T B^{-1} (A y / σ),
    Var(x*)   = K(x*, x*)
                - ||L_z^{-1} K_zt||²
                + ||L_B^{-1} L_z^{-1} K_zt||²,
    log N(y; 0, σ² I + Q)
              = -n/2 log(2πσ²)
                - ½ log|B|
                - 1/(2σ²) (||y||² - ||L_B^{-1} A y||²),
    collapsed ELBO
              = log N(y; 0, σ²I + Q)
                - 1/(2σ²) [tr(K_xx)_diag - tr(A A^T)].

Implementation notes
--------------------

* **Jittability**: all operations route through
  ``jnp.linalg.cholesky`` /
  ``jax.scipy.linalg.cho_solve`` /
  ``solve_triangular``; the full fit + predict pipeline compiles
  end-to-end under ``jax.jit``.
* **No equinox dependency**: the fitted state is a plain
  ``@dataclass(frozen=True, slots=True, kw_only=True)`` carrying
  ``jax.Array`` leaves. No tinygp / gpjax import.
* **Reference-checked correctness**: the algebra mirrors GPJax's
  ``collapsed_elbo`` and ``CollapsedVariationalGaussian.predict``
  exactly (zero-mean prior; same ``A / B / L_B`` factorisation).
  GPJax is the reference implementation, not a runtime dependency.

References
----------
* Titsias, M. K. 2009 — *Variational Learning of Inducing Variables in
  Sparse Gaussian Processes*, AISTATS (PRIMARY).
* Quinonero-Candela, J., Rasmussen, C. E. 2005 — *A unifying view of
  sparse approximate Gaussian process regression*, JMLR (SOR/DTC
  predictive equivalence).
* Hensman, J., Fusi, N., Lawrence, N. D. 2013 — *Gaussian Processes
  for Big Data*, UAI (stochastic-optimisation extension for
  non-conjugate likelihoods, deferred to a follow-up slice).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from opifex.uncertainty._predictive import gaussian_process_predictive
from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.exact import rbf_kernel
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


_SVGP_SOURCE_PACKAGE = "opifex.uncertainty.gp"


def _kernel_diagonal(
    kernel_fn: Callable[..., jax.Array],
    x: jax.Array,
    *,
    lengthscale: float,
    output_scale: float,
) -> jax.Array:
    r"""Return per-point ``K(x_i, x_i)`` of shape ``(n,)`` via ``jax.vmap``.

    For stationary single-output kernels this equals ``output_scale^2``
    at every point (and the vmap collapses to a trivial repeat). For
    **multi-output ICM/LCM kernels** (slices 2-3) the diagonal varies
    per point — ``k_base(x, x) * B[task, task]`` for ICM, the
    component sum for LCM — so the per-point evaluation is required
    for the ``K_{xx}``-diagonal term in the Titsias collapsed ELBO
    and the predictive variance.

    Args:
        kernel_fn: Any callable matching the standard kernel signature
            ``(x1, x2, *, lengthscale, output_scale) -> Gram``.
        x: Inputs of shape ``(n, d)``.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.

    Returns:
        ``(n,)`` array of diagonal entries.
    """
    return jax.vmap(
        lambda xi: kernel_fn(
            xi[None], xi[None], lengthscale=lengthscale, output_scale=output_scale
        )[0, 0]
    )(x)


@dataclass(frozen=True, slots=True, kw_only=True)
class SVGPState:
    """Fitted state for the Titsias-collapsed sparse-variational GP.

    Attributes:
        x_inducing: ``(m, d)`` inducing inputs ``Z``.
        cholesky_kmm: Lower-triangular ``L_z = chol(K_zz + jitter I)``.
        cholesky_b: Lower-triangular ``L_B = chol(I + A A^T)`` where
            ``A = L_z^{-1} K_zx / σ``.
        scaled_alpha: ``L_B^{-1} (A y / σ)`` of shape ``(m,)`` — the
            pre-solved coefficient vector used at predict time.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        noise_std: Observation noise scale ``σ``.
        jitter: Numerical jitter added to ``K_zz`` for PD stability.
        kernel_fn: Kernel callable.
        cached_y_squared_norm: ``||y||²`` cached for the collapsed-ELBO
            quadratic term.
        cached_a_y_inside_norm: ``||L_B^{-1} A y / σ||²`` cached
            for the collapsed-ELBO quadratic term.
        cached_trace_aat: ``tr(A A^T)`` cached for the
            collapsed-ELBO trace term.
        cached_kxx_diag_sum: ``Σ_i K(x_i, x_i)`` cached for the
            collapsed-ELBO trace term.
        cached_log_det_b: ``log|B| = 2 Σ_i log L_B[i, i]`` cached for
            the collapsed-ELBO log-det term.
        cached_n: Training-set size ``n`` (static int).
    """

    x_inducing: jax.Array
    cholesky_kmm: jax.Array
    cholesky_b: jax.Array
    scaled_alpha: jax.Array
    lengthscale: float
    output_scale: float
    noise_std: float
    jitter: float
    kernel_fn: Callable[..., jax.Array] = field(default=rbf_kernel)
    cached_y_squared_norm: jax.Array = field(default_factory=lambda: jnp.asarray(0.0))
    cached_a_y_inside_norm: jax.Array = field(default_factory=lambda: jnp.asarray(0.0))
    cached_trace_aat: jax.Array = field(default_factory=lambda: jnp.asarray(0.0))
    cached_kxx_diag_sum: jax.Array = field(default_factory=lambda: jnp.asarray(0.0))
    cached_log_det_b: jax.Array = field(default_factory=lambda: jnp.asarray(0.0))
    cached_n: int = 0


def fit_svgp(
    *,
    x_train: jax.Array,
    y_train: jax.Array,
    x_inducing: jax.Array,
    lengthscale: float,
    output_scale: float,
    noise_std: float,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
    jitter: float = 1e-6,
) -> SVGPState:
    r"""Fit the Titsias-collapsed sparse-variational GP.

    Args:
        x_train: ``(n, d)`` training inputs.
        y_train: ``(n,)`` training targets.
        x_inducing: ``(m, d)`` inducing inputs ``Z``. Choose ``m << n``
            for the scalability benefit; ``m == n`` recovers the
            exact GP up to numerical jitter.
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        noise_std: Observation noise scale ``σ`` (strictly positive).
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.
        jitter: Numerical jitter added to ``K_zz`` for PD stability.

    Returns:
        :class:`SVGPState` carrying the ``L_z`` / ``L_B`` Cholesky
        factors + pre-solved coefficient vector + cached ELBO terms.

    Raises:
        ValueError: If ``noise_std`` is not strictly positive.
    """
    if noise_std <= 0.0:
        raise ValueError(f"noise_std must be strictly positive; got {noise_std!r}.")
    n = int(x_train.shape[0])
    m = int(x_inducing.shape[0])
    sigma = noise_std

    k_zz = kernel_fn(x_inducing, x_inducing, lengthscale=lengthscale, output_scale=output_scale)
    k_zz = k_zz + jitter * jnp.eye(m)
    cholesky_kmm = jnp.linalg.cholesky(k_zz)

    k_zx = kernel_fn(x_inducing, x_train, lengthscale=lengthscale, output_scale=output_scale)
    # A = L_z^{-1} K_zx / sigma, shape (m, n).
    a_matrix = jax.scipy.linalg.solve_triangular(cholesky_kmm, k_zx, lower=True) / sigma
    # B = I + A A^T, shape (m, m).
    b_matrix = jnp.eye(m) + a_matrix @ a_matrix.T
    cholesky_b = jnp.linalg.cholesky(b_matrix)
    # scaled_alpha = L_B^{-1} (A y / sigma), shape (m,).
    scaled_alpha = jax.scipy.linalg.solve_triangular(
        cholesky_b, a_matrix @ y_train / sigma, lower=True
    )

    cached_y_squared_norm = jnp.dot(y_train, y_train)
    cached_a_y_inside_norm = jnp.dot(scaled_alpha, scaled_alpha)
    cached_trace_aat = jnp.sum(a_matrix * a_matrix)
    # diag K_xx via per-point vmap of kernel_fn. For stationary single-output
    # kernels this equals output_scale^2 everywhere; for multi-output ICM/LCM
    # kernels each diagonal entry is k_base(x_i, x_i) * B[task_i, task_i] (or
    # the LCM sum thereof), which varies per point. The vmap form handles
    # both transparently.
    kxx_diag = _kernel_diagonal(
        kernel_fn, x_train, lengthscale=lengthscale, output_scale=output_scale
    )
    cached_kxx_diag_sum = jnp.sum(kxx_diag)
    cached_log_det_b = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky_b)))
    return SVGPState(
        x_inducing=x_inducing,
        cholesky_kmm=cholesky_kmm,
        cholesky_b=cholesky_b,
        scaled_alpha=scaled_alpha,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
        jitter=jitter,
        kernel_fn=kernel_fn,
        cached_y_squared_norm=cached_y_squared_norm,
        cached_a_y_inside_norm=cached_a_y_inside_norm,
        cached_trace_aat=cached_trace_aat,
        cached_kxx_diag_sum=cached_kxx_diag_sum,
        cached_log_det_b=cached_log_det_b,
        cached_n=n,
    )


def predict_svgp(*, state: SVGPState, x_test: jax.Array) -> PredictiveDistribution:
    r"""Predictive distribution at ``x_test`` (Titsias 2009 SOR/DTC form).

    Args:
        state: Fitted :class:`SVGPState`.
        x_test: ``(t, d)`` test inputs.

    Returns:
        :class:`PredictiveDistribution` with ``mean`` ``(t,)``,
        ``variance`` ``(t,)``, ``epistemic == variance``, and metadata
        advertising ``estimator=titsias_collapsed_svgp``.
    """
    k_zt = state.kernel_fn(
        state.x_inducing,
        x_test,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
    )
    # v_test = L_z^{-1} K_zt, shape (m, t).
    v_test = jax.scipy.linalg.solve_triangular(state.cholesky_kmm, k_zt, lower=True)
    # mean(x*) = v_test^T @ L_B^{-T} L_B^{-1} (A y / sigma)
    # scaled_alpha is already L_B^{-1}(Ay/sigma), so we need v_test^T L_B^{-T} scaled_alpha,
    # so we need v_test^T @ L_B^{-T} @ scaled_alpha. Equivalent: (L_B^{-T} scaled_alpha)^T v_test.
    inner = jax.scipy.linalg.solve_triangular(
        state.cholesky_b, state.scaled_alpha, lower=True, trans="T"
    )
    mean = v_test.T @ inner
    # Var(x*) = K(x*, x*) - ||v_test||² + ||L_B^{-1} v_test||²
    kxx_diag = _kernel_diagonal(
        state.kernel_fn,
        x_test,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
    )
    l_b_inv_v = jax.scipy.linalg.solve_triangular(state.cholesky_b, v_test, lower=True)
    variance = kxx_diag - jnp.sum(v_test * v_test, axis=0) + jnp.sum(l_b_inv_v * l_b_inv_v, axis=0)
    return gaussian_process_predictive(
        mean,
        variance,
        epistemic=variance,
        total_uncertainty=variance,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_SVGP_SOURCE_PACKAGE,
            extra=(
                ("estimator", "titsias_collapsed_svgp"),
                ("paper", "Titsias 2009 (variational inducing variables)"),
            ),
        ),
    )


def svgp_collapsed_elbo(*, state: SVGPState) -> jax.Array:
    r"""Collapsed ELBO for a fitted :class:`SVGPState` (scalar, ``jit``-safe).

    Computes Titsias 2009 eq. 9 / GPJax ``collapsed_elbo`` exactly via
    the cached ``A``-Cholesky factorisation. Useful as the objective
    for hyperparameter optimisation when wrapped under ``jax.grad``.
    """
    n = state.cached_n
    sigma_sq = state.noise_std**2
    log_prob = -0.5 * (
        n * jnp.log(2.0 * jnp.pi * sigma_sq)
        + state.cached_log_det_b
        + (state.cached_y_squared_norm - state.cached_a_y_inside_norm) / sigma_sq
    )
    trace_term = (state.cached_kxx_diag_sum - state.cached_trace_aat * sigma_sq) / (2.0 * sigma_sq)
    # Note: cached_trace_aat = tr(A A^T) where A = L_z^{-1} K_zx / sigma, so
    # tr(K_xz K_zz^{-1} K_zx) = sigma^2 tr(A A^T) and tr(K_xx - Q) / (2 sigma^2)
    # = (sum K_diag - sigma^2 tr(A A^T)) / (2 sigma^2).
    return log_prob - trace_term


__all__ = [
    "SVGPState",
    "fit_svgp",
    "predict_svgp",
    "svgp_collapsed_elbo",
]
