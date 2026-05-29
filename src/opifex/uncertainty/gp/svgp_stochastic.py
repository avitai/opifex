r"""Stochastic SVGP for non-conjugate likelihoods — Task 11.1 D3.

Hensman, Fusi, Lawrence 2013 (UAI — *Gaussian Processes for Big Data*)
introduced the **stochastic uncollapsed SVGP** that fits any
factorising likelihood ``p(y_i | f_i)`` via minibatched gradient
optimisation of a variational ELBO. Hensman, Matthews, Ghahramani 2015
(AISTATS — *Scalable Variational Gaussian Process Classification*)
specialised the construction with the **whitened variational
parametrisation**, which is the form opifex ships.

Mathematical contract
---------------------

The variational family is

.. math::

    q(\mathbf{u}) = \mathcal{N}(\mathbf{L}_{z}\,\boldsymbol{\mu}_{w},\,
                                \mathbf{L}_{z}\,\mathbf{L}_{w}\,
                                \mathbf{L}_{w}^{\!\top}\,\mathbf{L}_{z}^{\!\top}),

with ``L_z = chol(K_zz + jitter I)`` and learnable whitened parameters
``μ_w ∈ R^m`` and lower-triangular ``L_w ∈ R^{m×m}``. The KL term has
a closed-form independent of ``K_zz``:

.. math::

    \operatorname{KL}\!\left[q(\mathbf{u})\;\big\|\;p(\mathbf{u})\right]
        = \frac{1}{2}\bigl(
              \lVert\boldsymbol{\mu}_{w}\rVert^{2}
              + \lVert\mathbf{L}_{w}\rVert_{F}^{2}
              - m
              - 2\,\textstyle\sum_{i=1}^{m}\log[\mathbf{L}_{w}]_{ii}
          \bigr).

Per-batch latent-marginal moments (``A = L_z^{-1}\,K_{zb}``):

.. math::

    \text{mean}_{b} &= \mathbf{A}^{\!\top}\,\boldsymbol{\mu}_{w}, \\
    \text{var}_{b}  &= \text{diag}(\mathbf{K}_{bb})
                      - \lVert\mathbf{A}\rVert_{\text{col}}^{2}
                      + \lVert\mathbf{L}_{w}^{\!\top}\mathbf{A}\rVert_{\text{col}}^{2}.

For non-Gaussian ``p(y_i | f_i)`` we evaluate

.. math::

    \mathbb{E}_{q}[\log p(y_i | f_i)]
        \approx \frac{1}{\sqrt{\pi}}\,
          \sum_{k=1}^{Q} w_k\,\log p\!\left(y_i \;\big|\;
            m_i + \sqrt{2\,V_i}\,\xi_k\right)

via Gauss-Hermite quadrature with ``Q`` static at trace time.

The minibatched ELBO is

.. math::

    \mathcal{L}(\boldsymbol{\mu}_{w}, \mathbf{L}_{w})
        = \frac{N}{|B|}\,\sum_{i \in B}
              \mathbb{E}_{q}[\log p(y_i | f_i)]
          - \operatorname{KL}.

opifex efficiency wins over GPJax
---------------------------------

GPJax's ``VariationalGaussian.prior_kl`` + ``predict`` each perform
their own ``chol(K_zz)``; the per-point predictive ``vmap`` in
``objectives.variational_expectation`` re-references ``K_zz`` and
``L_z`` inside every traced inner call (relying on XLA CSE for
deduplication). opifex computes ``L_z`` **once** per ELBO call and
threads it through both KL and per-batch predictive moments via
closed-form triangular solves; the whitened parametrisation removes
``K_zz`` from the KL entirely.

Reference implementations consulted (READ-ONLY)
-----------------------------------------------

* ``../GPJax/gpjax/variational_families.py:155-308`` (unwhitened
  ``VariationalGaussian``) — verified the unwhitened predictive
  formula matched by the whitened ↔ unwhitened equivalence test.
* ``../GPJax/gpjax/objectives.py:280-405`` (uncollapsed ELBO).

References
----------
* Hensman, J., Fusi, N., Lawrence, N. D. 2013 — *Gaussian Processes
  for Big Data*, UAI (PRIMARY).
* Hensman, J., Matthews, A., Ghahramani, Z. 2015 — *Scalable
  Variational Gaussian Process Classification*, AISTATS (whitened
  parametrisation).
* Salimbeni, H., Eleftheriadis, S., Hensman, J. 2018 — *Natural
  Gradients in Practice*, AISTATS (deferred natural-gradient form).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from opifex.uncertainty.adapters.base import compose_method_metadata
from opifex.uncertainty.gp.exact import rbf_kernel
from opifex.uncertainty.registry import DefaultStrategy
from opifex.uncertainty.types import PredictiveDistribution


_STOCHASTIC_SVGP_SOURCE_PACKAGE = "opifex.uncertainty.gp"


LogLikelihoodFn = Callable[[jax.Array, jax.Array], jax.Array]
"""Type alias: ``(f, y) -> log p(y_i | f_i)`` evaluated element-wise.

Both inputs and the returned array share the same shape ``(n,)`` (or any
broadcastable shape). The stochastic-SVGP ELBO calls this function on
Gauss-Hermite-quadrature samples of ``f``.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class StochasticSVGPState:
    """Whitened-parametrisation stochastic SVGP state.

    Attributes:
        x_inducing: ``(m, d)`` inducing inputs ``Z``.
        whitened_mean: ``(m,)`` ``μ_w`` — variational mean in the
            whitened basis (``μ = L_z μ_w`` in the unwhitened basis).
        whitened_root_cov: ``(m, m)`` lower-triangular ``L_w`` such
            that the whitened variational covariance is
            ``L_w L_w^T`` (unwhitened: ``L_z L_w L_w^T L_z^T``).
        lengthscale: Kernel length-scale.
        output_scale: Kernel output-scale.
        kernel_fn: Kernel callable.
        log_likelihood_fn: Per-observation ``log p(y_i | f_i)``.
        jitter: Numerical jitter on ``K_zz`` for PD stability.
    """

    x_inducing: jax.Array
    whitened_mean: jax.Array
    whitened_root_cov: jax.Array
    lengthscale: float
    output_scale: float
    kernel_fn: Callable[..., jax.Array] = field(default=rbf_kernel)
    log_likelihood_fn: LogLikelihoodFn = field(default=lambda f, y: -((y - f) ** 2))
    jitter: float = 1e-6


def init_stochastic_svgp_state(
    *,
    x_inducing: jax.Array,
    lengthscale: float,
    output_scale: float,
    log_likelihood_fn: LogLikelihoodFn,
    kernel_fn: Callable[..., jax.Array] = rbf_kernel,
    jitter: float = 1e-6,
) -> StochasticSVGPState:
    r"""Initialise ``q(u) = N(0, K_zz)`` (``μ_w = 0``, ``L_w = I``).

    Args:
        x_inducing: ``(m, d)`` inducing inputs.
        lengthscale: Kernel length-scale (strictly positive).
        output_scale: Kernel output-scale (strictly positive).
        log_likelihood_fn: Per-observation log-likelihood callable
            ``(f, y) -> log p(y | f)`` evaluated element-wise. Use
            :func:`bernoulli_log_likelihood` /
            :func:`poisson_log_likelihood` for the canonical cases or
            close over additional hyperparameters via a factory.
        kernel_fn: Kernel callable. Defaults to :func:`rbf_kernel`.
        jitter: Numerical jitter for ``K_zz`` factorisation.

    Returns:
        :class:`StochasticSVGPState` ready for gradient-based ELBO
        optimisation.

    Raises:
        ValueError: If ``lengthscale`` or ``output_scale`` is non-
            positive or ``x_inducing`` is not 2-D.
    """
    if lengthscale <= 0.0:
        raise ValueError(f"lengthscale must be strictly positive; got {lengthscale!r}.")
    if output_scale <= 0.0:
        raise ValueError(f"output_scale must be strictly positive; got {output_scale!r}.")
    if x_inducing.ndim != 2:
        raise ValueError(f"x_inducing must have shape (m, d); got {x_inducing.shape}.")
    num_inducing = x_inducing.shape[0]
    return StochasticSVGPState(
        x_inducing=x_inducing,
        whitened_mean=jnp.zeros((num_inducing,)),
        whitened_root_cov=jnp.eye(num_inducing),
        lengthscale=lengthscale,
        output_scale=output_scale,
        kernel_fn=kernel_fn,
        log_likelihood_fn=log_likelihood_fn,
        jitter=jitter,
    )


def _gauss_hermite_nodes_weights(
    num_points: int,
) -> tuple[jax.Array, jax.Array]:
    """Static Gauss-Hermite quadrature ``(nodes, weights)``.

    For ``∫ exp(-x²) g(x) dx ≈ Σ_k weights[k] g(nodes[k])``. Build via
    numpy at module-trace time so JIT sees concrete constants.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(num_points)
    return jnp.asarray(nodes), jnp.asarray(weights)


def _factorise_kzz(
    state: StochasticSVGPState,
) -> jax.Array:
    """Return ``L_z = chol(K_zz + jitter I)`` — a single call per ELBO/predict."""
    num_inducing = state.x_inducing.shape[0]
    k_zz = state.kernel_fn(
        state.x_inducing,
        state.x_inducing,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
    ) + state.jitter * jnp.eye(num_inducing)
    return jnp.linalg.cholesky(k_zz)


def _latent_marginal_moments(
    *,
    state: StochasticSVGPState,
    cholesky_kzz: jax.Array,
    x_batch: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Per-batch ``(mean_b, var_b)`` of the latent marginal ``q(f_b)``.

    With ``A = L_z^{-1} K_{zb}``:

    * ``mean_b = A^T μ_w``                              shape ``(b,)``
    * ``var_b  = diag(K_{bb}) - ‖A‖_{col}^2
                                + ‖L_w^T A‖_{col}^2``   shape ``(b,)``

    Stationary kernels have ``diag(K_{bb}) = output_scale²``.
    """
    k_zb = state.kernel_fn(
        state.x_inducing,
        x_batch,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
    )
    a_matrix = jax.scipy.linalg.solve_triangular(cholesky_kzz, k_zb, lower=True)
    mean_batch = a_matrix.T @ state.whitened_mean
    lw_a = state.whitened_root_cov.T @ a_matrix
    diag_k_bb = jnp.full((x_batch.shape[0],), state.output_scale**2)
    var_batch = diag_k_bb - jnp.sum(a_matrix * a_matrix, axis=0) + jnp.sum(lw_a * lw_a, axis=0)
    return mean_batch, jnp.clip(var_batch, a_min=1e-12)


def _whitened_kl_divergence(
    *,
    whitened_mean: jax.Array,
    whitened_root_cov: jax.Array,
) -> jax.Array:
    r"""Closed-form ``KL[N(μ_w, L_w L_w^T) || N(0, I)]`` (Hensman+ 2015).

    .. math::

        \operatorname{KL} = \tfrac{1}{2}\!\bigl(
            \lVert\mu_w\rVert^2 + \lVert L_w\rVert_F^2 - m
            - 2\,\textstyle\sum_i \log[L_w]_{ii}
        \bigr).

    Zero ``K_zz`` operations — the whitened-parametrisation payoff.
    """
    num_inducing = whitened_mean.shape[0]
    log_det = jnp.sum(jnp.log(jnp.abs(jnp.diag(whitened_root_cov))))
    return 0.5 * (
        jnp.sum(whitened_mean**2) + jnp.sum(whitened_root_cov**2) - num_inducing - 2.0 * log_det
    )


def _expected_log_likelihood_per_batch(
    *,
    log_likelihood_fn: LogLikelihoodFn,
    mean_batch: jax.Array,
    var_batch: jax.Array,
    y_batch: jax.Array,
    num_quadrature_points: int,
) -> jax.Array:
    r"""Compute ``E_{q(f_i)}[log p(y_i | f_i)]`` via Gauss-Hermite quadrature.

    Per-observation expectation under ``f_i ~ N(m_i, V_i)``:
    ``E[log p] ≈ Σ_k (w_k / √π) log p(y_i | m_i + √(2 V_i) ξ_k)``.

    Returns a ``(b,)`` array.
    """
    nodes, weights = _gauss_hermite_nodes_weights(num_quadrature_points)
    sqrt_two_var = jnp.sqrt(2.0 * var_batch)
    # f_samples shape (Q, b); broadcast y_batch over Q.
    f_samples = mean_batch[None, :] + sqrt_two_var[None, :] * nodes[:, None]
    y_broadcast = jnp.broadcast_to(y_batch[None, :], f_samples.shape)
    log_lik = log_likelihood_fn(f_samples, y_broadcast)  # (Q, b)
    weighted = weights[:, None] * log_lik
    return jnp.sum(weighted, axis=0) / jnp.sqrt(jnp.pi)


def stochastic_svgp_elbo(
    *,
    state: StochasticSVGPState,
    x_batch: jax.Array,
    y_batch: jax.Array,
    dataset_size: int,
    num_quadrature_points: int = 20,
) -> jax.Array:
    r"""Hensman+ 2013 minibatched ELBO (scalar, ``jit``-safe).

    ``ELBO = (N / |B|) Σ_{i ∈ B} E_q[log p(y_i | f_i)] - KL[q(u) || p(u)]``
    with the whitened ``KL`` and a Gauss-Hermite-quadrature data-fit.
    Exactly **one** ``chol(K_zz)`` per call.

    Args:
        state: Variational state to be differentiated through.
        x_batch: ``(b, d)`` minibatch inputs.
        y_batch: ``(b,)`` minibatch targets.
        dataset_size: ``N`` — full-dataset size. The data-fit term is
            scaled by ``N / b`` so the ELBO is an unbiased estimate
            of the full-data ELBO.
        num_quadrature_points: ``Q`` — Gauss-Hermite quadrature nodes
            (static; defaults to ``20``). Static under ``jax.jit``.

    Returns:
        Scalar ELBO array.
    """
    cholesky_kzz = _factorise_kzz(state)
    mean_batch, var_batch = _latent_marginal_moments(
        state=state, cholesky_kzz=cholesky_kzz, x_batch=x_batch
    )
    expected_log_lik_per_obs = _expected_log_likelihood_per_batch(
        log_likelihood_fn=state.log_likelihood_fn,
        mean_batch=mean_batch,
        var_batch=var_batch,
        y_batch=y_batch,
        num_quadrature_points=num_quadrature_points,
    )
    data_fit = jnp.sum(expected_log_lik_per_obs) * (dataset_size / x_batch.shape[0])
    kl_term = _whitened_kl_divergence(
        whitened_mean=state.whitened_mean,
        whitened_root_cov=state.whitened_root_cov,
    )
    return data_fit - kl_term


def predict_stochastic_svgp(
    *,
    state: StochasticSVGPState,
    x_test: jax.Array,
) -> PredictiveDistribution:
    r"""Closed-form latent predictive ``q(f(x_*))`` (Hensman+ 2015 eq. 8-9).

    Reuses :func:`_latent_marginal_moments` so the predictive moments
    are computed in ``O(m^2 t)`` after a single ``chol(K_zz)``.

    Args:
        state: Trained :class:`StochasticSVGPState`.
        x_test: ``(t, d)`` test inputs.

    Returns:
        :class:`PredictiveDistribution` whose ``mean`` and
        ``variance`` carry the latent marginal moments. Map through
        the appropriate response link (MacKay probit for classification,
        ``exp`` for Poisson, identity for regression) downstream.
    """
    cholesky_kzz = _factorise_kzz(state)
    mean_test, var_test = _latent_marginal_moments(
        state=state, cholesky_kzz=cholesky_kzz, x_batch=x_test
    )
    return PredictiveDistribution(
        mean=mean_test,
        variance=var_test,
        epistemic=var_test,
        total_uncertainty=var_test,
        metadata=compose_method_metadata(
            method=DefaultStrategy.GAUSSIAN_PROCESS.value,
            source_package=_STOCHASTIC_SVGP_SOURCE_PACKAGE,
            extra=(
                ("estimator", "stochastic_svgp"),
                ("paper", "Hensman+ 2013/2015 (UAI/AISTATS)"),
                ("parametrisation", "whitened"),
            ),
        ),
    )


# -----------------------------------------------------------------------------
# Built-in per-observation log-likelihood helpers
# -----------------------------------------------------------------------------


def bernoulli_log_likelihood(f: jax.Array, y: jax.Array) -> jax.Array:
    r"""``log p(y_i | f_i) = log σ(y_i f_i)`` for ``y ∈ {-1, +1}`` (RW06 §3.4).

    Uses :func:`jax.nn.log_sigmoid` for overflow / underflow safety.
    """
    return jax.nn.log_sigmoid(y * f)


def poisson_log_likelihood(f: jax.Array, y: jax.Array) -> jax.Array:
    r"""``log p(y_i | f_i) = y_i f_i - exp(f_i) - log Γ(y_i + 1)`` (``exp`` link).

    Matches the Poisson likelihood used by D5
    (:func:`opifex.uncertainty.gp.fit_poisson_laplace_gp`).
    """
    return y * f - jnp.exp(f) - jax.scipy.special.gammaln(y + 1.0)


# -----------------------------------------------------------------------------
# Natural-gradient updates (Salimbeni+ 2018) — D3 sub-item (b)
# -----------------------------------------------------------------------------


def natural_gradient_step(
    *,
    state: StochasticSVGPState,
    x_batch: jax.Array,
    y_batch: jax.Array,
    dataset_size: int,
    learning_rate: float,
    num_quadrature_points: int = 20,
) -> StochasticSVGPState:
    r"""One Salimbeni+ 2018 natural-gradient update of ``(μ_w, L_w)``.

    For the whitened variational Gaussian ``q(u_w) = N(μ_w, S_w)`` with
    ``S_w = L_w L_w^{T}``, the natural parameters are

    .. math::

        \theta_{1} = S_{w}^{-1}\,\mu_{w},
        \qquad \theta_{2} = -\tfrac{1}{2}\,S_{w}^{-1},

    and the expectation parameters are
    ``η_1 = μ_w``, ``η_2 = S_w + μ_w μ_w^{T}``. Salimbeni,
    Eleftheriadis, Hensman 2018 (AISTATS — *Natural Gradients in
    Practice*) shows that the natural-gradient step on ``θ`` equals
    the regular gradient on ``η``:

    .. math::

        \theta_{1}^{\text{new}} &= \theta_{1} + \rho\,\bigl(
            \partial \mathcal{L}/\partial \mu_{w}
            - 2\,(\partial \mathcal{L}/\partial S_{w})\,\mu_{w}\bigr),\\
        \theta_{2}^{\text{new}} &= \theta_{2} + \rho\,
            \partial \mathcal{L}/\partial S_{w}.

    After the update we recover ``(μ_w, L_w)`` via
    ``S_w = (-2 \theta_{2})^{-1}``, ``μ_w = S_w \theta_{1}``,
    ``L_w = chol(S_w)``. The natural-gradient direction is the
    Fisher-information-preconditioned direction; Salimbeni+ 2018 §4
    reports 10x-100x faster convergence than Adam on the variational
    parameters for non-Gaussian likelihoods.

    Performance: each step costs one regular ``jax.grad`` pass plus
    three ``O(m^{3})`` linear-algebra operations
    (``S_w`` inverse, ``S_w_new`` recovery, Cholesky). For ``m`` up
    to a few hundred, this is dominated by the gradient pass.

    Args:
        state: Current variational state.
        x_batch: ``(b, d)`` minibatch inputs.
        y_batch: ``(b,)`` minibatch targets.
        dataset_size: Full-dataset size ``N`` for the ELBO scaling.
        learning_rate: Natural-gradient step size ``ρ``. For the
            Gaussian-likelihood + conjugate case, ``ρ = 1`` reaches
            the variational optimum in a single step
            (Salimbeni+ 2018 Algorithm 1). For non-Gaussian
            likelihoods use ``ρ ≈ 0.1 - 1.0`` and decay over
            iterations.
        num_quadrature_points: Gauss-Hermite quadrature nodes for
            the ELBO data-fit term. Defaults to ``20``.

    Returns:
        :class:`StochasticSVGPState` with updated ``(μ_w, L_w)``;
        every other field is unchanged.
    """
    num_inducing = state.whitened_mean.shape[0]
    jitter_identity = state.jitter * jnp.eye(num_inducing)

    initial_s_w = state.whitened_root_cov @ state.whitened_root_cov.T

    def elbo_in_mu_and_s(mu_w_arg: jax.Array, s_w_arg: jax.Array) -> jax.Array:
        l_w_arg = jnp.linalg.cholesky(s_w_arg + jitter_identity)
        wrapped_state = StochasticSVGPState(
            x_inducing=state.x_inducing,
            whitened_mean=mu_w_arg,
            whitened_root_cov=l_w_arg,
            lengthscale=state.lengthscale,
            output_scale=state.output_scale,
            kernel_fn=state.kernel_fn,
            log_likelihood_fn=state.log_likelihood_fn,
            jitter=state.jitter,
        )
        return stochastic_svgp_elbo(
            state=wrapped_state,
            x_batch=x_batch,
            y_batch=y_batch,
            dataset_size=dataset_size,
            num_quadrature_points=num_quadrature_points,
        )

    grad_mu, grad_s = jax.grad(elbo_in_mu_and_s, argnums=(0, 1))(state.whitened_mean, initial_s_w)
    # Symmetrise the gradient w.r.t. S (the elbo depends on S only
    # through L_w = chol(S), so the autodiff gradient w.r.t. S is the
    # standard symmetric form modulo numerical noise — symmetrise to
    # ensure exact symmetry).
    grad_s_sym = 0.5 * (grad_s + grad_s.T)

    # Expectation-parameter gradient (Salimbeni+ 2018 eq. 7-8):
    # ∂L/∂η_1 = ∂L/∂μ - 2 (∂L/∂S) μ
    # ∂L/∂η_2 = ∂L/∂S
    grad_eta_1 = grad_mu - 2.0 * grad_s_sym @ state.whitened_mean
    grad_eta_2 = grad_s_sym

    # Current natural parameters in the whitened space.
    initial_s_inv = jnp.linalg.inv(initial_s_w + jitter_identity)
    initial_s_inv_sym = 0.5 * (initial_s_inv + initial_s_inv.T)
    theta_1 = initial_s_inv_sym @ state.whitened_mean
    theta_2 = -0.5 * initial_s_inv_sym

    # Natural-gradient step.
    theta_1_new = theta_1 + learning_rate * grad_eta_1
    theta_2_new = theta_2 + learning_rate * grad_eta_2
    theta_2_new_sym = 0.5 * (theta_2_new + theta_2_new.T)

    # Recover (μ_w, S_w, L_w) from the updated natural parameters.
    new_s_inv = -2.0 * theta_2_new_sym
    new_s_inv_sym = 0.5 * (new_s_inv + new_s_inv.T)
    new_s_w = jnp.linalg.inv(new_s_inv_sym + jitter_identity)
    new_s_w_sym = 0.5 * (new_s_w + new_s_w.T)
    new_mu_w = new_s_w_sym @ theta_1_new
    new_l_w = jnp.linalg.cholesky(new_s_w_sym + jitter_identity)

    return StochasticSVGPState(
        x_inducing=state.x_inducing,
        whitened_mean=new_mu_w,
        whitened_root_cov=new_l_w,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
        kernel_fn=state.kernel_fn,
        log_likelihood_fn=state.log_likelihood_fn,
        jitter=state.jitter,
    )


__all__ = [
    "LogLikelihoodFn",
    "StochasticSVGPState",
    "bernoulli_log_likelihood",
    "init_stochastic_svgp_state",
    "natural_gradient_step",
    "poisson_log_likelihood",
    "predict_stochastic_svgp",
    "stochastic_svgp_elbo",
]
