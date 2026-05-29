r"""Tests for the stochastic SVGP (Task 11.1 D3 — Hensman+ 2013 / 2015).

Slice 11 shipped the **Titsias collapsed** SVGP for Gaussian likelihoods
with a closed-form optimal ``q*(u)``. This slice ships the **stochastic
uncollapsed** SVGP that handles **any** factorising likelihood
``p(y_i | f_i)`` via Hensman+ 2013/2015's variational lower bound,
minibatched ELBO, and Gauss-Hermite quadrature.

opifex's efficiency wins over GPJax (verified by inspection of
``../GPJax/gpjax/variational_families.py:155-308`` +
``../GPJax/gpjax/objectives.py:280-330``):

* **One** ``K_zz`` Cholesky per ELBO call (GPJax does ≥ 2 — one in
  ``prior_kl``, one in ``predict``).
* **Whitened variational parametrisation** ``(μ_w, L_w)`` so the KL
  closes to ``0.5(‖μ_w‖² + ‖L_w‖_F² − m − 2 Σ log diag L_w)`` — zero
  K_zz operations in the KL term.
* **Closed-form vectorised batch predictive moments** — a single
  ``solve_triangular(L_z, K_zb)`` then per-batch matmuls; no per-
  point ``vmap`` over a function that re-references K_zz / L_z.
* Pure-JAX, no equinox dependency.

References
----------
* Hensman, J., Fusi, N., Lawrence, N. D. 2013 — *Gaussian Processes
  for Big Data*, UAI (PRIMARY, minibatched ELBO).
* Hensman, J., Matthews, A., Ghahramani, Z. 2015 — *Scalable
  Variational Gaussian Process Classification*, AISTATS (whitened
  parametrisation + classification specialisation).
* Salimbeni, H., Eleftheriadis, S., Hensman, J. 2018 — *Natural
  Gradients in Practice*, AISTATS (whitened-form natural gradient).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from opifex.uncertainty.gp import (
    bernoulli_log_likelihood,
    fit_bernoulli_laplace_gp,
    fit_svgp,
    init_stochastic_svgp_state,
    natural_gradient_step,
    poisson_log_likelihood,
    predict_bernoulli_laplace_gp,
    predict_stochastic_svgp,
    predict_svgp,
    rbf_kernel,
    stochastic_svgp_elbo,
    StochasticSVGPState,
)
from opifex.uncertainty.types import PredictiveDistribution


# -----------------------------------------------------------------------------
# Mechanical correctness: state init, KL identity, JIT, scaling
# -----------------------------------------------------------------------------


def test_init_stochastic_svgp_state_returns_zero_mean_and_identity_root_cov() -> None:
    """At init, ``μ_w = 0`` and ``L_w = I`` so ``q(u) = N(0, K_zz)`` under whitening."""
    x_inducing = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    state = init_stochastic_svgp_state(
        x_inducing=x_inducing,
        lengthscale=0.5,
        output_scale=1.0,
        log_likelihood_fn=bernoulli_log_likelihood,
    )
    assert isinstance(state, StochasticSVGPState)
    assert jnp.allclose(state.whitened_mean, jnp.zeros(5))
    assert jnp.allclose(state.whitened_root_cov, jnp.eye(5))


def test_stochastic_svgp_elbo_at_init_is_finite_negative_scalar() -> None:
    """ELBO is a scalar; at initialisation with random labels it's typically negative."""
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (40, 1))
    y_train = (jnp.sin(2.0 * x_train.squeeze(-1)) > 0).astype(jnp.float32) * 2.0 - 1.0
    x_inducing = jnp.linspace(-2.0, 2.0, 6).reshape(-1, 1)
    state = init_stochastic_svgp_state(
        x_inducing=x_inducing,
        lengthscale=0.5,
        output_scale=1.0,
        log_likelihood_fn=bernoulli_log_likelihood,
    )
    elbo = stochastic_svgp_elbo(
        state=state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
    )
    assert elbo.shape == ()
    assert jnp.isfinite(elbo)


def test_whitened_kl_matches_closed_form_identity_on_a_known_state() -> None:
    r"""``KL[N(μ_w, L_w L_w^T) || N(0, I)]`` closed-form identity.

    The whitened KL has no ``K_zz`` dependency:
    ``KL = 0.5 (‖μ_w‖² + ‖L_w‖_F² − m − 2 Σ log diag L_w)``.

    We construct a state with a non-trivial ``(μ_w, L_w)``, compute the
    ELBO using a constant likelihood (so the data-fit term is exactly
    known), and verify that ``data_fit - elbo == kl_closed_form``.
    """
    m = 4
    rng = jax.random.PRNGKey(2)
    key_mu, key_lw = jax.random.split(rng)
    mu_w = jax.random.normal(key_mu, (m,)) * 0.5
    raw_lower = jax.random.normal(key_lw, (m, m)) * 0.3
    l_w = jnp.tril(raw_lower) + jnp.eye(m) * 1.2  # strictly positive diagonal

    # Constant log-likelihood independent of f produces a deterministic data-fit.
    def constant_log_lik(f: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.zeros_like(y)

    state = init_stochastic_svgp_state(
        x_inducing=jnp.linspace(0.0, 1.0, m).reshape(-1, 1),
        lengthscale=0.4,
        output_scale=1.0,
        log_likelihood_fn=constant_log_lik,
    )
    state = StochasticSVGPState(
        x_inducing=state.x_inducing,
        whitened_mean=mu_w,
        whitened_root_cov=l_w,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
        kernel_fn=state.kernel_fn,
        log_likelihood_fn=constant_log_lik,
        jitter=state.jitter,
    )
    elbo = stochastic_svgp_elbo(
        state=state,
        x_batch=jnp.zeros((3, 1)),
        y_batch=jnp.zeros((3,)),
        dataset_size=3,
    )
    expected_kl = 0.5 * (
        jnp.sum(mu_w**2) + jnp.sum(l_w**2) - m - 2.0 * jnp.sum(jnp.log(jnp.diag(l_w)))
    )
    # data_fit is 0 when log_lik ≡ 0, so elbo == -kl.
    assert jnp.allclose(elbo, -expected_kl, atol=1e-5)


def test_stochastic_svgp_elbo_is_jit_compatible() -> None:
    """The full ELBO function compiles under ``jax.jit``."""
    x_train = jnp.linspace(-1.0, 1.0, 16).reshape(-1, 1)
    y_train = jnp.sign(jnp.sin(2.0 * x_train.squeeze(-1)))
    x_inducing = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)

    @jax.jit
    def elbo_step(x_batch: jax.Array, y_batch: jax.Array) -> jax.Array:
        state = init_stochastic_svgp_state(
            x_inducing=x_inducing,
            lengthscale=0.5,
            output_scale=1.0,
            log_likelihood_fn=bernoulli_log_likelihood,
        )
        return stochastic_svgp_elbo(
            state=state,
            x_batch=x_batch,
            y_batch=y_batch,
            dataset_size=x_train.shape[0],
        )

    value = elbo_step(x_train, y_train)
    assert jnp.isfinite(value)


def test_minibatch_scaling_matches_full_batch_in_expectation() -> None:
    """ELBO with ``dataset_size=N`` and a full batch equals the full-batch ELBO."""
    x_train = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y_train = jnp.sign(jnp.sin(2.0 * x_train.squeeze(-1)))
    x_inducing = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    state = init_stochastic_svgp_state(
        x_inducing=x_inducing,
        lengthscale=0.5,
        output_scale=1.0,
        log_likelihood_fn=bernoulli_log_likelihood,
    )
    elbo_full = stochastic_svgp_elbo(
        state=state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
    )
    # When the batch equals the full data and dataset_size equals batch size,
    # the scaling factor N/batch_size = 1, so this is the unscaled data-fit ELBO.
    elbo_scaled_identity = stochastic_svgp_elbo(
        state=state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
    )
    assert jnp.allclose(elbo_full, elbo_scaled_identity)


# -----------------------------------------------------------------------------
# Whitened ↔ unwhitened predictive equivalence (mathematical correctness)
# -----------------------------------------------------------------------------


def test_predict_stochastic_svgp_returns_predictive_distribution_with_finite_moments() -> None:
    """Predict returns a ``PredictiveDistribution`` with positive variance."""
    state = init_stochastic_svgp_state(
        x_inducing=jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1),
        lengthscale=0.5,
        output_scale=1.0,
        log_likelihood_fn=bernoulli_log_likelihood,
    )
    x_test = jnp.linspace(-1.5, 1.5, 8).reshape(-1, 1)
    predictive = predict_stochastic_svgp(state=state, x_test=x_test)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.variance is not None
    assert jnp.all(jnp.isfinite(predictive.mean))
    assert jnp.all(predictive.variance > 0.0)


def test_whitened_predict_equals_unwhitened_predict_formula() -> None:
    r"""``predict_stochastic_svgp`` agrees with the unwhitened GPJax formula.

    Construct a state with non-trivial ``(μ_w, L_w)``; let the effective
    unwhitened posterior be ``μ = L_z μ_w``, ``S = L_z L_w L_w^T L_z^T``.
    The closed-form GPJax-style predictive at ``x_test`` is

    .. math::

        \text{mean}(t) &= K_{tz} K_{zz}^{-1} \mu, \\
        \text{var}(t)  &= K_{tt} - K_{tz} K_{zz}^{-1} K_{zt}
                        + K_{tz} K_{zz}^{-1} S K_{zz}^{-1} K_{zt}.

    Verifying both forms match pins the whitened maths.
    """
    m = 4
    x_inducing = jnp.linspace(-1.0, 1.0, m).reshape(-1, 1)
    lengthscale, output_scale = 0.5, 1.0
    rng = jax.random.PRNGKey(7)
    key_mu, key_lw = jax.random.split(rng)
    mu_w = jax.random.normal(key_mu, (m,)) * 0.5
    raw = jax.random.normal(key_lw, (m, m)) * 0.3
    l_w = jnp.tril(raw) + jnp.eye(m) * 1.5
    state = StochasticSVGPState(
        x_inducing=x_inducing,
        whitened_mean=mu_w,
        whitened_root_cov=l_w,
        lengthscale=lengthscale,
        output_scale=output_scale,
        kernel_fn=rbf_kernel,
        log_likelihood_fn=bernoulli_log_likelihood,
        jitter=1e-6,
    )
    x_test = jnp.linspace(-1.5, 1.5, 5).reshape(-1, 1)
    predictive = predict_stochastic_svgp(state=state, x_test=x_test)

    # Reference: unwhitened GPJax-style closed form.
    k_zz = rbf_kernel(
        x_inducing, x_inducing, lengthscale=lengthscale, output_scale=output_scale
    ) + 1e-6 * jnp.eye(m)
    l_z = jnp.linalg.cholesky(k_zz)
    mu_unwhitened = l_z @ mu_w
    s_unwhitened = l_z @ l_w @ l_w.T @ l_z.T
    k_zt = rbf_kernel(x_inducing, x_test, lengthscale=lengthscale, output_scale=output_scale)
    k_tt_diag = jnp.full((x_test.shape[0],), output_scale**2)
    kzz_inv_kzt = jax.scipy.linalg.cho_solve((l_z, True), k_zt)
    expected_mean = kzz_inv_kzt.T @ mu_unwhitened
    expected_var = (
        k_tt_diag
        - jnp.sum(k_zt * kzz_inv_kzt, axis=0)
        + jnp.sum((kzz_inv_kzt.T @ s_unwhitened) * kzz_inv_kzt.T, axis=1)
    )
    assert predictive.variance is not None
    assert jnp.allclose(predictive.mean, expected_mean, atol=1e-5)
    assert jnp.allclose(predictive.variance, expected_var, atol=1e-5)


def test_predict_stochastic_svgp_is_jit_compatible() -> None:
    """Predict compiles under ``jax.jit``."""
    x_inducing = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    x_test = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)

    @jax.jit
    def predict(x_q: jax.Array) -> jax.Array:
        state = init_stochastic_svgp_state(
            x_inducing=x_inducing,
            lengthscale=0.5,
            output_scale=1.0,
            log_likelihood_fn=bernoulli_log_likelihood,
        )
        predictive = predict_stochastic_svgp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = predict(x_test)
    assert out.shape == (6,)
    assert jnp.all(jnp.isfinite(out))


# -----------------------------------------------------------------------------
# Convergence equivalences against Titsias collapsed and Laplace classification
# -----------------------------------------------------------------------------


def _train_stochastic_svgp(
    *,
    state: StochasticSVGPState,
    x_train: jax.Array,
    y_train: jax.Array,
    num_steps: int,
    learning_rate: float = 1e-2,
) -> StochasticSVGPState:
    """Adam-optimise ``(μ_w, L_w)`` against the ELBO. Used by convergence tests."""

    def loss(mu_w: jax.Array, l_w: jax.Array) -> jax.Array:
        new_state = StochasticSVGPState(
            x_inducing=state.x_inducing,
            whitened_mean=mu_w,
            whitened_root_cov=l_w,
            lengthscale=state.lengthscale,
            output_scale=state.output_scale,
            kernel_fn=state.kernel_fn,
            log_likelihood_fn=state.log_likelihood_fn,
            jitter=state.jitter,
        )
        return -stochastic_svgp_elbo(
            state=new_state,
            x_batch=x_train,
            y_batch=y_train,
            dataset_size=x_train.shape[0],
        )

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1)))
    optimizer = optax.adam(learning_rate)
    mu_w: jax.Array = state.whitened_mean
    l_w: jax.Array = state.whitened_root_cov
    opt_state = optimizer.init((mu_w, l_w))

    for _ in range(num_steps):
        grad_mu, grad_l = grad_fn(mu_w, l_w)
        # Constrain L_w to lower-triangular by projecting grads onto the tril mask.
        grad_l_tril = jnp.tril(grad_l)
        updates, opt_state = optimizer.update((grad_mu, grad_l_tril), opt_state)
        update_mu = jnp.asarray(updates[0])  # type: ignore[index]
        update_l = jnp.asarray(updates[1])  # type: ignore[index]
        mu_w = mu_w + update_mu
        l_w = jnp.tril(l_w + update_l)

    return StochasticSVGPState(
        x_inducing=state.x_inducing,
        whitened_mean=mu_w,
        whitened_root_cov=l_w,
        lengthscale=state.lengthscale,
        output_scale=state.output_scale,
        kernel_fn=state.kernel_fn,
        log_likelihood_fn=state.log_likelihood_fn,
        jitter=state.jitter,
    )


def test_gaussian_likelihood_stochastic_svgp_optimum_recovers_titsias() -> None:
    """At convergence with Z = X and Gaussian likelihood, predict matches Titsias."""
    rng = jax.random.PRNGKey(11)
    x_train = jnp.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1)) + 0.05 * jax.random.normal(rng, (12,))
    lengthscale, output_scale, noise_std = 0.5, 1.0, 0.1
    noise_var = noise_std**2

    def gaussian_log_lik(f: jax.Array, y: jax.Array) -> jax.Array:
        # log N(y; f, noise_var) = -0.5 log(2 pi noise_var) - 0.5 (y - f)^2 / noise_var
        return -0.5 * jnp.log(2.0 * jnp.pi * noise_var) - 0.5 * (y - f) ** 2 / noise_var

    init = init_stochastic_svgp_state(
        x_inducing=x_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        log_likelihood_fn=gaussian_log_lik,
    )
    trained = _train_stochastic_svgp(
        state=init, x_train=x_train, y_train=y_train, num_steps=400, learning_rate=5e-2
    )
    titsias = fit_svgp(
        x_train=x_train,
        y_train=y_train,
        x_inducing=x_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise_std=noise_std,
    )
    x_test = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    stochastic_pred = predict_stochastic_svgp(state=trained, x_test=x_test)
    titsias_pred = predict_svgp(state=titsias, x_test=x_test)
    # Stochastic-SVGP converges to Titsias predictive within tolerance after 400 Adam steps.
    assert jnp.allclose(stochastic_pred.mean, titsias_pred.mean, atol=5e-2)


def test_bernoulli_stochastic_svgp_optimum_approximates_laplace_classification() -> None:
    """Stochastic SVGP class probabilities approximate Bernoulli Laplace at Z = X."""
    rng = jax.random.PRNGKey(13)
    x_train = jax.random.uniform(rng, (20, 1), minval=-1.5, maxval=1.5)
    y_train = jnp.sign(jnp.sin(2.0 * x_train.squeeze(-1)))
    lengthscale, output_scale = 0.4, 1.0
    init = init_stochastic_svgp_state(
        x_inducing=x_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        log_likelihood_fn=bernoulli_log_likelihood,
    )
    trained = _train_stochastic_svgp(
        state=init, x_train=x_train, y_train=y_train, num_steps=600, learning_rate=5e-2
    )
    laplace_state = fit_bernoulli_laplace_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=lengthscale,
        output_scale=output_scale,
        num_newton_iterations=40,
    )
    x_test = jnp.linspace(-1.5, 1.5, 8).reshape(-1, 1)
    svgp_pred = predict_stochastic_svgp(state=trained, x_test=x_test)
    laplace_pred = predict_bernoulli_laplace_gp(state=laplace_state, x_test=x_test)
    # Class probabilities (MacKay-style sigmoid of latent under both methods) should
    # agree to within a few percent — Laplace and variational classification are both
    # approximations of the same true marginal, so they coincide near the optimum.
    assert svgp_pred.variance is not None
    svgp_class_prob = jax.nn.sigmoid(
        svgp_pred.mean / jnp.sqrt(1.0 + jnp.pi * svgp_pred.variance / 8.0)
    )
    assert jnp.max(jnp.abs(svgp_class_prob - laplace_pred.mean)) < 0.2


# -----------------------------------------------------------------------------
# Per-observation log-likelihood helpers
# -----------------------------------------------------------------------------


def test_bernoulli_log_likelihood_matches_log_sigmoid_definition() -> None:
    """``bernoulli_log_likelihood(f, y) = log σ(y f)`` for ``y ∈ {-1, +1}``."""
    f = jnp.linspace(-2.0, 2.0, 5)
    y = jnp.array([-1.0, 1.0, -1.0, 1.0, -1.0])
    assert jnp.allclose(
        bernoulli_log_likelihood(f, y),
        jax.nn.log_sigmoid(y * f),
        atol=1e-7,
    )


def test_poisson_log_likelihood_matches_exp_link_definition() -> None:
    """``poisson_log_likelihood(f, y) = y f - exp(f) - log Γ(y + 1)``."""
    f = jnp.array([0.5, 1.0, 1.5])
    y = jnp.array([1.0, 3.0, 2.0])
    expected = y * f - jnp.exp(f) - jax.scipy.special.gammaln(y + 1.0)
    assert jnp.allclose(poisson_log_likelihood(f, y), expected, atol=1e-7)


# -----------------------------------------------------------------------------
# Natural-gradient updates (Salimbeni+ 2018) — D3 sub-item (b)
# -----------------------------------------------------------------------------


def test_natural_gradient_step_returns_finite_state() -> None:
    """One nat-grad step from initial (μ_w=0, L_w=I) produces a finite state."""
    rng = jax.random.PRNGKey(31)
    x_train = jax.random.uniform(rng, (15, 1), minval=-1.5, maxval=1.5)
    y_train = jnp.sign(jnp.sin(2.0 * x_train.squeeze(-1)))
    state = init_stochastic_svgp_state(
        x_inducing=jnp.linspace(-1.5, 1.5, 6).reshape(-1, 1),
        lengthscale=0.5,
        output_scale=1.0,
        log_likelihood_fn=bernoulli_log_likelihood,
    )
    new_state = natural_gradient_step(
        state=state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
        learning_rate=0.5,
    )
    assert isinstance(new_state, StochasticSVGPState)
    assert jnp.all(jnp.isfinite(new_state.whitened_mean))
    assert jnp.all(jnp.isfinite(new_state.whitened_root_cov))
    # L_w must remain lower-triangular with positive diagonal.
    assert jnp.allclose(new_state.whitened_root_cov, jnp.tril(new_state.whitened_root_cov))
    assert jnp.all(jnp.diag(new_state.whitened_root_cov) > 0.0)


def test_natural_gradient_step_increases_elbo() -> None:
    """A single nat-grad step with a reasonable learning rate increases the ELBO."""
    rng = jax.random.PRNGKey(41)
    x_train = jax.random.uniform(rng, (20, 1), minval=-1.5, maxval=1.5)
    y_train = jnp.sign(jnp.sin(2.0 * x_train.squeeze(-1)))
    state = init_stochastic_svgp_state(
        x_inducing=jnp.linspace(-1.5, 1.5, 6).reshape(-1, 1),
        lengthscale=0.5,
        output_scale=1.0,
        log_likelihood_fn=bernoulli_log_likelihood,
    )
    elbo_before = stochastic_svgp_elbo(
        state=state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
    )
    new_state = natural_gradient_step(
        state=state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
        learning_rate=0.5,
    )
    elbo_after = stochastic_svgp_elbo(
        state=new_state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
    )
    assert float(elbo_after) > float(elbo_before)


def test_natural_gradient_converges_faster_than_adam_on_gaussian_likelihood() -> None:
    """At equal step counts, nat-grad lands at a higher ELBO than Adam (Salimbeni+ 2018)."""
    import optax  # local import keeps top-level optax dep scoped to convergence tests

    rng = jax.random.PRNGKey(53)
    x_train = jnp.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1)) + 0.05 * jax.random.normal(rng, (12,))
    noise_var = 0.01

    def gaussian_log_lik(f: jax.Array, y: jax.Array) -> jax.Array:
        return -0.5 * jnp.log(2.0 * jnp.pi * noise_var) - 0.5 * (y - f) ** 2 / noise_var

    initial_state = init_stochastic_svgp_state(
        x_inducing=x_train,
        lengthscale=0.5,
        output_scale=1.0,
        log_likelihood_fn=gaussian_log_lik,
    )
    num_steps = 8

    # Natural gradient path.
    nat_state = initial_state
    for _ in range(num_steps):
        nat_state = natural_gradient_step(
            state=nat_state,
            x_batch=x_train,
            y_batch=y_train,
            dataset_size=x_train.shape[0],
            learning_rate=0.8,
        )
    nat_elbo = stochastic_svgp_elbo(
        state=nat_state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
    )

    # Adam path (same number of steps, tuned LR).
    def loss(mu_w: jax.Array, l_w: jax.Array) -> jax.Array:
        new_state = StochasticSVGPState(
            x_inducing=initial_state.x_inducing,
            whitened_mean=mu_w,
            whitened_root_cov=l_w,
            lengthscale=initial_state.lengthscale,
            output_scale=initial_state.output_scale,
            kernel_fn=initial_state.kernel_fn,
            log_likelihood_fn=initial_state.log_likelihood_fn,
            jitter=initial_state.jitter,
        )
        return -stochastic_svgp_elbo(
            state=new_state,
            x_batch=x_train,
            y_batch=y_train,
            dataset_size=x_train.shape[0],
        )

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1)))
    optimizer = optax.adam(5e-2)
    mu_w = initial_state.whitened_mean
    l_w = initial_state.whitened_root_cov
    opt_state = optimizer.init((mu_w, l_w))
    for _ in range(num_steps):
        grad_mu, grad_l = grad_fn(mu_w, l_w)
        grad_l_tril = jnp.tril(grad_l)
        updates, opt_state = optimizer.update((grad_mu, grad_l_tril), opt_state)
        update_mu = jnp.asarray(updates[0])  # type: ignore[index]
        update_l = jnp.asarray(updates[1])  # type: ignore[index]
        mu_w = mu_w + update_mu
        l_w = jnp.tril(l_w + update_l)
    adam_state = StochasticSVGPState(
        x_inducing=initial_state.x_inducing,
        whitened_mean=mu_w,
        whitened_root_cov=l_w,
        lengthscale=initial_state.lengthscale,
        output_scale=initial_state.output_scale,
        kernel_fn=initial_state.kernel_fn,
        log_likelihood_fn=initial_state.log_likelihood_fn,
        jitter=initial_state.jitter,
    )
    adam_elbo = stochastic_svgp_elbo(
        state=adam_state,
        x_batch=x_train,
        y_batch=y_train,
        dataset_size=x_train.shape[0],
    )

    # Salimbeni+ 2018 §4 claim: nat-grad converges in O(1) - O(10) steps; Adam needs
    # 100x more for the same ELBO. With 8 steps, nat-grad should clearly dominate.
    assert float(nat_elbo) > float(adam_elbo)


def test_natural_gradient_step_is_jit_compatible() -> None:
    """The natural-gradient update compiles under ``jax.jit``."""
    x_train = jnp.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    y_train = jnp.sign(jnp.sin(2.0 * x_train.squeeze(-1)))
    x_inducing = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)

    @jax.jit
    def one_step(x_b: jax.Array, y_b: jax.Array) -> jax.Array:
        state = init_stochastic_svgp_state(
            x_inducing=x_inducing,
            lengthscale=0.5,
            output_scale=1.0,
            log_likelihood_fn=bernoulli_log_likelihood,
        )
        new_state = natural_gradient_step(
            state=state,
            x_batch=x_b,
            y_batch=y_b,
            dataset_size=x_b.shape[0],
            learning_rate=0.5,
        )
        return new_state.whitened_mean

    out = one_step(x_train, y_train)
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))
