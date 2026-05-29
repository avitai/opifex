r"""Tests for the orthogonal additive kernel (OAK) — Task 11.1 D2.

OAK (Lu, Boukouvalas, Hensman 2022 ICML — *Additive Gaussian Processes
Revisited*) extends the first-order additive kernel to higher-order
ANOVA decompositions:

.. math::

    k_{\text{OAK}}(\mathbf{x}, \mathbf{x}')
      = \sum_{\ell = 0}^{D_{\text{tilde}}}\sigma_{\ell}^{2}\,e_{\ell}\!\bigl(
            \tilde{k}_{1}(x_{1}, x'_{1}),\dots,\tilde{k}_{D}(x_{D}, x'_{D})
        \bigr),

where ``e_ℓ`` is the ``ℓ``-th elementary symmetric polynomial of the
``D`` per-dimension *constrained* (Gaussian-measure-orthogonal) base
kernels ``k̃_d`` and ``σ²_ℓ`` are non-negative order variances. The
elementary symmetric polynomials are evaluated via the **Newton-Girard
recursion** ``e_n = (1/n) Σ_{k=1}^{n} (-1)^{k-1} e_{n-k} s_k`` with
``s_k = Σ_d k̃_d^k`` (cheap, no combinatorial blow-up).

For the RBF base kernel under ``p(x) = N(μ, ζ²)`` Lu et al. 2022 derives
the closed-form projection (eq. 10):

.. math::

    \tilde{k}(x, x') = k(x, x') -
        \frac{\sigma_{f}^{2}\,\ell\,\sqrt{\ell^{2} + 2\zeta^{2}}}
             {\ell^{2} + \zeta^{2}}\,
        \exp\!\left(-\frac{(x - \mu)^{2} + (x' - \mu)^{2}}
                          {2(\ell^{2} + \zeta^{2})}\right).

Reference implementation consulted (READ-ONLY):
``../GPJax/gpjax/kernels/additive/oak.py``.

References
----------
* Lu, X., Boukouvalas, A., Hensman, J. 2022 — *Additive Gaussian
  Processes Revisited*, ICML (PRIMARY).
* Duvenaud, D., Nickisch, H., Rasmussen, C. E. 2011 — *Additive
  Gaussian Processes*, NeurIPS (predecessor).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    additive_kernel,
    constrained_rbf_kernel,
    fit_exact_gp,
    orthogonal_additive_kernel,
    predict_exact_gp,
    rbf_kernel,
)


# -----------------------------------------------------------------------------
# Constrained RBF — the canonical Gaussian-measure-orthogonal base kernel
# -----------------------------------------------------------------------------


def test_constrained_rbf_kernel_integral_against_constant_is_zero_under_gaussian_measure() -> None:
    r"""``∫ k̃(x, y) p(y) dy = 0`` when ``p = N(0, 1)`` (Lu 2022 Prop. 1).

    Monte-Carlo verification: sampling ``y ~ N(0, 1)``, the mean of
    ``k̃(x, y)`` for fixed ``x`` must converge to zero. The bare RBF
    integrates to a strictly positive number, so this test
    discriminates against an unconstrained-RBF baseline.
    """
    key = jax.random.PRNGKey(0)
    y_samples = jax.random.normal(key, (10_000, 1))
    x_query = jnp.asarray([[0.3]])
    kernel = constrained_rbf_kernel(input_mean=0.0, input_std=1.0)
    bare_baseline = rbf_kernel(x_query, y_samples, lengthscale=0.8, output_scale=1.0)
    constrained = kernel(x_query, y_samples, lengthscale=0.8, output_scale=1.0)
    # bare RBF has strictly positive Gaussian-measure mean
    assert float(jnp.mean(bare_baseline)) > 0.1
    # constrained kernel has near-zero Gaussian-measure mean
    assert float(jnp.abs(jnp.mean(constrained))) < 0.05


def test_constrained_rbf_kernel_returns_a_matrix_of_the_right_shape() -> None:
    """Standard ``(n, m)`` kernel-matrix output shape."""
    kernel = constrained_rbf_kernel()
    x1 = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    x2 = jnp.linspace(-0.5, 0.5, 3).reshape(-1, 1)
    gram = kernel(x1, x2, lengthscale=0.5, output_scale=1.0)
    assert gram.shape == (5, 3)
    assert jnp.all(jnp.isfinite(gram))


# -----------------------------------------------------------------------------
# OAK kernel — collapse / Newton-Girard / orthogonality / API
# -----------------------------------------------------------------------------


def test_oak_with_max_order_one_collapses_to_additive_kernel_of_the_same_base() -> None:
    """At ``max_order = 1`` and ``σ²_0 = 0, σ²_1 = 1``, OAK reduces to additive."""
    base = constrained_rbf_kernel()
    oak = orthogonal_additive_kernel(
        base_kernel_fns=(base, base),
        max_order=1,
        order_variances=jnp.asarray([0.0, 1.0]),
    )
    additive = additive_kernel(component_kernel_fns=(base, base))
    x = jnp.asarray([[0.2, -0.4], [0.7, 0.1], [-0.5, 0.5]])
    assert jnp.allclose(
        oak(x, x, lengthscale=0.6, output_scale=1.0),
        additive(x, x, lengthscale=0.6, output_scale=1.0),
        atol=1e-6,
    )


def test_oak_max_order_two_matches_explicit_newton_girard_expansion() -> None:
    r"""``e_2 = ½(s_1² - s_2)`` validated against per-dim products explicitly.

    With ``D = 3`` constrained kernels ``k̃_d`` and ``σ²_0 = 0, σ²_1 = 0,
    σ²_2 = 1``, OAK should compute

    .. math::

        \sum_{i < j} \tilde{k}_{i}(x_{i}, y_{i})\,\tilde{k}_{j}(x_{j}, y_{j})
            = \frac{1}{2}\!\left(s_{1}^{2} - s_{2}\right),

    where ``s_k = Σ_d k̃_d^k`` at the pair ``(x, y)``.
    """
    base = constrained_rbf_kernel()
    oak = orthogonal_additive_kernel(
        base_kernel_fns=(base, base, base),
        max_order=2,
        order_variances=jnp.asarray([0.0, 0.0, 1.0]),
    )
    x = jnp.asarray([[0.2, -0.4, 0.6], [0.7, 0.1, -0.3]])
    y = jnp.asarray([[0.1, 0.3, -0.2]])
    k_dim_0 = base(x[:, 0:1], y[:, 0:1], lengthscale=0.6, output_scale=1.0)
    k_dim_1 = base(x[:, 1:2], y[:, 1:2], lengthscale=0.6, output_scale=1.0)
    k_dim_2 = base(x[:, 2:3], y[:, 2:3], lengthscale=0.6, output_scale=1.0)
    expected = k_dim_0 * k_dim_1 + k_dim_0 * k_dim_2 + k_dim_1 * k_dim_2
    actual = oak(x, y, lengthscale=0.6, output_scale=1.0)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_oak_max_order_d_includes_full_product_interaction_term() -> None:
    r"""``e_D = Π_d k̃_d`` is the full ``D``-way interaction (top order)."""
    base = constrained_rbf_kernel()
    oak = orthogonal_additive_kernel(
        base_kernel_fns=(base, base, base),
        max_order=3,
        order_variances=jnp.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    x = jnp.asarray([[0.2, -0.4, 0.6]])
    y = jnp.asarray([[0.1, 0.3, -0.2]])
    k_dim_0 = base(x[:, 0:1], y[:, 0:1], lengthscale=0.6, output_scale=1.0)
    k_dim_1 = base(x[:, 1:2], y[:, 1:2], lengthscale=0.6, output_scale=1.0)
    k_dim_2 = base(x[:, 2:3], y[:, 2:3], lengthscale=0.6, output_scale=1.0)
    expected = k_dim_0 * k_dim_1 * k_dim_2
    actual = oak(x, y, lengthscale=0.6, output_scale=1.0)
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_oak_order_zero_term_is_a_constant_offset() -> None:
    r"""``e_0 = 1`` so ``σ²_0`` enters as an additive constant ``σ²_0 * J``."""
    base = constrained_rbf_kernel()
    oak_with_offset = orthogonal_additive_kernel(
        base_kernel_fns=(base, base),
        max_order=1,
        order_variances=jnp.asarray([2.5, 1.0]),
    )
    oak_no_offset = orthogonal_additive_kernel(
        base_kernel_fns=(base, base),
        max_order=1,
        order_variances=jnp.asarray([0.0, 1.0]),
    )
    x = jnp.asarray([[0.2, -0.4], [0.7, 0.1]])
    diff = oak_with_offset(x, x, lengthscale=0.6, output_scale=1.0) - oak_no_offset(
        x, x, lengthscale=0.6, output_scale=1.0
    )
    assert jnp.allclose(diff, 2.5, atol=1e-6)


def test_oak_is_jit_compatible_through_fit_exact_gp() -> None:
    """OAK + ``fit_exact_gp`` + ``predict_exact_gp`` compile under ``jax.jit``."""
    base = constrained_rbf_kernel()
    oak = orthogonal_additive_kernel(
        base_kernel_fns=(base, base),
        max_order=2,
        order_variances=jnp.asarray([0.0, 1.0, 0.5]),
    )
    x_train = jax.random.normal(jax.random.PRNGKey(0), (8, 2))
    y_train = jnp.sin(2.0 * x_train[:, 0]) + jnp.cos(x_train[:, 1])
    x_test = jax.random.normal(jax.random.PRNGKey(1), (3, 2))

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.05,
            kernel_fn=oak,
        )
        predictive = predict_exact_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x_train, y_train, x_test)
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_oak_rejects_empty_base_kernel_tuple() -> None:
    """At least one base kernel is required."""
    with pytest.raises(ValueError, match="base"):
        orthogonal_additive_kernel(
            base_kernel_fns=(),
            max_order=1,
            order_variances=jnp.asarray([0.0, 1.0]),
        )


def test_oak_rejects_max_order_greater_than_number_of_base_kernels() -> None:
    """``max_order`` must satisfy ``0 <= max_order <= D``."""
    base = constrained_rbf_kernel()
    with pytest.raises(ValueError, match="max_order"):
        orthogonal_additive_kernel(
            base_kernel_fns=(base, base),
            max_order=5,
            order_variances=jnp.ones(6),
        )


def test_oak_rejects_order_variance_shape_mismatch() -> None:
    """``order_variances`` must have shape ``(max_order + 1,)``."""
    base = constrained_rbf_kernel()
    with pytest.raises(ValueError, match="order_variances"):
        orthogonal_additive_kernel(
            base_kernel_fns=(base, base),
            max_order=2,
            order_variances=jnp.asarray([1.0, 1.0]),
        )
