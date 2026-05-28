r"""Tests for the direct-evaluation Matern kernel family + ICM multi-output.

The Matern family in *direct* (kernel-evaluation) form complements the
existing SDE state-space matern kernels at
``opifex.uncertainty.statespace.kernels`` (which return ``(F, L, Q_c,
H, P_∞)`` quadruples for Kalman filtering). For the exact-conjugate GP
fit/predict driver we need the direct evaluation
``k(r) = σ² (1 + …) exp(-c r/ℓ)`` instead.

Multi-output ICM (Intrinsic Coregionalisation Model, Alvarez et al.
2012 *Kernels for Vector-Valued Functions: A Review*, Foundations and
Trends in Machine Learning, vol. 4 no. 3) factorises the multi-output
kernel as a scalar base kernel times a coregionalisation matrix
``B ∈ R^{T × T}`` between ``T`` tasks: ``k_ICM((x, i), (x', j)) =
k_base(x, x') · B[i, j]``.

References
----------
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; §4.2 (Matern family, PRIMARY).
* Alvarez, M. A., Rosasco, L., Lawrence, N. D. 2012 — *Kernels for
  Vector-Valued Functions: A Review*, arXiv:1106.6251 (ICM, PRIMARY).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.gp import (
    fit_exact_gp,
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
    multi_output_icm_kernel,
    predict_exact_gp,
    rbf_kernel,
)


def test_matern12_matches_closed_form_at_unit_distance() -> None:
    r"""``Matern12(r=1, ℓ=1, σ=1) = exp(-1)``."""
    x = jnp.asarray([[0.0]])
    x_far = jnp.asarray([[1.0]])
    k = matern12_kernel(x, x_far, lengthscale=1.0, output_scale=1.0)
    assert jnp.allclose(k[0, 0], jnp.exp(-1.0), atol=1e-6)


def test_matern32_matches_closed_form_at_unit_distance() -> None:
    r"""``Matern32(r=1, ℓ=1, σ=1) = (1 + √3) exp(-√3)``."""
    x = jnp.asarray([[0.0]])
    x_far = jnp.asarray([[1.0]])
    k = matern32_kernel(x, x_far, lengthscale=1.0, output_scale=1.0)
    expected = (1.0 + jnp.sqrt(3.0)) * jnp.exp(-jnp.sqrt(3.0))
    assert jnp.allclose(k[0, 0], expected, atol=1e-6)


def test_matern52_matches_closed_form_at_unit_distance() -> None:
    r"""``Matern52(r=1, ℓ=1, σ=1) = (1 + √5 + 5/3) exp(-√5)``."""
    x = jnp.asarray([[0.0]])
    x_far = jnp.asarray([[1.0]])
    k = matern52_kernel(x, x_far, lengthscale=1.0, output_scale=1.0)
    expected = (1.0 + jnp.sqrt(5.0) + 5.0 / 3.0) * jnp.exp(-jnp.sqrt(5.0))
    assert jnp.allclose(k[0, 0], expected, atol=1e-6)


def test_matern_family_diagonal_equals_output_scale_squared() -> None:
    """At zero separation every Matern kernel returns ``σ_f²``."""
    x = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    for kernel_fn in (matern12_kernel, matern32_kernel, matern52_kernel):
        k = kernel_fn(x, x, lengthscale=0.4, output_scale=1.5)
        assert jnp.allclose(jnp.diag(k), jnp.full(4, 1.5**2), atol=1e-6), kernel_fn.__name__


def test_fit_exact_gp_accepts_matern32_kernel_via_kernel_fn() -> None:
    """The fit / predict driver routes through ``kernel_fn`` for any kernel."""
    x_train = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1))
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.05,
        kernel_fn=matern32_kernel,
    )
    predictive = predict_exact_gp(state=state, x_test=x_train)
    assert predictive.variance is not None
    # Matern32 GP also interpolates training points within a few noise scales.
    assert jnp.max(jnp.abs(predictive.mean - y_train)) < 5.0 * 0.05


def test_multi_output_icm_kernel_factorises_base_times_coregionalisation() -> None:
    r"""``k_ICM((x, i), (x', j)) = k_base(x, x') · B[i, j]``."""
    icm = multi_output_icm_kernel(
        base_kernel_fn=rbf_kernel,
        coregionalisation=jnp.asarray([[1.0, 0.3], [0.3, 0.5]]),
    )
    x1 = jnp.asarray([[0.0, 0], [0.5, 0], [0.0, 1], [0.5, 1]])  # last col = task index
    k = icm(x1, x1, lengthscale=1.0, output_scale=1.0)
    base = rbf_kernel(x1[:, :1], x1[:, :1], lengthscale=1.0, output_scale=1.0)
    expected = base * jnp.asarray(
        [[1.0, 1.0, 0.3, 0.3], [1.0, 1.0, 0.3, 0.3], [0.3, 0.3, 0.5, 0.5], [0.3, 0.3, 0.5, 0.5]]
    )
    assert jnp.allclose(k, expected, atol=1e-6)


def test_multi_output_icm_kernel_is_jit_compatible() -> None:
    """The ICM kernel compiles inside ``jax.jit``."""
    icm = multi_output_icm_kernel(
        base_kernel_fn=rbf_kernel,
        coregionalisation=jnp.asarray([[1.0, 0.1], [0.1, 0.8]]),
    )

    @jax.jit
    def evaluate(x: jax.Array) -> jax.Array:
        return icm(x, x, lengthscale=0.7, output_scale=1.2)

    x = jnp.asarray([[0.0, 0], [0.4, 1]])
    k = evaluate(x)
    assert k.shape == (2, 2)
    assert jnp.all(jnp.isfinite(k))


def test_multi_output_icm_gp_fit_predict_uses_task_aware_inputs() -> None:
    """Exact GP fit/predict via ICM kernel returns calibrated multi-output predictive."""
    x_train = jnp.asarray(
        [[-1.0, 0], [-0.5, 0], [0.0, 0], [0.5, 0], [1.0, 0], [-0.5, 1], [0.5, 1]]
    )
    y_train = jnp.asarray([0.0, 1.0, 0.0, -1.0, 0.0, 2.0, -2.0])
    icm = multi_output_icm_kernel(
        base_kernel_fn=rbf_kernel,
        coregionalisation=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    state = fit_exact_gp(
        x_train=x_train,
        y_train=y_train,
        lengthscale=0.5,
        output_scale=1.0,
        noise_std=0.05,
        kernel_fn=icm,
    )
    predictive = predict_exact_gp(state=state, x_test=x_train)
    assert predictive.variance is not None
    # Independent tasks (B=I): predictions interpolate within noise.
    assert jnp.max(jnp.abs(predictive.mean - y_train)) < 5.0 * 0.05
