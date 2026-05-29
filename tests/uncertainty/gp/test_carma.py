r"""CARMA(p, q) direct-evaluation kernel — Slice 29 (Task 11.1 carryover).

Phase 11 Task 11.1 body line 47-49 mandates ``CARMA / Celerite / SHO
state-space kernels (1D scalable) — cite tinygp.kernels.quasisep.
{CARMA, Celerite, SHO}``. Celerite (slice 14 D4) and SHO state-space
(slice 15 D1) shipped; this slice closes the **CARMA** half by
shipping the direct-evaluation form. The scalable state-space CARMA
variant is filed as a documented deferral in
``notes/06-task-11.1-deferred-items.md`` (D7).

Continuous-time autoregressive moving-average kernel from Kelly+
2014 (arXiv:1402.5978). The autocovariance is a closed-form sum of
real-and-complex exponentials parametrised by:

* ``alpha`` — length-``p`` array of autoregressive coefficients
  (the implicit ``α_p = 1`` is appended internally).
* ``beta`` — length-``(q+1)`` array of moving-average coefficients
  (``q + 1 ≤ p``). ``β_0 = sigma`` is absorbed into the overall
  amplitude.

The kernel reduces to known cases at low orders:

* CARMA(1, 0) with ``α = (a,)``, ``β = (1,)`` ≡ exponential /
  Matern-1/2 with ``lengthscale = 1/a``.
* CARMA(2, 1) is the canonical Celerite Complex term (Foreman-
  Mackey+ 2017).

Reference implementation consulted (READ-ONLY):
``../tinygp/src/tinygp/kernels/quasisep.py:CARMA`` (lines 672-885)
plus the ``carma_roots`` / ``carma_acvf`` helpers.

References
----------
* Kelly, Becker, Sobolewska, Siemiginowska, Uttley 2014 —
  *Flexible and scalable methods for quantifying stochastic
  variability in the era of massive time-domain astronomical data
  sets*, ApJ (arXiv:1402.5978).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def test_carma_1_0_matches_matern12_for_a_single_real_root() -> None:
    r"""CARMA(1, 0) with ``α = (a,)`` collapses to ``σ² exp(-a |τ|)``.

    Matern-1/2 (exponential) kernel
    ``k(τ) = output_scale² · exp(-|τ| / ℓ)`` with ``ℓ = 1/a`` is the
    exact CARMA(1, 0) acvf when ``α_0 = a`` and the MA polynomial has
    only the constant term ``β_0 = 1``.
    """
    from opifex.uncertainty.gp import carma_kernel, matern12_kernel

    a_value = 1.5  # decay rate
    kernel = carma_kernel(alpha=jnp.asarray([a_value]), beta=jnp.asarray([1.0]))
    x = jnp.linspace(0.0, 2.0, 6).reshape(-1, 1)
    carma_gram = kernel(x, x, lengthscale=1.0, output_scale=1.0)
    # CARMA(1, 0) ≡ Matern-1/2 with lengthscale = 1/a, output_scale matched.
    matern_gram = matern12_kernel(x, x, lengthscale=1.0 / a_value, output_scale=1.0)
    # Match the variance normalisation: CARMA(1,0) ACVF has amplitude
    # 1 / (2 a), Matern-1/2 has amplitude output_scale².  Compare the
    # *correlation* (gram normalised by the diagonal) which is
    # invariant to the global amplitude scaling.
    carma_corr = carma_gram / carma_gram[0, 0]
    matern_corr = matern_gram / matern_gram[0, 0]
    assert jnp.allclose(carma_corr, matern_corr, atol=1e-6)


def test_carma_kernel_returns_symmetric_positive_semidefinite_gram() -> None:
    """Gram matrices are symmetric and have non-negative eigenvalues."""
    from opifex.uncertainty.gp import carma_kernel

    kernel = carma_kernel(alpha=jnp.asarray([2.0, 1.5]), beta=jnp.asarray([1.0, 0.3]))
    x = jnp.linspace(0.0, 3.0, 8).reshape(-1, 1)
    gram = kernel(x, x, lengthscale=1.0, output_scale=1.0)
    # Symmetric (up to floating-point asymmetry from the eigendecomposition).
    assert jnp.allclose(gram, gram.T, atol=1e-5)
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (gram + gram.T))
    # Allow a tiny negative tolerance from numerical noise.
    assert jnp.all(eigenvalues > -1e-5)


def test_carma_kernel_diag_is_constant_and_positive() -> None:
    """At zero lag the CARMA kernel evaluates to a positive constant variance."""
    from opifex.uncertainty.gp import carma_kernel

    kernel = carma_kernel(alpha=jnp.asarray([2.0, 1.0]), beta=jnp.asarray([1.0, 0.2]))
    x = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    gram = kernel(x, x, lengthscale=1.0, output_scale=2.0)
    diag = jnp.diag(gram)
    assert jnp.all(diag > 0.0)
    assert jnp.allclose(diag, diag[0], atol=1e-6)


def test_carma_kernel_is_jit_compatible_through_fit_exact_gp() -> None:
    """End-to-end ``jax.jit`` compatibility with ``fit_exact_gp``."""
    from opifex.uncertainty.gp import (
        carma_kernel,
        fit_exact_gp,
        predict_exact_gp,
    )

    kernel = carma_kernel(alpha=jnp.asarray([2.0, 1.0]), beta=jnp.asarray([1.0, 0.2]))
    x_train = jnp.linspace(0.0, 2.5, 8).reshape(-1, 1)
    y_train = jnp.sin(2.0 * x_train.squeeze(-1)) * jnp.exp(-0.2 * x_train.squeeze(-1))
    x_test = jnp.linspace(0.0, 2.5, 4).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=1.0,
            output_scale=1.0,
            noise_std=0.05,
            kernel_fn=kernel,
        )
        predictive = predict_exact_gp(state=state, x_test=x_q)
        assert predictive.variance is not None
        return predictive.mean + predictive.variance

    out = fit_predict(x_train, y_train, x_test)
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


def test_carma_kernel_rejects_ma_order_exceeding_ar_order() -> None:
    """``q + 1 > p`` is invalid per Kelly+ 2014; the factory must reject it."""
    import pytest

    from opifex.uncertainty.gp import carma_kernel

    with pytest.raises(ValueError, match="beta"):
        carma_kernel(
            alpha=jnp.asarray([1.0]),
            beta=jnp.asarray([1.0, 0.5, 0.2]),  # q + 1 = 3 > p = 1
        )


def test_carma_kernel_output_scale_squared_amplitude() -> None:
    """Doubling ``output_scale`` multiplies the kernel by 4 (amplitude squared)."""
    from opifex.uncertainty.gp import carma_kernel

    kernel = carma_kernel(alpha=jnp.asarray([1.5, 0.8]), beta=jnp.asarray([1.0, 0.1]))
    x = jnp.linspace(0.0, 1.5, 4).reshape(-1, 1)
    gram_unit = kernel(x, x, lengthscale=1.0, output_scale=1.0)
    gram_double = kernel(x, x, lengthscale=1.0, output_scale=2.0)
    assert jnp.allclose(gram_double, 4.0 * gram_unit, atol=1e-6)
