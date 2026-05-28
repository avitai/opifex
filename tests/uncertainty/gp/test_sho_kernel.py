r"""Tests for the damped harmonic oscillator (SHO/celerite) kernel.

The underdamped SHO kernel (Foreman-Mackey, Agol, Ambikasaran, Angus
2017 — *Fast and scalable Gaussian process modelling with
applications to astronomical time series*, AJ, arXiv:1703.09710) has
the closed form

.. math::

    k(\tau) = \sigma^{2}\,\exp\!\left(-\frac{\omega\,|\tau|}{2 Q}\right)
        \left[\cos\!\left(\frac{g\,\omega\,|\tau|}{2 Q}\right)
              + \frac{1}{g}\sin\!\left(\frac{g\,\omega\,|\tau|}{2 Q}\right)\right]

for ``Q > 1/2`` (underdamped) where ``g = √(4 Q² − 1)``,
``τ = x_i − x_j``, and ``ω`` is the natural angular frequency. The
opifex implementation maps the standard kernel hyperparameters to
``output_scale = σ`` and ``lengthscale = 1 / ω``; the quality factor
``Q`` is closed over by the kernel-factory wrapper.

References
----------
* Foreman-Mackey, D., Agol, E., Ambikasaran, S., Angus, R. 2017 —
  *Fast and scalable Gaussian process modelling with applications to
  astronomical time series*, AJ, arXiv:1703.09710 (PRIMARY —
  celerite SHO kernel).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    damped_oscillator_kernel,
    fit_exact_gp,
    predict_exact_gp,
)


def _closed_form_sho(
    tau: jax.Array, *, omega: float, quality: float, sigma: float
) -> jax.Array:
    g = jnp.sqrt(4.0 * quality**2 - 1.0)
    coef = omega * jnp.abs(tau) / (2.0 * quality)
    g_coef = g * coef
    return (
        sigma**2
        * jnp.exp(-coef)
        * (jnp.cos(g_coef) + jnp.sin(g_coef) / g)
    )


def test_damped_oscillator_kernel_matches_closed_form_at_unit_distance() -> None:
    """Direct check against the Foreman-Mackey+ 2017 closed form."""
    kernel = damped_oscillator_kernel(quality_factor=2.0)
    x = jnp.asarray([[0.0]])
    x_far = jnp.asarray([[1.0]])
    k = kernel(x, x_far, lengthscale=1.0, output_scale=1.5)
    expected = _closed_form_sho(jnp.asarray(1.0), omega=1.0, quality=2.0, sigma=1.5)
    assert jnp.allclose(k[0, 0], expected, atol=1e-5)


def test_damped_oscillator_kernel_diagonal_equals_output_scale_squared() -> None:
    """At zero separation ``k(0) = σ_f²``."""
    kernel = damped_oscillator_kernel(quality_factor=2.0)
    x = jnp.linspace(0.0, 5.0, 5).reshape(-1, 1)
    k = kernel(x, x, lengthscale=1.0, output_scale=1.5)
    assert jnp.allclose(jnp.diag(k), jnp.full(5, 1.5**2), atol=1e-5)


def test_damped_oscillator_kernel_decays_exponentially_in_lag() -> None:
    """Envelope ``e^{-ω τ / (2 Q)}`` produces strict decay along the diagonal."""
    kernel = damped_oscillator_kernel(quality_factor=5.0)
    x = jnp.asarray([[0.0]])
    rows = []
    for tau in [0.0, 1.0, 5.0, 20.0]:
        rows.append(float(kernel(x, jnp.asarray([[tau]]), lengthscale=1.0, output_scale=1.0)[0, 0]))
    envelope = [abs(v) for v in rows]
    # Envelope strictly decreases beyond zero.
    assert envelope[0] >= envelope[1]
    assert envelope[2] >= envelope[3]


def test_damped_oscillator_kernel_oscillates_in_lag_for_high_quality() -> None:
    """High Q → multiple zero-crossings in the lag direction."""
    kernel = damped_oscillator_kernel(quality_factor=20.0)  # very underdamped
    x = jnp.asarray([[0.0]])
    taus = jnp.linspace(0.0, 30.0, 200).reshape(-1, 1)
    values = kernel(x, taus, lengthscale=1.0, output_scale=1.0)[0]
    signs = jnp.sign(values)
    crossings = jnp.sum((signs[:-1] * signs[1:] < 0).astype(jnp.int32))
    assert int(crossings) > 2


def test_damped_oscillator_kernel_plugs_into_fit_exact_gp() -> None:
    """The SHO kernel routes through ``fit_exact_gp(..., kernel_fn=…)``."""
    kernel = damped_oscillator_kernel(quality_factor=2.0)
    t_train = jnp.linspace(0.0, 6.0, 16).reshape(-1, 1)
    y_train = jnp.exp(-0.5 * t_train.squeeze(-1)) * jnp.cos(t_train.squeeze(-1))
    state = fit_exact_gp(
        x_train=t_train,
        y_train=y_train,
        lengthscale=1.0,
        output_scale=1.0,
        noise_std=0.05,
        kernel_fn=kernel,
    )
    predictive = predict_exact_gp(state=state, x_test=t_train)
    assert predictive.variance is not None
    assert jnp.max(jnp.abs(predictive.mean - y_train)) < 5.0 * 0.05


def test_damped_oscillator_kernel_is_jit_compatible() -> None:
    """End-to-end ``jax.jit`` compatibility."""
    kernel = damped_oscillator_kernel(quality_factor=2.0)
    t_train = jnp.linspace(0.0, 4.0, 8).reshape(-1, 1)
    y_train = jax.random.normal(jax.random.PRNGKey(0), (8,))
    t_test = jnp.linspace(0.0, 4.0, 3).reshape(-1, 1)

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array, x_q: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=1.0,
            output_scale=1.0,
            noise_std=0.1,
            kernel_fn=kernel,
        )
        pd = predict_exact_gp(state=state, x_test=x_q)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(t_train, y_train, t_test)
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))


def test_damped_oscillator_kernel_rejects_quality_at_or_below_half() -> None:
    """Underdamped form requires ``Q > 1/2``."""
    with pytest.raises(ValueError, match="quality"):
        damped_oscillator_kernel(quality_factor=0.5)
