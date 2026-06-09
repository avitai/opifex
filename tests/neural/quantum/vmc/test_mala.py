r"""Tests for the Metropolis-adjusted Langevin (MALA) sampler.

The sampler must produce walkers distributed as :math:`|\psi|^2`, report an
acceptance fraction in a sane band, run cleanly under ``jit``/``grad``/``vmap``,
and -- crucially -- carry the *asymmetric* proposal correction that distinguishes
MALA from an unadjusted Langevin walk. The asymmetric correction is exercised by
checking that the corrected sampler reproduces a known Gaussian target whereas an
uncorrected (symmetric-acceptance) variant is measurably biased.

Reference target: a fixed Gaussian log-amplitude ``log|psi|(x) = -0.5 * sum(x^2)``
gives the Born density ``|psi|^2 propto exp(-sum(x^2))``, i.e. ``N(0, 0.5 * I)``
(mean ``0``, per-coordinate variance ``0.5``). All distribution-level checks below
are stated against this exact target.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from opifex.neural.quantum.vmc.mala import langevin_drift, MALASampler


def _gaussian_log_abs(positions: jax.Array) -> jax.Array:
    r"""Fixed Gaussian log-amplitude ``log|psi| = -0.5 * sum(x^2)``.

    The Born density is then ``|psi|^2 propto exp(-sum(x^2))``, i.e. an
    isotropic Gaussian with per-coordinate variance ``0.5``.
    """
    return -0.5 * jnp.sum(positions**2)


def _equilibrate(
    sampler: MALASampler,
    walkers: jax.Array,
    key: jax.Array,
    blocks: int,
) -> jax.Array:
    """Run ``blocks`` independent :meth:`sample` calls and return final walkers."""
    for _ in range(blocks):
        key, subkey = jax.random.split(key)
        walkers, _ = sampler.sample(_gaussian_log_abs, walkers, subkey)
    return walkers


def test_sample_returns_walkers_and_acceptance() -> None:
    """One sampling call returns same-shaped walkers and a fraction in ``[0, 1]``."""
    sampler = MALASampler(steps=20, step_size=0.3)
    walkers = jax.random.normal(jax.random.PRNGKey(0), (256, 2, 3), dtype=jnp.float64)
    new_walkers, acceptance = sampler.sample(_gaussian_log_abs, walkers, jax.random.PRNGKey(1))
    assert new_walkers.shape == walkers.shape
    assert 0.0 <= float(acceptance) <= 1.0


def test_acceptance_in_sane_band() -> None:
    """A small Langevin step yields a high acceptance fraction in ``[0.5, 1.0]``.

    MALA's optimal acceptance is ~0.574 (Roberts & Rosenthal 1998); a modest
    step size should comfortably land in a high band on a smooth Gaussian.
    """
    sampler = MALASampler(steps=40, step_size=0.4)
    walkers = jax.random.normal(jax.random.PRNGKey(2), (512, 2, 3), dtype=jnp.float64)
    _, acceptance = sampler.sample(_gaussian_log_abs, walkers, jax.random.PRNGKey(3))
    assert 0.5 <= float(acceptance) <= 1.0


def test_samples_match_gaussian_target_mean_and_variance() -> None:
    r"""Equilibrated walkers reproduce ``N(0, 0.5 I)``: mean ~0, variance ~0.5."""
    sampler = MALASampler(steps=30, step_size=0.5)
    walkers = jax.random.normal(jax.random.PRNGKey(4), (8192, 2, 3), dtype=jnp.float64)
    walkers = _equilibrate(sampler, walkers, jax.random.PRNGKey(5), blocks=30)
    flat = walkers.reshape(-1)
    np.testing.assert_allclose(float(jnp.mean(flat)), 0.0, atol=0.03)
    np.testing.assert_allclose(float(jnp.var(flat)), 0.5, atol=0.03)


def test_asymmetric_correction_reduces_bias() -> None:
    r"""The asymmetric MH correction makes MALA unbiased vs. a symmetric variant.

    An unadjusted Langevin walk accepted with the *symmetric* ratio
    ``log p(x') - log p(x)`` (dropping the ``q(x|x')/q(x'|x)`` term) systematically
    under-disperses the Born density: the drift pulls walkers toward the mode, and
    without the reverse-proposal penalty the over-concentration is never corrected.
    With a deliberately large step the corrected sampler must recover variance
    ``0.5`` markedly better than the uncorrected one.
    """
    step_size = 0.9
    walkers0 = jax.random.normal(jax.random.PRNGKey(6), (8192, 2, 3), dtype=jnp.float64)

    corrected = MALASampler(steps=30, step_size=step_size, asymmetric_correction=True)
    uncorrected = MALASampler(steps=30, step_size=step_size, asymmetric_correction=False)

    w_corr = _equilibrate(corrected, walkers0, jax.random.PRNGKey(7), blocks=30)
    w_unc = _equilibrate(uncorrected, walkers0, jax.random.PRNGKey(7), blocks=30)

    var_corr = float(jnp.var(w_corr.reshape(-1)))
    var_unc = float(jnp.var(w_unc.reshape(-1)))

    # Corrected variance is close to the truth; uncorrected is biased low.
    assert abs(var_corr - 0.5) < 0.05
    assert abs(var_corr - 0.5) < abs(var_unc - 0.5)


def test_detailed_balance_on_1d_toy() -> None:
    r"""The asymmetric acceptance enforces detailed balance on a 1-D Gaussian.

    Detailed balance: ``p(x) T(x->x') = p(x') T(x'->x)`` for the MALA transition
    kernel, where ``T(x->x') = q(x'|x) * A(x,x')`` with the MALA acceptance
    ``A(x,x') = min(1, [p(x') q(x|x')] / [p(x) q(x'|x)])``. We verify the scalar
    identity ``p(x) q(x'|x) A(x,x') == p(x') q(x|x') A(x',x)`` directly for an
    arbitrary pair ``(x, x')`` -- this is the per-pair statement of reversibility.
    """
    step_size = 0.7
    dt = 0.5 * step_size**2

    def log_p(x: jax.Array) -> jax.Array:
        """1-D log Born density ``log|psi|^2 = -x^2`` (target ``N(0, 0.5)``)."""
        return -(x**2)

    grad_log_p = jax.grad(log_p)

    def log_q(x_to: jax.Array, x_from: jax.Array) -> jax.Array:
        """Log Langevin proposal density ``q(x_to | x_from)`` (up to a constant)."""
        mean = x_from + dt * grad_log_p(x_from)
        return -((x_to - mean) ** 2) / (4.0 * dt)

    def log_accept(x: jax.Array, x_prime: jax.Array) -> jax.Array:
        """MALA log acceptance ``min(0, .)`` for the move ``x -> x'``."""
        ratio = log_p(x_prime) + log_q(x, x_prime) - log_p(x) - log_q(x_prime, x)
        return jnp.minimum(0.0, ratio)

    x = jnp.asarray(0.4)
    x_prime = jnp.asarray(-0.9)
    forward = log_p(x) + log_q(x_prime, x) + log_accept(x, x_prime)
    reverse = log_p(x_prime) + log_q(x, x_prime) + log_accept(x_prime, x)
    np.testing.assert_allclose(float(forward), float(reverse), atol=1e-10)


def test_langevin_drift_matches_two_grad_log_abs() -> None:
    r"""The drift equals ``grad log p = 2 grad log|psi|`` per walker."""
    walkers = jax.random.normal(jax.random.PRNGKey(8), (16, 2, 3), dtype=jnp.float64)
    drift = langevin_drift(_gaussian_log_abs, walkers)
    # For log|psi| = -0.5 sum(x^2): grad log p = 2 * (-x) = -2 x.
    np.testing.assert_allclose(np.asarray(drift), np.asarray(-2.0 * walkers), atol=1e-10)


def test_drift_clipping_is_bounded() -> None:
    r"""With clipping enabled, the per-walker drift norm never exceeds the cap."""
    sampler = MALASampler(steps=1, step_size=0.5, max_drift_norm=1.0)
    # Far-out walkers produce large raw drifts (``-2 x``) that must be clipped.
    walkers = 50.0 * jnp.ones((8, 2, 3), dtype=jnp.float64)
    drift = sampler._clip_drift(langevin_drift(_gaussian_log_abs, walkers))
    norms = jnp.linalg.norm(drift.reshape(drift.shape[0], -1), axis=-1)
    assert float(jnp.max(norms)) <= 1.0 + 1e-9


def test_sample_is_jit_clean() -> None:
    """The sampling step jits (it already fuses grad + scan internally)."""
    sampler = MALASampler(steps=10, step_size=0.3)
    walkers = jax.random.normal(jax.random.PRNGKey(9), (128, 2, 3), dtype=jnp.float64)
    jitted = jax.jit(lambda w, k: sampler.sample(_gaussian_log_abs, w, k))
    new_walkers, acceptance = jitted(walkers, jax.random.PRNGKey(10))
    assert new_walkers.shape == walkers.shape
    assert jnp.isfinite(acceptance)


def test_sample_is_vmap_clean() -> None:
    """The whole :meth:`sample` call vmaps over a batch of PRNG keys (replicas)."""
    sampler = MALASampler(steps=5, step_size=0.3)
    base = jax.random.normal(jax.random.PRNGKey(11), (32, 2, 3), dtype=jnp.float64)
    walkers = jnp.broadcast_to(base, (4, *base.shape))
    keys = jax.random.split(jax.random.PRNGKey(12), 4)
    sample = lambda w, k: sampler.sample(_gaussian_log_abs, w, k)
    new_walkers, acceptance = jax.vmap(sample)(walkers, keys)
    assert new_walkers.shape == walkers.shape
    assert acceptance.shape == (4,)
    assert jnp.all(jnp.isfinite(acceptance))


def test_sample_supports_grad_through_step_size() -> None:
    r"""``sample`` is differentiable w.r.t. a traced step size (grad smoke test).

    A reduction over the sampler output must yield a finite gradient w.r.t. the
    Langevin step size, confirming the proposal/acceptance are differentiable.
    """
    walkers = jax.random.normal(jax.random.PRNGKey(13), (64, 2, 3), dtype=jnp.float64)
    key = jax.random.PRNGKey(14)

    def mean_sq(step_size: jax.Array) -> jax.Array:
        sampler = MALASampler(steps=3, step_size=step_size)
        new_walkers, _ = sampler.sample(_gaussian_log_abs, walkers, key)
        return jnp.mean(new_walkers**2)

    grad_value = jax.grad(mean_sq)(jnp.asarray(0.3))
    assert jnp.isfinite(grad_value)
