"""Likelihood log-density tests.

Backend-neutral likelihood log-density helpers. Container patterns:

* Likelihood spec metadata (``LikelihoodSpec``) → frozen+slotted hashable
  dataclass.
* Combined log-density containers carrying array values →
  ``@flax.struct.dataclass`` so they flow through ``jit`` / ``vmap``.
"""

from __future__ import annotations

import dataclasses
import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing import TraceCounter

from opifex.capabilities import require_uq_capability, UQ_CAPABILITIES
from opifex.uncertainty.likelihoods import (
    censored_gaussian_log_likelihood,
    gaussian_log_likelihood,
    heteroscedastic_gaussian_log_likelihood,
    laplace_log_likelihood,
    LIKELIHOOD_CAPABILITIES,
    LikelihoodSpec,
    mixture_log_likelihood,
    satisficing_censored_deficit,
    satisficing_censored_loss,
    student_t_log_likelihood,
    tobit_nll,
)


_EPS32 = float(np.finfo(np.float32).eps)
_EPS64 = float(np.finfo(np.float64).eps)


def _decimal_pi() -> Decimal:
    """Compute pi to the active Decimal precision via Machin's formula."""

    def _atan_inv(m: int) -> Decimal:
        inv_m = Decimal(1) / Decimal(m)
        inv_m2 = inv_m * inv_m
        term = inv_m
        total = term
        k = 1
        while term != 0:
            term = -term * inv_m2
            step = term / Decimal(2 * k + 1)
            if step == 0:
                break
            total += step
            k += 1
        return total

    return Decimal(4) * (Decimal(4) * _atan_inv(5) - _atan_inv(239))


def _decimal_erfc(x: Decimal, pi_val: Decimal) -> Decimal:
    """100+-digit ``erfc(x)`` for ``|x| <= 25`` via positive-term series in 340-digit context."""
    if x == 0:
        return Decimal(1)
    ax = abs(x)
    two_x2 = Decimal(2) * ax * ax
    term = ax
    series = term
    k = 1
    while True:
        term = term * two_x2 / Decimal(2 * k + 1)
        if term < Decimal("1e-330"):
            break
        series += term
        k += 1
    erf_ax = (Decimal(2) / pi_val.sqrt()) * (-ax * ax).exp() * series
    return Decimal(1) - erf_ax if x > 0 else Decimal(1) + erf_ax


def _oracle_censored_value_and_grads(
    lower: float,
    upper: float,
    mean: float,
    scale: float,
    *,
    exact: bool | None = None,
) -> tuple[float, float, float]:
    """100+-digit Decimal reference for ``(log_prob, d/dmean, d/dscale)`` on ``z in [-30, 30]``."""
    with localcontext() as ctx:
        ctx.prec = 340
        pi_val = _decimal_pi()
        sqrt_two = Decimal(2).sqrt()
        sqrt_two_pi = (Decimal(2) * pi_val).sqrt()
        mu = Decimal.from_float(mean)
        sig = Decimal.from_float(scale)

        is_exact = (
            exact
            if exact is not None
            else (math.isfinite(lower) and math.isfinite(upper) and lower == upper)
        )
        if is_exact:
            y = Decimal.from_float(lower)
            z = (y - mu) / sig
            ll = -(Decimal("0.5") * (Decimal(2) * pi_val).ln() + sig.ln() + Decimal("0.5") * z * z)
            grad_mu = z / sig
            grad_sig = (z * z - Decimal(1)) / sig
            return float(ll), float(grad_mu), float(grad_sig)

        if math.isinf(lower) and lower < 0 and math.isinf(upper) and upper > 0:
            return 0.0, 0.0, 0.0

        def _phi(z_val: Decimal) -> Decimal:
            return (-Decimal("0.5") * z_val * z_val).exp() / sqrt_two_pi

        def _cdf(z_val: Decimal) -> Decimal:
            return Decimal("0.5") * _decimal_erfc(-z_val / sqrt_two, pi_val)

        def _sf(z_val: Decimal) -> Decimal:
            return Decimal("0.5") * _decimal_erfc(z_val / sqrt_two, pi_val)

        if math.isinf(lower) and lower < 0:
            hi = Decimal.from_float(upper)
            z_hi = (hi - mu) / sig
            prob = _cdf(z_hi)
            phi_hi = _phi(z_hi)
            grad_mu = -phi_hi / (sig * prob)
            grad_sig = -(z_hi * phi_hi) / (sig * prob)
            return float(prob.ln()), float(grad_mu), float(grad_sig)

        if math.isinf(upper) and upper > 0:
            lo = Decimal.from_float(lower)
            z_lo = (lo - mu) / sig
            prob = _sf(z_lo)
            phi_lo = _phi(z_lo)
            grad_mu = phi_lo / (sig * prob)
            grad_sig = (z_lo * phi_lo) / (sig * prob)
            return float(prob.ln()), float(grad_mu), float(grad_sig)

        lo = Decimal.from_float(lower)
        hi = Decimal.from_float(upper)
        z_lo = (lo - mu) / sig
        z_hi = (hi - mu) / sig
        prob = _sf(z_lo) - _sf(z_hi) if (z_lo + z_hi) > 0 else _cdf(z_hi) - _cdf(z_lo)
        phi_lo = _phi(z_lo)
        phi_hi = _phi(z_hi)
        grad_mu = (phi_lo - phi_hi) / (sig * prob)
        grad_sig = (z_lo * phi_lo - z_hi * phi_hi) / (sig * prob)
        return float(prob.ln()), float(grad_mu), float(grad_sig)


def test_gaussian_log_likelihood_matches_hand_computed_value() -> None:
    # log N(y=1; mu=0, sigma=1) = -0.5 * (log(2*pi) + 1)
    y = jnp.array([1.0])
    mean = jnp.array([0.0])
    scale = jnp.array([1.0])
    expected = float(-0.5 * (jnp.log(2 * jnp.pi) + 1.0))
    result = float(gaussian_log_likelihood(y, mean=mean, scale=scale)[0])
    assert result == pytest.approx(expected, rel=1e-5)


def test_gaussian_log_likelihood_rejects_non_positive_scale() -> None:
    y = jnp.zeros(1)
    mean = jnp.zeros(1)
    with pytest.raises(ValueError, match=r"scale"):
        gaussian_log_likelihood(y, mean=mean, scale=jnp.array([0.0]))
    with pytest.raises(ValueError, match=r"scale"):
        gaussian_log_likelihood(y, mean=mean, scale=jnp.array([-1.0]))


def test_heteroscedastic_gaussian_log_likelihood_allows_per_point_scale() -> None:
    y = jnp.array([1.0, 2.0])
    mean = jnp.array([0.0, 2.5])
    scale = jnp.array([1.0, 0.5])
    result = heteroscedastic_gaussian_log_likelihood(y, mean=mean, scale=scale)
    expected_0 = float(-0.5 * (jnp.log(2 * jnp.pi) + 1.0))
    expected_1 = float(-0.5 * (jnp.log(2 * jnp.pi) + jnp.log(0.25) + (2.0 - 2.5) ** 2 / 0.25))
    assert float(jnp.sum(result)) == pytest.approx(expected_0 + expected_1, rel=1e-5)


def test_laplace_log_likelihood_matches_closed_form() -> None:
    # log Laplace(y=0; mu=0, b=1) = -log(2)
    y = jnp.array([0.0])
    location = jnp.array([0.0])
    scale = jnp.array([1.0])
    expected = float(-jnp.log(jnp.array(2.0)))
    result = float(laplace_log_likelihood(y, location=location, scale=scale)[0])
    assert result == pytest.approx(expected, rel=1e-5)


def test_student_t_log_likelihood_rejects_non_positive_df() -> None:
    y = jnp.zeros(1)
    location = jnp.zeros(1)
    scale = jnp.ones(1)
    with pytest.raises(ValueError, match=r"df"):
        student_t_log_likelihood(y, df=0.0, location=location, scale=scale)
    with pytest.raises(ValueError, match=r"df"):
        student_t_log_likelihood(y, df=-1.0, location=location, scale=scale)


def test_student_t_log_likelihood_approaches_gaussian_for_large_df() -> None:
    # df=1000 is "large enough" without hitting float32 gammaln precision loss
    # (df=1e6 collapses the gammaln difference to its float32 nearest representable value).
    y = jnp.array([0.3])
    location = jnp.array([0.0])
    scale = jnp.array([1.0])
    student = float(student_t_log_likelihood(y, df=1000.0, location=location, scale=scale)[0])
    gaussian = float(gaussian_log_likelihood(y, mean=location, scale=scale)[0])
    assert student == pytest.approx(gaussian, abs=2e-3)


def test_mixture_log_likelihood_rejects_invalid_weights() -> None:
    y = jnp.array([0.0])
    weights = jnp.array([0.5, 0.4])  # does not sum to 1
    means = jnp.array([0.0, 1.0])
    scales = jnp.array([1.0, 1.0])
    with pytest.raises(ValueError, match=r"weight"):
        mixture_log_likelihood(y, weights=weights, means=means, scales=scales)


def test_mixture_log_likelihood_matches_single_component_gaussian() -> None:
    y = jnp.array([0.5])
    weights = jnp.array([1.0])
    means = jnp.array([0.0])
    scales = jnp.array([1.0])
    expected = float(gaussian_log_likelihood(y, mean=means, scale=scales)[0])
    result = float(mixture_log_likelihood(y, weights=weights, means=means, scales=scales)[0])
    assert result == pytest.approx(expected, rel=1e-5)


def test_gaussian_log_likelihood_is_jit_and_grad_compatible() -> None:
    y = jnp.array([1.0, 2.0])
    mean = jnp.array([0.0, 1.5])
    scale = jnp.array([1.0, 0.5])
    jitted = jax.jit(lambda m: jnp.sum(gaussian_log_likelihood(y, mean=m, scale=scale)))
    assert jnp.isfinite(jitted(mean))
    grads = jax.grad(jitted)(mean)
    assert grads.shape == mean.shape
    assert jnp.all(jnp.isfinite(grads))


def test_likelihood_spec_is_pattern_a_frozen_dataclass() -> None:
    spec = LikelihoodSpec(
        name="gaussian",
        family="continuous",
        parameter_names=("mean", "scale"),
        supports_heteroscedastic=False,
    )
    assert dataclasses.is_dataclass(LikelihoodSpec)
    assert hasattr(LikelihoodSpec, "__slots__")
    assert not hasattr(spec, "__dict__")
    assert hash(spec) is not None


class TestCensoredGaussianLogLikelihood:
    """4-branch censored Gaussian log-likelihood value and gradient precision across z in [-30, 30]."""

    _CASES: tuple[tuple[float, float, float, float, bool | None], ...] = (
        # Exact observations across central and tail z
        (1.25, 1.25, 0.0, 1.0, True),
        (-18.0, -18.0, 2.0, 0.75, None),
        (27.5, 27.5, -2.5, 1.0, None),
        # Left-censored: (-inf, upper] across z_hi in [-30, +30]
        (-math.inf, -30.0, 0.0, 1.0, False),
        (-math.inf, -15.0, 0.0, 1.0, False),
        (-math.inf, -2.0, 0.0, 1.0, False),
        (-math.inf, 0.0, 0.0, 1.0, False),
        (-math.inf, 5.0, 0.0, 1.0, False),
        (-math.inf, 30.0, 0.0, 1.0, False),
        # Right-censored: [lower, +inf) across z_lo in [-30, +30]
        (-30.0, math.inf, 0.0, 1.0, False),
        (-5.0, math.inf, 0.0, 1.0, False),
        (0.0, math.inf, 0.0, 1.0, False),
        (2.0, math.inf, 0.0, 1.0, False),
        (15.0, math.inf, 0.0, 1.0, False),
        (30.0, math.inf, 0.0, 1.0, False),
        # Uninformative: (-inf, +inf)
        (-math.inf, math.inf, 1.5, 0.8, False),
        # Two-sided intervals: central, asymmetric, extreme left/right tails, and narrow intervals
        (-1.0, 1.0, 0.0, 1.0, False),
        (-0.2, 4.0, 0.0, 1.0, False),
        (-4.0, 0.2, 0.0, 1.0, False),
        (15.0, 16.0, 0.0, 1.0, False),
        (-16.0, -15.0, 0.0, 1.0, False),
        (29.0, 30.0, 0.0, 1.0, False),
        (-30.0, -29.0, 0.0, 1.0, False),
        (5.0, float(np.float32(5.0) + np.float32(1e-5)), 0.0, 1.0, False),
        (-25.0, float(np.float32(-25.0) + np.float32(1e-4)), 0.0, 1.0, False),
        (25.0, float(np.float32(25.0) + np.float32(1e-4)), 0.0, 1.0, False),
        # Shifted narrow interval where |mean| >> upper - lower
        (5.0, float(np.float32(5.0) + np.float32(1e-4)), -500.0, 20.0, False),
    )

    def test_float32_values_and_gradients_match_100_digit_oracle(self) -> None:
        lower = jnp.asarray([c[0] for c in self._CASES], dtype=jnp.float32)
        upper = jnp.asarray([c[1] for c in self._CASES], dtype=jnp.float32)
        mean = jnp.asarray([c[2] for c in self._CASES], dtype=jnp.float32)
        scale = jnp.asarray([c[3] for c in self._CASES], dtype=jnp.float32)
        exact_mask = jnp.asarray(
            [
                c[4]
                if c[4] is not None
                else (math.isfinite(c[0]) and math.isfinite(c[1]) and c[0] == c[1])
                for c in self._CASES
            ],
            dtype=jnp.bool_,
        )

        values = censored_gaussian_log_likelihood(
            lower, upper, mean=mean, scale=scale, exact_mask=exact_mask
        )
        grad_mean, grad_scale = jax.grad(
            lambda m, s: jnp.sum(
                censored_gaussian_log_likelihood(
                    lower, upper, mean=m, scale=s, exact_mask=exact_mask
                )
            ),
            argnums=(0, 1),
        )(mean, scale)

        assert bool(jnp.all(jnp.isfinite(values)))
        assert bool(jnp.all(jnp.isfinite(grad_mean)))
        assert bool(jnp.all(jnp.isfinite(grad_scale)))

        for idx, (*_, ex_f) in enumerate(self._CASES):
            lo32 = float(lower[idx])
            hi32 = float(upper[idx])
            mu32 = float(mean[idx])
            sig32 = float(scale[idx])
            ref_val, ref_g_mu, ref_g_sig = _oracle_censored_value_and_grads(
                lo32, hi32, mu32, sig32, exact=ex_f
            )
            val_err = abs(float(values[idx]) - ref_val)
            val_tol = 8.0 * _EPS32 * max(1.0, abs(ref_val))
            assert val_err <= val_tol, (idx, self._CASES[idx], float(values[idx]), ref_val, val_err)

            g_mu_err = abs(float(grad_mean[idx]) - ref_g_mu)
            g_mu_tol = 16.0 * _EPS32 * max(1.0, abs(ref_g_mu))
            assert g_mu_err <= g_mu_tol, (
                idx,
                self._CASES[idx],
                float(grad_mean[idx]),
                ref_g_mu,
                g_mu_err,
            )

            g_sig_err = abs(float(grad_scale[idx]) - ref_g_sig)
            g_sig_tol = 16.0 * _EPS32 * max(1.0, abs(ref_g_sig))
            assert g_sig_err <= g_sig_tol, (
                idx,
                self._CASES[idx],
                float(grad_scale[idx]),
                ref_g_sig,
                g_sig_err,
            )

    @pytest.mark.usefixtures("float64")
    def test_float64_values_and_central_difference_gradients_match_oracle(self) -> None:
        lower = jnp.asarray([c[0] for c in self._CASES], dtype=jnp.float64)
        upper = jnp.asarray([c[1] for c in self._CASES], dtype=jnp.float64)
        mean = jnp.asarray([c[2] for c in self._CASES], dtype=jnp.float64)
        scale = jnp.asarray([c[3] for c in self._CASES], dtype=jnp.float64)

        values = censored_gaussian_log_likelihood(lower, upper, mean=mean, scale=scale)
        grad_mean, grad_scale = jax.grad(
            lambda m, s: jnp.sum(censored_gaussian_log_likelihood(lower, upper, mean=m, scale=s)),
            argnums=(0, 1),
        )(mean, scale)

        for idx, (lo_f, hi_f, mu_f, sig_f, ex_f) in enumerate(self._CASES):
            ref_val, ref_g_mu, ref_g_sig = _oracle_censored_value_and_grads(
                lo_f, hi_f, mu_f, sig_f, exact=ex_f
            )
            assert abs(float(values[idx]) - ref_val) <= 16.0 * _EPS64 * max(1.0, abs(ref_val))
            assert abs(float(grad_mean[idx]) - ref_g_mu) <= 32.0 * _EPS64 * max(1.0, abs(ref_g_mu))
            assert abs(float(grad_scale[idx]) - ref_g_sig) <= 32.0 * _EPS64 * max(
                1.0, abs(ref_g_sig)
            )

        # Central finite-difference verification on a mixed batch (exact + left + right + interval)
        step = 1e-6
        fd_mean = (
            censored_gaussian_log_likelihood(lower, upper, mean=mean + step, scale=scale)
            - censored_gaussian_log_likelihood(lower, upper, mean=mean - step, scale=scale)
        ) / (2.0 * step)
        assert np.allclose(np.asarray(grad_mean), np.asarray(fd_mean), rtol=1e-7, atol=1e-8)

    @pytest.mark.parametrize("z_mid", [-20.0, -3.0, 0.0, 3.0, 20.0])
    def test_interval_branch_is_continuous_across_quadrature_boundary(self, z_mid: float) -> None:
        """Adjacent float32 widths straddling ``w * (1 + |z_mid|) == 0.5`` agree smoothly."""
        critical_width = np.float32(0.5 / (1.0 + abs(z_mid)))
        w_below = np.nextafter(critical_width, np.float32(0.0))
        w_above = np.nextafter(critical_width, np.float32(np.inf))
        widths = jnp.asarray([w_below, w_above], dtype=jnp.float32)
        mid = jnp.full((2,), z_mid, dtype=jnp.float32)
        lower = mid - 0.5 * widths
        upper = mid + 0.5 * widths
        vals = np.asarray(
            censored_gaussian_log_likelihood(
                lower,
                upper,
                mean=jnp.zeros(2, dtype=jnp.float32),
                scale=jnp.ones(2, dtype=jnp.float32),
                exact_mask=jnp.zeros(2, dtype=jnp.bool_),
            ),
            dtype=np.float64,
        )
        slope_bound = 2.0 / float(critical_width) + abs(z_mid) + 2.0
        tol = slope_bound * float(w_above - w_below) + 16.0 * _EPS32 * max(1.0, abs(vals[0]))
        assert abs(vals[1] - vals[0]) <= tol, (z_mid, vals, tol)

    def test_default_exact_mask_uses_strict_equality_not_isclose(self) -> None:
        """A narrow interval within default ``jnp.isclose`` tolerances stays an interval."""
        lo = jnp.array([5.0, 5.0], dtype=jnp.float32)
        hi = jnp.array([5.0, float(np.nextafter(np.float32(5.0), np.float32(np.inf)))])
        out = censored_gaussian_log_likelihood(lo, hi, mean=jnp.full(2, 5.0), scale=jnp.ones(2))
        # Entry 0 is an exact log-density (~ -0.9189); entry 1 is a 1-ULP interval log-mass (~ -15.3)
        assert float(out[0]) == pytest.approx(-0.5 * math.log(2.0 * math.pi), rel=1e-6)
        assert float(out[1]) < -10.0

    def test_collapsed_non_exact_interval_remains_finite_with_finite_gradients(self) -> None:
        """Explicit ``exact_mask=False`` with ``lower == upper`` uses tiny width without NaNs."""
        lo = jnp.array([3.0], dtype=jnp.float32)
        hi = jnp.array([3.0], dtype=jnp.float32)
        mask = jnp.array([False])
        val, (g_mu, g_sig) = jax.value_and_grad(
            lambda m, s: censored_gaussian_log_likelihood(lo, hi, mean=m, scale=s, exact_mask=mask)[
                0
            ],
            argnums=(0, 1),
        )(jnp.array([2.5], dtype=jnp.float32), jnp.array([0.8], dtype=jnp.float32))
        assert bool(jnp.isfinite(val))
        assert float(val) < -80.0
        assert bool(jnp.all(jnp.isfinite(g_mu)))
        assert bool(jnp.all(jnp.isfinite(g_sig)))

    def test_tobit_nll_negates_censored_log_likelihood(self) -> None:
        lo = jnp.array([1.0, -jnp.inf, 2.0, 0.5])
        hi = jnp.array([1.0, 0.0, jnp.inf, 1.5])
        mean = jnp.array([0.8, 0.2, 1.5, 1.0])
        scale = jnp.full(4, 0.7)
        nll = tobit_nll(lo, hi, mean=mean, scale=scale)
        ll = censored_gaussian_log_likelihood(lo, hi, mean=mean, scale=scale)
        assert jnp.allclose(nll, -ll, rtol=0.0, atol=0.0)

    def test_rejects_invalid_bounds_and_non_boolean_mask_eagerly(self) -> None:
        mean = jnp.zeros(1)
        scale = jnp.ones(1)
        with pytest.raises(ValueError, match=r"lower <= upper"):
            censored_gaussian_log_likelihood(
                jnp.array([2.0]), jnp.array([1.0]), mean=mean, scale=scale
            )
        with pytest.raises(ValueError, match=r"lower"):
            censored_gaussian_log_likelihood(
                jnp.array([jnp.inf]), jnp.array([jnp.inf]), mean=mean, scale=scale
            )
        with pytest.raises(ValueError, match=r"upper"):
            censored_gaussian_log_likelihood(
                jnp.array([-jnp.inf]), jnp.array([-jnp.inf]), mean=mean, scale=scale
            )
        with pytest.raises(ValueError, match=r"exact_mask"):
            censored_gaussian_log_likelihood(
                jnp.array([0.0]),
                jnp.array([1.0]),
                mean=mean,
                scale=scale,
                exact_mask=jnp.array([1]),
            )

    def test_jit_and_vmap_compile_once_and_match_eager(self) -> None:
        lo = jnp.array([[1.0, -jnp.inf, 2.0, 0.0], [0.5, -1.0, 3.0, -jnp.inf]])
        hi = jnp.array([[1.0, -0.5, jnp.inf, 1.0], [0.5, 1.0, jnp.inf, jnp.inf]])
        mean = jnp.zeros((2, 4))
        scale = jnp.ones((2, 4))

        counter = TraceCounter()
        compiled = jax.jit(
            counter.wrap(
                jax.vmap(
                    lambda l_i, u_i, m_i, s_i: censored_gaussian_log_likelihood(
                        l_i, u_i, mean=m_i, scale=s_i
                    )
                )
            )
        )
        with counter.expect(new_traces=1):
            out1 = compiled(lo, hi, mean, scale)
        with counter.expect(new_traces=0):
            out2 = compiled(lo, hi + 0.25, mean + 0.1, scale)

        eager1 = censored_gaussian_log_likelihood(lo, hi, mean=mean, scale=scale)
        assert jnp.allclose(out1, eager1, rtol=4.0 * _EPS32, atol=4.0 * _EPS32)
        assert bool(jnp.all(jnp.isfinite(out2)))


class TestSatisficingCensoredLoss:
    """Probability-satisficing censored loss (Mitropoulou et al., CensorADMET)."""

    def test_stops_pushing_once_interval_probability_reaches_tau(self) -> None:
        """Right-censored y >= 0: once P(Y >= 0) >= tau, deficit and gradient vanish."""
        lo = jnp.array([0.0])
        hi = jnp.array([jnp.inf])
        scale = jnp.array([1.0])

        # At mean = 3.0, P(Y >= 0) = Phi(3.0) = 0.99865 > tau = 0.9 -> zero deficit & zero grad
        satisfied_mean = jnp.array([3.0])
        deficit_sat, grad_sat = jax.value_and_grad(
            lambda m: satisficing_censored_deficit(lo, hi, mean=m, scale=scale, tau=0.9)[0]
        )(satisfied_mean)
        assert float(deficit_sat) == 0.0
        assert float(grad_sat[0]) == 0.0

        # At mean = 0.0, P(Y >= 0) = 0.5 < tau = 0.9 -> deficit = (0.9 - 0.5)^2 = 0.16
        violated_mean = jnp.array([0.0])
        deficit_viol, grad_viol = jax.value_and_grad(
            lambda m: satisficing_censored_deficit(lo, hi, mean=m, scale=scale, tau=0.9)[0]
        )(violated_mean)
        assert float(deficit_viol) == pytest.approx(0.16, rel=1e-6)
        # Pushing mean upward increases P(Y >= 0), so d(deficit)/d(mean) < 0
        assert float(grad_viol[0]) < 0.0
        expected_grad = -2.0 * (0.9 - 0.5) * (1.0 / math.sqrt(2.0 * math.pi))
        assert float(grad_viol[0]) == pytest.approx(expected_grad, rel=1e-5)

    def test_combined_loss_uses_exact_nll_on_exact_rows_and_deficit_on_censored_rows(self) -> None:
        lo = jnp.array([1.5, 2.0, -jnp.inf])
        hi = jnp.array([1.5, jnp.inf, -1.0])
        mean = jnp.array([1.0, 2.0, 0.0])
        scale = jnp.array([0.5, 1.0, 1.0])

        loss, grads = jax.value_and_grad(
            lambda m: jnp.sum(
                satisficing_censored_loss(lo, hi, mean=m, scale=scale, tau=0.8, censored_weight=2.5)
            )
        )(mean)
        per_row = satisficing_censored_loss(
            lo, hi, mean=mean, scale=scale, tau=0.8, censored_weight=2.5
        )
        expected_exact = float(-gaussian_log_likelihood(lo[:1], mean=mean[:1], scale=scale[:1])[0])
        expected_right = 2.5 * (0.8 - 0.5) ** 2
        assert float(per_row[0]) == pytest.approx(expected_exact, rel=1e-5)
        assert float(per_row[1]) == pytest.approx(expected_right, rel=1e-5)
        assert bool(jnp.isfinite(loss))
        assert bool(jnp.all(jnp.isfinite(grads)))

    def test_validates_tau_and_censored_weight(self) -> None:
        lo = jnp.array([0.0])
        hi = jnp.array([1.0])
        mean = jnp.zeros(1)
        scale = jnp.ones(1)
        with pytest.raises(ValueError, match=r"tau"):
            satisficing_censored_deficit(lo, hi, mean=mean, scale=scale, tau=0.0)
        with pytest.raises(ValueError, match=r"tau"):
            satisficing_censored_deficit(lo, hi, mean=mean, scale=scale, tau=1.1)
        with pytest.raises(ValueError, match=r"censored_weight"):
            satisficing_censored_loss(lo, hi, mean=mean, scale=scale, censored_weight=-0.5)


class TestLikelihoodCapabilities:
    """Censored likelihood capabilities are registered in UQ_CAPABILITIES."""

    def test_likelihood_capabilities_are_in_global_catalogue(self) -> None:
        assert "likelihood:censored_gaussian" in LIKELIHOOD_CAPABILITIES
        assert "likelihood:satisficing_censored" in LIKELIHOOD_CAPABILITIES
        for name, cap in LIKELIHOOD_CAPABILITIES.items():
            assert UQ_CAPABILITIES[name] is cap
            assert require_uq_capability(name) is cap
            assert cap.native_jax_kernel is True
            assert cap.native_distributional is True
