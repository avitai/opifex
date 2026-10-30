"""Split-conformal and CQR regression contracts.

References:

* Lei et al. 2018, "Distribution-Free Predictive Inference for Regression",
  JASA — split conformal absolute-residual score and finite-sample
  ``(n + 1) / n * (1 - alpha)`` rank correction.
* Romano, Patterson, Candes 2019, "Conformalized Quantile Regression",
  arXiv:1905.03222 — CQR score ``max(lo - y, y - hi)``.

Tests pin:

* Empirical coverage of the calibrated interval lands near ``1 - alpha`` on
  exchangeable synthetic regression data.
* ``predict`` before ``fit`` raises ``RuntimeError``.
* CQR widens the input quantile bounds when miscoverage is detected.
* Per-group split conformal recovers per-group coverage.
* Fitted state is a Pattern-B frozen ``flax.struct.dataclass``.
* JAX/NNX transform compatibility for the scoring + predict kernels.
"""

from __future__ import annotations

import dataclasses as dc
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_conformal():
    from opifex.uncertainty import conformal

    return conformal


def _import_regression_module():
    from opifex.uncertainty.conformal import regression

    return regression


def _split_dataset(
    n_calib: int = 512,
    n_test: int = 512,
    noise_std: float = 0.2,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    x_calib = rng.uniform(-1.0, 1.0, size=(n_calib,))
    y_calib = np.sin(np.pi * x_calib) + noise_std * rng.normal(size=(n_calib,))
    x_test = rng.uniform(-1.0, 1.0, size=(n_test,))
    y_test = np.sin(np.pi * x_test) + noise_std * rng.normal(size=(n_test,))
    return (
        jnp.asarray(x_calib),
        jnp.asarray(y_calib),
        jnp.asarray(x_test),
        jnp.asarray(y_test),
    )


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def test_absolute_residual_score_matches_definition() -> None:
    conformal = _import_conformal()
    predictions = jnp.array([0.0, 1.0, 2.0])
    targets = jnp.array([0.5, -0.5, 2.5])
    out = conformal.absolute_residual_score(predictions=predictions, targets=targets)
    assert bool(jnp.allclose(out, jnp.array([0.5, 1.5, 0.5])))


def test_cqr_score_matches_definition() -> None:
    """CQR score is ``max(lower - y, y - upper)`` per Romano et al. 2019."""
    conformal = _import_conformal()
    lower = jnp.array([0.0, 1.0, -1.0])
    upper = jnp.array([1.0, 2.0, 0.0])
    targets = jnp.array([0.5, 0.0, 1.0])
    # (0 - 0.5, 0.5 - 1) → max(-0.5, -0.5) = -0.5
    # (1 - 0, 0 - 2) → max(1, -2) = 1.0
    # (-1 - 1, 1 - 0) → max(-2, 1) = 1.0
    out = conformal.cqr_score(lower=lower, upper=upper, targets=targets)
    assert bool(jnp.allclose(out, jnp.array([-0.5, 1.0, 1.0])))


def test_conformal_quantile_applies_finite_sample_correction() -> None:
    """For n samples and target ``1 - alpha``, the quantile rank is
    ``ceil((n + 1)(1 - alpha)) / n``."""
    conformal = _import_conformal()
    scores = jnp.arange(1.0, 11.0)  # n=10, values 1..10
    out = float(conformal.conformal_quantile(scores=scores, alpha=0.1))
    # rank = ceil(11 * 0.9) / 10 = 10/10 = 1.0 → quantile at 1.0 → 10.0
    assert out == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Split conformal
# ---------------------------------------------------------------------------


def test_split_conformal_state_is_frozen() -> None:
    regression = _import_regression_module()
    state = regression.SplitConformalState(quantile=jnp.asarray(0.5), alpha=0.1)
    assert float(state.quantile) == pytest.approx(0.5)
    assert state.alpha == 0.1
    # struct.dataclass frozen → cannot mutate fields
    with pytest.raises(dc.FrozenInstanceError):
        state.quantile = jnp.asarray(0.6)  # type: ignore[misc]


def test_split_conformal_fit_then_predict_lands_near_target_coverage() -> None:
    regression = _import_regression_module()
    _, y_calib, _, y_test = _split_dataset(seed=1)
    # Deterministic "model": predict zero everywhere. Residuals = |y|.
    cal_preds = jnp.zeros_like(y_calib)
    test_preds = jnp.zeros_like(y_test)
    cp = regression.SplitConformalRegressor(alpha=0.1)
    state = cp.fit(predictions=cal_preds, targets=y_calib)
    interval = cp.with_state(state).predict(predictions=test_preds)
    covered = (y_test >= interval.lower) & (y_test <= interval.upper)
    empirical_coverage = float(jnp.mean(covered.astype(jnp.float32)))
    assert empirical_coverage == pytest.approx(0.9, abs=0.05)
    assert interval.coverage == 0.9
    assert interval.method == "split_conformal"


def test_split_conformal_predict_before_fit_raises() -> None:
    regression = _import_regression_module()
    cp = regression.SplitConformalRegressor(alpha=0.1)
    with pytest.raises(RuntimeError, match=r"(?i)(fit|calibrate)"):
        cp.predict(predictions=jnp.zeros((4,)))


def test_split_conformal_state_records_metadata() -> None:
    regression = _import_regression_module()
    _, y_calib, _, _ = _split_dataset(seed=2)
    cp = regression.SplitConformalRegressor(alpha=0.1)
    state = cp.fit(predictions=jnp.zeros_like(y_calib), targets=y_calib)
    md = dict(state.metadata)
    assert md["score_type"] == "absolute_residual"
    assert md["alpha"] == pytest.approx(0.1)
    assert int(md["calibration_size"]) == int(y_calib.shape[0])


def test_split_conformal_predict_is_jit_compatible() -> None:
    regression = _import_regression_module()
    _, y_calib, _, _ = _split_dataset(n_calib=128, seed=3)
    cp = regression.SplitConformalRegressor(alpha=0.1)
    state = cp.fit(predictions=jnp.zeros_like(y_calib), targets=y_calib)
    calibrator = cp.with_state(state)

    @jax.jit
    def jitted_predict(preds: jax.Array) -> jax.Array:
        out = calibrator.predict(predictions=preds)
        return out.upper - out.lower

    widths = jitted_predict(jnp.zeros((32,)))
    assert widths.shape == (32,)
    assert bool(jnp.all(jnp.isfinite(widths)))


# ---------------------------------------------------------------------------
# CQR
# ---------------------------------------------------------------------------


def test_cqr_widens_interval_when_undercovered() -> None:
    """CQR adjusts an under-covered quantile interval upward; coverage rises."""
    regression = _import_regression_module()
    rng = np.random.default_rng(4)
    n = 1024
    y = rng.standard_normal(n)
    # Narrow input intervals → low raw coverage.
    raw_lower = jnp.asarray(np.quantile(y, 0.40))
    raw_upper = jnp.asarray(np.quantile(y, 0.60))
    cal_lower = jnp.full((n,), float(raw_lower))
    cal_upper = jnp.full((n,), float(raw_upper))
    y_cal = jnp.asarray(y)
    cqr = regression.ConformalizedQuantileRegressor(alpha=0.1)
    state = cqr.fit(lower=cal_lower, upper=cal_upper, targets=y_cal)
    interval = cqr.with_state(state).predict(lower=cal_lower, upper=cal_upper)
    # Adjusted bounds should widen the original [40%, 60%] band.
    assert float(jnp.mean(interval.lower)) < float(raw_lower)
    assert float(jnp.mean(interval.upper)) > float(raw_upper)
    covered = (y_cal >= interval.lower) & (y_cal <= interval.upper)
    empirical = float(jnp.mean(covered.astype(jnp.float32)))
    assert empirical == pytest.approx(0.9, abs=0.05)


def test_cqr_predict_before_fit_raises() -> None:
    regression = _import_regression_module()
    cqr = regression.ConformalizedQuantileRegressor(alpha=0.1)
    with pytest.raises(RuntimeError, match=r"(?i)(fit|calibrate)"):
        cqr.predict(lower=jnp.zeros((4,)), upper=jnp.ones((4,)))


# ---------------------------------------------------------------------------
# Group-conditional split conformal
# ---------------------------------------------------------------------------


def test_grouped_split_conformal_recovers_per_group_coverage() -> None:
    regression = _import_regression_module()
    rng = np.random.default_rng(5)
    n_per_group = 512
    # Two heteroscedastic groups: group 0 noise σ=0.5, group 1 noise σ=2.0
    y_group0 = 0.5 * rng.standard_normal(n_per_group)
    y_group1 = 2.0 * rng.standard_normal(n_per_group)
    y_calib = jnp.asarray(np.concatenate([y_group0, y_group1]))
    groups = jnp.asarray(
        np.concatenate([np.zeros(n_per_group), np.ones(n_per_group)]).astype(np.int32)
    )
    preds = jnp.zeros_like(y_calib)
    cp = regression.GroupedSplitConformalRegressor(alpha=0.1)
    state = cp.fit(predictions=preds, targets=y_calib, groups=groups)
    interval = cp.with_state(state).predict(predictions=preds, groups=groups)
    covered = (y_calib >= interval.lower) & (y_calib <= interval.upper)
    # Per-group coverage near 0.9.
    g0_cov = float(jnp.mean(covered[:n_per_group].astype(jnp.float32)))
    g1_cov = float(jnp.mean(covered[n_per_group:].astype(jnp.float32)))
    assert g0_cov == pytest.approx(0.9, abs=0.05)
    assert g1_cov == pytest.approx(0.9, abs=0.05)
    # Group 1 (noisier) must produce wider intervals than group 0.
    widths = interval.upper - interval.lower
    w0 = float(jnp.mean(widths[:n_per_group]))
    w1 = float(jnp.mean(widths[n_per_group:]))
    assert w1 > w0 * 2.0


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_conformal_surface_includes_regression_components() -> None:
    conformal = _import_conformal()
    expected = {
        "absolute_residual_score",
        "cqr_score",
        "conformal_quantile",
        "SplitConformalRegressor",
        "SplitConformalState",
        "ConformalizedQuantileRegressor",
        "CQRState",
        "GroupedSplitConformalRegressor",
        "GroupedSplitConformalState",
    }
    missing = expected - set(dir(conformal))
    assert not missing, f"missing public conformal symbols: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Unattainable finite-sample rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.5, 0.2, 0.1, 0.05, 0.01])
def test_minimum_calibration_size_is_the_smallest_attainable_n(alpha: float) -> None:
    """``ceil((n + 1)(1 - alpha)) <= n`` first holds at ``minimum_calibration_size``."""
    conformal = _import_conformal()
    required = conformal.minimum_calibration_size(alpha=alpha)
    assert np.ceil((required + 1) * (1.0 - alpha)) <= required
    assert np.ceil(required * (1.0 - alpha)) > required - 1


@pytest.mark.parametrize("alpha", [0.2, 0.1, 0.05])
def test_conformal_quantile_is_infinite_below_the_minimum_size(alpha: float) -> None:
    """No threshold drawn from too few scores attains ``1 - alpha``, so it is ``+inf``.

    Returning ``max(scores)`` here would be finite, indistinguishable from a valid
    threshold, and short of the requested level by construction.
    """
    conformal = _import_conformal()
    required = conformal.minimum_calibration_size(alpha=alpha)
    scores = jnp.arange(1.0, float(required) + 1.0)

    for n in range(1, required):
        out = conformal.conformal_quantile(scores=scores[:n], alpha=alpha)
        assert bool(jnp.isinf(out)) and float(out) > 0.0, (alpha, n)

    at_minimum = conformal.conformal_quantile(scores=scores[:required], alpha=alpha)
    assert bool(jnp.isfinite(at_minimum)), (alpha, required)
    assert float(at_minimum) == pytest.approx(float(scores[:required].max()), abs=1e-6)


def test_conformal_quantile_coverage_holds_at_every_calibration_size() -> None:
    """Marginal coverage must reach ``1 - alpha`` for every ``n``, small ones included.

    The clamped rank gave exactly ``n / (n + 1)`` here — 0.667 at ``n=2`` against a
    requested 0.9 — which this asserts is gone. Coverage is marginal over the
    calibration draw, so each trial draws a fresh calibration set and a fresh test
    point; a single fixed calibration set would measure noise.
    """
    conformal = _import_conformal()
    alpha = 0.1
    trials = 20_000
    key = jax.random.key(11)
    quantile_of = jax.jit(jax.vmap(lambda s: conformal.conformal_quantile(scores=s, alpha=alpha)))

    for n in (2, 4, 8, 9, 19):
        key, k_cal, k_test = jax.random.split(key, 3)
        cal = jnp.abs(jax.random.normal(k_cal, (trials, n)))
        test = jnp.abs(jax.random.normal(k_test, (trials,)))
        covered = float(jnp.mean(test <= quantile_of(cal)))
        standard_error = (max(covered * (1.0 - covered), 1e-12) / trials) ** 0.5
        assert covered + 4.0 * standard_error >= 1.0 - alpha, (n, covered)


def test_conformal_quantile_is_jit_and_vmap_compatible_in_both_branches() -> None:
    """``n`` and ``alpha`` are static, so the ``+inf`` branch traces like the other."""
    conformal = _import_conformal()
    for n, expect_finite in ((4, False), (19, True)):
        scores = jnp.arange(1.0, float(n) + 1.0)
        jitted = jax.jit(lambda s: conformal.conformal_quantile(scores=s, alpha=0.1))
        assert bool(jnp.isfinite(jitted(scores))) is expect_finite, n
        batched = jax.vmap(lambda s: conformal.conformal_quantile(scores=s, alpha=0.1))
        out = batched(jnp.stack([scores, scores + 1.0]))
        assert out.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(out))) is expect_finite, n


def test_grouped_split_conformal_warns_and_names_underfilled_groups() -> None:
    """A group below the minimum size is named, recorded, and given an ``+inf`` width.

    The calibration set here holds 203 points, which looks ample until ``fit``
    partitions it: the three-point group cannot attain ``alpha=0.1`` on its own.
    """
    regression = _import_regression_module()
    conformal = _import_conformal()
    alpha = 0.1
    n_big, n_small = 200, 3
    rng = np.random.default_rng(7)
    targets = jnp.asarray(rng.standard_normal(n_big + n_small))
    groups = jnp.asarray(np.concatenate([np.zeros(n_big), np.ones(n_small)]).astype(np.int32))
    predictions = jnp.zeros_like(targets)

    cp = regression.GroupedSplitConformalRegressor(alpha=alpha)
    with pytest.warns(UserWarning, match=r"1 of 2 groups hold fewer: 1 \(n=3\)"):
        state = cp.fit(predictions=predictions, targets=targets, groups=groups)

    metadata = dict(state.metadata)
    assert metadata["min_group_size"] == conformal.minimum_calibration_size(alpha=alpha)
    assert metadata["underfilled_groups"] == ((1, 3),)
    assert bool(jnp.isfinite(state.quantiles[0]))
    assert bool(jnp.isinf(state.quantiles[1]))

    interval = cp.with_state(state).predict(
        predictions=jnp.zeros(2), groups=jnp.asarray([0, 1], dtype=jnp.int32)
    )
    widths = interval.upper - interval.lower
    assert bool(jnp.isfinite(widths[0]))
    assert bool(jnp.isinf(widths[1]))


def test_grouped_split_conformal_is_silent_when_every_group_is_large_enough() -> None:
    """The positive control: no warning and no ``+inf`` when the partition is adequate."""
    regression = _import_regression_module()
    rng = np.random.default_rng(8)
    n_per_group = 64
    targets = jnp.asarray(rng.standard_normal(2 * n_per_group))
    groups = jnp.asarray(
        np.concatenate([np.zeros(n_per_group), np.ones(n_per_group)]).astype(np.int32)
    )
    cp = regression.GroupedSplitConformalRegressor(alpha=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        state = cp.fit(predictions=jnp.zeros_like(targets), targets=targets, groups=groups)
    assert dict(state.metadata)["underfilled_groups"] == ()
    assert bool(jnp.all(jnp.isfinite(state.quantiles)))
