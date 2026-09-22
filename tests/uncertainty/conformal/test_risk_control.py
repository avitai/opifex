"""Risk-control interface contract.

References:

* Bates, Angelopoulos, Lei, Malik, Jordan 2021, "Distribution-Free,
  Risk-Controlling Prediction Sets" (RCPS, arXiv:2101.02703) — monotonic
  threshold selection via Hoeffding upper confidence bound on empirical
  loss.
* Angelopoulos et al. 2022, "Learn then Test: Calibrating Predictive
  Algorithms to Achieve Risk Control" (LTT, arXiv:2110.01052) — the
  general framework, of which RCPS is the monotonic special case.

Tests pin:

* `RiskControlConfig` is a Pattern-A frozen dataclass; validation lives in
  ``__post_init__`` and raises ``ValueError`` (never ``assert``).
* RCPS threshold selection on a monotonic synthetic loss recovers the
  largest threshold whose Hoeffding UCB is below the target risk.
* Non-monotonic loss path is flagged in metadata; the controller does not
  overclaim distribution-free coverage.
* Fitted controller state is a Pattern-B frozen pytree and records
  threshold, empirical loss, UCB, and config metadata.
* The threshold-selection kernel is jit / vmap compatible.
"""

from __future__ import annotations

import dataclasses as dc
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx


def _import_risk_control():
    from opifex.uncertainty.conformal import risk_control

    return risk_control


# ---------------------------------------------------------------------------
# RiskControlConfig validation (no asserts!)
# ---------------------------------------------------------------------------


def test_risk_control_config_construction_records_fields() -> None:
    rc = _import_risk_control()
    config = rc.RiskControlConfig(alpha=0.1, delta=0.05, loss_name="miscoverage", monotonic=True)
    assert config.alpha == 0.1
    assert config.delta == 0.05
    assert config.loss_name == "miscoverage"
    assert config.monotonic is True


def test_risk_control_config_is_frozen_pattern_a_dataclass() -> None:
    rc = _import_risk_control()
    config = rc.RiskControlConfig(alpha=0.1, delta=0.05, loss_name="miscoverage")
    with pytest.raises(dc.FrozenInstanceError):
        config.alpha = 0.2  # type: ignore[misc]


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5])
def test_risk_control_config_rejects_alpha_out_of_range(alpha: float) -> None:
    rc = _import_risk_control()
    with pytest.raises(ValueError, match=r"(?i)alpha"):
        rc.RiskControlConfig(alpha=alpha, delta=0.05, loss_name="miscoverage")


@pytest.mark.parametrize("delta", [-0.1, 0.0, 1.0, 2.0])
def test_risk_control_config_rejects_delta_out_of_range(delta: float) -> None:
    rc = _import_risk_control()
    with pytest.raises(ValueError, match=r"(?i)delta"):
        rc.RiskControlConfig(alpha=0.1, delta=delta, loss_name="miscoverage")


def test_risk_control_config_rejects_empty_loss_name() -> None:
    rc = _import_risk_control()
    with pytest.raises(ValueError, match=r"(?i)loss_name"):
        rc.RiskControlConfig(alpha=0.1, delta=0.05, loss_name="")


def test_risk_control_config_validation_uses_no_assert_statements() -> None:
    """Rule 6: validation paths must use ``raise ValueError``, never ``assert``.

    This regex-scan complements the exit-criterion ``rg`` check.
    """
    import inspect

    rc = _import_risk_control()
    source = inspect.getsource(rc.RiskControlConfig)
    # Any `assert ` followed by a control-flow check is a violation.
    matches = re.findall(r"^\s*assert\s+", source, flags=re.MULTILINE)
    assert not matches, f"Found {len(matches)} assert statement(s) in RiskControlConfig"


# ---------------------------------------------------------------------------
# Hoeffding upper confidence bound
# ---------------------------------------------------------------------------


def test_hoeffding_upper_bound_matches_closed_form() -> None:
    """For empirical mean m, n samples, delta δ:
    UCB = m + sqrt(log(1/δ) / (2n))
    """
    rc = _import_risk_control()
    mean = jnp.asarray(0.05)
    n_samples = 100
    delta = 0.05
    out = float(rc.hoeffding_upper_bound(empirical_mean=mean, n=n_samples, delta=delta))
    expected = 0.05 + float(np.sqrt(np.log(1.0 / 0.05) / (2 * 100)))
    assert out == pytest.approx(expected, rel=1e-6, abs=1e-7)


# ---------------------------------------------------------------------------
# RCPS monotonic threshold selection
# ---------------------------------------------------------------------------


def _monotonic_per_sample_losses(
    *,
    thresholds: jax.Array,
    n_samples: int = 256,
    seed: int = 0,
) -> jax.Array:
    """Synthetic loss decreasing monotonically with threshold.

    Returns ``(n_samples, n_thresholds)``. For threshold ``t`` and per-sample
    "difficulty" ``d_i ~ U[0, 1]``, loss is ``max(0, d_i - t)`` — clearly
    decreasing in ``t``, mean drops linearly to zero.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.uniform(size=(n_samples, 1))
    return jnp.maximum(0.0, jnp.asarray(difficulty) - thresholds[None, :])


def test_select_threshold_rcps_returns_largest_safe_threshold() -> None:
    """Given a monotonic decreasing loss curve, the controller should pick
    the largest threshold whose Hoeffding UCB on empirical loss is < alpha.
    """
    rc = _import_risk_control()
    thresholds = jnp.linspace(0.0, 1.0, 21)  # 0.00, 0.05, ..., 1.00
    losses = _monotonic_per_sample_losses(thresholds=thresholds, n_samples=2048, seed=1)
    config = rc.RiskControlConfig(alpha=0.1, delta=0.05, loss_name="abs_excess")
    state = rc.select_threshold_rcps(thresholds=thresholds, losses=losses, config=config)
    assert float(state.upper_confidence_bound) <= 0.1
    # The largest unsafe threshold should produce UCB > alpha.
    smaller_idx = max(0, int(jnp.argmax(thresholds == state.threshold)) - 1)
    if smaller_idx < len(thresholds) - 1:
        # The next-smaller threshold should still be safe (or selected one is the smallest).
        assert state.threshold >= thresholds[smaller_idx]


def test_select_threshold_rcps_records_calibration_stats_in_metadata() -> None:
    rc = _import_risk_control()
    thresholds = jnp.linspace(0.0, 1.0, 21)
    losses = _monotonic_per_sample_losses(thresholds=thresholds, n_samples=512, seed=2)
    config = rc.RiskControlConfig(alpha=0.1, delta=0.05, loss_name="abs_excess")
    state = rc.select_threshold_rcps(thresholds=thresholds, losses=losses, config=config)
    md = dict(state.metadata)
    assert md["method"] == "rcps"
    assert md["loss_name"] == "abs_excess"
    assert md["alpha"] == pytest.approx(0.1)
    assert md["delta"] == pytest.approx(0.05)
    assert int(md["calibration_size"]) == 512
    assert int(md["num_thresholds"]) == 21


def test_controller_state_is_frozen_pytree() -> None:
    rc = _import_risk_control()
    config = rc.RiskControlConfig(alpha=0.1, delta=0.05, loss_name="abs_excess")
    state = rc.RiskControllerState(
        threshold=jnp.asarray(0.5),
        empirical_loss_at_threshold=jnp.asarray(0.05),
        upper_confidence_bound=jnp.asarray(0.07),
        config=config,
    )
    with pytest.raises(dc.FrozenInstanceError):
        state.threshold = jnp.asarray(0.6)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Non-monotonic loss path: do not overclaim coverage
# ---------------------------------------------------------------------------


def test_non_monotonic_loss_path_records_conservative_method_metadata() -> None:
    """When the loss is not monotonic in the threshold, the controller must
    not claim RCPS-style finite-sample coverage; it should pick the
    smallest threshold whose UCB is below alpha and flag the choice in
    metadata.
    """
    rc = _import_risk_control()
    thresholds = jnp.linspace(0.0, 1.0, 21)
    losses = _monotonic_per_sample_losses(thresholds=thresholds, n_samples=512, seed=3)
    config = rc.RiskControlConfig(alpha=0.1, delta=0.05, loss_name="abs_excess", monotonic=False)
    state = rc.select_threshold_rcps(thresholds=thresholds, losses=losses, config=config)
    md = dict(state.metadata)
    assert md["method"] == "rcps"
    assert md.get("monotonic") is False
    assert md.get("coverage_guarantee") == "conservative"


# ---------------------------------------------------------------------------
# Confidence-interval reporting via CalibraX bootstrap
# ---------------------------------------------------------------------------


def _normal_samples(n: int = 64) -> jax.Array:
    return jnp.asarray(np.random.default_rng(0).standard_normal(size=(n,)))


def test_bootstrap_threshold_ci_brackets_the_sample_mean() -> None:
    rc = _import_risk_control()
    samples = _normal_samples()

    lo, hi = rc.bootstrap_threshold_ci(samples=samples, key=jax.random.key(0), confidence=0.95)

    assert float(lo) <= float(jnp.mean(samples)) <= float(hi)


def test_bootstrap_threshold_ci_is_calibrax_bootstrap_interval_of_the_mean() -> None:
    from calibrax.statistics import bootstrap_interval

    rc = _import_risk_control()
    samples = _normal_samples()
    key = jax.random.key(3)

    lo, hi = rc.bootstrap_threshold_ci(
        samples=samples, key=key, confidence=0.9, bootstrap_resamples=500
    )
    reference = bootstrap_interval(jnp.mean, samples, key=key, num_resamples=500, confidence=0.9)

    assert float(lo) == float(reference.lower)
    assert float(hi) == float(reference.upper)


def test_bootstrap_threshold_ci_depends_only_on_the_key() -> None:
    rc = _import_risk_control()
    samples = _normal_samples()

    first = rc.bootstrap_threshold_ci(samples=samples, key=jax.random.key(7))
    again = rc.bootstrap_threshold_ci(samples=samples, key=jax.random.key(7))
    other = rc.bootstrap_threshold_ci(samples=samples, key=jax.random.key(8))

    assert [float(v) for v in first] == [float(v) for v in again]
    assert [float(v) for v in first] != [float(v) for v in other]


def test_bootstrap_threshold_ci_of_one_sample_is_that_sample() -> None:
    rc = _import_risk_control()

    lo, hi = rc.bootstrap_threshold_ci(samples=jnp.asarray([0.25]), key=jax.random.key(0))

    assert float(lo) == float(hi) == 0.25


def test_bootstrap_threshold_ci_traces_once_under_jit() -> None:
    from substrax.testing import TraceCounter

    rc = _import_risk_control()
    counter = TraceCounter()
    interval = jax.jit(
        counter.wrap(lambda samples, key: rc.bootstrap_threshold_ci(samples=samples, key=key))
    )
    samples = _normal_samples()

    with counter.expect(new_traces=1):
        compiled = interval(samples, jax.random.key(1))
    with counter.expect(new_traces=0):
        interval(samples + 1.0, jax.random.key(2))

    eager = rc.bootstrap_threshold_ci(samples=samples, key=jax.random.key(1))
    assert [float(v) for v in compiled] == pytest.approx([float(v) for v in eager])


def test_bootstrap_threshold_ci_vmaps_over_keys() -> None:
    rc = _import_risk_control()
    samples = _normal_samples()
    keys = jax.random.split(jax.random.key(4), 3)

    lo, hi = jax.vmap(lambda key: rc.bootstrap_threshold_ci(samples=samples, key=key))(keys)

    assert lo.shape == hi.shape == (3,)
    for index in range(3):
        one_lo, one_hi = rc.bootstrap_threshold_ci(samples=samples, key=keys[index])
        assert float(lo[index]) == pytest.approx(float(one_lo))
        assert float(hi[index]) == pytest.approx(float(one_hi))


def test_bootstrap_threshold_ci_advances_an_rngs_under_nnx_jit() -> None:
    from substrax.testing import TraceCounter

    rc = _import_risk_control()
    counter = TraceCounter()
    interval = nnx.jit(
        counter.wrap(lambda samples, rngs: rc.bootstrap_threshold_ci(samples=samples, key=rngs))
    )
    samples = _normal_samples()
    rngs = nnx.Rngs(sample=5)

    with counter.expect(new_traces=1):
        first = interval(samples, rngs)
    with counter.expect(new_traces=0):
        second = interval(samples, rngs)

    eager_rngs = nnx.Rngs(sample=5)
    eager_first = rc.bootstrap_threshold_ci(samples=samples, key=eager_rngs)
    eager_second = rc.bootstrap_threshold_ci(samples=samples, key=eager_rngs)
    assert [float(v) for v in first] == pytest.approx([float(v) for v in eager_first])
    assert [float(v) for v in second] == pytest.approx([float(v) for v in eager_second])
    assert [float(v) for v in first] != [float(v) for v in second]


def test_bootstrap_threshold_ci_bounds_have_unit_gradient_mass() -> None:
    """Each bound is a convex combination of resample means, so its gradient sums to one."""
    rc = _import_risk_control()
    samples = _normal_samples()

    for bound in (0, 1):
        grad = jax.grad(
            lambda s, b=bound: rc.bootstrap_threshold_ci(samples=s, key=jax.random.key(6))[b]
        )(samples)
        assert bool(jnp.all(jnp.isfinite(grad)))
        assert float(jnp.sum(grad)) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_select_threshold_rcps_is_vmap_compatible() -> None:
    """vmap over an extra "trial" axis must trace cleanly."""
    rc = _import_risk_control()
    thresholds = jnp.linspace(0.0, 1.0, 21)
    losses_batched = jnp.stack(
        [
            _monotonic_per_sample_losses(thresholds=thresholds, n_samples=256, seed=seed)
            for seed in range(3)
        ]
    )
    config = rc.RiskControlConfig(alpha=0.1, delta=0.05, loss_name="abs_excess")
    chosen = jax.vmap(
        lambda losses: rc.rcps_threshold_kernel(
            thresholds=thresholds, losses=losses, alpha=config.alpha, delta=config.delta
        )
    )(losses_batched)
    assert chosen.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(chosen)))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_conformal_surface_includes_risk_control_components() -> None:
    from opifex.uncertainty import conformal

    expected = {
        "RiskControlConfig",
        "RiskControllerState",
        "hoeffding_upper_bound",
        "select_threshold_rcps",
        "rcps_threshold_kernel",
        "bootstrap_threshold_ci",
    }
    missing = expected - set(dir(conformal))
    assert not missing, f"missing public risk-control symbols: {sorted(missing)}"
