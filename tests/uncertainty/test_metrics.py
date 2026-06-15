"""Core UQ metric contracts.

Each metric has:

1. A closed-form / NumPy reference value on a small hand-picked example.
2. A ``jax.jit`` smoke check where applicable.

References:

* Gneiting & Raftery 2007, "Strictly Proper Scoring Rules, Prediction, and
  Estimation" (JASA) — interval / Winkler score
  ``IS_α(l, u, y) = (u - l) + (2/α)·(l - y)_+ + (2/α)·(y - u)_+``.
* Gal & Ghahramani 2016, "Dropout as a Bayesian Approximation" — ensemble
  predictive entropy and mutual-information decomposition into
  ``H(mean_m p_m) - mean_m H(p_m)``.

Brier / ECE / pinball / Gaussian-NLL live in
``opifex.uncertainty.calibration`` and are imported from there directly
(no forward re-exports from this module per the no-shim convention).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _import_metrics():
    from opifex.uncertainty import metrics

    return metrics


# ---------------------------------------------------------------------------
# Predictive entropy (ensemble form)
# ---------------------------------------------------------------------------


def test_predictive_entropy_matches_entropy_of_mean_probabilities() -> None:
    """For ensemble probs ``p_m``, predictive entropy is ``H(mean_m p_m)``."""
    metrics = _import_metrics()
    # Three ensemble members, two samples, two classes.
    ensemble_probs = jnp.array(
        [
            [[0.9, 0.1], [0.3, 0.7]],
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.8, 0.2], [0.5, 0.5]],
        ]
    )
    mean_probs = jnp.mean(ensemble_probs, axis=0)
    # H(p) = -sum_k p_k log(p_k)
    expected = -jnp.sum(mean_probs * jnp.log(mean_probs), axis=-1)
    out = metrics.predictive_entropy(ensemble_probabilities=ensemble_probs)
    assert bool(jnp.allclose(out, expected, atol=1e-6))


def test_predictive_entropy_is_zero_for_perfect_consensus() -> None:
    """When all members predict the same one-hot, predictive entropy is 0."""
    metrics = _import_metrics()
    ensemble_probs = jnp.tile(jnp.array([[1.0, 0.0]]), (5, 3, 1))
    out = metrics.predictive_entropy(ensemble_probabilities=ensemble_probs)
    assert bool(jnp.allclose(out, jnp.zeros((3,)), atol=1e-6))


# ---------------------------------------------------------------------------
# Mutual information (ensemble decomposition)
# ---------------------------------------------------------------------------


def test_mutual_information_decomposes_predictive_into_epistemic() -> None:
    """``MI = H(mean_m p_m) - mean_m H(p_m)`` per Gal & Ghahramani 2016."""
    metrics = _import_metrics()
    ensemble_probs = jnp.array(
        [
            [[0.9, 0.1], [0.5, 0.5]],
            [[0.1, 0.9], [0.5, 0.5]],
        ]
    )
    mean_probs = jnp.mean(ensemble_probs, axis=0)
    h_mean = -jnp.sum(mean_probs * jnp.log(mean_probs + 1e-12), axis=-1)
    h_members = -jnp.sum(ensemble_probs * jnp.log(ensemble_probs + 1e-12), axis=-1)
    expected_mi = h_mean - jnp.mean(h_members, axis=0)
    out = metrics.mutual_information(ensemble_probabilities=ensemble_probs)
    assert bool(jnp.allclose(out, expected_mi, atol=1e-5))


def test_mutual_information_is_zero_for_identical_members() -> None:
    """Identical ensemble members → zero epistemic uncertainty."""
    metrics = _import_metrics()
    base = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    ensemble_probs = jnp.tile(base, (4, 1, 1))
    out = metrics.mutual_information(ensemble_probabilities=ensemble_probs)
    assert bool(jnp.allclose(out, jnp.zeros((2,)), atol=1e-6))


# ---------------------------------------------------------------------------
# Interval / Winkler score (Gneiting & Raftery 2007)
# ---------------------------------------------------------------------------


def test_interval_score_matches_definition() -> None:
    metrics = _import_metrics()
    alpha = 0.1
    lower = jnp.array([0.0, 0.0, 0.0])
    upper = jnp.array([1.0, 1.0, 1.0])
    targets = jnp.array([0.5, -0.5, 1.5])  # in / below / above
    # IS = (u - l) + (2/alpha)(l - y)_+ + (2/alpha)(y - u)_+
    width = upper - lower
    expected = np.array(
        [
            float(width[0]),
            float(width[1]) + (2.0 / alpha) * 0.5,
            float(width[2]) + (2.0 / alpha) * 0.5,
        ]
    )
    out = metrics.interval_score(lower=lower, upper=upper, targets=targets, alpha=alpha)
    assert bool(jnp.allclose(out, jnp.asarray(expected), atol=1e-5))


def test_winkler_score_is_alias_of_interval_score() -> None:
    metrics = _import_metrics()
    lower = jnp.array([0.0])
    upper = jnp.array([1.0])
    targets = jnp.array([2.0])
    a = metrics.interval_score(lower=lower, upper=upper, targets=targets, alpha=0.1)
    b = metrics.winkler_score(lower=lower, upper=upper, targets=targets, alpha=0.1)
    assert bool(jnp.allclose(a, b))


def test_interval_score_lower_is_better_on_correct_intervals() -> None:
    metrics = _import_metrics()
    targets = jnp.array([0.5])
    narrow = metrics.interval_score(
        lower=jnp.array([0.4]), upper=jnp.array([0.6]), targets=targets, alpha=0.1
    )
    wide = metrics.interval_score(
        lower=jnp.array([0.0]), upper=jnp.array([1.0]), targets=targets, alpha=0.1
    )
    # Both intervals cover the target → score is just (u - l). Narrow < wide.
    assert float(narrow[0]) < float(wide[0])


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_metrics_are_jit_compatible() -> None:
    metrics = _import_metrics()
    rng = np.random.default_rng(0)
    ensemble = jnp.asarray(rng.uniform(0.0, 1.0, size=(5, 8, 3)))
    ensemble = ensemble / jnp.sum(ensemble, axis=-1, keepdims=True)
    lower = jnp.asarray(rng.standard_normal((8,)))
    upper = lower + jnp.asarray(rng.uniform(0.1, 1.0, size=(8,)))
    targets = jnp.asarray(rng.standard_normal((8,)))

    jitted_h = jax.jit(lambda probs: metrics.predictive_entropy(ensemble_probabilities=probs))
    jitted_mi = jax.jit(lambda probs: metrics.mutual_information(ensemble_probabilities=probs))
    jitted_is = jax.jit(
        lambda lo, up, y: metrics.interval_score(lower=lo, upper=up, targets=y, alpha=0.1)
    )

    h_out = jitted_h(ensemble)
    mi_out = jitted_mi(ensemble)
    is_out = jitted_is(lower, upper, targets)
    assert h_out.shape == (8,)
    assert mi_out.shape == (8,)
    assert is_out.shape == (8,)


def test_metrics_are_vmap_compatible() -> None:
    metrics = _import_metrics()
    rng = np.random.default_rng(0)
    base = rng.uniform(0.0, 1.0, size=(3, 5, 8, 4))
    base = base / base.sum(axis=-1, keepdims=True)
    ensemble_batch = jnp.asarray(base)
    out = jax.vmap(lambda probs: metrics.predictive_entropy(ensemble_probabilities=probs))(
        ensemble_batch
    )
    assert out.shape == (3, 8)
    assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_metrics_surface() -> None:
    """``metrics`` only exposes new ensemble / interval kernels — calibration
    metrics (Brier/ECE/pinball/Gaussian-NLL) stay in
    ``opifex.uncertainty.calibration`` to avoid forward shims."""
    metrics = _import_metrics()
    expected = {
        "interval_score",
        "mutual_information",
        "predictive_entropy",
        "winkler_score",
    }
    missing = expected - set(dir(metrics))
    assert not missing, f"missing public metric symbols: {sorted(missing)}"
