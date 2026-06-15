"""TemperatureScaling contract.

Guo et al. 2017 ("On Calibration of Modern Neural Networks", arXiv:1706.04599)
fit a single positive scalar ``T`` via NLL minimisation on a held-out
validation set; at predict time scaled probabilities are
``softmax(logits / T)``.

Tests pin:

* Frozen fitted-state container (pattern B per GUIDE_ALIGNMENT §5a).
* Pre-fit ``predict`` raises ``RuntimeError`` (not silent no-op).
* Over-confident logits → ``T > 1`` after fit (NLL minimum).
* Under-confident logits → ``T < 1`` after fit.
* ``predict(state, logits)`` is ``jax.jit`` compatible.
* The fit objective is differentiable in ``log_temperature``.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_module():
    from opifex.uncertainty.calibration import temperature

    return temperature


def _calibration_data(
    *,
    logit_boost: float,
    accuracy: float,
    num_classes: int = 3,
    n: int = 512,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Synthesise (logits, true_labels) where the model's confidence is decoupled from accuracy.

    The model picks a class to predict (``predicted``) and puts ``logit_boost``
    on that one channel; the true label equals ``predicted`` with probability
    ``accuracy`` and a uniformly-random other class otherwise. Over-confident
    regime: large ``logit_boost`` + low ``accuracy`` (softmax peak >> empirical
    accuracy → optimal ``T > 1``). Under-confident regime: small
    ``logit_boost`` + high ``accuracy`` (softmax flat << empirical accuracy →
    optimal ``T < 1``).
    """
    rng = np.random.default_rng(seed)
    predicted = rng.integers(low=0, high=num_classes, size=(n,))
    is_correct = rng.uniform(size=(n,)) < accuracy
    other_offset = 1 + rng.integers(low=0, high=num_classes - 1, size=(n,))
    other = (predicted + other_offset) % num_classes
    true_labels = np.where(is_correct, predicted, other)
    onehot = np.eye(num_classes)[predicted]
    logits = onehot * logit_boost
    return jnp.asarray(logits), jnp.asarray(true_labels.astype(np.int32))


def test_temperature_scaling_state_is_frozen_dataclass() -> None:
    temperature = _import_module()
    state = temperature.TemperatureScalingState(temperature=jnp.asarray(1.5))
    assert float(state.temperature) == pytest.approx(1.5, rel=1e-6)
    with pytest.raises(dc.FrozenInstanceError):
        state.temperature = jnp.asarray(2.0)  # type: ignore[misc]


def test_predict_before_fit_raises_runtime_error() -> None:
    temperature = _import_module()
    calibrator = temperature.TemperatureScaling()
    logits = jnp.array([[1.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match=r"(?i)(fit|calibrate)"):
        calibrator.predict(logits)


def test_fit_on_overconfident_logits_produces_temperature_above_one() -> None:
    temperature = _import_module()
    logits, labels = _calibration_data(logit_boost=8.0, accuracy=0.55, seed=1)
    calibrator = temperature.TemperatureScaling()
    state = calibrator.fit(logits=logits, targets=labels)
    assert float(state.temperature) > 1.0


def test_fit_on_underconfident_logits_produces_temperature_below_one() -> None:
    temperature = _import_module()
    logits, labels = _calibration_data(logit_boost=0.5, accuracy=0.85, seed=2)
    calibrator = temperature.TemperatureScaling()
    state = calibrator.fit(logits=logits, targets=labels)
    assert float(state.temperature) < 1.0


def test_fit_returns_positive_temperature() -> None:
    temperature = _import_module()
    logits, labels = _calibration_data(logit_boost=3.0, accuracy=0.7, n=256, seed=3)
    state = temperature.TemperatureScaling().fit(logits=logits, targets=labels)
    assert float(state.temperature) > 0.0


def test_predict_applies_state_temperature_to_softmax() -> None:
    temperature = _import_module()
    logits = jnp.array([[2.0, 1.0, -1.0]])
    state = temperature.TemperatureScalingState(temperature=jnp.asarray(2.0))
    calibrator = temperature.TemperatureScaling().with_state(state)
    out = calibrator.predict(logits)
    expected = jax.nn.softmax(logits / 2.0, axis=-1)
    assert bool(jnp.allclose(out, expected, rtol=1e-6, atol=1e-7))


def test_predict_is_jit_compatible() -> None:
    temperature = _import_module()
    logits, _ = _calibration_data(logit_boost=3.0, accuracy=0.7, n=64, seed=4)
    state = temperature.TemperatureScalingState(temperature=jnp.asarray(1.8))
    calibrator = temperature.TemperatureScaling().with_state(state)

    @jax.jit
    def jitted_predict(x: jax.Array) -> jax.Array:
        return calibrator.predict(x)

    out = jitted_predict(logits)
    eager_out = calibrator.predict(logits)
    assert bool(jnp.allclose(out, eager_out, rtol=1e-6, atol=1e-7))


def test_fit_objective_is_differentiable() -> None:
    """The internal NLL objective must produce finite gradients w.r.t. log T."""
    temperature = _import_module()
    logits, labels = _calibration_data(logit_boost=3.0, accuracy=0.7, n=128, seed=5)

    def loss_of_log_temp(log_t: jax.Array) -> jax.Array:
        return temperature.nll_loss_at_temperature(
            logits=logits, targets=labels, log_temperature=log_t
        )

    grad = jax.grad(loss_of_log_temp)(jnp.asarray(0.0))
    assert bool(jnp.isfinite(grad))
    assert bool(jnp.abs(grad) > 0.0)


def test_calibrator_records_fitted_state_metadata() -> None:
    temperature = _import_module()
    logits, labels = _calibration_data(logit_boost=3.0, accuracy=0.7, n=128, seed=6)
    state = temperature.TemperatureScaling().fit(logits=logits, targets=labels)
    md = dict(state.metadata)
    assert md.get("method") == "temperature_scaling"
    assert "calibration_size" in md
    assert int(md["calibration_size"]) == logits.shape[0]
