"""Tests for shared operator-learning error metrics (`opifex.core.metrics`).

TDD: these define the behaviour `relative_l2_error` must satisfy. The same
function backs the Trainer's ``relative_l2`` loss and the evaluation metric the
examples report, so a single implementation is exercised everywhere (DRY).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.core.metrics import per_sample_relative_l2, relative_l2_error


def test_relative_l2_error_is_jit_grad_vmap_compatible() -> None:
    """The metric backs the Trainer loss, so it must trace under jit/grad/vmap."""
    pred = jnp.ones((6, 5)) * 0.5
    target = jnp.ones((6, 5))
    assert jnp.isfinite(jax.jit(relative_l2_error)(pred, target))
    grad = jax.grad(relative_l2_error)(pred, target)  # differentiable as a loss
    assert jnp.all(jnp.isfinite(grad))
    batched = jax.vmap(per_sample_relative_l2)(pred[None], target[None])
    assert batched.shape == (1, 6)


def test_per_sample_returns_one_ratio_per_batch_element() -> None:
    """Per-sample variant returns a (batch,) vector, and the mean matches."""
    target = jnp.array([[3.0, 4.0], [6.0, 8.0]])  # norms 5, 10
    pred = jnp.zeros((2, 2))  # diff norms 5, 10 -> ratios 1.0, 1.0
    per = per_sample_relative_l2(pred, target)
    assert per.shape == (2,)
    assert jnp.allclose(per, jnp.array([1.0, 1.0]), atol=1e-4)
    assert float(relative_l2_error(pred, target)) == pytest.approx(float(jnp.mean(per)), rel=1e-6)


def test_identical_fields_have_zero_error() -> None:
    """Prediction equal to target gives zero relative L2."""
    y = jnp.arange(12.0).reshape(3, 4)
    assert float(relative_l2_error(y, y)) == pytest.approx(0.0, abs=1e-6)


def test_zero_prediction_against_unit_target_is_one() -> None:
    """Predicting zeros against a non-zero target gives a ratio of ~1."""
    target = jnp.ones((2, 5))
    pred = jnp.zeros((2, 5))
    assert float(relative_l2_error(pred, target)) == pytest.approx(1.0, rel=1e-4)


def test_is_per_sample_mean_of_norm_ratios() -> None:
    """Error is the mean over the batch of per-sample ||diff|| / ||target||."""
    target = jnp.array([[3.0, 4.0], [0.0, 5.0]])  # row norms 5, 5
    pred = jnp.array([[0.0, 0.0], [0.0, 0.0]])  # diff norms 5, 5
    # per-sample ratios: 5/5 and 5/5 -> mean 1.0
    assert float(relative_l2_error(pred, target)) == pytest.approx(1.0, rel=1e-4)


def test_flattens_trailing_dims_per_sample() -> None:
    """Multi-axis fields are flattened per leading (batch) sample."""
    target = jnp.ones((4, 8, 8, 1))
    pred = target * 1.1
    # every sample has identical 10% relative error
    assert float(relative_l2_error(pred, target)) == pytest.approx(0.1, rel=1e-3)


def test_zero_target_does_not_divide_by_zero() -> None:
    """A zero-norm target sample is regularised, not NaN/Inf."""
    target = jnp.zeros((1, 4))
    pred = jnp.ones((1, 4))
    out = float(relative_l2_error(pred, target))
    assert jnp.isfinite(out)


def test_matches_trainer_relative_l2_loss() -> None:
    """The metric reproduces the Trainer's inline ``relative_l2`` computation."""
    key_pred = jnp.linspace(-1.0, 1.0, 24).reshape(4, 6)
    target = jnp.linspace(0.5, 2.0, 24).reshape(4, 6)
    diff = (key_pred - target).reshape(4, -1)
    tgt = target.reshape(4, -1)
    expected = jnp.mean(jnp.linalg.norm(diff, axis=1) / (jnp.linalg.norm(tgt, axis=1) + 1e-8))
    assert float(relative_l2_error(key_pred, target)) == pytest.approx(float(expected), rel=1e-6)
