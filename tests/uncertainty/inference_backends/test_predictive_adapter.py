"""Tests for the inference-backend predictive adapter.

The adapter :func:`opifex.uncertainty._predictive.predictive_from_parameter_samples`
maps parameter-space posterior draws to a
:class:`opifex.uncertainty.types.PredictiveDistribution` via two paths:

* model-aware (``predict_fn`` supplied) — the genuine Bayesian posterior
  predictive, marginalising the model forward over the posterior parameter
  draws (Gelman et al. 2013, *Bayesian Data Analysis* 3rd ed., §3.2 — the
  posterior predictive distribution).
* lightweight (``predict_fn is None``) — a parameter-moment stand-in that
  broadcasts the posterior parameter mean / variance to ``x.shape``,
  byte-compatible with ``BlackJAXBackend``'s lightweight form.

The adapter must additionally pass ``jax.jit`` / ``jax.vmap`` and admit a
finite ``jax.grad`` through a scalar loss (a required JAX/NNX-transform
exit criterion).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty._predictive import predictive_from_parameter_samples
from opifex.uncertainty.types import PredictiveDistribution


def _linear_predict(params: jax.Array, x: jax.Array) -> jax.Array:
    """A known linear forward model ``x @ params`` over a ``(d,)`` parameter vector."""
    return x @ params


def test_predictive_from_parameter_samples_model_aware() -> None:
    """Model-aware path equals the manually-vmapped forward's empirical moments."""
    parameter_samples = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-1.0, 0.5]],
    )
    x = jnp.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])

    predictive = predictive_from_parameter_samples(
        parameter_samples,
        x,
        predict_fn=_linear_predict,
        metadata=(("backend", "test"),),
    )

    expected_preds = jax.vmap(lambda p: _linear_predict(p, x))(parameter_samples)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.samples is not None
    assert jnp.allclose(predictive.samples, expected_preds)
    assert jnp.allclose(predictive.mean, jnp.mean(expected_preds, axis=0))
    assert predictive.variance is not None
    assert jnp.allclose(predictive.variance, jnp.var(expected_preds, axis=0))
    assert predictive.mean.shape == (3,)


def test_predictive_from_parameter_samples_lightweight() -> None:
    """Lightweight path broadcasts the parameter moments to ``x.shape``."""
    parameter_samples = jnp.array(
        [[1.0, 0.0, -1.0], [0.0, 1.0, 2.0], [2.0, -1.0, 0.5]],
    )
    x = jnp.zeros((4, 3))

    predictive = predictive_from_parameter_samples(parameter_samples, x)

    expected_mean = jnp.broadcast_to(jnp.mean(parameter_samples, axis=0), x.shape)
    expected_var = jnp.broadcast_to(jnp.var(parameter_samples, axis=0), x.shape)
    assert isinstance(predictive, PredictiveDistribution)
    assert predictive.mean.shape == x.shape
    assert jnp.allclose(predictive.mean, expected_mean)
    assert predictive.variance is not None
    assert jnp.allclose(predictive.variance, expected_var)


def test_predictive_adapter_is_jit_grad_vmap_compatible() -> None:
    """The model-aware adapter survives jit / vmap and admits a finite grad."""
    parameter_samples = jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    x = jnp.array([[1.0, 2.0], [3.0, -1.0]])

    def adapter_mean(samples: jax.Array, inputs: jax.Array) -> jax.Array:
        return predictive_from_parameter_samples(samples, inputs, predict_fn=_linear_predict).mean

    eager = adapter_mean(parameter_samples, x)

    jitted = jax.jit(adapter_mean)(parameter_samples, x)
    assert jnp.allclose(jitted, eager)

    # vmap over a leading batch of independent parameter-sample sets.
    batched_samples = jnp.stack([parameter_samples, parameter_samples + 1.0])
    vmapped = jax.vmap(lambda s: adapter_mean(s, x))(batched_samples)
    assert vmapped.shape == (2, *eager.shape)
    assert jnp.allclose(vmapped[0], eager)

    def scalar_loss(samples: jax.Array) -> jax.Array:
        return jnp.sum(adapter_mean(samples, x) ** 2)

    grads = jax.grad(scalar_loss)(parameter_samples)
    assert grads.shape == parameter_samples.shape
    assert jnp.all(jnp.isfinite(grads))
