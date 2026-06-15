r"""Tests for ``active/deep_model_backends.py`` — Slice 23.

Phase 8 Task 8.3 mandates ``active/deep_model_backends.py`` wiring
deep-GP / deep-ensemble model backends from
``opifex.neural.bayesian`` into the AL acquisition loop.

The deep-model backend supplies a ``predict_distribution(...)``
callable that returns a :class:`PredictiveDistribution`; the wrapper
adapts heterogeneous deep-model surfaces into a uniform interface for
the AL loop.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — used at runtime in factory

import jax
import jax.numpy as jnp

from opifex.uncertainty.types import PredictiveDistribution


def test_deep_model_backend_wraps_predict_function_into_uniform_interface() -> None:
    """A user-supplied ``predict`` function plugs into the AL backend protocol."""
    from opifex.uncertainty.active.deep_model_backends import DeepModelBackend

    def predict(x: jax.Array) -> PredictiveDistribution:
        return PredictiveDistribution(mean=x.sum(axis=-1), variance=jnp.ones(x.shape[0]))

    backend = DeepModelBackend(predict_fn=predict, source_package="opifex.neural.bayesian")
    inputs = jnp.ones((4, 3))
    out = backend.predict(inputs)
    assert isinstance(out, PredictiveDistribution)
    assert out.mean.shape == (4,)
    assert backend.source_package == "opifex.neural.bayesian"


def test_deep_ensemble_backend_aggregates_member_predictions() -> None:
    """The deep-ensemble backend averages member predictions into one PD."""
    from opifex.uncertainty.active.deep_model_backends import DeepEnsembleBackend

    def make_member(offset: float) -> Callable[[jax.Array], PredictiveDistribution]:
        def _predict(x: jax.Array) -> PredictiveDistribution:
            return PredictiveDistribution(
                mean=jnp.full(x.shape[:-1], offset),
                variance=jnp.full(x.shape[:-1], 0.1),
            )

        return _predict

    backend = DeepEnsembleBackend(
        member_predict_fns=(make_member(1.0), make_member(2.0), make_member(3.0))
    )
    inputs = jnp.zeros((3, 2))
    aggregated = backend.predict(inputs)
    # Ensemble mean = mean of member means = 2.0
    assert jnp.allclose(aggregated.mean, 2.0)
    # Ensemble variance = average member variance + variance of member means
    # = 0.1 + var([1, 2, 3]) ≈ 0.1 + 2/3
    assert aggregated.variance is not None
    assert float(jnp.mean(aggregated.variance)) > 0.1
