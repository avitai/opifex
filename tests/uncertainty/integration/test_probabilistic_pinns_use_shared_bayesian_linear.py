"""``ProbabilisticPINN`` and ``MultiFidelityPINN`` use the shared Bayesian dense layer.

Pins the migration: both PINN classes construct their hidden layers from
:class:`opifex.uncertainty.layers.bayesian.BayesianLinear` and thread their
constructor-time ``nnx.Rngs`` through every forward-pass invocation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.bayesian.probabilistic_pinns import (
    MultiFidelityPINN,
    ProbabilisticPINN,
)
from opifex.uncertainty.layers.bayesian import BayesianLinear


def test_probabilistic_pinn_layers_are_bayesian_linear() -> None:
    model = ProbabilisticPINN(
        input_dim=2, output_dim=1, hidden_dims=(8, 8), use_bayesian=True, rngs=nnx.Rngs(0)
    )
    bayesian_layers = [layer for layer in model.layers if isinstance(layer, BayesianLinear)]
    assert len(bayesian_layers) == 2
    assert isinstance(model.output_layer, BayesianLinear)


def test_probabilistic_pinn_forward_pass_produces_finite_output() -> None:
    """End-to-end forward pass uses BayesianLinear, which samples from its own stream."""
    model = ProbabilisticPINN(
        input_dim=2, output_dim=1, hidden_dims=(8, 8), use_bayesian=True, rngs=nnx.Rngs(0)
    )
    x = jnp.ones((4, 2))
    out = model(x)
    assert out.shape == (4, 1)
    assert jnp.all(jnp.isfinite(out))


def test_multi_fidelity_pinn_low_fidelity_forward_uses_bayesian_linear() -> None:
    model = MultiFidelityPINN(
        input_dim=2,
        output_dim=1,
        low_fidelity_dims=(8, 8),
        high_fidelity_dims=(8, 8),
        rngs=nnx.Rngs(0),
    )
    bayesian_layers = [
        layer for layer in model.low_fidelity_layers if isinstance(layer, BayesianLinear)
    ]
    assert bayesian_layers, "MultiFidelityPINN low-fidelity layers must be BayesianLinear"

    x = jnp.ones((2, 2))
    result = model._low_fidelity_forward(x, deterministic=False)
    assert "low_fidelity_pred" in result
    assert jnp.all(jnp.isfinite(result["low_fidelity_pred"]))


def _assert_a_view_draws_no_key(model: nnx.Module, rngs: nnx.Rngs) -> None:
    """The PINN keeps no reference to the caller's ``nnx.Rngs``, as ``nnx.Dropout`` keeps none.

    ``nnx.view`` calls ``set_view`` on every node that has one, and an ``nnx.Rngs`` with a
    ``default`` stream answers any attribute name with that stream, so a stored ``Rngs``
    would have a key drawn from it by every view, and the view would fail under ``jax.jvp``.
    """
    count = int(rngs.default.count[...])
    nnx.view(model, deterministic=True)
    assert int(rngs.default.count[...]) == count


def test_a_view_of_a_probabilistic_pinn_draws_no_key_and_differentiates() -> None:
    rngs = nnx.Rngs(0)
    model = ProbabilisticPINN(input_dim=2, output_dim=1, hidden_dims=(8,), rngs=rngs)
    _assert_a_view_draws_no_key(model, rngs)

    def posterior_mean(points: jax.Array) -> jax.Array:
        return nnx.view(model, deterministic=True)(points)

    x = jnp.ones((3, 2))
    _, tangent = jax.jvp(posterior_mean, (x,), (jnp.ones_like(x),))
    assert jnp.all(jnp.isfinite(tangent))


def test_a_view_of_a_multi_fidelity_pinn_draws_no_key_and_differentiates() -> None:
    rngs = nnx.Rngs(0)
    model = MultiFidelityPINN(input_dim=2, output_dim=1, rngs=rngs)
    _assert_a_view_draws_no_key(model, rngs)

    def posterior_mean(points: jax.Array) -> jax.Array:
        result = nnx.view(model, deterministic=True)(points, fidelity_level="low")
        return result["low_fidelity_pred"]

    x = jnp.ones((3, 2))
    _, tangent = jax.jvp(posterior_mean, (x,), (jnp.ones_like(x),))
    assert jnp.all(jnp.isfinite(tangent))


def test_the_probabilistic_pinn_mode_decides_whether_it_samples() -> None:
    """Like flax's composites, the PINN takes no mode argument: its layers' mode decides."""
    model = ProbabilisticPINN(
        input_dim=2, output_dim=1, hidden_dims=(8, 8), use_bayesian=True, rngs=nnx.Rngs(0)
    )
    x = jnp.ones((4, 2))
    assert not jnp.array_equal(model(x), model(x))
    mean_view = nnx.view(model, deterministic=True)
    assert jnp.array_equal(mean_view(x), mean_view(x))
    model.eval()
    assert jnp.array_equal(model(x), mean_view(x))


def test_the_multi_fidelity_pinn_mode_decides_whether_it_samples() -> None:
    model = MultiFidelityPINN(
        input_dim=2,
        output_dim=1,
        low_fidelity_dims=(8, 8),
        high_fidelity_dims=(8, 8),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 2))
    sampled = model(x, fidelity_level="low")["low_fidelity_pred"]
    assert not jnp.array_equal(sampled, model(x, fidelity_level="low")["low_fidelity_pred"])
    model.eval()
    first = model(x, fidelity_level="low")["low_fidelity_pred"]
    assert jnp.array_equal(first, model(x, fidelity_level="low")["low_fidelity_pred"])
