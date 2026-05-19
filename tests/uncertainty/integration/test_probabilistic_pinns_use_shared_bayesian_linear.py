"""``ProbabilisticPINN`` and ``MultiFidelityPINN`` use the shared Bayesian dense layer.

Pins the migration: both PINN classes construct their hidden layers from
:class:`opifex.uncertainty.layers.bayesian.BayesianLinear` and thread their
constructor-time ``nnx.Rngs`` through every forward-pass invocation.
"""

from __future__ import annotations

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
    """End-to-end forward pass uses BayesianLinear and routes RNG through self.rngs."""
    model = ProbabilisticPINN(
        input_dim=2, output_dim=1, hidden_dims=(8, 8), use_bayesian=True, rngs=nnx.Rngs(0)
    )
    x = jnp.ones((4, 2))
    out = model(x, deterministic=False)
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


def test_multi_fidelity_pinn_constructor_stores_rngs_for_forward_pass() -> None:
    """Migration stores ``self.rngs`` so forward passes can satisfy BayesianLinear's RNG contract."""
    model = MultiFidelityPINN(input_dim=2, output_dim=1, rngs=nnx.Rngs(0))
    assert isinstance(model.rngs, nnx.Rngs)
