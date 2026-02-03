"""Smoke tests for Geometry-Informed Neural Operator (GINO).

Minimal API consistency checks: import, instantiation, and forward pass shape.
"""

from __future__ import annotations

import jax
from flax import nnx


class TestGINOSmoke:
    """Minimal smoke tests for GINO."""

    def test_import(self) -> None:
        """GINO class is importable from the specialized operators package."""
        from opifex.neural.operators.specialized.gino import (
            GeometryInformedNeuralOperator,
        )

        assert GeometryInformedNeuralOperator is not None

    def test_instantiation(self) -> None:
        """GINO can be instantiated with minimal parameters."""
        from opifex.neural.operators.specialized.gino import (
            GeometryInformedNeuralOperator,
        )

        model = GeometryInformedNeuralOperator(
            in_channels=3,
            out_channels=1,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=2,
            rngs=nnx.Rngs(0),
        )
        assert model is not None

    def test_forward_pass_shape(self) -> None:
        """Forward pass produces correct output shape."""
        from opifex.neural.operators.specialized.gino import (
            GeometryInformedNeuralOperator,
        )

        batch, height, width, in_ch, out_ch = 2, 16, 16, 3, 1
        model = GeometryInformedNeuralOperator(
            in_channels=in_ch,
            out_channels=out_ch,
            hidden_channels=16,
            modes=(4, 4),
            num_layers=2,
            rngs=nnx.Rngs(0),
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (batch, height, width, in_ch))
        y = model(x)
        assert y.shape[0] == batch
        assert y.shape[-1] == out_ch
