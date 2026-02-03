"""Smoke tests for Multipole Graph Neural Operator (MGNO).

Minimal API consistency checks: import, instantiation, and forward pass shape.
"""

from __future__ import annotations

import jax
from flax import nnx


class TestMGNOSmoke:
    """Minimal smoke tests for MGNO."""

    def test_import(self) -> None:
        """MGNO class is importable from the specialized operators package."""
        from opifex.neural.operators.specialized.mgno import (
            MultipoleGraphNeuralOperator,
        )

        assert MultipoleGraphNeuralOperator is not None

    def test_instantiation(self) -> None:
        """MGNO can be instantiated with minimal parameters."""
        from opifex.neural.operators.specialized.mgno import (
            MultipoleGraphNeuralOperator,
        )

        model = MultipoleGraphNeuralOperator(
            in_features=3,
            out_features=1,
            hidden_features=16,
            num_layers=2,
            max_degree=3,
            rngs=nnx.Rngs(0),
        )
        assert model is not None

    def test_forward_pass_shape(self) -> None:
        """Forward pass produces correct output shape."""
        from opifex.neural.operators.specialized.mgno import (
            MultipoleGraphNeuralOperator,
        )

        batch, num_points, in_feat, out_feat, coord_dim = 2, 16, 3, 1, 2
        model = MultipoleGraphNeuralOperator(
            in_features=in_feat,
            out_features=out_feat,
            hidden_features=16,
            num_layers=2,
            max_degree=3,
            rngs=nnx.Rngs(0),
        )
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (batch, num_points, in_feat))
        positions = jax.random.normal(k2, (batch, num_points, coord_dim))
        y = model(x, positions)
        assert y.shape == (batch, num_points, out_feat)
