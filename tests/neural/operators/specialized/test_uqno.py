"""Test the Uncertainty Quantification Neural Operator (UQNO).

After the Phase 3 migration UQNO exposes the shared platform surface:
deterministic / sampling forward pass via the canonical
``deterministic`` + caller-owned ``rngs`` keyword pair, and the
:meth:`predict_distribution` / :meth:`loss_components` /
:meth:`negative_elbo` methods. These tests pin the operator-level shape,
finiteness, and differentiability contracts; the deeper objective
surface is covered by ``test_uqno_objectives.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
)


class TestUncertaintyQuantificationNeuralOperator:
    """Operator-level UQNO contracts (shapes, finiteness, gradients)."""

    @pytest.fixture
    def rng_key(self) -> jax.Array:
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key: jax.Array) -> nnx.Rngs:
        return nnx.Rngs(rng_key)

    def test_uqno_initialization(self, rngs: nnx.Rngs) -> None:
        uqno = UncertaintyQuantificationNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes=(8, 8),
            num_layers=2,
            rngs=rngs,
        )
        assert uqno.in_channels == 2
        assert uqno.out_channels == 1
        assert uqno.hidden_channels == 32
        assert hasattr(uqno, "uqno_layers")
        assert len(uqno.uqno_layers) == 2

    def test_uqno_forward_pass_deterministic(self, rngs: nnx.Rngs, rng_key: jax.Array) -> None:
        """Deterministic forward returns a finite tensor of the expected shape."""
        uqno = UncertaintyQuantificationNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=4,
            modes=(2, 2),
            num_layers=1,
            rngs=rngs,
        )
        x = jax.random.normal(rng_key, (1, 4, 4, 1))
        out = uqno(x, deterministic=True)
        assert out.shape == (1, 4, 4, 1)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_uqno_forward_pass_sampling_requires_rngs(
        self, rngs: nnx.Rngs, rng_key: jax.Array
    ) -> None:
        """Non-deterministic forward needs caller-owned ``rngs`` — no hidden seed."""
        uqno = UncertaintyQuantificationNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=4,
            modes=(2, 2),
            num_layers=1,
            rngs=rngs,
        )
        x = jax.random.normal(rng_key, (1, 4, 4, 1))
        with pytest.raises(ValueError, match="requires caller-owned"):
            uqno(x)  # sampling mode without rngs

        out = uqno(x, rngs=nnx.Rngs(sample=7))
        assert out.shape == (1, 4, 4, 1)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_uqno_predict_distribution_shapes(self, rngs: nnx.Rngs, rng_key: jax.Array) -> None:
        """`predict_distribution` carries MC samples + variance + non-negative epistemic."""
        uqno = UncertaintyQuantificationNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            modes=(2, 2),
            num_layers=1,
            rngs=rngs,
        )
        x = jax.random.normal(rng_key, (1, 4, 4, 1))
        dist = uqno.predict_distribution(x, rngs=nnx.Rngs(sample=5), num_samples=5)

        assert dist.mean.shape == (1, 4, 4, 1)
        assert dist.samples is not None
        assert dist.samples.shape == (5, 1, 4, 4, 1)
        assert dist.variance is not None
        assert dist.variance.shape == (1, 4, 4, 1)
        assert dist.epistemic is not None
        assert bool(jnp.all(dist.epistemic >= 0.0))

    def test_uqno_differentiability(self, rngs: nnx.Rngs, rng_key: jax.Array) -> None:
        """`nnx.grad` on a deterministic UQNO output produces some non-zero gradients."""
        uqno = UncertaintyQuantificationNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            modes=(2, 2),
            num_layers=1,
            rngs=rngs,
        )

        def loss_fn(model: UncertaintyQuantificationNeuralOperator, x: jax.Array) -> jax.Array:
            return jnp.sum(model(x, deterministic=True) ** 2)

        x = jax.random.normal(rng_key, (1, 4, 4, 1))
        grads = nnx.grad(loss_fn)(uqno, x)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            float(jnp.linalg.norm(leaf)) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert grad_norms, "expected at least one gradient leaf"
        assert any(norm > 1e-8 for norm in grad_norms)
