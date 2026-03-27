"""Tests for DeepONet trainer adapter."""

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.deeponet.trainer_adapter import DeepONetTrainerAdapter


class MockDeepONet(nnx.Module):
    """Minimal DeepONet mock for testing the adapter."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.branch_net = nnx.Linear(4, 8, rngs=rngs)
        self.trunk_net = nnx.Linear(2, 8, rngs=rngs)

    def __call__(self, branch_input, trunk_input, **kwargs):
        b = self.branch_net(branch_input)
        t = self.trunk_net(trunk_input)
        return jnp.sum(b * t, axis=-1, keepdims=True)


class TestDeepONetTrainerAdapter:
    """Tests for DeepONetTrainerAdapter."""

    def test_forward_with_dict_input(self):
        """Adapter correctly unpacks dict into branch and trunk."""
        model = MockDeepONet(rngs=nnx.Rngs(0))
        adapter = DeepONetTrainerAdapter(model)

        x = {"branch": jnp.ones((2, 4)), "trunk": jnp.ones((2, 2))}
        result = adapter(x)
        assert result.shape == (2, 1)

    def test_rejects_non_dict_input(self):
        """Non-dict input raises TypeError."""
        model = MockDeepONet(rngs=nnx.Rngs(0))
        adapter = DeepONetTrainerAdapter(model)

        with pytest.raises(TypeError, match="dict input"):
            adapter(jnp.ones((2, 4)))  # pyright: ignore[reportArgumentType]

    def test_rejects_missing_keys(self):
        """Dict missing branch or trunk key raises TypeError."""
        model = MockDeepONet(rngs=nnx.Rngs(0))
        adapter = DeepONetTrainerAdapter(model)

        with pytest.raises(TypeError, match="missing keys"):
            adapter({"branch": jnp.ones((2, 4))})

    def test_wraps_nnx_module(self):
        """Adapter is itself an nnx.Module."""
        model = MockDeepONet(rngs=nnx.Rngs(0))
        adapter = DeepONetTrainerAdapter(model)
        assert isinstance(adapter, nnx.Module)
