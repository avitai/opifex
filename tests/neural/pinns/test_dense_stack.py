"""Tests for the shared PINN dense-stack and baseline-equivalence guards.

Task 12.3.7 consolidates the duplicated "build ``nnx.List`` of ``nnx.Linear``
then apply the activation over every layer except the last" pattern that was
re-implemented in six PINN networks.

The baseline tests below pin the *exact* forward output of every affected
network on a fixed input and seed. They are written before the refactor and
must keep passing after it, proving the extraction changed nothing.

The ``test_dense_stack_jit_grad_vmap`` smoke test guards JAX/NNX transform
compatibility of the shared module itself.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from opifex.neural.pinns.dense_stack import DenseStack
from opifex.neural.pinns.domain_decomposition.apinn import GatingNetwork
from opifex.neural.pinns.domain_decomposition.base import SubdomainNetwork
from opifex.neural.pinns.fpinn import FractionalPINN
from opifex.neural.pinns.gst_pinn import GradientEnhancedPINN
from opifex.neural.pinns.multi_scale import SimplePINN
from opifex.neural.pinns.vpinn import VPINN


# ---------------------------------------------------------------------------
# Pinned baselines (captured from the pre-refactor implementations)
# ---------------------------------------------------------------------------

# Every plain MLP network (SimplePINN, FractionalPINN, GradientEnhancedPINN,
# VPINN, SubdomainNetwork) performs the identical dense-stack computation, so a
# single pinned reference covers all five for the same dims/seed.
_MLP_BASELINE = jnp.asarray(
    [
        [0.4535508155822754, 0.3801552951335907],
        [-0.11338821798563004, -0.02297447808086872],
        [-0.5229619145393372, -0.17336246371269226],
    ]
)

_GATING_BASELINE_T1 = jnp.asarray(
    [
        [0.2960303723812103, 0.4357049763202667, 0.26826462149620056],
        [0.32931894063949585, 0.37053045630455017, 0.3001505732536316],
        [0.3555891215801239, 0.30688315629959106, 0.33752766251564026],
    ]
)

_GATING_BASELINE_T2 = jnp.asarray(
    [
        [0.3159421384334564, 0.3832972049713135, 0.3007606863975525],
        [0.33162936568260193, 0.3517681956291199, 0.3166023790836334],
        [0.3444397747516632, 0.31998202204704285, 0.33557820320129395],
    ]
)

_INPUT_DIM = 4
_OUTPUT_DIM = 2
_HIDDEN_DIMS = [8, 6]


def _mlp_input() -> jax.Array:
    """Fixed (3, 4) input shared by every plain-MLP baseline."""
    return jnp.linspace(-1.0, 1.0, 12).reshape(3, 4)


def _gating_input() -> jax.Array:
    """Fixed (3, 2) input for the gating network baseline."""
    return jnp.linspace(-1.0, 1.0, 6).reshape(3, 2)


# ---------------------------------------------------------------------------
# Baseline equivalence — the refactor must change nothing
# ---------------------------------------------------------------------------


class TestMLPNetworkBaselines:
    """Forward output of each plain-MLP network is bit-identical to baseline."""

    def test_simple_pinn(self) -> None:
        net = SimplePINN(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        np.testing.assert_allclose(net(_mlp_input()), _MLP_BASELINE, rtol=0, atol=0)

    def test_fractional_pinn(self) -> None:
        net = FractionalPINN(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        np.testing.assert_allclose(net(_mlp_input()), _MLP_BASELINE, rtol=0, atol=0)

    def test_gradient_enhanced_pinn(self) -> None:
        net = GradientEnhancedPINN(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        np.testing.assert_allclose(net(_mlp_input()), _MLP_BASELINE, rtol=0, atol=0)

    def test_vpinn(self) -> None:
        net = VPINN(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        np.testing.assert_allclose(net(_mlp_input()), _MLP_BASELINE, rtol=0, atol=0)

    def test_subdomain_network(self) -> None:
        net = SubdomainNetwork(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        np.testing.assert_allclose(net(_mlp_input()), _MLP_BASELINE, rtol=0, atol=0)


class TestGatingNetworkBaseline:
    """Gating network keeps its softmax behaviour after the extraction."""

    def test_default_temperature(self) -> None:
        net = GatingNetwork(
            input_dim=2,
            num_subdomains=3,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        np.testing.assert_allclose(net(_gating_input()), _GATING_BASELINE_T1, rtol=0, atol=0)

    def test_custom_temperature(self) -> None:
        net = GatingNetwork(
            input_dim=2,
            num_subdomains=3,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        np.testing.assert_allclose(
            net(_gating_input(), temperature=2.0), _GATING_BASELINE_T2, rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# Shared module behaviour
# ---------------------------------------------------------------------------


class TestDenseStack:
    """Direct tests of the shared :class:`DenseStack`."""

    def test_matches_mlp_baseline_without_dtype(self) -> None:
        """A plain dense-stack reproduces the shared MLP baseline."""
        stack = DenseStack(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        np.testing.assert_allclose(stack(_mlp_input()), _MLP_BASELINE, rtol=0, atol=0)

    def test_layer_count(self) -> None:
        """Hidden + output layers are all created."""
        stack = DenseStack(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        assert len(stack.layers) == len(_HIDDEN_DIMS) + 1

    def test_dtype_aware_output_dtype(self) -> None:
        """A compute dtype casts the output to that dtype."""
        stack = DenseStack(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            compute_dtype=jnp.float16,
            param_dtype=jnp.float16,
            rngs=nnx.Rngs(0),
        )
        out = stack(_mlp_input().astype(jnp.float16))
        assert out.dtype == jnp.float16

    def test_dense_stack_jit_grad_vmap(self) -> None:
        """Shared module survives jit, grad and vmap (NNX transforms)."""
        stack = DenseStack(
            input_dim=_INPUT_DIM,
            output_dim=_OUTPUT_DIM,
            hidden_dims=_HIDDEN_DIMS,
            rngs=nnx.Rngs(0),
        )
        x = _mlp_input()

        # jit
        jitted = nnx.jit(lambda m, inp: m(inp))
        out = jitted(stack, x)
        np.testing.assert_allclose(out, _MLP_BASELINE, rtol=1e-6, atol=1e-6)

        # grad — scalar loss through the module
        def loss_fn(model: DenseStack, inp: jax.Array) -> jax.Array:
            return jnp.sum(model(inp) ** 2)

        grads = nnx.grad(loss_fn)(stack, x)
        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves, "expected gradient leaves for the dense stack"
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

        # vmap over a batch dimension of single rows
        batched = jnp.stack([x, x, x])  # (3, 3, 4)
        vmapped = nnx.vmap(lambda m, inp: m(inp), in_axes=(None, 0))(stack, batched)
        assert vmapped.shape == (3, 3, _OUTPUT_DIM)
        for row in vmapped:
            np.testing.assert_allclose(row, _MLP_BASELINE, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
