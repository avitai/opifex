r"""Tests for the channel-wise (e3nn ``"uuu"``) tensor product.

Behaviour is specified against ``e3nn`` ``TensorProduct(connection_mode="uuu")``
(``../AIRS/OpenDFT/QHBench/QH9/models/QHNet.py`` uses it for the self / pair
interaction layers) and ``e3nn.elementwise_tensor_product``
(``../e3nn-jax/e3nn_jax/_src/tensor_products.py``): inputs and output share one
uniform multiplicity ``mul``; each allowed path ``(l1, l2, l3)`` couples the two
inputs channel-wise with the Clebsch-Gordan tensor and a per-channel weight
(``O(mul)`` parameters, not ``O(mul^3)``). Weights may be the module's internal
parameters or supplied per-sample (external), the latter being how QHNet injects
per-edge radial / inner-product modulation.

The load-bearing test is numerical equivariance ``TP(D . x, D . y) = D . TP(x, y)``
for a random rotation; output irreps, the internal/external weight paths,
``weight_numel`` and ``jit``/``grad``/``vmap`` are also checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import (
    ChannelwiseTensorProduct,
    Irreps,
    IrrepsArray,
    TensorProduct,
)


_RNG = SO3Group()


def _rotate(x: IrrepsArray, rotation: jax.Array) -> IrrepsArray:
    rotated_chunks = []
    for (_, irrep), chunk in zip(x.irreps.blocks, x.chunks, strict=True):
        matrix = wigner_d(irrep.l, rotation).astype(chunk.dtype)
        rotated_chunks.append(jnp.einsum("ij,...uj->...ui", matrix, chunk))
    flat = [c.reshape(*c.shape[:-2], -1) for c in rotated_chunks]
    return IrrepsArray(x.irreps, jnp.concatenate(flat, axis=-1))


def _random_input(irreps: Irreps, seed: int, dtype: jnp.dtype = jnp.float64) -> IrrepsArray:
    array = jax.random.normal(jax.random.PRNGKey(seed), (irreps.dim,), dtype=dtype)
    return IrrepsArray(irreps, array)


class TestChannelwiseTensorProduct:
    def test_is_tensor_product_protocol(self) -> None:
        layer = ChannelwiseTensorProduct("4x0e+4x1e", "4x0e+4x1e", "4x0e+4x1e", rngs=nnx.Rngs(0))
        assert isinstance(layer, TensorProduct)

    def test_output_irreps(self) -> None:
        layer = ChannelwiseTensorProduct("2x1e", "2x1e", "2x0e+2x1e+2x2e", rngs=nnx.Rngs(0))
        out = layer(_random_input(Irreps("2x1e"), 1), _random_input(Irreps("2x1e"), 2))
        assert out.irreps == Irreps("2x0e+2x1e+2x2e")

    def test_channelwise_param_count_is_linear_in_mul(self) -> None:
        """``"uuu"`` carries one weight per channel per path, not ``mul^3``."""
        layer = ChannelwiseTensorProduct("8x1e", "8x1e", "8x0e+8x1e+8x2e", rngs=nnx.Rngs(0))
        # 1e x 1e -> {0e, 1e, 2e}: three paths, each 8 channel weights = 24.
        assert layer.weight_numel == 24

    def test_equivariance_internal_weights(self) -> None:
        """``TP(D x, D y) = D TP(x, y)`` for the internal-weight forward."""
        irreps = Irreps("3x0e+3x1e+3x2e")
        layer = ChannelwiseTensorProduct(irreps, irreps, irreps, rngs=nnx.Rngs(3))
        x = _random_input(irreps, 4)
        y = _random_input(irreps, 5)
        rotation = _RNG.random_element(jax.random.PRNGKey(6)).astype(jnp.float64)
        lhs = _rotate(layer(x, y), rotation).array
        rhs = layer(_rotate(x, rotation), _rotate(y, rotation)).array
        assert jnp.allclose(lhs, rhs, atol=1e-5)

    def test_external_weights_override_internal(self) -> None:
        """Supplying ``weights`` uses them instead of the module parameters."""
        irreps = Irreps("2x0e+2x1e")
        layer = ChannelwiseTensorProduct(irreps, irreps, irreps, rngs=nnx.Rngs(7))
        x = _random_input(irreps, 8)
        y = _random_input(irreps, 9)
        zeros = jnp.zeros((layer.weight_numel,))
        out = layer(x, y, weights=zeros)
        assert jnp.allclose(out.array, 0.0, atol=1e-7)

    def test_external_weights_equivariant(self) -> None:
        """External-weight forward is equivariant for fixed (invariant) weights."""
        irreps = Irreps("4x0e+4x1e+4x2e")
        layer = ChannelwiseTensorProduct(irreps, irreps, irreps, rngs=nnx.Rngs(10))
        x = _random_input(irreps, 11)
        y = _random_input(irreps, 12)
        weights = jax.random.normal(jax.random.PRNGKey(13), (layer.weight_numel,))
        rotation = _RNG.random_element(jax.random.PRNGKey(14)).astype(jnp.float64)
        lhs = _rotate(layer(x, y, weights=weights), rotation).array
        rhs = layer(_rotate(x, rotation), _rotate(y, rotation), weights=weights).array
        assert jnp.allclose(lhs, rhs, atol=1e-5)

    def test_requires_uniform_multiplicity(self) -> None:
        """Mixed multiplicities are rejected (``"uuu"`` needs a single ``mul``)."""
        try:
            ChannelwiseTensorProduct("2x0e+4x1e", "2x0e+4x1e", "2x0e+4x1e", rngs=nnx.Rngs(0))
        except ValueError:
            return
        raise AssertionError("expected ValueError for non-uniform multiplicity")

    def test_jit_grad_vmap(self) -> None:
        """The forward is jit/grad/vmap clean over a batch of external weights."""
        irreps = Irreps("2x0e+2x1e")
        layer = ChannelwiseTensorProduct(irreps, irreps, irreps, rngs=nnx.Rngs(15))
        xb = IrrepsArray(irreps, jax.random.normal(jax.random.PRNGKey(16), (5, irreps.dim)))
        yb = IrrepsArray(irreps, jax.random.normal(jax.random.PRNGKey(17), (5, irreps.dim)))
        wb = jax.random.normal(jax.random.PRNGKey(18), (5, layer.weight_numel))

        @nnx.jit
        def run(module: ChannelwiseTensorProduct) -> jax.Array:
            out = jax.vmap(lambda a, b, w: module(a, b, weights=w).array)(xb, yb, wb)
            return jnp.sum(out**2)

        grad = nnx.grad(run)(layer)
        assert grad is not None
