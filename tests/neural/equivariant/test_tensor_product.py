r"""Tests for the equivariant tensor product.

Behaviour is specified against ``e3nn-jax``'s tensor product
(``../e3nn-jax/e3nn_jax/_src/tensor_products.py`` and the ``uvw`` connection of
``../e3nn-jax/e3nn_jax/_src/legacy/core_tensor_product.py``): for each path
``(i1) x (i2) -> i3`` allowed by the selection rule and present in the target
``irreps_out``, contract the two input blocks with the Clebsch-Gordan tensor and
a learnable ``(mul1, mul2, mul3)`` weight.

The load-bearing test is numerical equivariance ``TP(D . x, D . y) = D . TP(x, y)``
for a random rotation.  The ``TensorProduct`` protocol, output irreps,
learnability and ``jit``/``grad``/``vmap`` are also checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import (
    FullyConnectedTensorProduct,
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


def _random_input(irreps: Irreps, seed: int, dtype: jnp.dtype = jnp.float32) -> IrrepsArray:
    array = jax.random.normal(jax.random.PRNGKey(seed), (irreps.dim,), dtype=dtype)
    return IrrepsArray(irreps, array)


class TestFullyConnectedTensorProduct:
    def test_is_tensor_product_protocol(self) -> None:
        layer = FullyConnectedTensorProduct("2x0e+1x1o", "1x1o", "3x1o", rngs=nnx.Rngs(0))
        assert isinstance(layer, TensorProduct)

    def test_output_irreps(self) -> None:
        layer = FullyConnectedTensorProduct("1x1o", "1x1o", "2x0e+1x1e", rngs=nnx.Rngs(0))
        result = layer(_random_input(Irreps("1x1o"), 1), _random_input(Irreps("1x1o"), 2))
        assert result.irreps == Irreps("2x0e+1x1e")

    def test_scalar_times_scalar(self) -> None:
        """``0e x 0e -> 0e`` reduces to a learnable scalar bilinear form."""
        layer = FullyConnectedTensorProduct("1x0e", "1x0e", "1x0e", rngs=nnx.Rngs(0))
        x = IrrepsArray("1x0e", jnp.asarray([2.0]))
        y = IrrepsArray("1x0e", jnp.asarray([3.0]))
        result = layer(x, y)
        assert result.array.shape == (1,)

    def test_equivariance(self) -> None:
        """``TP(D . x, D . y) = D . TP(x, y)`` for a random rotation (float64)."""
        with jax.enable_x64(True):
            irreps1, irreps2 = Irreps("2x0e+2x1o"), Irreps("1x0e+2x1o+1x2e")
            irreps_out = Irreps("3x0e+2x1o+1x1e+1x2e")
            layer = FullyConnectedTensorProduct(irreps1, irreps2, irreps_out, rngs=nnx.Rngs(0))
            x = _random_input(irreps1, 3, dtype=jnp.float64)
            y = _random_input(irreps2, 4, dtype=jnp.float64)
            rotation = _RNG.random_element(jax.random.PRNGKey(5)).astype(jnp.float64)
            left = layer(_rotate(x, rotation), _rotate(y, rotation))
            right = _rotate(layer(x, y), rotation)
            assert jnp.allclose(left.array, right.array, atol=1e-8)

    def test_invariant_output_is_rotation_invariant(self) -> None:
        """A pure-scalar output is invariant under joint rotation of the inputs."""
        with jax.enable_x64(True):
            layer = FullyConnectedTensorProduct("2x1o", "2x1o", "4x0e", rngs=nnx.Rngs(0))
            x = _random_input(Irreps("2x1o"), 6, dtype=jnp.float64)
            y = _random_input(Irreps("2x1o"), 7, dtype=jnp.float64)
            rotation = _RNG.random_element(jax.random.PRNGKey(8)).astype(jnp.float64)
            base = layer(x, y).array
            rotated = layer(_rotate(x, rotation), _rotate(y, rotation)).array
            assert jnp.allclose(base, rotated, atol=1e-8)

    def test_weights_are_learnable(self) -> None:
        layer = FullyConnectedTensorProduct("2x0e+1x1o", "1x1o", "2x1o+1x0e", rngs=nnx.Rngs(0))
        leaves = jax.tree_util.tree_leaves(nnx.state(layer, nnx.Param))
        assert len(leaves) > 0
        assert all(jnp.asarray(leaf).size > 0 for leaf in leaves)

    def test_grad_flows_to_weights(self) -> None:
        layer = FullyConnectedTensorProduct("1x1o", "1x1o", "1x0e+1x1e", rngs=nnx.Rngs(0))
        x = _random_input(Irreps("1x1o"), 9)
        y = _random_input(Irreps("1x1o"), 10)

        def loss(model: FullyConnectedTensorProduct) -> jax.Array:
            return jnp.sum(model(x, y).array ** 2)

        grads = nnx.grad(loss)(layer)
        leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

    def test_jit_compatibility(self) -> None:
        layer = FullyConnectedTensorProduct("1x1o", "1x1o", "2x0e+1x1e", rngs=nnx.Rngs(0))
        x = _random_input(Irreps("1x1o"), 11)
        y = _random_input(Irreps("1x1o"), 12)
        graphdef, state = nnx.split(layer)

        @jax.jit
        def apply(state: nnx.State, ax: jax.Array, ay: jax.Array) -> jax.Array:
            model = nnx.merge(graphdef, state)
            return model(IrrepsArray("1x1o", ax), IrrepsArray("1x1o", ay)).array

        assert jnp.allclose(apply(state, x.array, y.array), layer(x, y).array, atol=1e-5)

    def test_vmap_over_batch(self) -> None:
        layer = FullyConnectedTensorProduct("1x1o", "1x1o", "2x0e+1x1e", rngs=nnx.Rngs(0))
        keys = jax.random.split(jax.random.PRNGKey(13), 2)
        batch_x = jax.random.normal(keys[0], (4, 3))
        batch_y = jax.random.normal(keys[1], (4, 3))
        result = layer(IrrepsArray("1x1o", batch_x), IrrepsArray("1x1o", batch_y))
        assert result.array.shape[0] == 4
        single = layer(IrrepsArray("1x1o", batch_x[0]), IrrepsArray("1x1o", batch_y[0]))
        assert jnp.allclose(result.array[0], single.array, atol=1e-5)
