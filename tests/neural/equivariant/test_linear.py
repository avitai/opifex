r"""Tests for the equivariant linear layer.

Behaviour is specified against ``e3nn-jax``'s ``FunctionalLinear``
(``../e3nn-jax/e3nn_jax/_src/linear.py``): a linear map that mixes only the
multiplicities of input irreps that share the *same* ``(l, p)`` with an output
irrep -- the constraint that makes the map equivariant.

The load-bearing test is numerical equivariance ``f(D . x) = D . f(x)``: applying
a Wigner-D rotation to each input block must commute with the layer.  Output
irreps, learnability and ``jit``/``grad``/``vmap`` are also checked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import EquivariantLinear, Irreps, IrrepsArray


_RNG = SO3Group()


def _rotate(x: IrrepsArray, rotation: jax.Array) -> IrrepsArray:
    """Apply the Wigner-D rotation block-by-block to an :class:`IrrepsArray`."""
    rotated_chunks = []
    for (_, irrep), chunk in zip(x.irreps.blocks, x.chunks, strict=True):
        matrix = wigner_d(irrep.l, rotation).astype(chunk.dtype)
        rotated_chunks.append(jnp.einsum("ij,...uj->...ui", matrix, chunk))
    flat = [c.reshape(*c.shape[:-2], -1) for c in rotated_chunks]
    return IrrepsArray(x.irreps, jnp.concatenate(flat, axis=-1))


def _random_input(irreps: Irreps, seed: int, dtype: jnp.dtype = jnp.float32) -> IrrepsArray:
    array = jax.random.normal(jax.random.PRNGKey(seed), (irreps.dim,), dtype=dtype)
    return IrrepsArray(irreps, array)


class TestEquivariantLinear:
    def test_output_irreps(self) -> None:
        layer = EquivariantLinear("4x0e+2x1o", "8x0e+3x1o", rngs=nnx.Rngs(0))
        result = layer(_random_input(Irreps("4x0e+2x1o"), 1))
        assert result.irreps == Irreps("8x0e+3x1o")
        assert result.array.shape == (Irreps("8x0e+3x1o").dim,)

    def test_only_matching_irreps_connected(self) -> None:
        """An output irrep with no matching input ``(l, p)`` yields zeros."""
        layer = EquivariantLinear("4x0e", "2x1o", rngs=nnx.Rngs(0))
        result = layer(_random_input(Irreps("4x0e"), 2))
        assert jnp.allclose(result.array, 0.0)

    def test_equivariance(self) -> None:
        """``f(D . x) = D . f(x)`` for a random rotation (float64)."""
        with jax.enable_x64(True):
            irreps_in, irreps_out = Irreps("3x0e+2x1o+1x2e"), Irreps("2x0e+3x1o+2x2e")
            layer = EquivariantLinear(irreps_in, irreps_out, rngs=nnx.Rngs(0))
            x = _random_input(irreps_in, 3, dtype=jnp.float64)
            rotation = _RNG.random_element(jax.random.PRNGKey(4)).astype(jnp.float64)
            left = layer(_rotate(x, rotation))
            right = _rotate(layer(x), rotation)
            assert jnp.allclose(left.array, right.array, atol=1e-8)

    def test_weights_are_learnable(self) -> None:
        layer = EquivariantLinear("4x0e+2x1o", "4x0e+2x1o", rngs=nnx.Rngs(0))
        params = nnx.state(layer, nnx.Param)
        leaves = jax.tree_util.tree_leaves(params)
        assert len(leaves) > 0
        assert all(jnp.asarray(leaf).size > 0 for leaf in leaves)

    def test_grad_flows_to_weights(self) -> None:
        layer = EquivariantLinear("4x0e+2x1o", "4x0e+2x1o", rngs=nnx.Rngs(0))
        x = _random_input(Irreps("4x0e+2x1o"), 5)

        def loss(model: EquivariantLinear) -> jax.Array:
            return jnp.sum(model(x).array ** 2)

        grads = nnx.grad(loss)(layer)
        leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

    def test_jit_compatibility(self) -> None:
        layer = EquivariantLinear("4x0e+2x1o", "6x0e+2x1o", rngs=nnx.Rngs(0))
        x = _random_input(Irreps("4x0e+2x1o"), 6)
        graphdef, state = nnx.split(layer)

        @jax.jit
        def apply(state: nnx.State, array: jax.Array) -> jax.Array:
            model = nnx.merge(graphdef, state)
            return model(IrrepsArray("4x0e+2x1o", array)).array

        assert jnp.allclose(apply(state, x.array), layer(x).array, atol=1e-5)

    def test_vmap_over_batch(self) -> None:
        layer = EquivariantLinear("4x0e+2x1o", "6x0e+2x1o", rngs=nnx.Rngs(0))
        irreps_in = Irreps("4x0e+2x1o")
        batch = jax.random.normal(jax.random.PRNGKey(7), (5, irreps_in.dim))
        result = layer(IrrepsArray(irreps_in, batch))
        assert result.array.shape == (5, Irreps("6x0e+2x1o").dim)
        single = layer(IrrepsArray(irreps_in, batch[0]))
        assert jnp.allclose(result.array[0], single.array, atol=1e-5)
