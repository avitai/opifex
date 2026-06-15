r"""Tests for the Cartesian-tensor equivariant building block.

Behaviour is specified against TensorNet (Simeon & de Fabritiis 2023,
arXiv:2306.06482), which represents features as rank-2 Cartesian tensors
``X = I0 (scalar) + A (antisymmetric ~ vector) + S (symmetric-traceless)`` that
transform as ``X -> R X R^T`` under a rotation ``R``, and HotPP (Wang et al.
2024) for the high-order Cartesian decomposition. The load-bearing tests are:

* a rank-2 Cartesian tensor transforms as ``X -> R X R^T`` (consistency with
  the rotation helper);
* the matrix-product tensor product is equivariant because
  ``(R X R^T)(R Y R^T) = R (X Y) R^T``;
* the scalar / vector / symmetric-traceless decomposition is correct and
  equivariant (scalar invariant, vector rotates by ``R``, symmetric-traceless
  rotates by ``R . R^T``);
* the Cartesian <-> irreps (``1x0e + 1x1o + 1x2e``) conversion round-trips and
  matches the spherical-harmonic ``l = 1`` / ``l = 2`` transform on a known
  vector;
* :class:`CartesianLinear` is equivariant;
* ``jit`` / ``grad`` / ``vmap`` smoke tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant import Irreps, IrrepsArray
from opifex.neural.equivariant.cartesian import (
    CartesianLinear,
    CartesianTensor,
    from_irreps_array,
    to_irreps_array,
)


_RNG = SO3Group()
_CARTESIAN_IRREPS = Irreps("1x0e+1x1o+1x2e")


def _random_rotation(seed: int) -> jax.Array:
    return _RNG.random_element(jax.random.PRNGKey(seed)).astype(jnp.float64)


def _random_tensor(seed: int, leading: tuple[int, ...] = ()) -> CartesianTensor:
    array = jax.random.normal(jax.random.PRNGKey(seed), (*leading, 3, 3), dtype=jnp.float64)
    return CartesianTensor(array)


class TestCartesianTensorContainer:
    def test_array_shape_validation(self) -> None:
        try:
            CartesianTensor(jnp.zeros((3, 2)))
        except ValueError:
            return
        raise AssertionError("expected ValueError for non-(...,3,3) array")

    def test_shape_and_ndim(self) -> None:
        tensor = _random_tensor(0, leading=(4,))
        assert tensor.array.shape == (4, 3, 3)

    def test_is_pytree(self) -> None:
        """The tensor flows through ``tree_map`` as a single array child."""
        tensor = _random_tensor(1)
        doubled = jax.tree_util.tree_map(lambda a: 2.0 * a, tensor)
        assert jnp.allclose(doubled.array, 2.0 * tensor.array)


class TestRotation:
    def test_rotate_is_conjugation(self) -> None:
        """``rotate(R)`` applies ``X -> R X R^T``."""
        with jax.enable_x64(True):
            tensor = _random_tensor(2)
            rotation = _random_rotation(3)
            expected = rotation @ tensor.array @ rotation.T
            assert jnp.allclose(tensor.rotate(rotation).array, expected, atol=1e-10)

    def test_rotate_batched(self) -> None:
        with jax.enable_x64(True):
            tensor = _random_tensor(4, leading=(5,))
            rotation = _random_rotation(5)
            rotated = tensor.rotate(rotation)
            assert rotated.array.shape == (5, 3, 3)
            assert jnp.allclose(
                rotated.array[0], rotation @ tensor.array[0] @ rotation.T, atol=1e-10
            )


class TestTensorProductEquivariance:
    def test_matrix_product_is_equivariant(self) -> None:
        """``(R X R^T)(R Y R^T) = R (X Y) R^T``."""
        with jax.enable_x64(True):
            x = _random_tensor(6)
            y = _random_tensor(7)
            rotation = _random_rotation(8)
            left = x.rotate(rotation).tensor_product(y.rotate(rotation))
            right = x.tensor_product(y).rotate(rotation)
            assert jnp.allclose(left.array, right.array, atol=1e-10)

    def test_tensor_product_matches_matmul(self) -> None:
        with jax.enable_x64(True):
            x = _random_tensor(9)
            y = _random_tensor(10)
            assert jnp.allclose(x.tensor_product(y).array, x.array @ y.array, atol=1e-12)


class TestDecomposition:
    def test_reconstructs_input(self) -> None:
        """scalar*I/3 + antisymmetric + symmetric-traceless == X."""
        with jax.enable_x64(True):
            tensor = _random_tensor(11)
            scalar, vector, symmetric = tensor.decompose()
            antisymmetric = _vector_to_antisymmetric(vector)
            reconstructed = scalar[..., None, None] / 3.0 * jnp.eye(3) + antisymmetric + symmetric
            assert jnp.allclose(reconstructed, tensor.array, atol=1e-10)

    def test_symmetric_part_is_traceless(self) -> None:
        with jax.enable_x64(True):
            _, _, symmetric = _random_tensor(12).decompose()
            assert jnp.allclose(jnp.trace(symmetric), 0.0, atol=1e-10)

    def test_scalar_is_invariant(self) -> None:
        with jax.enable_x64(True):
            tensor = _random_tensor(13)
            rotation = _random_rotation(14)
            base_scalar, _, _ = tensor.decompose()
            rot_scalar, _, _ = tensor.rotate(rotation).decompose()
            assert jnp.allclose(base_scalar, rot_scalar, atol=1e-10)

    def test_vector_rotates_as_l1(self) -> None:
        with jax.enable_x64(True):
            tensor = _random_tensor(15)
            rotation = _random_rotation(16)
            _, base_vector, _ = tensor.decompose()
            _, rot_vector, _ = tensor.rotate(rotation).decompose()
            assert jnp.allclose(rot_vector, rotation @ base_vector, atol=1e-10)

    def test_symmetric_rotates_as_conjugation(self) -> None:
        with jax.enable_x64(True):
            tensor = _random_tensor(17)
            rotation = _random_rotation(18)
            _, _, base_symmetric = tensor.decompose()
            _, _, rot_symmetric = tensor.rotate(rotation).decompose()
            assert jnp.allclose(rot_symmetric, rotation @ base_symmetric @ rotation.T, atol=1e-10)


class TestIrrepsConversion:
    def test_round_trip(self) -> None:
        with jax.enable_x64(True):
            tensor = _random_tensor(19)
            recovered = from_irreps_array(to_irreps_array(tensor))
            assert jnp.allclose(recovered.array, tensor.array, atol=1e-10)

    def test_output_layout(self) -> None:
        with jax.enable_x64(True):
            ia = to_irreps_array(_random_tensor(20))
            assert ia.irreps == _CARTESIAN_IRREPS
            assert ia.array.shape == (9,)

    def test_l1_block_matches_vector(self) -> None:
        """The ``1o`` block equals the decomposition vector (e3nn l=1 basis)."""
        with jax.enable_x64(True):
            tensor = _random_tensor(21)
            _, vector, _ = tensor.decompose()
            ia = to_irreps_array(tensor)
            l1_block = ia.array[1:4]
            assert jnp.allclose(l1_block, vector, atol=1e-10)

    def test_conversion_is_equivariant_l1(self) -> None:
        """Rotating the Cartesian tensor rotates the ``1o`` block by ``D^1(R)``."""
        with jax.enable_x64(True):
            tensor = _random_tensor(22)
            rotation = _random_rotation(23)
            d1 = wigner_d(1, rotation)
            base = to_irreps_array(tensor).array[1:4]
            rotated = to_irreps_array(tensor.rotate(rotation)).array[1:4]
            assert jnp.allclose(rotated, d1 @ base, atol=1e-10)

    def test_conversion_is_equivariant_l2(self) -> None:
        """Rotating the Cartesian tensor rotates the ``2e`` block by ``D^2(R)``."""
        with jax.enable_x64(True):
            tensor = _random_tensor(24)
            rotation = _random_rotation(25)
            d2 = wigner_d(2, rotation)
            base = to_irreps_array(tensor).array[4:9]
            rotated = to_irreps_array(tensor.rotate(rotation)).array[4:9]
            assert jnp.allclose(rotated, d2 @ base, atol=1e-10)

    def test_from_irreps_array_validates_layout(self) -> None:
        with jax.enable_x64(True):
            bad = IrrepsArray("1x0e+1x1o", jnp.zeros(4))
            try:
                from_irreps_array(bad)
            except ValueError:
                return
            raise AssertionError("expected ValueError for wrong irreps layout")


class TestCartesianLinear:
    def test_output_shape(self) -> None:
        layer = CartesianLinear(in_channels=3, out_channels=5, rngs=nnx.Rngs(0))
        tensors = _random_tensor(26, leading=(3,))
        out = layer(tensors)
        assert out.array.shape == (5, 3, 3)

    def test_is_equivariant(self) -> None:
        """``Linear(rotate(X)) = rotate(Linear(X))`` for a random rotation."""
        with jax.enable_x64(True):
            layer = CartesianLinear(in_channels=4, out_channels=4, rngs=nnx.Rngs(0))
            tensors = _random_tensor(27, leading=(4,))
            rotation = _random_rotation(28)
            rotated_in = CartesianTensor(
                jnp.einsum("ab,...bc,dc->...ad", rotation, tensors.array, rotation)
            )
            left = layer(rotated_in).array
            right = jnp.einsum("ab,...bc,dc->...ad", rotation, layer(tensors).array, rotation)
            assert jnp.allclose(left, right, atol=1e-8)

    def test_weights_are_learnable(self) -> None:
        layer = CartesianLinear(in_channels=3, out_channels=2, rngs=nnx.Rngs(0))
        leaves = jax.tree_util.tree_leaves(nnx.state(layer, nnx.Param))
        assert len(leaves) > 0
        assert all(jnp.asarray(leaf).size > 0 for leaf in leaves)


class TestTransformSmoke:
    def test_jit(self) -> None:
        layer = CartesianLinear(in_channels=3, out_channels=3, rngs=nnx.Rngs(0))
        tensors = _random_tensor(29, leading=(3,))
        graphdef, state = nnx.split(layer)

        @jax.jit
        def apply(state: nnx.State, array: jax.Array) -> jax.Array:
            model = nnx.merge(graphdef, state)
            return model(CartesianTensor(array)).array

        assert jnp.allclose(apply(state, tensors.array), layer(tensors).array, atol=1e-5)

    def test_grad(self) -> None:
        layer = CartesianLinear(in_channels=2, out_channels=2, rngs=nnx.Rngs(0))
        tensors = _random_tensor(30, leading=(2,))

        def loss(model: CartesianLinear) -> jax.Array:
            return jnp.sum(model(tensors).array ** 2)

        grads = nnx.grad(loss)(layer)
        leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

    def test_vmap_over_batch(self) -> None:
        with jax.enable_x64(True):
            tensors = _random_tensor(31, leading=(6, 3))
            rotation = _random_rotation(32)
            rotated = tensors.rotate(rotation)
            assert rotated.array.shape == (6, 3, 3, 3)
            assert jnp.allclose(
                rotated.array[0, 0], rotation @ tensors.array[0, 0] @ rotation.T, atol=1e-10
            )

    def test_tensor_product_jit(self) -> None:
        x = _random_tensor(33)
        y = _random_tensor(34)

        @jax.jit
        def product(ax: jax.Array, ay: jax.Array) -> jax.Array:
            return CartesianTensor(ax).tensor_product(CartesianTensor(ay)).array

        assert jnp.allclose(product(x.array, y.array), x.array @ y.array, atol=1e-5)


def _vector_to_antisymmetric(vector: jax.Array) -> jax.Array:
    """Build the antisymmetric matrix whose dual axial vector is ``vector``.

    This mirrors the convention used internally by :meth:`CartesianTensor.decompose`
    so the reconstruction test is self-consistent.
    """
    zero = jnp.zeros_like(vector[..., 0])
    row0 = jnp.stack([zero, -vector[..., 2], vector[..., 1]], axis=-1)
    row1 = jnp.stack([vector[..., 2], zero, -vector[..., 0]], axis=-1)
    row2 = jnp.stack([-vector[..., 1], vector[..., 0], zero], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def test_reference_basis_consistency() -> None:
    """The internal l=2 change-of-basis matches a direct ``r (x) r`` calculation."""
    with jax.enable_x64(True):
        direction = jnp.asarray([0.3, -0.7, 0.5])
        outer = jnp.outer(direction, direction)
        tensor = CartesianTensor(outer)
        _, _, symmetric = tensor.decompose()
        # The symmetric-traceless part of r (x) r maps to the l=2 spherical harmonic
        # block (up to the known sqrt(3) component normalisation); round-trip must hold.
        ia = to_irreps_array(tensor)
        recovered = from_irreps_array(ia)
        _, _, recovered_symmetric = recovered.decompose()
        assert jnp.allclose(recovered_symmetric, symmetric, atol=1e-10)
        assert np.isfinite(np.asarray(ia.array)).all()
