"""Tests for N-D / 3D domain decomposition (FBPINN tiling).

TDD: these tests define the expected behaviour of the N-D generalisation of
``uniform_partition`` and the FBPINN partition-of-unity for 3D domains.

Reference:
    Moseley, Markham, Nissen-Meyer (2023), "Finite Basis Physics-Informed
    Neural Networks", arXiv:2107.07871. The FBPINN decomposes the domain into
    overlapping subdomains laid out on a tensor-product grid, each with a smooth
    partition-of-unity window; the global solution is the windowed sum of the
    normalised local networks. The subdomain tiling and the window are tensor
    products across dimensions, so they extend naturally to N-D / 3D.

    Sibling reference: ``../FBPINNs`` -- ``fbpinns/decompositions.py``
    (``RectangularDecompositionND._get_level_params`` uses
    ``np.meshgrid(*subdomain_xs, indexing="ij")`` to lay out subdomains across
    arbitrary dimension counts).
"""

import itertools

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.pinns.domain_decomposition.base import uniform_partition
from opifex.neural.pinns.domain_decomposition.fbpinn import FBPINN, FBPINNConfig


class TestUniformPartition3D:
    """N-D tiling of a cube into the expected subdomains and interfaces."""

    def test_tiles_cube_into_expected_subdomains(self):
        """A (2,2,2) partition of the unit cube yields 8 subdomains."""
        subdomains, _interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2, 2),
        )

        assert len(subdomains) == 8
        for subdomain in subdomains:
            assert subdomain.bounds.shape == (3, 2)

    def test_non_uniform_partition_counts(self):
        """A (3,2,4) partition yields 3*2*4 = 24 subdomains."""
        subdomains, _interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 3.0], [0.0, 2.0], [0.0, 4.0]]),
            num_partitions=(3, 2, 4),
        )

        assert len(subdomains) == 24

    def test_subdomain_bounds_cover_cube_without_gaps(self):
        """Union of subdomain volumes equals the full cube volume."""
        subdomains, _interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2, 2),
        )

        total_volume = sum(float(s.volume) for s in subdomains)
        assert jnp.isclose(total_volume, 1.0)

    def test_corner_subdomain_has_expected_bounds(self):
        """First subdomain is the lower corner cell [0,0.5]^3."""
        subdomains, _interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2, 2),
        )

        assert jnp.allclose(
            subdomains[0].bounds,
            jnp.array([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]]),
        )

    def test_interface_count_matches_internal_faces(self):
        """A (2,2,2) cube has 3 axes * 1 internal cut * 4 transverse cells."""
        _subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2, 2),
        )

        # Per axis: (num-1) cuts * product of the other axes' cell counts.
        # 3 axes * 1 * 4 = 12 internal faces.
        assert len(interfaces) == 12

    def test_interfaces_have_axis_aligned_unit_normals(self):
        """Every 3D interface normal is an axis-aligned unit vector."""
        _subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2, 2),
        )

        for interface in interfaces:
            assert interface.normal.shape == (3,)
            assert jnp.isclose(jnp.linalg.norm(interface.normal), 1.0)
            # Exactly one component is non-zero.
            assert int(jnp.sum(interface.normal != 0.0)) == 1

    def test_interface_points_lie_on_shared_face(self):
        """Interface sample points have the shared coordinate fixed."""
        _subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2, 2),
            interface_points=5,
        )

        for interface in interfaces:
            assert interface.points.shape[1] == 3
            # The axis the normal points along has all points at one value.
            axis = int(jnp.argmax(jnp.abs(interface.normal)))
            coords = interface.points[:, axis]
            assert jnp.allclose(coords, coords[0])

    def test_interface_subdomain_ids_are_valid(self):
        """All interface subdomain ids index into the subdomain list."""
        subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2, 2),
        )

        num = len(subdomains)
        for interface in interfaces:
            left, right = interface.subdomain_ids
            assert 0 <= left < num
            assert 0 <= right < num
            assert left != right


class TestUniformPartitionGeneralND:
    """The N-D path must reproduce the original 1D / 2D behaviour and extend up."""

    def test_1d_unchanged(self):
        """1D partition still yields the documented bounds and one interface."""
        subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0]]),
            num_partitions=(2,),
        )

        assert len(subdomains) == 2
        assert len(interfaces) == 1
        assert jnp.allclose(subdomains[0].bounds, jnp.array([[0.0, 0.5]]))
        assert jnp.allclose(subdomains[1].bounds, jnp.array([[0.5, 1.0]]))

    def test_2d_unchanged(self):
        """2D (2,2) partition yields 4 subdomains and 4 internal interfaces."""
        subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2),
        )

        assert len(subdomains) == 4
        assert len(interfaces) == 4

    def test_subdomain_id_ordering_is_row_major(self):
        """Subdomain ids enumerate cells in row-major (C) order over the grid."""
        subdomains, _interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2),
        )

        # id == i*ny + j (row-major), matching the legacy 2D convention.
        assert jnp.allclose(subdomains[0].bounds, jnp.array([[0.0, 0.5], [0.0, 0.5]]))
        assert jnp.allclose(subdomains[1].bounds, jnp.array([[0.0, 0.5], [0.5, 1.0]]))
        assert jnp.allclose(subdomains[2].bounds, jnp.array([[0.5, 1.0], [0.0, 0.5]]))
        assert jnp.allclose(subdomains[3].bounds, jnp.array([[0.5, 1.0], [0.5, 1.0]]))

    def test_4d_partition(self):
        """The generalisation runs for 4D: (2,2,2,2) -> 16 subdomains."""
        subdomains, _interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0]] * 4),
            num_partitions=(2, 2, 2, 2),
        )

        assert len(subdomains) == 16

    def test_mismatched_lengths_raise(self):
        """num_partitions length must match the bounds dimension."""
        import pytest

        with pytest.raises(ValueError, match="must match"):
            uniform_partition(
                bounds=jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
                num_partitions=(2, 2),
            )


def _overlapping_cube_subdomains(half: float = 0.35):
    """Build 8 overlapping subdomains tiling the unit cube for FBPINN tests."""
    from opifex.neural.pinns.domain_decomposition.base import Subdomain

    centers = (0.25, 0.75)
    subdomains = []
    for idx, (cx, cy, cz) in enumerate(itertools.product(centers, centers, centers)):
        bounds = jnp.array(
            [
                [cx - half, cx + half],
                [cy - half, cy + half],
                [cz - half, cz + half],
            ]
        )
        subdomains.append(Subdomain(id=idx, bounds=bounds))
    return subdomains


class TestFBPINN3DPartitionOfUnity:
    """Partition-of-unity and local->global assembly in 3D."""

    def _make_model(self):
        return FBPINN(
            input_dim=3,
            output_dim=2,
            subdomains=_overlapping_cube_subdomains(),
            interfaces=[],
            hidden_dims=[8, 8],
            config=FBPINNConfig(window_type="cosine", normalize_windows=True),
            rngs=nnx.Rngs(0),
        )

    def test_window_weights_sum_to_one_over_3d_domain(self):
        """Normalised window weights form a partition of unity in the cube."""
        model = self._make_model()

        key = jax.random.PRNGKey(1)
        # Sample interior points well inside the overlapping cover.
        x = jax.random.uniform(key, (64, 3), minval=0.2, maxval=0.8)

        weights = model.compute_window_weights(x)
        sums = jnp.sum(weights, axis=-1)

        assert weights.shape == (64, 8)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_local_to_global_assembly_shape(self):
        """Global solution shape is (batch, output_dim) for a 3D FBPINN."""
        model = self._make_model()

        x = jax.random.uniform(jax.random.PRNGKey(2), (16, 3), minval=0.2, maxval=0.8)
        y = model(x)

        assert y.shape == (16, 2)
        assert jnp.isfinite(y).all()

    def test_windows_are_nonnegative(self):
        """Cosine windows are non-negative everywhere."""
        model = self._make_model()

        x = jax.random.uniform(jax.random.PRNGKey(3), (32, 3), minval=0.0, maxval=1.0)
        weights = model.compute_window_weights(x)

        assert (weights >= 0.0).all()


class TestFBPINN3DTransforms:
    """jit / grad / vmap smoke tests for the 3D FBPINN (JAX/NNX compatibility)."""

    def _make_model(self):
        return FBPINN(
            input_dim=3,
            output_dim=1,
            subdomains=_overlapping_cube_subdomains(),
            interfaces=[],
            hidden_dims=[8, 8],
            rngs=nnx.Rngs(0),
        )

    def test_jit(self):
        """The 3D forward pass compiles under nnx.jit."""
        model = self._make_model()

        @nnx.jit
        def forward(m, x):
            return m(x)

        x = jax.random.uniform(jax.random.PRNGKey(4), (8, 3), minval=0.2, maxval=0.8)
        y = forward(model, x)

        assert y.shape == (8, 1)
        assert jnp.isfinite(y).all()

    def test_grad(self):
        """A scalar loss is differentiable w.r.t. the 3D model parameters."""
        model = self._make_model()

        def loss_fn(m, x):
            return jnp.mean(m(x) ** 2)

        x = jax.random.uniform(jax.random.PRNGKey(5), (8, 3), minval=0.2, maxval=0.8)
        grads = nnx.grad(loss_fn)(model, x)

        leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
        assert leaves
        assert all(jnp.isfinite(leaf).all() for leaf in leaves)

    def test_vmap_over_inputs(self):
        """The window weights vectorise over a leading batch axis via vmap."""
        model = self._make_model()

        def per_point(x_row):
            return model.compute_window_weights(x_row[None, :])[0]

        x = jax.random.uniform(jax.random.PRNGKey(6), (10, 3), minval=0.2, maxval=0.8)
        weights = jax.vmap(per_point)(x)

        assert weights.shape == (10, 8)
        assert jnp.allclose(jnp.sum(weights, axis=-1), 1.0, atol=1e-5)
