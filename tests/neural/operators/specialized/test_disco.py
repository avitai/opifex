"""TDD contracts for discrete-continuous (DISCO) convolutions.

The shape contract mirrors the reference ``neuraloperator`` test
(``neuralop/layers/tests/test_disco_conv.py``): a point-cloud forward maps separate input/output
grids. The remaining tests pin the genuine DISCO properties that a standard discrete convolution
lacks — operation on irregular point sets, the per-output filter normalisation
(``torch_harmonics._normalize_convolution_filter_matrix``), discretisation invariance, and
JAX-transform compatibility. References: Ocampo et al. 2023 (``arXiv:2209.13603``);
``torch_harmonics``; ``neuraloperator``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.equivariant import PiecewiseLinearBasis
from opifex.neural.operators.specialized.disco import (
    build_disco_filter,
    DiscreteContinuousConv2d,
    regular_grid,
)


def test_disco_forward_shape_separate_in_out_grids() -> None:
    """Point-cloud forward maps an input grid to a different-resolution output grid (reference)."""
    in_coords, quad = regular_grid(16)  # 256 input points
    out_coords, _ = regular_grid(12)  # 144 output points
    conv = DiscreteContinuousConv2d(
        in_channels=6,
        out_channels=3,
        in_coords=in_coords,
        out_coords=out_coords,
        quad_weights=quad,
        num_basis=3,
        radius=0.25,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (4, 256, 6))
    out = conv(x)
    assert out.shape == (4, 144, 3)
    assert jnp.all(jnp.isfinite(out))


def test_disco_operates_on_irregular_point_set() -> None:
    """The genuine capability: DISCO convolves data on an irregular (random) point cloud."""
    key = jax.random.key(2)
    in_coords = jax.random.uniform(key, (200, 2))  # scattered, non-grid points
    out_coords = jax.random.uniform(jax.random.fold_in(key, 1), (50, 2))
    quad = jnp.full((200,), 1.0 / 200)  # Monte-Carlo quadrature weights
    conv = DiscreteContinuousConv2d(
        in_channels=2,
        out_channels=4,
        in_coords=in_coords,
        out_coords=out_coords,
        quad_weights=quad,
        num_basis=4,
        radius=0.3,
        rngs=nnx.Rngs(0),
    )
    out = conv(jax.random.normal(jax.random.fold_in(key, 2), (3, 200, 2)))
    assert out.shape == (3, 50, 4)
    assert jnp.all(jnp.isfinite(out))


def test_disco_filter_is_normalised_per_output_and_basis() -> None:
    """Each (output, basis) row of the filter sums to ~1 over inputs (reference normalisation)."""
    in_coords, quad = regular_grid(20)
    basis = PiecewiseLinearBasis(num_basis=4, cutoff=0.25)

    # Raw (un-normalised) support per (output, basis); rows whose raw support is well above the
    # eps floor must normalise to exactly 1 (partition of unity, per the reference).
    distances = jnp.linalg.norm(in_coords[:, None, :] - in_coords[None, :, :], axis=-1)
    raw_sums = jnp.sum(basis(distances) * quad[None, :, None], axis=1)  # (num_out, num_basis)

    psi = build_disco_filter(in_coords, in_coords, quad, basis)
    normalised_sums = jnp.sum(psi, axis=1)

    meaningful = raw_sums > 1e-4  # well above the 1e-9 normalisation floor
    assert bool(jnp.any(meaningful))
    assert jnp.allclose(normalised_sums[meaningful], 1.0, atol=1e-4)


def test_disco_is_discretisation_invariant() -> None:
    """The defining property: the SAME continuous kernel applied on a finer input grid yields a
    consistent output at fixed query points (the normalised quadrature converges to the integral).

    A standard pixel convolution has no such guarantee — its kernel is tied to the grid spacing.
    """
    # Fixed query (output) points, independent of input resolution.
    out_coords = jnp.array([[0.3, 0.3], [0.5, 0.5], [0.7, 0.4], [0.4, 0.6]])

    def smooth_field(coords: jax.Array) -> jax.Array:
        x, y = coords[:, 0], coords[:, 1]
        return (jnp.sin(2.0 * x) * jnp.cos(2.0 * y))[None, :, None]  # (1, N, 1)

    def disco_output(resolution: int) -> jax.Array:
        in_coords, quad = regular_grid(resolution)
        # Same rngs seed + same param shape => identical kernel weights across resolutions.
        conv = DiscreteContinuousConv2d(
            in_channels=1,
            out_channels=1,
            in_coords=in_coords,
            out_coords=out_coords,
            quad_weights=quad,
            num_basis=4,
            radius=0.3,
            use_bias=False,
            rngs=nnx.Rngs(0),
        )
        return conv(smooth_field(in_coords))[0, :, 0]

    coarse = disco_output(24)
    fine = disco_output(48)
    rel_diff = float(jnp.linalg.norm(coarse - fine) / (jnp.linalg.norm(fine) + 1e-9))
    assert rel_diff < 0.05  # consistent within 5% as the quadrature refines


def test_disco_is_jit_grad_vmap_safe() -> None:
    """A DISCO forward is jit/grad-compatible and vmaps over a batch of inputs."""
    in_coords, quad = regular_grid(12)
    conv = DiscreteContinuousConv2d(
        in_channels=1,
        out_channels=1,
        in_coords=in_coords,
        out_coords=in_coords,
        quad_weights=quad,
        num_basis=3,
        radius=0.3,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (2, 144, 1))

    graphdef, state = nnx.split(conv)

    @jax.jit
    def loss(state: nnx.State, x: jax.Array) -> jax.Array:
        return jnp.sum(nnx.merge(graphdef, state)(x) ** 2)

    grad = jax.grad(loss)(state, x)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grad))
    batched = jax.vmap(lambda xi: nnx.merge(graphdef, state)(xi[None])[0])(x)
    assert batched.shape == (2, 144, 1)
