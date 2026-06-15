"""Grid positional embedding for Fourier Neural Operators.

A spectral convolution is translation-equivariant, so an FNO whose only input is
the coefficient field cannot represent the *position-dependent* structure of a
boundary-value problem (e.g. the boundary layer of a Dirichlet Darcy solution) —
it can only memorise the training set. Concatenating normalised grid coordinates
as extra input channels gives the network absolute position and is the standard
fix (Li et al., 2020, "Fourier Neural Operator for Parametric PDEs"). On the
Darcy benchmark this is the difference between test rel-L2 ≈ 0.58 (no grid) and
≈ 0.01 (with grid).
"""

import jax
import jax.numpy as jnp


def append_grid_coordinates(x: jax.Array) -> jax.Array:
    """Concatenate normalised ``[0, 1]`` coordinate channels to a spatial field.

    Args:
        x: Input of shape ``(batch, channels, *spatial)`` with one or more
            spatial axes.

    Returns:
        Array of shape ``(batch, channels + n_spatial, *spatial)`` whose appended
        channels are the per-axis coordinate grids.
    """
    spatial = x.shape[2:]
    axes = [jnp.linspace(0.0, 1.0, size) for size in spatial]
    grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=0)
    grid = jnp.broadcast_to(grid[None], (x.shape[0], len(spatial), *spatial))
    return jnp.concatenate([x, grid], axis=1)
