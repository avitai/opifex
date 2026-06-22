"""Discrete-continuous (DISCO) convolutions on arbitrary point sets.

A DISCO convolution (Ocampo, Price & McEwen 2023, *Scalable and equivariant spherical CNNs by
discrete-continuous (DISCO) convolutions*, ``arXiv:2209.13603``; the algorithm implemented by
NVIDIA ``torch_harmonics`` and used in spherical neural operators for weather/climate)
parameterises the convolution kernel as a *continuous* function and evaluates the convolution as a
quadrature sum against the input samples:

    (kappa * f)(x_o) = integral kappa(x_o - x) f(x) dx  ~=  sum_i q_i kappa(x_o - x_i) f(x_i),

where ``q_i`` is the quadrature weight (the measure each input sample represents). The kernel
``kappa(r) = sum_k w_k phi_k(r)`` is a sum of fixed continuous radial basis functions ``phi_k`` with
learnable per-channel coefficients ``w_k``. The radial basis is the piecewise-linear hat basis of
the reference (opifex's :class:`~opifex.neural.equivariant.PiecewiseLinearBasis`, faithful to
``torch_harmonics``'s ``PiecewiseLinearFilterBasis``), and the filter is normalised per output point
and per basis function so each basis integrates to one against the quadrature
(``torch_harmonics``'s ``_normalize_convolution_filter_matrix``) — a partition of unity that gives
consistent magnitude and discretisation invariance.

Because the kernel is continuous and the sum is a quadrature, the operator is
*discretisation-aware*: it acts on arbitrary — including irregular and non-uniform — point
distributions, and the same learned kernel applied on a finer grid approximates the same
continuous operator. This is the capability standard discrete convolutions lack: the kernel lives
in physical coordinates, not pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.neural.equivariant import PiecewiseLinearBasis


if TYPE_CHECKING:
    from jaxtyping import Array, Float


def build_disco_filter(
    out_coords: jax.Array,
    in_coords: jax.Array,
    quad_weights: jax.Array,
    basis: PiecewiseLinearBasis,
    *,
    eps: float = 1e-9,
) -> jax.Array:
    """Build the normalised DISCO quadrature filter ``psi[o, i, k]``.

    For each output point ``o``, input point ``i`` and basis function ``k`` the entry is
    ``q_i * phi_k(|x_o - x_i|)`` normalised so ``sum_i psi[o, i, k] = 1``
    (``torch_harmonics``'s ``_normalize_convolution_filter_matrix``), making each basis a weighted
    average over the support.

    Args:
        out_coords: Output sample positions, shape ``(num_out, 2)``.
        in_coords: Input sample positions, shape ``(num_in, 2)``.
        quad_weights: Per-input quadrature weights, shape ``(num_in,)``.
        basis: The radial :class:`PiecewiseLinearBasis` (its ``cutoff`` is the support radius).
        eps: Normalisation floor.

    Returns:
        Filter tensor of shape ``(num_out, num_in, num_basis)``.
    """
    diff = out_coords[:, None, :] - in_coords[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)
    psi = basis(distances) * quad_weights[None, :, None]  # (num_out, num_in, num_basis)
    norm = jnp.sum(psi, axis=1, keepdims=True)  # (num_out, 1, num_basis)
    return psi / (norm + eps)


def regular_grid(resolution: int, extent: float = 1.0) -> tuple[jax.Array, jax.Array]:
    """Return ``(coords, quad_weights)`` for a uniform ``resolution x resolution`` grid on a square.

    ``coords`` is ``(resolution**2, 2)``; ``quad_weights`` is the uniform cell area
    ``(extent / resolution)**2`` for every point (the measure each sample represents).
    """
    axis = (jnp.arange(resolution) + 0.5) * (extent / resolution)
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    coords = jnp.stack([yy.reshape(-1), xx.reshape(-1)], axis=-1)
    cell_area = (extent / resolution) ** 2
    quad_weights = jnp.full((resolution * resolution,), cell_area)
    return coords, quad_weights


class DiscreteContinuousConv2d(nnx.Module):
    """Discrete-continuous convolution between two (possibly irregular) point sets.

    The convolution geometry — input/output sample positions and quadrature weights — is fixed
    at construction, so the normalised continuous-kernel quadrature filter is precomputed once. The
    learnable parameters are the per-basis channel-mixing weights
    ``(num_basis, in_channels, out_channels)`` and an optional bias; they are independent of the
    geometry, so the *same* learned kernel can be applied on a different grid (the resolution-
    transfer property).

    Forward input is ``(batch, num_in, in_channels)`` point features; output is
    ``(batch, num_out, out_channels)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_coords: jax.Array,
        out_coords: jax.Array,
        quad_weights: jax.Array,
        *,
        num_basis: int = 4,
        radius: float,
        use_bias: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        """Precompute the quadrature filter and initialise the per-basis channel weights."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_basis = num_basis
        self.radius = radius
        # Reference piecewise-linear radial basis; precompute the normalised (num_out, num_in,
        # num_basis) quadrature filter — a fixed function of the geometry, stored as a buffer.
        radial_basis = PiecewiseLinearBasis(num_basis=num_basis, cutoff=radius)
        self.filter = nnx.Variable(
            build_disco_filter(out_coords, in_coords, quad_weights, radial_basis)
        )
        scale = float(np.sqrt(1.0 / (in_channels * num_basis)))
        self.weight = nnx.Param(
            scale * jax.random.normal(rngs.params(), (num_basis, in_channels, out_channels))
        )
        self.bias = nnx.Param(jnp.zeros((out_channels,))) if use_bias else None

    def __call__(self, x: Float[Array, "batch num_in in_channels"]) -> jax.Array:
        """Apply the DISCO convolution: quadrature over input samples and the continuous kernel."""
        # out[b, o, d] = sum_{i, k} filter[o, i, k] * x[b, i, c] * weight[k, c, d]
        output = jnp.einsum("oik,bic,kcd->bod", self.filter.value, x, self.weight.value)
        if self.bias is not None:
            output = output + self.bias.value
        return output


__all__ = [
    "DiscreteContinuousConv2d",
    "build_disco_filter",
    "regular_grid",
]
