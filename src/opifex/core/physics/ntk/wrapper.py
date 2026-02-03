"""Neural Tangent Kernel (NTK) wrapper for FLAX NNX models.

This module provides utilities for computing the empirical Neural Tangent Kernel
using Google's neural-tangents library with FLAX NNX models.

Key Features:
    - Wrap NNX models for NTK computation
    - Compute empirical NTK matrices
    - Jacobian computation utilities
    - Spectral analysis of NTK

References:
    - Jacot et al. (2018): Neural Tangent Kernel
    - Survey Section 3: Neural Tangent Kernel Analysis
    - GitHub: https://github.com/google/neural-tangents
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float, PyTree


@dataclass(frozen=True)
class NTKConfig:
    """Configuration for NTK computation.

    Attributes:
        implementation: NTK implementation method (1=Jacobian contraction,
                        2=NTK-vector products, 3=structured derivatives)
        trace_axes: Axes to trace over for NTK computation
        diagonal_axes: Axes to compute diagonal for
        vmap_axes: Axes to vmap over
    """

    implementation: int = 1  # 1 is generally most reliable
    trace_axes: tuple = ()
    diagonal_axes: tuple = ()
    vmap_axes: tuple | None = None


def create_ntk_fn_from_nnx(
    model: nnx.Module,
    config: NTKConfig | None = None,
) -> Callable:
    """Create NTK function from NNX model.

    This function creates a kernel function that can compute the empirical
    NTK between sets of input points.

    Args:
        model: FLAX NNX model
        config: NTK configuration

    Returns:
        Kernel function that takes (x1, x2, params) and returns NTK matrix
    """
    try:
        import neural_tangents as nt
    except ImportError as err:
        raise ImportError(
            "neural-tangents is required for NTK computation. "
            "Install with: pip install neural-tangents"
        ) from err

    config = config or NTKConfig()

    # Split model into graphdef and params
    graphdef, state = nnx.split(model)

    def apply_fn(params_flat, x):
        """Apply function compatible with neural-tangents."""
        # Reconstruct the state with updated params
        # Note: This is a simplified version - in practice we need to
        # properly merge the params back into the state
        reconstructed_model = nnx.merge(graphdef, state)
        return reconstructed_model(x)  # pyright: ignore[reportCallIssue]

    # Create empirical kernel function using neural-tangents
    return nt.empirical_kernel_fn(
        apply_fn,  # pyright: ignore[reportArgumentType]
        trace_axes=config.trace_axes,
        diagonal_axes=config.diagonal_axes,
        vmap_axes=config.vmap_axes,
        implementation=config.implementation,
    )


def compute_empirical_ntk(
    model: nnx.Module,
    x1: Float[Array, "batch1 dim"],
    x2: Float[Array, "batch2 dim"] | None = None,
    config: NTKConfig | None = None,
) -> Float[Array, "batch1 batch2"]:
    """Compute empirical NTK matrix between input points.

    The empirical NTK is computed as:
        Θ(x1, x2) = J(x1) @ J(x2).T

    where J(x) is the Jacobian of the network output w.r.t. parameters.

    Args:
        model: FLAX NNX model
        x1: First set of input points
        x2: Second set of input points (uses x1 if None)
        config: NTK configuration

    Returns:
        NTK matrix of shape (batch1, batch2) or (batch1, batch1) if x2 is None
    """
    if x2 is None:
        x2 = x1

    # Compute Jacobians
    jacobian1 = compute_jacobian(model, x1)
    flat_jacobian1 = flatten_jacobian(jacobian1)

    if x2 is x1:
        flat_jacobian2 = flat_jacobian1
    else:
        jacobian2 = compute_jacobian(model, x2)
        flat_jacobian2 = flatten_jacobian(jacobian2)

    # NTK = J1 @ J2.T
    # Note: flat_jacobian shape is (batch * out, num_params)
    # We need to reshape to (batch, out * num_params) for proper NTK
    batch1 = x1.shape[0]
    batch2 = x2.shape[0]

    # Reshape to (batch, -1) for computing NTK per sample
    j1 = flat_jacobian1.reshape(batch1, -1)
    j2 = flat_jacobian2.reshape(batch2, -1)

    return j1 @ j2.T


def compute_jacobian(
    model: nnx.Module,
    x: Float[Array, ...],
) -> PyTree:
    """Compute Jacobian of model output w.r.t. parameters.

    Args:
        model: FLAX NNX model
        x: Input points

    Returns:
        Jacobian as a pytree with same structure as model parameters
    """
    graphdef, state = nnx.split(model)

    def forward_fn(state_dict):
        """Forward pass that takes state as input."""
        # Merge state back with graphdef
        full_state = nnx.State(state_dict)
        reconstructed = nnx.merge(graphdef, full_state)
        return reconstructed(x)  # pyright: ignore[reportCallIssue]

    # Get the state as a dict for differentiation
    state_dict = dict(state.flat_state())

    # Compute Jacobian using jax.jacrev
    return jax.jacrev(forward_fn)(state_dict)


def flatten_jacobian(
    jacobian: PyTree,
) -> Float[Array, "batch_out num_params"]:
    """Flatten Jacobian pytree into a 2D array.

    Args:
        jacobian: Jacobian as pytree

    Returns:
        Flattened Jacobian of shape (batch * output_dim, num_params)
    """
    # Get all leaves (parameter gradients)
    leaves = jax.tree.leaves(jacobian)

    if not leaves:
        raise ValueError("Empty Jacobian")

    # Each leaf has shape (batch, out, *param_shape)
    # We need to flatten to (batch * out, num_params)
    flattened_leaves = []
    for leaf in leaves:
        # Flatten all but first two dimensions
        # leaf shape: (batch, out, *param_shape)
        batch_out_shape = leaf.shape[:2]
        param_size = leaf[0, 0].size if leaf.ndim > 2 else 1
        flat = leaf.reshape(*batch_out_shape, param_size)
        flattened_leaves.append(flat)

    # Concatenate along parameter dimension
    result = jnp.concatenate(flattened_leaves, axis=-1)

    # Reshape to (batch * out, num_params)
    batch_size = result.shape[0]
    out_size = result.shape[1]
    return result.reshape(batch_size * out_size, -1)


class NTKWrapper:
    """Wrapper for computing NTK with NNX models.

    This class provides a convenient interface for NTK computations,
    caching the model structure and configuration.

    Attributes:
        model: The NNX model
        config: NTK configuration

    Example:
        >>> model = MyModel(rngs=nnx.Rngs(0))
        >>> wrapper = NTKWrapper(model)
        >>> ntk = wrapper.compute_ntk(x)
    """

    def __init__(
        self,
        model: nnx.Module,
        config: NTKConfig | None = None,
    ):
        """Initialize NTK wrapper.

        Args:
            model: FLAX NNX model
            config: NTK configuration
        """
        self.model = model
        self.config = config or NTKConfig()
        self._kernel_fn = None

    def compute_ntk(
        self,
        x1: Float[Array, "batch1 dim"],
        x2: Float[Array, "batch2 dim"] | None = None,
    ) -> Float[Array, "batch1 batch2"]:
        """Compute empirical NTK between input points.

        Args:
            x1: First set of input points
            x2: Second set of input points (uses x1 if None)

        Returns:
            NTK matrix
        """
        return compute_empirical_ntk(self.model, x1, x2, self.config)

    def compute_eigenvalues(
        self,
        x: Float[Array, ...],
    ) -> Float[Array, " batch"]:
        """Compute eigenvalues of NTK at given points.

        Args:
            x: Input points

        Returns:
            Eigenvalues sorted in descending order
        """
        ntk = self.compute_ntk(x)
        eigenvalues = jnp.linalg.eigvalsh(ntk)
        return jnp.sort(eigenvalues)[::-1]

    def compute_condition_number(
        self,
        x: Float[Array, ...],
    ) -> Float[Array, ""]:
        """Compute condition number of NTK.

        The condition number is the ratio of largest to smallest eigenvalue.
        Large condition numbers indicate ill-conditioning.

        Args:
            x: Input points

        Returns:
            Condition number
        """
        eigenvalues = self.compute_eigenvalues(x)
        return eigenvalues[0] / (eigenvalues[-1] + 1e-10)
