"""Neural Tangent Kernel (NTK) computation utilities.

This module implements efficient NTK computation using JAX automatic differentiation.
Formula: Θ_k = (1/m) J_k J_k^T
Where J_k is the Jacobian of the network outputs with respect to parameters.

References:
    - Survey Section 3.1: Neural Tangent Kernel
    - Jacot et al. (2018): Neural Tangent Kernel: Convergence and Generalization
"""

import jax
import jax.numpy as jnp
from flax import nnx


def compute_gradient_jacobian(model: nnx.Module, x: jax.Array) -> jax.Array:
    """Compute the flattened gradient Jacobian matrix.

    Args:
        model: Flax NNX module
        x: Input batch (batch_size, ...)

    Returns:
        Flattened Jacobian matrix of shape (batch_size, num_params)
    """
    # We need the Jacobian of the output w.r.t parameters for each sample
    # But NNX functional API handles state updates.
    # For NTK, we treat parameters as input to the function.

    # Using jax.jacobian on the functional API
    # 1. Get functional version
    graphdef, params = nnx.split(model)

    def functional_call(p, x_sample):
        m = nnx.merge(graphdef, p)
        # Assuming scalar output for simplest NTK case or flattening output
        out = m(x_sample)  # pyright: ignore[reportCallIssue]
        return out.reshape(-1)  # Flatten output dim

    # Jacobian: (batch_size, output_dim, num_params)
    # We want J wrt parameters.
    # For efficient computation, we vectorize over batch.

    # Single sample jacobian:
    # J_fn(params, x_i) -> (output_dim, num_params)

    # We need to flatten the params into a single vector first to get a matrix
    # Or keep them as PyTree and flatten later.

    batch_size = x.shape[0]

    # jacrev is efficient for wide networks (params > outputs)
    jac_fn = jax.jacrev(functional_call)

    # vmap over batch dimension of x
    batch_jac_fn = jax.vmap(jac_fn, in_axes=(None, 0))

    # Compute Jacobian PyTree: (batch, output_dim, params_pytree)
    J_pytree = batch_jac_fn(params, x)

    # Flatten parameters dimension
    # Leaves structure: (batch, output_dim, param_shape)
    leaves = jax.tree.leaves(J_pytree)

    # Reshape each leaf to (batch * output_dim, param_elements)
    # Then concat
    flat_leaves = []
    for leaf in leaves:
        # leaf shape: (batch, output_dim, p1, p2, ...)
        # flatten to (batch, output_dim, num_param_elements)
        # Actually for standard NTK we treat (batch) as the dimension of interest
        # The formula J J^T assumes we are looking at the kernel between samples.
        # If output_dim > 1, the NTK is usually defined as contracting over parameters
        # producing a (batch*output, batch*output) matrix or similar.
        # The survey defines: Θ_k = (1/m) J_k J_k^T
        # Usually for scalar output u(x), J is (batch, params).
        # For vector output, it's often block diagonal or tensor.
        # Let's assume user wants the scalar-like correlation or full kernel.
        # For simplicity in this implementation, and matching "1/m J J^T",
        # we treat the row index as "sample index".
        # If output > 1, effectively we have batch*output samples.

        # leaf_shape = leaf.shape
        # Flatten batch and output dims together for the "sample" dimension
        # But wait, NTK usually compares x_i and x_j.
        # For multi-output, we get a kernel K_{ij} which is a matrix.
        # Let's stick to the simplest interpretation:
        # Flatten output into the batch dimension? No, that mixes cross-correlations.

        # Let's flatten parameters only.
        # leaf: (batch, output_dim, p...)
        flat_leaf = leaf.reshape(
            batch_size, -1
        )  # (batch, output_dim * p...) is not right.

        # Let's restart the flattening logic carefully.
        # J should be (batch_size, num_params) if output is scalar.
        # If output is vector d_out, J is (batch_size, d_out, num_params).
        # Standard NTK contracts over parameters.
        # Result is (batch, batch, d_out, d_out) or (batch, d_out, batch, d_out).
        # The report says scalar u(x) in many places (PINN simple case).
        # Let's handle generic case:
        # Flatten params: (batch, d_out, P)
        # Flatten samples: (N, P) where N = batch * d_out

        # Reshape leaf to (batch, -1) effectively flattening output_dim and
        # param structure?
        # No, we want to preserve batch structure to return (batch, batch) if requested.

        # Let's implement the standard "empirical NTK" which is (batch, batch)
        # by contracting gradients. This implies scalar output or trace over outputs.
        # However, for survey compliance (2601.10222 eq 13), it treats u(theta)
        # as vector of size m.
        # This implies scalar output per sample.

        flat_leaf = leaf.reshape(batch_size, -1)
        flat_leaves.append(flat_leaf)

    # Concatenate all params: (batch, total_params)
    return jnp.concatenate(flat_leaves, axis=1)


def compute_ntk(model: nnx.Module, x: jax.Array) -> jax.Array:
    """Compute the Empirical Neural Tangent Kernel.

    Args:
        model: Flax NNX module
        x: Input batch (batch, ...)

    Returns:
        NTK matrix of shape (batch, batch).
        Contracted over output dimension if output > 1.
    """
    # Get Jacobian (batch, total_params)
    # Note: If output dim > 1, our helper flattens it into the param dimension
    # which effectively computes sum(grad_dim_k . grad_dim_k).
    # This is the trace of the NTK output-block.

    J = compute_gradient_jacobian(model, x)

    # Θ = (1/m) J J^T
    # The survey includes 1/m normalization factor in definition (Eq 13).
    m = x.shape[0]
    return (1.0 / m) * jnp.matmul(J, J.T)
