"""Spectrum Analysis utilities for Neural Tangent Kernel.

This module provides tools to analyze the spectral properties of the NTK,
which are crucial for understanding training dynamics and convergence rates.
It implements eigenvalue decomposition and related metrics.

References:
    - Survey Section 3.1: NTK Spectrum
    - Survey Section 7: Conditioning
    - Rahaman et al. (2019): On the Spectral Bias of Neural Networks
"""

import jax
import jax.numpy as jnp


def compute_ntk_spectrum(ntk_matrix: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute eigenvalues and eigenvectors of the NTK matrix.

    Args:
        ntk_matrix: Symmetrix positive semi-definite NTK matrix (N, N).

    Returns:
        Tuple of (eigenvalues, eigenvectors).
        Eigenvalues are sorted in descending order.
    """
    # jnp.linalg.eigh is for Hermitian/Symmetric matrices
    # It returns eigenvalues in ascending order
    eigenvalues, eigenvectors = jnp.linalg.eigh(ntk_matrix)

    # Sort descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    return eigenvalues, eigenvectors


def compute_condition_number(ntk_matrix: jax.Array) -> jax.Array:
    """Compute the condition number of the NTK matrix.

    Condition number kappa = lambda_max / lambda_min

    Args:
        ntk_matrix: NTK matrix (N, N).

    Returns:
        Condition number (scalar).
    """
    eigenvalues = jnp.linalg.eigvalsh(ntk_matrix)
    lambda_max = jnp.max(eigenvalues)
    lambda_min = jnp.min(eigenvalues)

    # Avoid division by zero
    lambda_min = jnp.where(lambda_min < 1e-10, 1e-10, lambda_min)

    return lambda_max / lambda_min


def effective_dimension(eigenvalues: jax.Array, gamma: float = 1e-4) -> jax.Array:
    """Compute the effective dimension N_eff(gamma).

    N_eff(gamma) = Sum_i (lambda_i / (lambda_i + gamma))

    This measures the number of relevant directions in parameter space
    given a regularization scale gamma.

    Args:
        eigenvalues: Vector of eigenvalues.
        gamma: Regularization parameter.

    Returns:
        Effective dimension (scalar).
    """
    return jnp.sum(eigenvalues / (eigenvalues + gamma))


def ntk_spectral_filtering(
    gradient_vector: jax.Array, eigenvectors: jax.Array, k: int
) -> jax.Array:
    """Filter gradient vector to keep only top-k spectral components.

    Project gradient onto the subspace spanned by the top-k eigenvectors
    of the NTK. This helps in analyzing which spectral modes are being learned.

    Args:
        gradient_vector: Flattened gradient vector (P,).
                         NOTE: This assumes the eigenvectors are providing a basis for
                         the parameter space (P, P).

                         BUT: If eigenvectors come from the Empirical NTK (N, N),
                         they span the *function output space* (on training data),
                         not the parameter space directly.

                         If the input is the gradient of the loss w.r.t parameters,
                         we need the SVD of the Jacobian to map between them.

                         However, usually spectral filtering in SciML context refers
                         to filtering the *residual* or the *output* in function space.

                         Let's assume the user passes a vector in the same space as
                         the eigenvectors (i.e., size N, representing function values
                         or residuals on the training set).

        eigenvectors: Eigenvectors of NTK (N, N).
        k: Number of top modes to keep.

    Returns:
        Filtered vector (N,).
    """
    # Assuming gradient_vector is in function space (e.g. residual vector) of size N
    # and eigenvectors are (N, N).

    # Top k eigenvectors
    # They are already sorted descending in compute_ntk_spectrum
    # Shape: (N, N), columns are eigenvectors

    # U_k = matrix of first k columns
    U_k = eigenvectors[:, :k]  # (N, k)

    # Project v onto U_k: P_k v = U_k U_k^T v
    # Coefficients c = U_k^T v (k,)
    coeffs = jnp.dot(U_k.T, gradient_vector)

    # Reconstruct: U_k c (N,)
    return jnp.dot(U_k, coeffs)
