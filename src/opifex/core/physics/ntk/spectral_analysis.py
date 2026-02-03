"""NTK Spectral Analysis for training diagnostics.

This module provides tools for analyzing the spectral properties of the
Neural Tangent Kernel, which are fundamental for understanding training
dynamics and convergence properties.

Key Features:
    - Eigenvalue decomposition of NTK
    - Condition number computation
    - Effective rank estimation
    - Mode-wise convergence analysis
    - Spectral bias detection

References:
    - Survey Section 3.2: Mode-wise Error Decay
    - Jacot et al. (2018): Neural Tangent Kernel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from opifex.core.physics.ntk.wrapper import NTKWrapper


if TYPE_CHECKING:
    from flax import nnx
    from jaxtyping import Array, Float


@dataclass
class NTKDiagnostics:
    """Diagnostics from NTK spectral analysis.

    Attributes:
        eigenvalues: Sorted eigenvalues (descending order)
        condition_number: Ratio of largest to smallest eigenvalue
        effective_rank: Effective rank based on eigenvalue distribution
        spectral_bias_indicator: Indicator of spectral bias
        mode_convergence_rates: Per-eigenmode convergence rates
    """

    eigenvalues: Float[Array, ...]
    condition_number: float
    effective_rank: float
    spectral_bias_indicator: float
    mode_convergence_rates: Float[Array, ...] | None = None

    @classmethod
    def from_ntk(
        cls,
        ntk: Float[Array, ...],
        learning_rate: float = 0.01,
    ) -> NTKDiagnostics:
        """Create diagnostics from NTK matrix.

        Args:
            ntk: NTK matrix (should be symmetric positive semi-definite)
            learning_rate: Learning rate for convergence rate computation

        Returns:
            NTKDiagnostics instance
        """
        # Compute eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(ntk)
        eigenvalues = jnp.sort(eigenvalues)[::-1]  # Descending order

        # Compute diagnostics
        cond = compute_condition_number(eigenvalues)
        rank = compute_effective_rank(eigenvalues)
        bias = compute_spectral_bias_indicator(eigenvalues)
        rates = compute_mode_convergence_rates(eigenvalues, learning_rate)

        return cls(
            eigenvalues=eigenvalues,
            condition_number=float(cond),
            effective_rank=float(rank),
            spectral_bias_indicator=float(bias),
            mode_convergence_rates=rates,
        )


def compute_condition_number(
    eigenvalues: Float[Array, ...],
) -> Float[Array, ""]:
    """Compute condition number from eigenvalues.

    The condition number is the ratio of the largest to smallest eigenvalue.
    Large condition numbers indicate ill-conditioning.

    Args:
        eigenvalues: Eigenvalues (should be non-negative)

    Returns:
        Condition number (scalar)
    """
    # Filter out very small eigenvalues to avoid division by zero
    min_eigenvalue = jnp.min(eigenvalues)
    max_eigenvalue = jnp.max(eigenvalues)

    # Add small epsilon to denominator
    return max_eigenvalue / (min_eigenvalue + 1e-10)


def compute_condition_number_from_ntk(
    ntk: Float[Array, ...],
) -> Float[Array, ""]:
    """Compute condition number directly from NTK matrix.

    Args:
        ntk: NTK matrix

    Returns:
        Condition number
    """
    eigenvalues = jnp.linalg.eigvalsh(ntk)
    return compute_condition_number(eigenvalues)


def compute_effective_rank(
    eigenvalues: Float[Array, ...],
) -> Float[Array, ""]:
    """Compute effective rank from eigenvalue distribution.

    The effective rank is computed using the entropy-based definition:
        effective_rank = exp(entropy(p))

    where p is the normalized eigenvalue distribution.

    This gives a smooth measure of how many "significant" eigenvalues exist.

    Args:
        eigenvalues: Eigenvalues (should be non-negative)

    Returns:
        Effective rank (scalar between 1 and len(eigenvalues))
    """
    # Ensure non-negative
    eigenvalues = jnp.maximum(eigenvalues, 0.0)

    # Normalize to get probability distribution
    total = jnp.sum(eigenvalues)
    p = eigenvalues / (total + 1e-10)

    # Compute entropy
    # Use where to handle log(0)
    log_p = jnp.where(p > 1e-10, jnp.log(p), 0.0)
    entropy = -jnp.sum(p * log_p)

    # Effective rank = exp(entropy)
    return jnp.exp(entropy)


def compute_spectral_bias_indicator(
    eigenvalues: Float[Array, ...],
) -> Float[Array, ""]:
    """Compute spectral bias indicator from eigenvalue decay.

    The spectral bias indicator measures how quickly eigenvalues decay.
    Higher values indicate stronger spectral bias (faster decay).

    We use the ratio of the geometric mean to the arithmetic mean,
    normalized so that larger values indicate more decay.

    Args:
        eigenvalues: Eigenvalues (should be non-negative)

    Returns:
        Spectral bias indicator (scalar, higher = more bias)
    """
    # Ensure non-negative
    eigenvalues = jnp.maximum(eigenvalues, 1e-10)

    # Arithmetic mean
    arith_mean = jnp.mean(eigenvalues)

    # Geometric mean (via log)
    geom_mean = jnp.exp(jnp.mean(jnp.log(eigenvalues)))

    # Ratio (0 to 1, lower = more uniform, higher = more decay)
    ratio = geom_mean / (arith_mean + 1e-10)

    # Invert so higher = more spectral bias
    return 1.0 - ratio


def compute_mode_convergence_rates(
    eigenvalues: Float[Array, ...],
    learning_rate: float,
) -> Float[Array, ...]:
    """Compute convergence rates for each eigenmode.

    For gradient descent with learning rate α, the error in eigenmode k
    decays as (1 - α * λ_k)^t, where λ_k is the k-th eigenvalue.

    The convergence rate is 1 - α * λ_k, with smaller values indicating
    faster convergence.

    From Survey Section 3.2: e_k = Σᵢ cᵢ(1 - αλᵢ)^k qᵢ

    Args:
        eigenvalues: Eigenvalues
        learning_rate: Learning rate α

    Returns:
        Per-mode convergence rates (values in [0, 1] for stable training)
    """
    # Convergence rate per mode
    rates = 1.0 - learning_rate * eigenvalues

    # Clip to [0, 1] for interpretation
    # Values > 1 indicate divergence, values < 0 indicate oscillation
    rates = jnp.abs(rates)
    return jnp.minimum(rates, 1.0)


def estimate_pde_order(
    eigenvalues: Float[Array, ...],
) -> Float[Array, ""]:
    """Estimate PDE order from eigenvalue spectrum.

    For PDE of order p, the condition number scales as:
        κ(H) ≈ (ω_max / ω_min)^(2p)

    where ω are the characteristic frequencies. This function estimates p
    from the spectral decay pattern.

    Args:
        eigenvalues: Eigenvalues (sorted descending)

    Returns:
        Estimated PDE order (scalar)
    """
    # Ensure sorted descending and positive
    eigenvalues = jnp.sort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(eigenvalues, 1e-10)

    n = len(eigenvalues)
    if n < 2:
        return jnp.array(0.0)

    # Compute log ratio of successive eigenvalues
    log_ratios = jnp.log(eigenvalues[:-1]) - jnp.log(eigenvalues[1:])

    # Average log ratio gives an estimate of the decay rate
    _ = jnp.mean(log_ratios)  # Used for future extensions

    # For polynomial decay λ_k ~ k^(-2p), we have
    # log(λ_k / λ_{k+1}) ≈ 2p * log((k+1)/k) ≈ 2p/k
    # This is a rough estimate

    # Simple estimate: if decay is exponential-like, estimate order from condition
    log_cond = jnp.log(eigenvalues[0] / eigenvalues[-1])

    # Assuming n eigenvalues span roughly n frequency modes
    # and κ ~ (n)^(2p), we get p ~ log(κ) / (2 log(n))
    estimated_order = log_cond / (2 * jnp.log(n + 1) + 1e-10)

    return jnp.clip(estimated_order, 0.0, 10.0)


class NTKSpectralAnalyzer:
    """Analyzer for NTK spectral properties.

    This class provides a convenient interface for analyzing NTK
    eigenvalue distributions and tracking them during training.

    Attributes:
        model: The NNX model to analyze
        ntk_wrapper: NTK computation wrapper
        history: History of diagnostics during training

    Example:
        >>> model = MyModel(rngs=nnx.Rngs(0))
        >>> analyzer = NTKSpectralAnalyzer(model)
        >>> diagnostics = analyzer.analyze(x_train)
        >>> print(f"Condition number: {diagnostics.condition_number}")
    """

    def __init__(self, model: nnx.Module):
        """Initialize spectral analyzer.

        Args:
            model: FLAX NNX model to analyze
        """
        self.model = model
        self.ntk_wrapper = NTKWrapper(model)
        self.history: list[NTKDiagnostics] = []

    def analyze(
        self,
        x: Float[Array, ...],
        learning_rate: float = 0.01,
        track: bool = False,
    ) -> NTKDiagnostics:
        """Analyze NTK spectral properties at given points.

        Args:
            x: Input points to analyze NTK at
            learning_rate: Learning rate for convergence rate computation
            track: Whether to store result in history

        Returns:
            NTKDiagnostics with spectral analysis results
        """
        # Compute NTK
        ntk = self.ntk_wrapper.compute_ntk(x)

        # Create diagnostics
        diagnostics = NTKDiagnostics.from_ntk(ntk, learning_rate)

        if track:
            self.history.append(diagnostics)

        return diagnostics

    def get_condition_number_history(self) -> Float[Array, ...]:
        """Get history of condition numbers during training.

        Returns:
            Array of condition numbers from tracked analyses
        """
        return jnp.array([d.condition_number for d in self.history])

    def get_effective_rank_history(self) -> Float[Array, ...]:
        """Get history of effective ranks during training.

        Returns:
            Array of effective ranks from tracked analyses
        """
        return jnp.array([d.effective_rank for d in self.history])

    def clear_history(self):
        """Clear the tracking history."""
        self.history.clear()
