"""NTK-based Training Diagnostics and Callbacks.

This module provides tools for diagnosing training dynamics using the
Neural Tangent Kernel, including mode-wise error decay prediction
and training callbacks for monitoring NTK evolution.

Key Features:
    - Mode-wise error decay computation
    - Convergence prediction from NTK eigenvalues
    - Spectral bias detection and monitoring
    - Training callbacks for NTK diagnostics

References:
    - Survey Section 3.2: Mode-wise Error Decay
    - e_k = Σᵢ cᵢ(1 - αλᵢ)^k qᵢ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp

from opifex.core.physics.ntk.spectral_analysis import (
    compute_condition_number,
    compute_effective_rank,
)
from opifex.core.physics.ntk.wrapper import NTKWrapper


if TYPE_CHECKING:
    from flax import nnx
    from jaxtyping import Array, Float


def compute_mode_coefficients(
    residuals: Float[Array, ...],
    eigenvectors: Float[Array, ...],
) -> Float[Array, ...]:
    """Project residuals onto eigenvector basis.

    The initial error can be decomposed into eigenmodes:
        e_0 = Σᵢ cᵢ qᵢ

    where qᵢ are the eigenvectors and cᵢ are the coefficients.

    Args:
        residuals: Residual vector (e.g., training loss per sample)
        eigenvectors: Eigenvector matrix (columns are eigenvectors)

    Returns:
        Coefficients for each eigenmode
    """
    # Project residuals onto eigenbasis: c = Q^T @ r
    return eigenvectors.T @ residuals


def compute_mode_decay_factors(
    eigenvalues: Float[Array, ...],
    learning_rate: float,
    iteration: int,
) -> Float[Array, ...]:
    """Compute decay factors for each eigenmode at given iteration.

    For gradient descent, each mode decays as:
        (1 - α * λᵢ)^k

    where α is the learning rate, λᵢ is the eigenvalue, and k is the iteration.

    Args:
        eigenvalues: NTK eigenvalues
        learning_rate: Learning rate
        iteration: Training iteration number

    Returns:
        Decay factors for each mode
    """
    decay_rates = 1.0 - learning_rate * eigenvalues
    return jnp.power(decay_rates, iteration)


def predict_mode_errors(
    initial_coeffs: Float[Array, ...],
    eigenvalues: Float[Array, ...],
    learning_rate: float,
    iteration: int,
) -> Float[Array, ...]:
    """Predict per-mode errors at given iteration.

    The error in each mode at iteration k is:
        e_k^(i) = c_i * (1 - α * λᵢ)^k

    Args:
        initial_coeffs: Initial mode coefficients
        eigenvalues: NTK eigenvalues
        learning_rate: Learning rate
        iteration: Training iteration

    Returns:
        Predicted error for each mode
    """
    decay_factors = compute_mode_decay_factors(eigenvalues, learning_rate, iteration)
    return initial_coeffs * decay_factors


def detect_spectral_bias(
    eigenvalues: Float[Array, ...],
) -> Float[Array, ""]:
    """Detect spectral bias from eigenvalue distribution.

    Spectral bias occurs when eigenvalues span many orders of magnitude,
    causing slow convergence for modes with small eigenvalues.

    Args:
        eigenvalues: NTK eigenvalues

    Returns:
        Spectral bias indicator (higher = more bias)
    """
    eigenvalues = jnp.maximum(eigenvalues, 1e-10)

    # Use log ratio of max to min as bias indicator
    return jnp.log(jnp.max(eigenvalues) / jnp.min(eigenvalues))


def identify_slow_modes(
    eigenvalues: Float[Array, ...],
    learning_rate: float,
    threshold: float = 0.99,
) -> Float[Array, ...]:
    """Identify slow-converging modes.

    A mode is slow if its per-step decay factor (1 - α * λ) is above
    the threshold, indicating minimal error reduction per step.

    Args:
        eigenvalues: NTK eigenvalues
        learning_rate: Learning rate
        threshold: Decay factor threshold (modes with rate > threshold are slow)

    Returns:
        Boolean mask indicating slow modes
    """
    decay_rates = jnp.abs(1.0 - learning_rate * eigenvalues)
    return decay_rates > threshold


def estimate_convergence_rate(
    eigenvalues: Float[Array, ...],
    learning_rate: float,
) -> Float[Array, ""]:
    """Estimate overall convergence rate from eigenvalues.

    The convergence rate is determined by the slowest mode (smallest eigenvalue).

    Args:
        eigenvalues: NTK eigenvalues
        learning_rate: Learning rate

    Returns:
        Convergence rate (smaller = faster convergence)
    """
    min_eigenvalue = jnp.min(jnp.abs(eigenvalues))
    return jnp.abs(1.0 - learning_rate * min_eigenvalue)


def estimate_epochs_to_convergence(
    eigenvalues: Float[Array, ...],
    learning_rate: float,
    target_reduction: float = 0.01,
) -> Float[Array, ""]:
    """Estimate epochs needed to reach target error reduction.

    Based on the slowest mode: (1 - α * λ_min)^n = target_reduction

    Args:
        eigenvalues: NTK eigenvalues
        learning_rate: Learning rate
        target_reduction: Target error reduction factor (e.g., 0.01 = 100x reduction)

    Returns:
        Estimated number of epochs (iterations)
    """
    rate = estimate_convergence_rate(eigenvalues, learning_rate)

    # Solve rate^n = target for n
    # n = log(target) / log(rate)
    # Handle edge cases
    rate = jnp.clip(rate, 1e-10, 1.0 - 1e-10)
    epochs = jnp.log(target_reduction) / jnp.log(rate)

    return jnp.abs(epochs)


@dataclass
class NTKTrainingDiagnostics:
    """Training diagnostics based on NTK analysis.

    Tracks NTK evolution during training and provides convergence predictions.

    Attributes:
        eigenvalues: Current eigenvalues
        eigenvectors: Current eigenvectors
        condition_number: Current condition number
        initial_coeffs: Initial mode coefficients
        history: History of diagnostics
    """

    track_history: bool = True
    eigenvalues: Float[Array, ...] | None = None
    eigenvectors: Float[Array, ...] | None = None
    condition_number: float = 0.0
    effective_rank: float = 0.0
    initial_coeffs: Float[Array, ...] | None = None
    learning_rate: float = 0.01
    history: list[dict] = field(default_factory=list)

    def update(
        self,
        ntk: Float[Array, ...],
        residuals: Float[Array, ...],
        learning_rate: float,
        iteration: int,
    ) -> None:
        """Update diagnostics with new NTK information.

        Args:
            ntk: Current NTK matrix
            residuals: Current residuals
            learning_rate: Learning rate
            iteration: Current iteration
        """
        self.learning_rate = learning_rate

        # Compute eigendecomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(ntk)

        # Sort descending
        idx = jnp.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

        # Compute diagnostics
        self.condition_number = float(
            compute_condition_number(self.eigenvalues)  # pyright: ignore[reportArgumentType]
        )
        self.effective_rank = float(
            compute_effective_rank(self.eigenvalues)  # pyright: ignore[reportArgumentType]
        )

        # Compute initial coefficients
        if self.initial_coeffs is None or iteration == 0:
            self.initial_coeffs = compute_mode_coefficients(
                residuals,
                self.eigenvectors,  # pyright: ignore[reportArgumentType]
            )

        # Track history
        if self.track_history:
            self.history.append(
                {
                    "iteration": iteration,
                    "condition_number": self.condition_number,
                    "effective_rank": self.effective_rank,
                    "eigenvalues": self.eigenvalues.copy(),  # pyright: ignore[reportOptionalMemberAccess]
                }
            )

    def predict_errors_at(self, iteration: int) -> Float[Array, ...]:
        """Predict per-mode errors at given iteration.

        Args:
            iteration: Future iteration to predict

        Returns:
            Predicted mode errors
        """
        if self.initial_coeffs is None or self.eigenvalues is None:
            raise ValueError("Must call update() before predicting errors")

        return predict_mode_errors(
            self.initial_coeffs,
            self.eigenvalues,
            self.learning_rate,
            iteration,
        )


class NTKDiagnosticsCallback:
    """Callback for NTK diagnostics during training.

    Computes and tracks NTK properties at specified intervals.

    Attributes:
        frequency: How often to compute NTK (every N steps)
        history: List of diagnostic dictionaries
    """

    def __init__(
        self,
        compute_frequency: int = 100,
    ):
        """Initialize callback.

        Args:
            compute_frequency: Compute NTK every N steps
        """
        self.frequency = compute_frequency
        self._history: list[dict] = []
        self._wrapper: NTKWrapper | None = None

    def on_step_end(
        self,
        model: nnx.Module,
        x: Float[Array, ...],
        step: int,
    ) -> None:
        """Called at end of each training step.

        Args:
            model: Current model state
            x: Sample inputs for NTK computation
            step: Current training step
        """
        if step % self.frequency != 0:
            return

        # Create or update wrapper
        if self._wrapper is None:
            self._wrapper = NTKWrapper(model)
        else:
            self._wrapper.model = model

        # Compute NTK
        ntk = self._wrapper.compute_ntk(x)

        # Compute eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(ntk)
        eigenvalues = jnp.sort(eigenvalues)[::-1]

        # Compute diagnostics
        cond = compute_condition_number(eigenvalues)
        rank = compute_effective_rank(eigenvalues)

        # Store history
        self._history.append(
            {
                "step": step,
                "condition_number": float(cond),
                "effective_rank": float(rank),
                "eigenvalues": eigenvalues,
                "max_eigenvalue": float(eigenvalues[0]),
                "min_eigenvalue": float(eigenvalues[-1]),
            }
        )

    def get_history(self) -> list[dict]:
        """Get history of diagnostics.

        Returns:
            List of diagnostic dictionaries
        """
        return self._history

    def get_condition_numbers(self) -> Float[Array, ...]:
        """Get array of condition numbers from history.

        Returns:
            Condition numbers at each tracked step
        """
        return jnp.array([h["condition_number"] for h in self._history])
