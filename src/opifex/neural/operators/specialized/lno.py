"""Laplace Neural Operator (LNO).

Implements the LNO architecture from Cao et al. (2023), which uses the
Laplace transform with a pole-residue decomposition instead of FFT. This
enables handling of non-periodic signals and transient responses with
exponential convergence.

Reference:
    Cao, Q., Goswami, S., & Karniadakis, G. E. (2023).
    "LNO: Laplace Neural Operator for Solving Differential Equations."
    arXiv:2303.10528.
    GitHub: qianyingcao/Laplace-Neural-Operator

Architecture (from reference):
    1. Lift (fc0): Linear map from input to hidden channels
    2. Laplace Layer (PR): FFT → transfer function H(s)=res/(s-pole) →
       dual-path output (IFFT steady-state + exponential transient)
    3. Local linear (w0): Conv1d skip connection
    4. Project (fc1, fc2): MLP back to output channels

This module reuses:
- ``StandardMLP`` from ``opifex.neural.base`` for lifting/projection
- ``get_activation`` from ``opifex.neural.activations``
"""

from __future__ import annotations

import dataclasses
import logging

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.activations import get_activation
from opifex.neural.base import StandardMLP


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LaplaceLayerConfig:
    """Configuration for a single Laplace layer.

    Frozen dataclass following Artifex layer-level config pattern.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_poles: Number of learnable poles (complex exponentials).
    """

    in_channels: int
    out_channels: int
    num_poles: int = 16

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")
        if self.num_poles <= 0:
            raise ValueError(f"num_poles must be positive, got {self.num_poles}")


# ---------------------------------------------------------------------------
# Laplace Layer (pole-residue decomposition)
#
# Reference: ``PR`` class in qianyingcao/Laplace-Neural-Operator
#
# Architecture:
#   1. FFT on input -> frequency-domain poles and coefficients
#   2. Learnable transfer function: H(s) = residue / (s - pole)
#   3. Steady-state path: multiply in frequency domain -> IFFT
#   4. Transient path: residue * exp(pole * t) summed over poles
#   5. Output = steady_state + transient + local_linear_skip
# ---------------------------------------------------------------------------


class LaplaceLayer(nnx.Module):
    """Laplace-domain convolution layer via pole-residue decomposition.

    Faithfully ports the ``PR`` class from the reference implementation.
    Uses complex-valued learnable poles and residues, with a dual-path
    output: IFFT-based steady-state response and exponential-kernel
    transient response.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_poles: Number of learnable poles (modes).
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_poles: int = 16,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_poles = num_poles

        # Scale factor matching reference: 1 / (in * out)
        scale = 1.0 / (in_channels * out_channels)

        # Learnable complex poles: real and imaginary parts stored separately
        # (JAX doesn't natively support complex nnx.Param gradient flow as
        #  cleanly as PyTorch, so we keep real/imag split)
        # Shape: (in_channels, out_channels, num_poles) — matching reference
        key1, key2 = jax.random.split(rngs.params())
        self.weights_pole = nnx.Param(
            scale
            * jax.random.uniform(
                key1,
                (in_channels, out_channels, num_poles),
            )
        )
        self.weights_pole_imag = nnx.Param(
            scale
            * jax.random.uniform(
                key2,
                (in_channels, out_channels, num_poles),
            )
        )

        key3, key4 = jax.random.split(rngs.params())
        self.weights_residue = nnx.Param(
            scale
            * jax.random.uniform(
                key3,
                (in_channels, out_channels, num_poles),
            )
        )
        self.weights_residue_imag = nnx.Param(
            scale
            * jax.random.uniform(
                key4,
                (in_channels, out_channels, num_poles),
            )
        )

        # Local linear transform (skip connection, like FNO's Conv1d w/ kernel=1)
        self.local_linear = nnx.Linear(in_channels, out_channels, rngs=rngs)

    def _build_complex_weights(
        self,
    ) -> tuple[jax.Array, jax.Array]:
        """Build complex pole and residue tensors from real/imag parts."""
        poles = self.weights_pole + 1j * self.weights_pole_imag
        residues = self.weights_residue + 1j * self.weights_residue_imag
        return poles, residues

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply Laplace layer (pole-residue operation).

        Follows reference ``PR.forward()`` architecture:
        1. FFT to get input poles/residues
        2. Apply transfer function H(s) = res/(s - pole)
        3. Dual-path output: IFFT (steady-state) + exp kernel (transient)

        Args:
            x: Input tensor ``(B, C_in, N)`` in channels-first format.

        Returns:
            Output tensor ``(B, C_out, N)``.
        """
        _B, _C_in, N = x.shape
        poles, residues = self._build_complex_weights()

        # --- Step 1: FFT to get input spectral coefficients ---
        # alpha = FFT(x): (B, C_in, N) -> (B, C_in, N) complex
        alpha = jnp.fft.fft(x)

        # Compute input frequency-domain poles (λ = j·ω)
        # Using normalized grid with dt = 1/N
        dt = 1.0 / N
        lambda0 = jnp.fft.fftfreq(N, dt) * 2 * jnp.pi * 1j  # (N,)
        # Reshape for broadcasting: (N, 1, 1, 1)
        lambda1 = lambda0[:, None, None, None]

        # --- Step 2: Apply transfer function H(s) = res / (s - pole) ---
        # poles: (C_in, C_out, P), lambda1: (N, 1, 1, 1)
        # H(s): (N, C_in, C_out, P)
        term1 = 1.0 / (lambda1 - poles[None, :, :, :])
        Hw = residues[None, :, :, :] * term1  # (N, C_in, C_out, P)

        # Contract: alpha*H for steady-state, alpha*(-H) for transient
        # alpha: (B, C_in, N), Hw: (N, C_in, C_out, P)
        # Steady-state residues: sum over C_in, keep (B, C_out, N)
        output_residue1 = jnp.einsum("bin,niop->bon", alpha, Hw)
        # Transient: sum over C_in and N, keep (B, C_out, P)
        output_residue2 = jnp.einsum("bin,niop->bop", alpha, -Hw)

        # --- Step 3a: Steady-state via IFFT ---
        x1 = jnp.fft.ifft(output_residue1, n=N)
        x1 = jnp.real(x1)

        # --- Step 3b: Transient via exponential kernel ---
        t = jnp.linspace(0, 1, N)  # (N,)
        # Compute exp(pole * t): poles (C_in, C_out, P), t (N,)
        # → (C_in, C_out, P, N) then einsum with residues
        exp_term = jnp.exp(jnp.einsum("iop,n->iopn", poles, t.astype(jnp.complex64)))
        # output_residue2: (B, C_out, P), exp_term: (C_in, C_out, P, N)
        # → contract over P: (B, C_out, N)
        x2 = jnp.einsum("bop,iopn->bon", output_residue2, exp_term)
        x2 = jnp.real(x2) / N

        # --- Step 4: Local linear skip connection ---
        # x: (B, C_in, N) → (B, N, C_in) → linear → (B, N, C_out) → (B, C_out, N)
        skip = self.local_linear(x.transpose(0, 2, 1)).transpose(0, 2, 1)

        return x1 + x2 + skip


# ---------------------------------------------------------------------------
# LNO Model (matches reference LNO1d)
# ---------------------------------------------------------------------------


class LaplaceNeuralOperator(nnx.Module):
    """Laplace Neural Operator for non-periodic and transient signals.

    Architecture matches reference ``LNO1d``:
        Lift (fc0) -> [LaplaceLayer + Skip + Act] -> Project

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        hidden_channels: Hidden dimension (``width`` in reference).
        num_layers: Number of Laplace layers (reference uses 1).
        num_poles: Number of poles per layer (``modes`` in reference).
        activation: Activation function (reference uses ``sin``).
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        num_poles: int = 16,
        activation: str = "gelu",
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_poles = num_poles

        # Lifting layer: in_channels → hidden_channels (fc0 in reference)
        self.lift = StandardMLP(
            layer_sizes=[in_channels, hidden_channels],
            activation=activation,
            rngs=rngs,
        )

        # Stacked Laplace layers (reference uses 1 layer)
        layers = []
        for _ in range(num_layers):
            layer = LaplaceLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                num_poles=num_poles,
                rngs=rngs,
            )
            layers.append(layer)
        self.laplace_layers = nnx.List(layers)

        # Activation (reference uses sin; we default to gelu for general use)
        self.activation_fn = get_activation(activation)

        # Project: hidden → 128 → out (fc1, fc2 in reference)
        self.project = StandardMLP(
            layer_sizes=[hidden_channels, hidden_channels, out_channels],
            activation=activation,
            rngs=rngs,
        )

        logger.info(
            "LNO initialized: %d layers, %d poles, hidden=%d",
            num_layers,
            num_poles,
            hidden_channels,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input ``(B, C_in, N)`` in channels-first format.
            deterministic: If True, disable dropout (API consistency).

        Returns:
            Output ``(B, C_out, N)``.
        """
        # Lift: (B, C_in, N) → (B, N, C_in) → MLP → (B, N, hidden) → (B, H, N)
        h = self.lift(x.transpose(0, 2, 1), deterministic=deterministic).transpose(
            0, 2, 1
        )

        # Laplace layers: each applies PR + skip + activation
        for layer in self.laplace_layers:
            h = self.activation_fn(layer(h))

        # Project: (B, H, N) -> (B, N, H) -> MLP -> (B, N, C_out) -> (B, C_out, N)
        return self.project(
            h.transpose(0, 2, 1), deterministic=deterministic
        ).transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_lno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    num_layers: int = 1,
    num_poles: int = 16,
    activation: str = "gelu",
    *,
    rngs: nnx.Rngs,
) -> LaplaceNeuralOperator:
    """Create a Laplace Neural Operator.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        hidden_channels: Hidden dimension.
        num_layers: Number of Laplace layers (reference uses 1).
        num_poles: Number of poles per layer.
        activation: Activation function name.
        rngs: Flax NNX random number generators.

    Returns:
        Configured LaplaceNeuralOperator.
    """
    return LaplaceNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_poles=num_poles,
        activation=activation,
        rngs=rngs,
    )


__all__ = [
    "LaplaceLayer",
    "LaplaceLayerConfig",
    "LaplaceNeuralOperator",
    "create_lno",
]
