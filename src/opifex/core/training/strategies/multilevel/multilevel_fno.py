"""Multilevel Training for Fourier Neural Operators.

This module implements multilevel training strategies specifically designed
for Fourier Neural Operators, where the hierarchy is based on Fourier modes.

Key Features:
    - Mode-based coarsening (fewer modes = coarser level)
    - Transfer operators for spectral weights
    - Cascade training from few modes to many modes

The key insight is that training with fewer Fourier modes first captures
low-frequency features quickly, then finer modes capture high-frequency details.

References:
    - Survey Section 8.2: Multilevel Training
    - Li et al. (2020): Fourier Neural Operator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float


@dataclass(frozen=True)
class MultilevelFNOConfig:
    """Configuration for multilevel FNO training.

    Attributes:
        num_levels: Number of levels in the hierarchy
        base_modes: Number of Fourier modes at finest level
        mode_reduction_factor: Factor to reduce modes at each coarser level
        level_epochs: Epochs to train at each level
    """

    num_levels: int = 3
    base_modes: int = 12
    mode_reduction_factor: int = 2
    level_epochs: list[int] = field(default_factory=lambda: [50, 100, 150])


def create_mode_hierarchy(
    base_modes: int,
    num_levels: int,
    reduction_factor: int = 2,
) -> list[int]:
    """Create hierarchy of mode counts from coarse to fine.

    The finest level (highest index) uses base_modes.
    Coarser levels use progressively fewer modes.

    Args:
        base_modes: Number of modes at finest level
        num_levels: Number of levels in hierarchy
        reduction_factor: Factor to reduce modes at each coarser level

    Returns:
        List of mode counts from coarsest to finest
    """
    hierarchy = []

    for level in range(num_levels):
        # Compute factor for this level (finest = 1, coarser = smaller)
        factor = reduction_factor ** (num_levels - level - 1)

        # Compute modes for this level
        modes = max(base_modes // factor, 1)
        hierarchy.append(modes)

    return hierarchy


class SpectralConv1d(nnx.Module):
    """1D Spectral Convolution layer for FNO.

    Performs convolution in Fourier space by multiplying
    with learned complex weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spectral convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            modes: Number of Fourier modes to keep
            rngs: Random number generators
        """
        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Complex weights for Fourier modes
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes)
        key = rngs.params()
        weights_real = jax.random.normal(key, shape) * scale
        key = rngs.params()
        weights_imag = jax.random.normal(key, shape) * scale

        self.weights_real = nnx.Param(weights_real)
        self.weights_imag = nnx.Param(weights_imag)

    def __call__(
        self, x: Float[Array, "batch spatial channels"]
    ) -> Float[Array, "batch spatial channels"]:
        """Apply spectral convolution.

        Args:
            x: Input tensor (batch, spatial, channels)

        Returns:
            Output tensor (batch, spatial, out_channels)
        """
        batch, spatial, _ = x.shape

        # FFT along spatial dimension
        x_ft = jnp.fft.rfft(x, axis=1)

        # Truncate to modes
        x_ft = x_ft[:, : self.modes, :]

        # Complex weights
        weights = self.weights_real[...] + 1j * self.weights_imag[...]

        # Multiply in Fourier space: (batch, modes, in) x (in, out, modes)
        out_ft = jnp.einsum("bmi,iom->bmo", x_ft, weights)

        # Pad back to original size
        pad_shape = (batch, spatial // 2 + 1, self.out_channels)
        out_ft_padded = jnp.zeros(pad_shape, dtype=out_ft.dtype)
        out_ft_padded = out_ft_padded.at[:, : self.modes, :].set(out_ft)

        # Inverse FFT
        return jnp.fft.irfft(out_ft_padded, n=spatial, axis=1)


class SimpleFNO(nnx.Module):
    """Simple FNO for multilevel training.

    A simplified FNO with configurable modes for multilevel hierarchy.

    Attributes:
        modes: Number of Fourier modes
        width: Hidden channel width
    """

    def __init__(
        self,
        modes: int,
        width: int,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 4,
        activation: Callable[[Array], Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize simple FNO.

        Args:
            modes: Number of Fourier modes
            width: Hidden channel width
            input_dim: Input channels
            output_dim: Output channels
            num_layers: Number of FNO layers
            activation: Activation function
            rngs: Random number generators
        """
        self.modes = modes
        self.width = width
        self.activation = activation

        # Lifting layer
        self.lift = nnx.Linear(input_dim, width, rngs=rngs)

        # FNO layers
        spectral_convs = []
        linear_convs = []
        for _ in range(num_layers):
            spectral_convs.append(SpectralConv1d(width, width, modes, rngs=rngs))
            linear_convs.append(nnx.Linear(width, width, rngs=rngs))

        self.spectral_convs = nnx.List(spectral_convs)
        self.linear_convs = nnx.List(linear_convs)

        # Projection layer
        self.project = nnx.Linear(width, output_dim, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch spatial channels"]
    ) -> Float[Array, "batch spatial channels"]:
        """Forward pass.

        Args:
            x: Input (batch, spatial, channels)

        Returns:
            Output (batch, spatial, output_dim)
        """
        # Lift to higher dimension
        x = self.lift(x)

        # FNO layers
        for spectral, linear in zip(
            list(self.spectral_convs), list(self.linear_convs), strict=False
        ):
            x1 = spectral(x)
            x2 = linear(x)
            x = self.activation(x1 + x2)

        # Project to output
        return self.project(x)


def create_fno_hierarchy(
    base_modes: int,
    width: int,
    input_dim: int,
    output_dim: int,
    num_levels: int,
    reduction_factor: int = 2,
    *,
    num_layers: int = 4,
    activation: Callable[[Array], Array] = nnx.gelu,
    rngs: nnx.Rngs,
) -> list[SimpleFNO]:
    """Create hierarchy of FNOs from coarse to fine.

    Args:
        base_modes: Modes at finest level
        width: Hidden channel width (same for all levels)
        input_dim: Input channels
        output_dim: Output channels
        num_levels: Number of levels
        reduction_factor: Mode reduction per level
        num_layers: FNO layers per network
        activation: Activation function
        rngs: Random number generators

    Returns:
        List of FNOs from coarsest to finest
    """
    mode_hierarchy = create_mode_hierarchy(base_modes, num_levels, reduction_factor)

    hierarchy = []
    for modes in mode_hierarchy:
        fno = SimpleFNO(
            modes=modes,
            width=width,
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            activation=activation,
            rngs=rngs,
        )
        hierarchy.append(fno)

    return hierarchy


def prolongate_fno_modes(
    coarse_fno: SimpleFNO,
    fine_fno: SimpleFNO,
) -> SimpleFNO:
    """Transfer spectral weights from coarse to fine FNO.

    Copies the lower-frequency modes from the coarse network to the
    corresponding modes in the fine network.

    Args:
        coarse_fno: Coarse FNO (fewer modes)
        fine_fno: Fine FNO (more modes, modified in place)

    Returns:
        Fine FNO with prolongated weights
    """
    coarse_modes = coarse_fno.modes
    fine_modes = fine_fno.modes

    # Transfer spectral convolution weights
    for coarse_spec, fine_spec in zip(
        list(coarse_fno.spectral_convs), list(fine_fno.spectral_convs), strict=False
    ):
        # Copy coarse weights to fine (lower modes)
        min_modes = min(coarse_modes, fine_modes)

        # Real weights
        fine_real = fine_spec.weights_real[...]
        fine_real = fine_real.at[:, :, :min_modes].set(
            coarse_spec.weights_real[...][:, :, :min_modes]
        )
        fine_spec.weights_real[...] = fine_real

        # Imaginary weights
        fine_imag = fine_spec.weights_imag[...]
        fine_imag = fine_imag.at[:, :, :min_modes].set(
            coarse_spec.weights_imag[...][:, :, :min_modes]
        )
        fine_spec.weights_imag[...] = fine_imag

    # Transfer lifting and projection layers
    # pyright: ignore[reportOptionalMemberAccess] - FLAX NNX attributes are initialized
    fine_fno.lift.kernel[...] = coarse_fno.lift.kernel[...]  # type: ignore[union-attr]
    fine_fno.lift.bias[...] = coarse_fno.lift.bias[...]  # type: ignore[union-attr]
    fine_fno.project.kernel[...] = coarse_fno.project.kernel[...]  # type: ignore[union-attr]
    fine_fno.project.bias[...] = coarse_fno.project.bias[...]  # type: ignore[union-attr]

    # Transfer linear convolution weights
    for coarse_lin, fine_lin in zip(
        list(coarse_fno.linear_convs), list(fine_fno.linear_convs), strict=False
    ):
        fine_lin.kernel[...] = coarse_lin.kernel[...]
        fine_lin.bias[...] = coarse_lin.bias[...]

    return fine_fno


def restrict_fno_modes(
    fine_fno: SimpleFNO,
    coarse_fno: SimpleFNO,
) -> SimpleFNO:
    """Transfer spectral weights from fine to coarse FNO.

    Copies the lower-frequency modes from the fine network to the
    coarse network (truncation).

    Args:
        fine_fno: Fine FNO (more modes)
        coarse_fno: Coarse FNO (fewer modes, modified in place)

    Returns:
        Coarse FNO with restricted weights
    """
    coarse_modes = coarse_fno.modes

    # Transfer spectral convolution weights
    for fine_spec, coarse_spec in zip(
        list(fine_fno.spectral_convs), list(coarse_fno.spectral_convs), strict=False
    ):
        # Take lower modes from fine
        real_slice = fine_spec.weights_real[...][:, :, :coarse_modes]
        imag_slice = fine_spec.weights_imag[...][:, :, :coarse_modes]
        coarse_spec.weights_real[...] = real_slice
        coarse_spec.weights_imag[...] = imag_slice

    # Transfer other layers
    # pyright: ignore[reportOptionalMemberAccess] - FLAX NNX attributes are initialized
    coarse_fno.lift.kernel[...] = fine_fno.lift.kernel[...]  # type: ignore[union-attr]
    coarse_fno.lift.bias[...] = fine_fno.lift.bias[...]  # type: ignore[union-attr]
    coarse_fno.project.kernel[...] = fine_fno.project.kernel[...]  # type: ignore[union-attr]
    coarse_fno.project.bias[...] = fine_fno.project.bias[...]  # type: ignore[union-attr]

    for fine_lin, coarse_lin in zip(
        list(fine_fno.linear_convs), list(coarse_fno.linear_convs), strict=False
    ):
        coarse_lin.kernel[...] = fine_lin.kernel[...]
        coarse_lin.bias[...] = fine_lin.bias[...]

    return coarse_fno


class MultilevelFNOTrainer:
    """Trainer for multilevel FNO.

    Trains FNOs from coarse to fine modes, transferring learned
    spectral weights between levels.

    Attributes:
        config: Multilevel configuration
        hierarchy: List of FNOs from coarse to fine
        current_level: Current training level
    """

    def __init__(
        self,
        width: int,
        input_dim: int,
        output_dim: int,
        config: MultilevelFNOConfig | None = None,
        *,
        num_layers: int = 4,
        activation: Callable[[Array], Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ):
        """Initialize multilevel FNO trainer.

        Args:
            width: Hidden channel width
            input_dim: Input channels
            output_dim: Output channels
            config: Multilevel configuration
            num_layers: FNO layers per network
            activation: Activation function
            rngs: Random number generators
        """
        self.config = config or MultilevelFNOConfig()

        self.hierarchy = create_fno_hierarchy(
            base_modes=self.config.base_modes,
            width=width,
            input_dim=input_dim,
            output_dim=output_dim,
            num_levels=self.config.num_levels,
            reduction_factor=self.config.mode_reduction_factor,
            num_layers=num_layers,
            activation=activation,
            rngs=rngs,
        )

        self.current_level = 0

    def get_current_model(self) -> SimpleFNO:
        """Get FNO at current level.

        Returns:
            Current level FNO
        """
        return self.hierarchy[self.current_level]

    def advance_level(self) -> bool:
        """Advance to next finer level.

        Transfers learned weights from current level to next level.

        Returns:
            True if advanced successfully, False if at finest level
        """
        if self.current_level >= len(self.hierarchy) - 1:
            return False

        # Prolongate weights to next level
        coarse_fno = self.hierarchy[self.current_level]
        fine_fno = self.hierarchy[self.current_level + 1]
        prolongate_fno_modes(coarse_fno, fine_fno)

        self.current_level += 1
        return True

    def is_at_finest(self) -> bool:
        """Check if at finest level.

        Returns:
            True if at finest level
        """
        return self.current_level >= len(self.hierarchy) - 1

    def get_epochs_for_current_level(self) -> int:
        """Get epochs for current level from config.

        Returns:
            Number of epochs to train at current level
        """
        if self.current_level < len(self.config.level_epochs):
            return self.config.level_epochs[self.current_level]
        return self.config.level_epochs[-1]
