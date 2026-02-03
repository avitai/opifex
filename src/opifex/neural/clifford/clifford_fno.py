"""Clifford Fourier Neural Operator (CliffordFNO).

Wraps Artifex Clifford spectral convolution layers into a
full FNO architecture for 2D/3D PDE solving on multivector
fields. Supports Cl(2) (2D) and Cl(3) (3D) algebras.

Reference:
    Pepe et al. "Fengbo: Clifford Neural Operator for 3D CFD"
    (ICLR 2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from artifex.generative_models.core.layers.clifford import (
    CliffordConv2d,
    CliffordConv3d,
    CliffordSpectralConv2d,
    CliffordSpectralConv3d,
)
from flax import nnx


if TYPE_CHECKING:
    import jax


@dataclass(frozen=True, slots=True, kw_only=True)
class CliffordFNOConfig:
    """Configuration for Clifford FNO.

    Attributes:
        metric: Clifford algebra signature, e.g. (1,1) or
            (1,1,1).
        in_channels: Input field channels.
        out_channels: Output field channels.
        hidden_channels: Hidden channels per Fourier block.
        num_layers: Number of CliffordFourierBlocks.
        modes: Fourier modes per spatial dimension.
    """

    metric: tuple[int, ...] = (1, 1, 1)
    in_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 20
    num_layers: int = 4
    modes: tuple[int, ...] = (12, 12, 12)

    @property
    def dim(self) -> int:
        """Spatial dimension from metric."""
        return len(self.metric)

    @property
    def n_blades(self) -> int:
        """Number of Clifford algebra blades (2^dim)."""
        return 2**self.dim

    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.modes) != len(self.metric):
            msg = (
                f"modes length ({len(self.modes)}) must "
                f"match metric dim ({len(self.metric)})"
            )
            raise ValueError(msg)
        if self.hidden_channels < 1:
            msg = f"hidden_channels must be >= 1, got {self.hidden_channels}"
            raise ValueError(msg)
        if self.num_layers < 1:
            msg = f"num_layers must be >= 1, got {self.num_layers}"
            raise ValueError(msg)


def _make_spectral_conv(
    config: CliffordFNOConfig,
    in_ch: int,
    out_ch: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Build spectral conv for the right dimension. DRY."""
    if config.dim == 2:
        return CliffordSpectralConv2d(
            metric=config.metric,
            in_channels=in_ch,
            out_channels=out_ch,
            modes1=config.modes[0],
            modes2=config.modes[1],
            rngs=rngs,
        )
    if config.dim == 3:
        return CliffordSpectralConv3d(
            metric=config.metric,
            in_channels=in_ch,
            out_channels=out_ch,
            modes1=config.modes[0],
            modes2=config.modes[1],
            modes3=config.modes[2],
            rngs=rngs,
        )
    msg = f"Only 2D/3D supported, got dim={config.dim}"
    raise ValueError(msg)


def _make_bypass_conv(
    config: CliffordFNOConfig,
    in_ch: int,
    out_ch: int,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Build 1x1 bypass convolution. DRY."""
    if config.dim == 2:
        return CliffordConv2d(
            metric=config.metric,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            rngs=rngs,
        )
    if config.dim == 3:
        return CliffordConv3d(
            metric=config.metric,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            rngs=rngs,
        )
    msg = f"Only 2D/3D supported, got dim={config.dim}"
    raise ValueError(msg)


class CliffordFourierBlock(nnx.Module):
    """Single Clifford FNO block.

    Spectral path + 1x1 bypass convolution + GELU activation.
    Preserves spatial resolution and blade structure.

    Architecture:
        y = gelu(spectral_conv(x) + bypass_conv(x))

    Args:
        config: CliffordFNOConfig.
        rngs: Random number generators.
    """

    def __init__(
        self,
        *,
        config: CliffordFNOConfig,
        rngs: nnx.Rngs,
    ) -> None:
        ch = config.hidden_channels
        self.spectral = _make_spectral_conv(
            config,
            ch,
            ch,
            rngs,
        )
        self.bypass = _make_bypass_conv(
            config,
            ch,
            ch,
            rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward: spectral + bypass + activation.

        Args:
            x: Multivector field, shape
                2D: (B, H, W, C, 4)
                3D: (B, D, H, W, C, 8)
            deterministic: Unused, API consistency.

        Returns:
            Same shape as input.
        """
        return nnx.gelu(
            self.spectral(x, deterministic=deterministic)  # type: ignore[reportCallIssue]
            + self.bypass(x, deterministic=deterministic)  # type: ignore[reportCallIssue]
        )


class CliffordFNO(nnx.Module):
    """Clifford Fourier Neural Operator.

    Full FNO using Clifford spectral convolution blocks.
    Lifts scalar input fields into multivector representation,
    processes through Fourier blocks, projects back to scalar.

    Pipeline:
        input (B, *D, C_in)
        → lift to multivector (B, *D, C_hidden, n_blades)
        → N × CliffordFourierBlock
        → project to output (B, *D, C_out)

    Args:
        config: CliffordFNOConfig.
        rngs: Random number generators.
    """

    def __init__(
        self,
        *,
        config: CliffordFNOConfig,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config
        ch = config.hidden_channels
        nb = config.n_blades

        # Lifting: (B, *D, C_in) → (B, *D, C_hidden, n_blades)
        self.lift = nnx.Linear(
            in_features=config.in_channels,
            out_features=ch * nb,
            rngs=rngs,
        )

        # Fourier blocks
        self.blocks = nnx.List(
            [
                CliffordFourierBlock(config=config, rngs=rngs)
                for _ in range(config.num_layers)
            ]
        )

        # Projection: (B, *D, C_hidden * n_blades) → (B, *D, C_out)
        self.project = nnx.Linear(
            in_features=ch * nb,
            out_features=config.out_channels,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input field.
                2D: (B, H, W, C_in)
                3D: (B, D, H, W, C_in)
            deterministic: Passed to sub-layers.

        Returns:
            2D: (B, H, W, C_out)
            3D: (B, D, H, W, C_out)
        """
        cfg = self.config
        nb = cfg.n_blades

        # Lift to multivector
        h = self.lift(x)  # (B, *D, ch*nb)

        # Reshape to separate blade dimension
        spatial = h.shape[1:-1]
        h = h.reshape(
            h.shape[0],
            *spatial,
            cfg.hidden_channels,
            nb,
        )

        # Apply Fourier blocks
        for block in self.blocks:
            h = block(h, deterministic=deterministic)

        # Flatten blade dim and project
        h = h.reshape(
            h.shape[0],
            *spatial,
            cfg.hidden_channels * nb,
        )
        return self.project(h)


def create_clifford_fno(
    *,
    config: CliffordFNOConfig | None = None,
    rngs: nnx.Rngs,
) -> CliffordFNO:
    """Factory function for CliffordFNO.

    Args:
        config: Optional CliffordFNOConfig.
        rngs: Random number generators.

    Returns:
        CliffordFNO instance.
    """
    if config is None:
        config = CliffordFNOConfig()
    return CliffordFNO(config=config, rngs=rngs)
