"""Physics-Informed Kolmogorov-Arnold Networks (PIKAN).

Wraps Artifex KAN layers into multi-layer networks with
physics-informed training utilities. Supports all Artifex
KAN variants: dense, efficient, chebyshev, fourier, legendre,
rbf, sine.

Reference:
    Hao et al. "From PINNs to PIKANs" (arXiv, Oct 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from artifex.generative_models.core.layers.kan import (
    create_kan_layer,
)
from flax import nnx


if TYPE_CHECKING:
    import jax


# Valid KAN types that map to Artifex registry
_VALID_KAN_TYPES = frozenset(
    {
        "dense",
        "efficient",
        "chebyshev",
        "fourier",
        "legendre",
        "rbf",
        "sine",
    }
)

# Types that support spline grid update
_SPLINE_KAN_TYPES = frozenset({"dense", "efficient"})


@dataclass(frozen=True, slots=True, kw_only=True)
class PIKANConfig:
    """Configuration for physics-informed KAN networks.

    Attributes:
        n_layers: Number of KAN layers (must be >= 1).
        hidden_dim: Hidden dimension between layers.
        kan_type: Artifex KAN layer type.
        k: B-spline order (spline variants only).
        grid_intervals: Number of grid intervals.
        grid_range: Initial grid range.
        grid_e: Grid mixing parameter.
        pde_weight: Weight for PDE residual loss.
        bc_weight: Weight for boundary condition loss.
    """

    n_layers: int = 4
    hidden_dim: int = 32
    kan_type: str = "dense"
    k: int = 3
    grid_intervals: int = 5
    grid_range: tuple[float, float] = (-1.0, 1.0)
    grid_e: float = 0.05
    pde_weight: float = 1.0
    bc_weight: float = 10.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_layers < 1:
            msg = f"n_layers must be >= 1, got {self.n_layers}"
            raise ValueError(msg)
        if self.kan_type not in _VALID_KAN_TYPES:
            msg = (
                f"kan_type must be one of "
                f"{sorted(_VALID_KAN_TYPES)}, "
                f"got '{self.kan_type}'"
            )
            raise ValueError(msg)


def _build_kan_layer(
    n_in: int,
    n_out: int,
    config: PIKANConfig,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Build a single KAN layer from config.

    Shared helper — DRY construction for PIKAN and SincKAN.
    """
    kan_type = config.kan_type
    kwargs: dict = {
        "n_in": n_in,
        "n_out": n_out,
        "rngs": rngs,
    }
    # Spline-specific params
    if kan_type in _SPLINE_KAN_TYPES:
        kwargs["k"] = config.k
        kwargs["grid_intervals"] = config.grid_intervals
        kwargs["grid_range"] = config.grid_range
        kwargs["grid_e"] = config.grid_e
    # Basis layers use degree (D)
    elif kan_type in ("chebyshev", "legendre", "rbf") or kan_type in {
        "fourier",
        "sine",
    }:
        kwargs["D"] = config.grid_intervals

    return create_kan_layer(kan_type, **kwargs)  # type: ignore[reportReturnType]


class PIKAN(nnx.Module):
    """Physics-Informed Kolmogorov-Arnold Network.

    Multi-layer KAN network wrapping Artifex KAN layers with
    PDE-specific utilities (grid refinement, combined loss).

    Architecture: in → [KAN₁ → KAN₂ → ... → KANₙ] → out

    Args:
        in_dim: Input spatial dimension.
        out_dim: Number of output fields.
        config: PIKANConfig with layer hyperparameters.
        rngs: Random number generators.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        config: PIKANConfig | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        if config is None:
            config = PIKANConfig()
        self.config = config

        # Build layer stack
        dims = [in_dim] + [config.hidden_dim] * (config.n_layers - 1) + [out_dim]
        self.layers = nnx.List(
            [
                _build_kan_layer(
                    dims[i],
                    dims[i + 1],
                    config,
                    rngs,
                )
                for i in range(len(dims) - 1)
            ]
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass through KAN stack.

        Args:
            x: Input coordinates, shape (batch, in_dim).
            deterministic: Passed to each KAN layer.

        Returns:
            Output fields, shape (batch, out_dim).
        """
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)  # type: ignore[reportCallIssue]
        return x

    def update_grids(
        self,
        x: jax.Array,
        new_intervals: int,
    ) -> None:
        """Refine spline grids using input data.

        Only applies to spline-based KAN types (dense,
        efficient). No-op for polynomial/basis variants.

        Args:
            x: Sample data for grid adaptation.
            new_intervals: New grid interval count.
        """
        if self.config.kan_type not in _SPLINE_KAN_TYPES:
            return
        h = x
        for layer in self.layers:
            layer.update_grid(h, new_intervals)  # type: ignore[reportAttributeAccessIssue]
            h = layer(h)  # type: ignore[reportCallIssue]


class SincKAN(nnx.Module):
    """KAN with sinc/sinusoidal activations.

    Designed for PDEs with singularities (fracture mechanics,
    shock waves). Uses SineKANLayer from Artifex internally.

    Args:
        in_dim: Input spatial dimension.
        out_dim: Number of output fields.
        config: PIKANConfig (should use kan_type="sine").
        rngs: Random number generators.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        config: PIKANConfig | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        if config is None:
            config = PIKANConfig(kan_type="sine")
        self.config = config

        # Build layer stack
        dims = [in_dim] + [config.hidden_dim] * (config.n_layers - 1) + [out_dim]
        self.layers = nnx.List(
            [
                _build_kan_layer(
                    dims[i],
                    dims[i + 1],
                    config,
                    rngs,
                )
                for i in range(len(dims) - 1)
            ]
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Forward pass through sinc KAN stack.

        Args:
            x: Input, shape (batch, in_dim).
            deterministic: Passed to each layer.

        Returns:
            Output, shape (batch, out_dim).
        """
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)  # type: ignore[reportCallIssue]
        return x


def create_pikan(
    in_dim: int,
    out_dim: int,
    *,
    kan_type: str = "dense",
    config: PIKANConfig | None = None,
    rngs: nnx.Rngs,
) -> PIKAN | SincKAN:
    """Factory function for physics-informed KAN models.

    Returns SincKAN when kan_type is "sine", PIKAN otherwise.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        kan_type: KAN layer type (overridden if config given).
        config: Optional PIKANConfig.
        rngs: Random number generators.

    Returns:
        A PIKAN or SincKAN instance.
    """
    if config is None:
        config = PIKANConfig(kan_type=kan_type)

    if config.kan_type == "sine":
        return SincKAN(
            in_dim=in_dim,
            out_dim=out_dim,
            config=config,
            rngs=rngs,
        )
    return PIKAN(
        in_dim=in_dim,
        out_dim=out_dim,
        config=config,
        rngs=rngs,
    )
