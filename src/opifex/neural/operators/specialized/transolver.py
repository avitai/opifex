"""Transolver — Physics-Attention Neural Operator.

Implements the Transolver architecture from Wu et al. (2024), which uses a
**Slice → Attend → Deslice** mechanism to achieve linear complexity attention
for PDE operator learning on irregular and structured meshes.

Reference:
    Wu, H., Luo, H., Wang, H., Wang, J., & Long, M. (2024).
    "Transolver: A Fast Transformer Solver for PDEs on General Geometries."
    ICML 2024.

This module reuses:
- ``StandardMLP`` from ``opifex.neural.base`` for feed-forward networks
- ``get_activation`` from ``opifex.neural.activations`` for activation lookup
"""

from __future__ import annotations

import dataclasses
import logging

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.base import StandardMLP


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TransolverConfig:
    """Configuration for Transolver model.

    Frozen dataclass following Artifex layer-level config pattern.

    Args:
        space_dim: Spatial dimensionality of input coordinates.
        fun_dim: Dimensionality of input function values. 0 means coords only.
        out_dim: Output dimensionality.
        hidden_dim: Hidden dimension throughout the transformer blocks.
        num_heads: Number of attention heads.
        num_layers: Number of Transolver blocks.
        slice_num: Number of physics-aware slices (G in the paper).
        dropout_rate: Dropout probability.
        mlp_ratio: FFN hidden dim multiplier relative to hidden_dim.
        activation: Activation function name (resolved via get_activation).
    """

    space_dim: int
    fun_dim: int
    out_dim: int
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 5
    slice_num: int = 32
    dropout_rate: float = 0.0
    mlp_ratio: int = 1
    activation: str = "gelu"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.slice_num <= 0:
            raise ValueError(f"slice_num must be positive, got {self.slice_num}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")


# ---------------------------------------------------------------------------
# Physics Attention (Slice → Attend → Deslice)
# ---------------------------------------------------------------------------


class PhysicsAttention(nnx.Module):
    """Physics-aware attention via Slice → Attend → Deslice.

    Projects input tokens into *G* physics-aware slices using
    temperature-scaled softmax, performs standard multi-head attention
    among slice tokens (linear in *N*), then deslices back.

    Complexity: O(N·G + G²) instead of O(N²).

    Args:
        dim: Model dimension (C).
        num_heads: Number of attention heads (H).
        dim_head: Dimension per head (D = C / H).
        slice_num: Number of slices (G).
        dropout_rate: Dropout probability.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        slice_num: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads

        # Learnable temperature for softmax sharpness (per head)
        self.temperature = nnx.Param(jnp.full((1, num_heads, 1, 1), 0.5))

        # Input projections
        self.in_project_x = nnx.Linear(dim, inner_dim, rngs=rngs)
        self.in_project_fx = nnx.Linear(dim, inner_dim, rngs=rngs)

        # Slice projection — orthogonal init for principled assignment
        self.in_project_slice = nnx.Linear(dim_head, slice_num, rngs=rngs)
        # Apply orthogonal initialization to slice weights
        ortho_key = rngs.params()
        ortho_weight = jax.random.orthogonal(ortho_key, n=max(dim_head, slice_num))[
            :dim_head, :slice_num
        ]
        self.in_project_slice.kernel.value = ortho_weight

        # QKV projections (no bias, per reference)
        self.to_q = nnx.Linear(dim_head, dim_head, use_bias=False, rngs=rngs)
        self.to_k = nnx.Linear(dim_head, dim_head, use_bias=False, rngs=rngs)
        self.to_v = nnx.Linear(dim_head, dim_head, use_bias=False, rngs=rngs)

        # Output projection
        self.out_proj = nnx.Linear(inner_dim, dim, rngs=rngs)

        # Dropout (Artifex pattern: conditional with rate guard)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout: nnx.Dropout | None = None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Apply physics attention.

        Args:
            x: Input tensor of shape ``(B, N, C)``.
            deterministic: If True, disable dropout.

        Returns:
            Output tensor of shape ``(B, N, C)``.
        """
        B, N, _C = x.shape

        # --- (1) Slice ---
        # Project to multi-head representations: (B, N, C) → (B, H, N, D)
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.num_heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )

        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.num_heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )

        # Temperature-scaled soft assignment: (B, H, N, D) → (B, H, N, G)
        temp = jnp.clip(self.temperature.value, 0.1, 5.0)
        slice_weights = jax.nn.softmax(self.in_project_slice(x_mid) / temp, axis=-1)

        # Aggregate into slice tokens: (B, H, G, D) via weighted average
        slice_norm = slice_weights.sum(axis=2)  # (B, H, G)
        slice_token = jnp.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[:, :, :, None] + 1e-5)

        # --- (2) Attend among slice tokens ---
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = jax.nn.softmax(dots, axis=-1)
        if self.dropout is not None and not deterministic:
            attn = self.dropout(attn)
        out_slice = jnp.matmul(attn, v)  # (B, H, G, D)

        # --- (3) Deslice ---
        out_x = jnp.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        # (B, H, N, D) → (B, N, H*D)
        out_x = out_x.transpose(0, 2, 1, 3).reshape(B, N, -1)

        return self.out_proj(out_x)


# ---------------------------------------------------------------------------
# Transolver Block (pre-norm transformer)
# ---------------------------------------------------------------------------


class TransolverBlock(nnx.Module):
    """Pre-norm Transformer block with Physics Attention + FFN.

    Structure::

        x → LayerNorm → PhysicsAttention → + residual → LayerNorm → FFN → + residual

    If ``last_layer=True``, an additional LayerNorm + Linear projects
    to ``out_dim`` after the residual.

    Args:
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        slice_num: Number of physics slices.
        dropout_rate: Dropout probability.
        mlp_ratio: FFN hidden dim multiplier.
        activation: Activation function name.
        last_layer: Whether this is the final block (adds output projection).
        out_dim: Output dimension (only used when ``last_layer=True``).
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        slice_num: int,
        dropout_rate: float = 0.0,
        mlp_ratio: int = 4,
        activation: str = "gelu",
        last_layer: bool = False,
        out_dim: int = 1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer

        dim_head = hidden_dim // num_heads

        # Pre-norm 1 + Physics Attention
        self.ln_1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.attn = PhysicsAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            slice_num=slice_num,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        # Pre-norm 2 + FFN (reuses StandardMLP from opifex.neural.base)
        self.ln_2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.ffn = StandardMLP(
            layer_sizes=[hidden_dim, hidden_dim * mlp_ratio, hidden_dim],
            activation=activation,
            dropout_rate=dropout_rate,
            rngs=rngs,
        )

        # Optional output projection for last layer
        if last_layer:
            self.ln_3 = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.out_proj = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor ``(B, N, hidden_dim)``.
            deterministic: If True, disable dropout.

        Returns:
            ``(B, N, hidden_dim)`` or ``(B, N, out_dim)`` if last layer.
        """
        # Pre-norm attention + residual
        x = self.attn(self.ln_1(x), deterministic=deterministic) + x
        # Pre-norm FFN + residual
        x = self.ffn(self.ln_2(x), deterministic=deterministic) + x

        if self.last_layer:
            return self.out_proj(self.ln_3(x))
        return x


# ---------------------------------------------------------------------------
# Full Transolver Model
# ---------------------------------------------------------------------------


class Transolver(nnx.Module):
    """Transolver — Fast Transformer Solver for PDEs on General Geometries.

    Takes spatial coordinates ``x`` (and optional function values ``fx``),
    projects them into a hidden space, passes through stacked
    ``TransolverBlock`` layers, and outputs predictions.

    Args:
        config: ``TransolverConfig`` frozen dataclass.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        config: TransolverConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.config = config

        # Input projection: (space_dim + fun_dim) → hidden_dim
        in_dim = config.space_dim + max(config.fun_dim, 0)
        self.preprocess = StandardMLP(
            layer_sizes=[in_dim, config.hidden_dim * 2, config.hidden_dim],
            activation=config.activation,
            dropout_rate=0.0,  # no dropout on input projection
            rngs=rngs,
        )

        # Learnable placeholder bias (adds position-agnostic info)
        self.placeholder = nnx.Param(
            jax.random.normal(rngs.params(), (config.hidden_dim,)) / config.hidden_dim
        )

        # Stacked Transolver blocks
        blocks = []
        for i in range(config.num_layers):
            is_last = i == config.num_layers - 1
            block = TransolverBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                slice_num=config.slice_num,
                dropout_rate=config.dropout_rate,
                mlp_ratio=config.mlp_ratio,
                activation=config.activation,
                last_layer=is_last,
                out_dim=config.out_dim,
                rngs=rngs,
            )
            blocks.append(block)
        self.blocks = nnx.List(blocks)

        logger.info(
            "Transolver initialized: %d layers, %d heads, %d slices, hidden_dim=%d",
            config.num_layers,
            config.num_heads,
            config.slice_num,
            config.hidden_dim,
        )

    def __call__(
        self,
        x: jax.Array,
        fx: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Forward pass.

        Args:
            x: Spatial coordinates ``(B, N, space_dim)``.
            fx: Function values ``(B, N, fun_dim)`` or ``None``.
            deterministic: If True, disable dropout.

        Returns:
            Predictions ``(B, N, out_dim)``.
        """
        # Concatenate coordinates and function values
        h = jnp.concatenate([x, fx], axis=-1) if fx is not None else x

        # Project to hidden dimension
        h = self.preprocess(h, deterministic=deterministic)

        # Add learnable placeholder bias
        h = h + self.placeholder.value[None, None, :]

        # Pass through Transolver blocks
        for block in self.blocks:
            h = block(h, deterministic=deterministic)

        return h


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_transolver(
    space_dim: int,
    fun_dim: int,
    out_dim: int,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 5,
    slice_num: int = 32,
    dropout_rate: float = 0.0,
    *,
    rngs: nnx.Rngs,
) -> Transolver:
    """Create a Transolver model with standard configuration.

    Args:
        space_dim: Spatial dimensionality.
        fun_dim: Input function dimensionality.
        out_dim: Output dimensionality.
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Transolver blocks.
        slice_num: Number of physics slices.
        dropout_rate: Dropout probability.
        rngs: Flax NNX random number generators.

    Returns:
        Configured Transolver model.
    """
    config = TransolverConfig(
        space_dim=space_dim,
        fun_dim=fun_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        slice_num=slice_num,
        dropout_rate=dropout_rate,
    )
    return Transolver(config, rngs=rngs)


__all__ = [
    "PhysicsAttention",
    "Transolver",
    "TransolverBlock",
    "TransolverConfig",
    "create_transolver",
]
