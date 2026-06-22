"""U-shaped Neural Operator (U-NO).

A faithful Flax-NNX reimplementation of the U-shaped Neural Operator of
Rahman, Ross & Azizzadenesheli, "U-NO: U-shaped Neural Operators", TMLR 2022
(https://arxiv.org/abs/2204.11127), mirroring the reference implementation in
``neuralop.models.uno.UNO``.

Unlike a conv U-Net, U-NO changes spatial resolution ONLY in the Fourier domain
(via :class:`~opifex.neural.operators.fno.base.SpectralConvResize` /
:func:`~opifex.neural.operators.fno.base.spectral_resample`) — there are no
strided convolutions and no pixel pooling or interpolation anywhere. This makes
the operator discretisation invariant, giving genuine zero-shot
super-resolution: a model trained at one grid resolution evaluates accurately at
a finer one.

Architecture (channels-first, ``(batch, channels, *spatial)``):

- a lifting :class:`ChannelMLP` (``in + grid -> hidden``),
- ``n_layers`` FNO blocks, each a spectral convolution with per-layer resolution
  scaling ``uno_scalings[i]`` followed by a channel MLP, a linear skip resampled
  to the block's output resolution, and a GELU non-linearity,
- horizontal U-skips (``horizontal_skips_map``) that resample a stored encoder
  feature into the current resolution in Fourier space and concatenate it on the
  channel axis,
- a projection :class:`ChannelMLP` (``hidden -> out``).
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.fno.base import (
    _resolve_output_size,
    spectral_resample,
    SpectralConvResize,
)


class ChannelMLP(nnx.Module):
    """Pointwise (1x1) channel-mixing MLP over a channels-first spatial field.

    Applies ``n_layers`` 1x1 "convolutions" (pointwise linear maps over the
    channel axis) with a GELU between hidden layers, matching
    ``neuralop.layers.channel_mlp.ChannelMLP``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        n_layers: int = 2,
        *,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the channel MLP.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            hidden_channels: Width of the hidden layers; defaults to
                ``in_channels``.
            n_layers: Number of pointwise linear layers (>= 1).
            activation: Non-linearity applied between hidden layers.
            rngs: Random number generators (keyword-only).
        """
        super().__init__()
        self.activation = activation
        hidden = hidden_channels if hidden_channels is not None else in_channels
        widths = [in_channels] + [hidden] * (n_layers - 1) + [out_channels]
        self.layers = nnx.List(
            [
                nnx.Linear(in_features=widths[i], out_features=widths[i + 1], rngs=rngs)
                for i in range(n_layers)
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the channel MLP to a ``(batch, channels, *spatial)`` field."""
        # Move channels last for the pointwise Linear, then restore.
        spatial = len(x.shape) - 2
        perm = [0, *range(2, 2 + spatial), 1]
        inv_perm = [0, spatial + 1, *range(1, spatial + 1)]
        h = jnp.transpose(x, perm)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return jnp.transpose(h, inv_perm)


class UNOBlock(nnx.Module):
    """A single U-NO Fourier block: spectral conv (+resize) + channel MLP + skip.

    Mirrors ``neuralop.layers.fno_block.FNOBlocks`` for one layer: the spectral
    convolution applies the per-layer resolution scaling, a linear skip is
    resampled to the same output resolution and added, a channel-mixing MLP
    refines the result, and a GELU non-linearity closes the block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        scaling: Sequence[float],
        *,
        channel_mlp_expansion: float = 0.5,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_modes: Retained Fourier modes ``(modes_h, modes_w)``.
            scaling: Per-axis resolution scaling for this block.
            channel_mlp_expansion: Hidden width of the channel MLP as a fraction
                of ``out_channels``.
            activation: Block non-linearity.
            rngs: Random number generators (keyword-only).
        """
        super().__init__()
        self.scaling = tuple(float(s) for s in scaling)
        self.activation = activation
        self.spectral_conv = SpectralConvResize(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            rngs=rngs,
        )
        # Linear skip connection (1x1 conv), resampled to the block output size.
        self.skip = nnx.Linear(in_features=in_channels, out_features=out_channels, rngs=rngs)
        mlp_hidden = max(1, round(out_channels * channel_mlp_expansion))
        self.channel_mlp = ChannelMLP(
            in_channels=out_channels,
            out_channels=out_channels,
            hidden_channels=mlp_hidden,
            n_layers=2,
            activation=activation,
            rngs=rngs,
        )

    def _apply_skip(self, x: jax.Array, output_size: tuple[int, ...]) -> jax.Array:
        """Pointwise linear skip, resampled to ``output_size`` in Fourier space."""
        spatial = len(x.shape) - 2
        perm = [0, *range(2, 2 + spatial), 1]
        inv_perm = [0, spatial + 1, *range(1, spatial + 1)]
        h = jnp.transpose(self.skip(jnp.transpose(x, perm)), inv_perm)
        if h.shape[-spatial:] != tuple(output_size):
            h = spectral_resample(h, tuple(output_size), axes=tuple(range(-spatial, 0)))
        return h

    def __call__(
        self,
        x: jax.Array,
        *,
        output_shape: tuple[int, ...] | None = None,
    ) -> jax.Array:
        """Apply the block.

        Args:
            x: Input ``(batch, in_channels, height, width)``.
            output_shape: Explicit output spatial size; overrides the per-layer
                scaling factor (used by the final layer to hit an exact size).

        Returns:
            Output ``(batch, out_channels, *output_size)``.
        """
        output_size = _resolve_output_size(x.shape[-2:], output_shape, self.scaling)
        spectral = self.spectral_conv(x, output_shape=output_size)
        skip = self._apply_skip(x, output_size)
        h = self.activation(spectral + skip)
        return self.channel_mlp(h)


class UNeuralOperator(nnx.Module):
    """U-shaped Neural Operator (U-NO).

    Discretisation-invariant operator that performs all resolution changes in
    the Fourier domain. See the module docstring for the architecture and the
    reference (Rahman et al., TMLR 2022).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        uno_out_channels: Sequence[int],
        uno_n_modes: Sequence[Sequence[int]],
        uno_scalings: Sequence[Sequence[float]],
        n_layers: int,
        *,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        channel_mlp_expansion: float = 0.5,
        horizontal_skips_map: dict[int, int] | None = None,
        activation: Callable[[jax.Array], jax.Array] = nnx.gelu,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the U-NO.

        Args:
            in_channels: Number of input channels (including any grid channels).
            out_channels: Number of output channels.
            hidden_channels: Lifting width fed into the first Fourier block.
            uno_out_channels: Output channels of each Fourier block (length
                ``n_layers``).
            uno_n_modes: Retained Fourier modes per block, ``[[mh, mw], ...]``
                (length ``n_layers``).
            uno_scalings: Per-axis resolution scaling per block,
                ``[[sh, sw], ...]`` (length ``n_layers``). The product across
                blocks is the end-to-end scaling and is typically 1.0.
            n_layers: Number of Fourier blocks.
            lifting_channels: Hidden width of the lifting MLP.
            projection_channels: Hidden width of the projection MLP.
            channel_mlp_expansion: Channel-MLP expansion inside each block.
            horizontal_skips_map: ``{dst: src}`` horizontal U-skip map. Defaults
                to ``{n-1-i: i for i in range(n // 2)}`` (e.g. ``{4: 0, 3: 1}``
                for ``n_layers=5``).
            activation: Block / projection non-linearity.
            rngs: Random number generators (keyword-only).

        Raises:
            ValueError: If the per-layer list lengths do not match ``n_layers``.
        """
        super().__init__()
        for name, seq in (
            ("uno_out_channels", uno_out_channels),
            ("uno_n_modes", uno_n_modes),
            ("uno_scalings", uno_scalings),
        ):
            if len(seq) != n_layers:
                raise ValueError(f"{name} must have length n_layers={n_layers}, got {len(seq)}")

        self.n_layers = n_layers
        self.n_dim = len(uno_n_modes[0])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.uno_out_channels = list(uno_out_channels)
        self.uno_n_modes = [tuple(m) for m in uno_n_modes]
        self.uno_scalings = [[float(s) for s in sc] for sc in uno_scalings]

        # Default horizontal skip map: {n-1: 0, n-2: 1, ...} for the first half.
        if horizontal_skips_map is None:
            horizontal_skips_map = {n_layers - i - 1: i for i in range(n_layers // 2)}
        self.horizontal_skips_map = horizontal_skips_map

        # End-to-end scaling = product of per-layer scalings (per axis).
        scaling = [1.0] * self.n_dim
        for layer_scaling in self.uno_scalings:
            scaling = [a * b for a, b in zip(scaling, layer_scaling, strict=True)]
        self.end_to_end_scaling_factor = scaling

        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
            activation=activation,
            rngs=rngs,
        )

        blocks: list[UNOBlock] = []
        horizontal_skips: dict[str, nnx.Linear] = {}
        prev_out = hidden_channels
        for i in range(n_layers):
            if i in self.horizontal_skips_map:
                prev_out += self.uno_out_channels[self.horizontal_skips_map[i]]
            blocks.append(
                UNOBlock(
                    in_channels=prev_out,
                    out_channels=self.uno_out_channels[i],
                    n_modes=self.uno_n_modes[i],
                    scaling=self.uno_scalings[i],
                    channel_mlp_expansion=channel_mlp_expansion,
                    activation=activation,
                    rngs=rngs,
                )
            )
            if i in self.horizontal_skips_map.values():
                horizontal_skips[str(i)] = nnx.Linear(
                    in_features=self.uno_out_channels[i],
                    out_features=self.uno_out_channels[i],
                    rngs=rngs,
                )
            prev_out = self.uno_out_channels[i]

        self.fno_blocks = nnx.List(blocks)
        self.horizontal_skips = nnx.Dict(horizontal_skips)

        self.projection = ChannelMLP(
            in_channels=prev_out,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            activation=activation,
            rngs=rngs,
        )

    def _apply_horizontal_skip(self, skip_idx: int, x: jax.Array) -> jax.Array:
        """Pointwise linear horizontal skip on a stored encoder feature."""
        layer = self.horizontal_skips[str(skip_idx)]
        spatial = len(x.shape) - 2
        perm = [0, *range(2, 2 + spatial), 1]
        inv_perm = [0, spatial + 1, *range(1, spatial + 1)]
        return jnp.transpose(layer(jnp.transpose(x, perm)), inv_perm)

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:  # noqa: ARG002 - nnx forward interface carries a deterministic flag
        """Apply the U-NO.

        Args:
            x: Input of shape ``(batch, in_channels, height, width)``
                (channels-first).
            deterministic: Present for interface compatibility; U-NO has no
                stochastic layers.

        Returns:
            Output of shape ``(batch, out_channels, *output_size)`` where the
            spatial size is ``round(input_size * end_to_end_scaling_factor)``.
        """
        x = self.lifting(x)

        # Final end-to-end output resolution (applied exactly on the last block).
        output_shape = tuple(
            round(size * scale)
            for size, scale in zip(
                x.shape[-self.n_dim :], self.end_to_end_scaling_factor, strict=True
            )
        )

        skip_outputs: dict[int, jax.Array] = {}
        for layer_idx in range(self.n_layers):
            if layer_idx in self.horizontal_skips_map:
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                if skip_val.shape[-self.n_dim :] != x.shape[-self.n_dim :]:
                    skip_val = spectral_resample(
                        skip_val,
                        tuple(x.shape[-self.n_dim :]),
                        axes=tuple(range(-self.n_dim, 0)),
                    )
                skip_val = self._apply_horizontal_skip(
                    self.horizontal_skips_map[layer_idx], skip_val
                )
                x = jnp.concatenate([x, skip_val], axis=1)

            block_output_shape = output_shape if layer_idx == self.n_layers - 1 else None
            x = self.fno_blocks[layer_idx](x, output_shape=block_output_shape)

            if layer_idx in self.horizontal_skips_map.values():
                skip_outputs[layer_idx] = x

        return self.projection(x)


def create_uno(
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_channels: int = 64,
    *,
    uno_out_channels: Sequence[int] | None = None,
    uno_n_modes: Sequence[Sequence[int]] | None = None,
    uno_scalings: Sequence[Sequence[float]] | None = None,
    n_layers: int = 5,
    rngs: nnx.Rngs,
) -> UNeuralOperator:
    """Create a U-NO with a Darcy-style default configuration.

    The default mirrors the reference ``examples/models/plot_UNO_darcy.py``: a
    five-layer encoder/decoder with channels ``[32, 64, 64, 64, 32]``, modes
    ``[8, 8]`` per block, and scalings whose product is 1.0 (output resolution
    equals input resolution).

    Args:
        in_channels: Number of input channels (including grid channels).
        out_channels: Number of output channels.
        hidden_channels: Lifting width.
        uno_out_channels: Per-block output channels (defaults to the Darcy set).
        uno_n_modes: Per-block Fourier modes (defaults to the Darcy set).
        uno_scalings: Per-block resolution scalings (defaults to the Darcy set).
        n_layers: Number of Fourier blocks.
        rngs: Random number generators (keyword-only).

    Returns:
        A configured :class:`UNeuralOperator`.
    """
    if uno_out_channels is None:
        uno_out_channels = [32, 64, 64, 64, 32]
    if uno_n_modes is None:
        uno_n_modes = [[8, 8], [8, 8], [4, 4], [8, 8], [8, 8]]
    if uno_scalings is None:
        uno_scalings = [[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [1.0, 1.0]]

    return UNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        uno_out_channels=uno_out_channels,
        uno_n_modes=uno_n_modes,
        uno_scalings=uno_scalings,
        n_layers=n_layers,
        rngs=rngs,
    )


__all__ = [
    "ChannelMLP",
    "UNOBlock",
    "UNeuralOperator",
    "create_uno",
]
