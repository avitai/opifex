# FILE PLACEMENT: opifex/neural/operators/fno/tensorized.py

"""Tensorized Fourier Neural Operator with genuine low-rank spectral weights.

The spectral convolution weight ``(out_channels, in_channels, *modes)`` is stored
in a factorized CP / Tucker / Tensor-Train form whose parameter count is ``<<``
the dense weight at low rank. The factorized contraction (input against the
factors directly) and the reconstruct formulas are ports of established
references centralised in :mod:`._factorized`:

- Reconstruct: tensorly 0.9.0 ``cp_to_tensor`` / ``tucker_to_tensor`` /
  ``tt_to_tensor``.
- Contraction: neuraloperator ``_contract_cp`` / ``_contract_tucker`` /
  ``_contract_tt``.
- Factor layouts + init std: tltorch ``factorized_tensors.py`` / ``init.py``.
- Paper: Kossaifi et al., "Multi-Grid Tensorized Fourier Neural Operator".

Complex weights are stored as two real ``nnx.Param`` tensors per factor / core /
CP-weight (``*_real`` and ``*_imag``) and recombined as ``real + 1j * imag``
inside reconstruct / contract. This split avoids the JAX complex-gradient
convention issue (optax #196) that previously produced a Tensor-Train
complex-gradient bug; do not collapse it to a single complex parameter.
"""

from collections.abc import Sequence
from math import prod
from typing import Literal

import jax
from flax import nnx

from opifex.neural.operators.fno._decompositions import (
    CPDecomposition,
    make_decomposition,
    TensorTrainDecomposition,
    TuckerDecomposition,
)
from opifex.neural.operators.fno._factorized import factorized_spectral_conv
from opifex.neural.operators.fno.base import FourierNeuralOperator


class TensorizedSpectralConvolution(nnx.Module):
    """Spectral convolution whose weight is a low-rank CP / Tucker / TT factorization.

    Transforms a real spatial field to the Fourier domain, contracts the centered
    low-frequency band against the factorized weight (keeping both positive and
    negative low frequencies) and transforms back — see
    :func:`opifex.neural.operators.fno._factorized.factorized_spectral_conv`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        decomposition_type: Literal["tucker", "cp", "tt"] = "tucker",
        rank: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(modes)
        self.decomposition_type = decomposition_type

        tensor_shape = (out_channels, in_channels, *modes)
        self.decomposition = make_decomposition(decomposition_type, tensor_shape, rank, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the factorized spectral convolution to a real spatial field.

        Args:
            x: Real input of shape ``(batch, in_channels, *spatial)``.

        Returns:
            Real output of shape ``(batch, out_channels, *spatial)``.
        """
        return factorized_spectral_conv(self.decomposition, x, self.modes)

    def get_compression_stats(self) -> dict[str, float]:
        """Get compression statistics."""
        factorized_params = self.decomposition.parameter_count()
        dense_params = self.in_channels * self.out_channels * prod(self.modes)
        compression_ratio = factorized_params / dense_params if dense_params > 0 else 0

        return {
            "factorized_parameters": float(factorized_params),
            "equivalent_dense_parameters": float(dense_params),
            "compression_ratio": float(compression_ratio),
            "parameter_reduction": float(1 - compression_ratio),
        }


class TensorizedFourierNeuralOperator(FourierNeuralOperator):
    """Tensorized FNO — a Fourier Neural Operator with low-rank spectral weights.

    Thin specialisation of :class:`~opifex.neural.operators.fno.base.FourierNeuralOperator`
    that stores each spectral-convolution weight as a CP / Tucker / Tensor-Train
    factorization (Kossaifi et al., "Multi-Grid Tensorized Fourier Neural
    Operator"). It inherits the full, correct FNO forward pass — lifting, the
    ``activation(spectral + skip)`` Fourier blocks, grid positional embedding, and
    the two-layer projection head — so the only difference from a dense FNO is the
    factorized weight. This deletes the previously duplicated (and incorrect)
    spectral/forward implementation in favour of the single shared one (Rule 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        modes: Sequence[int] | int = (16, 16),
        num_layers: int = 4,
        factorization: Literal["tucker", "cp", "tt"] = "tucker",
        rank: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        modes_tuple = (modes,) if isinstance(modes, int) else tuple(modes)
        if len(set(modes_tuple)) > 1:
            raise ValueError(
                f"TFNO uses an isotropic mode count per axis; got modes={modes_tuple}."
            )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes_tuple[0],
            num_layers=num_layers,
            factorization_type=factorization,
            factorization_rank=rank,
            spatial_dims=len(modes_tuple),
            positional_embedding=True,
            rngs=rngs,
        )


# Factory functions for convenience
def create_tucker_fno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (16, 16),
    rank: float = 0.1,
    num_layers: int = 4,
    *,
    rngs: nnx.Rngs,
) -> TensorizedFourierNeuralOperator:
    """Create Tucker factorized FNO."""
    return TensorizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_layers=num_layers,
        factorization="tucker",
        rank=rank,
        rngs=rngs,
    )


def create_cp_fno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (16, 16),
    rank: float = 0.1,
    num_layers: int = 4,
    *,
    rngs: nnx.Rngs,
) -> TensorizedFourierNeuralOperator:
    """Create CP factorized FNO."""
    return TensorizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_layers=num_layers,
        factorization="cp",
        rank=rank,
        rngs=rngs,
    )


def create_tt_fno(
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
    modes: Sequence[int] = (16, 16),
    rank: float = 0.1,
    num_layers: int = 4,
    *,
    rngs: nnx.Rngs,
) -> TensorizedFourierNeuralOperator:
    """Create Tensor Train factorized FNO."""
    return TensorizedFourierNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        modes=modes,
        num_layers=num_layers,
        factorization="tt",
        rank=rank,
        rngs=rngs,
    )


# Re-export the factorized weights (moved to ._decompositions) so existing
# ``from ...fno.tensorized import TuckerDecomposition`` imports keep working.
__all__ = [
    "CPDecomposition",
    "TensorTrainDecomposition",
    "TensorizedFourierNeuralOperator",
    "TensorizedSpectralConvolution",
    "TuckerDecomposition",
    "create_cp_fno",
    "create_tt_fno",
    "create_tucker_fno",
    "make_decomposition",
]
