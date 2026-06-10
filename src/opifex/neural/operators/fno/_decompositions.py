"""Learnable low-rank factorized spectral weights (CP / Tucker / Tensor-Train).

These ``nnx.Module`` weights store a spectral-convolution tensor
``(out_channels, in_channels, *modes)`` in factorized form whose parameter count
is ``<<`` the dense weight at low rank. The reconstruct formulas and the
memory-optimal factorized contractions are ported from established references and
live in :mod:`._factorized`; this module wraps them as learnable parameters.

It is a leaf module (it imports only :mod:`._factorized` and Flax) so that both
:mod:`.base` and :mod:`.tensorized` can depend on it without an import cycle.

Complex weights are stored as two real ``nnx.Param`` tensors per factor / core /
CP-weight (``*_real`` and ``*_imag``) and recombined as ``real + 1j * imag``.
This split avoids the JAX complex-gradient convention issue (optax #196) that
otherwise corrupts Tensor-Train complex gradients; do not collapse it.
"""

from collections.abc import Sequence
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.fno._factorized import (
    contract_cp,
    contract_tt,
    contract_tucker,
    cp_factor_std,
    cp_parameter_count,
    cp_to_tensor,
    tt_factor_std,
    tt_parameter_count,
    tt_ranks,
    tt_to_tensor,
    tucker_factor_std,
    tucker_parameter_count,
    tucker_ranks,
    tucker_to_tensor,
)


class TensorDecomposition(Protocol):
    """Protocol for tensor decomposition strategies."""

    def multiply_factorized(self, input_tensor: jax.Array) -> jax.Array:
        """Efficient multiplication using factorized form."""
        ...

    def reconstruct(self) -> jax.Array:
        """Reconstruct full tensor from factors."""
        ...

    def parameter_count(self) -> int:
        """Count parameters in factorized representation."""
        ...


def _split_real_imag(
    key: jax.Array, shape: tuple[int, ...], std: float
) -> tuple[nnx.Param, nnx.Param]:
    """Create a real/imag-split complex factor as two ``nnx.Param`` tensors.

    The split is the optax #196 workaround (see module docstring): each complex
    factor is two independent real Gaussian tensors with the given std.
    """
    real_key, imag_key = jax.random.split(key)
    return (
        nnx.Param(jax.random.normal(real_key, shape) * std),
        nnx.Param(jax.random.normal(imag_key, shape) * std),
    )


class TuckerDecomposition(nnx.Module):
    """Tucker low-rank factorization of a spectral convolution weight.

    Stores a complex core ``(rank_in, rank_out, rank_mode_0, ...)`` and one
    complex factor matrix per mode. The factorized contraction is a port of
    neuraloperator ``_contract_tucker``; the reconstruct is tensorly
    ``tucker_to_tensor``. Internal mode order is ``(in, out, *modes)``; the public
    weight is ``(out_channels, in_channels, *modes)``.
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        rank: float | Sequence[int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the Tucker core and factor matrices for the given tensor shape."""
        super().__init__()
        self.tensor_shape = tuple(tensor_shape)
        self.out_channels = tensor_shape[0]
        self.in_channels = tensor_shape[1]
        self.modes = tuple(tensor_shape[2:])
        self.ndim = len(self.modes)  # Spatial dimensions

        # Internal (in, out, *modes) factor dimensions and matching ranks.
        self._internal_dims = (self.in_channels, self.out_channels, *self.modes)
        self.ranks = tucker_ranks(self._internal_dims, rank)

        order = len(self._internal_dims)
        std = tucker_factor_std(self.ranks, order)
        keys = jax.random.split(rngs.params(), order + 1)
        self.core_real, self.core_imag = _split_real_imag(keys[0], self.ranks, std)
        factor_real: list[nnx.Param] = []
        factor_imag: list[nnx.Param] = []
        for i, (dim, rk) in enumerate(zip(self._internal_dims, self.ranks, strict=False)):
            real, imag = _split_real_imag(keys[i + 1], (dim, rk), std)
            factor_real.append(real)
            factor_imag.append(imag)
        self.factors_real = nnx.List(factor_real)
        self.factors_imag = nnx.List(factor_imag)

    def _complex_parts(self) -> tuple[jax.Array, list[jax.Array]]:
        """Assemble the complex Tucker core and factor matrices from real/imag parts."""
        core = self.core_real[...] + 1j * self.core_imag[...]
        factors = [
            r[...] + 1j * im[...]
            for r, im in zip(self.factors_real, self.factors_imag, strict=False)
        ]
        return core, factors

    def multiply_factorized(self, x: jax.Array) -> jax.Array:
        """Contract input with the factorized weight.

        Args:
            x: Input tensor of shape (batch, in_channels, *spatial_modes).

        Returns:
            Output tensor of shape (batch, out_channels, *spatial_modes).
        """
        core, factors = self._complex_parts()
        return contract_tucker(x, core, factors)

    def reconstruct(self) -> jax.Array:
        """Reconstruct the full weight tensor as (out_channels, in_channels, *modes)."""
        core, factors = self._complex_parts()
        full = tucker_to_tensor(core, factors)  # (in, out, *modes)
        return jnp.swapaxes(full, 0, 1)

    def parameter_count(self) -> int:
        """Count parameters in the factorized representation."""
        return tucker_parameter_count(self._internal_dims, self.ranks)


class CPDecomposition(nnx.Module):
    """CP / PARAFAC low-rank factorization of a spectral convolution weight.

    Stores complex CP weights ``(rank,)`` and one complex ``(dim, rank)`` factor
    per mode. The factorized contraction is a port of neuraloperator
    ``_contract_cp``; the reconstruct is tensorly ``cp_to_tensor``. Internal mode
    order is ``(in, out, *modes)``; the public weight is
    ``(out_channels, in_channels, *modes)``.
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        rank: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the CP rank-one factor matrices for the given tensor shape."""
        super().__init__()
        self.tensor_shape = tuple(tensor_shape)
        self.out_channels = tensor_shape[0]
        self.in_channels = tensor_shape[1]
        self.modes = tuple(tensor_shape[2:])
        self.ndim = len(self.modes)
        self.rank = max(1, int(rank))

        self._internal_dims = (self.in_channels, self.out_channels, *self.modes)
        order = len(self._internal_dims)
        std = cp_factor_std(self.rank, order)
        keys = jax.random.split(rngs.params(), order + 1)
        # CP weights are initialised to one (tltorch cp_init); kept as a learnable
        # real/imag-split parameter for symmetry with the factors.
        self.weights_real = nnx.Param(jnp.ones((self.rank,)))
        self.weights_imag = nnx.Param(jnp.zeros((self.rank,)))
        factor_real: list[nnx.Param] = []
        factor_imag: list[nnx.Param] = []
        for i, dim in enumerate(self._internal_dims):
            real, imag = _split_real_imag(keys[i + 1], (dim, self.rank), std)
            factor_real.append(real)
            factor_imag.append(imag)
        self.factors_real = nnx.List(factor_real)
        self.factors_imag = nnx.List(factor_imag)

    def _complex_parts(self) -> tuple[jax.Array, list[jax.Array]]:
        """Assemble the complex CP weights and factor matrices from real/imag parts."""
        weights = self.weights_real[...] + 1j * self.weights_imag[...]
        factors = [
            r[...] + 1j * im[...]
            for r, im in zip(self.factors_real, self.factors_imag, strict=False)
        ]
        return weights, factors

    def multiply_factorized(self, x: jax.Array) -> jax.Array:
        """Contract input with the factorized weight -> (batch, out_channels, *modes)."""
        weights, factors = self._complex_parts()
        return contract_cp(x, weights, factors)

    def reconstruct(self) -> jax.Array:
        """Reconstruct the full weight tensor as (out_channels, in_channels, *modes)."""
        weights, factors = self._complex_parts()
        full = cp_to_tensor(weights, factors)  # (in, out, *modes)
        return jnp.swapaxes(full, 0, 1)

    def parameter_count(self) -> int:
        """Count parameters in the factorized representation."""
        return cp_parameter_count(self._internal_dims, self.rank)


class TensorTrainDecomposition(nnx.Module):
    """Tensor-Train low-rank factorization of a spectral convolution weight.

    Stores complex 3D cores ``[(1, in, r1), (r1, out, r2), (r2, mode_0, r3), ...]``
    with TT-ranks capped at ``max_rank``. The factorized contraction is a port of
    neuraloperator ``_contract_tt``; the reconstruct is tensorly ``tt_to_tensor``.
    Internal mode order is ``(in, out, *modes)``; the public weight is
    ``(out_channels, in_channels, *modes)``.
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        max_rank: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the tensor-train cores for the given tensor shape."""
        super().__init__()
        self.tensor_shape = tuple(tensor_shape)
        self.out_channels = tensor_shape[0]
        self.in_channels = tensor_shape[1]
        self.modes = tuple(tensor_shape[2:])
        self.ndim = len(self.modes)
        self.max_rank = max(1, int(max_rank))

        self._internal_dims = (self.in_channels, self.out_channels, *self.modes)
        self.tt_rank_chain = tt_ranks(self._internal_dims, self.max_rank)

        order = len(self._internal_dims)
        std = tt_factor_std(self.tt_rank_chain, order)
        keys = jax.random.split(rngs.params(), order)
        core_real: list[nnx.Param] = []
        core_imag: list[nnx.Param] = []
        for i, dim in enumerate(self._internal_dims):
            shape = (self.tt_rank_chain[i], dim, self.tt_rank_chain[i + 1])
            real, imag = _split_real_imag(keys[i], shape, std)
            core_real.append(real)
            core_imag.append(imag)
        self.cores_real = nnx.List(core_real)
        self.cores_imag = nnx.List(core_imag)

    def _complex_cores(self) -> list[jax.Array]:
        """Assemble the complex tensor-train cores from their real/imag parts."""
        return [
            r[...] + 1j * im[...] for r, im in zip(self.cores_real, self.cores_imag, strict=False)
        ]

    def multiply_factorized(self, x: jax.Array) -> jax.Array:
        """Contract input with the factorized weight -> (batch, out_channels, *modes)."""
        return contract_tt(x, self._complex_cores())

    def reconstruct(self) -> jax.Array:
        """Reconstruct the full weight tensor as (out_channels, in_channels, *modes)."""
        full = tt_to_tensor(self._complex_cores())  # (in, out, *modes)
        return jnp.swapaxes(full, 0, 1)

    def parameter_count(self) -> int:
        """Count parameters in the factorized representation."""
        return tt_parameter_count(self._internal_dims, self.max_rank)


def make_decomposition(
    factorization: str,
    tensor_shape: Sequence[int],
    rank: float,
    *,
    rngs: nnx.Rngs,
) -> TuckerDecomposition | CPDecomposition | TensorTrainDecomposition:
    """Build the requested low-rank factorization of a spectral weight.

    Centralises the rank convention so every caller allocates identical
    factorized weights (Rule 1, no duplication):

    - ``tucker``: ``rank`` is a per-mode compression ratio (or explicit sequence).
    - ``cp`` / ``tt``: ``rank`` is a ratio of ``min(tensor_shape)`` giving the
      (clamped) CP rank / maximum TT rank.

    Args:
        factorization: Decomposition family (``tucker`` / ``cp`` / ``tt``).
        tensor_shape: Full weight shape ``(out, in, *modes)``.
        rank: Compression ratio (see above).
        rngs: Random number generators.

    Returns:
        The instantiated decomposition module.

    Raises:
        ValueError: If ``factorization`` is not one of ``tucker``/``cp``/``tt``.
    """
    if factorization == "tucker":
        return TuckerDecomposition(tensor_shape, rank, rngs=rngs)
    if factorization == "cp":
        return CPDecomposition(tensor_shape, max(1, int(rank * min(tensor_shape))), rngs=rngs)
    if factorization == "tt":
        return TensorTrainDecomposition(
            tensor_shape, max(1, int(rank * min(tensor_shape))), rngs=rngs
        )
    raise ValueError(f"Unknown decomposition type: {factorization}")


__all__ = [
    "CPDecomposition",
    "TensorDecomposition",
    "TensorTrainDecomposition",
    "TuckerDecomposition",
    "make_decomposition",
]


# ``Literal`` re-exported for callers that constrain the factorization family.
FactorizationType = Literal["tucker", "cp", "tt"]
