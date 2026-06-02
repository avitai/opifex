"""Pure-JAX low-rank tensor factorizations for spectral convolution weights.

This module provides transform-safe (jit / grad / vmap compatible) building
blocks shared by :mod:`tensorized` and :mod:`tensorly_integration`: faithful
ports of the CP / Tucker / Tensor-Train reconstruct formulas and the
memory-optimal factorized contractions used by tensorized FNOs. Centralising
the decomposition math here keeps the two public modules DRY (Rule 1).

The factorized weight has the internal layout ``(in, out, *modes)`` (matching
neuraloperator's spectral-convolution weight ordering); callers that expose the
opifex convention ``(out, in, *modes)`` swap the first two axes at the boundary.

References (cloned sibling repos, read before porting -- never invent formulas):

Reconstruct formulas (tensorly 0.9.0):
- CP     : ``tensorly/tensorly/cp_tensor.py:433`` ``cp_to_tensor`` -- the
  Khatri-Rao sum, written here as the equivalent ``sum_r w_r outer(U_k[:, r])``
  einsum (the alternate form noted in that docstring).
- Tucker : ``tensorly/tensorly/tucker_tensor.py:50`` ``tucker_to_tensor`` =
  ``multi_mode_dot(core, factors)`` (``tenalg/core_tenalg/n_mode_product.py:79``).
- TT     : ``tensorly/tensorly/tt_tensor.py:61`` ``tt_to_tensor`` -- sequential
  reshape + dot over the 3D cores.

Memory-optimal factorized contractions (neuraloperator):
- CP     : ``neuraloperator/neuralop/layers/spectral_convolution.py:55``
  ``_contract_cp``.
- Tucker : ``...:76`` ``_contract_tucker``.
- TT     : ``...:106`` ``_contract_tt``.

Complex factorization layout + learnable-factor init (tltorch):
- ``tltorch/tltorch/factorized_tensors/factorized_tensors.py`` (factor shapes:
  CP line 102, Tucker line 241, TT line 386).
- ``tltorch/tltorch/factorized_tensors/init.py`` (``cp_init`` / ``tucker_init`` /
  ``tt_init`` standard deviations).

Paper: Kossaifi et al., "Multi-Grid Tensorized Fourier Neural Operator".
"""

from collections.abc import Sequence
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np


# Lowercase-then-uppercase pool, identical to neuraloperator's ``einsum_symbols``
# (spectral_convolution.py:18) so the ported equations are byte-faithful.
EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------------------- #
# Reconstruct formulas (tensorly 0.9.0 ports)
# --------------------------------------------------------------------------- #
def cp_to_tensor(weights: jax.Array, factors: Sequence[jax.Array]) -> jax.Array:
    """Reconstruct a full tensor from a CP / PARAFAC factorization.

    Port of ``tensorly.cp_tensor.cp_to_tensor`` (cp_tensor.py:433), using the
    equivalent ``sum_r weights[r] * outer_k(factors[k][:, r])`` einsum form (the
    alternate implementation noted in that function's docstring) so the result
    is a single transform-safe contraction.

    Args:
        weights: CP weights of shape ``(rank,)``.
        factors: One factor matrix ``(dim_k, rank)`` per tensor mode ``k``.

    Returns:
        Full tensor of shape ``(factors[0].shape[0], ..., factors[-1].shape[0])``.
    """
    order = len(factors)
    mode_syms = EINSUM_SYMBOLS[:order]
    rank_sym = EINSUM_SYMBOLS[order]
    factor_terms = ",".join(f"{mode_syms[k]}{rank_sym}" for k in range(order))
    equation = f"{factor_terms},{rank_sym}->{mode_syms}"
    return jnp.einsum(equation, *factors, weights)


def tucker_to_tensor(core: jax.Array, factors: Sequence[jax.Array]) -> jax.Array:
    """Reconstruct a full tensor from a Tucker factorization.

    Port of ``tensorly.tucker_tensor.tucker_to_tensor`` (tucker_tensor.py:50),
    which is ``multi_mode_dot(core, factors)``. Expressed here as the single
    einsum ``core x_k factors[k]`` equivalent to the chained mode products in
    ``tenalg/core_tenalg/n_mode_product.py:79``.

    Args:
        core: Tucker core of shape ``(rank_0, ..., rank_{N-1})``.
        factors: One factor matrix ``(dim_k, rank_k)`` per core mode ``k``.

    Returns:
        Full tensor of shape ``(factors[0].shape[0], ..., factors[-1].shape[0])``.
    """
    order = len(factors)
    core_syms = EINSUM_SYMBOLS[:order]
    out_syms = EINSUM_SYMBOLS[order : 2 * order]
    factor_terms = ",".join(f"{out_syms[k]}{core_syms[k]}" for k in range(order))
    equation = f"{core_syms},{factor_terms}->{out_syms}"
    return jnp.einsum(equation, core, *factors)


def tt_to_tensor(cores: Sequence[jax.Array]) -> jax.Array:
    """Reconstruct a full tensor from a Tensor-Train (MPS) factorization.

    Port of ``tensorly.tt_tensor.tt_to_tensor`` (tt_tensor.py:61): start from the
    mode-0 unfolding of the first 3D core and sequentially reshape + matmul each
    remaining core ``(rank_prev, dim, rank_next)``.

    Args:
        cores: TT cores; ``cores[k]`` has shape ``(rank_k, dim_k, rank_{k+1})``
            with ``rank_0 == rank_N == 1``.

    Returns:
        Full tensor of shape ``(dim_0, ..., dim_{N-1})``.
    """
    full_shape = [int(core.shape[1]) for core in cores]
    full_tensor = cores[0].reshape(full_shape[0], -1)
    for core in cores[1:]:
        rank_prev, _, rank_next = core.shape
        full_tensor = full_tensor @ core.reshape(rank_prev, -1)
        full_tensor = full_tensor.reshape(-1, rank_next)
    return full_tensor.reshape(full_shape)


# --------------------------------------------------------------------------- #
# Factorized contractions (neuraloperator ports), x: (batch, in, *modes)
# --------------------------------------------------------------------------- #
def contract_cp(x: jax.Array, weights: jax.Array, factors: Sequence[jax.Array]) -> jax.Array:
    """Contract input directly with CP factors (no full reconstruct).

    Port of ``neuraloperator._contract_cp`` (spectral_convolution.py:55) for the
    non-separable case. The equation is built programmatically from ``x.ndim`` so
    1-D / 2-D / 3-D inputs all work, with the channel axis (position 1) mapped
    from ``in`` to ``out``.

    Args:
        x: Input of shape ``(batch, in_channels, *modes)``.
        weights: CP weights ``(rank,)``.
        factors: ``[factor_in (in, rank), factor_out (out, rank),
            factor_mode_k (mode_k, rank) ...]``.

    Returns:
        Output of shape ``(batch, out_channels, *modes)``.
    """
    order = x.ndim
    x_syms = EINSUM_SYMBOLS[:order]
    rank_sym = EINSUM_SYMBOLS[order]
    out_sym = EINSUM_SYMBOLS[order + 1]
    out_syms = list(x_syms)
    out_syms[1] = out_sym
    factor_syms = [EINSUM_SYMBOLS[1] + rank_sym, out_sym + rank_sym]
    factor_syms += [f"{xs}{rank_sym}" for xs in x_syms[2:]]
    equation = f"{x_syms},{rank_sym},{','.join(factor_syms)}->{''.join(out_syms)}"
    return jnp.einsum(equation, x, weights, *factors)


def contract_tucker(x: jax.Array, core: jax.Array, factors: Sequence[jax.Array]) -> jax.Array:
    """Contract input directly with Tucker core and factors (no full reconstruct).

    Port of ``neuraloperator._contract_tucker`` (spectral_convolution.py:76) for
    the non-separable case.

    Args:
        x: Input of shape ``(batch, in_channels, *modes)``.
        core: Tucker core ``(rank_in, rank_out, rank_mode_0, ...)``.
        factors: ``[factor_in (in, rank_in), factor_out (out, rank_out),
            factor_mode_k (mode_k, rank_mode_k) ...]``.

    Returns:
        Output of shape ``(batch, out_channels, *modes)``.
    """
    order = x.ndim
    x_syms = EINSUM_SYMBOLS[:order]
    out_sym = EINSUM_SYMBOLS[order]
    out_syms = list(x_syms)
    out_syms[1] = out_sym
    core_syms = EINSUM_SYMBOLS[order + 1 : 2 * order + 1]
    factor_syms = [EINSUM_SYMBOLS[1] + core_syms[0], out_sym + core_syms[1]]
    factor_syms += [f"{xs}{rs}" for xs, rs in zip(x_syms[2:], core_syms[2:], strict=False)]
    equation = f"{x_syms},{core_syms},{','.join(factor_syms)}->{''.join(out_syms)}"
    return jnp.einsum(equation, x, core, *factors)


def contract_tt(x: jax.Array, cores: Sequence[jax.Array]) -> jax.Array:
    """Contract input directly with TT cores (no full reconstruct).

    Port of ``neuraloperator._contract_tt`` (spectral_convolution.py:106) for the
    non-separable case. The weight modes are ``(in, out, *modes)`` -- ``out`` is
    inserted after ``in`` -- and each core contributes the index triple
    ``(rank_k, mode_sym_k, rank_{k+1})``.

    Args:
        x: Input of shape ``(batch, in_channels, *modes)``.
        cores: TT cores ``[(1, in, r1), (r1, out, r2), (r2, mode_0, r3), ...]``.

    Returns:
        Output of shape ``(batch, out_channels, *modes)``.
    """
    order = x.ndim
    x_syms = list(EINSUM_SYMBOLS[:order])
    weight_syms = list(x_syms[1:])  # drop batch
    weight_syms.insert(1, EINSUM_SYMBOLS[order])  # insert output symbol after input
    out_syms = list(weight_syms)
    out_syms[0] = x_syms[0]
    rank_syms = list(EINSUM_SYMBOLS[order + 1 :])
    tt_syms = [[rank_syms[i], sym, rank_syms[i + 1]] for i, sym in enumerate(weight_syms)]
    equation = (
        "".join(x_syms)
        + ","
        + ",".join("".join(triple) for triple in tt_syms)
        + "->"
        + "".join(out_syms)
    )
    return jnp.einsum(equation, x, *cores)


# --------------------------------------------------------------------------- #
# Centered-band spectral convolution (shared by tensorized.py and base.py)
# --------------------------------------------------------------------------- #
class _FactorizedWeight(Protocol):
    """Anything exposing the memory-optimal factorized contraction."""

    out_channels: int

    def multiply_factorized(self, x: jax.Array) -> jax.Array:
        """Contract ``(batch, in, *modes)`` against the factorized weight."""
        ...


def factorized_spectral_conv(
    weight: "_FactorizedWeight", x: jax.Array, modes: Sequence[int]
) -> jax.Array:
    """Spectral convolution over the centered low-frequency band with a factorized weight.

    Mirrors the neuraloperator spectral-convolution frequency handling: take the
    real FFT, ``fftshift`` every axis except the (already one-sided) last so the
    zero frequency is centered, keep a symmetric band of ``modes_k`` coefficients
    around it (the low band from DC on the half-spectrum last axis), contract the
    band against the factorized weight, scatter back, and invert. Keeping the
    *centered* band retains both positive and negative low frequencies — unlike a
    plain ``[:modes]`` slice, which discards the negative half and is lossy.

    Args:
        weight: Factorized weight exposing ``multiply_factorized`` and
            ``out_channels`` (a CP / Tucker / TT decomposition).
        x: Real input of shape ``(batch, in_channels, *spatial)``.
        modes: Number of retained Fourier modes per spatial axis.

    Returns:
        Real output of shape ``(batch, out_channels, *spatial)``.
    """
    axes = tuple(range(2, x.ndim))
    spatial = x.shape[2:]
    x_ft = jnp.fft.rfftn(x, axes=axes)
    shifted_axes = axes[:-1]  # every spatial axis except the one-sided rfft axis
    if shifted_axes:
        x_ft = jnp.fft.fftshift(x_ft, axes=shifted_axes)

    band: list[slice] = [slice(None), slice(None)]
    for dim, n_modes in enumerate(modes):
        full = x_ft.shape[2 + dim]
        kept = min(n_modes, full)
        if dim < len(modes) - 1:  # shifted axis: centered band around the DC bin
            start = full // 2 - kept // 2
            band.append(slice(start, start + kept))
        else:  # one-sided rfft axis: low band starting at DC
            band.append(slice(0, kept))
    band_index = tuple(band)

    out_band = weight.multiply_factorized(x_ft[band_index])
    out_ft = jnp.zeros((x.shape[0], weight.out_channels, *x_ft.shape[2:]), dtype=x_ft.dtype)
    out_ft = out_ft.at[band_index].set(out_band)
    if shifted_axes:
        out_ft = jnp.fft.ifftshift(out_ft, axes=shifted_axes)
    return jnp.fft.irfftn(out_ft, s=spatial, axes=axes)


# --------------------------------------------------------------------------- #
# Rank derivation + parameter-count helpers
# --------------------------------------------------------------------------- #
def tucker_ranks(tensor_shape: Sequence[int], rank: float | Sequence[int]) -> tuple[int, ...]:
    """Resolve Tucker ranks (one per mode) from a ratio or explicit sequence.

    A ``float`` ratio scales each mode independently (``max(1, round(rank * dim))``)
    and is clamped to the mode dimension, matching tltorch's proportional ranks.

    Args:
        tensor_shape: Full weight shape ``(out, in, *modes)``.
        rank: Compression ratio in ``(0, 1]`` or an explicit per-mode rank sequence.

    Returns:
        One integer rank per tensor mode.
    """
    if isinstance(rank, float):
        return tuple(max(1, min(dim, round(rank * dim))) for dim in tensor_shape)
    if isinstance(rank, int):
        return tuple(min(rank, dim) for dim in tensor_shape)
    ranks = tuple(int(r) for r in rank)
    if len(ranks) != len(tensor_shape):
        raise ValueError(
            f"Tucker rank sequence has {len(ranks)} entries but tensor has "
            f"{len(tensor_shape)} modes."
        )
    return tuple(min(r, dim) for r, dim in zip(ranks, tensor_shape, strict=False))


def tt_ranks(tensor_shape: Sequence[int], max_rank: int) -> tuple[int, ...]:
    """Compute clamped TT-ranks for a tensor, capped at ``max_rank``.

    Applies the standard TT-rank bound (the realisable rank cannot exceed the
    product of the dimensions on either side of a split, see
    ``tensorly.tt_tensor.validate_tt_rank``) and additionally clamps every
    internal rank to ``max_rank``. Boundary ranks are ``1``.

    Args:
        tensor_shape: Full weight shape ``(out, in, *modes)`` in the order the TT
            cores will iterate over (the caller supplies ``(in, out, *modes)``).
        max_rank: Maximum internal TT rank.

    Returns:
        ``len(tensor_shape) + 1`` ranks with unit first and last entries.
    """
    order = len(tensor_shape)
    ranks = [1] * (order + 1)
    for k in range(1, order):
        left = int(np.prod(tensor_shape[:k]))
        right = int(np.prod(tensor_shape[k:]))
        ranks[k] = int(min(max_rank, left, right))
    return tuple(ranks)


def cp_parameter_count(tensor_shape: Sequence[int], rank: int) -> int:
    """Parameters in a CP factorization: ``rank * sum(dims) + rank`` (weights)."""
    return int(rank * sum(tensor_shape) + rank)


def tucker_parameter_count(tensor_shape: Sequence[int], ranks: Sequence[int]) -> int:
    """Parameters in a Tucker factorization: ``prod(ranks) + sum(rank_k * dim_k)``."""
    core = int(np.prod(ranks))
    factors = sum(int(r) * int(d) for r, d in zip(ranks, tensor_shape, strict=False))
    return core + factors


def tt_parameter_count(tensor_shape: Sequence[int], max_rank: int) -> int:
    """Parameters in a TT factorization: ``sum(r_{k-1} * dim_k * r_k)``."""
    ranks = tt_ranks(tensor_shape, max_rank)
    return int(sum(ranks[k] * tensor_shape[k] * ranks[k + 1] for k in range(len(tensor_shape))))


# --------------------------------------------------------------------------- #
# Learnable-factor init standard deviations (tltorch init.py)
# --------------------------------------------------------------------------- #
def cp_factor_std(rank: int, order: int, std: float = 0.02) -> float:
    """CP factor std so the reconstruction has standard deviation ``std``.

    Port of ``tltorch.factorized_tensors.init.cp_init``: weights are set to one
    and each factor is drawn from ``N(0, (std / sqrt(rank)) ** (1 / order))``.
    """
    return float((std / np.sqrt(rank)) ** (1.0 / order))


def tucker_factor_std(ranks: Sequence[int], order: int, std: float = 0.02) -> float:
    """Tucker core/factor std so the reconstruction has standard deviation ``std``.

    Port of ``tltorch.factorized_tensors.init.tucker_init``: with
    ``r = prod(sqrt(rank_k))`` the core and every factor are drawn from
    ``N(0, (std / r) ** (1 / (order + 1)))``.
    """
    r = float(np.prod([np.sqrt(rank) for rank in ranks]))
    return float((std / r) ** (1.0 / (order + 1)))


def tt_factor_std(ranks: Sequence[int], order: int, std: float = 0.02) -> float:
    """TT core std so the reconstruction has standard deviation ``std``.

    Port of ``tltorch.factorized_tensors.init.tt_init``: with ``r = prod(rank)``
    each core is drawn from ``N(0, (std / r) ** (1 / order))``.
    """
    r = float(np.prod(ranks))
    return float((std / r) ** (1.0 / order))
