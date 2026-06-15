r"""Low-rank (LoRA) adapter for ``nnx.Linear`` layers in atomistic models.

LoRA (Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models",
arXiv:2106.09685) freezes a pre-trained weight ``W`` and learns a low-rank
correction so that the effective weight becomes

.. math:: W_{\text{eff}} = W + \frac{\alpha}{r}\, A B,

with ``A`` of shape ``(in, r)`` and ``B`` of shape ``(r, out)``. ``A`` is
initialised random-small and ``B`` is initialised to **zero**, so at the start of
fine-tuning the correction ``A B`` is exactly zero and the adapter reproduces the
base layer -- training therefore begins from the pre-trained model and only the
tiny ``A`` / ``B`` factors are optimised.

This is the JAX / Flax-NNX analogue of the ``../mace`` ``mace/modules/lora.py``
``LoRAFCLayer`` (which adapts the scalar MLP layers of an equivariant MACE model
with the same ``delta = A @ B`` low-rank, ``A`` random-small / ``B`` zero, and
the e3nn ``(in, out)`` weight layout). Flax's ``nnx.Linear`` uses the same
``kernel`` layout ``(in_features, out_features)``, so the correction is added
directly to ``kernel`` and the adapter is **equivariant-safe**: it acts per
output channel (each output feature is corrected independently), so wrapping the
per-irrep linear blocks of an equivariant backbone never mixes irreps.

:class:`LoRALinear` is an ``nnx.Module`` whose forward is a single fused matmul
with the effective kernel, so it is ``jit`` / ``grad`` / ``vmap`` clean and
introduces no Python-side control flow.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002


logger = logging.getLogger(__name__)

_DEFAULT_RANK = 4
_DEFAULT_ALPHA = 1.0
_LORA_A_INIT_STD = 1e-3
"""Standard deviation of the random small ``A`` init (matches ``mace`` LoRA)."""


class LoRALinear(nnx.Module):
    r"""A LoRA-adapted wrapper around a frozen ``nnx.Linear`` base layer.

    The forward computes ``x @ W_eff + b`` with
    ``W_eff = W + (alpha / rank) * A @ B`` (Hu et al. 2021, arXiv:2106.09685;
    the ``../mace`` ``mace/modules/lora.py`` ``LoRAFCLayer``). The base
    ``kernel`` and ``bias`` are stored as data leaves so that they travel with
    the model state but are typically excluded from the fine-tune gradient by a
    backbone-freeze filter (see
    :func:`opifex.neural.atomistic.foundation.trainable_filter`).

    Args:
        base: The pre-trained ``nnx.Linear`` to adapt. Its weights are reused
            (not copied); only the low-rank ``A`` / ``B`` factors are new
            parameters.
        rank: Low-rank bottleneck width ``r`` (the number of ``A`` columns /
            ``B`` rows). Must be a positive integer.
        alpha: LoRA scaling numerator; the correction is scaled by
            ``alpha / rank``.
        rngs: Random number generators seeding the random-small ``A`` init.
            Defaults to a fixed seed (the init is tiny, so the seed is
            inconsequential at the start of training).

    Raises:
        ValueError: If ``rank`` is not a positive integer.
    """

    def __init__(
        self,
        base: nnx.Linear,
        *,
        rank: int = _DEFAULT_RANK,
        alpha: float = _DEFAULT_ALPHA,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Store the base layer and create the zero-``B`` / small-``A`` factors."""
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be a positive integer, got {rank}.")
        in_features = base.in_features
        out_features = base.out_features
        self.base = base
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = float(alpha) / float(rank)

        rngs = rngs if rngs is not None else nnx.Rngs(0)
        a_init = _LORA_A_INIT_STD * jax.random.normal(
            rngs.params(), (in_features, rank), dtype=base.kernel.value.dtype
        )
        self.lora_a = nnx.Param(a_init)
        self.lora_b = nnx.Param(jnp.zeros((rank, out_features), dtype=base.kernel.value.dtype))

    def effective_kernel(self) -> Float[Array, "in_features out_features"]:
        """Return the fused kernel ``W + (alpha / rank) * A @ B``."""
        return self.base.kernel.value + self.scaling * (self.lora_a.value @ self.lora_b.value)

    def __call__(self, x: Float[Array, "... in_features"]) -> Float[Array, "... out_features"]:
        """Apply the LoRA-adapted linear map ``x @ W_eff + b``.

        Args:
            x: Input array whose last axis is ``in_features``.

        Returns:
            ``x @ (W + (alpha/rank) A @ B) + b`` with the base bias (if any).
        """
        out = x @ self.effective_kernel()
        if self.base.bias is not None:
            out = out + self.base.bias.value
        return out


def apply_lora(
    base: nnx.Linear,
    *,
    rank: int = _DEFAULT_RANK,
    alpha: float = _DEFAULT_ALPHA,
    rngs: nnx.Rngs | None = None,
) -> LoRALinear:
    """Wrap an ``nnx.Linear`` in a :class:`LoRALinear` adapter.

    A thin functional alias for :class:`LoRALinear` mirroring the ``../mace``
    ``inject_lora`` entry point.

    Args:
        base: The pre-trained ``nnx.Linear`` to adapt.
        rank: Low-rank bottleneck width.
        alpha: LoRA scaling numerator (correction scaled by ``alpha / rank``).
        rngs: Random number generators for the ``A`` init.

    Returns:
        The LoRA-adapted layer.
    """
    return LoRALinear(base, rank=rank, alpha=alpha, rngs=rngs)


__all__ = ["LoRALinear", "apply_lora"]
