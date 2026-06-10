r"""Equivariant gate nonlinearity.

A pointwise activation cannot be applied to non-scalar (``l > 0``) channels
without breaking equivariance.  The *gate* (Weiler et al. 2018; Geiger & Smidt
2022) keeps equivariance by:

* activating the scalar (``l = 0``) channels directly, and
* scaling each higher-``l`` multiplicity by an *activated scalar gate* -- a
  multiplication of an equivariant vector by an invariant scalar is equivariant.

Ported from ``e3nn-jax`` (``../e3nn-jax/e3nn_jax/_src/gate.py``).  As in the
reference, the gate scalars are the **rightmost** scalars of the input: with
``n`` non-scalar multiplicities, the last ``n`` scalar channels become gates and
the remaining scalars are "extra" channels that pass through ``even_act`` /
``odd_act``.  The output drops the gate scalars.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.equivariant._assembly import from_chunks
from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray


def soft_odd(x: jax.Array) -> jax.Array:
    r"""Smooth odd activation ``(1 - exp(-x^2)) x`` for odd scalars.

    The default activation for ``0o`` channels in ``e3nn-jax``: odd
    (``f(-x) = -f(x)``) so that it preserves the parity of an odd scalar.
    """
    return (1.0 - jnp.exp(-(x**2))) * x


def gate(
    x: IrrepsArray,
    *,
    even_act: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
    odd_act: Callable[[jax.Array], jax.Array] = soft_odd,
    gate_act: Callable[[jax.Array], jax.Array] = jax.nn.sigmoid,
) -> IrrepsArray:
    r"""Apply the equivariant gate nonlinearity.

    Args:
        x: Input feature.  Its scalar (``l = 0``) blocks must come first and
            number at least the count of non-scalar multiplicities (the gates).
        even_act: Activation for even scalars (``0e``). Default :func:`jax.nn.gelu`.
        odd_act: Activation for odd scalars (``0o``). Default :func:`soft_odd`.
        gate_act: Activation applied to the gate scalars. Default
            :func:`jax.nn.sigmoid`.

    Returns:
        An :class:`IrrepsArray` whose scalar blocks are the activated extra
        scalars and whose non-scalar blocks are the gated (scaled) inputs.

    Raises:
        ValueError: If there are fewer scalar channels than non-scalar
            multiplicities (no gate available for some vector).
    """
    scalar_blocks = [
        (index, block) for index, block in enumerate(x.irreps.blocks) if block[1].l == 0
    ]
    vector_blocks = [
        (index, block) for index, block in enumerate(x.irreps.blocks) if block[1].l > 0
    ]
    chunks = x.chunks
    leading_shape = x.array.shape[:-1]
    dtype = x.array.dtype

    num_gates = sum(mul for _, (mul, _) in vector_blocks)
    if num_gates == 0:
        activated: list[jax.Array | None] = [
            _activate_scalar(chunks[index], irrep.p, even_act, odd_act)
            for index, (_, irrep) in scalar_blocks
        ]
        return from_chunks(x.irreps, activated, leading_shape, dtype)

    num_scalars = sum(mul for _, (mul, _) in scalar_blocks)
    if num_scalars < num_gates:
        raise ValueError(
            f"gate requires at least as many scalars ({num_scalars}) as non-scalar "
            f"irreps ({num_gates}); input irreps {x.irreps!r}"
        )

    # Flatten scalar multiplicities, split into "extra" (left) and "gates" (right).
    scalar_flat = jnp.concatenate(
        [chunks[index].reshape(*leading_shape, mul) for index, (mul, _) in scalar_blocks], axis=-1
    )
    num_extra = num_scalars - num_gates
    extra_values = scalar_flat[..., :num_extra]
    gate_values = gate_act(scalar_flat[..., num_extra:])

    output_irreps, output_chunks = _build_gated_output(
        scalar_blocks,
        vector_blocks,
        chunks,
        extra_values,
        gate_values,
        num_extra,
        leading_shape,
        even_act,
        odd_act,
    )
    return from_chunks(output_irreps, output_chunks, leading_shape, dtype)


def _activate_scalar(
    chunk: jax.Array,
    parity: int,
    even_act: Callable[[jax.Array], jax.Array],
    odd_act: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Apply the parity-appropriate scalar activation to a ``0e``/``0o`` chunk."""
    return even_act(chunk) if parity == 1 else odd_act(chunk)


def _build_gated_output(
    scalar_blocks: list[tuple[int, tuple[int, Irrep]]],
    vector_blocks: list[tuple[int, tuple[int, Irrep]]],
    chunks: list[jax.Array],
    extra_values: jax.Array,
    gate_values: jax.Array,
    num_extra: int,
    leading_shape: tuple[int, ...],  # noqa: ARG001 - gated-output builder interface
    even_act: Callable[[jax.Array], jax.Array],
    odd_act: Callable[[jax.Array], jax.Array],
) -> tuple[Irreps, list[jax.Array | None]]:
    """Assemble the output irreps/chunks: activated extra scalars + gated vectors."""
    out_blocks: list[tuple[int, Irrep]] = []
    out_chunks: list[jax.Array | None] = []

    # Extra scalars: re-split into their original blocks and activate per parity.
    cursor = 0
    for _, (mul, irrep) in scalar_blocks:
        if cursor >= num_extra:
            break
        take = min(mul, num_extra - cursor)
        block_values = extra_values[..., cursor : cursor + take]
        out_blocks.append((take, irrep))
        out_chunks.append(
            _activate_scalar(
                block_values[..., None, :].swapaxes(-1, -2), irrep.p, even_act, odd_act
            )
        )
        cursor += take

    # Gated vectors: scale each multiplicity by its corresponding gate scalar.
    gate_cursor = 0
    for index, (mul, irrep) in vector_blocks:
        block_gates = gate_values[..., gate_cursor : gate_cursor + mul]
        gated = chunks[index] * block_gates[..., None]
        out_blocks.append((mul, irrep))
        out_chunks.append(gated)
        gate_cursor += mul

    return Irreps(tuple(out_blocks)), out_chunks


class Gate(nnx.Module):
    """Configured wrapper around :func:`gate` for use in module stacks.

    Stores the activation choices (and validates the input irreps on call) so the
    gate can be dropped into an ``nnx`` module sequence like a layer.  It has no
    learnable parameters.
    """

    def __init__(
        self,
        irreps_in: Irreps | str,
        *,
        even_act: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
        odd_act: Callable[[jax.Array], jax.Array] = soft_odd,
        gate_act: Callable[[jax.Array], jax.Array] = jax.nn.sigmoid,
    ) -> None:
        """Configure the gate for a fixed input layout.

        Args:
            irreps_in: Expected input layout (validated on call).
            even_act: Activation for even scalars. Default :func:`jax.nn.gelu`.
            odd_act: Activation for odd scalars. Default :func:`soft_odd`.
            gate_act: Activation for the gate scalars. Default
                :func:`jax.nn.sigmoid`.
        """
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self._even_act = even_act
        self._odd_act = odd_act
        self._gate_act = gate_act

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        """Apply the configured gate.

        Args:
            x: Input feature with ``x.irreps == self.irreps_in``.

        Returns:
            The gated :class:`IrrepsArray`.

        Raises:
            ValueError: If ``x.irreps`` does not match the configured layout.
        """
        if x.irreps != self.irreps_in:
            raise ValueError(f"Gate expected input irreps {self.irreps_in!r}, got {x.irreps!r}")
        return gate(x, even_act=self._even_act, odd_act=self._odd_act, gate_act=self._gate_act)
