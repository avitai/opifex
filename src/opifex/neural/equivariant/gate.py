r"""Equivariant gate nonlinearity.

A pointwise activation cannot be applied to non-scalar (``l > 0``) channels
without breaking equivariance.  The *gate* (Weiler et al. 2018; Geiger & Smidt
2022) keeps equivariance by:

* activating the scalar (``l = 0``) channels directly, and
* scaling each higher-``l`` multiplicity by an *activated scalar gate* -- a
  multiplication of an equivariant vector by an invariant scalar is equivariant.

The gate scalars are the **rightmost** scalars of the input: with ``n`` non-scalar
multiplicities, the last ``n`` scalar channels become gates and the remaining
scalars are "extra" channels that pass through ``even_act`` / ``odd_act``.  The
output drops the gate scalars.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.neural.equivariant._assembly import from_chunks
from opifex.neural.equivariant._invariants import norm
from opifex.neural.equivariant.irreps import Irrep, Irreps, IrrepsArray


# Probabilists' Hermite (Gauss-Hermite) nodes/weights for the standard normal,
# used to normalise activations to unit second moment. Module-level constants so
# the resulting scalar factor is a compile-time fold.
_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite_e.hermegauss(64)
_GAUSSIAN_WEIGHT_NORM = float(np.sqrt(2.0 * np.pi))


def soft_odd(x: jax.Array) -> jax.Array:
    r"""Smooth odd activation ``(1 - exp(-x^2)) x`` for odd scalars.

    The standard activation for ``0o`` channels: odd (``f(-x) = -f(x)``) so that
    it preserves the parity of an odd scalar.
    """
    return (1.0 - jnp.exp(-(x**2))) * x


def normalize_activation(
    act: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    r"""Rescale ``act`` to unit second moment under a standard normal input.

    Returns ``x -> act(x) / sqrt(E_{z~N(0,1)}[act(z)^2])``, so an activation fed
    unit-variance scalars emits unit-variance outputs and the downstream
    weight-init variance budget is preserved. The second moment is evaluated once
    by Gauss-Hermite quadrature; the resulting scalar factor is constant-folded by
    XLA.
    """
    nodes = jnp.asarray(_HERMITE_NODES)
    weights = jnp.asarray(_HERMITE_WEIGHTS)
    second_moment = jnp.sum(weights * act(nodes) ** 2) / _GAUSSIAN_WEIGHT_NORM
    factor = jax.lax.rsqrt(second_moment)
    return lambda x: act(x) * factor


def gate(
    x: IrrepsArray,
    *,
    even_act: Callable[[jax.Array], jax.Array] = jax.nn.gelu,
    odd_act: Callable[[jax.Array], jax.Array] = soft_odd,
    gate_act: Callable[[jax.Array], jax.Array] = jax.nn.sigmoid,
    normalize_act: bool = False,
) -> IrrepsArray:
    r"""Apply the equivariant gate nonlinearity.

    Args:
        x: Input feature.  Its scalar (``l = 0``) blocks must come first and
            number at least the count of non-scalar multiplicities (the gates).
        even_act: Activation for even scalars (``0e``). Default :func:`jax.nn.gelu`.
        odd_act: Activation for odd scalars (``0o``). Default :func:`soft_odd`.
        gate_act: Activation applied to the gate scalars. Default
            :func:`jax.nn.sigmoid`.
        normalize_act: If ``True``, each activation is rescaled to unit second
            moment under a standard-normal input (see :func:`normalize_activation`)
            so feature magnitudes do not drift across stacked gated layers. Default
            ``False`` (the consumer opts in -- e.g. the NequIP backbone).

    Returns:
        An :class:`IrrepsArray` whose scalar blocks are the activated extra
        scalars and whose non-scalar blocks are the gated (scaled) inputs.

    Raises:
        ValueError: If there are fewer scalar channels than non-scalar
            multiplicities (no gate available for some vector).
    """
    if normalize_act:
        even_act = normalize_activation(even_act)
        odd_act = normalize_activation(odd_act)
        gate_act = normalize_activation(gate_act)
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


class NormGate(nnx.Module):
    r"""Norm-gated equivariant nonlinearity.

    Unlike :func:`gate` -- which consumes the *rightmost* scalars as dedicated
    gates -- this gate drives **every** multiplicity from a learnable MLP of the
    input scalars concatenated with the per-irrep **norms** of the non-scalar
    channels (the gating signal is therefore rotation-invariant). The MLP output
    replaces the scalar channels and scales each non-scalar multiplicity, so the
    output layout equals the input layout. This is the nonlinearity used
    throughout the QHNet self / pair interaction layers (Yu et al. 2023).

    Scaling an equivariant feature by an invariant gate is equivariant, so the
    module is rotation-equivariant; reusing :func:`opifex.neural.equivariant.norm`
    gives a NaN-safe gradient at zero vectors.
    """

    def __init__(self, irreps: Irreps | str, *, rngs: nnx.Rngs) -> None:
        """Build the gate MLP for a fixed input layout.

        Args:
            irreps: Expected input layout (validated on call). Must carry at least
                one ``l = 0`` scalar channel to drive the gates.
            rngs: Random number generators (keyword-only) seeding the MLP.

        Raises:
            ValueError: If ``irreps`` has no scalar channel.
        """
        super().__init__()
        self.irreps = Irreps(irreps)
        self._num_scalars = sum(mul for mul, irrep in self.irreps.blocks if irrep.l == 0)
        self._num_vectors = sum(mul for mul, irrep in self.irreps.blocks if irrep.l > 0)
        if self._num_scalars == 0:
            raise ValueError(
                f"NormGate needs at least one scalar channel to gate, got {self.irreps!r}"
            )
        width = self._num_scalars + self._num_vectors
        self.hidden = nnx.Linear(width, width, rngs=rngs)
        self.readout = nnx.Linear(width, width, rngs=rngs)

    def _gate_signal(self, x: IrrepsArray) -> tuple[Float[Array, "... width"], list[Array]]:
        """Return the MLP gate vector and the per-block scalar input chunks."""
        leading = x.array.shape[:-1]
        scalar_chunks = [
            chunk.reshape(*leading, mul)
            for (mul, irrep), chunk in zip(self.irreps.blocks, x.chunks, strict=True)
            if irrep.l == 0
        ]
        scalars = jnp.concatenate(scalar_chunks, axis=-1)
        if self._num_vectors > 0:
            vector_blocks = tuple((mul, irrep) for mul, irrep in self.irreps.blocks if irrep.l > 0)
            vector_flat = jnp.concatenate(
                [
                    chunk.reshape(*leading, mul * irrep.dim)
                    for (mul, irrep), chunk in zip(self.irreps.blocks, x.chunks, strict=True)
                    if irrep.l > 0
                ],
                axis=-1,
            )
            vector_norms = norm(IrrepsArray(Irreps(vector_blocks), vector_flat)).array
            signal = jnp.concatenate([scalars, vector_norms], axis=-1)
        else:
            signal = scalars
        gates = self.readout(jax.nn.silu(self.hidden(signal)))
        return gates, scalar_chunks

    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        """Apply the norm gate.

        Args:
            x: Input feature with ``x.irreps == self.irreps``.

        Returns:
            The gated :class:`IrrepsArray` with the same layout as ``x``.

        Raises:
            ValueError: If ``x.irreps`` does not match the configured layout.
        """
        if x.irreps != self.irreps:
            raise ValueError(f"NormGate expected input irreps {self.irreps!r}, got {x.irreps!r}")
        leading = x.array.shape[:-1]
        gates, _ = self._gate_signal(x)
        scalar_gates = gates[..., : self._num_scalars]
        vector_gates = gates[..., self._num_scalars :]
        out_chunks: list[Array | None] = []
        scalar_cursor = 0
        vector_cursor = 0
        for (mul, irrep), chunk in zip(self.irreps.blocks, x.chunks, strict=True):
            if irrep.l == 0:
                block = scalar_gates[..., scalar_cursor : scalar_cursor + mul]
                out_chunks.append(block.reshape(*leading, mul, 1))
                scalar_cursor += mul
            else:
                block_gates = vector_gates[..., vector_cursor : vector_cursor + mul]
                out_chunks.append(chunk * block_gates[..., None])
                vector_cursor += mul
        return from_chunks(self.irreps, out_chunks, leading, x.array.dtype)
