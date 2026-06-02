r"""Irreducible-representation bookkeeping for E(3)-equivariant networks.

A native, dependency-free reimplementation of the irreps data model used by
``e3nn`` / ``e3nn-jax`` (Geiger & Smidt 2022, arXiv:2207.09453; reference:
``../e3nn-jax/e3nn_jax/_src/irreps.py`` and ``irreps_array.py``). Building this in
opifex (rather than depending on ``e3nn-jax``) keeps the equivariant core
efficient and fully ``jax``/``flax.nnx`` transform-compatible, and lets every
quantum-SciML family (interatomic potentials, equivariant Hamiltonian
prediction) share one implementation.

Conventions:

* An :class:`Irrep` is a pair ``(l, p)`` -- angular-momentum degree
  ``l in {0, 1, 2, ...}`` and parity ``p in {+1, -1}`` (``e``/``o``). Its
  representation space has dimension ``2 l + 1``. Under a rotation ``R`` it
  transforms by the Wigner-D matrix :math:`D^{l}(R)` (times ``p`` under
  inversion).
* The tensor product ``Irrep(l1, p1) * Irrep(l2, p2)`` decomposes as
  :math:`\bigoplus_{l_3 = |l_1 - l_2|}^{l_1 + l_2} (l_3, p_1 p_2)`.
* An :class:`Irreps` is an ordered direct sum of ``mul x Irrep`` blocks.
* An :class:`IrrepsArray` couples a JAX array with its :class:`Irreps` layout.
  It is a pytree whose single child is the array and whose (static, hashable)
  aux-data is the :class:`Irreps`, so it flows through ``jit``/``grad``/``vmap``.
"""

from __future__ import annotations

from collections.abc import Iterator  # noqa: TC003

import jax
from jaxtyping import Array, Float  # noqa: TC002


_PARITY_TO_CHAR = {1: "e", -1: "o"}
_CHAR_TO_PARITY = {"e": 1, "o": -1}


class Irrep:
    """A single irreducible representation ``(l, p)`` of O(3).

    Construct from a string (``Irrep("1o")``) or from explicit degree and parity
    (``Irrep(1, -1)``).
    """

    __slots__ = ("l", "p")

    l: int
    p: int

    def __init__(self, degree: int | str | Irrep, parity: int | None = None) -> None:
        """Build from a string/``Irrep`` (single arg) or explicit ``(l, p)`` ints."""
        if parity is None:
            if isinstance(degree, Irrep):
                resolved_l, resolved_p = degree.l, degree.p
            else:
                text = str(degree).strip()
                try:
                    resolved_p = _CHAR_TO_PARITY[text[-1]]
                    resolved_l = int(text[:-1])
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"Cannot parse irrep {degree!r}; expected e.g. '0e', '1o', '2e'"
                    ) from error
        else:
            resolved_l, resolved_p = int(degree), int(parity)  # type: ignore[arg-type]
        if resolved_l < 0:
            raise ValueError(f"Irrep degree l must be non-negative, got {resolved_l}")
        if resolved_p not in (1, -1):
            raise ValueError(f"Irrep parity p must be +1 or -1, got {resolved_p}")
        self.l = resolved_l
        self.p = resolved_p

    @property
    def dim(self) -> int:
        """Dimension ``2 l + 1`` of the representation space."""
        return 2 * self.l + 1

    def __mul__(self, other: Irrep) -> Iterator[Irrep]:
        """Yield the irreps of the tensor product, ascending in ``l`` (selection rule)."""
        parity = self.p * other.p
        for degree in range(abs(self.l - other.l), self.l + other.l + 1):
            yield Irrep(degree, parity)

    def __eq__(self, other: object) -> bool:
        """Two irreps are equal iff degree and parity match."""
        return isinstance(other, Irrep) and (self.l, self.p) == (other.l, other.p)

    def __lt__(self, other: Irrep) -> bool:
        """Order by degree then even-before-odd parity."""
        return (self.l, -self.p) < (other.l, -other.p)

    def __hash__(self) -> int:
        """Hash by ``(l, p)`` so irreps are usable as dict keys / set members."""
        return hash((self.l, self.p))

    def __repr__(self) -> str:
        """Return the compact ``"1o"``-style representation."""
        return f"{self.l}{_PARITY_TO_CHAR[self.p]}"


def _coerce_blocks(value: Irreps | Irrep | str) -> tuple[tuple[int, Irrep], ...]:
    if isinstance(value, Irreps):
        return value.blocks
    if isinstance(value, Irrep):
        return ((1, value),)
    blocks: list[tuple[int, Irrep]] = []
    for term in str(value).split("+"):
        token = term.strip()
        if not token:
            continue
        if "x" in token:
            mul_text, irrep_text = token.split("x", 1)
            mul = int(mul_text)
        else:
            mul, irrep_text = 1, token
        if mul < 0:
            raise ValueError(f"Irrep multiplicity must be non-negative, got {mul}")
        blocks.append((mul, Irrep(irrep_text)))
    return tuple(blocks)


class Irreps:
    """An ordered direct sum of ``mul x Irrep`` blocks (a steerable feature layout).

    Construct from a string (``Irreps("8x0e + 4x1o")``), an :class:`Irrep`, another
    :class:`Irreps`, or a tuple of ``(mul, Irrep)`` blocks.
    """

    __slots__ = ("blocks",)

    blocks: tuple[tuple[int, Irrep], ...]

    def __init__(self, value: Irreps | Irrep | str | tuple[tuple[int, Irrep], ...]) -> None:
        """Coerce the input into the canonical tuple of ``(mul, Irrep)`` blocks."""
        if isinstance(value, tuple) and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Irrep)
            for item in value
        ):
            self.blocks = value  # type: ignore[assignment]
        else:
            self.blocks = _coerce_blocks(value)  # type: ignore[arg-type]

    @property
    def dim(self) -> int:
        """Total dimension ``sum(mul * (2l + 1))`` of the feature vector."""
        return sum(mul * irrep.dim for mul, irrep in self.blocks)

    @property
    def num_irreps(self) -> int:
        """Total number of irreps (multiplicities summed)."""
        return sum(mul for mul, _ in self.blocks)

    def slices(self) -> list[slice]:
        """Return the contiguous slice into the feature axis for each block."""
        result: list[slice] = []
        start = 0
        for mul, irrep in self.blocks:
            width = mul * irrep.dim
            result.append(slice(start, start + width))
            start += width
        return result

    def __add__(self, other: Irreps | Irrep | str) -> Irreps:
        """Concatenate two layouts."""
        return Irreps(self.blocks + Irreps(other).blocks)

    def __iter__(self) -> Iterator[tuple[int, Irrep]]:
        """Iterate over ``(mul, Irrep)`` blocks."""
        return iter(self.blocks)

    def __eq__(self, other: object) -> bool:
        """Two layouts are equal iff their block sequences match."""
        return isinstance(other, Irreps) and self.blocks == other.blocks

    def __hash__(self) -> int:
        """Hash by the block tuple so :class:`Irreps` is usable as static aux-data."""
        return hash(self.blocks)

    def __repr__(self) -> str:
        """Return the compact ``"8x0e+4x1o"``-style representation."""
        return "+".join(f"{mul}x{irrep}" for mul, irrep in self.blocks)


@jax.tree_util.register_pytree_node_class
class IrrepsArray:
    """A JAX array tagged with its :class:`Irreps` layout (an equivariant feature).

    The array's last axis is the feature axis and must equal ``irreps.dim``. The
    object is a pytree: the array is the single child (traced) and the
    :class:`Irreps` is static aux-data, so equivariant features pass through
    ``jit``/``grad``/``vmap`` unchanged.
    """

    __slots__ = ("array", "irreps")

    irreps: Irreps
    array: Float[Array, "... dim"]

    def __init__(self, irreps: Irreps | Irrep | str, array: Float[Array, "... dim"]) -> None:
        """Build an :class:`IrrepsArray`, validating that the last axis matches ``irreps.dim``."""
        resolved = Irreps(irreps)
        if array.shape[-1] != resolved.dim:
            raise ValueError(
                f"IrrepsArray last dimension {array.shape[-1]} does not match "
                f"irreps {resolved!r} of dimension {resolved.dim}"
            )
        self.irreps = resolved
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        return self.array.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying array."""
        return self.array.ndim

    @property
    def chunks(self) -> list[Float[Array, "... mul dim_l"]]:
        """Split the feature axis into per-block arrays of shape ``(..., mul, 2l+1)``."""
        result: list[Float[Array, "... mul dim_l"]] = []
        leading = self.array.shape[:-1]
        for (mul, irrep), block_slice in zip(self.irreps.blocks, self.irreps.slices(), strict=True):
            block = self.array[..., block_slice]
            result.append(block.reshape(*leading, mul, irrep.dim))
        return result

    def tree_flatten(self) -> tuple[tuple[Float[Array, "... dim"]], Irreps]:
        """Pytree flatten: array is the child, irreps is static aux-data."""
        return (self.array,), self.irreps

    @classmethod
    def tree_unflatten(
        cls, aux_data: Irreps, children: tuple[Float[Array, "... dim"]]
    ) -> IrrepsArray:
        """Pytree unflatten, bypassing dimension re-validation for traced leaves."""
        instance = object.__new__(cls)
        instance.irreps = aux_data
        instance.array = children[0]
        return instance

    def __repr__(self) -> str:
        """Return a short representation with the layout and array shape."""
        return f"IrrepsArray({self.irreps!r}, shape={self.array.shape})"
