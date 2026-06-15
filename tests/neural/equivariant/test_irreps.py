"""Tests for irreps bookkeeping (Irrep / Irreps / IrrepsArray).

Behaviour is specified against the e3nn / e3nn-jax conventions (Geiger & Smidt
2022, arXiv:2207.09453): an irrep is ``(l, p)`` with dimension ``2l + 1``; the
tensor product ``l1 x l2`` yields ``l3`` for ``|l1-l2| <= l3 <= l1+l2`` with
parity ``p1*p2``; an ``Irreps`` is an ordered direct sum of ``mul x irrep``;
``IrrepsArray`` couples a JAX array with its ``Irreps`` layout and is a pytree
(static irreps metadata, traced array) so it survives jit/grad/vmap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.neural.equivariant import Irrep, Irreps, IrrepsArray


class TestIrrep:
    def test_parses_string_and_dimension(self) -> None:
        scalar = Irrep("0e")
        vector = Irrep("1o")
        assert (scalar.l, scalar.p) == (0, 1)
        assert (vector.l, vector.p) == (1, -1)
        assert scalar.dim == 1
        assert vector.dim == 3
        assert Irrep("2e").dim == 5

    def test_tensor_product_selection_rule_and_parity(self) -> None:
        # 1o x 1o -> 0e + 1e + 2e  (parities multiply: (-1)*(-1)=+1)
        out = list(Irrep("1o") * Irrep("1o"))
        assert out == [Irrep("0e"), Irrep("1e"), Irrep("2e")]
        # 1o x 0e -> 1o (odd * even = odd)
        assert list(Irrep("1o") * Irrep("0e")) == [Irrep("1o")]

    def test_is_orderable_and_hashable(self) -> None:
        assert Irrep("0e") < Irrep("1o")
        assert len({Irrep("1o"), Irrep("1o")}) == 1


class TestIrreps:
    def test_parses_and_reports_dimension(self) -> None:
        irreps = Irreps("8x0e + 4x1o")
        assert irreps.dim == 8 * 1 + 4 * 3
        assert irreps.num_irreps == 12  # multiplicities summed

    def test_slices_index_each_block(self) -> None:
        irreps = Irreps("2x0e + 1x1o")
        slices = irreps.slices()
        assert slices[0] == slice(0, 2)  # 2 scalars
        assert slices[1] == slice(2, 5)  # 1 vector (3 dims)

    def test_addition_concatenates(self) -> None:
        assert (Irreps("2x0e") + Irreps("1x1o")).dim == Irreps("2x0e + 1x1o").dim


class TestIrrepsArray:
    def test_construction_validates_dimension(self) -> None:
        x = IrrepsArray("1x0e + 1x1o", jnp.arange(4.0))
        assert x.irreps == Irreps("1x0e + 1x1o")
        assert x.shape == (4,)
        with pytest.raises(ValueError, match="last dimension"):
            IrrepsArray("1x1o", jnp.arange(4.0))  # 1o needs dim 3, not 4

    def test_chunks_split_by_irrep_block(self) -> None:
        x = IrrepsArray("2x0e + 1x1o", jnp.arange(5.0))
        chunks = x.chunks
        assert len(chunks) == 2
        assert chunks[0].shape == (2, 1)  # (mul, 2l+1)
        assert chunks[1].shape == (1, 3)

    def test_is_a_pytree_through_jit_and_vmap(self) -> None:
        irreps = Irreps("1x0e + 1x1o")

        @jax.jit
        def double(z: IrrepsArray) -> IrrepsArray:
            return IrrepsArray(z.irreps, z.array * 2)

        x = IrrepsArray(irreps, jnp.arange(4.0))
        out = double(x)
        assert out.irreps == irreps
        assert jnp.allclose(out.array, x.array * 2)

        batch = IrrepsArray(irreps, jnp.arange(12.0).reshape(3, 4))
        vout = jax.vmap(double)(batch)
        assert vout.array.shape == (3, 4)
        assert vout.irreps == irreps
