"""Tests for the symmetric tensor-product (U-matrix) basis builder.

Validates the invariants of the symmetric coupling basis against the e3nn-jax
``reduced_symmetric_tensor_product_basis`` reference: orthonormal independent
paths, the correct number of symmetric paths per output irrep for known small
cases, invariance under permuting the input axes, and rotational equivariance of
the resulting contraction.
"""

from __future__ import annotations

import itertools

import numpy as np

from opifex.geometry.algebra import SO3Group
from opifex.geometry.algebra.wigner import wigner_d
from opifex.neural.equivariant._reduced_tensor_product import (
    gram_schmidt,
    reduced_symmetric_tensor_product_basis,
)
from opifex.neural.equivariant.irreps import Irrep


class TestGramSchmidt:
    def test_orthonormal_rows(self) -> None:
        rows = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        q = gram_schmidt(rows)
        np.testing.assert_allclose(q @ q.T, np.eye(q.shape[0]), atol=1e-9)

    def test_drops_dependent_rows(self) -> None:
        rows = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])  # rank 2
        assert gram_schmidt(rows).shape[0] == 2


class TestSymmetricBasis:
    def test_degree_one_is_identity_per_irrep(self) -> None:
        """Degree 1 returns one orthonormal path per input multiplicity."""
        basis = reduced_symmetric_tensor_product_basis("2x0e + 1x1o", degree=1)
        # d = 2*1 + 1*3 = 5; shape is (num_paths, d, ir.dim); 0e has mul 2 -> 2 paths.
        assert basis[Irrep("0e")].shape == (2, 5, 1)
        assert basis[Irrep("1o")].shape == (1, 5, 3)

    def test_degree_two_path_counts(self) -> None:
        """Symmetric square of (0e + 1o): 0e x2, 1o x1, 2e x1; no antisymmetric 1e."""
        basis = reduced_symmetric_tensor_product_basis("1x0e + 1x1o", degree=2)
        counts = {ir: u.shape[0] for ir, u in basis.items()}
        assert counts.get(Irrep("0e")) == 2  # 0e*0e and sym(1o*1o)
        assert counts.get(Irrep("1o")) == 1  # sym(0e*1o)
        assert counts.get(Irrep("2e")) == 1  # sym(1o*1o)
        assert Irrep("1e") not in counts  # antisymmetric 1o*1o is dropped

    def test_keep_ir_filters_outputs(self) -> None:
        basis = reduced_symmetric_tensor_product_basis("1x0e + 1x1o", degree=2, keep_ir="0e")
        assert set(basis) == {Irrep("0e")}

    def test_paths_are_orthonormal(self) -> None:
        basis = reduced_symmetric_tensor_product_basis("1x0e + 1x1o", degree=3)
        for u in basis.values():
            flat = u.reshape(u.shape[0], -1)
            np.testing.assert_allclose(flat @ flat.T, np.eye(flat.shape[0]), atol=1e-8)

    def test_invariant_under_input_axis_permutation(self) -> None:
        """U is symmetric: permuting its degree input axes leaves it unchanged."""
        degree = 3
        basis = reduced_symmetric_tensor_product_basis("1x0e + 1x1o", degree=degree)
        for u in basis.values():
            for perm in itertools.permutations(range(degree)):
                # axes: (path,) + input axes (1..degree) + (ir.dim,)
                axes = (0, *(1 + p for p in perm), degree + 1)
                np.testing.assert_allclose(np.transpose(u, axes), u, atol=1e-8)

    def test_contraction_is_rotation_equivariant(self) -> None:
        """Contracting U with degree copies of v transforms by the output Wigner-D.

        Uses irreps = 1x1o (d=3): the degree-2 contraction sends v in R^3 to the
        symmetric outputs; under a rotation R the 0e output is invariant and the
        2e output transforms by D^2(R).
        """
        import jax
        import jax.numpy as jnp

        degree = 2
        basis = reduced_symmetric_tensor_product_basis("1x1o", degree=degree)
        rotation = jnp.asarray(SO3Group().random_element(jax.random.PRNGKey(3)))
        d1 = np.asarray(wigner_d(1, rotation))  # input (l=1) transforms by D^1(R)
        vector = np.array([0.3, -0.5, 0.8])
        rotated = d1 @ vector

        for ir, u in basis.items():
            # out[path, k] = sum_{i,j} U[path, i, j, k] v_i v_j
            out = np.einsum("pijk,i,j->pk", u, vector, vector)
            out_rot = np.einsum("pijk,i,j->pk", u, rotated, rotated)
            d_out = np.asarray(wigner_d(ir.l, rotation))
            # ``wigner_d`` follows the active JAX precision: under the default
            # CI config (JAX_ENABLE_X64=0) it is float32, so the equivariance
            # identity holds only to float32 precision. Use a float32-appropriate
            # tolerance so the check is deterministic regardless of whether x64
            # happens to be enabled.
            np.testing.assert_allclose(out_rot, out @ d_out.T, atol=1e-5)
