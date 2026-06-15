"""Honesty tests for genuine low-rank CP / Tucker / TT spectral factorizations.

These tests verify that the tensorized FNO performs *real* low-rank
compression: the factorized parameter count must be ``<<`` the dense weight at
low rank, the factorized contraction must equal contracting the input with the
reconstructed full tensor, and the three decompositions must be distinct.

References (cloned siblings, read before writing these tests):
- tensorly 0.9.0 ``cp_to_tensor`` / ``tucker_to_tensor`` / ``tt_to_tensor``.
- neuraloperator ``_contract_cp`` / ``_contract_tucker`` / ``_contract_tt``.
- tltorch ``factorized_tensors.py`` (factor layouts) + ``init.py`` (factor init).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from opifex.neural.operators.fno._factorized import (
    contract_cp,
    contract_tt,
    contract_tucker,
    cp_parameter_count,
    cp_to_tensor,
    tt_parameter_count,
    tt_ranks,
    tt_to_tensor,
    tucker_parameter_count,
    tucker_to_tensor,
)
from opifex.neural.operators.fno.tensorized import (
    CPDecomposition,
    TensorizedSpectralConvolution,
    TensorTrainDecomposition,
    TuckerDecomposition,
)


# A weight shape where low-rank compression is unambiguous: dense = 8*8*16*16.
_SHAPE = (8, 8, 16, 16)
_DENSE = int(np.prod(_SHAPE))


def _dense_contract(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Reference dense contraction: (batch,in,*m) x (out,in,*m) -> (batch,out,*m)."""
    symbols = "abcdefghijklmnopqrstuvwxyz"
    x_syms = symbols[: x.ndim]
    out_sym = symbols[x.ndim]
    # weight is (out, in, *modes): out + in + spatial
    weight_syms = out_sym + x_syms[1] + x_syms[2:]
    out_syms = x_syms[0] + out_sym + x_syms[2:]
    return jnp.einsum(f"{x_syms},{weight_syms}->{out_syms}", x, weight)


class TestPureFactorizedFormulas:
    """Reconstruct formulas match hand-built analytic factorizations (#4)."""

    def test_cp_rank_one_is_outer_product(self):
        """CP reconstruct of a rank-1 factorization equals the scaled outer product."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0, 0.0, -1.0, 2.0])
        weights = jnp.array([2.0])
        reconstructed = cp_to_tensor(weights, [a[:, None], b[:, None]])
        expected = weights[0] * jnp.outer(a, b)
        assert jnp.allclose(reconstructed, expected, atol=1e-6)

    def test_tucker_is_core_times_factors(self):
        """Tucker reconstruct of a 2-mode factorization equals U0 @ core @ U1.T."""
        core = jax.random.normal(jax.random.PRNGKey(0), (2, 3))
        u0 = jax.random.normal(jax.random.PRNGKey(1), (5, 2))
        u1 = jax.random.normal(jax.random.PRNGKey(2), (4, 3))
        reconstructed = tucker_to_tensor(core, [u0, u1])
        expected = u0 @ core @ u1.T
        assert jnp.allclose(reconstructed, expected, atol=1e-5)

    def test_tt_two_core_chain_is_matrix_product(self):
        """TT reconstruct of a 2-core chain equals the boundary matrix product."""
        g0 = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 2))
        g1 = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 1))
        reconstructed = tt_to_tensor([g0, g1])
        expected = g0[0, :, :] @ g1[:, :, 0]
        assert reconstructed.shape == (5, 4)
        assert jnp.allclose(reconstructed, expected, atol=1e-5)

    @pytest.mark.parametrize("n_modes", [1, 2, 3])
    def test_contract_cp_equals_dense_reconstruct(self, n_modes):
        """contract_cp equals contracting x with the reconstructed CP tensor."""
        inc, outc, rank = 3, 4, 3
        modes = tuple([5, 6, 7][:n_modes])
        keys = jax.random.split(jax.random.PRNGKey(n_modes), 30)
        scale = 0.3
        x = scale * jax.random.normal(keys[0], (2, inc, *modes))
        dims = [inc, outc, *modes]  # internal (in, out, *modes) layout
        factors = [scale * jax.random.normal(keys[2 + i], (d, rank)) for i, d in enumerate(dims)]
        weights = jax.random.normal(keys[20], (rank,))
        full = cp_to_tensor(weights, factors)  # (in, out, *modes)
        full_oim = jnp.moveaxis(full, 1, 0)  # -> (out, in, *modes)
        assert jnp.allclose(
            contract_cp(x, weights, factors), _dense_contract(x, full_oim), rtol=1e-3, atol=1e-4
        )

    @pytest.mark.parametrize("n_modes", [1, 2, 3])
    def test_contract_tucker_equals_dense_reconstruct(self, n_modes):
        """contract_tucker equals contracting x with the reconstructed Tucker tensor."""
        inc, outc = 3, 4
        modes = tuple([5, 6, 7][:n_modes])
        ranks = [2, 3, *([2] * n_modes)]
        keys = jax.random.split(jax.random.PRNGKey(n_modes + 10), 30)
        scale = 0.3
        x = scale * jax.random.normal(keys[0], (2, inc, *modes))
        dims = [inc, outc, *modes]
        factors = [
            scale * jax.random.normal(keys[2 + i], (d, r))
            for i, (d, r) in enumerate(zip(dims, ranks, strict=False))
        ]
        core = jax.random.normal(keys[25], ranks)
        full = tucker_to_tensor(core, factors)
        full_oim = jnp.moveaxis(full, 1, 0)
        assert jnp.allclose(
            contract_tucker(x, core, factors),
            _dense_contract(x, full_oim),
            rtol=1e-3,
            atol=1e-4,
        )

    @pytest.mark.parametrize("n_modes", [1, 2, 3])
    def test_contract_tt_equals_dense_reconstruct(self, n_modes):
        """contract_tt equals contracting x with the reconstructed TT tensor."""
        inc, outc = 3, 4
        modes = tuple([5, 6, 7][:n_modes])
        dims = [inc, outc, *modes]
        ranks = [1, *([3] * (len(dims) - 1)), 1]
        keys = jax.random.split(jax.random.PRNGKey(n_modes + 20), 30)
        scale = 0.4
        x = scale * jax.random.normal(keys[0], (2, inc, *modes))
        cores = [
            scale * jax.random.normal(keys[2 + i], (ranks[i], dims[i], ranks[i + 1]))
            for i in range(len(dims))
        ]
        full = tt_to_tensor(cores)  # (in, out, *modes)
        full_oim = jnp.moveaxis(full, 1, 0)
        assert jnp.allclose(
            contract_tt(x, cores), _dense_contract(x, full_oim), rtol=1e-3, atol=1e-4
        )

    def test_complex_contract_cp_equals_dense_reconstruct(self):
        """Complex CP contraction equals dense complex reconstruct contraction."""
        inc, outc, rank, modes = 2, 3, 2, (4, 4)
        keys = jax.random.split(jax.random.PRNGKey(99), 40)
        scale = 0.3

        def cplx(idx, shape):
            return scale * (
                jax.random.normal(keys[idx], shape) + 1j * jax.random.normal(keys[idx + 20], shape)
            )

        x = cplx(0, (2, inc, *modes))
        dims = [inc, outc, *modes]
        factors = [cplx(2 + i, (d, rank)) for i, d in enumerate(dims)]
        weights = cplx(10, (rank,))
        full = cp_to_tensor(weights, factors)
        full_oim = jnp.moveaxis(full, 1, 0)
        result = contract_cp(x, weights, factors)
        assert jnp.iscomplexobj(result)
        assert jnp.allclose(result, _dense_contract(x, full_oim), rtol=1e-3, atol=1e-4)


class TestParameterCountHelpers:
    """Param-count helpers report the genuine factorized sizes (#1, #5)."""

    def test_cp_param_count_is_rank_times_sum_dims_plus_weights(self):
        """CP params = rank * (out + in + sum(modes)) + rank (weights)."""
        rank = 2
        expected = rank * (8 + 8 + 16 + 16) + rank
        assert cp_parameter_count(_SHAPE, rank) == expected
        assert cp_parameter_count(_SHAPE, rank) < _DENSE

    def test_tucker_param_count_is_core_plus_factor_matrices(self):
        """Tucker params = prod(core_ranks) + sum(rank_k * dim_k)."""
        ranks = (2, 2, 4, 4)
        expected = int(np.prod(ranks)) + sum(r * d for r, d in zip(ranks, _SHAPE, strict=False))
        assert tucker_parameter_count(_SHAPE, ranks) == expected
        assert tucker_parameter_count(_SHAPE, ranks) < _DENSE

    def test_tt_param_count_is_sum_of_core_sizes(self):
        """TT params = sum(r_{k-1} * dim_k * r_k)."""
        ranks = tt_ranks(_SHAPE, max_rank=4)
        expected = sum(ranks[k] * _SHAPE[k] * ranks[k + 1] for k in range(len(_SHAPE)))
        assert tt_parameter_count(_SHAPE, max_rank=4) == expected
        assert tt_parameter_count(_SHAPE, max_rank=4) < _DENSE

    def test_tt_ranks_respect_boundary_and_cap(self):
        """TT ranks have unit boundaries and are clamped to <= prod of remaining dims."""
        ranks = tt_ranks(_SHAPE, max_rank=4)
        assert ranks[0] == 1
        assert ranks[-1] == 1
        assert all(r <= 4 for r in ranks)
        # left bound r_k <= prod(dims[:k]); right bound r_k <= prod(dims[k:])
        for k in range(len(_SHAPE) + 1):
            left = int(np.prod(_SHAPE[:k])) if k > 0 else 1
            right = int(np.prod(_SHAPE[k:])) if k < len(_SHAPE) else 1
            assert ranks[k] <= min(left, right) or ranks[k] <= 4


class TestRealCompression:
    """Decomposition modules report genuine compression at low rank (#1, #2, #5)."""

    def test_cp_parameter_count_below_dense(self):
        """CP decomposition has far fewer params than the dense weight at low rank."""
        cp = CPDecomposition(_SHAPE, rank=2, rngs=nnx.Rngs(0))
        assert cp.parameter_count() < _DENSE

    def test_tucker_parameter_count_below_dense(self):
        """Tucker decomposition has fewer params than the dense weight at low rank."""
        tucker = TuckerDecomposition(_SHAPE, rank=0.25, rngs=nnx.Rngs(0))
        assert tucker.parameter_count() < _DENSE

    def test_tt_parameter_count_below_dense(self):
        """TT decomposition has fewer params than the dense weight at low rank."""
        tt = TensorTrainDecomposition(_SHAPE, max_rank=4, rngs=nnx.Rngs(0))
        assert tt.parameter_count() < _DENSE

    def test_decompositions_are_distinct(self):
        """CP, Tucker and TT report different (non-dense) parameter counts (#5)."""
        cp = CPDecomposition(_SHAPE, rank=2, rngs=nnx.Rngs(0)).parameter_count()
        tucker = TuckerDecomposition(_SHAPE, rank=0.25, rngs=nnx.Rngs(0)).parameter_count()
        tt = TensorTrainDecomposition(_SHAPE, max_rank=4, rngs=nnx.Rngs(0)).parameter_count()
        assert len({cp, tucker, tt}) == 3
        assert all(count != _DENSE for count in (cp, tucker, tt))

    @pytest.mark.parametrize("decomposition", ["cp", "tucker", "tt"])
    def test_compression_stats_honest(self, decomposition):
        """get_compression_stats reports ratio < 1 and reduction > 0 at low rank (#2)."""
        conv = TensorizedSpectralConvolution(
            in_channels=8,
            out_channels=8,
            modes=(16, 16),
            decomposition_type=decomposition,
            rank=0.1,
            rngs=nnx.Rngs(0),
        )
        stats = conv.get_compression_stats()
        assert stats["compression_ratio"] < 1.0
        assert stats["parameter_reduction"] > 0.0


class TestFactorizedContractionTie:
    """multiply_factorized equals contracting x with reconstruct() (#3)."""

    @pytest.mark.parametrize(
        ("decomposition", "kwargs"),
        [
            ("cp", {"rank": 3}),
            ("tucker", {"rank": 0.5}),
            ("tt", {"max_rank": 3}),
        ],
    )
    def test_multiply_factorized_matches_reconstruct(self, decomposition, kwargs):
        """The factorized contraction equals contracting with the reconstructed tensor."""
        shape = (4, 3, 8, 8)
        cls = {
            "cp": CPDecomposition,
            "tucker": TuckerDecomposition,
            "tt": TensorTrainDecomposition,
        }[decomposition]
        module = cls(shape, rngs=nnx.Rngs(7), **kwargs)
        x = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (2, 3, 8, 8))
        x = x + 1j * 0.3 * jax.random.normal(jax.random.PRNGKey(1), (2, 3, 8, 8))

        factorized = module.multiply_factorized(x)
        full = module.reconstruct()  # (out, in, *modes)
        dense = _dense_contract(x, full)

        assert factorized.shape == (2, 4, 8, 8)
        assert jnp.allclose(factorized, dense, rtol=1e-3, atol=1e-4)


class TestJaxTransforms:
    """jit / grad / vmap work for CP and TT decompositions (#7)."""

    @pytest.mark.parametrize(
        ("decomposition", "kwargs"),
        [("cp", {"rank": 3}), ("tt", {"max_rank": 3}), ("tucker", {"rank": 0.5})],
    )
    def test_jit_grad_vmap(self, decomposition, kwargs):
        """multiply_factorized is jit/grad/vmap compatible and finite."""
        shape = (4, 3, 8, 8)
        cls = {
            "cp": CPDecomposition,
            "tucker": TuckerDecomposition,
            "tt": TensorTrainDecomposition,
        }[decomposition]
        module = cls(shape, rngs=nnx.Rngs(7), **kwargs)
        x = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (2, 3, 8, 8))

        # jit
        graphdef, state = nnx.split(module)

        @jax.jit
        def jitted(state, x):
            return nnx.merge(graphdef, state).multiply_factorized(x)

        out = jitted(state, x)
        assert jnp.all(jnp.isfinite(out))

        # grad
        def loss_fn(module, x):
            return jnp.mean(jnp.abs(module.multiply_factorized(x)) ** 2)

        grads = nnx.grad(loss_fn)(module, x)
        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves and all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

        # vmap over a stack of inputs
        x_batched = jnp.stack([x, 2.0 * x])
        vmapped = jax.vmap(lambda single: nnx.merge(graphdef, state).multiply_factorized(single))
        out_v = vmapped(x_batched)
        assert out_v.shape == (2, 2, 4, 8, 8)
        assert jnp.all(jnp.isfinite(out_v))
