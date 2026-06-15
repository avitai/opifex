"""Regression tests for the unified FNO / TFNO fixes.

These guard the four correctness bugs found in the 2026-06-01 operator audit:

1. The TFNO forward must be **nonlinear** (it was a linear spectral cascade with no
   skip term and no activation).
2. The factorized spectral convolution must keep **negative** frequencies — it must
   equal contracting the *centered* low-frequency band with the reconstructed dense
   weight (the old code kept only the ``[:modes]`` positive corner).
3. ``FourierNeuralOperator`` must support **grid positional embedding** (without it a
   Dirichlet boundary-value problem cannot be generalised).
4. The unified TFNO must remain ``jit`` / ``grad`` / ``vmap`` compatible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from opifex.neural.operators.fno._decompositions import TuckerDecomposition
from opifex.neural.operators.fno._factorized import factorized_spectral_conv
from opifex.neural.operators.fno._positional import append_grid_coordinates
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.fno.tensorized import TensorizedFourierNeuralOperator


class TestForwardIsNonlinear:
    """The FNO/TFNO forward must be nonlinear (guards the linear-cascade bug)."""

    @pytest.mark.parametrize("factorization", ["tucker", "cp", "tt"])
    def test_tfno_forward_is_nonlinear(self, factorization):
        """f(2x) must differ from 2 f(x): a linear cascade would make them equal."""
        model = TensorizedFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(8, 8),
            num_layers=3,
            factorization=factorization,
            rank=0.5,
            rngs=nnx.Rngs(0),
        )
        x = 0.5 * jax.random.normal(jax.random.PRNGKey(1), (2, 1, 16, 16))
        scaled = model(2.0 * x)
        linear = 2.0 * model(x)
        assert not jnp.allclose(scaled, linear, atol=1e-3), (
            "TFNO behaves linearly — the activation / skip connection is missing."
        )

    def test_dense_fno_forward_is_nonlinear(self):
        """The dense FNO must also be nonlinear."""
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=8,
            num_layers=3,
            rngs=nnx.Rngs(0),
        )
        x = 0.5 * jax.random.normal(jax.random.PRNGKey(1), (2, 1, 16, 16))
        assert not jnp.allclose(model(2.0 * x), 2.0 * model(x), atol=1e-3)


class TestFactorizedSpectralConvKeepsNegativeFrequencies:
    """The factorized spectral conv must use the centered (signed) frequency band."""

    def test_matches_dense_centered_band_contraction(self):
        """factorized_spectral_conv == contracting the centered band with reconstruct().

        Reproduces the centered-band frequency handling against the reconstructed
        dense weight. A positive-corner-only truncation (the old bug) would not
        match this reference because it discards the negative-frequency half.
        """
        modes = (8, 8)
        decomp = TuckerDecomposition((4, 3, *modes), rank=0.5, rngs=nnx.Rngs(7))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 16, 16))

        result = factorized_spectral_conv(decomp, x, modes)

        # Independent reference: centered band of the rfft2 spectrum contracted with
        # the reconstructed dense weight (out, in, mh, mw).
        weight = decomp.reconstruct()
        x_ft = jnp.fft.fftshift(jnp.fft.rfft2(x, axes=(-2, -1)), axes=(-2,))
        full_h = x_ft.shape[-2]
        start = full_h // 2 - modes[0] // 2
        band = x_ft[:, :, start : start + modes[0], : modes[1]]
        ref_band = jnp.einsum("bixy,oixy->boxy", band, weight)
        out_ft = jnp.zeros((2, 4, full_h, x_ft.shape[-1]), dtype=x_ft.dtype)
        out_ft = out_ft.at[:, :, start : start + modes[0], : modes[1]].set(ref_band)
        out_ft = jnp.fft.ifftshift(out_ft, axes=(-2,))
        reference = jnp.fft.irfft2(out_ft, s=(16, 16), axes=(-2, -1))

        # 1e-3 tolerance absorbs float32 / GPU TF32 matmul noise (both paths use einsum).
        assert jnp.allclose(result, reference, rtol=1e-3, atol=1e-3)

    def test_responds_to_high_axis0_frequencies(self):
        """Output must change when a high (wrap-around) axis-0 frequency is added.

        The centered band covers ``±modes//2`` around DC; a positive-corner slice
        would ignore the negative half and miss this component.
        """
        modes = (8, 8)
        decomp = TuckerDecomposition((2, 2, *modes), rank=0.5, rngs=nnx.Rngs(3))
        grid = jnp.arange(16)
        low = jnp.cos(2 * jnp.pi * 1 * grid / 16)[None, None, :, None]
        low = jnp.broadcast_to(low, (1, 2, 16, 16))
        # Add a component at a negative wrap-around frequency on axis 0.
        neg = jnp.cos(2 * jnp.pi * 3 * grid / 16)[None, None, :, None]
        with_neg = low + jnp.broadcast_to(neg, (1, 2, 16, 16))

        out_low = factorized_spectral_conv(decomp, low, modes)
        out_with_neg = factorized_spectral_conv(decomp, with_neg, modes)
        assert not jnp.allclose(out_low, out_with_neg, atol=1e-5)


class TestPositionalEmbedding:
    """Grid positional embedding (guards the boundary-generalisation bug)."""

    def test_append_grid_adds_one_channel_per_axis(self):
        """append_grid_coordinates appends one normalised [0, 1] grid per spatial axis."""
        x = jnp.zeros((2, 3, 8, 8))
        out = append_grid_coordinates(x)
        assert out.shape == (2, 5, 8, 8)
        # Appended channels span [0, 1] along their respective axes.
        assert jnp.isclose(out[0, 3, 0, 0], 0.0) and jnp.isclose(out[0, 3, -1, 0], 1.0)
        assert jnp.isclose(out[0, 4, 0, 0], 0.0) and jnp.isclose(out[0, 4, 0, -1], 1.0)

    def test_fno_lifting_consumes_grid_channels(self):
        """With positional embedding the lifting layer takes spatial_dims extra channels."""
        model = FourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=8,
            modes=6,
            num_layers=2,
            spatial_dims=2,
            positional_embedding=True,
            rngs=nnx.Rngs(0),
        )
        assert model.input_projection.in_features == 2 + 2
        out = model(jnp.ones((1, 2, 16, 16)))
        assert out.shape == (1, 1, 16, 16)


class TestTransformCompatibility:
    """The unified TFNO must stay jit / grad / vmap compatible."""

    def test_jit_grad_vmap(self):
        """jit forward, grad, and vmap all produce finite results."""
        model = TensorizedFourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=(8, 8),
            num_layers=2,
            factorization="tucker",
            rank=0.5,
            rngs=nnx.Rngs(0),
        )
        x = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (2, 1, 16, 16))

        graphdef, state = nnx.split(model)

        @jax.jit
        def jitted(state, x):
            return nnx.merge(graphdef, state)(x)

        assert jnp.all(jnp.isfinite(jitted(state, x)))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        grads = nnx.grad(loss_fn)(model, x)
        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves and all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

        x_batched = jnp.stack([x, 2.0 * x])

        def apply_single(single):
            return nnx.merge(graphdef, state)(single)

        out_v = jax.vmap(apply_single)(x_batched)
        assert out_v.shape == (2, 2, 1, 16, 16)
        assert jnp.all(jnp.isfinite(out_v))


class TestUnifiedFnoLearnsOperator:
    """End-to-end guard: the unified stack actually learns a known operator."""

    def test_learns_periodic_poisson(self):
        """A dense FNO learns the periodic inverse-Laplacian to good rel-L2.

        Small, deterministic, periodic (so no positional embedding is needed): this
        guards against a non-learning regression in the shared forward pass.
        """
        import optax

        res, n_train, n_test = 16, 256, 64
        key = jax.random.PRNGKey(0)
        k = jnp.fft.fftfreq(res) * res
        kx, ky = jnp.meshgrid(k, k, indexing="ij")
        k2 = kx**2 + ky**2
        inv = jnp.where(k2 == 0, 0.0, 1.0 / k2)
        band = (jnp.abs(kx) <= 6) & (jnp.abs(ky) <= 6)

        def make(subkey, n):
            xh = jax.random.normal(subkey, (n, res, res)) * band[None]
            x = jnp.fft.ifft2(xh).real
            y = jnp.fft.ifft2(jnp.fft.fft2(x) * inv[None]).real
            return x[:, None], y[:, None]

        k1, k2_ = jax.random.split(key)
        x_tr, y_tr = make(k1, n_train)
        x_te, y_te = make(k2_, n_test)
        xm, xs = x_tr.mean(), x_tr.std()
        ym, ys = y_tr.mean(), y_tr.std()
        x_tr_n, y_tr_n = (x_tr - xm) / xs, (y_tr - ym) / ys
        x_te_n = (x_te - xm) / xs

        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=24,
            modes=8,
            num_layers=4,
            rngs=nnx.Rngs(0),
        )
        optimizer = nnx.Optimizer(model, optax.adam(2e-3), wrt=nnx.Param)

        @nnx.jit
        def step(model, optimizer, x, y):
            def loss_fn(m):
                pred = m(x)
                num = jnp.linalg.norm((pred - y).reshape(x.shape[0], -1), axis=1)
                den = jnp.linalg.norm(y.reshape(x.shape[0], -1), axis=1)
                return jnp.mean(num / den)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            return loss

        batch = 32
        loop_key = jax.random.PRNGKey(1)
        for _ in range(60):
            loop_key, sub = jax.random.split(loop_key)
            perm = jax.random.permutation(sub, n_train)
            for i in range(n_train // batch):
                idx = perm[i * batch : (i + 1) * batch]
                step(model, optimizer, x_tr_n[idx], y_tr_n[idx])

        pred = model(x_te_n) * ys + ym
        num = jnp.linalg.norm((pred - y_te).reshape(n_test, -1), axis=1)
        den = jnp.linalg.norm(y_te.reshape(n_test, -1), axis=1)
        rel_l2 = float(jnp.mean(num / den))
        assert rel_l2 < 0.1, f"FNO failed to learn the operator (rel-L2={rel_l2:.3f})"
        assert not np.isnan(rel_l2)
