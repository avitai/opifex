"""Tests for the U-shaped Neural Operator (UNO).

Mirrors the reference test suite ``neuralop/models/tests/test_uno.py`` and the
discretisation-invariance property that distinguishes a genuine neural operator
from a conv U-Net:

- output-shape correctness (end-to-end scaling product == 1.0),
- forward at a resolution DIFFERENT from another input yields a correctly
  scaled output shape,
- discretisation invariance of the resolution-scaling spectral convolution
  (the property pixel pooling / strided convs lack),
- jit / grad / vmap safety.

Reference: Rahman, Ross, Azizzadenesheli, "U-NO: U-shaped Neural Operators",
TMLR 2022, https://arxiv.org/abs/2204.11127, and the neuraloperator library.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.base import spectral_resample, SpectralConvResize
from opifex.neural.operators.specialized.uno import (
    create_uno,
    UNeuralOperator,
)


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Deterministic RNG bundle for the model tests."""
    return nnx.Rngs(0)


def _smooth_field(size: int, n_channels: int, key: jax.Array) -> jax.Array:
    """Band-limited 2D field sampled at ``size`` so it is resolution-free.

    Built from a fixed set of low Fourier modes evaluated on a ``[0, 1)`` grid,
    so the SAME continuous function can be sampled at any resolution. Shape:
    ``(1, n_channels, size, size)`` (channels-first).
    """
    coords = jnp.linspace(0.0, 1.0, size, endpoint=False)
    yy, xx = jnp.meshgrid(coords, coords, indexing="ij")
    keys = jax.random.split(key, n_channels)
    channels = []
    for ck in keys:
        amp_key, phase_key = jax.random.split(ck)
        amps = jax.random.normal(amp_key, (3, 3))
        phases = jax.random.uniform(phase_key, (3, 3)) * 2 * jnp.pi
        field = jnp.zeros((size, size))
        for kx in range(3):
            for ky in range(3):
                field = field + amps[kx, ky] * jnp.cos(
                    2 * jnp.pi * (kx * xx + ky * yy) + phases[kx, ky]
                )
        channels.append(field)
    return jnp.stack(channels, axis=0)[None]


# =========================================================================
# spectral_resample: Fourier-domain resolution scaling
# =========================================================================


class TestSpectralResample:
    """The pure Fourier-domain resize helper."""

    def test_upsample_shape(self) -> None:
        """Resampling to a larger grid yields the requested shape."""
        x = _smooth_field(16, 2, jax.random.PRNGKey(0))
        y = spectral_resample(x, (32, 32), axes=(-2, -1))
        assert y.shape == (1, 2, 32, 32)

    def test_downsample_shape(self) -> None:
        """Resampling to a smaller grid yields the requested shape."""
        x = _smooth_field(32, 2, jax.random.PRNGKey(0))
        y = spectral_resample(x, (16, 16), axes=(-2, -1))
        assert y.shape == (1, 2, 16, 16)

    def test_identity(self) -> None:
        """Resampling to the same size is (numerically) the identity."""
        x = _smooth_field(24, 3, jax.random.PRNGKey(1))
        y = spectral_resample(x, (24, 24), axes=(-2, -1))
        assert jnp.allclose(x, y, atol=1e-5)

    def test_discretisation_invariance(self) -> None:
        """Up- then down-sampling a band-limited field recovers the original.

        A genuine spectral resize moves data only between Fourier modes, so a
        band-limited signal sampled coarsely, upsampled, and brought back agrees
        with the direct fine sampling on the shared coarse grid.
        """
        x_coarse = _smooth_field(16, 1, jax.random.PRNGKey(2))
        x_fine = spectral_resample(x_coarse, (48, 48), axes=(-2, -1))
        back = spectral_resample(x_fine, (16, 16), axes=(-2, -1))
        assert jnp.allclose(x_coarse, back, atol=1e-4)

    def test_jit(self) -> None:
        """Resample traces under jit (output size is a static tuple)."""
        x = _smooth_field(16, 2, jax.random.PRNGKey(0))
        fn = jax.jit(lambda a: spectral_resample(a, (32, 32), axes=(-2, -1)))
        assert fn(x).shape == (1, 2, 32, 32)


# =========================================================================
# SpectralConvResize: spectral conv with resolution scaling
# =========================================================================


class TestSpectralConvResize:
    """The learnable spectral convolution that changes resolution in Fourier."""

    def test_output_shape_scaled(self, rngs: nnx.Rngs) -> None:
        """A 0.5 scaling halves the spatial extent."""
        conv = SpectralConvResize(in_channels=4, out_channels=6, n_modes=(6, 6), rngs=rngs)
        x = jnp.ones((2, 4, 32, 32))
        y = conv(x, output_scaling_factor=(0.5, 0.5))
        assert y.shape == (2, 6, 16, 16)

    def test_output_shape_default(self, rngs: nnx.Rngs) -> None:
        """No scaling preserves the spatial extent."""
        conv = SpectralConvResize(in_channels=4, out_channels=4, n_modes=(6, 6), rngs=rngs)
        x = jnp.ones((1, 4, 24, 24))
        assert conv(x).shape == (1, 4, 24, 24)

    def test_explicit_output_shape(self, rngs: nnx.Rngs) -> None:
        """An explicit output_shape overrides the scaling factor."""
        conv = SpectralConvResize(in_channels=2, out_channels=2, n_modes=(4, 4), rngs=rngs)
        x = jnp.ones((1, 2, 20, 20))
        assert conv(x, output_shape=(13, 17)).shape == (1, 2, 13, 17)

    def test_discretisation_invariance(self, rngs: nnx.Rngs) -> None:
        """Same weights on a field at N and 2N agree on a common readout grid.

        This is the load-bearing property: applying the spectral conv to a
        band-limited field sampled at N and at 2N, then reading both out at a
        common coarse grid, gives matching results — a strided conv / pooling
        hybrid cannot do this.
        """
        conv = SpectralConvResize(in_channels=1, out_channels=1, n_modes=(6, 6), rngs=rngs)
        key = jax.random.PRNGKey(7)
        x_n = _smooth_field(16, 1, key)
        x_2n = _smooth_field(32, 1, key)  # same continuous field, finer sampling

        common = 16
        out_n = conv(x_n, output_shape=(common, common))
        out_2n = conv(x_2n, output_shape=(common, common))
        rel = jnp.linalg.norm(out_n - out_2n) / jnp.linalg.norm(out_n)
        assert float(rel) < 5e-2

    def test_jit_grad(self, rngs: nnx.Rngs) -> None:
        """Spectral conv is jit- and grad-safe."""
        conv = SpectralConvResize(in_channels=2, out_channels=2, n_modes=(4, 4), rngs=rngs)
        x = jnp.ones((1, 2, 16, 16))

        @nnx.jit
        def loss_fn(m: SpectralConvResize) -> jax.Array:
            return jnp.mean(m(x, output_scaling_factor=(0.5, 0.5)) ** 2)

        grads = nnx.grad(loss_fn)(conv)
        leaves = [g for g in jax.tree.leaves(grads) if hasattr(g, "shape")]
        assert any(jnp.any(g != 0) for g in leaves)

    def test_vmap(self, rngs: nnx.Rngs) -> None:
        """Spectral conv vmaps over a leading example axis."""
        conv = SpectralConvResize(in_channels=2, out_channels=2, n_modes=(4, 4), rngs=rngs)
        batch = jnp.ones((3, 1, 2, 16, 16))
        graphdef, state = nnx.split(conv)

        def apply(x: jax.Array) -> jax.Array:
            return nnx.merge(graphdef, state)(x)

        out = jax.vmap(apply)(batch)
        assert out.shape == (3, 1, 2, 16, 16)


# =========================================================================
# UNeuralOperator: full architecture (mirrors neuralop test_uno.py)
# =========================================================================


def _darcy_uno(rngs: nnx.Rngs, n_layers: int = 5) -> UNeuralOperator:
    """Reference-style 5-layer UNO config with end-to-end scaling 1.0."""
    return UNeuralOperator(
        in_channels=3,
        out_channels=3,
        hidden_channels=16,
        uno_out_channels=[16, 32, 32, 32, 16],
        uno_n_modes=[[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
        uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0], [1.0, 1.0]],
        n_layers=n_layers,
        rngs=rngs,
    )


class TestUNeuralOperator:
    """The rebuilt resolution-invariant UNO."""

    def test_init(self, rngs: nnx.Rngs) -> None:
        """UNO initialises with the reference-style config."""
        model = _darcy_uno(rngs)
        assert model is not None
        assert model.end_to_end_scaling_factor == [1.0, 1.0]

    @pytest.mark.parametrize("size", [(64, 64), (48, 56)])
    def test_forward_shape(self, rngs: nnx.Rngs, size: tuple[int, int]) -> None:
        """End-to-end scaling 1.0 -> output spatial size == input spatial size."""
        model = _darcy_uno(rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, *size))
        y = model(x)
        assert y.shape == (2, 3, *size)

    def test_forward_different_resolution(self, rngs: nnx.Rngs) -> None:
        """A second input at a different resolution scales correctly too."""
        model = _darcy_uno(rngs)
        x1 = jax.random.normal(jax.random.PRNGKey(0), (1, 3, 32, 32))
        x2 = jax.random.normal(jax.random.PRNGKey(1), (1, 3, 64, 64))
        assert model(x1).shape == (1, 3, 32, 32)
        assert model(x2).shape == (1, 3, 64, 64)

    def test_net_scaling(self, rngs: nnx.Rngs) -> None:
        """A net 2x scaling doubles the output spatial size."""
        model = UNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 8, 8],
            uno_n_modes=[[4, 4], [4, 4], [4, 4]],
            uno_scalings=[[2.0, 2.0], [1.0, 1.0], [1.0, 1.0]],
            n_layers=3,
            rngs=rngs,
        )
        x = jnp.ones((1, 1, 16, 16))
        assert model(x).shape == (1, 1, 32, 32)

    def test_finite_output(self, rngs: nnx.Rngs) -> None:
        """Output is finite for random input."""
        model = _darcy_uno(rngs)
        x = jax.random.normal(jax.random.PRNGKey(3), (1, 3, 32, 32))
        assert jnp.all(jnp.isfinite(model(x)))

    def test_discretisation_invariance(self, rngs: nnx.Rngs) -> None:
        """Same weights applied at N and 2N agree on a common readout grid.

        The whole point of UNO over a conv U-Net: feeding a band-limited field
        at 16x16 and at 32x32 and reading both outputs at 16x16 gives matching
        predictions because all resolution changes happen in Fourier space.
        """
        model = _darcy_uno(rngs)
        key = jax.random.PRNGKey(11)
        x_n = _smooth_field(16, 3, key)
        x_2n = _smooth_field(32, 3, key)

        out_n = model(x_n)
        out_2n = model(x_2n)
        out_2n_coarse = spectral_resample(out_2n, (16, 16), axes=(-2, -1))
        rel = jnp.linalg.norm(out_n - out_2n_coarse) / jnp.linalg.norm(out_n)
        assert float(rel) < 0.15

    def test_jit(self, rngs: nnx.Rngs) -> None:
        """UNO runs under jit."""
        model = _darcy_uno(rngs)
        x = jnp.ones((1, 3, 32, 32))

        @nnx.jit
        def forward(m: UNeuralOperator, a: jax.Array) -> jax.Array:
            return m(a)

        y = forward(model, x)
        assert y.shape == (1, 3, 32, 32)
        assert jnp.all(jnp.isfinite(y))

    def test_gradient_flow(self, rngs: nnx.Rngs) -> None:
        """Gradients flow to the spectral weights."""
        model = _darcy_uno(rngs)
        x = jnp.ones((1, 3, 32, 32))

        @nnx.jit
        def loss_fn(m: UNeuralOperator) -> jax.Array:
            return jnp.mean(m(x) ** 2)

        grads = nnx.grad(loss_fn)(model)
        leaves = [g for g in jax.tree.leaves(grads) if hasattr(g, "shape")]
        assert any(jnp.any(g != 0) for g in leaves)


class TestCreateUNO:
    """The create_uno factory."""

    def test_factory_creates_model(self, rngs: nnx.Rngs) -> None:
        """Factory returns a UNeuralOperator."""
        model = create_uno(in_channels=1, out_channels=1, hidden_channels=16, rngs=rngs)
        assert isinstance(model, UNeuralOperator)

    def test_factory_forward(self, rngs: nnx.Rngs) -> None:
        """Factory model maps channels-first input to output channels."""
        model = create_uno(in_channels=2, out_channels=3, hidden_channels=16, rngs=rngs)
        x = jnp.ones((1, 2, 32, 32))
        assert model(x).shape == (1, 3, 32, 32)

    def test_factory_net_scaling_is_identity(self, rngs: nnx.Rngs) -> None:
        """The default factory config has end-to-end scaling 1.0."""
        model = create_uno(in_channels=1, out_channels=1, hidden_channels=16, rngs=rngs)
        assert model.end_to_end_scaling_factor == [1.0, 1.0]
