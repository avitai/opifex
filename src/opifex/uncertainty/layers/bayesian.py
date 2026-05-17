"""Phase 2 Task 2.1 — shared :class:`BayesianLinear` NNX module.

Single canonical implementation of the variational diagonal-Gaussian dense
layer. Replaces the two parallel implementations in
``src/opifex/neural/bayesian/layers.py`` and
``src/opifex/neural/operators/specialized/uqno.py``; Phase 2 Task 2.3 and
Phase 3 Task 3.4 migrate those call sites to import from here.

RNG safety (GUIDE_ALIGNMENT items 4, 4a, 5, 7, 9):

* Constructor ``rngs`` initializes parameters only.
* Stochastic sampling routes every call through
  ``artifex.generative_models.core.rng.extract_rng_key`` — caller-owned
  ``nnx.Rngs`` (advancing the ``"posterior"`` stream) or an explicit
  ``jax.Array`` key. No hidden ``jax.random.PRNGKey(...)`` seeds in the
  production path.

KL math: delegated to
:func:`opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl` which itself
delegates to Artifex ``gaussian_kl_divergence`` for the N(0,1) case
(GUIDE_ALIGNMENT item 8). One formula, owned in one place.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx

from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl


# Named-stream resolution order, matching the Avitai canonical convention.
_POSTERIOR_STREAMS: tuple[str, ...] = ("posterior", "sample", "default")


class BayesianLinear(nnx.Module):
    """Variational diagonal-Gaussian dense layer.

    Weight and bias each carry a ``(mean, log-variance)`` posterior; sampling
    uses the reparameterization trick. Pre-defined ``__call__`` flags:

    * ``training=False`` — return the posterior-mean prediction (no sampling).
    * ``sample=False`` — same; equivalent to deterministic mode.
    * ``training=True, sample=True`` — sample weights / bias from the
      diagonal-Gaussian posterior; ``rngs`` MUST be provided by the caller.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize variational parameters; ``rngs`` initializes parameters only."""
        if prior_std <= 0.0:
            raise ValueError(f"prior_std must be > 0; got {prior_std!r}.")

        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.weight_mean = nnx.Param(
            nnx.initializers.xavier_normal()(rngs.params(), (out_features, in_features))
        )
        self.weight_logvar = nnx.Param(jnp.full((out_features, in_features), -10.0))
        self.bias_mean = nnx.Param(jnp.zeros(out_features))
        self.bias_logvar = nnx.Param(jnp.full(out_features, -10.0))

    def __call__(
        self,
        x: jax.Array,
        *,
        training: bool = True,
        sample: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass; samples weights when ``training and sample``."""
        if training and sample:
            if rngs is None:
                raise ValueError(
                    "BayesianLinear sampling requires caller-owned `rngs` "
                    "(nnx.Rngs with a posterior/sample/default stream, or a "
                    "jax.Array key)."
                )
            key = extract_rng_key(
                rngs,
                streams=_POSTERIOR_STREAMS,
                context="BayesianLinear sampling",
            )
            weight_key, bias_key = jax.random.split(key)
            weight = _reparameterize(self.weight_mean[...], self.weight_logvar[...], weight_key)
            bias = _reparameterize(self.bias_mean[...], self.bias_logvar[...], bias_key)
        else:
            weight = self.weight_mean[...]
            bias = self.bias_mean[...]

        return jnp.dot(x, weight.T) + bias

    def kl_divergence(self) -> jax.Array:
        """Total KL divergence (weights + bias) under the layer's diagonal Gaussian prior."""
        weight_kl = diagonal_gaussian_kl(
            self.weight_mean[...],
            self.weight_logvar[...],
            prior_mean=0.0,
            prior_std=self.prior_std,
        )
        bias_kl = diagonal_gaussian_kl(
            self.bias_mean[...],
            self.bias_logvar[...],
            prior_mean=0.0,
            prior_std=self.prior_std,
        )
        return weight_kl + bias_kl


def _reparameterize(mean: jax.Array, logvar: jax.Array, key: jax.Array) -> jax.Array:
    """Reparameterization-trick sample from ``N(mean, exp(logvar))``."""
    std = jnp.exp(0.5 * logvar)
    noise = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    return mean + std * noise


class BayesianSpectralConvolution(nnx.Module):
    """Variational Fourier-spectral convolution with complex Gaussian weights.

    Implements the canonical Zongyi Li Fourier Neural Operator spectral block
    (Li et al. 2021, ``arXiv:2010.08895``; reference implementation:
    ``../deeponet-fno/src/darcy_rectangular_pwc/fourier_2d.py:SpectralConv2d``)
    with a variational diagonal-Gaussian posterior over each complex weight.

    The trainable Fourier weights split into real and imaginary parts; each
    part carries a diagonal-Gaussian posterior ``(mean, log-variance)``.
    Sampling uses the reparameterization trick and combines the parts into a
    complex weight tensor for the spectral convolution.

    **Mode handling.** For ``jnp.fft.rfft``-style transforms:

    * 1D: only the real-FFT axis exists, low-frequency modes are
      ``[:modes[0]]``. One weight tensor of shape ``(out, in, modes[0])``.
    * 2D: ``jnp.fft.rfftn(x, axes=(-2, -1))`` is a full FFT on the H axis and
      a real FFT on the W axis. The H axis therefore carries BOTH positive
      ``[:modes[0]]`` and negative ``[-modes[0]:]`` low-frequency modes; the
      W axis carries only ``[:modes[1] // 2 + 1]``. Following Li, TWO weight
      tensors of shape ``(out, in, modes[0], modes[1] // 2 + 1)`` are used —
      one for each H-frequency band — so the spectral kernel captures the
      full low-frequency response rather than a single quadrant.

    Output spatial shape matches the input; only ``in_channels`` becomes
    ``out_channels``. Aleatoric / epistemic uncertainty extraction is the
    caller's responsibility — this layer returns only the convolved tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, ...],
        prior_std: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize complex weight posteriors; ``rngs`` initializes parameters only."""
        if prior_std <= 0.0:
            raise ValueError(f"prior_std must be > 0; got {prior_std!r}.")
        if len(modes) not in (1, 2):
            raise ValueError(
                f"BayesianSpectralConvolution supports 1D or 2D modes; got "
                f"{len(modes)}D ({modes!r}). Higher-rank spectral kernels are out of scope."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.prior_std = prior_std

        if len(modes) == 1:
            base_shape: tuple[int, ...] = (out_channels, in_channels, modes[0])
        else:
            base_shape = (out_channels, in_channels, modes[0], modes[1] // 2 + 1)

        init = nnx.initializers.normal(stddev=0.1)
        # Positive H-axis (or only-axis in 1D) weight tensor.
        self.weight_mean = nnx.Param(init(rngs.params(), base_shape))
        self.weight_logvar = nnx.Param(jnp.full(base_shape, -3.0))
        self.weight_imag_mean = nnx.Param(init(rngs.params(), base_shape))
        self.weight_imag_logvar = nnx.Param(jnp.full(base_shape, -3.0))

        # 2D: also need a second weight tensor for the negative H-axis modes
        # (canonical Li FNO; see class docstring).
        if len(modes) == 2:
            self.weight_neg_h_mean = nnx.Param(init(rngs.params(), base_shape))
            self.weight_neg_h_logvar = nnx.Param(jnp.full(base_shape, -3.0))
            self.weight_neg_h_imag_mean = nnx.Param(init(rngs.params(), base_shape))
            self.weight_neg_h_imag_logvar = nnx.Param(jnp.full(base_shape, -3.0))

    def __call__(
        self,
        x: jax.Array,
        *,
        training: bool = True,
        sample: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
    ) -> jax.Array:
        """Apply Bayesian Fourier-spectral convolution to ``x``."""
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, got {x.shape[1]} from "
                f"input shape {x.shape!r}."
            )

        if training and sample:
            if rngs is None:
                raise ValueError(
                    "BayesianSpectralConvolution sampling requires caller-owned "
                    "`rngs` (nnx.Rngs with a posterior/sample/default stream, or a "
                    "jax.Array key)."
                )
            key = extract_rng_key(
                rngs,
                streams=_POSTERIOR_STREAMS,
                context="BayesianSpectralConvolution sampling",
            )
            weights = self._sample_weights(key)
        else:
            weights = self._mean_weights()

        return _spectral_convolve(x, weights, modes=self.modes)

    def _mean_weights(self) -> tuple[jax.Array, ...]:
        """Return posterior-mean complex weights (no sampling)."""
        positive = self.weight_mean[...] + 1j * self.weight_imag_mean[...]
        if len(self.modes) == 1:
            return (positive,)
        negative = self.weight_neg_h_mean[...] + 1j * self.weight_neg_h_imag_mean[...]
        return (positive, negative)

    def _sample_weights(self, key: jax.Array) -> tuple[jax.Array, ...]:
        """Sample complex weights via the reparameterization trick."""
        if len(self.modes) == 1:
            real_key, imag_key = jax.random.split(key)
            real = _reparameterize(self.weight_mean[...], self.weight_logvar[...], real_key)
            imag = _reparameterize(
                self.weight_imag_mean[...], self.weight_imag_logvar[...], imag_key
            )
            return (real + 1j * imag,)

        # 2D: four independent samples (positive/negative H × real/imag).
        keys = jax.random.split(key, 4)
        pos_real = _reparameterize(self.weight_mean[...], self.weight_logvar[...], keys[0])
        pos_imag = _reparameterize(
            self.weight_imag_mean[...], self.weight_imag_logvar[...], keys[1]
        )
        neg_real = _reparameterize(
            self.weight_neg_h_mean[...], self.weight_neg_h_logvar[...], keys[2]
        )
        neg_imag = _reparameterize(
            self.weight_neg_h_imag_mean[...], self.weight_neg_h_imag_logvar[...], keys[3]
        )
        return (pos_real + 1j * pos_imag, neg_real + 1j * neg_imag)

    def kl_divergence(self) -> jax.Array:
        """Total KL divergence summed over every (real/imag, positive/negative-H) weight posterior."""
        total = diagonal_gaussian_kl(
            self.weight_mean[...],
            self.weight_logvar[...],
            prior_mean=0.0,
            prior_std=self.prior_std,
        ) + diagonal_gaussian_kl(
            self.weight_imag_mean[...],
            self.weight_imag_logvar[...],
            prior_mean=0.0,
            prior_std=self.prior_std,
        )
        if len(self.modes) == 2:
            total = (
                total
                + diagonal_gaussian_kl(
                    self.weight_neg_h_mean[...],
                    self.weight_neg_h_logvar[...],
                    prior_mean=0.0,
                    prior_std=self.prior_std,
                )
                + diagonal_gaussian_kl(
                    self.weight_neg_h_imag_mean[...],
                    self.weight_neg_h_imag_logvar[...],
                    prior_mean=0.0,
                    prior_std=self.prior_std,
                )
            )
        return total


def _spectral_convolve(
    x: jax.Array, weights: tuple[jax.Array, ...], *, modes: tuple[int, ...]
) -> jax.Array:
    """Pure-JAX FNO-style spectral convolution with complex Bayesian-sampled weights.

    Implements the Li 2D FNO formulation: positive- and negative-H-axis bands
    are convolved with independent weight tensors. 1D uses the single-band
    formulation (no negative-frequency band in 1D rfft).
    """
    if len(modes) == 2:
        weights_pos_h, weights_neg_h = weights
        modes_h, modes_w = modes[0], modes[1]
        batch_size, _in_channels, height, width = x.shape
        out_channels = weights_pos_h.shape[0]

        x_ft = jnp.fft.rfftn(x, axes=(-2, -1))
        out_ft = jnp.zeros((batch_size, out_channels, height, width // 2 + 1), dtype=x_ft.dtype)
        # Positive H-axis low-frequency modes.
        out_ft = out_ft.at[:, :, :modes_h, : modes_w // 2 + 1].set(
            jnp.einsum(
                "bcij,ocij->boij",
                x_ft[:, :, :modes_h, : modes_w // 2 + 1],
                weights_pos_h,
            )
        )
        # Negative H-axis low-frequency modes (canonical Li FNO).
        out_ft = out_ft.at[:, :, -modes_h:, : modes_w // 2 + 1].set(
            jnp.einsum(
                "bcij,ocij->boij",
                x_ft[:, :, -modes_h:, : modes_w // 2 + 1],
                weights_neg_h,
            )
        )
        return jnp.real(jnp.fft.irfftn(out_ft, s=(height, width), axes=(-2, -1)))

    (weights_1d,) = weights
    modes_m = modes[0]
    batch_size, _in_channels, length = x.shape
    out_channels = weights_1d.shape[0]

    x_ft = jnp.fft.rfft(x, axis=-1)
    out_ft = jnp.zeros((batch_size, out_channels, length // 2 + 1), dtype=x_ft.dtype)
    out_ft = out_ft.at[:, :, :modes_m].set(
        jnp.einsum("bci,oci->boi", x_ft[:, :, :modes_m], weights_1d)
    )
    return jnp.real(jnp.fft.irfft(out_ft, n=length, axis=-1))


__all__ = ["BayesianLinear", "BayesianSpectralConvolution"]
