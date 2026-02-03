# FILE PLACEMENT: opifex/neural/operators/specialized/uqno.py
#
# FIXED Uncertainty Quantification Neural Operator Implementation
# Fixes shape mismatches in spectral convolutions and skip connections
#
# This file should REPLACE: opifex/neural/operators/specialized/uqno.py

"""
Uncertainty Quantification Neural Operator (UQNO)

Advanced neural operator with built-in uncertainty quantification for
safety-critical applications. Provides both epistemic and aleatoric
uncertainty estimates using Bayesian neural networks and ensemble methods.

Key Features:
- Bayesian spectral convolutions with weight uncertainty
- Epistemic uncertainty through weight distributions
- Aleatoric uncertainty through learned noise parameters
- Monte Carlo sampling for uncertainty propagation
- Safety-critical application support
"""

import logging
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


# Set up logger for this module
logger = logging.getLogger(__name__)


class BayesianLinear(nnx.Module):
    """
    Bayesian linear layer with weight uncertainty.

    Implements variational Bayesian linear layer where weights are
    distributions rather than point estimates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize Bayesian linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_std: Standard deviation of weight prior
            rngs: Random number generator state
        """
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Weight mean parameters  handling
        self.weight_mean = nnx.Param(
            nnx.initializers.normal(stddev=0.1)(
                rngs.params(), (out_features, in_features)
            )
        )
        self.weight_logvar = nnx.Param(
            nnx.initializers.constant(-3.0)(rngs.params(), (out_features, in_features))
        )

        # Bias parameters  handling
        self.bias_mean = nnx.Param(jnp.zeros((out_features,)))
        self.bias_logvar = nnx.Param(
            nnx.initializers.constant(-3.0)(rngs.params(), (out_features,))
        )

    def __call__(
        self, x: jax.Array, training: bool = True, sample: bool = True
    ) -> jax.Array:
        """
        Forward pass with Bayesian sampling.

        Args:
            x: Input tensor
            training: Whether in training mode
            sample: Whether to sample weights

        Returns:
            Output tensor with uncertainty
        """

        if training and sample:
            # Sample weights from posterior
            weight_std = jnp.exp(0.5 * self.weight_logvar.value)
            bias_std = jnp.exp(0.5 * self.bias_logvar.value)

            eps_w = jax.random.normal(
                jax.random.PRNGKey(0),
                self.weight_mean.value.shape,
            )
            eps_b = jax.random.normal(
                jax.random.PRNGKey(1),
                self.bias_mean.value.shape,
            )
            weight = self.weight_mean.value + weight_std * eps_w
            bias = self.bias_mean.value + bias_std * eps_b
        else:
            # Use mean values
            weight = self.weight_mean.value
            bias = self.bias_mean.value

        return x @ weight.T + bias

    def kl_divergence(self) -> jax.Array:
        """
        Compute KL divergence between posterior and prior.

        Returns:
            KL divergence scalar
        """
        # KL divergence for Gaussian distributions
        weight_var = jnp.exp(self.weight_logvar.value)
        bias_var = jnp.exp(self.bias_logvar.value)

        weight_kl = 0.5 * jnp.sum(
            (self.weight_mean.value**2 + weight_var) / (self.prior_std**2)
            - 1
            - self.weight_logvar.value
            + 2 * jnp.log(self.prior_std)
        )

        bias_kl = 0.5 * jnp.sum(
            (self.bias_mean.value**2 + bias_var) / (self.prior_std**2)
            - 1
            - self.bias_logvar.value
            + 2 * jnp.log(self.prior_std)
        )

        return weight_kl + bias_kl


class BayesianSpectralConvolution(nnx.Module):
    """
    Bayesian spectral convolution with proper shape handling.

    Implements spectral convolution in Fourier domain with Bayesian weights
    for uncertainty quantification.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        prior_std: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize Bayesian spectral convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Fourier modes for each spatial dimension
            prior_std: Standard deviation of weight prior
            rngs: Random number generator state
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.prior_std = prior_std

        # Calculate weight dimensions - shape for einsum "bc...ij,oc...ij->bo...ij"
        # FIXED: Proper weight shape for 2D spectral convolution
        if len(modes) == 1:
            weight_shape: tuple[int, ...] = (out_channels, in_channels, modes[0])
        elif len(modes) == 2:
            weight_shape = (out_channels, in_channels, modes[0], modes[1] // 2 + 1)
        else:
            raise ValueError(f"Unsupported number of modes: {len(modes)}")

        # Weight mean parameters  handling
        self.weight_mean = nnx.Param(
            nnx.initializers.normal(stddev=0.1)(rngs.params(), weight_shape)
        )
        self.weight_logvar = nnx.Param(
            nnx.initializers.constant(-3.0)(rngs.params(), weight_shape)
        )

        # Separate imaginary parts for complex weights  handling
        self.weight_imag_mean = nnx.Param(
            nnx.initializers.normal(stddev=0.1)(rngs.params(), weight_shape)
        )
        self.weight_imag_logvar = nnx.Param(
            nnx.initializers.constant(-3.0)(rngs.params(), weight_shape)
        )

    def _sample_weights(
        self, training: bool, sample: bool
    ) -> tuple[jax.Array, jax.Array]:
        """Sample or use mean weights for spectral convolution."""
        if training and sample:
            real_std = jnp.exp(0.5 * self.weight_logvar.value)
            imag_std = jnp.exp(0.5 * self.weight_imag_logvar.value)

            eps_real = jax.random.normal(
                jax.random.PRNGKey(2),
                self.weight_mean.value.shape,
            )
            eps_imag = jax.random.normal(
                jax.random.PRNGKey(3),
                self.weight_imag_mean.value.shape,
            )
            weight_real = self.weight_mean.value + real_std * eps_real
            weight_imag = self.weight_imag_mean.value + imag_std * eps_imag
            weights = weight_real + 1j * weight_imag

            # Compute aleatoric uncertainty for this sample
            aleatoric_var = real_std**2 + imag_std**2
            aleatoric_uncertainty = jnp.sqrt(jnp.mean(aleatoric_var))
        else:
            weights = self.weight_mean.value + 1j * self.weight_imag_mean.value
            aleatoric_uncertainty = jnp.zeros(())

        return weights, aleatoric_uncertainty

    def _perform_spectral_convolution(
        self, x: jax.Array, weights: jax.Array, batch_size: int, height: int, width: int
    ) -> jax.Array:
        """
        Perform spectral convolution with FFT operations.

        Args:
            x: Input tensor in spatial domain
            weights: Complex weights for spectral convolution
            batch_size: Batch size
            height: Spatial height dimension
            width: Spatial width dimension

        Returns:
            Output tensor after spectral convolution
        """
        # Forward FFT to frequency domain
        x_ft = jnp.fft.rfftn(x, axes=(-2, -1))

        # Crop to specified modes based on dimensionality
        if len(self.modes) == 2:
            modes_h, modes_w = self.modes[0], self.modes[1]
            x_ft_cropped = x_ft[:, :, :modes_h, : modes_w // 2 + 1]
        else:
            modes_h = self.modes[0]
            x_ft_cropped = x_ft[:, :, :modes_h]

        # Spectral convolution via einsum
        out_ft = jnp.einsum("bc...ij,oc...ij->bo...ij", x_ft_cropped, weights)

        # Pad and inverse FFT back to spatial domain
        if len(self.modes) == 2:
            # 2D case: pad and use irfftn
            out_ft_padded = jnp.zeros(
                (batch_size, self.out_channels, height, width // 2 + 1)
            )
            out_ft_padded = out_ft_padded.at[:, :, :modes_h, : modes_w // 2 + 1].set(
                out_ft
            )
            return jnp.fft.irfftn(out_ft_padded, s=(height, width), axes=(-2, -1))
        # 1D case: pad and use irfft
        out_ft_padded = jnp.zeros((batch_size, self.out_channels, height))
        out_ft_padded = out_ft_padded.at[:, :, :modes_h].set(out_ft)
        return jnp.fft.irfft(out_ft_padded, n=width, axis=-1)

    def __call__(
        self, x: jax.Array, training: bool = True, sample: bool = True
    ) -> tuple[jax.Array, jax.Array]:
        """
        Apply Bayesian spectral convolution with proper shape handling.

        Args:
            x: Input tensor (batch, in_channels, height, width)
            training: Whether in training mode
            sample: Whether to sample weights

        Returns:
            Tuple of (output_mean, aleatoric_uncertainty)
        """
        batch_size, in_channels, height, width = x.shape

        # Validate input channels
        if in_channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {in_channels}"
            )

        # Sample weights and get aleatoric uncertainty
        weights, aleatoric_uncertainty = self._sample_weights(training, sample)

        # Perform spectral convolution
        output_mean = self._perform_spectral_convolution(
            x, weights, batch_size, height, width
        )
        output_mean = jnp.real(output_mean)

        # Ensure aleatoric uncertainty has correct shape  handling
        aleatoric_uncertainty = jnp.broadcast_to(
            aleatoric_uncertainty, output_mean.shape
        )

        return output_mean, aleatoric_uncertainty

    def kl_divergence(self) -> jax.Array:
        """Compute KL divergence for weight distributions."""
        # KL for real part
        real_var = jnp.exp(self.weight_logvar.value)
        kl_real = 0.5 * jnp.sum(
            (self.weight_mean.value**2 + real_var) / (self.prior_std**2)
            - 1
            - self.weight_logvar.value
            + 2 * jnp.log(self.prior_std)
        )

        # KL for imaginary part
        imag_var = jnp.exp(self.weight_imag_logvar.value)
        kl_imag = 0.5 * jnp.sum(
            (self.weight_imag_mean.value**2 + imag_var) / (self.prior_std**2)
            - 1
            - self.weight_imag_logvar.value
            + 2 * jnp.log(self.prior_std)
        )

        return kl_real + kl_imag


class UQNOLayer(nnx.Module):
    """
    UQNO layer with proper shape handling for skip connections.

    Combines Bayesian spectral convolution with local operations
    and proper channel dimension handling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        use_skip_connection: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize UQNO layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Fourier modes for spectral convolution
            use_skip_connection: Whether to use skip connections
            rngs: Random number generator state
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_skip_connection = use_skip_connection
        # Bayesian spectral convolution
        self.spectral_conv = BayesianSpectralConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            rngs=rngs,
        )

        # Local convolution for comparison
        self.local_conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        # FIXED: Channel projection for skip connection if needed
        if use_skip_connection and in_channels != out_channels:
            self.channel_proj = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                padding="SAME",
                rngs=rngs,
            )
        else:
            self.channel_proj = None  # type: ignore[assignment]

    def __call__(
        self, x: jax.Array, training: bool = True
    ) -> tuple[jax.Array, jax.Array]:
        """
        Forward pass with proper shape handling.

        Args:
            x: Input tensor (batch, in_channels, height, width)
            training: Whether in training mode

        Returns:
            Tuple of (output, aleatoric_uncertainty)
        """

        # Apply spectral convolution
        x_spec, aleatoric_std = self.spectral_conv(x, training=training)

        # Convert to channels-last for Conv
        x_channels_last = x.transpose(0, 2, 3, 1)
        conv_out = self.local_conv(x_channels_last)
        # Convert back to channels-first
        x_local = conv_out.transpose(0, 3, 1, 2)

        # Handle skip connection with proper channel matching
        if self.use_skip_connection:
            if self.channel_proj is not None:
                x_channels_last = x.transpose(0, 2, 3, 1)
                proj_out = self.channel_proj(x_channels_last)
                x_skip = proj_out.transpose(0, 3, 1, 2)
            else:
                x_skip = x
        else:
            x_skip = jnp.zeros_like(x_spec)

        # FIXED: Combine all paths
        output = x_spec + x_local + x_skip

        return nnx.gelu(output), aleatoric_std

    def kl_divergence(self) -> jax.Array:
        """Get KL divergence from spectral convolution."""
        return self.spectral_conv.kl_divergence()


class UncertaintyQuantificationNeuralOperator(nnx.Module):
    """
    Complete Uncertainty Quantification Neural Operator.

    Neural operator with built-in uncertainty quantification for
    safety-critical applications and robust predictions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        modes: Sequence[int] = (16, 16),
        num_layers: int = 4,
        use_epistemic: bool = True,
        use_aleatoric: bool = True,
        ensemble_size: int = 10,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize UQNO with uncertainty quantification.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Hidden layer width
            modes: Fourier modes for spectral convolution
            num_layers: Number of UQNO layers
            use_epistemic: Whether to use epistemic uncertainty
            use_aleatoric: Whether to use aleatoric uncertainty
            ensemble_size: Size for Monte Carlo sampling
            rngs: Random number generator state
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_epistemic = use_epistemic
        self.use_aleatoric = use_aleatoric
        self.ensemble_size = ensemble_size
        self.rngs = rngs  # Store rngs for later use
        # Input lifting
        self.lifting = BayesianLinear(in_channels, hidden_channels, rngs=rngs)

        # UQNO layers
        uqno_layers_temp = []
        for _ in range(num_layers):
            layer_in_channels = hidden_channels
            layer_out_channels = hidden_channels

            layer = UQNOLayer(
                in_channels=layer_in_channels,
                out_channels=layer_out_channels,
                modes=modes,
                rngs=rngs,
            )
            uqno_layers_temp.append(layer)
            self.uqno_layers = nnx.List(uqno_layers_temp)

        # Output projection
        self.projection = BayesianLinear(hidden_channels, out_channels, rngs=rngs)

        # Epistemic uncertainty head (if enabled)
        if use_epistemic:
            # The epistemic head will be initialized during the first forward pass
            # when we know the actual input dimensions
            self.epistemic_head: BayesianLinear | None = None

    def _compute_epistemic_uncertainty(
        self, x: jax.Array, mean_pred: jax.Array, training: bool
    ) -> jax.Array:
        """Compute epistemic uncertainty if enabled."""
        if self.use_epistemic and training:
            # Reshape 4D input to 2D for linear layer
            batch_size, height, width, hidden_channels = x.shape
            x_reshaped = x.reshape(batch_size, height * width * hidden_channels)

            # Initialize epistemic head if not already done
            if self.epistemic_head is None:
                self.epistemic_head = BayesianLinear(
                    height * width * hidden_channels, self.out_channels, rngs=self.rngs
                )

            epistemic_logvar = self.epistemic_head(
                x_reshaped, training=training
            )  # (batch, out_channels)
            epistemic_std = jnp.exp(0.5 * epistemic_logvar)  # epistemic_std
            return epistemic_std.reshape(
                batch_size, height, width, self.out_channels
            )  # Reshape back to 4D
        return jnp.zeros_like(mean_pred)

    def _compute_aleatoric_uncertainty(
        self, aleatoric_stds: list[jax.Array], mean_pred: jax.Array
    ) -> jax.Array:
        """Compute and project aleatoric uncertainty."""
        if not (self.use_aleatoric and aleatoric_stds):
            return jnp.zeros_like(mean_pred)

        # Combine uncertainties from all layers (still in channels-first format)
        combined_aleatoric_channels_first = jnp.sqrt(
            jnp.sum(
                jnp.stack([std**2 for std in aleatoric_stds], axis=0),
                axis=0,
            )
        )  # Shape: (batch, hidden_channels, height, width)

        # Convert to channels-last for projection
        combined_aleatoric_for_projection = combined_aleatoric_channels_first.transpose(
            0, 2, 3, 1
        )
        # Shape: (batch, height, width, hidden_channels)

        # Project to output dimension using a linear transformation
        # Note: For uncertainty, we use a simplified projection
        # (mean of uncertainty across channels)
        if self.hidden_channels != self.out_channels:
            # Average across hidden channels to get output channels
            combined_aleatoric = jnp.mean(
                combined_aleatoric_for_projection.reshape(
                    (
                        *combined_aleatoric_for_projection.shape[:-1],
                        self.out_channels,
                        self.hidden_channels // self.out_channels,
                    )
                ),
                axis=-1,
            )
        else:
            combined_aleatoric = combined_aleatoric_for_projection

        return combined_aleatoric

    def __call__(self, x: jax.Array, training: bool = True) -> dict[str, jax.Array]:
        """
        Forward pass with uncertainty quantification.

        Args:
            x: Input tensor (batch, height, width, in_channels)
            training: Whether in training mode

        Returns:
            Dictionary with mean prediction and uncertainties
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.ndim}D")

        _batch_size, _height, _width, _in_channels = x.shape

        # Lifting with shape handling
        x = self.lifting(
            x, training=training
        )  # (batch, height, width, hidden_channels)

        # Convert to channels-first for convolution layers
        x = x.transpose(0, 3, 1, 2)  # (batch, hidden_channels, height, width)

        # Collect aleatoric uncertainties
        aleatoric_stds = []

        # Apply UQNO layers
        for layer in self.uqno_layers:
            x, aleatoric_std = layer(x, training=training)
            aleatoric_stds.append(aleatoric_std)

        # Convert back to channels-last for linear layers
        x = x.transpose(0, 2, 3, 1)  # (batch, height, width, hidden_channels)

        # Output projection
        mean_pred = self.projection(x, training=training)

        # Compute uncertainties
        epistemic_std = self._compute_epistemic_uncertainty(x, mean_pred, training)
        combined_aleatoric = self._compute_aleatoric_uncertainty(
            aleatoric_stds, mean_pred
        )

        # Total uncertainty
        total_uncertainty = jnp.sqrt(epistemic_std**2 + combined_aleatoric**2)

        return {
            "mean": mean_pred,
            "epistemic_uncertainty": epistemic_std,
            "aleatoric_uncertainty": combined_aleatoric,
            "total_uncertainty": total_uncertainty,
        }

    def predict_with_uncertainty(
        self, x: jax.Array, num_samples: int = 10, key: jax.Array | None = None
    ) -> dict[str, jax.Array]:
        """
        Predict with Monte Carlo uncertainty estimation.

        Args:
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            key: Random key for sampling

        Returns:
            Dictionary with prediction statistics
        """
        # Ensure key is not None
        sampling_key: jax.Array = key if key is not None else jax.random.PRNGKey(0)

        predictions = []
        aleatoric_uncertainties = []

        for _ in range(num_samples):
            sampling_key, _subkey = jax.random.split(sampling_key)
            pred = self(x, training=True)
            predictions.append(pred["mean"])
            aleatoric_uncertainties.append(pred["aleatoric_uncertainty"])

        # Compute statistics across samples
        means = jnp.stack(predictions)
        aleatoric_stack = jnp.stack(aleatoric_uncertainties)

        prediction_mean = jnp.mean(means, axis=0)
        epistemic_uncertainty = jnp.std(means, axis=0)
        aleatoric_uncertainty = jnp.mean(aleatoric_stack, axis=0)

        total_uncertainty = jnp.sqrt(
            epistemic_uncertainty**2 + aleatoric_uncertainty**2
        )

        return {
            "mean": prediction_mean,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "total_uncertainty": total_uncertainty,
        }

    def kl_divergence(self) -> jax.Array:
        """Compute total KL divergence for all Bayesian layers."""
        kl_div = self.lifting.kl_divergence() + self.projection.kl_divergence()

        for layer in self.uqno_layers:
            kl_div += layer.kl_divergence()

        if self.use_epistemic and self.epistemic_head is not None:
            kl_div += self.epistemic_head.kl_divergence()

        return kl_div


# Factory functions for different UQNO configurations
def create_safety_critical_uqno(
    in_channels: int, out_channels: int, *, rngs: nnx.Rngs
) -> UncertaintyQuantificationNeuralOperator:
    """Create UQNO for safety-critical applications."""
    return UncertaintyQuantificationNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,
        modes=(32, 32),
        num_layers=6,
        use_epistemic=True,
        use_aleatoric=True,
        ensemble_size=20,
        rngs=rngs,
    )


def create_robust_design_uqno(
    in_channels: int, out_channels: int, *, rngs: nnx.Rngs
) -> UncertaintyQuantificationNeuralOperator:
    """Create UQNO for robust engineering design."""
    return UncertaintyQuantificationNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=96,
        modes=(24, 24),
        num_layers=5,
        use_epistemic=True,
        use_aleatoric=True,
        ensemble_size=15,
        rngs=rngs,
    )


def create_bayesian_inverse_uqno(
    in_channels: int, out_channels: int, *, rngs: nnx.Rngs
) -> UncertaintyQuantificationNeuralOperator:
    """Create UQNO for Bayesian inverse problems."""
    return UncertaintyQuantificationNeuralOperator(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        modes=(16, 16),
        num_layers=4,
        use_epistemic=True,
        use_aleatoric=False,  # Focus on epistemic uncertainty
        ensemble_size=25,
        rngs=rngs,
    )
