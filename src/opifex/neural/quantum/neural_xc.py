"""Neural exchange-correlation functional for density functional theory.

Implements modern neural XC functionals with attention mechanisms for capturing
non-local correlations and enhanced physics constraint enforcement for
chemical accuracy.
"""

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadAttention(nnx.Module):
    """Multi-head attention mechanism for non-local correlations in XC functionals.

    Implements scaled dot-product attention to capture long-range electron correlations
    that are crucial for accurate exchange-correlation energy predictions.
    """

    def __init__(
        self,
        features: int,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize multi-head attention with enhanced capabilities.

        Args:
            features: Number of input/output features
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for attention weights
            rngs: Random number generators for initialization
        """
        super().__init__()
        self.features = features
        self.num_heads = num_heads
        self.head_dim = features // num_heads
        self.dropout_rate = dropout_rate

        if features % num_heads != 0:
            raise ValueError(
                f"Features {features} must be divisible by num_heads {num_heads}"
            )

        # Enhanced Query, Key, Value projections  handling
        self.q_proj = nnx.Linear(features, features, rngs=rngs)
        self.k_proj = nnx.Linear(features, features, rngs=rngs)
        self.v_proj = nnx.Linear(features, features, rngs=rngs)
        self.out_proj = nnx.Linear(features, features, rngs=rngs)

        # Dropout for regularization
        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        # Scale factor for numerical stability
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
        """Apply multi-head attention with enhanced stability.

        Args:
            x: Input tensor [batch, seq_len, features]
            deterministic: Whether to use deterministic computation (disables dropout)

        Returns:
            Attention output [batch, seq_len, features]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.ndim}D")

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V with input validation
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation: [batch, heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention with numerical stability
        attn_weights = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply softmax with numerical stability
        attn_weights = nnx.softmax(attn_weights, axis=-1)

        # Apply dropout if training
        if self.dropout_rate > 0.0 and not deterministic:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, v)

        # Transpose back and reshape: [batch, seq_len, features]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.features)

        # Final projection
        return self.out_proj(attn_output)


class DensityFeatureExtractor(nnx.Module):
    """Extract physics-informed features from density and gradients for XC functional.

    Implements advanced feature extraction that captures both local and semi-local
    density information crucial for accurate exchange-correlation predictions.
    """

    def __init__(
        self,
        feature_dim: int,
        use_advanced_features: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize density feature extractor with enhanced capabilities.

        Args:
            feature_dim: Output feature dimension
            use_advanced_features: Whether to include advanced physics features
            rngs: Random number generators for initialization
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.use_advanced_features = use_advanced_features

        # Enhanced feature extraction layers
        if use_advanced_features:
            # Ensure dimensions sum to feature_dim exactly
            density_features = feature_dim // 3
            gradient_features = feature_dim // 3
            advanced_features = (
                feature_dim - density_features - gradient_features
            )  # Remainder
        else:
            density_features = feature_dim // 2
            gradient_features = feature_dim - density_features  # Remainder
            advanced_features = 0

        self.density_proj = nnx.Linear(
            2, density_features, rngs=rngs
        )  # density + log(density)
        self.gradient_proj = nnx.Linear(
            4, gradient_features, rngs=rngs
        )  # 3 gradients + magnitude

        if use_advanced_features:
            # Advanced physics features (kinetic energy density, etc.)
            self.advanced_proj = nnx.Linear(3, advanced_features, rngs=rngs)

        # Feature normalization for stability
        self.feature_norm = nnx.LayerNorm(feature_dim, rngs=rngs)

        # Numerical stability parameters  handling
        self.eps = jnp.finfo(jnp.float64).eps
        self.density_cutoff = nnx.Param(jnp.array(1e-10))

    def __call__(
        self, density: jax.Array, gradients: jax.Array, *, deterministic: bool = False
    ) -> jax.Array:
        """Extract physics-informed features from density and gradients.

        Args:
            density: Electron density [batch, grid_points]
            gradients: Density gradients [batch, grid_points, 3]
            deterministic: Whether to use deterministic computation

        Returns:
            Combined features [batch, grid_points, feature_dim]
        """
        # Input validation
        if density.ndim != 2:
            raise ValueError(f"Expected 2D density tensor, got {density.ndim}D")
        if gradients.ndim != 3 or gradients.shape[-1] != 3:
            raise ValueError(
                f"Expected 3D gradient tensor with 3 components, got {gradients.shape}"
            )
        if density.shape != gradients.shape[:-1]:
            raise ValueError(
                f"Density and gradient shapes incompatible: {density.shape} vs "
                f"{gradients.shape[:-1]}"
            )

        # Ensure numerical stability for density
        safe_density = jnp.maximum(density, self.density_cutoff.value)

        # Create enhanced density features
        log_density = jnp.log(safe_density + self.eps)
        density_features = jnp.stack([safe_density, log_density], axis=-1)
        density_features = self.density_proj(density_features)

        # Create enhanced gradient features
        gradient_magnitude = jnp.linalg.norm(gradients, axis=-1, keepdims=True)
        gradient_features = jnp.concatenate([gradients, gradient_magnitude], axis=-1)
        gradient_features = self.gradient_proj(gradient_features)

        # Combine base features
        if self.use_advanced_features:
            # Advanced physics features
            # Reduced density gradient (important for GGA functionals)
            reduced_gradient = gradient_magnitude / (
                safe_density[..., None] ** (4 / 3) + self.eps
            )

            # Approximate kinetic energy density
            kinetic_energy_density = (
                0.5
                * jnp.sum(gradients**2, axis=-1, keepdims=True)
                / (safe_density[..., None] + self.eps)
            )

            # Local Fermi wavevector
            fermi_wavevector = (3 * jnp.pi**2 * safe_density) ** (1 / 3)
            fermi_wavevector = fermi_wavevector[..., None]

            advanced_features = jnp.concatenate(
                [reduced_gradient, kinetic_energy_density, fermi_wavevector], axis=-1
            )
            advanced_features = self.advanced_proj(advanced_features)

            combined_features = jnp.concatenate(
                [density_features, gradient_features, advanced_features], axis=-1
            )
        else:
            combined_features = jnp.concatenate(
                [density_features, gradient_features], axis=-1
            )

        # Apply normalization for numerical stability
        return self.feature_norm(combined_features)


class NeuralXCFunctional(nnx.Module):
    """Neural exchange-correlation functional for DFT calculations.

    Implements a modern neural XC functional with attention mechanisms for
    capturing non-local correlations, enhanced physics constraints, and
    chemical accuracy optimization.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (128, 128, 64),
        activation: Callable = nnx.gelu,
        use_attention: bool = True,
        num_attention_heads: int = 8,
        use_advanced_features: bool = True,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize neural XC functional with enhanced capabilities.

        Args:
            hidden_sizes: Sequence of hidden layer sizes
            activation: Activation function to use
            use_attention: Whether to use attention mechanism for non-local correlations
            num_attention_heads: Number of attention heads
            use_advanced_features: Whether to include advanced physics features
            dropout_rate: Dropout rate for regularization
            rngs: Random number generators
        """
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.use_attention = use_attention
        self.use_advanced_features = use_advanced_features
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Enhanced feature extraction
        feature_dim = hidden_sizes[0]
        self.feature_extractor = DensityFeatureExtractor(
            feature_dim, use_advanced_features, rngs=rngs
        )

        # Attention mechanism for non-local correlations
        if use_attention:
            self.attention = MultiHeadAttention(
                feature_dim, num_attention_heads, dropout_rate, rngs=rngs
            )

        # Enhanced neural network layers with residual connections
        layers_temp = []
        for i, size in enumerate(hidden_sizes):
            if i == 0:
                continue  # Skip first layer as we use feature extractor

            layers_temp.append(nnx.Linear(hidden_sizes[i - 1], size, rngs=rngs))

        self.layers = nnx.List(layers_temp)

        # Add dropout for regularization if specified
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        # Final output layer (XC energy per particle)
        self.output_layer = nnx.Linear(hidden_sizes[-1], 1, rngs=rngs)

        # Enhanced physics constraint enforcement  handling
        self.constraint_scale = nnx.Param(jnp.array(1.0))
        self.exchange_weight = nnx.Param(jnp.array(0.7))  # Typical exchange fraction
        self.correlation_weight = nnx.Param(jnp.array(0.3))

        # Numerical stability parameters  handling
        self.energy_clamp_min = nnx.Param(
            jnp.array(-10.0)
        )  # Reasonable XC energy bounds
        self.energy_clamp_max = nnx.Param(jnp.array(0.0))

    def _enforce_physics_constraints(
        self, xc_energy: jax.Array, density: jax.Array, *, deterministic: bool = False
    ) -> jax.Array:
        """Enforce physics constraints on XC energy with enhanced validation.

        Args:
            xc_energy: Raw XC energy per particle
            density: Electron density
            deterministic: Whether to use deterministic computation

        Returns:
            Constrained XC energy satisfying physical principles
        """
        # Ensure XC energy is negative (attractive) with proper scaling
        constrained_energy = -jnp.abs(xc_energy) * self.constraint_scale

        # Clamp energy to reasonable physical bounds
        constrained_energy = jnp.clip(
            constrained_energy, self.energy_clamp_min.value, self.energy_clamp_max.value
        )

        # Scale appropriately with density for numerical stability
        eps = jnp.finfo(jnp.float64).eps
        density_factor = jnp.tanh(density + eps)

        # Apply physics-based density scaling
        # XC energy per particle should scale properly with density
        scaled_energy = constrained_energy * density_factor

        # Ensure smooth behavior at low densities
        low_density_cutoff = 1e-8
        smooth_factor = jnp.where(
            density < low_density_cutoff, density / low_density_cutoff, 1.0
        )

        return scaled_energy * smooth_factor

    def __call__(
        self, density: jax.Array, gradients: jax.Array, *, deterministic: bool = False
    ) -> jax.Array:
        """Compute XC energy per particle with enhanced physics constraints.

        Args:
            density: Electron density [batch, grid_points]
            gradients: Density gradients [batch, grid_points, 3]
            deterministic: Whether to use deterministic computation

        Returns:
            XC energy per particle [batch, grid_points]
        """
        # Input validation
        if density.ndim != 2:
            raise ValueError(f"Expected 2D density tensor, got {density.ndim}D")
        if gradients.ndim != 3:
            raise ValueError(f"Expected 3D gradient tensor, got {gradients.ndim}D")

        # Extract physics-informed features
        features = self.feature_extractor(
            density, gradients, deterministic=deterministic
        )

        # Apply attention for non-local correlations
        if self.use_attention:
            attention_output = self.attention(features, deterministic=deterministic)
            features = features + attention_output  # Residual connection

            # Process through neural network layers with skip connections
        x = features
        for i, layer in enumerate(self.layers):
            # Apply linear layer
            layer_output = layer(x)

            # Add residual connection for deeper layers
            if layer_output.shape == x.shape and i > 0:
                layer_output = layer_output + x

            x = self.activation(layer_output)

            # Apply dropout if available
            if self.dropout_rate > 0.0:
                x = self.dropout(x, deterministic=deterministic)

        # Final output layer without activation
        xc_energy_raw = self.output_layer(x)

        # Remove last dimension and apply physics constraints
        xc_energy_raw = jnp.squeeze(xc_energy_raw, axis=-1)
        return self._enforce_physics_constraints(
            xc_energy_raw, density, deterministic=deterministic
        )

    def compute_functional_derivative(
        self, density: jax.Array, gradients: jax.Array, *, deterministic: bool = False
    ) -> jax.Array:
        """Compute functional derivative of XC energy with respect to density.

        Args:
            density: Electron density
            gradients: Density gradients
            deterministic: Whether to use deterministic computation

        Returns:
            Functional derivative ∂E_xc/∂ρ with enhanced numerical stability
        """

        def xc_energy_fn(rho):
            # Create approximate gradients for derivative computation
            grad_shape = (*rho.shape, 3)
            grads = jnp.zeros(grad_shape)
            return jnp.sum(self(rho, grads, deterministic=deterministic) * rho)

        # Compute derivative using JAX automatic differentiation
        derivative = jax.grad(xc_energy_fn)(density)

        # Apply numerical stability measures
        eps = jnp.finfo(jnp.float64).eps
        result = jnp.where(
            jnp.abs(derivative) < eps, jnp.sign(derivative) * eps, derivative
        )
        # Ensure we return a single Array, not a tuple
        return jnp.asarray(result)

    def assess_chemical_accuracy(
        self,
        density: jax.Array,
        gradients: jax.Array,
        reference_energy: jax.Array | None = None,
        *,
        deterministic: bool = False,
    ) -> dict[str, float]:
        """Assess chemical accuracy of XC functional predictions.

        Args:
            density: Electron density
            gradients: Density gradients
            reference_energy: Reference XC energy for comparison (optional)
            deterministic: Whether to use deterministic computation

        Returns:
            Dictionary containing accuracy metrics
        """
        predicted_energy = self(density, gradients, deterministic=deterministic)

        metrics = {
            "predicted_xc_energy": float(jnp.sum(predicted_energy)),
            "energy_per_particle_mean": float(jnp.mean(predicted_energy)),
            "energy_per_particle_std": float(jnp.std(predicted_energy)),
            "density_integrated": float(jnp.sum(density)),
            "numerical_stability": float(jnp.all(jnp.isfinite(predicted_energy))),
        }

        if reference_energy is not None:
            error = predicted_energy - reference_energy
            metrics.update(
                {
                    "absolute_error": float(jnp.mean(jnp.abs(error))),
                    "relative_error": float(
                        jnp.mean(jnp.abs(error) / (jnp.abs(reference_energy) + 1e-10))
                    ),
                    "max_error": float(jnp.max(jnp.abs(error))),
                    "chemical_accuracy_achieved": float(
                        jnp.mean(jnp.abs(error)) < 1e-3
                    ),  # ~1 kcal/mol
                }
            )

        return metrics
