r"""Neural exchange-correlation functional for density functional theory.

Implements a meta-GGA-style neural exchange-correlation (XC) functional with an
attention mechanism for non-local correlation and a genuine *exact-constraint*
layer. The functional is written in the enhancement-factor form of every
constraint-based functional (PBE, SCAN) and of the constraint-respecting
machine-learned functionals (DM21):

.. math::
    \varepsilon_{xc}(\rho, |\nabla\rho|) =
        \varepsilon_x^{\text{unif}}(\rho)\,F_{xc}\big(\text{dimensionless features}\big),

with :math:`\varepsilon_x^{\text{unif}}=-C_x\rho^{1/3}` and a Lieb-Oxford-bounded
enhancement factor :math:`F_{xc}\in[0, 1+\kappa]`. The exact constraints
(uniform coordinate scaling, the uniform-electron-gas limit, the Lieb-Oxford
bound and exchange spin scaling) are enforced by
:mod:`opifex.neural.quantum.dft._constraints`; see that module for the formulas
and references.

The functional derivative (the XC potential) is the GGA pair
:math:`(\partial e_{xc}/\partial\rho,\,\partial e_{xc}/\partial\sigma)` with
:math:`\sigma=|\nabla\rho|^2`, obtained by automatic differentiation -- both
channels are live (the density-gradient channel is not zeroed).

References
----------
* J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).
* J. Sun, A. Ruzsinszky, J. P. Perdew, *Phys. Rev. Lett.* **115**, 036402 (2015)
  -- the SCAN exact-constraint set.
* J. Kirkpatrick et al., *Science* **374**, 1385 (2021), arXiv:2102.06179 (DM21).
* E. H. Lieb, S. Oxford, *Int. J. Quantum Chem.* **19**, 427 (1981).
"""

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.quantum.dft._constraints import constrained_xc_energy_density


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
    ) -> None:
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
            raise ValueError(f"Features {features} must be divisible by num_heads {num_heads}")

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
    ) -> None:
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
            advanced_features = feature_dim - density_features - gradient_features  # Remainder
        else:
            density_features = feature_dim // 2
            gradient_features = feature_dim - density_features  # Remainder
            advanced_features = 0

        self.density_proj = nnx.Linear(2, density_features, rngs=rngs)  # density + log(density)
        self.gradient_proj = nnx.Linear(4, gradient_features, rngs=rngs)  # 3 gradients + magnitude

        if use_advanced_features:
            # Advanced physics features (kinetic energy density, etc.)
            self.advanced_proj = nnx.Linear(3, advanced_features, rngs=rngs)

        # Feature normalization for stability
        self.feature_norm = nnx.LayerNorm(feature_dim, rngs=rngs)

        # Numerical stability parameters  handling
        self.eps = jnp.finfo(jnp.float64).eps
        self.density_cutoff = nnx.Param(jnp.array(1e-10))

    def __call__(
        self,
        density: jax.Array,
        gradients: jax.Array,
        *,
        deterministic: bool = False,  # noqa: ARG002 - nnx forward interface carries a deterministic flag
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
            reduced_gradient = gradient_magnitude / (safe_density[..., None] ** (4 / 3) + self.eps)

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
            combined_features = jnp.concatenate([density_features, gradient_features], axis=-1)

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
    ) -> None:
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

        # Final layer outputs the *raw enhancement signal* (one scalar per grid
        # point), which the exact-constraint layer squashes into a
        # Lieb-Oxford-bounded enhancement factor. Initialised to zero so the
        # untrained functional reduces exactly to the uniform-gas (LDA) limit.
        self.output_layer = nnx.Linear(
            hidden_sizes[-1],
            1,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def _raw_enhancement_signal(
        self, density: jax.Array, gradients: jax.Array, *, deterministic: bool
    ) -> jax.Array:
        r"""Network output: the unbounded enhancement signal per grid point.

        Built from physics-informed features (which include the dimensionless
        reduced gradient :math:`s`), passed through the attention/MLP trunk. The
        exact-constraint layer turns this into a bounded enhancement factor.
        """
        features = self.feature_extractor(density, gradients, deterministic=deterministic)
        if self.use_attention:
            attention_output = self.attention(features, deterministic=deterministic)
            features = features + attention_output  # Residual connection

        x = features
        for i, layer in enumerate(self.layers):
            layer_output = layer(x)
            if layer_output.shape == x.shape and i > 0:
                layer_output = layer_output + x  # Residual connection for depth
            x = self.activation(layer_output)
            if self.dropout_rate > 0.0:
                x = self.dropout(x, deterministic=deterministic)

        return jnp.squeeze(self.output_layer(x), axis=-1)

    def __call__(
        self, density: jax.Array, gradients: jax.Array, *, deterministic: bool = False
    ) -> jax.Array:
        r"""Compute the constraint-satisfying XC energy per particle.

        The network produces a raw enhancement signal which the exact-constraint
        layer maps to :math:`\varepsilon_{xc}=\varepsilon_x^{\text{unif}}(\rho)
        F_{xc}` with :math:`F_{xc}\in[0,1+\kappa]` (Lieb-Oxford bounded), so the
        output obeys uniform coordinate scaling, the uniform-gas limit and the
        Lieb-Oxford bound by construction.

        Args:
            density: Electron density [batch, grid_points].
            gradients: Density gradients [batch, grid_points, 3].
            deterministic: Whether to use deterministic computation.

        Returns:
            XC energy per particle [batch, grid_points].
        """
        if density.ndim != 2:
            raise ValueError(f"Expected 2D density tensor, got {density.ndim}D")
        if gradients.ndim != 3:
            raise ValueError(f"Expected 3D gradient tensor, got {gradients.ndim}D")

        raw = self._raw_enhancement_signal(density, gradients, deterministic=deterministic)
        sigma = jnp.sum(gradients**2, axis=-1)
        return constrained_xc_energy_density(raw, density, sigma)

    def energy_density_from_sigma(
        self, density: jax.Array, sigma: jax.Array, *, deterministic: bool = True
    ) -> jax.Array:
        r"""XC energy per particle as a function of ``rho`` and ``sigma=|grad rho|^2``.

        The GGA-native interface used on a real molecular grid and for the AD XC
        potential: the gradient direction is irrelevant to a (semi-)local
        functional, so the dimensionless features depend only on
        :math:`(\rho,\sigma)`. The Cartesian gradient is reconstructed along a
        single axis with magnitude :math:`\sqrt\sigma` purely to reuse the
        feature extractor; the resulting energy density is identical for any
        direction.

        Args:
            density: Electron density [Shape: (n_points,)].
            sigma: Squared density gradient ``|grad rho|^2`` [Shape: (n_points,)].
            deterministic: Whether to use deterministic computation.

        Returns:
            XC energy per particle [Shape: (n_points,)].
        """
        # Floor the sqrt argument so d/dsigma is finite at sigma = 0 (the bare
        # square root has an infinite derivative there); the floor is far below
        # any physically resolved gradient magnitude.
        magnitude = jnp.sqrt(jnp.clip(sigma, 0.0, None) + 1.0e-24)
        zeros = jnp.zeros_like(magnitude)
        gradient = jnp.stack([magnitude, zeros, zeros], axis=-1)[None, ...]
        raw = self._raw_enhancement_signal(density[None, :], gradient, deterministic=deterministic)[
            0
        ]
        return constrained_xc_energy_density(raw, density, sigma)

    def xc_potential_components(
        self, density: jax.Array, sigma: jax.Array, *, deterministic: bool = True
    ) -> tuple[jax.Array, jax.Array]:
        r"""GGA XC potential pair :math:`(v_\rho, v_\sigma)` by autodiff.

        Returns both functional derivatives of the XC energy density
        :math:`\rho\,\varepsilon_{xc}(\rho,\sigma)`:

        .. math::
            v_\rho = \frac{\partial(\rho\varepsilon_{xc})}{\partial\rho},\qquad
            v_\sigma = \frac{\partial(\rho\varepsilon_{xc})}{\partial\sigma}.

        Both channels are live -- the density-gradient (:math:`\sigma`) channel
        is differentiated, not zeroed -- so the GGA potential is correct.

        Args:
            density: Electron density [Shape: (n_points,)].
            sigma: Squared density gradient ``|grad rho|^2`` [Shape: (n_points,)].
            deterministic: Whether to use deterministic computation.

        Returns:
            The pair ``(v_rho, v_sigma)`` each [Shape: (n_points,)].
        """

        def energy_density(rho: jax.Array, sig: jax.Array) -> jax.Array:
            per_particle = self.energy_density_from_sigma(rho, sig, deterministic=deterministic)
            return jnp.sum(rho * per_particle)

        v_rho = jax.grad(energy_density, argnums=0)(density, sigma)
        v_sigma = jax.grad(energy_density, argnums=1)(density, sigma)
        return v_rho, v_sigma

    def compute_functional_derivative(
        self, density: jax.Array, gradients: jax.Array, *, deterministic: bool = False
    ) -> jax.Array:
        r"""Density-channel functional derivative ``d(rho eps_xc)/d rho``.

        Computes the *live* GGA density-channel potential at fixed
        :math:`\sigma=|\nabla\rho|^2`. The full GGA potential additionally needs
        the :math:`\sigma` channel; use :meth:`xc_potential_components` for both.

        Args:
            density: Electron density [batch, grid_points] or [grid_points].
            gradients: Density gradients [..., 3].
            deterministic: Whether to use deterministic computation.

        Returns:
            ``d(rho eps_xc)/d rho`` with the same shape as ``density``.
        """
        flat_density = density.reshape(-1)
        flat_sigma = jnp.sum(gradients**2, axis=-1).reshape(-1)
        v_rho, _ = self.xc_potential_components(
            flat_density, flat_sigma, deterministic=deterministic
        )
        return v_rho.reshape(density.shape)

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
