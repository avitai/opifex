# FILE PLACEMENT: opifex/neural/bayesian/probabilistic_pinns.py
#
# FIXED MultiFidelityPINN Implementation
# Fixes missing configuration keys and proper initialization
#
# This file should REPLACE: opifex/neural/bayesian/probabilistic_pinns.py

"""
Probabilistic Physics-Informed Neural Networks

Advanced PINN implementations with uncertainty quantification and
multi-fidelity modeling for robust scientific computing applications.

Key Features:
- Multi-fidelity neural networks with adaptive selection
- Bayesian uncertainty quantification
- Proper configuration validation and key handling
- Epistemic and aleatoric uncertainty estimation
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx

from opifex.neural.bayesian.config import FidelityConfig, MultiFidelityConfig
from opifex.uncertainty.layers.bayesian import BayesianLinear
from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
from opifex.uncertainty.types import PredictiveDistribution, PredictiveMode


_PINN_RNG_STREAMS = ("sample", "default")

# Boundary-condition types supported by :func:`compute_boundary_residual`.
# Dirichlet uses the field value directly; Neumann/Robin need the model's
# normal derivative and therefore a differentiable ``model`` callable.
_BC_TYPE_DIRICHLET = "dirichlet"
_BC_TYPE_NEUMANN = "neumann"
_BC_TYPE_ROBIN = "robin"
_SUPPORTED_BC_TYPES = (_BC_TYPE_DIRICHLET, _BC_TYPE_NEUMANN, _BC_TYPE_ROBIN)


def _resolve_bc_target(value: Any, coords: jax.Array) -> jax.Array:
    """Resolve a boundary target ``g`` to an array broadcastable to predictions.

    ``value`` may be a constant (scalar / array) or a callable ``g(x)`` that
    maps boundary coordinates to target values (deepxde ``DirichletBC`` pattern,
    ``../deepxde/deepxde/icbc/boundary_conditions.py``).
    """
    if callable(value):
        return jnp.asarray(value(coords))
    return jnp.asarray(value)


def _normal_derivative(model: Callable[[jax.Array], jax.Array], coords: jax.Array) -> jax.Array:
    """Return ``du/dn`` at ``coords`` via JVP of the model along the outward normal.

    For the 1-D / axis-aligned boundaries handled here the outward normal is the
    sum of input-coordinate partials, matching deepxde ``NeumannBC.error``
    (``../deepxde``). Implemented with :func:`jax.jvp` so it composes with
    ``jit``/``grad``/``vmap``.
    """
    ones = jnp.ones_like(coords)
    _, directional = jax.jvp(model, (coords,), (ones,))
    return directional


def compute_boundary_residual(
    x: jax.Array,
    y_pred: jax.Array,
    boundary: Mapping[str, Any],
) -> jax.Array:
    """Compute the per-point boundary-condition residual ``r_b`` for a B-PINN.

    Implements the boundary-condition likelihood residual of a Bayesian
    Physics-Informed Neural Network (Yang, Meng & Karniadakis 2021,
    *"B-PINNs"*, J. Comput. Phys. 425:109913, arXiv:2003.06097): the residual
    placed under a Gaussian BC likelihood alongside the PDE residual. The
    Dirichlet/Neumann/Robin residual forms follow deepxde's ``DirichletBC`` /
    ``NeumannBC`` / ``RobinBC`` ``error`` methods
    (``../deepxde/deepxde/icbc/boundary_conditions.py``).

    Args:
        x: Boundary coordinates ``x_b`` at which the BC is enforced.
        y_pred: Model prediction ``u_theta(x_b)`` at those coordinates.
        boundary: BC specification. Recognised keys:

            * ``type`` — ``"dirichlet"`` (default), ``"neumann"`` or
              ``"robin"``.
            * ``value`` — target ``g``; a constant or a callable ``g(x)``.
              Robin targets are callables ``g(x, u)`` of coordinate and field.
            * ``model`` — differentiable ``u_theta(x)`` callable; REQUIRED for
              Neumann/Robin to form the normal derivative.

    Returns:
        The residual array (NOT yet squared / reduced) shaped like ``y_pred``.

    Raises:
        ValueError: when ``type`` is unsupported, or a Neumann/Robin BC omits
            the ``model`` callable needed for the normal derivative.
    """
    bc_type = str(boundary.get("type", _BC_TYPE_DIRICHLET)).lower()
    if bc_type not in _SUPPORTED_BC_TYPES:
        raise ValueError(
            f"unsupported boundary-condition type {bc_type!r}; "
            f"supported types: {list(_SUPPORTED_BC_TYPES)!r}."
        )
    raw_value = boundary.get("value", 0.0)

    if bc_type == _BC_TYPE_DIRICHLET:
        target = _resolve_bc_target(raw_value, x)
        return y_pred - target

    model: Callable[[jax.Array], jax.Array] | None = boundary.get("model")
    if model is None or not callable(model):
        raise ValueError(
            f"boundary-condition type {bc_type!r} requires a 'model' callable "
            "(differentiable u_theta(x)) to evaluate the normal derivative."
        )
    normal_grad = _normal_derivative(model, x)
    if bc_type == _BC_TYPE_NEUMANN:
        target = _resolve_bc_target(raw_value, x)
        return normal_grad - target
    # Robin: du/dn - g(x, u); the target depends on both coordinate and field.
    robin_target = (
        jnp.asarray(raw_value(x, y_pred)) if callable(raw_value) else jnp.asarray(raw_value)
    )
    return normal_grad - robin_target


def _boundary_loss_term(
    interior_x: jax.Array,
    interior_pred: jax.Array,
    boundary: Mapping[str, Any],
    model: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Return the scalar BC MSE for one ``boundary_conditions`` mapping.

    The BC is evaluated at the dedicated boundary points ``boundary_x`` when
    supplied (the B-PINN contract — BC data live on the boundary, not the
    interior PDE collocation points). For backward compatibility, when no
    ``boundary_x`` is given the BC is scored on the interior batch (legacy
    single-value-over-domain behaviour). The returned value is the RAW mean
    squared residual — no hidden multiplier — so ``ObjectiveConfig`` weights
    remain the single source of truth.
    """
    boundary_x = boundary.get("boundary_x")
    if boundary_x is not None:
        x_b = jnp.asarray(boundary_x)
        y_b = model(x_b)
    else:
        x_b = interior_x
        y_b = interior_pred
    # Neumann/Robin need a differentiable model callable for the normal
    # derivative; inject the wrapped model when the spec does not carry one.
    spec = boundary
    if boundary.get("model") is None and str(boundary.get("type", "")).lower() in (
        _BC_TYPE_NEUMANN,
        _BC_TYPE_ROBIN,
    ):
        spec = {**boundary, "model": model}
    residual = compute_boundary_residual(x_b, y_b, spec)
    return jnp.mean(residual**2)


def _coerce_predictive_mode(mode: PredictiveMode | str) -> PredictiveMode:
    """Coerce ``mode`` to :class:`PredictiveMode`; raise ``ValueError`` on miss.

    Raised messages name the offending mode and the legal values so the
    caller can correct the call site.
    """
    if isinstance(mode, PredictiveMode):
        return mode
    try:
        return PredictiveMode(mode)
    except (ValueError, TypeError) as exc:
        valid = sorted(m.value for m in PredictiveMode)
        raise ValueError(f"Unknown predictive mode {mode!r}; legal values: {valid!r}.") from exc


class MultiFidelityPINN(nnx.Module):
    """
    Multi-Fidelity Physics-Informed Neural Network with proper configuration.

    Implements adaptive multi-fidelity modeling with uncertainty quantification
    for physics-informed neural networks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        low_fidelity_dims: tuple[int, ...] = (32, 32),
        high_fidelity_dims: tuple[int, ...] = (64, 64),
        fusion_dims: tuple[int, ...] = (48,),
        config: MultiFidelityConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """
        Initialize MultiFidelityPINN with full configuration.

        Args:
            input_dim: Dimension of input space
            output_dim: Dimension of output space
            low_fidelity_dims: Hidden dimensions for low fidelity network
            high_fidelity_dims: Hidden dimensions for high fidelity network
            fusion_dims: Hidden dimensions for fusion network
            config: Multi-fidelity configuration (optional, uses defaults if None)
            rngs: Caller-owned ``nnx.Rngs`` bundle; required (no hidden fallback).
        """
        if rngs is None:
            raise ValueError(
                "rngs is required; pass a caller-owned nnx.Rngs bundle to "
                "avoid hidden fixed-seed Monte Carlo paths."
            )
        # Persist the RNG bundle so forward passes can pass it through to
        # BayesianLinear sampling without a per-call argument from callers.
        self.rngs = rngs

        # Store network architecture parameters
        self.low_fidelity_dims = low_fidelity_dims
        self.high_fidelity_dims = high_fidelity_dims
        self.fusion_dims = fusion_dims

        # FIXED: Use default config if none provided and validate
        if config is None:
            config = MultiFidelityConfig()

        # Ensure configuration is complete and validated
        if not isinstance(config, MultiFidelityConfig):
            raise TypeError("config must be MultiFidelityConfig instance")

        config._validate_config()  # Explicit validation
        self.config = config

        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Extract network parameters
        hidden_layers = list(low_fidelity_dims)  # Use provided dimensions
        activation = config.network_params.get("activation", "tanh")
        use_bayesian = config.uncertainty_params.get("use_epistemic", True)

        # Build multi-fidelity networks
        self._build_networks(hidden_layers, activation, use_bayesian, rngs)

        # Initialize prediction statistics for adaptive selection
        self._initialize_prediction_stats()

    def _build_networks(
        self,
        hidden_layers: list[int],
        activation: str,
        use_bayesian: bool,
        rngs: nnx.Rngs,
    ) -> None:
        """Build low and high fidelity networks."""
        self._set_activation_function(activation)
        self._build_low_fidelity_network(hidden_layers, use_bayesian, rngs)
        self._build_high_fidelity_networks(hidden_layers, use_bayesian, rngs)

    def _set_activation_function(self, activation: str) -> None:
        """Set the activation function based on the activation string."""
        activation_map = {
            "tanh": nnx.tanh,
            "relu": nnx.relu,
            "gelu": nnx.gelu,
        }
        self.activation = activation_map.get(activation, nnx.tanh)

    def _create_layer(self, in_dim: int, out_dim: int, use_bayesian: bool, rngs: nnx.Rngs):
        """Create a layer (either Bayesian or Linear) based on the configuration."""
        if use_bayesian:
            return BayesianLinear(in_dim, out_dim, rngs=rngs)
        return nnx.Linear(in_dim, out_dim, rngs=rngs)

    def _build_low_fidelity_network(
        self, hidden_layers: list[int], use_bayesian: bool, rngs: nnx.Rngs
    ) -> None:
        """Build the low-fidelity network layers."""
        low_fidelity_layers_temp = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_layers:
            layer = self._create_layer(prev_dim, hidden_dim, use_bayesian, rngs)
            low_fidelity_layers_temp.append(layer)
            prev_dim = hidden_dim

        self.low_fidelity_layers = nnx.List(low_fidelity_layers_temp)
        # self.layers = nnx.List(layers_temp) - REMOVED: undefined variable

        # Low-fidelity output layer
        self.low_fidelity_output = self._create_layer(prev_dim, self.output_dim, use_bayesian, rngs)

    def _build_high_fidelity_networks(
        self, hidden_layers: list[int], use_bayesian: bool, rngs: nnx.Rngs
    ) -> None:
        """Build the high-fidelity correction networks."""
        high_fidelity_networks_temp = []

        for _ in range(self.config.high_fidelity_count):
            network_layers = self._build_single_high_fidelity_network(
                hidden_layers, use_bayesian, rngs
            )
            high_fidelity_networks_temp.append(nnx.List(network_layers))
        self.high_fidelity_networks = nnx.List(high_fidelity_networks_temp)

    def _build_single_high_fidelity_network(
        self, hidden_layers: list[int], use_bayesian: bool, rngs: nnx.Rngs
    ) -> list:
        """Build a single high-fidelity correction network."""
        network_layers = []
        # Input: original input + low-fidelity prediction
        prev_dim = self.input_dim + self.output_dim

        for hidden_dim in hidden_layers:
            layer = self._create_layer(prev_dim, hidden_dim, use_bayesian, rngs)
            network_layers.append(layer)
            prev_dim = hidden_dim

        # Output layer
        output_layer = self._create_layer(prev_dim, self.output_dim, use_bayesian, rngs)
        network_layers.append(output_layer)

        return network_layers

    def _initialize_prediction_stats(self) -> None:
        """Initialize prediction statistics for monitoring."""
        # These will be tracked during training/inference
        self.prediction_stats = {
            "low_fidelity_calls": 0,
            "high_fidelity_calls": 0,
            "total_predictions": 0,
            "average_uncertainty": 0.0,
            "uncertainty_threshold_hits": 0,
        }

    def _low_fidelity_forward(
        self, x: jax.Array, *, deterministic: bool | None = None
    ) -> dict[str, jax.Array]:
        """Low-fidelity forward pass with uncertainty estimation.

        Args:
            x: Input tensor
            deterministic: Per-call override of ``self.deterministic`` for
                Bayesian-layer mode (mirrors :class:`nnx.Dropout`).

        Returns:
            Dictionary with prediction and uncertainty
        """
        h = x
        for layer in self.low_fidelity_layers:
            if isinstance(layer, BayesianLinear):
                h = self.activation(layer(h, deterministic=deterministic, rngs=self.rngs))
            else:
                h = self.activation(layer(h))

        # Output prediction
        if isinstance(self.low_fidelity_output, BayesianLinear):
            prediction = self.low_fidelity_output(h, deterministic=deterministic, rngs=self.rngs)
        else:
            prediction = self.low_fidelity_output(h)

        # Estimate uncertainty (simplified - could be improved with ensemble)
        uncertainty = jnp.std(prediction, axis=-1, keepdims=True) + 1e-6

        return {"low_fidelity_pred": prediction, "uncertainty_estimate": uncertainty}

    def _high_fidelity_forward(
        self, x: jax.Array, *, deterministic: bool | None = None
    ) -> dict[str, jax.Array]:
        """High-fidelity forward pass using first correction network."""
        # Get low-fidelity prediction first
        low_result = self._low_fidelity_forward(x, deterministic=deterministic)
        low_pred = low_result["low_fidelity_pred"]

        # High-fidelity correction using first network
        correction_input = jnp.concatenate([x, low_pred], axis=-1)
        h = correction_input

        for layer in self.high_fidelity_networks[0][:-1]:
            if isinstance(layer, BayesianLinear):
                h = self.activation(layer(h, deterministic=deterministic, rngs=self.rngs))
            else:
                h = self.activation(layer(h))

        # Correction prediction
        if isinstance(self.high_fidelity_networks[0][-1], BayesianLinear):
            correction = self.high_fidelity_networks[0][-1](
                h, deterministic=deterministic, rngs=self.rngs
            )
        else:
            correction = self.high_fidelity_networks[0][-1](h)

        high_pred = low_pred + correction

        # Reduced uncertainty for high-fidelity
        uncertainty = low_result["uncertainty_estimate"] * 0.5

        return {
            "high_fidelity_pred": high_pred,
            "low_fidelity_pred": low_pred,
            "uncertainty_estimate": uncertainty,
        }

    def _adaptive_forward(
        self, x: jax.Array, *, deterministic: bool | None = None
    ) -> dict[str, jax.Array]:
        """Adaptive forward pass with fidelity selection based on uncertainty."""
        # Start with low-fidelity
        low_result = self._low_fidelity_forward(x, deterministic=deterministic)
        uncertainty = low_result["uncertainty_estimate"]

        # Decide whether to use high-fidelity based on uncertainty threshold
        use_high_fidelity = uncertainty > self.config.uncertainty_threshold

        # FIXED: Count predictions by fidelity level (convert to Python int)
        low_fidelity_count = int(jnp.sum(~use_high_fidelity))
        high_fidelity_count = int(jnp.sum(use_high_fidelity))

        # Apply high-fidelity where needed
        if jnp.any(use_high_fidelity):
            high_result = self._high_fidelity_forward(x, deterministic=deterministic)

            # Select predictions based on uncertainty
            final_pred = jnp.where(
                use_high_fidelity,
                high_result["high_fidelity_pred"],
                low_result["low_fidelity_pred"],
            )

            final_uncertainty = jnp.where(
                use_high_fidelity,
                high_result["uncertainty_estimate"],
                low_result["uncertainty_estimate"],
            )
        else:
            final_pred = low_result["low_fidelity_pred"]
            final_uncertainty = low_result["uncertainty_estimate"]

        # FIXED: Return dictionary with all required keys - convert scalars to Arrays
        return {
            "prediction": final_pred,
            "low_fidelity_pred": low_result["low_fidelity_pred"],
            "uncertainty_estimate": final_uncertainty,
            "use_high_fidelity": jnp.array(use_high_fidelity),
            "low_fidelity_count": jnp.array(low_fidelity_count),
            "high_fidelity_count": jnp.array(high_fidelity_count),
            # Additional statistics
            "total_samples": jnp.array(low_fidelity_count + high_fidelity_count),
            "high_fidelity_ratio": jnp.array(
                high_fidelity_count / (low_fidelity_count + high_fidelity_count + 1e-8)
            ),
        }

    def __call__(
        self,
        x: jax.Array,
        fidelity_level: str = "adaptive",
        *,
        deterministic: bool | None = None,
    ) -> dict[str, jax.Array]:
        """Forward pass with fidelity selection.

        Args:
            x: Input tensor
            fidelity_level: Type of fidelity ('low', 'high', 'adaptive')
            deterministic: Per-call mode override for Bayesian-layer sampling
                (mirrors :class:`nnx.Dropout`); falls back to
                ``self.deterministic`` when ``None``.

        Returns:
            Dictionary with predictions and metadata
        """
        if fidelity_level == "low":
            return self._low_fidelity_forward(x, deterministic=deterministic)
        if fidelity_level == "high":
            return self._high_fidelity_forward(x, deterministic=deterministic)
        if fidelity_level == "adaptive":
            return self._adaptive_forward(x, deterministic=deterministic)
        raise ValueError(f"Unknown fidelity level: {fidelity_level}")

    def predict_with_uncertainty(
        self, x: jax.Array, num_samples: int = 10, fidelity_level: str = "adaptive"
    ) -> dict[str, jax.Array]:
        """
        Predict with Monte Carlo uncertainty estimation.

        Args:
            x: Input tensor
            num_samples: Number of MC samples
            fidelity_level: Type of fidelity to use

        Returns:
            Dictionary with prediction statistics
        """
        predictions = []
        uncertainties = []

        for _ in range(num_samples):
            result = self(x, fidelity_level=fidelity_level, deterministic=False)
            pred = result.get("prediction")
            if pred is None:
                pred = result.get("low_fidelity_pred")
            if pred is not None:
                predictions.append(pred)

            uncertainty = result.get("uncertainty_estimate")
            if uncertainty is not None:
                uncertainties.append(uncertainty)

        # Compute statistics
        predictions_stack = jnp.stack(predictions)
        uncertainties_stack = jnp.stack(uncertainties)

        mean_prediction = jnp.mean(predictions_stack, axis=0)
        epistemic_uncertainty = jnp.std(predictions_stack, axis=0)
        aleatoric_uncertainty = jnp.mean(uncertainties_stack, axis=0)
        total_uncertainty = jnp.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)

        return {
            "mean": mean_prediction,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "total_uncertainty": total_uncertainty,
            "predictions": predictions_stack,
            "num_samples": jnp.array(num_samples),
        }

    def adaptive_prediction(
        self,
        x: jax.Array,
        uncertainty_threshold: float = 0.1,  # noqa: ARG002 - prediction interface accepts an uncertainty threshold
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        """
        Adaptive prediction with automatic fidelity selection.

        Args:
            x: Input tensor
            uncertainty_threshold: Threshold for fidelity selection

        Returns:
            Tuple of (predictions, uncertainties, info_dict)
        """
        # Get adaptive results — adaptive inference uses posterior means.
        result = self._adaptive_forward(x, deterministic=True)

        predictions = result["prediction"]
        uncertainties = result["uncertainty_estimate"]

        info = {
            "low_fidelity_count": result["low_fidelity_count"],
            "high_fidelity_count": result["high_fidelity_count"],
            "total_samples": result["total_samples"],
        }

        return predictions, uncertainties, info

    def _fusion_prediction(self, x: jax.Array, *, deterministic: bool | None = None) -> jax.Array:
        """Internal fusion prediction method for test compatibility."""
        result = self._adaptive_forward(x, deterministic=deterministic)
        return result["prediction"]


class ProbabilisticPINN(nnx.Module):
    """
    Single-fidelity Probabilistic PINN with uncertainty quantification.

    Standard PINN with Bayesian layers for uncertainty estimation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] | list[int] = (64, 64, 64),
        use_bayesian: bool = True,
        physics_loss_weight: float = 1.0,
        uncertainty_weight: float = 0.1,
        deterministic: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize probabilistic PINN.

        ``deterministic`` ships as ``False`` so the module is in sampling
        mode by default; flip via the NNX inference-mode toggle to disable
        posterior sampling globally (matches :class:`nnx.Dropout`).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bayesian = use_bayesian
        self.physics_loss_weight = physics_loss_weight
        self.uncertainty_weight = uncertainty_weight
        self.deterministic = deterministic

        if rngs is None:
            raise ValueError(
                "rngs is required; pass a caller-owned nnx.Rngs bundle to "
                "avoid hidden fixed-seed Monte Carlo paths."
            )
        self.rngs = rngs

        # Build network
        layers_temp = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            if use_bayesian:
                layer = BayesianLinear(prev_dim, hidden_dim, rngs=rngs)
            else:
                layer = nnx.Linear(prev_dim, hidden_dim, rngs=rngs)
            layers_temp.append(layer)
            prev_dim = hidden_dim
        # Assignment outside the loop avoids the NNX hazard of rebinding
        # ``self.layers`` on every iteration (Rule 0).
        self.layers = nnx.List(layers_temp)

        # Output layer
        if use_bayesian:
            self.output_layer = BayesianLinear(prev_dim, output_dim, rngs=rngs)
        else:
            self.output_layer = nnx.Linear(prev_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool | None = None) -> jax.Array:
        """Forward pass through probabilistic PINN using ``self.rngs``.

        ``deterministic`` per-call override mirrors :class:`nnx.Dropout`;
        falls back to ``self.deterministic`` when ``None``.
        """
        return self._forward(x, rngs=self.rngs, deterministic=deterministic)

    def _forward(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
        deterministic: bool | None = None,
    ) -> jax.Array:
        """Forward pass routing every Bayesian-layer sample through ``rngs``."""
        h = x
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                h = nnx.tanh(layer(h, deterministic=deterministic, rngs=rngs))
            else:
                h = nnx.tanh(layer(h))
        if isinstance(self.output_layer, BayesianLinear):
            return self.output_layer(h, deterministic=deterministic, rngs=rngs)
        return self.output_layer(h)

    def kl_divergence(self) -> jax.Array:
        """Sum the per-layer KL divergences for every Bayesian layer.

        Returns ``jnp.array(0.0)`` when ``use_bayesian=False`` (no
        variational parameters → vacuous KL).
        """
        total = jnp.array(0.0)
        if not self.use_bayesian:
            return total
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                total = total + layer.kl_divergence()
        if isinstance(self.output_layer, BayesianLinear):
            total = total + self.output_layer.kl_divergence()
        return total

    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs,
        num_samples: int = 10,
        mode: PredictiveMode = PredictiveMode.PREDICTIVE,
    ) -> PredictiveDistribution:
        """Return a :class:`PredictiveDistribution` via MC sampling of the posterior.

        For deterministic models (``use_bayesian=False``) returns a single
        forward pass tagged with ``method="deterministic"`` and zero
        epistemic uncertainty per the platform contract.
        """
        coerced_mode = _coerce_predictive_mode(mode)

        if not self.use_bayesian:
            pred = self._forward(x, rngs=rngs, deterministic=True)
            samples = pred[None, ...]
            return PredictiveDistribution(
                mean=pred,
                samples=samples,
                variance=jnp.zeros_like(pred),
                epistemic=jnp.zeros_like(pred),
                metadata=(("method", "deterministic"), ("num_samples", 1)),
            )

        key = extract_rng_key(
            rngs, streams=_PINN_RNG_STREAMS, context="ProbabilisticPINN.predict_distribution"
        )
        sample_keys = jax.random.split(key, num_samples)
        # Force sampling for every MC draw regardless of the module's
        # deterministic attribute.
        samples = jnp.stack(
            [self._forward(x, rngs=nnx.Rngs(sample=sk), deterministic=False) for sk in sample_keys],
            axis=0,
        )
        mean = jnp.mean(samples, axis=0)
        variance = jnp.var(samples, axis=0)
        quantiles = {
            0.025: jnp.quantile(samples, 0.025, axis=0),
            0.5: jnp.quantile(samples, 0.5, axis=0),
            0.975: jnp.quantile(samples, 0.975, axis=0),
        }
        return PredictiveDistribution(
            mean=mean,
            samples=samples,
            variance=variance,
            epistemic=variance,
            quantiles=quantiles,
            metadata=(("method", coerced_mode.value), ("num_samples", int(num_samples))),
        )

    def boundary_coverage(
        self,
        boundary_x: jax.Array,
        *,
        boundary_value: float | jax.Array | Callable[[jax.Array], jax.Array] = 0.0,
        rngs: nnx.Rngs,
        num_samples: int = 32,
        credible_level: float = 0.95,
    ) -> dict[str, jax.Array]:
        """Check that the predictive distribution covers a Dirichlet BC value.

        Calibration contract for a B-PINN (Yang/Meng/Karniadakis 2021): a
        well-trained posterior should produce a predictive whose MEAN matches
        the boundary value and whose credible interval COVERS it at the boundary
        points (low predictive std + correct mean). This method draws
        ``num_samples`` posterior predictive samples at ``boundary_x`` and
        reports the agreement.

        Args:
            boundary_x: Boundary coordinates ``x_b`` to evaluate.
            boundary_value: Target Dirichlet value ``g``; constant or callable.
            rngs: Caller-owned RNG bundle for posterior sampling.
            num_samples: Number of posterior predictive draws.
            credible_level: Central credible level for the interval check.

        Returns:
            Mapping with ``mean_abs_error`` (mean ``|E[u] - g|`` over boundary
            points), ``coverage`` (fraction of boundary points whose central
            credible interval contains ``g``), and ``predictive_std`` (mean
            predictive standard deviation at the boundary).
        """
        predictive = self.predict_distribution(boundary_x, rngs=rngs, num_samples=num_samples)
        target = _resolve_bc_target(boundary_value, boundary_x)
        target = jnp.broadcast_to(target, predictive.mean.shape)

        mean_abs_error = jnp.mean(jnp.abs(predictive.mean - target))

        samples = predictive.samples
        if samples is None:
            raise ValueError(
                "boundary_coverage requires predictive samples; predict_distribution returned none."
            )
        tail = (1.0 - credible_level) / 2.0
        lower = jnp.quantile(samples, tail, axis=0)
        upper = jnp.quantile(samples, 1.0 - tail, axis=0)
        covered = (target >= lower) & (target <= upper)
        coverage = jnp.mean(covered.astype(jnp.float32))

        variance = predictive.variance
        predictive_std = jnp.mean(jnp.sqrt(variance)) if variance is not None else jnp.array(0.0)
        return {
            "mean_abs_error": mean_abs_error,
            "coverage": coverage,
            "predictive_std": predictive_std,
        }

    def loss_components(
        self,
        batch: Mapping[str, Any],
        *,
        rngs: nnx.Rngs,
        objective: ObjectiveConfig,
    ) -> UQLossComponents:
        """Compute the per-batch UQ loss components for variational training.

        Required batch fields: ``x``, ``y``. Optional fields used when
        present: ``pde_residual_fn`` (PDE residual callable),
        ``boundary_conditions`` (mapping with ``value``/``weight``).
        """
        missing = [field for field in ("x", "y") if field not in batch]
        if missing:
            raise ValueError(f"batch missing required field(s): {missing!r}")
        x = batch["x"]
        y = batch["y"]
        pde_residual_fn = batch.get("pde_residual_fn")
        boundary_conditions = batch.get("boundary_conditions")

        # Sampling-forward for variational loss; deterministic=False forces
        # posterior sampling regardless of the module's inference-mode flag.
        y_pred = self._forward(x, rngs=rngs, deterministic=False)

        data = jnp.mean((y_pred - y) ** 2)
        physics_residual = None
        if pde_residual_fn is not None:
            residual = pde_residual_fn(x, y_pred)
            physics_residual = jnp.mean(residual**2)
        boundary = None
        if boundary_conditions is not None:
            # B-PINN BC likelihood: score the residual on the dedicated
            # boundary points (Yang/Meng/Karniadakis 2021), sampling the
            # posterior so the BC term enters the variational objective.
            def bc_model(coords: jax.Array) -> jax.Array:
                return self._forward(coords, rngs=rngs, deterministic=False)

            boundary = _boundary_loss_term(x, y_pred, boundary_conditions, bc_model)
        kl = self.kl_divergence()

        return UQLossComponents.from_components(
            config=objective,
            data=data,
            physics_residual=physics_residual,
            boundary=boundary,
            kl=kl,
            metadata=(("source", "probabilistic_pinn"),),
        )

    def negative_elbo(
        self,
        batch: Mapping[str, Any],
        *,
        rngs: nnx.Rngs,
        objective: ObjectiveConfig,
    ) -> UQLossComponents:
        """Return loss components with the ``negative_elbo`` field populated.

        The total is unchanged; ``negative_elbo`` is set to ``total`` so
        downstream code can read ``components.negative_elbo`` without
        recomputing the weight-driven sum.
        """
        base = self.loss_components(batch, rngs=rngs, objective=objective)
        return dataclasses.replace(base, negative_elbo=base.total)

    def predict_with_uncertainty(self, x: jax.Array, num_samples: int = 10) -> dict[str, jax.Array]:
        """
        Predict with uncertainty estimation for single-fidelity PINN.

        Args:
            x: Input tensor
            num_samples: Number of MC samples for uncertainty

        Returns:
            Dictionary with prediction statistics
        """
        if not self.use_bayesian:
            # Deterministic call — posterior mean only, zero uncertainty.
            pred = self(x, deterministic=True)
            return {
                "mean": pred,
                "std": jnp.zeros_like(pred),
                "total_uncertainty": jnp.zeros(pred.shape[0]),
            }

        # Monte Carlo sampling for Bayesian networks.
        predictions = []
        for _ in range(num_samples):
            pred = self(x, deterministic=False)  # Force sampling
            predictions.append(pred)

        predictions_stack = jnp.stack(predictions)
        mean_prediction = jnp.mean(predictions_stack, axis=0)
        std_prediction = jnp.std(predictions_stack, axis=0)
        total_uncertainty = jnp.mean(std_prediction, axis=-1)

        # Calculate 95% confidence intervals (1.96 * std for normal distribution)
        confidence_95_lower = mean_prediction - 1.96 * std_prediction
        confidence_95_upper = mean_prediction + 1.96 * std_prediction

        return {
            "mean": mean_prediction,
            "std": std_prediction,
            "total_uncertainty": total_uncertainty,
            "confidence_95_lower": confidence_95_lower,
            "confidence_95_upper": confidence_95_upper,
        }

    def physics_loss(
        self,
        x: jax.Array,
        pde_residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
        boundary_conditions: dict[str, Any] | None = None,
    ) -> jax.Array:
        """
        Compute physics loss for PINN training.

        Args:
            x: Input coordinates
            pde_residual_fn: Function computing PDE residual
            boundary_conditions: Optional boundary conditions

        Returns:
            Physics loss scalar
        """
        # Forward prediction at posterior mean — physics loss should not
        # add posterior-sampling noise.
        y_pred = self(x, deterministic=True)

        # Compute PDE residual
        residual = pde_residual_fn(x, y_pred)
        physics_loss = jnp.mean(residual**2)

        # Add boundary-condition loss if provided. The BC residual follows the
        # B-PINN / deepxde form (Dirichlet/Neumann/Robin) and is evaluated at
        # dedicated boundary points when supplied; ``weight`` (default 1.0) is
        # the only multiplier — no hidden scale.
        if boundary_conditions is not None:
            bc_weight = boundary_conditions.get("weight", 1.0)

            def bc_model(coords: jax.Array) -> jax.Array:
                return self(coords, deterministic=True)

            bc_loss = _boundary_loss_term(x, y_pred, boundary_conditions, bc_model)
            physics_loss = physics_loss + bc_weight * bc_loss

        return physics_loss

    def robust_loss(
        self,
        x: jax.Array,
        y_true: jax.Array,
        noise_scale: float = 0.01,
    ) -> jax.Array:
        """
        Compute robust loss with noise perturbation.

        Args:
            x: Input coordinates
            y_true: True values
            noise_scale: Scale of adversarial noise

        Returns:
            Robust loss scalar
        """
        # Clean prediction at posterior mean.
        clean_pred = self(x, deterministic=True)

        # Noisy prediction; perturbation key comes from caller-owned rngs
        # (advancing the ``noise`` stream when present, falling back to
        # ``default``).
        key = extract_rng_key(self.rngs, streams=("noise", "default"), context="robust_loss noise")
        noise = jax.random.normal(key, x.shape) * noise_scale
        x_noisy = x + noise
        noisy_pred = self(x_noisy, deterministic=True)

        # Data loss
        data_loss = jnp.mean((clean_pred - y_true) ** 2)

        # Robustness penalty
        robustness_penalty = jnp.mean((clean_pred - noisy_pred) ** 2) / (noise_scale**2)

        return data_loss + self.uncertainty_weight * robustness_penalty


def _extract_prediction_array(model_output: jax.Array | dict[str, Any]) -> jax.Array:
    """Return a single prediction ``jax.Array`` from a PINN forward output.

    Single-fidelity PINNs return a plain ``jax.Array``; multifidelity PINNs
    return a dict keyed by ``prediction``, ``mean``, or
    ``low_fidelity_pred`` depending on fidelity level.
    """
    if not isinstance(model_output, dict):
        return model_output
    for key in ("prediction", "mean", "low_fidelity_pred"):
        value = model_output.get(key)
        if value is not None:
            return value
    raise ValueError("Could not extract prediction from model output dict")


class RobustPINNOptimizer(nnx.Module):
    """
    Robust optimizer for Physics-Informed Neural Networks.

    Provides robust optimization with uncertainty quantification,
    physics loss integration, and adaptive sampling strategies.
    """

    def __init__(self, model: ProbabilisticPINN | MultiFidelityPINN) -> None:
        """Initialize robust PINN optimizer.

        The optimizer owns no trainable state of its own and no hidden RNG;
        every stochastic step requires caller-owned ``rngs`` at the method
        boundary. Loss-component weights live in :class:`ObjectiveConfig`
        and are passed per call into :meth:`compute_loss_components`.
        """
        self.model = model

    def compute_loss_components(
        self,
        batch: Mapping[str, Any],
        *,
        rngs: nnx.Rngs,
        objective: ObjectiveConfig,
    ) -> UQLossComponents:
        """Compute UQ loss components for robust PINN training.

        Required batch fields: ``x``, ``y_true``. Optional fields:
        ``pde_residual_fn`` (callable producing per-point residuals),
        ``boundary_conditions`` (mapping with ``value``/``weight``),
        ``noise_scale`` (perturbation magnitude for the robustness penalty;
        defaults to ``0.01``).

        Component mapping:

        * ``data`` — supervised MSE between posterior-mean prediction and
          ``y_true``.
        * ``physics_residual`` — ``mean(pde_residual_fn(x, y_pred)**2)``
          when supplied.
        * ``boundary`` — Dirichlet/etc penalty when ``boundary_conditions``
          is supplied.
        * ``regularization`` — perturbation-based robustness penalty.
        * ``kl`` — ``self.model.kl_divergence()`` when the underlying model
          exposes a Bayesian KL.

        All component weights are applied by ``ObjectiveConfig`` inside
        :meth:`UQLossComponents.from_components`.
        """
        missing = [field for field in ("x", "y_true") if field not in batch]
        if missing:
            raise ValueError(f"batch missing required field(s): {missing!r}")
        x = batch["x"]
        y_true = batch["y_true"]
        pde_residual_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = batch.get(
            "pde_residual_fn"
        )
        boundary_conditions = batch.get("boundary_conditions")
        noise_scale = batch.get("noise_scale", 0.01)

        # Posterior-mean prediction for robust scoring.
        model_output = self.model(x, deterministic=True)
        y_pred = _extract_prediction_array(model_output)

        data = jnp.mean((y_pred - y_true) ** 2)

        physics_residual = None
        if pde_residual_fn is not None:
            residual = pde_residual_fn(x, y_pred)
            physics_residual = jnp.mean(residual**2)

        boundary = None
        if boundary_conditions is not None:
            boundary = self._compute_boundary_loss(x, y_pred, boundary_conditions)

        regularization = self._compute_robustness_penalty(x, noise_scale, rngs=rngs)

        kl: jax.Array | None = None
        kl_fn: Callable[[], jax.Array] | None = getattr(self.model, "kl_divergence", None)
        if kl_fn is not None and callable(kl_fn):
            kl = kl_fn()

        return UQLossComponents.from_components(
            config=objective,
            data=data,
            physics_residual=physics_residual,
            boundary=boundary,
            regularization=regularization,
            kl=kl,
            metadata=(("source", "robust_pinn_optimizer"),),
        )

    def _compute_robustness_penalty(
        self, x: jax.Array, noise_scale: float, *, rngs: nnx.Rngs
    ) -> jax.Array:
        """Compute the perturbation-based robustness penalty.

        The penalty is the squared sensitivity of the posterior-mean
        prediction to a Gaussian perturbation drawn from caller-owned
        ``rngs``; no hidden seed.
        """
        key = extract_rng_key(
            rngs, streams=("noise", "default"), context="robustness penalty noise"
        )
        noise = jax.random.normal(key, x.shape) * noise_scale
        x_noisy = x + noise

        if hasattr(self.model, "predict_with_uncertainty"):
            clean_result = self.model.predict_with_uncertainty(x, num_samples=3)
            noisy_result = self.model.predict_with_uncertainty(x_noisy, num_samples=3)
            clean_pred = clean_result["mean"]
            noisy_pred = noisy_result["mean"]
        else:
            clean_pred = _extract_prediction_array(self.model(x))
            noisy_pred = _extract_prediction_array(self.model(x_noisy))

        return jnp.mean((clean_pred - noisy_pred) ** 2) / (noise_scale**2)

    def _compute_boundary_loss(
        self,
        x: jax.Array,
        y_pred: jax.Array,
        boundary_conditions: dict[str, Any],
    ) -> jax.Array:
        """Compute the raw boundary-condition MSE for the wrapped model.

        Delegates to :func:`compute_boundary_residual` so Dirichlet, Neumann
        and Robin BCs share one reference-cited implementation (B-PINN /
        deepxde). The BC is evaluated at the dedicated ``boundary_x`` points
        when supplied; otherwise on the interior batch (legacy behaviour). The
        returned MSE carries NO hidden multiplier — the ``boundary_weight`` of
        :class:`ObjectiveConfig` is the single scaling source.

        Args:
            x: Interior collocation coordinates (fallback boundary points).
            y_pred: Posterior-mean prediction at ``x``.
            boundary_conditions: BC specification (see
                :func:`compute_boundary_residual`).

        Returns:
            Scalar boundary-condition mean squared residual.
        """

        def bc_model(coords: jax.Array) -> jax.Array:
            return _extract_prediction_array(self.model(coords, deterministic=True))

        return _boundary_loss_term(x, y_pred, boundary_conditions, bc_model)

    def uncertainty_guided_sampling(
        self,
        x_candidates: jax.Array,
        num_samples: int,
        *,
        rngs: nnx.Rngs,
        uncertainty_threshold: float = 0.1,  # noqa: ARG002 - sampling interface accepts an uncertainty threshold
    ) -> jax.Array:
        """Select candidate points by predictive uncertainty.

        ``rngs`` is required keyword-only — there is no instance-stored
        fallback. When the model does not expose
        ``predict_with_uncertainty``, the active-learning step degrades to
        a caller-keyed uniform selection rather than a silent fixed seed.
        """
        if hasattr(self.model, "predict_with_uncertainty"):
            pred_result = self.model.predict_with_uncertainty(x_candidates, num_samples=10)
            uncertainties = pred_result.get(
                "total_uncertainty",
                pred_result.get("std", jnp.zeros(x_candidates.shape[0])),
            )
        else:
            key = extract_rng_key(
                rngs,
                streams=("active_learning", "default"),
                context="active learning random selection",
            )
            uncertainties = jax.random.uniform(key, (x_candidates.shape[0],))

        if uncertainties.ndim > 1:
            uncertainties = jnp.mean(uncertainties, axis=-1)

        top_indices = jnp.argsort(uncertainties)[-num_samples:]
        return x_candidates[top_indices]


# Factory functions for easy creation
def create_multifidelity_pinn(
    input_dim: int,
    output_dim: int,
    config_dict: dict[str, Any] | None = None,
    rngs: nnx.Rngs | None = None,
) -> MultiFidelityPINN:
    """
    Create MultiFidelityPINN with proper configuration handling.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        config_dict: Optional configuration dictionary
        rngs: Random number generator state

    Returns:
        Configured MultiFidelityPINN instance
    """
    if rngs is None:
        raise ValueError(
            "rngs is required; pass a caller-owned nnx.Rngs bundle so the "
            "factory does not seed a hidden fixed-key Monte Carlo path."
        )

    # Create configuration from dict if provided. ``MultiFidelityConfig`` is a
    # frozen dataclass, so overrides are applied immutably via
    # ``dataclasses.replace`` rather than in-place ``setattr``.
    if config_dict is not None:
        valid_fields = {f.name for f in dataclasses.fields(MultiFidelityConfig)}
        overrides = {key: value for key, value in config_dict.items() if key in valid_fields}

        # Normalise a nested ``low_fidelity`` dict into a ``FidelityConfig``.
        low_fidelity = overrides.get("low_fidelity")
        if isinstance(low_fidelity, dict):
            overrides["low_fidelity"] = FidelityConfig(**low_fidelity)

        config = dataclasses.replace(MultiFidelityConfig(), **overrides)
    else:
        config = MultiFidelityConfig()

    return MultiFidelityPINN(input_dim, output_dim, config=config, rngs=rngs)


def create_probabilistic_pinn(
    input_dim: int,
    output_dim: int,
    hidden_layers: list[int] | None = None,
    use_bayesian: bool = True,
    rngs: nnx.Rngs | None = None,
) -> ProbabilisticPINN:
    """Create a probabilistic PINN with default configuration."""
    if hidden_layers is None:
        hidden_layers = [64, 64, 64]
    if rngs is None:
        raise ValueError(
            "rngs is required; pass a caller-owned nnx.Rngs bundle so the "
            "factory does not seed a hidden fixed-key Monte Carlo path."
        )

    return ProbabilisticPINN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_layers,
        use_bayesian=use_bayesian,
        rngs=rngs,
    )


# Update factory functions to include RobustPINNOptimizer
def create_robust_pinn_optimizer(
    model: ProbabilisticPINN | MultiFidelityPINN,
) -> RobustPINNOptimizer:
    """Create a robust PINN optimizer.

    Loss-component weights are no longer instance-stored; pass an
    :class:`ObjectiveConfig` per call into
    :meth:`RobustPINNOptimizer.compute_loss_components` and supply
    caller-owned ``rngs`` for every stochastic step.
    """
    return RobustPINNOptimizer(model=model)
