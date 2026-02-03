"""Unified operator network interface for easy experimentation.

This module provides a common interface for different neural operator
architectures to enable easy switching and comparison between different
operator types.
"""

from __future__ import annotations

from typing import Any

import jax  # noqa: TC002  # Used in runtime method signature (__call__ -> jax.Array)
from flax import nnx

from opifex.neural.operators.deeponet import DeepONet
from opifex.neural.operators.fno import FourierNeuralOperator


# Type-safe imports for enhanced operators


def _lazy_import_fourier_enhanced_deeponet():
    """Lazy import for FourierEnhancedDeepONet."""
    try:
        from opifex.neural.operators.deeponet import FourierEnhancedDeepONet

        return FourierEnhancedDeepONet
    except ImportError as e:
        raise ValueError(
            f"FourierEnhancedDeepONet is not available. Import error: {e}"
        ) from e


def _lazy_import_adaptive_deeponet():
    """Lazy import for AdaptiveDeepONet."""
    try:
        from opifex.neural.operators.deeponet import AdaptiveDeepONet

        return AdaptiveDeepONet
    except ImportError as e:
        raise ValueError(f"AdaptiveDeepONet is not available. Import error: {e}") from e


def _lazy_import_graph_neural_operator():
    """Lazy import for GraphNeuralOperator."""
    try:
        from opifex.neural.operators.graph import GraphNeuralOperator

        return GraphNeuralOperator
    except ImportError as e:
        raise ValueError(
            f"GraphNeuralOperator is not available. Import error: {e}"
        ) from e


def _lazy_import_multiphysics_deeponet():
    """Lazy import for MultiPhysicsDeepONet."""
    try:
        from opifex.neural.operators.deeponet import MultiPhysicsDeepONet

        return MultiPhysicsDeepONet
    except ImportError:
        return None  # Optional dependency


def _lazy_import_latent_neural_operator():
    """Lazy import for LatentNeuralOperator."""
    try:
        from opifex.neural.operators.specialized.latent import LatentNeuralOperator

        return LatentNeuralOperator
    except ImportError as e:
        raise ValueError(
            f"LatentNeuralOperator is not available. Import error: {e}"
        ) from e


class OperatorNetwork(nnx.Module):
    """Unified interface for different operator network types.

    This class provides a common interface for different neural operator
    architectures (FNO, DeepONet, etc.) to enable easy experimentation
    and comparison.
    """

    def __init__(
        self,
        operator_type: str,
        config: dict[str, Any],
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize operator network.

        Args:
            operator_type: Type of operator
                ('fno', 'deeponet', 'fourier_deeponet', 'adaptive_deeponet', etc.)
            config: Configuration dictionary for the operator
            rngs: Random number generators
        """
        super().__init__()
        self.operator_type = operator_type
        self.config = config

        # Get activation function - ensure it's callable
        activation = self._get_activation(config.get("activation", nnx.gelu))

        # Initialize operator based on type
        self.operator = self._create_operator(operator_type, config, activation, rngs)

    def _get_activation(self, activation):
        """Get activation function from string or callable."""
        if isinstance(activation, str):
            # Handle string activation names
            activation_mapping = {
                "relu": nnx.relu,
                "gelu": nnx.gelu,
                "tanh": nnx.tanh,
                "sigmoid": nnx.sigmoid,
                "swish": nnx.swish,
                "silu": nnx.silu,
            }
            return activation_mapping.get(activation, nnx.gelu)
        return activation

    def _create_operator(
        self, operator_type: str, config: dict[str, Any], activation, rngs: nnx.Rngs
    ) -> Any:
        """Create operator instance based on type."""
        if operator_type == "fno":
            return self._create_fno_operator(config, activation, rngs)
        if operator_type == "deeponet":
            return self._create_deeponet_operator(config, activation, rngs)
        if operator_type == "fourier_deeponet":
            return self._create_fourier_deeponet_operator(config, activation, rngs)
        if operator_type == "adaptive_deeponet":
            return self._create_adaptive_deeponet_operator(config, activation, rngs)
        if operator_type == "gno":
            return self._create_gno_operator(config, activation, rngs)
        if operator_type == "latent":
            return self._create_latent_operator(config, activation, rngs)
        raise ValueError(f"Unknown operator type: {operator_type}")

    def _create_fno_operator(
        self, config: dict[str, Any], activation, rngs: nnx.Rngs
    ) -> Any:
        """Create FNO operator."""
        return FourierNeuralOperator(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            hidden_channels=config["hidden_channels"],
            modes=config["modes"],
            num_layers=config["num_layers"],
            activation=activation,
            factorization_type=config.get("factorization_type"),
            factorization_rank=config.get("factorization_rank"),
            use_mixed_precision=config.get("use_mixed_precision", False),
            rngs=rngs,
        )

    def _create_deeponet_operator(
        self, config: dict[str, Any], activation, rngs: nnx.Rngs
    ) -> Any:
        """Create DeepONet operator."""
        if config.get("enhanced", False):
            MultiPhysicsDeepONet = _lazy_import_multiphysics_deeponet()
            if MultiPhysicsDeepONet is not None:
                return MultiPhysicsDeepONet(
                    branch_input_dim=config["branch_input_dim"],
                    trunk_input_dim=config["trunk_input_dim"],
                    branch_hidden_dims=config["branch_hidden_dims"],
                    trunk_hidden_dims=config["trunk_hidden_dims"],
                    latent_dim=config["latent_dim"],
                    num_physics_systems=config.get("num_physics_systems", 1),
                    use_attention=config.get("use_attention", True),
                    attention_heads=config.get("attention_heads", 8),
                    physics_constraints=config.get("physics_constraints"),
                    sensor_optimization=config.get("sensor_optimization", True),
                    num_sensors=config.get("num_sensors"),
                    activation=activation,
                    rngs=rngs,
                )
        # Build sizes from input/hidden dims for new API
        branch_sizes = (
            [config["branch_input_dim"]]
            + config["branch_hidden_dims"]
            + [config["latent_dim"]]
        )
        trunk_sizes = (
            [config["trunk_input_dim"]]
            + config["trunk_hidden_dims"]
            + [config["latent_dim"]]
        )
        return DeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            activation=activation,
            rngs=rngs,
        )

    def _create_fourier_deeponet_operator(
        self, config: dict[str, Any], activation, rngs: nnx.Rngs
    ) -> Any:
        """Create Fourier-enhanced DeepONet operator."""
        FourierEnhancedDeepONet = _lazy_import_fourier_enhanced_deeponet()

        # Build sizes from input/hidden dims for new API
        branch_sizes = (
            [config["branch_input_dim"]]
            + config["branch_hidden_dims"]
            + [config["latent_dim"]]
        )
        trunk_sizes = (
            [config["trunk_input_dim"]]
            + config["trunk_hidden_dims"]
            + [config["latent_dim"]]
        )
        return FourierEnhancedDeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            fourier_modes=config.get("fourier_modes", 16),
            use_spectral_branch=config.get("use_spectral_branch", True),
            use_spectral_trunk=config.get("use_spectral_trunk", False),
            activation=activation,
            rngs=rngs,
        )

    def _create_adaptive_deeponet_operator(
        self, config: dict[str, Any], activation, rngs: nnx.Rngs
    ) -> Any:
        """Create adaptive DeepONet operator."""
        AdaptiveDeepONet = _lazy_import_adaptive_deeponet()

        return AdaptiveDeepONet(
            branch_input_dim=config["branch_input_dim"],
            trunk_input_dim=config["trunk_input_dim"],
            base_latent_dim=config["base_latent_dim"],
            num_resolution_levels=config.get("num_resolution_levels", 3),
            adaptive_latent_scaling=config.get("adaptive_latent_scaling", True),
            use_residual_connections=config.get("use_residual_connections", True),
            activation=activation,
            rngs=rngs,
        )

    def _create_gno_operator(
        self, config: dict[str, Any], activation, rngs: nnx.Rngs
    ) -> Any:
        """Create Graph Neural Operator."""
        GraphNeuralOperator = _lazy_import_graph_neural_operator()

        return GraphNeuralOperator(
            node_dim=config["node_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            edge_dim=config.get("edge_dim", 0),
            activation=activation,
            rngs=rngs,
        )

    def _create_latent_operator(
        self, config: dict[str, Any], activation, rngs: nnx.Rngs
    ) -> Any:
        """Create Latent Neural Operator."""
        LatentNeuralOperator = _lazy_import_latent_neural_operator()

        return LatentNeuralOperator(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            latent_dim=config["latent_dim"],
            num_latent_tokens=config["num_latent_tokens"],
            num_attention_heads=config.get("num_attention_heads", 8),
            num_encoder_layers=config.get("num_encoder_layers", 4),
            num_decoder_layers=config.get("num_decoder_layers", 4),
            physics_constraints=config.get("physics_constraints"),
            rngs=rngs,
        )

    def __call__(self, *args, **kwargs) -> jax.Array:
        """Apply the operator network.

        Args:
            *args: Positional arguments (depends on operator type)
            **kwargs: Keyword arguments

        Returns:
            Output tensor
        """
        return self.operator(*args, **kwargs)
