"""Activation functions optimized for scientific neural networks.

This module provides a comprehensive collection of activation functions
specifically optimized for scientific machine learning applications.
All functions are fully compatible with Flax NNX patterns and JAX transformations.

MODERNIZATION APPLIED:
- Full Flax NNX compliance with proper type annotations
- Enhanced activation function selection with error handling
- Optimized implementations for scientific computing
- Support for both standard and specialized activation patterns
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


# Global registry for custom activation functions
_CUSTOM_ACTIVATIONS: dict[str, Callable] = {}


def get_activation(name: str | Callable) -> Any:
    """Get activation function by name or return function if already callable.

    Args:
        name: Name of the activation function (case-insensitive) or callable function

    Returns:
        JAX activation function or callable

    Raises:
        ValueError: If activation function is not found
    """
    # If already a callable function, return it directly
    if callable(name):
        return name

    # If not a string, convert to string
    if not isinstance(name, str):
        raise TypeError(f"Activation must be string or callable, got {type(name)}")

    name_lower = name.lower().strip()

    # Dictionary-based lookup to reduce complexity
    activation_functions = _get_activation_map()

    if name_lower in activation_functions:
        return activation_functions[name_lower]

    raise ValueError(f"Unknown activation function: {name}")


def _get_activation_map() -> dict[str, Any]:
    """Get the activation function mapping dictionary."""
    return {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
        "softmax": jax.nn.softmax,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "swish": jax.nn.silu,  # swish is alias for silu
        "elu": jax.nn.elu,
        "leaky_relu": jax.nn.leaky_relu,
        "relu6": jax.nn.relu6,
        "hard_tanh": jax.nn.hard_tanh,
        "log_sigmoid": jax.nn.log_sigmoid,
        "softplus": jax.nn.softplus,
        "mish": mish,
        "snake": snake_activation,
        "gaussian": gaussian_activation,
        "normalized_tanh": normalized_tanh,
        "soft_exponential": soft_exponential,
        "hard_swish": jax.nn.hard_swish,
        "hard_sigmoid": jax.nn.hard_sigmoid,
        "prelu": jnp.maximum,  # Simplified version
        "celu": jax.nn.celu,
        "selu": jax.nn.selu,
        "linear": lambda x: x,
        "identity": lambda x: x,
        "none": lambda x: x,
        **_CUSTOM_ACTIVATIONS,  # Include custom registered activations
    }


def list_activations() -> list[str]:
    """List all available activation functions.

    Returns:
        List of activation function names

    Examples:
        >>> activations = list_activations()
        >>> print(f"Available activations: {', '.join(activations)}")
    """
    # Standard activations from jax.nn
    standard_activations = [
        "gelu",
        "relu",
        "tanh",
        "sigmoid",
        "silu",
        "swish",
        "leaky_relu",
        "elu",
        "softplus",
    ]

    # Custom scientific activations
    custom_activations = [
        "mish",
        "snake",
        "gaussian",
        "normalized_tanh",
        "soft_exponential",
    ]

    # Registered custom activations
    registered_activations = list(_CUSTOM_ACTIVATIONS.keys())

    return standard_activations + custom_activations + registered_activations


def register_activation(name: str, func: Callable) -> None:
    """Register a custom activation function.

    Args:
        name: Name of the activation function
        func: The activation function (should accept and return JAX arrays)

    Examples:
        >>> def my_activation(x):
        ...     return x ** 3
        >>> register_activation("cubic", my_activation)
        >>> cubic_fn = get_activation("cubic")
    """
    if not callable(func):
        raise TypeError(f"Activation function must be callable, got {type(func)}")

    name_lower = name.lower().strip()
    if not name_lower:
        raise ValueError("Activation name cannot be empty")

    _CUSTOM_ACTIVATIONS[name_lower] = func


def mish(x: jax.Array) -> jax.Array:
    """Mish activation function: x * tanh(softplus(x)).

    Mish is a self-gated activation function that has shown excellent
    performance in deep networks. It's smooth and non-monotonic.

    Mathematical definition: f(x) = x * tanh(ln(1 + exp(x)))

    Args:
        x: Input array

    Returns:
        Output array with Mish activation applied

    Note:
        This implementation uses softplus(x) = ln(1 + exp(x)) for numerical stability.
    """
    return x * jnp.tanh(jax.nn.softplus(x))


def snake_activation(x: jax.Array, a: float = 1.0) -> jax.Array:
    """Snake activation function: x + sin²(αx)/α.

    Snake activation has been shown to work well for certain scientific
    applications, particularly those involving periodic patterns.

    Mathematical definition: f(x) = x + (1/α) * sin²(αx)

    Args:
        x: Input array
        a: Frequency parameter (default: 1.0)

    Returns:
        Output array with Snake activation applied

    Note:
        The frequency parameter α controls the oscillation frequency.
        Higher values create more frequent oscillations.
    """
    if a <= 0:
        raise ValueError(f"Frequency parameter 'a' must be positive, got {a}")

    return x + jnp.sin(a * x) ** 2 / a


def gaussian_activation(x: jax.Array, sigma: float = 1.0) -> jax.Array:
    """Gaussian activation function: exp(-x²/(2σ²)).

    Gaussian activation can be useful for radial basis function networks
    and certain scientific applications where localized responses are desired.

    Mathematical definition: f(x) = exp(-x²/(2σ²))

    Args:
        x: Input array
        sigma: Standard deviation parameter (default: 1.0)

    Returns:
        Output array with Gaussian activation applied

    Note:
        The σ parameter controls the width of the Gaussian.
        Smaller values create sharper peaks.
    """
    if sigma <= 0:
        raise ValueError(f"Sigma parameter must be positive, got {sigma}")

    return jnp.exp(-0.5 * (x / sigma) ** 2)


# Specialized activation functions for scientific computing
def normalized_tanh(x: jax.Array) -> jax.Array:
    """Normalized tanh activation: 1.7159 * tanh(2x/3).

    This is a normalized version of tanh that has unit variance
    for normalized inputs, which can help with training stability.

    Args:
        x: Input array

    Returns:
        Output array with normalized tanh applied
    """
    return 1.7159 * jnp.tanh(2.0 * x / 3.0)


def soft_exponential(x: jax.Array, alpha: float = 0.0) -> jax.Array:
    """Soft exponential activation function.

    This is a parameterized activation that interpolates between
    different behaviors based on the alpha parameter.

    Mathematical definition:
    - If α < 0: -ln(1 - α(x + α)) / α
    - If α = 0: x
    - If α > 0: (exp(αx) - 1) / α + α

    Args:
        x: Input array
        alpha: Shape parameter

    Returns:
        Output array with soft exponential applied
    """
    if alpha == 0:
        return x
    if alpha < 0:
        return -jnp.log(1 - alpha * (x + alpha)) / alpha
    return (jnp.exp(alpha * x) - 1) / alpha + alpha


def get_derivative_activation(name: str) -> Any:
    """Get the derivative of an activation function.

    This is useful for implementations that need explicit derivatives
    rather than relying on automatic differentiation.

    Args:
        name: Name of the activation function

    Returns:
        Derivative function of the specified activation

    Raises:
        ValueError: If activation name is not recognized or derivative not available
    """
    name_lower = name.lower().strip()

    if name_lower == "relu":
        return lambda x: jnp.asarray(x > 0)
    if name_lower == "tanh":
        return lambda x: 1 - jnp.tanh(x) ** 2
    if name_lower == "sigmoid":
        return lambda x: jax.nn.sigmoid(x) * (1 - jax.nn.sigmoid(x))
    if name_lower == "leaky_relu":
        return lambda x: jnp.where(x > 0, 1.0, 0.01)

    raise ValueError(
        f"Derivative not implemented for activation: '{name}'. "
        f"Consider using JAX automatic differentiation instead."
    )
