"""Activation functions optimized for scientific neural networks.

This module provides a full collection of activation functions
specifically optimized for scientific machine learning applications.
All functions are fully compatible with Flax NNX patterns and JAX transformations.

MODERNIZATION APPLIED:
- Full Flax NNX compliance with proper type annotations
- Enhanced activation function selection with error handling
- Optimized implementations for scientific computing
- Support for both standard and specialized activation patterns
"""

from __future__ import annotations

from typing import Any, Final, TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


# Global registry for custom activation functions
_CUSTOM_ACTIVATIONS: dict[str, Callable] = {}


_PARAMETRIC_ELSEWHERE: Final = {"prelu": "flax.nnx.PReLU, whose negative slope is learned"}
"""Activations that carry parameters, and the module that owns each."""


def get_activation(name: str | Callable) -> Any:
    """Get activation function by name or return function if already callable.

    Args:
        name: Name of the activation function (case-insensitive) or callable function

    Returns:
        JAX activation function or callable

    Raises:
        ValueError: If no activation of that name is registered, or the name is one that
            carries parameters and so lives in a module.
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

    if name_lower in _PARAMETRIC_ELSEWHERE:
        raise ValueError(
            f"{name} learns its parameters, so it is a module, not a function here: "
            f"use {_PARAMETRIC_ELSEWHERE[name_lower]}"
        )
    raise ValueError(
        f"Unknown activation function: {name}; the names are "
        f"{', '.join(sorted(activation_functions))}"
    )


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
        "mish": jax.nn.mish,
        "snake": snake_activation,
        "gaussian": gaussian_activation,
        "normalized_tanh": normalized_tanh,
        "soft_exponential": soft_exponential,
        "hard_swish": jax.nn.hard_swish,
        "hard_sigmoid": jax.nn.hard_sigmoid,
        "celu": jax.nn.celu,
        "selu": jax.nn.selu,
        "linear": lambda x: x,
        "identity": lambda x: x,
        "none": lambda x: x,
        **_CUSTOM_ACTIVATIONS,  # Include custom registered activations
    }


def list_activations() -> list[str]:
    """The names :func:`get_activation` accepts, registered ones included.

    Returns:
        The names, sorted.

    Examples:
        >>> activations = list_activations()
        >>> print(f"Available activations: {', '.join(activations)}")
    """
    return sorted(_get_activation_map())


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
