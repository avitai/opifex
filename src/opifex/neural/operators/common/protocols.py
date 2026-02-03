"""Common protocols for neural operators.

This module contains shared protocols and interfaces used across all neural operator
implementations.
"""

from typing import Any, Protocol

import jax


class CallableModule(Protocol):
    """Protocol for callable neural network modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> jax.Array:
        """Call the module with arbitrary arguments."""
        ...
