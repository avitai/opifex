"""Shared dense-stack module for PINN architectures.

Many PINN networks share the same fully-connected backbone: a stack of
:class:`flax.nnx.Linear` layers where the configured activation is applied
after every layer except the last. This module extracts that single pattern so
it is implemented once and composed (never inherited) by the individual
networks.

The stack is dtype-aware. When ``compute_dtype`` is provided the input is cast
to the compute dtype, the linear layers are constructed with the matching
``dtype``/``param_dtype`` and the output is cast back to the compute dtype.
When it is ``None`` the layers are built without any dtype overrides and no
casts are applied, reproducing the plain-precision behaviour exactly.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.neural.dtypes import as_compute_array, canonicalize_dtype


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class DenseStack(nnx.Module):
    """Fully-connected backbone shared by PINN networks.

    Builds ``len(hidden_dims) + 1`` linear layers mapping
    ``input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1] -> output_dim`` and
    applies ``activation`` after every layer except the final one.

    Attributes:
        layers: ``nnx.List`` of the linear layers (length ``len(hidden_dims) + 1``).
        activation: Activation applied after each non-final layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: Callable[[Array], Array] = jnp.tanh,
        compute_dtype: Any | None = None,
        param_dtype: Any | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the dense stack.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            hidden_dims: Hidden layer widths, applied in order.
            activation: Activation function applied after each hidden layer.
            compute_dtype: Computation dtype. ``None`` keeps the default
                precision and skips input/output casts.
            param_dtype: Parameter storage dtype. Ignored when
                ``compute_dtype`` is ``None``.
            rngs: Random number generators for parameter initialisation.
        """
        super().__init__()
        self.activation = activation
        self._compute_dtype = None if compute_dtype is None else canonicalize_dtype(compute_dtype)
        resolved_param_dtype = None if compute_dtype is None else canonicalize_dtype(param_dtype)

        dims = [input_dim, *hidden_dims, output_dim]
        linear_kwargs: dict[str, Any] = {}
        if self._compute_dtype is not None:
            linear_kwargs = {
                "dtype": self._compute_dtype,
                "param_dtype": resolved_param_dtype,
            }

        layers = [
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs, **linear_kwargs)
            for i in range(len(dims) - 1)
        ]
        # nnx.List keeps the layers registered as submodules (FLAX NNX 0.12.0).
        self.layers = nnx.List(layers)

    def __call__(self, x: Array) -> Array:
        """Run the dense stack.

        Args:
            x: Input array of shape ``(..., input_dim)``.

        Returns:
            Output array of shape ``(..., output_dim)``.
        """
        h = x if self._compute_dtype is None else as_compute_array(x, self._compute_dtype)

        layers = list(self.layers)
        for layer in layers[:-1]:
            h = self.activation(layer(h))
        out = layers[-1](h)

        return out if self._compute_dtype is None else out.astype(self._compute_dtype)
