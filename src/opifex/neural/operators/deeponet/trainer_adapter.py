"""DeepONet adapter for BasicTrainer compatibility.

DeepONet requires two separate inputs (branch, trunk) but BasicTrainer
expects models with a single input. This adapter bridges the gap.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from flax import nnx


if TYPE_CHECKING:
    import jax


class DeepONetTrainerAdapter(nnx.Module):
    """Wraps a DeepONet for single-input BasicTrainer compatibility.

    BasicTrainer calls models as ``model(x)`` with a single input.
    DeepONet requires ``model(branch_input, trunk_input)``.

    This adapter accepts a dict input with ``'branch'`` and ``'trunk'``
    keys and delegates to the underlying DeepONet.

    Example::

        deeponet = DeepONet(branch_sizes=[100, 64, 64, 32],
                            trunk_sizes=[2, 64, 64, 32], rngs=rngs)
        adapter = DeepONetTrainerAdapter(deeponet)

        x_train = {'branch': branch_data, 'trunk': trunk_data}
        trainer.train((x_train, y_train))
    """

    def __init__(self, model: nnx.Module) -> None:
        self.model = model

    def __call__(
        self,
        x: dict[str, jax.Array],
        *,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> jax.Array:
        """Forward pass with dict input.

        Args:
            x: Dict with 'branch' and 'trunk' keys containing the
               respective input arrays.
            deterministic: Whether to use deterministic mode.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            Operator output at evaluation locations.

        Raises:
            TypeError: If x is not a dict with required keys.
        """
        if not isinstance(x, dict):
            raise TypeError(
                f"DeepONetTrainerAdapter expects a dict input with 'branch' and "
                f"'trunk' keys, got {type(x).__name__}"
            )
        missing = {"branch", "trunk"} - set(x.keys())
        if missing:
            raise TypeError(
                f"DeepONetTrainerAdapter input dict missing keys: {missing}"
            )
        return self.model(  # pyright: ignore[reportCallIssue]
            x["branch"], x["trunk"], deterministic=deterministic, **kwargs
        )
