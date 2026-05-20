"""JAX-native :class:`PointwiseQuantileLoss` for conformal-residual training.

Mirrors the canonical PyTorch reference at
``../neuraloperator/neuralop/losses/data_losses.py::PointwiseQuantileLoss``
(Ma, Pitt, Azizzadenesheli, Anandkumar — TMLR 2024,
`arXiv:2402.01960 <https://arxiv.org/abs/2402.01960>`_). The loss is the
self-scaling pinball loss used to train a UQNO residual operator to
predict per-grid-point quantile widths.

Formula (matches the reference numerically; cross-checked in
``tests/uncertainty/losses/test_pointwise_quantile.py``):

.. code-block:: text

    quantile = 1 - alpha
    y_abs    = abs(y)
    diff     = y_abs - y_pred
    yscale   = max(y_abs, axis=0) + eps        # per-batch element max
    ptwise   = max(quantile * diff, -(1 - quantile) * diff)
    scaled   = ptwise / (2 * quantile * (1 - quantile) * yscale)
    ptavg    = mean over spatial axes (keeps batch + channel)
    loss     = sum or mean over batch + channel

``y`` carries the *true* pointwise residuals (``model_pred -
y_true``) at training time; ``y_pred`` is the residual operator's
predicted quantile widths.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp


_ALLOWED_REDUCTIONS = ("sum", "mean")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PointwiseQuantileLoss:
    """Self-scaling pinball loss for UQNO residual-quantile training.

    Args:
        alpha: Pointwise miscoverage rate in ``(0, 1)``. Together with
            ``delta`` (used downstream in conformal calibration) it
            controls the target coverage proportion ``1 - alpha`` of
            the predicted quantile interval.
        reduction: Reduction across batch + channel dimensions —
            ``"sum"`` (default, matches reference) or ``"mean"``.
            Spatial dimensions are always averaged before reduction.

    """

    alpha: float
    reduction: Literal["sum", "mean"] = "sum"

    def __post_init__(self) -> None:
        """Validate ``alpha ∈ (0, 1)`` and a supported ``reduction``."""
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(
                f"PointwiseQuantileLoss.alpha must lie strictly in (0, 1); got {self.alpha!r}."
            )
        if self.reduction not in _ALLOWED_REDUCTIONS:
            raise ValueError(
                f"PointwiseQuantileLoss.reduction must be one of "
                f"{_ALLOWED_REDUCTIONS}; got {self.reduction!r}."
            )

    def __call__(self, *, y_pred: jax.Array, y: jax.Array, eps: float = 1e-7) -> jax.Array:
        """Compute the loss.

        Args:
            y_pred: Predicted quantile widths; same shape as ``y``.
                Conventionally non-negative (the residual model is
                typically followed by ``softplus`` / ``exp`` so the
                outputs are widths, not signed displacements).
            y: True pointwise residuals (``base_model(x) - y_true``).
            eps: Floor added to the per-batch max before division to
                stop the scaling factor from blowing up on
                near-zero rows. Matches the PyTorch reference's
                ``eps=1e-7``.

        Returns:
            Scalar loss (after the configured reduction).

        """
        quantile = 1.0 - self.alpha
        y_abs = jnp.abs(y)
        diff = y_abs - y_pred
        yscale = jnp.max(y_abs, axis=0) + eps
        ptwise = jnp.maximum(quantile * diff, -(1.0 - quantile) * diff)
        scaled = ptwise / (2.0 * quantile * (1.0 - quantile) * yscale)
        batch = scaled.shape[0]
        ptavg = jnp.mean(scaled.reshape(batch, -1), axis=1, keepdims=True)
        if self.reduction == "sum":
            return jnp.sum(ptavg)
        return jnp.mean(ptavg)


__all__ = ["PointwiseQuantileLoss"]
