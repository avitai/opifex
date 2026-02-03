"""Split conformal prediction for calibrated prediction intervals.

Provides model-wrapping conformal prediction that produces prediction intervals
with finite-sample coverage guarantees, without distributional assumptions.

The split conformal method:
1. Uses a held-out calibration set to compute nonconformity scores |y - f(x)|.
2. Selects the ceil((n+1)(1-alpha))/n quantile of those scores as the interval
   half-width.
3. At prediction time, returns (f(x) - q, f(x) + q) as the prediction interval.

Reference:
    Vovk, Gammerman, Shafer. "Algorithmic Learning in a Random World" (2005).
    Lei et al. "Distribution-Free Predictive Inference for Regression" (2018).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp


if TYPE_CHECKING:
    import jax
    from flax import nnx


@dataclass(frozen=True)
class ConformalConfig:
    """Configuration for conformal prediction.

    Attributes:
        alpha: Miscoverage level. The target coverage probability is 1 - alpha.
            Must be in the open interval (0, 1).
    """

    alpha: float = 0.1

    def __post_init__(self) -> None:
        """Validate that alpha is in (0, 1)."""
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


class ConformalPredictor:
    """Split conformal prediction for calibrated prediction intervals.

    Wraps any point predictor (PINN, neural operator, etc.) and provides
    calibrated prediction intervals without distributional assumptions.

    The predictor must be calibrated on a held-out calibration set before
    prediction intervals can be computed.

    Attributes:
        model: The wrapped NNX module used for point predictions.
        config: Conformal prediction configuration.
    """

    def __init__(
        self,
        model: nnx.Module,
        config: ConformalConfig | None = None,
    ) -> None:
        """Initialize the conformal predictor.

        Args:
            model: Any Flax NNX module that maps inputs to predictions.
                Must implement ``__call__(x) -> jax.Array``.
            config: Conformal prediction configuration. If ``None``, uses
                default ``ConformalConfig(alpha=0.1)``.
        """
        self.model = model
        self.config = config if config is not None else ConformalConfig()
        self._quantile: float | None = None
        self._is_calibrated: bool = False

    def calibrate(
        self,
        x_cal: jax.Array,
        y_cal: jax.Array,
    ) -> None:
        """Compute nonconformity scores on a calibration set.

        Runs the wrapped model on ``x_cal``, computes absolute residuals
        against ``y_cal``, and stores the conformal quantile.

        Args:
            x_cal: Calibration inputs with shape ``(n, ...)``.
            y_cal: Calibration targets with shape ``(n, ...)``.
        """
        predictions = self.model(x_cal)  # pyright: ignore[reportCallIssue]
        scores = jnp.abs(y_cal - predictions)

        # Flatten to 1D for quantile computation (handles multi-output)
        scores_flat = scores.flatten()
        n = scores_flat.shape[0]

        # Split conformal quantile level: ceil((n+1)(1-alpha)) / n
        # Clipped to [0, 1] to handle edge cases with very small n
        quantile_level = min(math.ceil((n + 1) * (1.0 - self.config.alpha)) / n, 1.0)

        self._quantile = float(jnp.quantile(scores_flat, quantile_level))
        self._is_calibrated = True

    def predict_with_intervals(
        self,
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return point predictions with calibrated prediction intervals.

        Args:
            x: Input array with shape ``(n, ...)``.

        Returns:
            A tuple of ``(predictions, lower_bounds, upper_bounds)`` where each
            array has the same shape as the model output.

        Raises:
            RuntimeError: If ``calibrate()`` has not been called yet.
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "Must call calibrate() before predict_with_intervals(). "
                "Provide a calibration dataset first."
            )

        predictions = self.model(x)  # pyright: ignore[reportCallIssue]
        half_width = self._quantile  # type: ignore[assignment]
        lower = predictions - half_width
        upper = predictions + half_width

        return predictions, lower, upper
