"""Metric-driven training control: early stopping and learning-rate plateau decay.

Both react to a stream of per-epoch validation metrics and carry no JAX state,
so they compose with any training loop, including the scan-fused atomistic
epoch whose learning rate is a mutable ``optax.inject_hyperparams`` field
updated between epochs. The best-value tracker, :class:`EarlyStopping` (the
Keras / Lightning semantics: stop after ``patience`` epochs without a
``min_delta`` improvement) and :class:`PlateauMode` are substrax's;
:class:`ReduceLROnPlateau` composes the tracker with the PyTorch scheduler's
semantics (scale the rate by ``factor`` after ``patience`` stagnant epochs,
floored at ``min_lr``). It stays here because it acts once per epoch on a
validation metric between scanned epochs, which ``optax.contrib.reduce_on_plateau``,
a per-step gradient transformation fed at update time, does not express.
"""

from __future__ import annotations

from substrax.callbacks import BestMetricTracker, EarlyStopping, PlateauMode


class ReduceLROnPlateau(BestMetricTracker):
    """Scale a learning rate down when a monitored metric plateaus.

    Args:
        factor: Multiplicative factor applied to the rate on a plateau
            (``0 < factor < 1``).
        patience: Epochs without a ``min_delta`` improvement before reducing.
        min_lr: Lower bound for the reduced learning rate.
        min_delta: Minimum absolute change counted as an improvement.
        mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).

    Raises:
        ValueError: If ``factor`` is not in ``(0, 1)`` or ``patience`` is not
            positive.
    """

    def __init__(
        self,
        *,
        factor: float,
        patience: int,
        min_lr: float = 0.0,
        min_delta: float = 0.0,
        mode: PlateauMode | str = PlateauMode.MIN,
    ) -> None:
        """Initialise the scheduler."""
        super().__init__(mode=mode, min_delta=min_delta)
        if not 0.0 < factor < 1.0:
            raise ValueError(f"factor must be in (0, 1), got {factor}.")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")
        self._factor = float(factor)
        self._patience = patience
        self._min_lr = float(min_lr)

    def update(self, value: float, learning_rate: float) -> float:
        """Record the metric and return the (possibly reduced) learning rate.

        Reduces the rate by ``factor`` (floored at ``min_lr``) once the metric has
        stagnated for ``patience`` epochs, then resets the stagnation counter so
        the next reduction waits a further ``patience`` epochs.
        """
        self.register(value)
        if self._num_bad_epochs >= self._patience and learning_rate > self._min_lr:
            self._reset_stagnation()
            return max(learning_rate * self._factor, self._min_lr)
        return learning_rate


__all__ = ["EarlyStopping", "PlateauMode", "ReduceLROnPlateau"]
