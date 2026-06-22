"""Metric-driven training-control callbacks: early stopping + LR plateau decay.

Both react to a stream of per-epoch validation metrics. They are framework
agnostic (pure-Python scalar bookkeeping, no JAX state), so they compose with any
training loop -- including the scan-fused atomistic epoch, whose learning rate is a
mutable ``optax.inject_hyperparams`` field updated between epochs.

Semantics follow the established references: :class:`EarlyStopping` mirrors the
Keras / Lightning callback (stop after ``patience`` epochs without a
``min_delta`` improvement) and :class:`ReduceLROnPlateau` mirrors the PyTorch
scheduler of the same name (scale the rate by ``factor`` after ``patience``
stagnant epochs, floored at ``min_lr``).
"""

from __future__ import annotations

from enum import StrEnum


class PlateauMode(StrEnum):
    """Whether a monitored metric improves by decreasing or increasing."""

    MIN = "min"
    MAX = "max"


class _BestMetricTracker:
    """Shared best-so-far + stagnation bookkeeping for the plateau callbacks.

    Tracks the best monitored value and the number of consecutive updates without
    a ``min_delta`` improvement, in either ``"min"`` or ``"max"`` mode.
    """

    def __init__(self, *, mode: PlateauMode | str, min_delta: float) -> None:
        """Initialise the tracker.

        Args:
            mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).
            min_delta: Minimum absolute change counted as an improvement.

        Raises:
            ValueError: If ``min_delta`` is negative.
        """
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}.")
        self._mode = PlateauMode(mode)
        self._min_delta = float(min_delta)
        self._best = float("inf") if self._mode is PlateauMode.MIN else float("-inf")
        self._num_bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        """Return whether ``value`` beats the best by at least ``min_delta``."""
        if self._mode is PlateauMode.MIN:
            return value < self._best - self._min_delta
        return value > self._best + self._min_delta

    def register(self, value: float) -> bool:
        """Record ``value``; return ``True`` if it improves on the best so far.

        Updates the best value and resets the stagnation counter on an
        improvement; otherwise increments the stagnation counter.
        """
        if self._is_improvement(value):
            self._best = float(value)
            self._num_bad_epochs = 0
            return True
        self._num_bad_epochs += 1
        return False

    @property
    def best(self) -> float:
        """The best monitored value seen so far."""
        return self._best

    @property
    def num_bad_epochs(self) -> int:
        """Consecutive updates without a ``min_delta`` improvement."""
        return self._num_bad_epochs

    def _reset_stagnation(self) -> None:
        """Clear the stagnation counter (e.g. after an LR reduction)."""
        self._num_bad_epochs = 0


class EarlyStopping(_BestMetricTracker):
    """Signal to stop once a monitored metric stops improving.

    Args:
        patience: Epochs without a ``min_delta`` improvement before stopping.
        min_delta: Minimum absolute change counted as an improvement.
        mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).

    Raises:
        ValueError: If ``patience`` is not positive.
    """

    def __init__(
        self,
        *,
        patience: int,
        min_delta: float = 0.0,
        mode: PlateauMode | str = PlateauMode.MIN,
    ) -> None:
        """Initialise the stopper."""
        super().__init__(mode=mode, min_delta=min_delta)
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")
        self._patience = patience

    def update(self, value: float) -> bool:
        """Record the latest metric; return ``True`` if it improved on the best."""
        return self.register(value)

    @property
    def should_stop(self) -> bool:
        """Whether the metric has stagnated for ``patience`` epochs."""
        return self._num_bad_epochs >= self._patience


class ReduceLROnPlateau(_BestMetricTracker):
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
