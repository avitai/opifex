"""Tests for the metric-driven training-control callbacks."""

from __future__ import annotations

import pytest

from opifex.core.training.callbacks import EarlyStopping, ReduceLROnPlateau


class TestEarlyStopping:
    def test_stops_after_patience_without_improvement(self) -> None:
        stopper = EarlyStopping(patience=3)
        assert stopper.update(1.0) is True  # first value is an improvement
        assert not stopper.should_stop
        for _ in range(2):
            assert stopper.update(1.0) is False  # no improvement
            assert not stopper.should_stop
        assert stopper.update(1.0) is False  # third stagnant epoch
        assert stopper.should_stop

    def test_improvement_resets_the_counter(self) -> None:
        stopper = EarlyStopping(patience=2)
        stopper.update(1.0)
        stopper.update(1.0)  # 1 bad epoch
        assert stopper.update(0.5) is True  # improvement resets
        assert not stopper.should_stop
        assert stopper.best == pytest.approx(0.5)

    def test_min_delta_requires_meaningful_improvement(self) -> None:
        stopper = EarlyStopping(patience=5, min_delta=0.1)
        stopper.update(1.0)
        assert stopper.update(0.95) is False  # below the 0.1 threshold

    def test_max_mode_tracks_increasing_metric(self) -> None:
        stopper = EarlyStopping(patience=2, mode="max")
        assert stopper.update(0.5) is True
        assert stopper.update(0.7) is True
        assert stopper.update(0.6) is False

    def test_rejects_non_positive_patience(self) -> None:
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0)


class TestReduceLROnPlateau:
    def test_reduces_after_patience(self) -> None:
        scheduler = ReduceLROnPlateau(factor=0.5, patience=2)
        lr = 1.0
        lr = scheduler.update(1.0, lr)  # improvement
        assert lr == pytest.approx(1.0)
        lr = scheduler.update(1.0, lr)  # 1 bad
        assert lr == pytest.approx(1.0)
        lr = scheduler.update(1.0, lr)  # 2 bad -> reduce
        assert lr == pytest.approx(0.5)

    def test_floors_at_min_lr(self) -> None:
        scheduler = ReduceLROnPlateau(factor=0.5, patience=1, min_lr=0.3)
        lr = 0.4
        lr = scheduler.update(1.0, lr)  # improvement
        lr = scheduler.update(1.0, lr)  # plateau -> 0.2 clamped to 0.3
        assert lr == pytest.approx(0.3)
        lr = scheduler.update(1.0, lr)  # already at min_lr, no further change
        assert lr == pytest.approx(0.3)

    def test_improvement_prevents_reduction(self) -> None:
        scheduler = ReduceLROnPlateau(factor=0.5, patience=2)
        lr = 1.0
        for value in (1.0, 0.9, 0.8, 0.7):  # steady improvement
            lr = scheduler.update(value, lr)
        assert lr == pytest.approx(1.0)

    def test_rejects_factor_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="factor"):
            ReduceLROnPlateau(factor=1.5, patience=1)
