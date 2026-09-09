"""The training infrastructure opifex shares with its siblings comes from substrax and calibrax.

The checkpoint store, the metric-driven callbacks' tracker and early stopping,
and FLOP counting are substrax's and calibrax's objects, not copies. What stays
in opifex is composed on top of them (``ReduceLROnPlateau``, the trainer, the
serving layer).
"""

from __future__ import annotations

import importlib

import pytest
from calibrax.profiling import FlopsCounter as CalibraxFlopsCounter
from substrax.callbacks import BestMetricTracker, EarlyStopping, PlateauMode
from substrax.checkpoint import CheckpointStore, OrbaxCheckpointStore

from opifex.benchmarking.profiling import FlopsCounter
from opifex.core.training import callbacks
from opifex.core.training.components import checkpoint_store


def test_checkpoint_store_is_substrax_store() -> None:
    assert checkpoint_store.OrbaxCheckpointStore is OrbaxCheckpointStore
    assert checkpoint_store.CheckpointStore is CheckpointStore
    assert checkpoint_store.__all__ == ["CheckpointStore", "ModelLike", "OrbaxCheckpointStore"]


def test_callbacks_compose_substrax_tracker() -> None:
    assert callbacks.EarlyStopping is EarlyStopping
    assert callbacks.PlateauMode is PlateauMode
    assert issubclass(callbacks.ReduceLROnPlateau, BestMetricTracker)
    assert callbacks.__all__ == ["EarlyStopping", "PlateauMode", "ReduceLROnPlateau"]


def test_flops_counter_is_calibrax_and_the_estimator_is_gone() -> None:
    assert FlopsCounter is CalibraxFlopsCounter
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("opifex.core.training.monitoring.flops")
