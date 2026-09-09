"""Checkpoint persistence: substrax's Orbax-backed store.

``OrbaxCheckpointStore`` writes NNX modules, Flax ``TrainState`` objects and
plain dictionaries with Orbax and their metadata with JSON, addresses
checkpoints by integer step, prunes to ``max_to_keep`` and picks the best step
by a recorded metric. It is the same object every Avitai library uses; the
trainer and the serving layer compose it.
"""

from __future__ import annotations

from substrax.checkpoint import CheckpointStore, ModelLike, OrbaxCheckpointStore


__all__ = ["CheckpointStore", "ModelLike", "OrbaxCheckpointStore"]
