"""Checkpoint persistence: substrax's Orbax-backed store.

``OrbaxCheckpointStore`` writes named items (``model``, ``optimizer``, ``rng``, ...)
with Orbax and a JSON record beside them: the step, the epoch, the metrics, the
producer and the caller's own values. It addresses checkpoints by integer step,
prunes to ``max_to_keep`` and picks the best step by a recorded metric. It reads
the one checkpoint format it writes and refuses any other with substrax's
``UnsupportedCheckpointError``. It is the same object every Avitai library uses;
the trainer and the serving layer compose it.
"""

from __future__ import annotations

from substrax.checkpoint import CheckpointStore, OrbaxCheckpointStore


__all__ = ["CheckpointStore", "OrbaxCheckpointStore"]
