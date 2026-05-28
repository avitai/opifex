"""Data-assimilation / digital-twin state utilities (Task 6.7).

A thin orchestration layer over :mod:`opifex.uncertainty.statespace`:
this package supplies the digital-twin-aware containers
(:class:`AssimilationState`), sensor-mask helpers, and a
``jax.lax.scan``-driven sequential update loop. The actual Kalman
math is re-exported from the canonical state-space module so callers
have a one-stop import surface, but no formula bodies are
re-implemented here (Phase 9 Task 9.3 enforces this).
"""

from opifex.uncertainty.assimilation.state import (
    AssimilationState,
    build_default_metadata,
)
from opifex.uncertainty.assimilation.updates import (
    annotate_metadata,
    observation_matrix_from_mask,
    predict,
    sequential_update,
    update,
)

# Re-exports of the underlying Kalman math, so importers can use the
# whole pipeline through ``opifex.uncertainty.assimilation`` if they
# don't want to reach into the state-space subpackage directly.
from opifex.uncertainty.statespace import (
    kalman_predict,
    kalman_update,
)


__all__ = [
    "AssimilationState",
    "annotate_metadata",
    "build_default_metadata",
    "kalman_predict",
    "kalman_update",
    "observation_matrix_from_mask",
    "predict",
    "sequential_update",
    "update",
]
