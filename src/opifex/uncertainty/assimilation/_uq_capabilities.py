"""UQ capability declarations for the data-assimilation and digital-twin surfaces.

Static, module-level constants, collected into :data:`opifex.capabilities.UQ_CAPABILITIES`;
importing this module changes no global state. ``opifex.uncertainty.assimilation`` is a thin
orchestration layer over the state-space primitives (CAKF/CAKS and square-root Kalman), so
the declarations set ``default_strategy=DefaultStrategy.STATE_SPACE_FILTERING`` and
``native_jax_kernel=True``.
"""

from __future__ import annotations

from opifex.core.capabilities import DefaultStrategy, UQCapability


# Static digital-twin-aware state container. Frozen Pattern A dataclass;
# carries the Kalman-filter state + provenance metadata.
_ASSIMILATION_STATE_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.STATE_SPACE_FILTERING,
    native_jax_kernel=True,
    source_package="opifex",
    notes=(
        "AssimilationState is the Pattern A frozen container threaded "
        "through the Task 6.7 assimilation/digital-twin loop. Wraps a "
        "Kalman state (mean / covariance) over the Task 6.3 "
        "state-space primitives."
    ),
)


# Sequential update loop over an observation stream. ``jax.lax.scan``
# native_jax_kernel; the Kalman math is re-exported from
# ``opifex.uncertainty.statespace``.
_SEQUENTIAL_UPDATE_CAPABILITY = UQCapability(
    supports_solver_uncertainty=True,
    default_strategy=DefaultStrategy.STATE_SPACE_FILTERING,
    native_jax_kernel=True,
    source_package="opifex",
    notes=(
        "sequential_update is a jax.lax.scan-driven assimilation loop "
        "over an observation stream. Builds on the CAKF/CAKS and "
        "square-root Kalman primitives from "
        "opifex.uncertainty.statespace (Task 6.3)."
    ),
)


# Single-step predict / update wrappers around the canonical Kalman math.
_PREDICT_CAPABILITY = UQCapability(
    supports_solver_uncertainty=True,
    default_strategy=DefaultStrategy.STATE_SPACE_FILTERING,
    native_jax_kernel=True,
    source_package="opifex",
    notes=(
        "Single-step Kalman predict over an AssimilationState. Thin "
        "wrapper around opifex.uncertainty.statespace.kalman_predict."
    ),
)


_UPDATE_CAPABILITY = UQCapability(
    supports_solver_uncertainty=True,
    default_strategy=DefaultStrategy.STATE_SPACE_FILTERING,
    native_jax_kernel=True,
    source_package="opifex",
    notes=(
        "Single-step Kalman update over an AssimilationState. Thin "
        "wrapper around opifex.uncertainty.statespace.kalman_update."
    ),
)


ASSIMILATION_CAPABILITIES: dict[str, UQCapability] = {
    "assimilation:AssimilationState": _ASSIMILATION_STATE_CAPABILITY,
    "assimilation:sequential_update": _SEQUENTIAL_UPDATE_CAPABILITY,
    "assimilation:predict": _PREDICT_CAPABILITY,
    "assimilation:update": _UPDATE_CAPABILITY,
}


__all__ = ["ASSIMILATION_CAPABILITIES"]
