"""UQ capability declarations for the MLOps surfaces.

``opifex.mlops`` is a metric-publication surface: ``ExperimentTracker`` forwards
UQ-flavoured metrics (Brier, ECE, NLL, coverage) to a run, but computes no
uncertainty itself, so its capability is ``UNSUPPORTED``. The table is
collected by :mod:`opifex.capabilities`; importing the package registers nothing.
"""

from opifex.core.capabilities import DefaultStrategy, UQCapability


_EXPERIMENT_TRACKER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="opifex",
    notes=(
        "ExperimentTracker publishes UQ-flavoured metrics (Brier / ECE / "
        "NLL / coverage / interval width) to MLflow or other registered "
        "backends. It does not own any uncertainty computation, which "
        "lives in opifex.uncertainty.monitoring and the calibration and "
        "conformal adapters."
    ),
)


MLOPS_CAPABILITIES: dict[str, UQCapability] = {
    "mlops:ExperimentTracker": _EXPERIMENT_TRACKER_CAPABILITY,
}


__all__ = ["MLOPS_CAPABILITIES"]
