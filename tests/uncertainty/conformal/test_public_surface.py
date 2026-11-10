"""The conformal package re-exports the types its calibrators return (issue #66).

``PredictionInterval`` and ``PredictionSet`` are defined in ``opifex.uncertainty.types``
with the rest of the predictive value objects, and every conformal calibrator returns one
of them. Both statements are documented and correct; together they left the return type
unimportable from the module that returns it, so annotating a function that wraps a
calibrator failed on the natural import.

The state types are the control: this module already re-exported ``SplitConformalState``
and its siblings, which is why the omission read as an oversight rather than a layering
decision.
"""

from __future__ import annotations

import importlib

import pytest


#: Re-exported here because the calibrators return them.
RETURNED_TYPES = ("PredictionInterval", "PredictionSet")

#: Already re-exported before this change, and so the control: if these were missing the
#: module's surface would be broken in a different way and the assertions below would be
#: measuring that instead.
CONTROL_NAMES = (
    "SplitConformalState",
    "GroupedSplitConformalState",
    "LACConformalState",
    "conformal_quantile",
)


@pytest.mark.parametrize("name", CONTROL_NAMES)
def test_the_control_names_are_exported(name: str) -> None:
    """The surface this module already had, unchanged."""
    conformal = importlib.import_module("opifex.uncertainty.conformal")
    assert hasattr(conformal, name)
    assert name in conformal.__all__


@pytest.mark.parametrize("name", RETURNED_TYPES)
def test_the_returned_types_are_importable_from_the_module_that_returns_them(
    name: str,
) -> None:
    """``from opifex.uncertainty.conformal import PredictionInterval`` resolves."""
    conformal = importlib.import_module("opifex.uncertainty.conformal")
    assert hasattr(conformal, name), f"{name} is not exported from opifex.uncertainty.conformal"
    assert name in conformal.__all__, f"{name} is absent from __all__"


@pytest.mark.parametrize("name", RETURNED_TYPES)
def test_the_re_export_is_the_same_object_as_the_definition(name: str) -> None:
    """A re-export, not a redefinition: one type, so ``isinstance`` holds either way."""
    conformal = importlib.import_module("opifex.uncertainty.conformal")
    types_module = importlib.import_module("opifex.uncertainty.types")
    assert getattr(conformal, name) is getattr(types_module, name)


def test_a_calibrator_returns_the_re_exported_type() -> None:
    """The type a calibrator actually returns is the one the module exports.

    Without this the exports above could drift to names nothing returns.
    """
    import jax.numpy as jnp
    import numpy as np

    from opifex.uncertainty.conformal import PredictionInterval, SplitConformalRegressor

    rng = np.random.default_rng(0)
    predictions = jnp.asarray(rng.normal(size=40).astype(np.float32))
    targets = predictions + jnp.asarray(rng.normal(scale=0.3, size=40).astype(np.float32))
    calibrator = SplitConformalRegressor(alpha=0.10)
    state = calibrator.fit(predictions=predictions, targets=targets)
    interval = calibrator.with_state(state).predict(predictions=predictions[:3])
    assert isinstance(interval, PredictionInterval)
