"""Tests for the five named GP adapter specs.

Each spec is a Pattern-A frozen dataclass declaring metadata for a
Gaussian-process backend. Two specs (``GPJaxAdapterSpec``,
``TinygpAdapterSpec``) are user-installed (``OPTIONAL``); three
(``MarkovflowAdapterSpec``, ``BayesnewtonAdapterSpec``,
``KalmanJaxAdapterSpec``) are metadata-only because their algorithms
are vendored elsewhere or live in non-JAX frameworks.

Canonical reference:
* ``../gpjax/gpjax/*`` for GPJax family enumeration.
* ``../tinygp/tinygp/*`` for tinygp family enumeration.
* ``../markovflow``, ``../bayesnewton``, ``../kalman-jax`` for the
  state-space lineage.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from opifex.uncertainty.adapters import (
    BayesnewtonAdapterSpec,
    GPJaxAdapterSpec,
    KalmanJaxAdapterSpec,
    MarkovflowAdapterSpec,
    TinygpAdapterSpec,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_GP_SPECS: tuple[tuple[type, DefaultStrategy], ...] = (
    (GPJaxAdapterSpec, DefaultStrategy.GAUSSIAN_PROCESS),
    (TinygpAdapterSpec, DefaultStrategy.GAUSSIAN_PROCESS),
    (MarkovflowAdapterSpec, DefaultStrategy.STATE_SPACE_FILTERING),
    (BayesnewtonAdapterSpec, DefaultStrategy.STATE_SPACE_FILTERING),
    (KalmanJaxAdapterSpec, DefaultStrategy.STATE_SPACE_FILTERING),
)


@pytest.mark.parametrize(("spec_cls", "expected_strategy"), _GP_SPECS)
def test_gp_adapter_specs_are_frozen_dataclasses_with_capability_metadata(
    spec_cls: type, expected_strategy: DefaultStrategy
) -> None:
    """Every GP adapter spec is a frozen dataclass with the expected metadata."""
    import dataclasses as dc

    assert dc.is_dataclass(spec_cls)
    spec: Any = spec_cls()
    assert spec.default_strategy is expected_strategy
    assert isinstance(spec.source_package, str)
    assert isinstance(spec.required_capabilities, tuple)
    assert isinstance(spec.family_tags, tuple)
    assert all(isinstance(tag, str) for tag in spec.family_tags)
    with pytest.raises(dc.FrozenInstanceError):
        spec.source_package = "tampered"  # type: ignore[misc]


def test_gpjax_adapter_spec_family_tags_match_design_enumeration() -> None:
    """``GPJaxAdapterSpec`` exposes the 9-tag GP family enumeration."""
    spec = GPJaxAdapterSpec()
    expected = {
        "exact_gp",
        "conjugate_gaussian",
        "svgp",
        "non_conjugate",
        "multi_output",
        "deep_kernel",
        "stochastic_variational",
        "natural_gradient",
        "rff_approximation",
    }
    assert set(spec.family_tags) == expected
    # GPJax 0.14 dropped flax.nnx — capability must reflect that.
    assert "native_nnx_module" not in spec.required_capabilities
    assert "native_jax" in spec.required_capabilities


def test_tinygp_adapter_spec_family_tags_match_design_enumeration() -> None:
    """``TinygpAdapterSpec`` exposes the 4-tag family enumeration."""
    spec = TinygpAdapterSpec()
    expected = {
        "exact_gp",
        "conjugate_gaussian",
        "stationary_kernel",
        "quasisep_1d_state_space",
    }
    assert set(spec.family_tags) == expected
    assert "native_jax" in spec.required_capabilities


@pytest.mark.parametrize("spec_cls", [GPJaxAdapterSpec, TinygpAdapterSpec])
def test_gp_adapter_spec_wrap_raises_actionable_error(spec_cls: type) -> None:
    """User-install GP specs raise an actionable error from ``wrap``."""
    spec: Any = spec_cls()
    capability = UQCapability(default_strategy=spec.default_strategy)
    with pytest.raises(NotImplementedError, match=spec.default_strategy.value):
        spec.wrap(model=None, capability=capability)


@pytest.mark.parametrize(
    "spec_cls", [MarkovflowAdapterSpec, BayesnewtonAdapterSpec, KalmanJaxAdapterSpec]
)
def test_state_space_adapter_specs_advertise_vendored_algorithms(spec_cls: type) -> None:
    """Metadata-only specs point users at the vendored statespace module."""
    spec: Any = spec_cls()
    notes_text = spec.notes.lower()
    assert "vendored" in notes_text or "statespace" in notes_text


def test_kalman_jax_adapter_spec_emits_deprecation_warning() -> None:
    """``KalmanJaxAdapterSpec`` emits a ``DeprecationWarning`` pointing at
    the bayesnewton successor (per kalman-jax's own README:1)."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        KalmanJaxAdapterSpec()
    deprecation_messages = [
        str(warning.message)
        for warning in captured
        if issubclass(warning.category, DeprecationWarning)
    ]
    assert any("bayesnewton" in message.lower() for message in deprecation_messages), (
        f"Expected a DeprecationWarning pointing at bayesnewton; got {deprecation_messages!r}"
    )
