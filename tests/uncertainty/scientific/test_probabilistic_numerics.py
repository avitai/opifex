"""Tests for the probabilistic-numerics adapter catalogue.

21 Pattern-A frozen dataclasses declared in
:mod:`opifex.uncertainty.scientific.probabilistic_numerics`. The catalogue
covers solver adapters, likelihood adapters, prior adapters, and the
seven solver axis specs from the design enumeration.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.scientific import (
    ApplyDiffusionSpec,
    CalibrationSpec,
    CorrectionSpec,
    CubatureRuleSpec,
    DaltonAdapterSpec,
    DenseOutputSamplingSpec,
    DiffeqzooAdapterSpec,
    DiffusionSpec,
    FenrirAdapterSpec,
    InitSchemeSpec,
    IOUPPriorSpec,
    IWPPriorSpec,
    ManifoldUpdateSpec,
    MaternPriorSpec,
    PerturbedStepSolverSpec,
    ProbdiffeqAdapterSpec,
    ProbfindiffAdapterSpec,
    ProbnumAdapterSpec,
    SsmFactSpec,
    StrategySpec,
    TornadoxAdapterSpec,
)


_AXIS_SPECS: tuple[type, ...] = (
    SsmFactSpec,
    InitSchemeSpec,
    CorrectionSpec,
    CubatureRuleSpec,
    StrategySpec,
    CalibrationSpec,
    DiffusionSpec,
)


_DEFERRED_ADAPTER_SPECS: tuple[type, ...] = (
    ProbdiffeqAdapterSpec,
    ProbnumAdapterSpec,
    ProbfindiffAdapterSpec,
    DiffeqzooAdapterSpec,
    ManifoldUpdateSpec,
    PerturbedStepSolverSpec,
    DenseOutputSamplingSpec,
    ApplyDiffusionSpec,
)


_CONCRETIZED_PRIOR_SPECS: tuple[type, ...] = (
    # Task 6.3.11: these three return concrete (drift, dispersion) SDE tuples.
    IOUPPriorSpec,
    MaternPriorSpec,
    IWPPriorSpec,
)


_CONCRETIZED_LIKELIHOOD_SPECS: tuple[type, ...] = (
    # Task 6.3.10: Fenrir / DALTON wrap() returns a JAX-native log-likelihood
    # callable suitable for hyperparameter learning via jax.grad / jax.jit.
    FenrirAdapterSpec,
    DaltonAdapterSpec,
)


_CONCRETIZED_ADAPTER_SPECS: tuple[type, ...] = (
    _CONCRETIZED_PRIOR_SPECS + _CONCRETIZED_LIKELIHOOD_SPECS
)


# All non-axis adapter specs (deferred + concretized), used by structural
# tests that don't care which family the spec is in.
_ADAPTER_SPECS: tuple[type, ...] = _DEFERRED_ADAPTER_SPECS + _CONCRETIZED_ADAPTER_SPECS


def test_catalogue_has_exactly_twenty_one_named_specs() -> None:
    """21 named specs = 7 axes + 14 adapter specs (incl. deprecated TornadoxAdapterSpec)."""
    assert len(_AXIS_SPECS) + len(_ADAPTER_SPECS) + 1 == 21  # +1 for TornadoxAdapterSpec


@pytest.mark.parametrize("spec_cls", _AXIS_SPECS)
def test_axis_spec_is_frozen_dataclass_with_literal_choice(spec_cls: type) -> None:
    """Each axis spec is a frozen Pattern-A dataclass with a ``choice`` field."""
    import dataclasses as dc

    assert dc.is_dataclass(spec_cls)
    spec: Any = spec_cls()
    assert hasattr(spec, "choice")
    assert isinstance(spec.choice, str)
    with pytest.raises(dc.FrozenInstanceError):
        spec.choice = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize("spec_cls", _ADAPTER_SPECS)
def test_adapter_spec_is_frozen_dataclass_with_probnum_strategy(spec_cls: type) -> None:
    """Each adapter spec advertises ``DefaultStrategy.PROBABILISTIC_NUMERICS``."""
    import dataclasses as dc

    assert dc.is_dataclass(spec_cls)
    spec: Any = spec_cls()
    assert spec.default_strategy is DefaultStrategy.PROBABILISTIC_NUMERICS
    assert isinstance(spec.family_tags, tuple)
    assert isinstance(spec.source_package, str)
    with pytest.raises(dc.FrozenInstanceError):
        spec.source_package = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize("spec_cls", _DEFERRED_ADAPTER_SPECS)
def test_deferred_adapter_spec_wrap_raises_actionable_error(spec_cls: type) -> None:
    """Each deferred adapter spec raises an actionable error from ``wrap``."""
    spec: Any = spec_cls()
    capability = UQCapability(default_strategy=spec.default_strategy)
    with pytest.raises(NotImplementedError, match=type(spec).__name__):
        spec.wrap(model=None, capability=capability)


@pytest.mark.parametrize("spec_cls", _CONCRETIZED_PRIOR_SPECS)
def test_concretized_prior_spec_wrap_returns_sde_tuple(spec_cls: type) -> None:
    """IOUP / Matern / IWP prior specs now return concrete SDE matrices.

    Per Task 6.3.11 these three were promoted from deferred-metadata to
    concrete builders. ``wrap`` returns a ``(drift, dispersion)`` pair
    suitable for :func:`opifex.uncertainty.statespace.discretize_lti_sde`.
    """
    spec: Any = spec_cls()
    capability = UQCapability(default_strategy=spec.default_strategy)
    drift, dispersion = spec.wrap(model=None, capability=capability)
    assert drift.ndim == 2
    assert dispersion.ndim == 2
    assert drift.shape[0] == drift.shape[1]
    assert drift.shape[0] == dispersion.shape[0]


@pytest.mark.parametrize("spec_cls", _CONCRETIZED_LIKELIHOOD_SPECS)
def test_concretized_likelihood_spec_wrap_returns_callable(spec_cls: type) -> None:
    """Fenrir / DALTON likelihood specs now return JAX-native callables.

    Per Task 6.3.10 these two were promoted from deferred-metadata to
    concrete log-likelihood combinators usable for hyperparameter learning.
    """
    spec: Any = spec_cls()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)


def test_probdiffeq_adapter_spec_exposes_four_axis_fields() -> None:
    """``ProbdiffeqAdapterSpec`` carries the four solver-axis fields."""
    spec = ProbdiffeqAdapterSpec()
    assert isinstance(spec.ssm_fact, SsmFactSpec)
    assert isinstance(spec.init_scheme, InitSchemeSpec)
    assert isinstance(spec.correction, CorrectionSpec)
    assert isinstance(spec.cubature, CubatureRuleSpec)
    # pn_observation_noise is optional — defaults to None.
    assert spec.pn_observation_noise is None


def test_probdiffeq_adapter_spec_accepts_custom_axis_choices() -> None:
    """Custom axis choices flow through dataclass construction."""
    spec = ProbdiffeqAdapterSpec(
        ssm_fact=SsmFactSpec(choice="blockdiag"),
        correction=CorrectionSpec(choice="ts1"),
        cubature=CubatureRuleSpec(choice="unscented"),
        pn_observation_noise=0.01,
    )
    assert spec.ssm_fact.choice == "blockdiag"
    assert spec.correction.choice == "ts1"
    assert spec.cubature.choice == "unscented"
    assert spec.pn_observation_noise == 0.01


def test_tornadox_adapter_spec_emits_deprecation_warning() -> None:
    """``TornadoxAdapterSpec`` emits a ``DeprecationWarning`` at construction."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        TornadoxAdapterSpec()
    deprecation = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert any("ProbdiffeqAdapterSpec" in str(w.message) for w in deprecation)


def test_fenrir_and_dalton_specs_cite_their_arxiv_papers() -> None:
    """Fenrir / DALTON specs reference their published papers."""
    assert "2202.01287" in FenrirAdapterSpec().notes
    assert "2306.05566" in DaltonAdapterSpec().notes


def test_ioup_prior_spec_advertises_three_rate_modes() -> None:
    """``IOUPPriorSpec`` declares scalar / vector / matrix rate-parameter modes."""
    spec = IOUPPriorSpec()
    assert "scalar_rate" in spec.family_tags
    assert "vector_rate" in spec.family_tags
    assert "matrix_rate" in spec.family_tags
    assert "2305.14978" in spec.notes


def test_probnum_adapter_spec_advertises_vendored_algorithms() -> None:
    """``ProbnumAdapterSpec`` notes its algorithm vendoring strategy."""
    notes = ProbnumAdapterSpec().notes
    assert "vendored" in notes.lower()
    assert "statespace" in notes
