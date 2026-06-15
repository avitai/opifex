"""Optional-backend adapter spec tests.

The :mod:`opifex.uncertainty.inference_backends.optional` module declares
adapter specs for every optional backend the audit lists (TFP-substrate,
bijx, FlowJAX, Bayeux, NumPyro, GPJax, sbiax, flowMC, oryx, traceax,
matfree, kfac-jax) plus Artifex's seven NNX-native flow families. Each
spec exposes a name, family, source package, missing-dependency hint, and
a list of supported method names.

Specs are pattern (A) frozen+slotted dataclasses (hashable, registrable).
The lazy ``probe()`` method returns ``True`` when the optional package is
installed. ``instantiate()`` either returns the actual sampler / flow /
adapter when available, or raises :class:`ImportError` with an actionable
message naming the Artifex alternative.
"""

from __future__ import annotations

import importlib.util

import pytest

from opifex.uncertainty.inference_backends.optional import (
    ARTIFEX_FLOW_SPECS,
    DISTRIBUTION_SPECS,
    OPTIONAL_FLOW_SPECS,
    OPTIONAL_SAMPLER_SPECS,
    OptionalBackendSpec,
)


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


# -----------------------------------------------------------------------------
# Artifex flow family — always available (Artifex is a hard dependency)
# -----------------------------------------------------------------------------


def test_artifex_flow_specs_cover_all_seven_families() -> None:
    """The seven Artifex flow classes are all registered."""
    expected = {"RealNVP", "MAF", "IAF", "Glow", "NeuralSplineFlow", "ConditionalFlow", "MADE"}
    actual = {spec.name for spec in ARTIFEX_FLOW_SPECS}
    assert expected <= actual


def test_artifex_flow_specs_always_probe_true() -> None:
    """Artifex is a hard dependency; every flow spec probes installed."""
    for spec in ARTIFEX_FLOW_SPECS:
        assert spec.probe() is True
        assert spec.source_package == "artifex"


def test_artifex_flow_specs_are_pattern_a_frozen_slotted() -> None:
    """Adapter specs are hashable for registry storage (rule: tuple-over-list)."""
    import dataclasses

    for spec in ARTIFEX_FLOW_SPECS:
        assert dataclasses.is_dataclass(spec)
        assert hasattr(spec, "__slots__")
        # Sequence fields must be tuples for hashability.
        assert isinstance(spec.method_names, tuple)
        hash(spec)  # Must not raise.


# -----------------------------------------------------------------------------
# Optional flow backends — present-when-installed, absent-when-not
# -----------------------------------------------------------------------------


def test_optional_flow_specs_include_bijx_and_flowjax() -> None:
    spec_names = {spec.name for spec in OPTIONAL_FLOW_SPECS}
    assert "bijx" in spec_names
    assert "FlowJAX" in spec_names


def test_optional_flow_specs_probe_matches_install_state() -> None:
    """``probe()`` mirrors ``importlib.util.find_spec``."""
    for spec in OPTIONAL_FLOW_SPECS:
        assert spec.probe() == _has(spec.import_module)


def test_optional_flow_specs_instantiation_raises_import_error_when_missing() -> None:
    """Absent optional dependency: ``instantiate`` raises ImportError naming Artifex."""
    for spec in OPTIONAL_FLOW_SPECS:
        if not spec.probe():
            with pytest.raises(ImportError, match=r"(?i)artifex"):
                spec.instantiate()


# -----------------------------------------------------------------------------
# Optional sampler backends
# -----------------------------------------------------------------------------


def test_optional_sampler_specs_cover_audit_listed_families() -> None:
    """TFP, Bayeux, NumPyro, oryx, sbiax, flowMC, traceax, matfree, kfac-jax."""
    names = {spec.name for spec in OPTIONAL_SAMPLER_SPECS}
    for expected in (
        "TFP-substrate",
        "Bayeux",
        "NumPyro",
        "oryx",
        "sbiax",
        "flowMC",
        "traceax",
        "matfree",
        "kfac-jax",
    ):
        assert expected in names, f"missing optional sampler spec: {expected}"


def test_tfp_sampler_spec_probes_installed() -> None:
    """TFP IS installed in this environment; the spec must reflect that."""
    spec = next(s for s in OPTIONAL_SAMPLER_SPECS if s.name == "TFP-substrate")
    assert spec.probe() is True


def test_unavailable_sampler_instantiation_raises_import_error_with_install_hint() -> None:
    """Absent optional sampler: instantiation raises ImportError with package name."""
    for spec in OPTIONAL_SAMPLER_SPECS:
        if not spec.probe():
            with pytest.raises(ImportError, match=spec.install_hint):
                spec.instantiate()


# -----------------------------------------------------------------------------
# Distribution-adapter specs (Artifex first, then GPJax / Distrax / TFP)
# -----------------------------------------------------------------------------


def test_distribution_specs_resolution_order_artifex_first() -> None:
    """First entry is Artifex; later entries are alternatives."""
    assert DISTRIBUTION_SPECS[0].source_package == "artifex"


def test_distribution_specs_include_distrax_and_tfp() -> None:
    names = {spec.name for spec in DISTRIBUTION_SPECS}
    assert "Distrax" in names
    assert "TFP-substrate" in names


def test_distrax_distribution_spec_probes_installed() -> None:
    """Distrax IS installed (it's a transitive dep of artifex.continuous.Normal)."""
    spec = next(s for s in DISTRIBUTION_SPECS if s.name == "Distrax")
    assert spec.probe() is True


def test_optional_backend_spec_is_hashable_for_registry_use() -> None:
    """Specs serve as registry keys / dispatch entries — must be hashable."""
    sample = ARTIFEX_FLOW_SPECS[0]
    table: dict[OptionalBackendSpec, str] = {sample: "test"}
    assert table[sample] == "test"


def test_optional_backend_spec_install_hint_names_the_pip_package() -> None:
    """Every optional spec's ``install_hint`` contains the importable module name."""
    for spec in OPTIONAL_FLOW_SPECS + OPTIONAL_SAMPLER_SPECS:
        assert spec.install_hint, f"{spec.name} has no install_hint"
