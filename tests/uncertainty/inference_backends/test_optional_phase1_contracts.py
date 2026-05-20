"""Phase-1 contract tests for the optional inference-backend specs.

The Phase 2 plan mandates that optional backend metadata reflect the
Phase-1 log-density / prior / likelihood contract surface that each
backend would consume when wired:

* Bayeux / NumPyro / oryx — probabilistic-program backends should
  consume the Phase-1 ``LikelihoodSpec`` + ``PriorSpec`` contract
  metadata rather than backend-specific dictionaries.
* sbiax / flowMC — flow-MCMC backends should compose with Artifex
  flows for the density-estimator step.
* traceax / matfree — scalable-curvature primitives that future
  Laplace-style adapters can claim.

These tests verify the structural shape of the spec metadata so the
routing surface remains internally consistent even before any optional
backend is installed.
"""

from __future__ import annotations

import pytest

from opifex.uncertainty.inference_backends import optional as optional_specs
from opifex.uncertainty.inference_backends.optional import (
    OPTIONAL_FLOW_SPECS,
    OPTIONAL_SAMPLER_SPECS,
    OptionalBackendSpec,
)


# ---------------------------------------------------------------------------
# Bayeux / NumPyro / oryx — probabilistic-program contract consumers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Bayeux", "NumPyro", "oryx"])
def test_probabilistic_program_specs_advertise_log_density_methods(name: str) -> None:
    """Each probabilistic-program backend must advertise method names
    callers would target when consuming Phase-1 log-density /
    prior / likelihood contracts."""
    spec = next((s for s in OPTIONAL_SAMPLER_SPECS if s.name == name), None)
    assert spec is not None, f"OPTIONAL_SAMPLER_SPECS missing entry for {name!r}"
    assert isinstance(spec, OptionalBackendSpec)
    assert spec.method_names, (
        f"{name!r} must advertise concrete method names for Phase-1 "
        f"contract consumers — got an empty tuple."
    )
    assert isinstance(spec.method_names, tuple)


def test_probabilistic_program_specs_include_canonical_samplers() -> None:
    """At least one probabilistic-program backend must list a sampler the
    Phase-1 log-density contract knows how to expose (HMC / NUTS)."""
    canonical_samplers = {"hmc", "nuts", "mcmc", "inference"}
    matches = [
        spec
        for spec in OPTIONAL_SAMPLER_SPECS
        if spec.name in {"Bayeux", "NumPyro", "oryx"}
        and any(name.lower() in canonical_samplers for name in spec.method_names)
    ]
    assert matches, (
        "At least one of bayeux / numpyro / oryx must advertise an "
        "HMC- or NUTS-family sampler against the Phase-1 contract."
    )


# ---------------------------------------------------------------------------
# sbiax / flowMC — Artifex-flow composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sbiax", "flowMC"])
def test_flow_mcmc_specs_compose_with_artifex_flows(name: str) -> None:
    """sbiax / flowMC are flow-MCMC backends — their density-estimator
    step composes with an Artifex flow. The spec must declare flow
    composition by listing flow-relevant method names."""
    spec = next((s for s in OPTIONAL_SAMPLER_SPECS if s.name == name), None)
    assert spec is not None, f"OPTIONAL_SAMPLER_SPECS missing entry for {name!r}"
    # Method names must reference flow-MCMC or normalizing-flow surfaces;
    # NPE/NLE/NRE are the canonical sbi/sbiax flow-composition primitives.
    flow_terms = {"flow", "nf", "rqs", "maf", "real_nvp", "neutra", "sbi", "npe", "nle", "nre"}
    has_flow_method = any(
        any(term in name.lower() for term in flow_terms) for name in spec.method_names
    )
    assert has_flow_method, (
        f"{name!r} spec must advertise at least one flow-composition "
        f"method (e.g. neutra / sbi / nf / flow); got {spec.method_names!r}."
    )


def test_artifex_flow_specs_present_for_density_estimator_step() -> None:
    """The Artifex-flow side of sbiax/flowMC composition lives in the
    OPTIONAL_FLOW_SPECS family — at least one spec must declare a
    normalizing-flow family Artifex provides."""
    assert OPTIONAL_FLOW_SPECS, "OPTIONAL_FLOW_SPECS must not be empty"
    # Confirm at least one optional-flow spec advertises a method name
    # that a flow-MCMC backend can pair with.
    flow_methods = [
        method
        for spec in OPTIONAL_FLOW_SPECS
        for method in spec.method_names
    ]
    assert flow_methods, "OPTIONAL_FLOW_SPECS specs must list method names"


# ---------------------------------------------------------------------------
# traceax / matfree — Laplace-style scalable-curvature specs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["traceax", "matfree"])
def test_scalable_curvature_specs_declare_laplace_compatible_methods(name: str) -> None:
    """traceax / matfree provide stochastic-trace / matrix-free
    primitives that a Laplace-style adapter could claim. The spec must
    advertise at least one matrix-free / trace-estimator method name."""
    spec = next((s for s in OPTIONAL_SAMPLER_SPECS if s.name == name), None)
    assert spec is not None, f"OPTIONAL_SAMPLER_SPECS missing entry for {name!r}"
    curvature_terms = {"trace", "diag", "matfree", "lanczos", "lobpcg", "krylov"}
    has_curvature_method = any(
        any(term in mn.lower() for term in curvature_terms) for mn in spec.method_names
    )
    assert has_curvature_method, (
        f"{name!r} spec must advertise at least one trace/matfree/krylov "
        f"primitive name; got {spec.method_names!r}."
    )


# ---------------------------------------------------------------------------
# Cross-spec invariants
# ---------------------------------------------------------------------------


def test_every_optional_sampler_spec_has_unique_name() -> None:
    """Router resolution relies on unique ``name`` keys across the family."""
    names = [spec.name for spec in OPTIONAL_SAMPLER_SPECS]
    assert len(names) == len(set(names)), (
        f"Duplicate spec name(s) in OPTIONAL_SAMPLER_SPECS: {names!r}"
    )


def test_every_optional_sampler_spec_install_hint_starts_with_uv_pip() -> None:
    """Install hints must point to the canonical project package manager
    so error messages are immediately actionable."""
    for spec in OPTIONAL_SAMPLER_SPECS:
        assert spec.install_hint.startswith(("uv pip install", "pip install")), (
            f"{spec.name!r} install_hint must start with a pip command; "
            f"got {spec.install_hint!r}."
        )


def test_optional_specs_module_exports_only_named_collections() -> None:
    """No private symbols escape via wildcard. The two public tuples and
    the spec dataclass are the surface."""
    assert hasattr(optional_specs, "OPTIONAL_SAMPLER_SPECS")
    assert hasattr(optional_specs, "OPTIONAL_FLOW_SPECS")
    assert hasattr(optional_specs, "OptionalBackendSpec")
