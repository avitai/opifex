r"""Phase 9 distribution-adapter spec closure — Slice 21 (audit finding #3).

Phase 9 final-validation (``09-phase-final-validation.md:780-782``) requires:

    [ ] ``bijx`` is declared the preferred NNX-native normalizing-flow
    backend in the distribution adapter registry; FlowJAX and
    ``distrax`` are alternative backends behind the same protocol.

This test file pins the contract for the three adapter specs that
satisfy that exit criterion:

* ``BijxAdapterSpec`` — preferred NNX-native normalizing-flow backend
  (carries the ``preferred_nf`` family tag).
* ``FlowJAXAdapterSpec`` — equinox-based alternative.
* ``DistraxAdapterSpec`` — TFP-on-JAX fallback.

References
----------
* Phase 9 final-validation plan :780-782.
* Reference packages: ``bijx``, ``flowjax``, ``distrax``.
"""

from __future__ import annotations


def test_phase9_named_distribution_specs_are_importable() -> None:
    """Phase 9 :780-782 gate — all three named distribution adapter specs exist."""
    import opifex.uncertainty.adapters.distribution as dist_specs

    for name in ("BijxAdapterSpec", "FlowJAXAdapterSpec", "DistraxAdapterSpec"):
        assert hasattr(dist_specs, name), (
            f"Distribution adapter spec '{name}' missing from "
            f"opifex.uncertainty.adapters.distribution."
        )


def test_bijx_adapter_spec_advertises_preferred_nf_tag() -> None:
    """``BijxAdapterSpec`` family tags must include ``preferred_nf`` and ``nnx_native``."""
    from opifex.uncertainty.adapters.distribution import BijxAdapterSpec

    spec = BijxAdapterSpec()
    assert "preferred_nf" in spec.family_tags
    assert "nnx_native" in spec.family_tags
    assert "normalizing_flow" in spec.family_tags
    assert spec.source_package == "bijx"


def test_flowjax_adapter_spec_advertises_equinox_native_tag() -> None:
    """``FlowJAXAdapterSpec`` family tags identify the equinox-native lineage."""
    from opifex.uncertainty.adapters.distribution import FlowJAXAdapterSpec

    spec = FlowJAXAdapterSpec()
    assert "equinox_native" in spec.family_tags
    assert "normalizing_flow" in spec.family_tags
    assert spec.source_package == "flowjax"


def test_distrax_adapter_spec_advertises_tfp_substrate_tag() -> None:
    """``DistraxAdapterSpec`` family tags identify the TFP-on-JAX heritage."""
    from opifex.uncertainty.adapters.distribution import DistraxAdapterSpec

    spec = DistraxAdapterSpec()
    assert "tfp_substrate" in spec.family_tags
    assert "distribution" in spec.family_tags
    assert spec.source_package == "distrax"


def test_distribution_specs_share_installation_probe_contract() -> None:
    """All three specs expose ``is_installed`` returning a bool."""
    from opifex.uncertainty.adapters.distribution import (
        BijxAdapterSpec,
        DistraxAdapterSpec,
        FlowJAXAdapterSpec,
    )

    for spec_cls in (BijxAdapterSpec, FlowJAXAdapterSpec, DistraxAdapterSpec):
        installed = spec_cls.is_installed()
        assert isinstance(installed, bool), (
            f"{spec_cls.__name__}.is_installed() must return bool; got {type(installed).__name__}."
        )


def test_distribution_specs_are_frozen_slotted_dataclasses() -> None:
    """All three specs follow the opifex Pattern-A frozen-slotted-kw-only contract."""
    import dataclasses

    from opifex.uncertainty.adapters.distribution import (
        BijxAdapterSpec,
        DistraxAdapterSpec,
        FlowJAXAdapterSpec,
    )

    for spec_cls in (BijxAdapterSpec, FlowJAXAdapterSpec, DistraxAdapterSpec):
        assert dataclasses.is_dataclass(spec_cls)
        spec = spec_cls()
        # Frozen: attribute assignment must raise.
        import pytest

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            spec.source_package = "mutated"  # type: ignore[misc,assignment]
