"""Backend-router tests.

The :class:`opifex.uncertainty.inference_backends.router.BackendRouter`
selects the highest-priority available backend for a given family
(``"flow"`` / ``"sampler"`` / ``"distribution"``) following the
Artifex-first resolution order. Unknown family names raise a clear
``ValueError`` listing the registered families.
"""

from __future__ import annotations

import pytest

from opifex.uncertainty.inference_backends.optional import (
    ARTIFEX_FLOW_SPECS,
    DISTRIBUTION_SPECS,
    OPTIONAL_FLOW_SPECS,
)
from opifex.uncertainty.inference_backends.router import BackendRouter


def test_router_resolves_flow_family_to_artifex_first() -> None:
    router = BackendRouter()
    chosen = router.resolve("flow")
    assert chosen.source_package == "artifex"
    # First Artifex flow advertised wins.
    assert chosen == ARTIFEX_FLOW_SPECS[0]


def test_router_lists_available_flows_including_optional_installed_ones() -> None:
    router = BackendRouter()
    available = router.available("flow")
    artifex_names = {spec.name for spec in ARTIFEX_FLOW_SPECS}
    assert artifex_names <= {spec.name for spec in available}
    # Optional flows that probe True are also present (likely none in this env).
    for spec in OPTIONAL_FLOW_SPECS:
        if spec.probe():
            assert spec in available


def test_router_resolves_sampler_family_picks_first_available() -> None:
    router = BackendRouter()
    chosen = router.resolve("sampler")
    # Must pick BlackJAX (delegated through Artifex — always-available).
    assert chosen.name in {"BlackJAX", "TFP-substrate"}


def test_router_distribution_family_resolves_to_artifex_first() -> None:
    router = BackendRouter()
    chosen = router.resolve("distribution")
    assert chosen == DISTRIBUTION_SPECS[0]
    assert chosen.source_package == "artifex"


def test_router_unknown_family_raises_value_error_listing_available_families() -> None:
    router = BackendRouter()
    with pytest.raises(ValueError, match=r"family"):
        router.resolve("not-a-family")


def test_router_resolve_by_name_picks_specific_backend() -> None:
    router = BackendRouter()
    spec = router.resolve("sampler", name="BlackJAX")
    assert spec.name == "BlackJAX"


def test_router_resolve_by_name_raises_for_unavailable_backend() -> None:
    """An ``unavailable`` backend (probe False) raises ImportError on instantiation,
    but ``resolve`` returns the spec so callers can read its install_hint.
    """
    router = BackendRouter()
    bijx_spec = next(s for s in OPTIONAL_FLOW_SPECS if s.name == "bijx")
    if not bijx_spec.probe():
        spec = router.resolve("flow", name="bijx")
        assert spec.name == "bijx"
        with pytest.raises(ImportError):
            spec.instantiate()


def test_router_resolve_by_name_raises_for_unknown_name() -> None:
    router = BackendRouter()
    with pytest.raises(ValueError, match=r"unknown backend"):
        router.resolve("sampler", name="totally-not-a-real-backend")


def test_router_all_families_listed_in_unknown_family_error() -> None:
    """Error message must list the registered families so callers know what's valid."""
    router = BackendRouter()
    with pytest.raises(ValueError, match=r"unknown family") as excinfo:
        router.resolve("not-a-family")
    message = str(excinfo.value)
    for family in ("flow", "sampler", "distribution"):
        assert family in message


def test_router_includes_sampler_spec_for_optional_sampler_families() -> None:
    """Every optional sampler family is registered in the router's spec table."""
    router = BackendRouter()
    sampler_names = {spec.name for spec in router.available("sampler")}
    for expected in ("TFP-substrate", "Bayeux", "NumPyro", "sbiax", "flowMC"):
        assert expected in sampler_names
