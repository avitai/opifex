"""Tests for the named Bayesian-quadrature adapter specs.

Pattern-A frozen dataclasses for the algorithmic BQ methods listed in
the design notes (WSABI-L coexists with VanillaBQ; SOBER vs FFBQ
split; emukit metadata-only).
"""

from __future__ import annotations

from typing import Any

import pytest

from opifex.uncertainty.quadrature import (
    EmukitQuadratureAdapterSpec,
    FFBQAdapterSpec,
    SOBERAdapterSpec,
    VanillaBayesianQuadratureAdapterSpec,
    WSABILAdapterSpec,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_CONCRETIZED_BQ_SPECS: tuple[type, ...] = (
    # Task 6.3.13: Vanilla BQ + WSABI-L vendored in bayesian_quadrature.py.
    VanillaBayesianQuadratureAdapterSpec,
    WSABILAdapterSpec,
    # Task 6.3.14a: SOBER kernel-recombination vendored in sober.py.
    SOBERAdapterSpec,
    # Task 6.3.14b: Frank-Wolfe BQ vendored in frank_wolfe_bq.py.
    FFBQAdapterSpec,
)


_DEFERRED_BQ_SPECS: tuple[type, ...] = (EmukitQuadratureAdapterSpec,)


_BQ_SPECS: tuple[type, ...] = _CONCRETIZED_BQ_SPECS + _DEFERRED_BQ_SPECS


@pytest.mark.parametrize("spec_cls", _BQ_SPECS)
def test_bq_adapter_spec_is_frozen_dataclass_with_bq_strategy(spec_cls: type) -> None:
    """Each BQ spec advertises ``DefaultStrategy.BAYESIAN_QUADRATURE``."""
    import dataclasses as dc

    assert dc.is_dataclass(spec_cls)
    spec: Any = spec_cls()
    assert spec.default_strategy is DefaultStrategy.BAYESIAN_QUADRATURE
    assert isinstance(spec.family_tags, tuple)
    assert all(isinstance(tag, str) for tag in spec.family_tags)
    with pytest.raises(dc.FrozenInstanceError):
        spec.source_package = "tampered"  # type: ignore[misc]


@pytest.mark.parametrize("spec_cls", _DEFERRED_BQ_SPECS)
def test_deferred_bq_adapter_spec_wrap_raises_actionable_error(spec_cls: type) -> None:
    """Every deferred BQ spec raises an actionable error from ``wrap``."""
    spec: Any = spec_cls()
    capability = UQCapability(default_strategy=spec.default_strategy)
    with pytest.raises(NotImplementedError, match=spec.default_strategy.value):
        spec.wrap(model=None, capability=capability)


@pytest.mark.parametrize("spec_cls", _CONCRETIZED_BQ_SPECS)
def test_concretized_bq_adapter_spec_wrap_returns_callable(spec_cls: type) -> None:
    """Vanilla BQ / WSABI-L specs now return JAX-native callables (Task 6.3.13)."""
    spec: Any = spec_cls()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)


def test_wsabi_l_and_vanilla_bq_coexist_in_bayesian_quadrature_module() -> None:
    """WSABI-L and VanillaBQ notes both reference ``bayesian_quadrature.py``."""
    wsabi = WSABILAdapterSpec()
    vanilla = VanillaBayesianQuadratureAdapterSpec()
    assert "bayesian_quadrature.py" in wsabi.notes
    assert "bayesian_quadrature.py" in vanilla.notes


def test_sober_and_ffbq_advertise_separate_modules() -> None:
    """SOBER and FFBQ live in distinct modules (per the design split, fix #190)."""
    sober = SOBERAdapterSpec()
    ffbq = FFBQAdapterSpec()
    assert "sober.py" in sober.notes
    # FFBQ is Frank-Wolfe BQ per Briol+ 2015; the design notes pin the
    # file to frank_wolfe_bq.py rather than the legacy mnemonic ffbq.py.
    assert "frank_wolfe_bq.py" in ffbq.notes


def test_emukit_quadrature_spec_is_metadata_only_reference() -> None:
    """``EmukitQuadratureAdapterSpec`` advertises its vendored-reference role."""
    spec = EmukitQuadratureAdapterSpec()
    assert "Metadata-only" in spec.notes or "metadata-only" in spec.notes.lower()
    assert spec.source_package == "emukit"
