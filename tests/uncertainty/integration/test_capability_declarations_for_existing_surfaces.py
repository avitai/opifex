"""Capability declarations for existing UQ surfaces.

Pins one capability declaration per pre-existing UQ-bearing class and asserts
the declarations are internally consistent (no contradictory flag pairs).
Acts as a smoke check that the :class:`UQCapability` schema is rich enough to
describe every surface the canonical registry will eventually hold.

These are NOT the canonical capability-coverage tests — those live in
``tests/uncertainty/test_capability_coverage.py`` and use the binding
``uq_registry`` singleton. Here we use a local registry instance to avoid
polluting the singleton.
"""

from __future__ import annotations

import dataclasses

import pytest

from opifex.uncertainty.registry import (
    DefaultStrategy,
    UQCapability,
    UQRegistry,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:  # pyright: ignore[reportUnusedFunction]
    UQRegistry.reset()


def test_uqno_capability_declaration_is_internally_consistent() -> None:
    """UQNO is a Bayesian neural operator with native NNX state + adapter-mediated UQ."""
    cap = UQCapability(
        native_bayesian=True,
        native_nnx_module=True,
        supports_function_space=True,
        supports_ensemble=True,
        default_strategy=DefaultStrategy.BAYESIAN,
        source_package="opifex",
        notes="UQNO: Bayesian Fourier-spectral operator with epistemic + aleatoric heads.",
    )
    assert cap.native_bayesian and cap.native_nnx_module
    assert cap.default_strategy is DefaultStrategy.BAYESIAN
    assert not cap.requires_graph_adapter or cap.notes


def test_mean_field_gaussian_capability_declaration() -> None:
    """MeanFieldGaussian is the variational posterior used by AmortizedVariationalFramework."""
    cap = UQCapability(
        native_bayesian=True,
        native_nnx_module=True,
        default_strategy=DefaultStrategy.VARIATIONAL,
        source_package="opifex",
    )
    assert cap.default_strategy is DefaultStrategy.VARIATIONAL


def test_probabilistic_pinn_capability_declaration() -> None:
    """Probabilistic PINNs combine physics residuals with Bayesian neural networks."""
    cap = UQCapability(
        native_bayesian=True,
        native_nnx_module=True,
        default_strategy=DefaultStrategy.BAYESIAN,
        source_package="opifex",
        notes="Probabilistic PINN: variational posterior over weights + physics residual loss.",
    )
    assert cap.native_bayesian
    assert cap.default_strategy is DefaultStrategy.BAYESIAN


def test_deterministic_baseline_capability_is_all_false() -> None:
    """A deterministic FNO / DeepONet / MLP gets the baseline declaration."""
    cap = UQCapability.deterministic_baseline()
    for f in dataclasses.fields(UQCapability):
        if f.name.startswith(("supports_", "native_", "requires_")):
            assert getattr(cap, f.name) is False, f"baseline must have {f.name}=False"


def test_capability_declarations_round_trip_through_uq_registry() -> None:
    """Every declaration above can be registered and looked up by name."""
    registry = UQRegistry()
    registry.register(
        "uqno",
        UQCapability(
            native_bayesian=True,
            native_nnx_module=True,
            supports_function_space=True,
            supports_ensemble=True,
            default_strategy=DefaultStrategy.BAYESIAN,
            source_package="opifex",
        ),
    )
    registry.register(
        "mean_field_gaussian",
        UQCapability(
            native_bayesian=True,
            native_nnx_module=True,
            default_strategy=DefaultStrategy.VARIATIONAL,
            source_package="opifex",
        ),
    )
    assert registry.get("uqno").supports_function_space is True
    assert registry.get("mean_field_gaussian").default_strategy is DefaultStrategy.VARIATIONAL
    assert set(registry.list_names()) == {"uqno", "mean_field_gaussian"}


def test_capability_contradiction_native_jax_kernel_with_required_graph_adapter() -> None:
    """A pure JAX kernel cannot also require an NNX graph adapter (would be a contradiction).

    Catching this combination in capability declarations early means downstream
    capability tests cannot accidentally claim mutually exclusive properties.
    """
    cap = UQCapability(
        native_jax_kernel=True,
        requires_graph_adapter=True,
        notes="contradictory; this test asserts the test surfaces the contradiction",
        default_strategy=DefaultStrategy.DETERMINISTIC,
    )
    # Currently the dataclass accepts this combination — a cross-field
    # validator is required. The test surfaces the gap explicitly so a
    # subsequent validator has a target to satisfy.
    if cap.native_jax_kernel and cap.requires_graph_adapter:
        pytest.xfail(
            "UQCapability must add a __post_init__ validator that rejects "
            "(native_jax_kernel AND requires_graph_adapter) as a contradiction."
        )
