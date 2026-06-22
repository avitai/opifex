"""Phase 8 Task 8.5 capability-coverage tests.

Plan reference: ``08-phase-pac-bayes-sbi-active-stochastic-fields.md``
lines 755-790.

Tasks 8.1 (PAC-Bayes), 8.2 (SBI), 8.3 (active learning + trainer rewrites
+ GP-acquisition L2O), and 8.4 (stochastic-field surrogates) all landed
before this task, so the registry MUST now contain at least one surface
in each of the four capability buckets the plan demands:

1. ``supports_pac_bayes_certificate=True`` — PAC-Bayes certificate driver
   and Bayesian models with a variational posterior consumable by it.
2. ``supports_likelihood_free=True`` — SBI surfaces (NPE / NLE / NRE).
3. ``supports_active_learning=True`` — acquisition surfaces and the
   rewritten ``UncertaintyGuidedTrainer`` / ``ActiveUncertaintyLearner``.
4. ``supports_stochastic_field_input=True`` — KLE / PCE / stochastic-
   Galerkin / stochastic-collocation surrogates from
   ``opifex.uncertainty.scientific``.

In addition to membership, the plan requires honest provenance: every
capability declaration that opts into one of these strategies records
:attr:`UQCapability.source_package` (non-empty string).
"""

from __future__ import annotations

import pytest

# The per-subsystem ``from .._uq_capabilities import ...`` lines below
# transitively import each Phase 8 subpackage's ``__init__``, which seeds
# the singleton :class:`UQRegistry` via its idempotent registration loop.
# We therefore do NOT need explicit ``import opifex.<sub>`` side-effect
# lines — the capability-table imports already pull the subpackage in.
from opifex.neural.bayesian._uq_capabilities import BAYESIAN_MODEL_CAPABILITIES
from opifex.training._uq_capabilities import TRAINING_CAPABILITIES
from opifex.uncertainty.active._uq_capabilities import ACTIVE_CAPABILITIES
from opifex.uncertainty.pac_bayes._uq_capabilities import PAC_BAYES_CAPABILITIES
from opifex.uncertainty.registry import DefaultStrategy, UQCapability, UQRegistry
from opifex.uncertainty.sbi._uq_capabilities import SBI_CAPABILITIES
from opifex.uncertainty.scientific._uq_capabilities import (
    SCIENTIFIC_FIELD_CAPABILITIES,
)


# Aggregate Phase 8 capability tables for the autouse seeder. The
# ``test_registry.py`` suite uses an autouse :meth:`UQRegistry.reset`
# fixture that wipes every registration; we re-seed here so this suite
# is order-independent (mirrors the Phase 7 coverage tests).
_ALL_PHASE_8_CAPABILITIES: dict[str, UQCapability] = {
    **PAC_BAYES_CAPABILITIES,
    **SBI_CAPABILITIES,
    **ACTIVE_CAPABILITIES,
    **TRAINING_CAPABILITIES,
    **SCIENTIFIC_FIELD_CAPABILITIES,
    **BAYESIAN_MODEL_CAPABILITIES,
}


@pytest.fixture(autouse=True)
def _seed_registry() -> None:  # pyright: ignore[reportUnusedFunction]
    """Re-seed the shared singleton ``UQRegistry`` with every Phase 8 entry.

    Idempotent — registers only names not already present so consecutive
    fixture invocations are safe.
    """
    registry = UQRegistry()
    for name, capability in _ALL_PHASE_8_CAPABILITIES.items():
        if name not in registry:
            registry.register(name, capability)


@pytest.fixture
def uq_registry() -> UQRegistry:
    """Return the shared singleton ``UQRegistry`` after subpackage imports."""
    return UQRegistry()


# ---------------------------------------------------------------------------
# Plan exit criterion 1: at least one surface advertises
# ``supports_pac_bayes_certificate=True``.
# ---------------------------------------------------------------------------


def test_at_least_one_surface_supports_pac_bayes_certificate(
    uq_registry: UQRegistry,
) -> None:
    """Task 8.1 landed → the registry MUST advertise at least one
    PAC-Bayes-capable surface (the certificate driver + a model that
    owns a posterior with ``kl_divergence``)."""
    pac_bayes_capable = [
        name
        for name in uq_registry.list_names()
        if uq_registry.require(name).supports_pac_bayes_certificate
    ]
    assert pac_bayes_capable, (
        "Phase 8 Task 8.5 requires ≥ 1 surface with "
        "supports_pac_bayes_certificate=True; registry advertises none."
    )


def test_pac_bayes_certificate_surface_uses_pac_bayes_strategy(
    uq_registry: UQRegistry,
) -> None:
    """The dedicated ``pac_bayes:certificate`` entry advertises the
    plan-mandated :attr:`DefaultStrategy.PAC_BAYES` strategy."""
    cap = uq_registry.require("pac_bayes:certificate")
    assert cap.supports_pac_bayes_certificate is True
    assert cap.default_strategy is DefaultStrategy.PAC_BAYES


def test_probabilistic_pinn_opts_into_pac_bayes_certificate(
    uq_registry: UQRegistry,
) -> None:
    """``ProbabilisticPINN`` owns a mean-field variational posterior with
    a ``kl_divergence`` method (the certificate driver's only structural
    requirement) so its capability MUST advertise opt-in."""
    cap = uq_registry.require("model:ProbabilisticPINN")
    assert cap.supports_pac_bayes_certificate is True


# ---------------------------------------------------------------------------
# Plan exit criterion 2: at least one inverse-problem surface advertises
# ``supports_likelihood_free=True``.
# ---------------------------------------------------------------------------


def test_at_least_one_inverse_problem_surface_supports_likelihood_free(
    uq_registry: UQRegistry,
) -> None:
    """Task 8.2 SBI surfaces (NPE / NLE / NRE) MUST advertise the
    likelihood-free flag."""
    likelihood_free_capable = [
        name
        for name in uq_registry.list_names()
        if uq_registry.require(name).supports_likelihood_free
    ]
    assert likelihood_free_capable, (
        "Phase 8 Task 8.5 requires ≥ 1 surface with "
        "supports_likelihood_free=True; registry advertises none."
    )


@pytest.mark.parametrize("name", ["sbi:npe", "sbi:nle", "sbi:nre"])
def test_sbi_surface_advertises_likelihood_free_sbi_strategy(
    name: str, uq_registry: UQRegistry
) -> None:
    """Each SBI estimator advertises the plan-mandated
    :attr:`DefaultStrategy.LIKELIHOOD_FREE_SBI` strategy."""
    cap = uq_registry.require(name)
    assert cap.supports_likelihood_free is True
    assert cap.default_strategy is DefaultStrategy.LIKELIHOOD_FREE_SBI


# ---------------------------------------------------------------------------
# Plan exit criterion 3: at least one trainer / learner advertises
# ``supports_active_learning=True``.
# ---------------------------------------------------------------------------


def test_at_least_one_trainer_or_learner_supports_active_learning(
    uq_registry: UQRegistry,
) -> None:
    """Task 8.3 rewrote ``UncertaintyGuidedTrainer`` and
    ``ActiveUncertaintyLearner`` to invoke real uncertainty quantifiers
    + active-learning acquisitions → both MUST advertise opt-in now."""
    active_learning_capable = [
        name
        for name in uq_registry.list_names()
        if uq_registry.require(name).supports_active_learning
    ]
    assert active_learning_capable, (
        "Phase 8 Task 8.5 requires ≥ 1 surface with "
        "supports_active_learning=True; registry advertises none."
    )


@pytest.mark.parametrize(
    "name",
    ["trainer:UncertaintyGuidedTrainer", "trainer:ActiveUncertaintyLearner"],
)
def test_active_learning_trainer_uses_active_learning_strategy(
    name: str, uq_registry: UQRegistry
) -> None:
    cap = uq_registry.require(name)
    assert cap.supports_active_learning is True
    assert cap.default_strategy is DefaultStrategy.ACTIVE_LEARNING


def test_multi_fidelity_trainer_default_strategy_flipped(
    uq_registry: UQRegistry,
) -> None:
    """Task 8.3 rewrote ``MultiFidelityUncertaintyTrainer`` to call both
    high/low fidelity models for real and combine their ensemble
    decompositions → the strategy must reflect the ensemble flavour."""
    cap = uq_registry.require("trainer:MultiFidelityUncertaintyTrainer")
    assert cap.default_strategy is DefaultStrategy.ENSEMBLE
    assert cap.supports_ensemble is True


# ---------------------------------------------------------------------------
# Plan exit criterion 4: at least one solver / scientific-UQ surface
# advertises ``supports_stochastic_field_input=True``.
# ---------------------------------------------------------------------------


def test_at_least_one_scientific_surface_supports_stochastic_field_input(
    uq_registry: UQRegistry,
) -> None:
    """Task 8.4 landed → the registry MUST advertise ≥ 1 stochastic-field
    surrogate that accepts KLE/PCE-parameterized random-field inputs."""
    stochastic_field_capable = [
        name
        for name in uq_registry.list_names()
        if uq_registry.require(name).supports_stochastic_field_input
    ]
    assert stochastic_field_capable, (
        "Phase 8 Task 8.5 requires ≥ 1 surface with "
        "supports_stochastic_field_input=True; registry advertises none."
    )


@pytest.mark.parametrize(
    ("name", "expected_strategy"),
    [
        ("stochastic_fields:KLE", DefaultStrategy.KARHUNEN_LOEVE),
        ("stochastic_fields:PCE", DefaultStrategy.POLYNOMIAL_CHAOS),
        ("stochastic_fields:StochasticGalerkin", DefaultStrategy.STOCHASTIC_GALERKIN),
        ("stochastic_fields:StochasticCollocation", DefaultStrategy.STOCHASTIC_GALERKIN),
    ],
)
def test_stochastic_field_surface_default_strategy(
    name: str, expected_strategy: DefaultStrategy, uq_registry: UQRegistry
) -> None:
    cap = uq_registry.require(name)
    assert cap.supports_stochastic_field_input is True
    assert cap.default_strategy is expected_strategy


# ---------------------------------------------------------------------------
# Plan exit criterion 5: declarations that opt in record ``source_package``.
# ---------------------------------------------------------------------------


def test_phase_8_capability_declarations_record_source_package(
    uq_registry: UQRegistry,
) -> None:
    """Every capability that opts into a Phase 8 strategy MUST record a
    non-empty ``source_package`` so the registry display can attribute
    the implementation honestly (Opifex-local vs. sibling-backed)."""
    phase_8_strategies: frozenset[DefaultStrategy] = frozenset(
        {
            DefaultStrategy.PAC_BAYES,
            DefaultStrategy.LIKELIHOOD_FREE_SBI,
            DefaultStrategy.ACTIVE_LEARNING,
            DefaultStrategy.KARHUNEN_LOEVE,
            DefaultStrategy.POLYNOMIAL_CHAOS,
            DefaultStrategy.STOCHASTIC_GALERKIN,
        }
    )
    opted_in = [
        name
        for name in uq_registry.list_names()
        if uq_registry.require(name).default_strategy in phase_8_strategies
    ]
    assert opted_in, "Sanity: at least one Phase 8 opt-in surface must exist."
    for name in opted_in:
        cap = uq_registry.require(name)
        assert cap.source_package, (
            f"Capability {name!r} opts into {cap.default_strategy.value!r} but "
            f"records empty source_package."
        )


def test_phase_8_opt_in_flags_pair_with_provenance_metadata(
    uq_registry: UQRegistry,
) -> None:
    """Any surface advertising one of the four Phase 8 capability flags
    MUST also record :attr:`source_package` (the sibling-reuse audit
    relies on this provenance field for the Phase 9 final coverage)."""
    flag_attrs: tuple[str, ...] = (
        "supports_pac_bayes_certificate",
        "supports_likelihood_free",
        "supports_active_learning",
        "supports_stochastic_field_input",
    )
    for name in uq_registry.list_names():
        cap = uq_registry.require(name)
        if any(getattr(cap, attr) for attr in flag_attrs):
            assert cap.source_package, (
                f"Capability {name!r} advertises a Phase 8 flag but records empty source_package."
            )
