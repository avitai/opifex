"""Tests for the scientific-domain UQ capability declarations (Task 7.5).

Plan exit criteria (``07-phase-registry-docs-examples.md`` lines 572-680):

1. Every expected scientific-domain name (equation discovery, quantum
   chemistry, L2O, training trainers, data assimilation, mlops,
   monitoring/reporting) is registered in the singleton
   :class:`UQRegistry`.
2. ``l2o:BayesianSchedulerOptimizer`` carries
   ``native_bayesian=False`` and
   ``default_strategy=DefaultStrategy.UNSUPPORTED`` with a docstring
   note pointing at Phase 8 Task 8.3 / 8.5.
3. ``trainer:UncertaintyGuidedTrainer`` and
   ``trainer:MultiFidelityUncertaintyTrainer`` carry
   ``default_strategy=DefaultStrategy.UNSUPPORTED`` with mocked-
   prediction notes pointing at Phase 8 Task 8.3 / 8.5.
4. Assimilation declarations carry
   ``default_strategy=DefaultStrategy.STATE_SPACE_FILTERING`` and
   ``native_jax_kernel=True`` per the Task 6.7 thin-layer-over-
   statespace architecture.
5. ``mlops:ExperimentTracker`` carries
   ``default_strategy=DefaultStrategy.UNSUPPORTED`` per the
   "metric-publication-only" rule.
6. Equation-discovery surfaces declare honest UQ strategies
   (deterministic + adapter strategies for SINDy / WeakSINDy /
   distill_ude_residual; ENSEMBLE for EnsembleSINDy; UNSUPPORTED for
   SymbolicRegressor).
7. Quantum chemistry surfaces declare the deterministic baseline +
   three adapter strategies via :meth:`UQCapability.with_adapter`.
8. Monitoring/reporting surfaces declare CALIBRATION default with
   pure-function builder semantics.
"""

from __future__ import annotations

import pytest

from opifex.discovery._uq_capabilities import DISCOVERY_CAPABILITIES
from opifex.discovery.sindy._uq_capabilities import SINDY_CAPABILITIES
from opifex.mlops._uq_capabilities import MLOPS_CAPABILITIES
from opifex.neural.quantum._uq_capabilities import QUANTUM_CAPABILITIES
from opifex.optimization.l2o._uq_capabilities import L2O_CAPABILITIES
from opifex.training._uq_capabilities import TRAINING_CAPABILITIES
from opifex.uncertainty.assimilation._uq_capabilities import ASSIMILATION_CAPABILITIES
from opifex.uncertainty.monitoring._uq_capabilities import MONITORING_CAPABILITIES
from opifex.uncertainty.registry import DefaultStrategy, UQCapability, UQRegistry


_ALL_TASK_7_5_CAPABILITIES: dict[str, UQCapability] = {
    **DISCOVERY_CAPABILITIES,
    **SINDY_CAPABILITIES,
    **QUANTUM_CAPABILITIES,
    **L2O_CAPABILITIES,
    **TRAINING_CAPABILITIES,
    **ASSIMILATION_CAPABILITIES,
    **MLOPS_CAPABILITIES,
    **MONITORING_CAPABILITIES,
}


@pytest.fixture(autouse=True)
def _seed_registry() -> None:  # pyright: ignore[reportUnusedFunction]
    """Re-seed the shared singleton ``UQRegistry`` with the Task 7.5 entries.

    Idempotent — registers only names not already present so consecutive
    fixture invocations are safe. Sibling tests (notably
    ``test_registry.py``) use an ``autouse`` :meth:`UQRegistry.reset`
    fixture that wipes every registration; we re-seed here so this
    suite is order-independent.
    """
    registry = UQRegistry()
    for name, capability in _ALL_TASK_7_5_CAPABILITIES.items():
        if name not in registry:
            registry.register(name, capability)


# ---------------------------------------------------------------------------
# Expected registry contents (Task 7.5).
# ---------------------------------------------------------------------------


_TASK_7_5_DISCOVERY_NAMES: frozenset[str] = frozenset(
    {
        "discovery:SymbolicRegressor",
        "discovery:SINDy",
        "discovery:WeakSINDy",
        "discovery:EnsembleSINDy",
        "discovery:distill_ude_residual",
    }
)


_TASK_7_5_QUANTUM_NAMES: frozenset[str] = frozenset(
    {
        "quantum:NeuralDFT",
        "quantum:NeuralSCFSolver",
        "quantum:NeuralXCFunctional",
    }
)


_TASK_7_5_L2O_NAMES: frozenset[str] = frozenset(
    {
        "l2o:BayesianSchedulerOptimizer",
    }
)


_TASK_7_5_TRAINER_NAMES: frozenset[str] = frozenset(
    {
        "trainer:UncertaintyGuidedTrainer",
        "trainer:MultiFidelityUncertaintyTrainer",
    }
)


_TASK_7_5_ASSIMILATION_NAMES: frozenset[str] = frozenset(
    {
        "assimilation:AssimilationState",
        "assimilation:sequential_update",
        "assimilation:predict",
        "assimilation:update",
    }
)


_TASK_7_5_MLOPS_NAMES: frozenset[str] = frozenset(
    {
        "mlops:ExperimentTracker",
    }
)


_TASK_7_5_MONITORING_NAMES: frozenset[str] = frozenset(
    {
        "monitoring:build_reliability_report",
        "monitoring:MonitoringInputs",
    }
)


_TASK_7_5_EXPECTED: frozenset[str] = (
    _TASK_7_5_DISCOVERY_NAMES
    | _TASK_7_5_QUANTUM_NAMES
    | _TASK_7_5_L2O_NAMES
    | _TASK_7_5_TRAINER_NAMES
    | _TASK_7_5_ASSIMILATION_NAMES
    | _TASK_7_5_MLOPS_NAMES
    | _TASK_7_5_MONITORING_NAMES
)


# ---------------------------------------------------------------------------
# Coverage tests — every expected name is registered.
# ---------------------------------------------------------------------------


@pytest.fixture
def uq_registry() -> UQRegistry:
    """Return the shared singleton ``UQRegistry`` after subpackage imports."""
    return UQRegistry()


@pytest.mark.parametrize("name", sorted(_TASK_7_5_EXPECTED))
def test_task_7_5_capability_name_is_registered(name: str, uq_registry: UQRegistry) -> None:
    assert name in uq_registry, f"Task 7.5 expected name {name!r} missing from UQRegistry."


# ---------------------------------------------------------------------------
# Equation-discovery declarations.
# ---------------------------------------------------------------------------


def test_symbolic_regressor_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("discovery:SymbolicRegressor")
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED
    assert cap.source_package == "pysr"


@pytest.mark.parametrize(
    "name",
    [
        "discovery:SINDy",
        "discovery:WeakSINDy",
        "discovery:distill_ude_residual",
    ],
)
def test_sindy_family_deterministic_baseline(name: str, uq_registry: UQRegistry) -> None:
    """SINDy / WeakSINDy / UDE distillation are deterministic + adapter-mediated."""
    cap = uq_registry.require(name)
    assert cap.default_strategy is DefaultStrategy.DETERMINISTIC
    assert cap.native_jax_kernel is True
    assert cap.supports_conformal is True
    assert cap.supports_calibration is True


def test_ensemble_sindy_native_ensemble_strategy(uq_registry: UQRegistry) -> None:
    """EnsembleSINDy advertises native bootstrap-aggregated ensemble UQ."""
    cap = uq_registry.require("discovery:EnsembleSINDy")
    assert cap.default_strategy is DefaultStrategy.ENSEMBLE
    assert cap.supports_ensemble is True
    assert cap.supports_calibration is True
    assert cap.native_jax_kernel is True


# ---------------------------------------------------------------------------
# Quantum chemistry declarations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    sorted(_TASK_7_5_QUANTUM_NAMES),
)
def test_quantum_surface_declares_three_adapter_strategies(
    name: str, uq_registry: UQRegistry
) -> None:
    """Each quantum chemistry surface inherits the three-adapter baseline."""
    cap = uq_registry.require(name)
    assert cap.default_strategy is DefaultStrategy.DETERMINISTIC
    assert cap.native_nnx_module is True
    assert cap.supports_ensemble is True
    assert cap.supports_conformal is True
    assert cap.supports_calibration is True


# ---------------------------------------------------------------------------
# L2O declarations.
# ---------------------------------------------------------------------------


def test_bayesian_scheduler_optimizer_capability_flags(uq_registry: UQRegistry) -> None:
    """Phase 7 records BayesianSchedulerOptimizer honestly as UNSUPPORTED.

    The current ``_bayesian_parameter_suggestion`` is a random heuristic
    (adaptive_schedulers.py); the Bayesian flag flips in Phase 8 Task 8.5
    after Phase 8 Task 8.3 lands the real GP-acquisition implementation.
    """
    cap = uq_registry.require("l2o:BayesianSchedulerOptimizer")
    assert cap.native_bayesian is False
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED
    assert cap.native_nnx_module is True
    # Plan-mandated docstring breadcrumb pointing at Phase 8.
    assert "Phase 8 Task 8.3" in cap.notes
    assert "Phase 8 Task 8.5" in cap.notes


# ---------------------------------------------------------------------------
# Training-side trainer declarations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    sorted(_TASK_7_5_TRAINER_NAMES),
)
def test_trainer_capability_is_unsupported_with_phase_8_note(
    name: str, uq_registry: UQRegistry
) -> None:
    """UncertaintyGuidedTrainer + MultiFidelityUncertaintyTrainer record honestly.

    Both classes use a hardcoded ``jax.random.PRNGKey(42)`` mock prediction
    today (basic_trainer.py:1168/1280); Phase 8 Task 8.3 rewrites them to
    call the real model and Phase 8 Task 8.5 flips the capability flags.
    """
    cap = uq_registry.require(name)
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED
    assert "Phase 8 Task 8.3" in cap.notes
    assert "Phase 8 Task 8.5" in cap.notes


def test_uncertainty_guided_trainer_active_learning_flag_off(
    uq_registry: UQRegistry,
) -> None:
    """Phase 7 records ``supports_active_learning=False``; flips in Phase 8."""
    cap = uq_registry.require("trainer:UncertaintyGuidedTrainer")
    assert cap.supports_active_learning is False


def test_multi_fidelity_uncertainty_trainer_distributional_flag_off(
    uq_registry: UQRegistry,
) -> None:
    """Phase 7 records ``native_distributional=False``; flips in Phase 8."""
    cap = uq_registry.require("trainer:MultiFidelityUncertaintyTrainer")
    assert cap.native_distributional is False


# ---------------------------------------------------------------------------
# Data-assimilation / digital-twin declarations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    sorted(_TASK_7_5_ASSIMILATION_NAMES),
)
def test_assimilation_capability_state_space_filtering(name: str, uq_registry: UQRegistry) -> None:
    """Task 6.7 assimilation surfaces inherit the STATE_SPACE_FILTERING strategy."""
    cap = uq_registry.require(name)
    assert cap.default_strategy is DefaultStrategy.STATE_SPACE_FILTERING
    assert cap.native_jax_kernel is True
    assert cap.source_package == "opifex"


def test_assimilation_update_loops_propagate_solver_uncertainty(
    uq_registry: UQRegistry,
) -> None:
    """The predict / update / sequential_update loops carry solver-uncertainty.

    The pure ``AssimilationState`` container is metadata only and does
    not flip ``supports_solver_uncertainty``; the three update primitives
    do.
    """
    for name in (
        "assimilation:predict",
        "assimilation:update",
        "assimilation:sequential_update",
    ):
        cap = uq_registry.require(name)
        assert cap.supports_solver_uncertainty is True, name


# ---------------------------------------------------------------------------
# MLOps declarations.
# ---------------------------------------------------------------------------


def test_mlops_experiment_tracker_unsupported(uq_registry: UQRegistry) -> None:
    """``opifex.mlops`` is a metric-publication surface — UNSUPPORTED placeholder."""
    cap = uq_registry.require("mlops:ExperimentTracker")
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED
    assert cap.source_package == "opifex"
    # Plan-mandated breadcrumb tying the placeholder to the underlying
    # monitoring/calibration surface.
    assert "opifex.uncertainty.monitoring" in cap.notes


# ---------------------------------------------------------------------------
# Monitoring / reporting declarations.
# ---------------------------------------------------------------------------


def test_build_reliability_report_calibration_strategy(uq_registry: UQRegistry) -> None:
    """The Phase 5 reliability-report builder is a pure-function CALIBRATION surface."""
    cap = uq_registry.require("monitoring:build_reliability_report")
    assert cap.default_strategy is DefaultStrategy.CALIBRATION
    assert cap.native_jax_kernel is True
    assert cap.supports_calibration is True


def test_monitoring_inputs_metadata_only(uq_registry: UQRegistry) -> None:
    """``MonitoringInputs`` is a provenance container; no computation flag."""
    cap = uq_registry.require("monitoring:MonitoringInputs")
    assert cap.default_strategy is DefaultStrategy.CALIBRATION
    assert cap.source_package == "opifex"


# ---------------------------------------------------------------------------
# Closed-world membership — every Task 7.5 prefix is populated.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("discovery:", _TASK_7_5_DISCOVERY_NAMES),
        ("quantum:", _TASK_7_5_QUANTUM_NAMES),
        ("l2o:", _TASK_7_5_L2O_NAMES),
        ("trainer:", _TASK_7_5_TRAINER_NAMES),
        ("assimilation:", _TASK_7_5_ASSIMILATION_NAMES),
        ("mlops:", _TASK_7_5_MLOPS_NAMES),
        ("monitoring:", _TASK_7_5_MONITORING_NAMES),
    ],
)
def test_task_7_5_prefix_membership(
    prefix: str, expected: frozenset[str], uq_registry: UQRegistry
) -> None:
    """Every Task 7.5 prefix is populated by exactly the expected name set."""
    observed = {name for name in uq_registry.list_names() if name.startswith(prefix)}
    assert observed == expected, (
        f"Task 7.5 prefix {prefix!r} expected {sorted(expected)!r}; got {sorted(observed)!r}."
    )


def test_task_7_5_capability_tables_use_only_task_7_5_prefixes() -> None:
    """The Task 7.5 capability tables never write under another phase's prefix."""
    allowed_prefixes = (
        "discovery:",
        "quantum:",
        "l2o:",
        "trainer:",
        "assimilation:",
        "mlops:",
        "monitoring:",
    )
    for name in _ALL_TASK_7_5_CAPABILITIES:
        assert any(name.startswith(prefix) for prefix in allowed_prefixes), (
            f"Task 7.5 capability table registers {name!r}, which does not use a Task 7.5 prefix."
        )
