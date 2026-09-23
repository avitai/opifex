"""Tests for the scientific-domain UQ capability declarations (Task 7.5).

Plan exit criteria (``07-phase-registry-docs-examples.md`` lines 572-680):

1. Every expected scientific-domain name (equation discovery, quantum
   chemistry, L2O, training trainers, data assimilation, mlops,
   monitoring/reporting) is declared in the capability catalogue
   :data:`opifex.capabilities.UQ_CAPABILITIES`.
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

from opifex.capabilities import require_uq_capability, UQ_CAPABILITIES
from opifex.core.capabilities import DefaultStrategy, UQCapability
from opifex.discovery._uq_capabilities import DISCOVERY_CAPABILITIES
from opifex.discovery.sindy._uq_capabilities import SINDY_CAPABILITIES
from opifex.mlops._uq_capabilities import MLOPS_CAPABILITIES
from opifex.neural.quantum._uq_capabilities import QUANTUM_CAPABILITIES
from opifex.optimization.l2o import L2O_CAPABILITIES
from opifex.training._uq_capabilities import TRAINING_CAPABILITIES
from opifex.uncertainty.assimilation._uq_capabilities import ASSIMILATION_CAPABILITIES
from opifex.uncertainty.monitoring._uq_capabilities import MONITORING_CAPABILITIES


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
        "quantum:NeuralXCFunctional",
    }
)


_L2O_NAMES: frozenset[str] = frozenset(
    {
        "l2o:L2OEngine",
        "l2o:LearnedOptimizer",
    }
)


_TASK_7_5_TRAINER_NAMES: frozenset[str] = frozenset(
    {
        "trainer:UncertaintyGuidedTrainer",
        "trainer:MultiFidelityUncertaintyTrainer",
        # Phase 8 Task 8.5 added the third trainer entry — flipped from
        # UNSUPPORTED placeholders to ACTIVE_LEARNING after the Task 8.3
        # rewrite landed.
        "trainer:ActiveUncertaintyLearner",
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
    | _L2O_NAMES
    | _TASK_7_5_TRAINER_NAMES
    | _TASK_7_5_ASSIMILATION_NAMES
    | _TASK_7_5_MLOPS_NAMES
    | _TASK_7_5_MONITORING_NAMES
)


# ---------------------------------------------------------------------------
# Coverage tests — every expected name is registered.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_TASK_7_5_EXPECTED))
def test_task_7_5_capability_name_is_registered(name: str) -> None:
    assert name in UQ_CAPABILITIES, (
        f"Task 7.5 expected name {name!r} missing from the capability catalogue."
    )


# ---------------------------------------------------------------------------
# Equation-discovery declarations.
# ---------------------------------------------------------------------------


def test_symbolic_regressor_capability_flags() -> None:
    cap = require_uq_capability("discovery:SymbolicRegressor")
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
def test_sindy_family_deterministic_baseline(name: str) -> None:
    """SINDy / WeakSINDy / UDE distillation are deterministic + adapter-mediated."""
    cap = require_uq_capability(name)
    assert cap.default_strategy is DefaultStrategy.DETERMINISTIC
    assert cap.native_jax_kernel is True
    assert cap.supports_conformal is True
    assert cap.supports_calibration is True


def test_ensemble_sindy_native_ensemble_strategy() -> None:
    """EnsembleSINDy advertises native bootstrap-aggregated ensemble UQ."""
    cap = require_uq_capability("discovery:EnsembleSINDy")
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
def test_quantum_surface_declares_three_adapter_strategies(name: str) -> None:
    """Each quantum chemistry surface inherits the three-adapter baseline."""
    cap = require_uq_capability(name)
    assert cap.default_strategy is DefaultStrategy.DETERMINISTIC
    assert cap.native_nnx_module is True
    assert cap.supports_ensemble is True
    assert cap.supports_conformal is True
    assert cap.supports_calibration is True


# ---------------------------------------------------------------------------
# Learn-to-optimize declarations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_L2O_NAMES))
def test_l2o_surface_declares_no_uncertainty_strategy(name: str) -> None:
    """Learned optimisers report measured point summaries, not predictive distributions."""
    cap = require_uq_capability(name)
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED
    assert cap.native_bayesian is False
    assert cap.native_distributional is False
    assert cap.source_package == "opifex"
    assert cap.notes


# ---------------------------------------------------------------------------
# Training-side trainer declarations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    sorted(_TASK_7_5_TRAINER_NAMES),
)
def test_trainer_capability_records_phase_8_provenance(name: str) -> None:
    """All three trainer declarations record their Phase 8 provenance.

    The Phase 7 UNSUPPORTED placeholders flipped in Phase 8 Task 8.5
    after Task 8.3 rewrote both trainers to invoke the wrapped model +
    uncertainty quantifier for real and added the
    :class:`ActiveUncertaintyLearner` entry. The notes string MUST still
    mention the upgrade path so the audit can trace it.
    """
    cap = require_uq_capability(name)
    assert cap.default_strategy is not DefaultStrategy.UNSUPPORTED
    assert "Phase 8 Task 8.3" in cap.notes
    assert cap.source_package == "opifex"


def test_uncertainty_guided_trainer_active_learning_flag_on() -> None:
    """Phase 8 Task 8.5 flipped supports_active_learning ``False → True``."""
    cap = require_uq_capability("trainer:UncertaintyGuidedTrainer")
    assert cap.supports_active_learning is True
    assert cap.default_strategy is DefaultStrategy.ACTIVE_LEARNING


def test_multi_fidelity_uncertainty_trainer_strategy_is_ensemble() -> None:
    """Phase 8 Task 8.5 flipped the strategy to ENSEMBLE.

    Task 8.3 rewrote the trainer to invoke both high/low fidelity
    models via :func:`_stochastic_ensemble_from_model` and combine
    their ensemble decompositions with Kennedy-O'Hagan additive linear
    weighting — the strategy bucket reflects that flavour.
    """
    cap = require_uq_capability("trainer:MultiFidelityUncertaintyTrainer")
    assert cap.default_strategy is DefaultStrategy.ENSEMBLE
    assert cap.supports_ensemble is True


def test_active_uncertainty_learner_default_strategy() -> None:
    """Task 8.5 added the third trainer entry — ACTIVE_LEARNING default."""
    cap = require_uq_capability("trainer:ActiveUncertaintyLearner")
    assert cap.supports_active_learning is True
    assert cap.default_strategy is DefaultStrategy.ACTIVE_LEARNING


# ---------------------------------------------------------------------------
# Data-assimilation / digital-twin declarations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    sorted(_TASK_7_5_ASSIMILATION_NAMES),
)
def test_assimilation_capability_state_space_filtering(name: str) -> None:
    """Task 6.7 assimilation surfaces inherit the STATE_SPACE_FILTERING strategy."""
    cap = require_uq_capability(name)
    assert cap.default_strategy is DefaultStrategy.STATE_SPACE_FILTERING
    assert cap.native_jax_kernel is True
    assert cap.source_package == "opifex"


def test_assimilation_update_loops_propagate_solver_uncertainty() -> None:
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
        cap = require_uq_capability(name)
        assert cap.supports_solver_uncertainty is True, name


# ---------------------------------------------------------------------------
# MLOps declarations.
# ---------------------------------------------------------------------------


def test_mlops_experiment_tracker_unsupported() -> None:
    """``opifex.mlops`` is a metric-publication surface — UNSUPPORTED placeholder."""
    cap = require_uq_capability("mlops:ExperimentTracker")
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED
    assert cap.source_package == "opifex"
    # Plan-mandated breadcrumb tying the placeholder to the underlying
    # monitoring/calibration surface.
    assert "opifex.uncertainty.monitoring" in cap.notes


# ---------------------------------------------------------------------------
# Monitoring / reporting declarations.
# ---------------------------------------------------------------------------


def test_build_reliability_report_calibration_strategy() -> None:
    """The Phase 5 reliability-report builder is a pure-function CALIBRATION surface."""
    cap = require_uq_capability("monitoring:build_reliability_report")
    assert cap.default_strategy is DefaultStrategy.CALIBRATION
    assert cap.native_jax_kernel is True
    assert cap.supports_calibration is True


def test_monitoring_inputs_metadata_only() -> None:
    """``MonitoringInputs`` is a provenance container; no computation flag."""
    cap = require_uq_capability("monitoring:MonitoringInputs")
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
        ("l2o:", _L2O_NAMES),
        ("trainer:", _TASK_7_5_TRAINER_NAMES),
        ("assimilation:", _TASK_7_5_ASSIMILATION_NAMES),
        ("mlops:", _TASK_7_5_MLOPS_NAMES),
        ("monitoring:", _TASK_7_5_MONITORING_NAMES),
    ],
)
def test_task_7_5_prefix_membership(prefix: str, expected: frozenset[str]) -> None:
    """Every Task 7.5 prefix is populated by exactly the expected name set."""
    observed = {name for name in UQ_CAPABILITIES if name.startswith(prefix)}
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
