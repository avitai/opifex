"""Tests for the model/solver/calibration UQ capability declarations (Task 7.2).

The plan exit criteria are:

1. Every expected name (model / solver / aggregation / subpackage /
   adapter / backend) is registered in the singleton
   :class:`UQRegistry` with the declared flags.
2. Subpackage entries (linalg / quadrature / statespace / curvature)
   carry ``native_jax_kernel=True`` and the plan-mandated
   :class:`DefaultStrategy` value.
3. Inference-backend entries (Pathfinder / SVGD / ADVI) carry
   ``default_strategy=DefaultStrategy.VARIATIONAL`` and
   ``source_package="blackjax"``.
4. ``ProbabilisticPINN`` declares ``native_bayesian=True``,
   ``supports_calibration=True``,
   ``default_strategy=DefaultStrategy.VARIATIONAL``, and
   ``native_nnx_module=True``.
5. ``MultiFidelityPINN`` declares the three adapter strategies via
   :meth:`UQCapability.with_adapter`.
6. Solver-side aggregation utilities declare
   ``supports_solver_uncertainty=True``,
   ``native_jax_kernel=True``,
   ``default_strategy=DefaultStrategy.PROBABILISTIC_NUMERICS``, and
   ``source_package="opifex"``.
7. Task 7.2 does NOT add capability declarations for scientific-domain
   surfaces (equation discovery, quantum, optimization/L2O,
   data-assimilation, monitoring/reporting) — those belong to Task 7.5.

Closing the catalogue (criterion 7) is enforced by asserting the
registry currently exposes exactly the union of Task 7.1 + Task 7.2
names; any Task 7.5 addition will need to extend this expected set.
"""

from __future__ import annotations

import pytest

from opifex.neural.bayesian._uq_capabilities import BAYESIAN_MODEL_CAPABILITIES
from opifex.neural.operators._uq_capabilities import _OPERATOR_CAPABILITIES
from opifex.solvers._uq_capabilities import SOLVER_CAPABILITIES
from opifex.uncertainty.adapters._uq_capabilities import ADAPTER_CAPABILITIES
from opifex.uncertainty.curvature._uq_capabilities import CURVATURE_CAPABILITIES
from opifex.uncertainty.inference_backends._uq_capabilities import (
    INFERENCE_BACKEND_CAPABILITIES,
)
from opifex.uncertainty.linalg._uq_capabilities import LINALG_CAPABILITIES
from opifex.uncertainty.quadrature._uq_capabilities import QUADRATURE_CAPABILITIES
from opifex.uncertainty.registry import DefaultStrategy, UQCapability, UQRegistry
from opifex.uncertainty.statespace._uq_capabilities import STATESPACE_CAPABILITIES


# Other test modules (``test_registry.py`` and the capability-declarations
# integration test) use an ``autouse`` :meth:`UQRegistry.reset` fixture to
# isolate their own state. That fixture wipes every registration this file
# depends on, so we re-seed the singleton here before each test rather than
# relying on import-time side effects (Rule 13 — no mutable side effects
# beyond constants + idempotent registration).
_ALL_TASK_7_X_CAPABILITIES: dict[str, UQCapability] = {
    **_OPERATOR_CAPABILITIES,
    **BAYESIAN_MODEL_CAPABILITIES,
    **SOLVER_CAPABILITIES,
    **ADAPTER_CAPABILITIES,
    **CURVATURE_CAPABILITIES,
    **INFERENCE_BACKEND_CAPABILITIES,
    **LINALG_CAPABILITIES,
    **QUADRATURE_CAPABILITIES,
    **STATESPACE_CAPABILITIES,
}


@pytest.fixture(autouse=True)
def _seed_registry() -> None:  # pyright: ignore[reportUnusedFunction]
    """Re-seed the singleton ``UQRegistry`` with every Task 7.1 + 7.2 entry.

    Idempotent — registers only names not already present so consecutive
    fixture invocations are safe.
    """
    registry = UQRegistry()
    for name, capability in _ALL_TASK_7_X_CAPABILITIES.items():
        if name not in registry:
            registry.register(name, capability)


# ---------------------------------------------------------------------------
# Expected registry contents (Task 7.1 + Task 7.2).
# ---------------------------------------------------------------------------


# Task 7.1 — 19 operator declarations registered under bare operator
# names ("FNO", "UQNO", ...).
_TASK_7_1_OPERATOR_NAMES: frozenset[str] = frozenset(
    {
        "FNO",
        "DeepONet",
        "PINO",
        "TFNO",
        "UFNO",
        "SFNO",
        "LocalFNO",
        "AM-FNO",
        "MS-FNO",
        "FourierDeepONet",
        "AdaptiveDeepONet",
        "MultiPhysicsDeepONet",
        "GINO",
        "MGNO",
        "UQNO",
        "LNO",
        "WNO",
        "GNO",
        "OperatorNet",
    }
)


_TASK_7_2_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "model:ProbabilisticPINN",
        "model:MultiFidelityPINN",
        "model:deterministic_baseline",
    }
)


_TASK_7_2_SOLVER_NAMES: frozenset[str] = frozenset(
    {
        "solver:PINNSolver",
        "solver:HybridSolver",
        "solver:NeuralOperatorSolver",
        "solver:aggregate_solver_solutions",
        "solver:summarize_stacked_sample_solution",
    }
)


_TASK_7_2_SUBPACKAGE_NAMES: frozenset[str] = frozenset(
    {
        "subpackage:linalg",
        "subpackage:quadrature",
        "subpackage:statespace",
        "subpackage:curvature",
    }
)


_TASK_7_2_BACKEND_NAMES: frozenset[str] = frozenset(
    {
        "backend:pathfinder",
        "backend:svgd",
        "backend:advi",
    }
)


_TASK_7_2_ADAPTER_NAMES: frozenset[str] = frozenset(
    {
        # GP adapter specs (Task 6.3.4 inventory).
        "adapter:GPJaxAdapterSpec",
        "adapter:TinygpAdapterSpec",
        "adapter:MarkovflowAdapterSpec",
        "adapter:BayesnewtonAdapterSpec",
        "adapter:KalmanJaxAdapterSpec",
        # Phase 4 model-uncertainty adapter specs.
        "adapter:LaplaceAdapterSpec",
        "adapter:MCDropoutAdapter",
        "adapter:BayesianLastLayerAdapterSpec",
        "adapter:SNGPAdapterSpec",
        "adapter:VBLLAdapterSpec",
        # Phase 4 ensemble adapter specs.
        "adapter:DeepEnsembleAdapter",
        "adapter:SnapshotEnsembleAdapterSpec",
        "adapter:SWAGAdapterSpec",
        "adapter:BatchEnsembleAdapterSpec",
        "adapter:DUEAdapterSpec",
        "adapter:TestTimeAugmentationAdapterSpec",
        # Phase 4 calibration / conformal concrete calibrators.
        "calibration:TemperatureScaling",
        "conformal:SplitConformalRegressor",
        "conformal:ConformalizedQuantileRegressor",
        "conformal:GroupedSplitConformalRegressor",
        "conformal:LACConformalClassifier",
        "conformal:FieldSplitConformalRegressor",
        "conformal:RiskControllerState",
    }
)


_TASK_7_2_EXPECTED: frozenset[str] = (
    _TASK_7_2_MODEL_NAMES
    | _TASK_7_2_SOLVER_NAMES
    | _TASK_7_2_SUBPACKAGE_NAMES
    | _TASK_7_2_BACKEND_NAMES
    | _TASK_7_2_ADAPTER_NAMES
)


_REGISTRY_EXPECTED: frozenset[str] = _TASK_7_1_OPERATOR_NAMES | _TASK_7_2_EXPECTED


# ---------------------------------------------------------------------------
# Coverage tests — every expected name is present.
# ---------------------------------------------------------------------------


@pytest.fixture
def uq_registry() -> UQRegistry:
    """Return the shared singleton ``UQRegistry`` after subpackage imports."""
    return UQRegistry()


@pytest.mark.parametrize("name", sorted(_TASK_7_2_EXPECTED))
def test_task_7_2_capability_name_is_registered(name: str, uq_registry: UQRegistry) -> None:
    assert name in uq_registry, f"Task 7.2 expected name {name!r} missing from UQRegistry."


# ---------------------------------------------------------------------------
# Per-surface flag assertions.
# ---------------------------------------------------------------------------


def test_probabilistic_pinn_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("model:ProbabilisticPINN")
    assert cap.native_bayesian is True
    assert cap.supports_calibration is True
    assert cap.default_strategy is DefaultStrategy.VARIATIONAL
    assert cap.native_nnx_module is True


def test_multi_fidelity_pinn_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("model:MultiFidelityPINN")
    assert cap.default_strategy is DefaultStrategy.DETERMINISTIC
    assert cap.native_nnx_module is True
    assert cap.supports_ensemble is True
    assert cap.supports_conformal is True
    assert cap.supports_calibration is True


def test_deterministic_baseline_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("model:deterministic_baseline")
    assert cap.default_strategy is DefaultStrategy.DETERMINISTIC
    assert cap.native_nnx_module is True
    assert cap.supports_ensemble is True
    assert cap.supports_conformal is True
    assert cap.supports_calibration is True


@pytest.mark.parametrize(
    "name",
    [
        "solver:PINNSolver",
        "solver:HybridSolver",
        "solver:NeuralOperatorSolver",
    ],
)
def test_solver_entry_point_capability_flags(name: str, uq_registry: UQRegistry) -> None:
    """Each solver entry point inherits the three-adapter baseline."""
    cap = uq_registry.require(name)
    assert cap.default_strategy is DefaultStrategy.DETERMINISTIC
    assert cap.native_nnx_module is True
    assert cap.supports_ensemble is True
    assert cap.supports_conformal is True
    assert cap.supports_calibration is True


@pytest.mark.parametrize(
    "name",
    [
        "solver:aggregate_solver_solutions",
        "solver:summarize_stacked_sample_solution",
    ],
)
def test_solver_aggregation_utility_capability_flags(name: str, uq_registry: UQRegistry) -> None:
    cap = uq_registry.require(name)
    assert cap.supports_solver_uncertainty is True
    assert cap.native_jax_kernel is True
    assert cap.default_strategy is DefaultStrategy.PROBABILISTIC_NUMERICS
    assert cap.source_package == "opifex"


# ---------------------------------------------------------------------------
# Subpackage capability declarations.
# ---------------------------------------------------------------------------


def test_linalg_subpackage_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("subpackage:linalg")
    assert cap.native_jax_kernel is True
    assert cap.default_strategy is DefaultStrategy.RANDOMIZED_LINALG
    assert cap.source_package == "matfree+traceax"


def test_quadrature_subpackage_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("subpackage:quadrature")
    assert cap.native_jax_kernel is True
    assert cap.default_strategy is DefaultStrategy.BAYESIAN_QUADRATURE
    assert cap.source_package == "emukit"


def test_statespace_subpackage_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("subpackage:statespace")
    assert cap.native_jax_kernel is True
    assert cap.default_strategy is DefaultStrategy.STATE_SPACE_FILTERING
    # Plan-mandated CAKF reference cite.
    assert "Pförtner" in cap.notes
    assert "arXiv:2306.07879" in cap.notes


def test_curvature_subpackage_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("subpackage:curvature")
    assert cap.native_jax_kernel is True
    assert cap.default_strategy is DefaultStrategy.LAPLACE
    assert cap.source_package == "traceax+matfree+kfac-jax"


# ---------------------------------------------------------------------------
# Inference-backend declarations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["backend:pathfinder", "backend:svgd", "backend:advi"],
)
def test_inference_backend_capability_flags(name: str, uq_registry: UQRegistry) -> None:
    cap = uq_registry.require(name)
    assert cap.native_jax_kernel is True
    assert cap.default_strategy is DefaultStrategy.VARIATIONAL
    assert cap.source_package == "blackjax"


# ---------------------------------------------------------------------------
# GP adapter spec declarations — five named entries (plan §7.2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_source_package"),
    [
        ("adapter:GPJaxAdapterSpec", "gpjax"),
        ("adapter:TinygpAdapterSpec", "tinygp"),
        ("adapter:MarkovflowAdapterSpec", "markovflow"),
        ("adapter:BayesnewtonAdapterSpec", "bayesnewton"),
        ("adapter:KalmanJaxAdapterSpec", "kalman-jax"),
    ],
)
def test_gp_adapter_spec_capability_source_package(
    name: str, expected_source_package: str, uq_registry: UQRegistry
) -> None:
    cap = uq_registry.require(name)
    assert cap.source_package == expected_source_package
    # Plan §7.2: GP adapter specs declare UNSUPPORTED until a backend lands.
    assert cap.default_strategy is DefaultStrategy.UNSUPPORTED


# ---------------------------------------------------------------------------
# Phase 4 conformal / calibration adapter declarations.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "conformal:SplitConformalRegressor",
        "conformal:ConformalizedQuantileRegressor",
        "conformal:GroupedSplitConformalRegressor",
        "conformal:LACConformalClassifier",
        "conformal:FieldSplitConformalRegressor",
        "conformal:RiskControllerState",
    ],
)
def test_conformal_calibrator_capability_flags(name: str, uq_registry: UQRegistry) -> None:
    cap = uq_registry.require(name)
    assert cap.supports_conformal is True
    assert cap.default_strategy is DefaultStrategy.CONFORMAL


def test_temperature_scaling_capability_flags(uq_registry: UQRegistry) -> None:
    cap = uq_registry.require("calibration:TemperatureScaling")
    assert cap.supports_calibration is True
    assert cap.default_strategy is DefaultStrategy.CALIBRATION


def test_risk_controller_uses_calibrax_source(uq_registry: UQRegistry) -> None:
    """RiskController reuses calibrax.statistics.analyzer for the inner statistics."""
    cap = uq_registry.require("conformal:RiskControllerState")
    assert cap.source_package == "calibrax"


# ---------------------------------------------------------------------------
# Scope closure — Task 7.5 surfaces must NOT be registered yet.
# ---------------------------------------------------------------------------


_TASK_7_5_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "equation_discovery:",
    "quantum:",
    "optimization:",
    "data_assimilation:",
    "monitoring:",
    "reporting:",
)


def test_task_7_2_does_not_add_task_7_5_surfaces(uq_registry: UQRegistry) -> None:
    """Phase 7 Task 7.2 scope excludes scientific-domain surfaces (Task 7.5)."""
    for name in uq_registry.list_names():
        for forbidden in _TASK_7_5_FORBIDDEN_PREFIXES:
            assert not name.startswith(forbidden), (
                f"Capability name {name!r} appears in Task 7.2 scope but its "
                f"prefix {forbidden!r} belongs to Task 7.5."
            )


def test_registry_contents_match_task_7_1_plus_task_7_2(uq_registry: UQRegistry) -> None:
    """Closed-world assertion — every registered name is from Task 7.1 or 7.2."""
    registered = frozenset(uq_registry.list_names())
    unexpected = registered - _REGISTRY_EXPECTED
    missing = _REGISTRY_EXPECTED - registered
    assert not unexpected, f"Unexpected capability names registered: {sorted(unexpected)!r}."
    assert not missing, f"Missing expected capability names: {sorted(missing)!r}."
