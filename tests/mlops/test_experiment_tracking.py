"""The experiment tracker, its configuration records and the mlops package surface."""

from __future__ import annotations

import importlib
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest

import opifex.mlops
from opifex.mlops import register_mlops_capabilities
from opifex.mlops._uq_capabilities import MLOPS_CAPABILITIES
from opifex.mlops.backends.mlflow_backend import MLflowBackend
from opifex.mlops.experiment import (
    Experiment,
    ExperimentConfig,
    Framework,
    PhysicsDomain,
    PhysicsMetadata,
)
from opifex.mlops.tracker import ExperimentTracker
from opifex.uncertainty.registry import DefaultStrategy, UQRegistry


if TYPE_CHECKING:
    from typing import Any


def _config(**overrides: Any) -> ExperimentConfig:
    values: dict[str, Any] = {
        "name": "test_experiment",
        "physics_domain": PhysicsDomain.NEURAL_OPERATORS,
        "framework": Framework.JAX,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


class TestPhysicsMetadata:
    def test_every_field_defaults_to_none(self) -> None:
        assert all(value is None for value in asdict(PhysicsMetadata()).values())

    def test_fields_are_kept_as_given(self) -> None:
        metadata = PhysicsMetadata(
            pde_type="navier_stokes",
            dimensionality=2,
            boundary_conditions=["dirichlet"],
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            time_horizon=1.0,
            material_properties={"viscosity": 0.01},
            system_parameters={"reynolds_number": 1000},
        )

        assert asdict(metadata) == {
            **asdict(PhysicsMetadata()),
            "pde_type": "navier_stokes",
            "dimensionality": 2,
            "boundary_conditions": ["dirichlet"],
            "domain_bounds": (0.0, 1.0, 0.0, 1.0),
            "time_horizon": 1.0,
            "material_properties": {"viscosity": 0.01},
            "system_parameters": {"reynolds_number": 1000},
        }


class TestExperimentConfig:
    def test_defaults(self) -> None:
        config = _config()

        assert config.description is None
        assert config.tags == []
        assert config.backend == "auto"
        assert config.physics_metadata is None
        assert config.research_group is None

    def test_custom_values_are_kept(self) -> None:
        metadata = PhysicsMetadata(pde_type="navier_stokes")
        config = _config(
            name="fluid_dynamics_experiment",
            tags=["neural_operator", "fluid_dynamics"],
            physics_metadata=metadata,
            backend="mlflow",
            research_group="opifex_team",
            random_seed=42,
        )

        assert config.name == "fluid_dynamics_experiment"
        assert config.tags == ["neural_operator", "fluid_dynamics"]
        assert config.physics_metadata is metadata
        assert config.backend == "mlflow"
        assert config.research_group == "opifex_team"
        assert config.random_seed == 42


class TestEnums:
    def test_physics_domains(self) -> None:
        assert {domain.value for domain in PhysicsDomain} == {
            "neural-operators",
            "l2o",
            "neural-dft",
            "pinn",
            "quantum-computing",
        }

    def test_jax_is_the_only_framework(self) -> None:
        assert [framework.value for framework in Framework] == ["jax"]


class _NullExperiment(Experiment):
    """An experiment that records nothing."""

    async def start(self) -> str:
        return "null"

    async def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        pass

    async def log_physics_metrics(self, metrics: object, step: int | None = None) -> None:
        pass

    async def log_parameters(self, params: dict[str, Any]) -> None:
        pass

    async def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        pass

    async def log_model(
        self, model: object, model_name: str, physics_metadata: PhysicsMetadata | None = None
    ) -> None:
        pass

    async def end(self, status: str = "completed") -> None:
        pass


class TestExperimentTracker:
    def test_mlflow_is_registered_by_default(self) -> None:
        tracker = ExperimentTracker()

        assert tracker.default_backend == "mlflow"
        assert tracker.backends == ("mlflow",)

    def test_registered_backends_are_listed(self) -> None:
        tracker = ExperimentTracker()

        tracker.register_backend("null", _NullExperiment)

        assert tracker.backends == ("mlflow", "null")

    @pytest.mark.asyncio
    async def test_auto_resolves_to_the_tracker_default(self) -> None:
        tracker = ExperimentTracker(default_backend="null")
        tracker.register_backend("null", _NullExperiment)

        experiment = await tracker.create_experiment(_config(backend="auto"))

        assert isinstance(experiment, _NullExperiment)

    @pytest.mark.asyncio
    async def test_the_config_backend_wins_over_the_default(self) -> None:
        tracker = ExperimentTracker(default_backend="null")
        tracker.register_backend("null", _NullExperiment)

        experiment = await tracker.create_experiment(_config(backend="mlflow"))

        assert isinstance(experiment, MLflowBackend)

    @pytest.mark.asyncio
    async def test_an_unknown_backend_names_the_registered_ones(self) -> None:
        tracker = ExperimentTracker()

        with pytest.raises(ValueError, match=r"'wandb'.*mlflow"):
            await tracker.create_experiment(_config(backend="wandb"))


class TestPackageSurface:
    def test_public_names(self) -> None:
        assert set(opifex.mlops.__all__) == {
            "MLOPS_CAPABILITIES",
            "Experiment",
            "ExperimentConfig",
            "ExperimentTracker",
            "Framework",
            "L2OMetrics",
            "MLflowBackend",
            "NeuralDFTMetrics",
            "NeuralOperatorMetrics",
            "PINNMetrics",
            "PhysicsDomain",
            "PhysicsMetadata",
            "QuantumMetrics",
            "register_mlops_capabilities",
        }
        assert opifex.mlops.ExperimentTracker is ExperimentTracker
        for name in (
            "SUPPORTED_FRAMEWORKS",
            "SUPPORTED_BACKENDS",
            "MLFLOW_AVAILABLE",
            "__version__",
        ):
            assert not hasattr(opifex.mlops, name), name

    def test_importing_the_package_does_not_register_capabilities(self) -> None:
        UQRegistry.reset()
        try:
            importlib.reload(opifex.mlops)

            assert "mlops:ExperimentTracker" not in UQRegistry()
        finally:
            register_mlops_capabilities(UQRegistry())

    def test_register_mlops_capabilities_is_idempotent(self) -> None:
        registry = UQRegistry()

        register_mlops_capabilities(registry)
        register_mlops_capabilities(registry)

        capability = registry.require("mlops:ExperimentTracker")
        assert capability is MLOPS_CAPABILITIES["mlops:ExperimentTracker"]
        assert capability.default_strategy is DefaultStrategy.UNSUPPORTED
