"""The pure record shaping the MLflow backend logs: metadata parameters and flat metrics."""

from __future__ import annotations

import dataclasses
import statistics

import pytest
from hypothesis import given, settings, strategies as st

from opifex.mlops.experiment import L2OMetrics, NeuralOperatorMetrics, PhysicsMetadata
from opifex.mlops.records import flatten_physics_metrics, physics_metadata_params


_KEYS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8)
_FINITE = st.floats(allow_nan=False, allow_infinity=False, width=32)
_NESTED = st.recursive(
    _FINITE,
    lambda children: st.dictionaries(_KEYS, children, min_size=1, max_size=4),
    max_leaves=12,
)


def _l2o(**overrides: object) -> L2OMetrics:
    values: dict[str, object] = {
        "final_objective_value": 0.5,
        "convergence_iterations": 12,
        "convergence_time_seconds": 3.5,
        "meta_loss": 0.2,
        "adaptation_loss": 0.3,
        "meta_gradient_norm": 1.0,
        "few_shot_performance": {"task_a": 0.9},
        "zero_shot_performance": {"task_a": 0.7},
        "update_magnitude": 0.01,
        "gradient_scaling_factor": 2.0,
    }
    values.update(overrides)
    return L2OMetrics(**values)  # type: ignore[arg-type]


def _leaves(tree: dict[str, object], prefix: str) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_leaves(value, path))  # type: ignore[arg-type]
        else:
            flat[path] = float(value)  # type: ignore[arg-type]
    return flat


class TestFlattenPhysicsMetrics:
    """Every recorded value becomes one float under a dotted key."""

    def test_scalars_keep_their_field_names(self) -> None:
        metrics = NeuralOperatorMetrics(
            train_loss=0.5,
            val_loss=0.6,
            spectral_accuracy=0.9,
            relative_l2_error=0.1,
            max_absolute_error=0.2,
            pde_residual=0.01,
            conservation_error=0.001,
            boundary_condition_error=0.002,
            stability_measure=1.0,
            physical_consistency=0.99,
            inference_time_per_sample=0.003,
            memory_usage_mb=512.0,
        )

        flat = flatten_physics_metrics(metrics)

        assert flat["train_loss"] == 0.5
        assert flat["memory_usage_mb"] == 512.0
        assert all(isinstance(value, float) for value in flat.values())

    def test_unset_optional_fields_are_absent(self) -> None:
        flat = flatten_physics_metrics(_l2o())

        assert "task_similarity_scores" not in flat
        assert "learned_lr_schedule" not in flat

    def test_slotted_dataclasses_are_read_through_their_fields(self) -> None:
        metrics = _l2o()
        assert not hasattr(metrics, "__dict__")

        assert flatten_physics_metrics(metrics)["meta_loss"] == 0.2

    @settings(max_examples=60, deadline=None)
    @given(tree=_NESTED)
    def test_nested_mappings_flatten_to_dotted_leaves(
        self, tree: dict[str, object] | float
    ) -> None:
        if not isinstance(tree, dict):
            tree = {"leaf": tree}
        flat = flatten_physics_metrics(_l2o(few_shot_performance=tree))
        expected = _leaves(tree, "few_shot_performance")

        assert {key: flat[key] for key in expected} == expected
        assert len([key for key in flat if key.startswith("few_shot_performance.")]) == len(
            expected
        )

    @settings(max_examples=60, deadline=None)
    @given(values=st.lists(_FINITE, min_size=1, max_size=20))
    def test_number_sequences_become_mean_min_max(self, values: list[float]) -> None:
        flat = flatten_physics_metrics(_l2o(learned_lr_schedule=values))

        assert flat["learned_lr_schedule.min"] == min(values)
        assert flat["learned_lr_schedule.max"] == max(values)
        assert flat["learned_lr_schedule.mean"] == pytest.approx(statistics.fmean(values))

    def test_empty_sequences_are_absent(self) -> None:
        flat = flatten_physics_metrics(_l2o(learned_lr_schedule=[]))

        assert not any(key.startswith("learned_lr_schedule") for key in flat)

    def test_non_numeric_values_are_an_error(self) -> None:
        with pytest.raises(TypeError, match="learned_lr_schedule"):
            flatten_physics_metrics(_l2o(learned_lr_schedule=["fast", "slow"]))


class TestPhysicsMetadataParams:
    """Metadata becomes MLflow parameters under the ``physics.`` prefix."""

    def test_empty_metadata_has_no_params(self) -> None:
        assert physics_metadata_params(PhysicsMetadata()) == {}

    def test_every_kind_of_field_is_rendered(self) -> None:
        metadata = PhysicsMetadata(
            pde_type="navier_stokes",
            dimensionality=2,
            boundary_conditions=["dirichlet", "neumann"],
            physical_constants={"viscosity": 0.01},
            domain_bounds=(0.0, 1.0),
            grid_resolution=(64, 64),
            time_horizon=1.5,
            system_parameters={"reynolds_number": 1000},
            analytical_solution="taylor_green",
        )

        params = physics_metadata_params(metadata)

        assert params == {
            "physics.pde_type": "navier_stokes",
            "physics.dimensionality": 2,
            "physics.boundary_conditions": "dirichlet,neumann",
            "physics.physical_constants.viscosity": 0.01,
            "physics.domain_bounds": "(0.0, 1.0)",
            "physics.grid_resolution": "(64, 64)",
            "physics.time_horizon": 1.5,
            "physics.system_parameters.reynolds_number": 1000,
            "physics.analytical_solution": "taylor_green",
        }

    def test_every_metadata_field_has_a_rendering(self) -> None:
        samples: dict[str, object] = {
            "str": "x",
            "int": 1,
            "float": 1.0,
            "list": ["a"],
            "dict": {"k": 1.0},
            "tuple": (1, 2),
        }
        for field in dataclasses.fields(PhysicsMetadata):
            origin = str(field.type).split("|")[0].strip().split("[")[0]
            value = samples[origin]
            params = physics_metadata_params(PhysicsMetadata(**{field.name: value}))  # type: ignore[arg-type]

            assert params, field.name
            assert all(key.startswith(f"physics.{field.name}") for key in params), field.name
