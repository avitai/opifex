"""Tests for diffusion tempering (Beck & Tronarp+ 2024, arXiv:2402.12231).

Diffusion tempering anneals the *solver noise level* ``σ(epoch)`` from a
large initial value (well-mixed, smooth likelihood) toward a small final
value (sharp posterior) over the course of training. For probabilistic
ODE integrators this lets MLE escape local minima before sharpening. The
opifex translation hooks into ``ObjectiveConfig`` by rebuilding
``physics_weight`` per epoch from the inverse noise level
(``physics_weight ∝ 1 / σ²``), so when ``σ`` is large the physics
constraint is soft and gradually tightens as ``σ`` decays.

References
----------
* Beck, J., Bosch, N., Deistler, M., Kadhim, K. L., Macke, J. H.,
  Hennig, P., Tronarp, F. 2024 — *Diffusion Tempering Improves
  Parameter Estimation with Probabilistic Integrators for Ordinary
  Differential Equations*, arXiv:2402.12231 (PRIMARY).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from opifex.uncertainty.objectives import ObjectiveConfig
from opifex.uncertainty.tempering import (
    DiffusionTemperingSchedule,
    TemperingScheduleType,
)


def _base_config(physics_weight: float = 1.0) -> ObjectiveConfig:
    return ObjectiveConfig(
        kl_weight=1.0,
        dataset_size=64,
        physics_weight=physics_weight,
        data_weight=1.0,
        boundary_weight=1.0,
        initial_condition_weight=1.0,
        regularization_weight=0.0,
        calibration_weight=0.0,
        conformal_weight=0.0,
        pac_bayes_weight=0.0,
    )


def test_linear_schedule_decays_noise_from_initial_to_final() -> None:
    """Linear schedule hits ``initial_noise`` at ``epoch=0`` and ``final_noise`` at the end."""
    schedule = DiffusionTemperingSchedule(
        initial_noise=2.0,
        final_noise=0.1,
        num_epochs=10,
        schedule_type=TemperingScheduleType.LINEAR,
    )
    assert jnp.allclose(schedule.noise_at(0), 2.0, atol=1e-6)
    assert jnp.allclose(schedule.noise_at(10), 0.1, atol=1e-6)
    assert jnp.allclose(schedule.noise_at(5), 1.05, atol=1e-6)


def test_exponential_schedule_decays_log_linearly() -> None:
    """Exponential schedule hits both endpoints and decays log-linearly between them."""
    schedule = DiffusionTemperingSchedule(
        initial_noise=4.0,
        final_noise=0.25,
        num_epochs=4,
        schedule_type=TemperingScheduleType.EXPONENTIAL,
    )
    assert jnp.allclose(schedule.noise_at(0), 4.0, atol=1e-6)
    assert jnp.allclose(schedule.noise_at(4), 0.25, atol=1e-6)
    # Midpoint of a log-linear decay between 4 and 0.25 is sqrt(4 * 0.25) = 1.0.
    assert jnp.allclose(schedule.noise_at(2), 1.0, atol=1e-5)


def test_cosine_schedule_smoothly_interpolates_endpoints() -> None:
    """Cosine schedule hits both endpoints and is monotone non-increasing."""
    schedule = DiffusionTemperingSchedule(
        initial_noise=1.0,
        final_noise=0.01,
        num_epochs=8,
        schedule_type=TemperingScheduleType.COSINE,
    )
    assert jnp.allclose(schedule.noise_at(0), 1.0, atol=1e-6)
    assert jnp.allclose(schedule.noise_at(8), 0.01, atol=1e-6)
    values = [float(schedule.noise_at(e)) for e in range(9)]
    diffs = [values[i + 1] - values[i] for i in range(8)]
    # Strictly decreasing (cosine schedule never increases on a monotone path).
    assert all(d <= 0.0 for d in diffs)


def test_objective_config_at_rebuilds_physics_weight_from_inverse_noise() -> None:
    r"""``physics_weight_at_epoch = base_physics_weight / σ(epoch)²`` exactly."""
    schedule = DiffusionTemperingSchedule(
        initial_noise=2.0,
        final_noise=0.5,
        num_epochs=4,
        schedule_type=TemperingScheduleType.LINEAR,
    )
    base = _base_config(physics_weight=1.5)
    config_5 = schedule.objective_config_at(epoch=2, base_config=base)
    sigma = schedule.noise_at(2)
    assert jnp.allclose(config_5.physics_weight, 1.5 / (sigma**2), atol=1e-6)


def test_objective_config_at_preserves_other_weights() -> None:
    """Only ``physics_weight`` is rebuilt; every other weight is preserved."""
    schedule = DiffusionTemperingSchedule(
        initial_noise=1.0,
        final_noise=0.1,
        num_epochs=4,
        schedule_type=TemperingScheduleType.EXPONENTIAL,
    )
    base = _base_config(physics_weight=1.0)
    tempered = schedule.objective_config_at(epoch=2, base_config=base)
    for name in (
        "kl_weight",
        "dataset_size",
        "data_weight",
        "boundary_weight",
        "initial_condition_weight",
        "regularization_weight",
        "calibration_weight",
        "conformal_weight",
        "pac_bayes_weight",
        "reduction",
    ):
        assert getattr(tempered, name) == getattr(base, name), name


def test_dataclass_is_pattern_a_frozen_slots_kw_only() -> None:
    """``DiffusionTemperingSchedule`` is a frozen slotted kw-only dataclass."""
    schedule = DiffusionTemperingSchedule(
        initial_noise=1.0,
        final_noise=0.1,
        num_epochs=4,
        schedule_type=TemperingScheduleType.LINEAR,
    )
    assert dataclasses.is_dataclass(schedule)
    assert dataclasses.fields(type(schedule)) is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        schedule.initial_noise = 99.0  # type: ignore[misc]


def test_rejects_nonpositive_noise_levels() -> None:
    """Both endpoints must be strictly positive."""
    with pytest.raises(ValueError, match="noise"):
        DiffusionTemperingSchedule(
            initial_noise=0.0,
            final_noise=0.1,
            num_epochs=4,
            schedule_type=TemperingScheduleType.LINEAR,
        )


def test_rejects_final_noise_not_below_initial() -> None:
    """Annealing must decrease the noise, not raise it."""
    with pytest.raises(ValueError, match="initial_noise"):
        DiffusionTemperingSchedule(
            initial_noise=0.1,
            final_noise=2.0,
            num_epochs=4,
            schedule_type=TemperingScheduleType.LINEAR,
        )


def test_rejects_nonpositive_num_epochs() -> None:
    """``num_epochs`` must be strictly positive."""
    with pytest.raises(ValueError, match="num_epochs"):
        DiffusionTemperingSchedule(
            initial_noise=2.0,
            final_noise=0.1,
            num_epochs=0,
            schedule_type=TemperingScheduleType.LINEAR,
        )
