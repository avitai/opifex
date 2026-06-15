"""Tests for multi-fidelity PINN configuration immutability.

Tests for :class:`opifex.neural.bayesian.config.MultiFidelityConfig`.

Regression guard for R5 (immutable data containers): the config must be
a frozen dataclass with a ``low_fidelity`` default supplied via
``field(default_factory=...)`` rather than mutated in ``__post_init__``,
and the ``create_multifidelity_pinn`` factory must build configured
instances via ``dataclasses.replace`` rather than ``setattr``.
"""

from __future__ import annotations

import dataclasses

import pytest
from flax import nnx

from opifex.neural.bayesian.config import FidelityConfig, MultiFidelityConfig
from opifex.neural.bayesian.probabilistic_pinns import create_multifidelity_pinn


def test_multi_fidelity_config_is_frozen() -> None:
    """Reassigning a field on a built config must raise ``FrozenInstanceError``."""
    config = MultiFidelityConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.high_fidelity_count = 5  # type: ignore[misc]


def test_multi_fidelity_config_default_low_fidelity_populated() -> None:
    """The default ``low_fidelity`` is a populated ``FidelityConfig``."""
    config = MultiFidelityConfig()
    assert isinstance(config.low_fidelity, FidelityConfig)
    assert config.low_fidelity.data_points == 1000
    assert config.low_fidelity.spatial_resolution == 32


def test_multi_fidelity_config_replace_preserves_other_fields() -> None:
    """``dataclasses.replace`` updates one field and keeps defaults intact."""
    config = MultiFidelityConfig()
    updated = dataclasses.replace(config, high_fidelity_count=4)
    assert updated.high_fidelity_count == 4
    assert updated.uncertainty_threshold == config.uncertainty_threshold
    assert updated.fidelity_weights == config.fidelity_weights


def test_factory_applies_config_dict_overrides() -> None:
    """The factory builds a configured model honouring ``config_dict``."""
    rngs = nnx.Rngs(0)
    custom_low = {
        "data_points": 500,
        "noise_level": 0.02,
        "spatial_resolution": 16,
        "temporal_resolution": 25,
    }
    model = create_multifidelity_pinn(
        input_dim=2,
        output_dim=1,
        config_dict={
            "low_fidelity": custom_low,
            "high_fidelity_count": 3,
            "uncertainty_threshold": 0.25,
        },
        rngs=rngs,
    )

    assert model.config.high_fidelity_count == 3
    assert model.config.uncertainty_threshold == 0.25
    assert isinstance(model.config.low_fidelity, FidelityConfig)
    assert model.config.low_fidelity.data_points == 500
    assert model.config.low_fidelity.spatial_resolution == 16


def test_factory_accepts_fidelity_config_instance() -> None:
    """A ``FidelityConfig`` instance passed via ``config_dict`` is used directly."""
    rngs = nnx.Rngs(0)
    low = FidelityConfig(
        data_points=750,
        noise_level=0.03,
        spatial_resolution=8,
        temporal_resolution=10,
    )
    model = create_multifidelity_pinn(
        input_dim=2,
        output_dim=1,
        config_dict={"low_fidelity": low},
        rngs=rngs,
    )
    assert model.config.low_fidelity is low
