"""Tests for the datarax-backed PDE loader factory.

The factory generates a dataset eagerly (jit+vmap) and serves it through
datarax ``MemorySource`` + ``Pipeline``, returning a :class:`PDELoaders`
(train + val), mirroring ``create_rmd17_loader``. These tests pin the public
contract: batch shapes/keys, the train/val split, and shuffle determinism.
"""

import numpy as np
import pytest

from opifex.data.loaders.factory import (
    create_burgers_loader,
    create_darcy_loader,
    create_diffusion_loader,
    create_navier_stokes_loader,
    create_shallow_water_loader,
    PDELoaders,
)


# (factory, channels, spatial_ndim) — channels-first {"input","output"} layout.
_PDE_CASES = [
    (create_burgers_loader, 1, 1),
    (create_darcy_loader, 1, 2),
    (create_diffusion_loader, 1, 2),
    (create_navier_stokes_loader, 2, 2),
    (create_shallow_water_loader, 3, 2),
]
_FACTORIES = [case[0] for case in _PDE_CASES]


def _first_batch(pipeline) -> dict[str, np.ndarray]:
    """Return the first batch yielded by a datarax pipeline."""
    for batch in pipeline:
        return batch
    raise AssertionError("pipeline yielded no batches")


@pytest.mark.parametrize("factory", _FACTORIES)
def test_loader_returns_pde_loaders_with_split(factory) -> None:
    """Every factory returns a PDELoaders whose train/val sizes match the split."""
    loaders = factory(n_samples=10, resolution=16, batch_size=4, val_fraction=0.2, seed=0)

    assert isinstance(loaders, PDELoaders)
    assert loaders.n_train == 8
    assert loaders.n_val == 2
    assert loaders.n_train + loaders.n_val == 10
    assert loaders.resolution == 16


@pytest.mark.parametrize(("factory", "channels", "spatial_ndim"), _PDE_CASES)
def test_batches_have_channel_first_input_output(factory, channels, spatial_ndim) -> None:
    """Batches expose only ``input``/``output`` with channels-first spatial shape."""
    loaders = factory(n_samples=10, resolution=16, batch_size=4, seed=0)

    batch = _first_batch(loaders.train)
    assert set(batch) == {"input", "output"}
    spatial = (16,) * spatial_ndim
    assert batch["input"].shape == (4, channels, *spatial)
    assert batch["output"].shape == (4, channels, *spatial)
    assert np.all(np.isfinite(batch["input"]))
    assert np.all(np.isfinite(batch["output"]))


def test_validation_pipeline_is_sequential_and_deterministic() -> None:
    """The val split is unshuffled, so two fresh loaders yield identical first batches.

    A datarax ``Pipeline`` is single-pass (its ``_position`` advances and does
    not auto-reset), so determinism is checked across two freshly built loaders
    rather than by re-iterating one exhausted pipeline.
    """
    first = _first_batch(
        create_burgers_loader(n_samples=10, resolution=16, batch_size=2, seed=0).val
    )["input"]
    second = _first_batch(
        create_burgers_loader(n_samples=10, resolution=16, batch_size=2, seed=0).val
    )["input"]
    np.testing.assert_array_equal(first, second)


def test_same_seed_reproduces_training_order() -> None:
    """Identical seeds reproduce the shuffled training order; a new seed differs."""
    same_a = _first_batch(
        create_burgers_loader(n_samples=12, resolution=16, batch_size=4, seed=0).train
    )["input"]
    same_b = _first_batch(
        create_burgers_loader(n_samples=12, resolution=16, batch_size=4, seed=0).train
    )["input"]
    other = _first_batch(
        create_burgers_loader(n_samples=12, resolution=16, batch_size=4, seed=1).train
    )["input"]

    np.testing.assert_array_equal(same_a, same_b)
    assert not np.array_equal(same_a, other)
