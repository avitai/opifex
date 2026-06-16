"""Tests for the vmapped PDE-dataset generation module.

Written test-first (TDD). Every generator returns the uniform contract

    {"input": (n, C_in, *spatial) float32, "output": (n, C_out, *spatial) float32}

(channels-first, FNO layout) mapping a conditioning field to the FINAL-time
solution. These tests assert that contract: exact shapes, ``float32`` dtype,
all-finite values, and determinism (same seed -> identical arrays; different
seed -> different arrays).
"""

import numpy as np
import pytest

from opifex.data.sources.pde_generation import (
    generate_burgers,
    generate_darcy,
    generate_diffusion,
    generate_navier_stokes,
    generate_shallow_water,
)


# Small configs keep the suite fast (the PDE solvers are the cost driver).
N_SAMPLES = 4
RESOLUTION = 16


def _assert_contract(
    dataset: dict[str, np.ndarray],
    *,
    n_samples: int,
    channels_in: int,
    channels_out: int,
    spatial: tuple[int, ...],
) -> None:
    """Assert the uniform generation contract for a returned dataset."""
    assert set(dataset) == {"input", "output"}
    input_array, output_array = dataset["input"], dataset["output"]

    assert input_array.shape == (n_samples, channels_in, *spatial)
    assert output_array.shape == (n_samples, channels_out, *spatial)

    assert input_array.dtype == np.float32
    assert output_array.dtype == np.float32

    assert np.all(np.isfinite(input_array))
    assert np.all(np.isfinite(output_array))


def _assert_deterministic(generate, kwargs: dict) -> None:
    """Same seed reproduces arrays; a different seed changes them."""
    first = generate(**kwargs)
    same = generate(**kwargs)
    np.testing.assert_array_equal(first["input"], same["input"])
    np.testing.assert_array_equal(first["output"], same["output"])

    different = generate(**{**kwargs, "seed": kwargs["seed"] + 1})
    assert not np.array_equal(first["input"], different["input"])
    assert not np.array_equal(first["output"], different["output"])


class TestGenerateBurgers:
    """1D Burgers: IC field -> final-time solution, 1 channel each."""

    def _kwargs(self) -> dict:
        return {"n_samples": N_SAMPLES, "resolution": RESOLUTION, "seed": 7}

    def test_contract(self) -> None:
        dataset = generate_burgers(**self._kwargs())
        _assert_contract(
            dataset,
            n_samples=N_SAMPLES,
            channels_in=1,
            channels_out=1,
            spatial=(RESOLUTION,),
        )

    def test_determinism(self) -> None:
        _assert_deterministic(generate_burgers, self._kwargs())


class TestGenerateDarcy:
    """Darcy: permeability coefficient -> steady solution, 1 channel each."""

    def _kwargs(self) -> dict:
        return {"n_samples": N_SAMPLES, "resolution": RESOLUTION, "seed": 11}

    def test_contract_smooth(self) -> None:
        dataset = generate_darcy(**self._kwargs())
        _assert_contract(
            dataset,
            n_samples=N_SAMPLES,
            channels_in=1,
            channels_out=1,
            spatial=(RESOLUTION, RESOLUTION),
        )

    def test_contract_binary(self) -> None:
        dataset = generate_darcy(field_type="binary", **self._kwargs())
        _assert_contract(
            dataset,
            n_samples=N_SAMPLES,
            channels_in=1,
            channels_out=1,
            spatial=(RESOLUTION, RESOLUTION),
        )

    def test_determinism(self) -> None:
        _assert_deterministic(generate_darcy, self._kwargs())


class TestGenerateDiffusion:
    """Diffusion-advection: IC field -> final state, 1 channel each."""

    def _kwargs(self) -> dict:
        return {"n_samples": N_SAMPLES, "resolution": RESOLUTION, "seed": 13}

    def test_contract(self) -> None:
        dataset = generate_diffusion(**self._kwargs())
        _assert_contract(
            dataset,
            n_samples=N_SAMPLES,
            channels_in=1,
            channels_out=1,
            spatial=(RESOLUTION, RESOLUTION),
        )

    def test_determinism(self) -> None:
        _assert_deterministic(generate_diffusion, self._kwargs())


class TestGenerateNavierStokes:
    """Navier-Stokes: [u0, v0] -> final [u, v], 2 channels each."""

    def _kwargs(self) -> dict:
        return {"n_samples": N_SAMPLES, "resolution": RESOLUTION, "seed": 17}

    def test_contract(self) -> None:
        dataset = generate_navier_stokes(**self._kwargs())
        _assert_contract(
            dataset,
            n_samples=N_SAMPLES,
            channels_in=2,
            channels_out=2,
            spatial=(RESOLUTION, RESOLUTION),
        )

    def test_determinism(self) -> None:
        _assert_deterministic(generate_navier_stokes, self._kwargs())


class TestGenerateShallowWater:
    """Shallow water: [h, u, v] init -> final [h, u, v], 3 channels each."""

    def _kwargs(self) -> dict:
        return {"n_samples": N_SAMPLES, "resolution": RESOLUTION, "seed": 19}

    def test_contract(self) -> None:
        dataset = generate_shallow_water(**self._kwargs())
        _assert_contract(
            dataset,
            n_samples=N_SAMPLES,
            channels_in=3,
            channels_out=3,
            spatial=(RESOLUTION, RESOLUTION),
        )

    def test_determinism(self) -> None:
        _assert_deterministic(generate_shallow_water, self._kwargs())


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
