"""Tests for the vmapped PDE-dataset generation module.

Written test-first (TDD). Every generator returns the uniform contract

    {"input": (n, C_in, *spatial) float32, "output": (n, C_out, *spatial) float32}

(channels-first, FNO layout) mapping a conditioning field to the FINAL-time
solution. These tests assert that contract: exact shapes, ``float32`` dtype,
all-finite values, and determinism (same seed -> identical arrays; different
seed -> different arrays).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.data.sources.pde_generation import (
    _diffusion_ic,
    _wrapped_gaussian,
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

    def test_shipped_defaults_are_finite_and_physical(self) -> None:
        """At the shipped resolution the labels obey the PDE's conservation laws.

        Advection-diffusion on a periodic domain conserves the mean exactly and
        strictly dissipates the L2 norm of every non-constant field when ``D > 0``.
        """
        dataset = generate_diffusion(n_samples=16)
        initial = dataset["input"].astype(np.float64)
        final = dataset["output"].astype(np.float64)
        assert np.all(np.isfinite(final))

        scale = np.abs(initial).max(axis=(1, 2, 3))
        mean_drift = np.abs(final.mean(axis=(1, 2, 3)) - initial.mean(axis=(1, 2, 3)))
        assert np.all(mean_drift <= 8 * np.finfo(np.float32).eps * scale)

        centred_initial = initial - initial.mean(axis=(1, 2, 3), keepdims=True)
        centred_final = final - final.mean(axis=(1, 2, 3), keepdims=True)
        energy_initial = np.sum(centred_initial**2, axis=(1, 2, 3))
        energy_final = np.sum(centred_final**2, axis=(1, 2, 3))
        assert np.all(energy_final < energy_initial)

    def test_zero_time_returns_the_initial_condition(self) -> None:
        dataset = generate_diffusion(time_final=0.0, **self._kwargs())
        np.testing.assert_allclose(
            dataset["output"], dataset["input"], rtol=0.0, atol=8 * float(np.finfo(np.float32).eps)
        )

    def test_initial_conditions_nest_across_resolutions(self) -> None:
        """The IC is sampled at ``x_j = j / N``, so a 2N grid contains the N grid."""
        for seed in range(12):
            key = jax.random.key(seed)
            coarse = _diffusion_ic(key, 16)
            fine = _diffusion_ic(key, 32)
            np.testing.assert_allclose(np.asarray(fine[::2, ::2]), np.asarray(coarse), atol=1e-6)

    def test_gaussian_initial_condition_is_periodic(self) -> None:
        """The widest Gaussian nearest the boundary is still one smooth periodic field."""
        grid = jnp.linspace(0.0, 1.0, 9)
        grid_x, grid_y = jnp.meshgrid(grid, grid, indexing="ij")
        bump = _wrapped_gaussian(grid_x, grid_y, jnp.asarray(0.3), jnp.asarray(0.7), 0.15)
        shifted = _wrapped_gaussian(
            grid_x + 1.0, grid_y - 1.0, jnp.asarray(0.3), jnp.asarray(0.7), 0.15
        )
        np.testing.assert_allclose(np.asarray(shifted), np.asarray(bump), atol=1e-6)
        np.testing.assert_allclose(np.asarray(bump[0]), np.asarray(bump[-1]), atol=1e-6)


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
