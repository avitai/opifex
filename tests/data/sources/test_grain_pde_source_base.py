"""Equivalence tests for shared Grain PDE source index validation + key derivation.

Task 12.3.10 consolidates an 11-line index-validation and deterministic-key block
that was byte-identical across all five Grain data sources. These tests pin the
exact public behaviour (valid sample shape, determinism, and the precise
``TypeError``/``IndexError`` raised for bad indices) so the refactor can be proven
behaviour-preserving.

The same suite must pass before the refactor (against the inlined blocks) and after
it (against the shared :class:`GrainPDESource` base).
"""

from typing import Any

import jax
import pytest

from opifex.data.sources import (
    BurgersDataSource,
    DarcyDataSource,
    DiffusionDataSource,
    NavierStokesDataSource,
    ShallowWaterDataSource,
)


def _build_sources() -> dict[str, tuple[Any, tuple[int, ...]]]:
    """Construct one small instance per source with its expected input shape.

    Returns:
        Mapping of source name to ``(source, expected_input_shape)``.
    """
    return {
        "burgers": (BurgersDataSource(n_samples=5, resolution=16, seed=42), (16, 16)),
        "darcy": (DarcyDataSource(n_samples=5, resolution=17, seed=42), (17, 17)),
        "diffusion": (DiffusionDataSource(n_samples=5, resolution=16, seed=42), (16, 16)),
        "navier_stokes": (
            NavierStokesDataSource(n_samples=5, resolution=32, seed=42),
            (2, 32, 32),
        ),
        "shallow_water": (
            ShallowWaterDataSource(n_samples=5, resolution=16, seed=42),
            (3, 16, 16),
        ),
    }


SOURCE_NAMES = list(_build_sources().keys())


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_valid_index_returns_expected_input_shape(name: str) -> None:
    """A valid integer index yields a dict whose ``input`` has the expected shape."""
    source, expected_shape = _build_sources()[name]
    sample = source[2]
    assert isinstance(sample, dict)
    assert "input" in sample
    assert tuple(sample["input"].shape) == expected_shape


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_deterministic_same_seed_same_output(name: str) -> None:
    """Two sources with the same seed return byte-identical samples for an index."""
    source_a, _ = _build_sources()[name]
    source_b, _ = _build_sources()[name]
    sample_a = source_a[3]
    sample_b = source_b[3]
    assert jax.numpy.array_equal(sample_a["input"], sample_b["input"])
    assert jax.numpy.array_equal(sample_a["output"], sample_b["output"])


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_slice_index_raises_type_error(name: str) -> None:
    """Slicing is rejected with the exact ``TypeError`` message."""
    source, _ = _build_sources()[name]
    with pytest.raises(TypeError, match="Slicing not supported, use integer index"):
        source[0:2]


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_float_index_raises_type_error(name: str) -> None:
    """A non-integer index is rejected with the exact ``TypeError`` message."""
    source, _ = _build_sources()[name]
    with pytest.raises(TypeError, match=r"Index must be an integer, got <class 'float'>"):
        source[1.0]  # type: ignore[index]


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_negative_index_raises_index_error(name: str) -> None:
    """A negative index is out of bounds with the exact ``IndexError`` message."""
    source, _ = _build_sources()[name]
    with pytest.raises(IndexError, match=r"Index -1 out of bounds for source with 5 samples"):
        source[-1]


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_out_of_bounds_index_raises_index_error(name: str) -> None:
    """An index >= n_samples is out of bounds with the exact ``IndexError`` message."""
    source, _ = _build_sources()[name]
    with pytest.raises(IndexError, match=r"Index 5 out of bounds for source with 5 samples"):
        source[5]


@pytest.mark.parametrize("name", SOURCE_NAMES)
def test_len_returns_n_samples(name: str) -> None:
    """``__len__`` reports the configured number of samples."""
    source, _ = _build_sources()[name]
    assert len(source) == 5
