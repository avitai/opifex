"""Fixtures shared by the uncertainty tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def float64() -> Iterator[None]:
    """Run the test with x64 enabled; the root bookend switches it off again after."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)
