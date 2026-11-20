"""Refusals of :class:`ServableModelRegistry`, the name-to-class map behind reconstruction.

The registry is process-wide, so each test registers under a name of its own and removes
it afterwards.
"""

from collections.abc import Iterator

import pytest
from calibrax.core import DuplicateRegistrationError
from flax import nnx

from opifex.deployment.servable_registry import register_servable_model, ServableModelRegistry


_NAME = "servable_registry_refusal_probe"


class _FirstModel(nnx.Module):
    """A servable module registered first."""


class _SecondModel(nnx.Module):
    """A different servable module offered under the same name."""


@pytest.fixture
def registered_first() -> Iterator[None]:
    """Register :class:`_FirstModel` under the probe name, and remove it afterwards."""
    register_servable_model(_NAME)(_FirstModel)
    try:
        yield
    finally:
        ServableModelRegistry().remove(_NAME)


@pytest.mark.usefixtures("registered_first")
def test_a_taken_name_is_refused_and_keeps_the_first_class() -> None:
    """Registering a second class under a taken name raises and changes nothing."""
    with pytest.raises(DuplicateRegistrationError, match=f"'{_NAME}' is already registered"):
        register_servable_model(_NAME)(_SecondModel)

    assert ServableModelRegistry().require(_NAME) is _FirstModel


def test_a_class_that_is_not_an_nnx_module_is_refused() -> None:
    """Only ``nnx.Module`` subclasses can be reconstructed, so nothing else registers."""

    class _NotAModule:
        pass

    with pytest.raises(TypeError, match=r"must be an nnx\.Module subclass"):
        ServableModelRegistry().register(_NAME, _NotAModule)  # pyright: ignore[reportArgumentType]

    assert _NAME not in ServableModelRegistry()
