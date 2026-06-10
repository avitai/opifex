"""In-process registry of servable ``nnx.Module`` classes.

Cross-process model reconstruction needs to map a stable, persisted *name*
back to the concrete :class:`flax.nnx.Module` subclass to instantiate. A fresh
process cannot trust a free-form import path, so the registry name is the
cross-process contract: registering a class makes it reconstructable, and
loading an unregistered name fails fast with an actionable error.

Mirrors :mod:`opifex.core.quantum.registry` and
:mod:`opifex.uncertainty.registry` — each reuses
``calibrax.core.registry.SingletonRegistry`` and layers two policies on top:
duplicate-registration rejection (registrations are canonical) and a
:meth:`require` error that lists the available names. The ``@register_servable_model``
class-decorator self-registers at import time, so importing a model module
makes it discoverable for deserialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from calibrax.core.registry import SingletonRegistry
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable


_M = TypeVar("_M", bound=type[nnx.Module])


class ServableModelRegistry(SingletonRegistry[type[nnx.Module]]):
    """Singleton registry mapping a name to a servable ``nnx.Module`` class.

    Extends CalibraX's :class:`SingletonRegistry` with duplicate-rejection and
    an actionable :meth:`require` error, matching the other Opifex per-family
    registries.
    """

    def register(self, name: str, item: type[nnx.Module]) -> None:
        """Register ``item`` under ``name``, rejecting duplicate names.

        Raises:
            TypeError: If ``item`` is not an :class:`flax.nnx.Module` subclass.
            ValueError: If ``name`` is already registered (registrations are
                canonical; ``calibrax`` would silently overwrite).
        """
        if not (isinstance(item, type) and issubclass(item, nnx.Module)):
            raise TypeError(
                f"Servable model {name!r} must be an nnx.Module subclass, got {item!r}."
            )
        if name in self:
            raise ValueError(
                f"Servable model {name!r} is already registered. Registrations are "
                "canonical -- use require()/get() to read, or choose a new name."
            )
        super().register(name, item)

    def require(self, name: str) -> type[nnx.Module]:
        """Return the class registered under ``name``.

        Raises:
            KeyError: If ``name`` is not registered; the message lists the
                available names so the caller can correct the typo or import
                the module that registers the class.
        """
        if name not in self:
            available = sorted(self.list_names())
            raise KeyError(
                f"Servable model {name!r} is not registered. Available: {available!r}. "
                "Import the module that defines and registers the class before loading."
            )
        return self.get(name)


def register_servable_model(name: str) -> Callable[[_M], _M]:
    """Class decorator registering an ``nnx.Module`` class under ``name``.

    Args:
        name: Stable identifier persisted in a model's reconstruction recipe.

    Returns:
        The decorator, which registers the class and returns it unchanged.
    """

    def decorator(cls: _M) -> _M:
        ServableModelRegistry().register(name, cls)
        return cls

    return decorator


__all__ = [
    "ServableModelRegistry",
    "register_servable_model",
]
