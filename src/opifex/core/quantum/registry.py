r"""Per-family in-process registries for atomistic models, backbones and heads.

Each family has its own singleton registry keyed by name, a
``calibrax.core.registry.SingletonRegistry``. That registry refuses a name it already
holds with ``DuplicateRegistrationError``, so registrations are canonical; Opifex adds
an actionable :meth:`require` error listing the available names. ``register_*``
class-decorators self-register at import time so simply importing a
model/backbone/head module makes it discoverable.

This is **in-process** lookup only (swap a backbone/head by name). Persisted or
versioned model *artifacts* belong to ``opifex.platform.registry`` instead.
"""

from collections.abc import Callable
from typing import TypeVar

from calibrax.core.registry import SingletonRegistry


_C = TypeVar("_C", bound=type)


class _AtomisticRegistry(SingletonRegistry[type]):
    """Base class adding an actionable ``require``.

    Subclassed once per family so each family is an independent singleton.
    """

    def require(self, name: str) -> type:
        """Return the class registered under ``name``.

        Raises:
            KeyError: If ``name`` is not registered; the message lists the
                available names so the caller can correct the typo.
        """
        if name not in self:
            available = sorted(self.list_names())
            raise KeyError(
                f"{name!r} not registered in {type(self).__name__}. Available: {available!r}."
            )
        return self.get(name)


class AtomisticModelRegistry(_AtomisticRegistry):
    """Singleton registry of assembled atomistic-model classes."""


class BackboneRegistry(_AtomisticRegistry):
    """Singleton registry of backbone classes (embedding producers)."""


class PropertyHeadRegistry(_AtomisticRegistry):
    """Singleton registry of property-head classes (typed readouts)."""


def _make_decorator(registry: _AtomisticRegistry, name: str) -> Callable[[_C], _C]:
    def decorator(cls: _C) -> _C:
        registry.register(name, cls)
        return cls

    return decorator


def register_atomistic_model(name: str) -> Callable[[_C], _C]:
    """Class decorator registering an atomistic-model class under ``name``."""
    return _make_decorator(AtomisticModelRegistry(), name)


def register_backbone(name: str) -> Callable[[_C], _C]:
    """Class decorator registering a backbone class under ``name``."""
    return _make_decorator(BackboneRegistry(), name)


def register_property_head(name: str) -> Callable[[_C], _C]:
    """Class decorator registering a property-head class under ``name``."""
    return _make_decorator(PropertyHeadRegistry(), name)


__all__ = [
    "AtomisticModelRegistry",
    "BackboneRegistry",
    "PropertyHeadRegistry",
    "register_atomistic_model",
    "register_backbone",
    "register_property_head",
]
