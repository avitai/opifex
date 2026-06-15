r"""Per-family in-process registries for atomistic models, backbones and heads.

Mirrors ``src/opifex/uncertainty/registry.py`` (which wraps
``calibrax.core.registry.SingletonRegistry``): each family has its own singleton
registry keyed by name, with two Opifex policies layered on top --
duplicate-registration rejection and an actionable :meth:`require` error listing
the available names. ``register_*`` class-decorators self-register at import time
so simply importing a model/backbone/head module makes it discoverable.

This is **in-process** lookup only (swap a backbone/head by name). Persisted or
versioned model *artifacts* belong to ``opifex.platform.registry`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from calibrax.core.registry import SingletonRegistry


if TYPE_CHECKING:
    from collections.abc import Callable


_C = TypeVar("_C", bound=type)


class _AtomisticRegistry(SingletonRegistry[type]):
    """Base class adding duplicate-rejection and an actionable ``require``.

    Subclassed once per family so each family is an independent singleton.
    """

    def register(self, name: str, item: type) -> None:
        """Register ``item`` under ``name``, rejecting duplicate names.

        Raises:
            ValueError: If ``name`` is already registered (registrations are
                canonical; ``calibrax`` would silently overwrite).
        """
        if name in self:
            raise ValueError(
                f"{name!r} is already registered in {type(self).__name__}. "
                "Registrations are canonical -- use require()/get() to read, or "
                "choose a new name."
            )
        super().register(name, item)

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
