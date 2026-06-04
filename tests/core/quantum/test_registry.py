r"""Tests for the per-family atomistic registries and ``register_*`` decorators.

The registries mirror ``src/opifex/uncertainty/registry.py`` (which wraps
``calibrax.core.registry.SingletonRegistry``): in-process name-keyed lookup with
duplicate-rejection and an actionable ``require()`` error listing available
names. Tests reset each singleton for isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from opifex.core.quantum.registry import (
    AtomisticModelRegistry,
    BackboneRegistry,
    PropertyHeadRegistry,
    register_atomistic_model,
    register_backbone,
    register_property_head,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_registries() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Give each test empty registries, then restore the import-time state.

    The registries are process-wide singletons populated by import-time
    ``register_*`` decorators, so they need a clean slate per test. A bare
    ``reset()`` would, however, leak the wipe to every later test module whose
    backbones/heads self-registered at import (those decorators do not re-run --
    the modules are already cached). Snapshot first, reset for isolation, then
    restore on teardown so the wipe cannot escape this file.
    """
    registries = (AtomisticModelRegistry, BackboneRegistry, PropertyHeadRegistry)
    saved = {
        registry: {name: registry().get(name) for name in registry().list_names()}
        for registry in registries
    }
    for registry in registries:
        registry.reset()
    try:
        yield
    finally:
        for registry in registries:
            registry.reset()
            for name, item in saved[registry].items():
                registry().register(name, item)


class TestRegistration:
    def test_register_and_require_model(self) -> None:
        @register_atomistic_model("dummy")
        class _DummyModel:
            pass

        assert AtomisticModelRegistry().require("dummy") is _DummyModel

    def test_register_backbone(self) -> None:
        @register_backbone("dummy_backbone")
        class _DummyBackbone:
            pass

        assert BackboneRegistry().require("dummy_backbone") is _DummyBackbone

    def test_register_property_head(self) -> None:
        @register_property_head("dummy_head")
        class _DummyHead:
            pass

        assert PropertyHeadRegistry().require("dummy_head") is _DummyHead

    def test_decorator_returns_class_unchanged(self) -> None:
        @register_atomistic_model("identity")
        class _Model:
            pass

        assert _Model.__name__ == "_Model"


class TestPolicies:
    def test_duplicate_registration_rejected(self) -> None:
        @register_backbone("collide")
        class _First:  # pyright: ignore[reportUnusedClass]
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_backbone("collide")
            class _Second:  # pyright: ignore[reportUnusedClass]
                pass

    def test_require_unknown_lists_available(self) -> None:
        @register_property_head("present")
        class _Head:  # pyright: ignore[reportUnusedClass]
            pass

        with pytest.raises(KeyError, match="present"):
            PropertyHeadRegistry().require("absent")

    def test_registries_are_independent(self) -> None:
        @register_backbone("shared_name")
        class _Backbone:
            pass

        # The same name in a different family is fine — registries are separate.
        @register_atomistic_model("shared_name")
        class _Model:
            pass

        assert BackboneRegistry().require("shared_name") is _Backbone
        assert AtomisticModelRegistry().require("shared_name") is _Model
