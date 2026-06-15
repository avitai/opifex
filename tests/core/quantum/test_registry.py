r"""Tests for the per-family atomistic registries and ``register_*`` decorators.

The registries mirror ``src/opifex/uncertainty/registry.py`` (which wraps
``calibrax.core.registry.SingletonRegistry``): in-process name-keyed lookup with
duplicate-rejection and an actionable ``require()`` error listing available
names. Tests reset each singleton for isolation.
"""

from __future__ import annotations

import pytest

from opifex.core.quantum.registry import (
    AtomisticModelRegistry,
    BackboneRegistry,
    PropertyHeadRegistry,
    register_atomistic_model,
    register_backbone,
    register_property_head,
)


@pytest.fixture(autouse=True)
def _reset_registries() -> None:  # pyright: ignore[reportUnusedFunction]
    AtomisticModelRegistry.reset()
    BackboneRegistry.reset()
    PropertyHeadRegistry.reset()


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
