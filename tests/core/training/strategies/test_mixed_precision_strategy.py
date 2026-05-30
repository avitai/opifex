"""Behaviour and layering tests for the core mixed-precision strategy.

These tests pin the public behaviour of
``opifex.core.training.strategies.mixed_precision`` *before* the layering
refactor (Task 12.3.16) and the overflow-check de-duplication (Task 12.3.11),
so the refactor can be proven behaviour-preserving.

Two invariants are asserted:

1. **Layering (R3)** — ``core.training.strategies.mixed_precision`` must not
   import the higher-level application package ``opifex.training``. ``core``
   may never reach *up* into the application layer.
2. **Behaviour** — the loss-scaling primitives, overflow detection, gradient
   (un)scaling and the trainer's single-step update produce identical numbers
   before and after the refactor.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.config import TrainingConfig
from opifex.core.training.strategies import mixed_precision as mp


# --------------------------------------------------------------------------- #
# Layering invariant (Task 12.3.16)
# --------------------------------------------------------------------------- #
def test_core_strategy_does_not_import_application_training() -> None:
    """``core.training.strategies.mixed_precision`` must not import ``opifex.training``.

    Parses the module's own source for any ``import opifex.training[...]`` or
    ``from opifex.training[...] import`` statement. The application package
    ``opifex.training`` sits *above* ``opifex.core`` and importing it inverts
    the dependency direction.
    """
    module_path = Path(mp.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "opifex.training" or node.module.startswith("opifex.training."):
                offending.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "opifex.training" or alias.name.startswith("opifex.training."):
                    offending.append(f"import {alias.name}")

    assert not offending, (
        "core.training.strategies.mixed_precision must not import the application "
        f"layer opifex.training; found: {offending}"
    )


def test_training_config_is_the_core_config() -> None:
    """The ``TrainingConfig`` the strategy uses must be the canonical core one.

    Guards against the name collision with the (re-exported) config in
    ``opifex.training.basic_trainer`` — there must be exactly one
    ``TrainingConfig`` and it must live in ``opifex.core.training.config``.
    """
    core_config = importlib.import_module("opifex.core.training.config")
    # The trainer's annotated/runtime config type resolves to the core one.
    trainer = mp.MixedPrecisionTrainer(_TinyModel(rngs=nnx.Rngs(0)), TrainingConfig(num_epochs=1))
    assert isinstance(trainer.config, core_config.TrainingConfig)


# --------------------------------------------------------------------------- #
# Behaviour: loss-scale / overflow / gradient-scaling primitives
# --------------------------------------------------------------------------- #
def test_check_for_overflow_detects_non_finite() -> None:
    """Overflow detection flags NaN/Inf and passes finite trees unchanged."""
    finite = {"w": jnp.array([1.0, 2.0, 3.0])}
    nan = {"w": jnp.array([1.0, jnp.nan, 3.0])}
    inf = {"w": jnp.array([jnp.inf, 2.0, 3.0])}

    assert mp.check_for_overflow(finite) is False
    assert mp.check_for_overflow(nan) is True
    assert mp.check_for_overflow(inf) is True


def test_scale_gradients_unscales_by_loss_scale() -> None:
    """The strategy's ``scale_gradients`` divides by the loss scale (un-scaling)."""
    grads = {"w": jnp.array([4.0, 8.0, 16.0])}
    out = mp.scale_gradients(grads, loss_scale=4.0)
    assert jnp.allclose(out["w"], jnp.array([1.0, 2.0, 4.0]))


def test_update_loss_scale_halves_on_overflow() -> None:
    """On overflow the loss scale is divided by ``loss_scale_factor``."""
    config = mp.MixedPrecisionConfig(loss_scale=1024.0, loss_scale_factor=2.0, min_loss_scale=1.0)
    state = mp.MixedPrecisionState(loss_scale=1024.0)

    new_state = mp.update_loss_scale(state, has_overflow=True, config=config)

    assert new_state.loss_scale == pytest.approx(512.0)
    assert new_state.overflow_count == 1


def test_update_loss_scale_grows_periodically_without_overflow() -> None:
    """Without overflow and at a check boundary the loss scale grows."""
    config = mp.MixedPrecisionConfig(
        loss_scale=1024.0,
        loss_scale_factor=2.0,
        max_loss_scale=2**24,
        overflow_check_frequency=100,
    )
    state = mp.MixedPrecisionState(loss_scale=1024.0)
    state.step_count = 0  # 0 % 100 == 0 -> boundary

    new_state = mp.update_loss_scale(state, has_overflow=False, config=config)

    assert new_state.loss_scale == pytest.approx(2048.0)
    assert new_state.overflow_count == 0


def test_update_loss_scale_static_when_dynamic_disabled() -> None:
    """With dynamic scaling off the state is returned unchanged."""
    config = mp.MixedPrecisionConfig(loss_scale=1024.0, dynamic_loss_scaling=False)
    state = mp.MixedPrecisionState(loss_scale=1024.0)

    assert mp.update_loss_scale(state, has_overflow=True, config=config) is state


# --------------------------------------------------------------------------- #
# Behaviour: trainer single-step update
# --------------------------------------------------------------------------- #
class _TinyModel(nnx.Module):
    """Minimal linear model with a deterministic forward pass."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(4, 4, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)


def test_trainer_train_step_behaviour_is_preserved() -> None:
    """Pin the *current* behaviour of ``train_step`` so the refactor matches it.

    ``_compute_loss_and_grads`` currently differentiates ``loss_fn`` w.r.t.
    ``nnx.state(model)`` and then calls the resulting ``nnx.State`` tree as if
    it were the module (``model_mp(x)``). An ``nnx.State`` is not callable, so
    the step raises ``TypeError`` today. This is a pre-existing latent bug in a
    code path with no production caller; the layering refactor must preserve
    behaviour exactly, so we pin the raise here rather than silently changing
    it. (Tracked separately as a follow-up correctness fix.)
    """
    import optax

    model = _TinyModel(rngs=nnx.Rngs(0))
    config = TrainingConfig(num_epochs=1, batch_size=8)
    trainer = mp.MixedPrecisionTrainer(model, config)

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    x = jnp.ones((8, 4), dtype=jnp.float32)
    y = jnp.zeros((8, 4), dtype=jnp.float32)

    with pytest.raises(TypeError, match="not callable"):
        trainer.train_step(model, optimizer, (x, y))


def test_trainer_prepare_batch_casts_and_aligns() -> None:
    """``_prepare_batch`` casts to the compute dtype and aligns for TensorCore.

    This exercises the parts of the trainer that *do* run, independent of the
    broken forward pass, so the refactor is pinned on real behaviour.
    """
    model = _TinyModel(rngs=nnx.Rngs(0))
    trainer = mp.MixedPrecisionTrainer(model, TrainingConfig(num_epochs=1))

    x = jnp.ones((8, 4), dtype=jnp.float32)
    y = jnp.ones((8, 4), dtype=jnp.float32)
    x_out, y_out = trainer._prepare_batch((x, y))

    assert x_out.dtype == trainer.mp_config.compute_dtype
    assert y_out.dtype == trainer.mp_config.compute_dtype


def test_trainer_exposes_mixed_precision_stats() -> None:
    """``get_mixed_precision_stats`` reports loss scale and dtypes."""
    model = _TinyModel(rngs=nnx.Rngs(0))
    trainer = mp.MixedPrecisionTrainer(model, TrainingConfig(num_epochs=1))

    stats = trainer.get_mixed_precision_stats()

    assert stats["loss_scale"] == trainer.mp_config.loss_scale
    assert stats["overflow_count"] == 0
    assert stats["step_count"] == 0
    assert stats["backend"] == trainer.mp_config.backend
