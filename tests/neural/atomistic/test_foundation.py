r"""Tests for the atomistic foundation-model fine-tuning mechanism.

The :mod:`~opifex.neural.atomistic.foundation` module provides the transfer
primitives for atomistic foundation models:

* :func:`~opifex.neural.atomistic.foundation.remap_element_table` -- map a source
  model's per-element arrays (e.g. reference energies ``E0`` or element
  embeddings) keyed by source atomic numbers onto a target element set, copying
  the rows of shared elements and initialising novel elements to a documented
  default (the ``../mace`` ``mace/tools/finetuning_utils.py``
  ``load_foundations_elements`` index-remap ``source[indices_weights]`` and the
  ``mace/data/utils.py`` novel-element ``0.0`` E0 default);
* :func:`~opifex.neural.atomistic.foundation.freeze_backbone` -- partition a
  model's parameters into the trainable (heads + LoRA) and frozen (backbone)
  groups so a fine-tune optimises only the heads, with the backbone frozen
  (Flax NNX filter / ``nnx.split`` partitioning).

Load-bearing checks:

* the remap copies shared-element rows verbatim and fills novel rows with the
  default;
* :func:`freeze_backbone` yields a partition whose gradient is absent on
  backbone params and present (nonzero) on heads / LoRA params.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.atomistic.foundation import (
    FineTuneConfig,
    freeze_backbone,
    remap_element_table,
    trainable_filter,
)
from opifex.neural.atomistic.lora import LoRALinear


class TestRemapElementTable:
    def test_shared_elements_copied_verbatim(self) -> None:
        """Rows for atomic numbers present in both tables are copied exactly."""
        source_numbers = (1, 6, 8)  # H, C, O
        # One row per source element; distinct so we can track them.
        source_array = jnp.array([[1.0, 1.1], [6.0, 6.6], [8.0, 8.8]])
        target_numbers = (8, 1)  # O, H (subset, reordered)
        remapped = remap_element_table(
            source_array,
            source_numbers=source_numbers,
            target_numbers=target_numbers,
        )
        # Row 0 is O (source row 2), row 1 is H (source row 0).
        assert jnp.allclose(remapped[0], source_array[2])
        assert jnp.allclose(remapped[1], source_array[0])

    def test_novel_elements_use_default(self) -> None:
        """Atomic numbers absent from the source are initialised to the default."""
        source_numbers = (1, 6)
        source_array = jnp.array([[1.0], [6.0]])
        target_numbers = (6, 7)  # C shared, N novel
        remapped = remap_element_table(
            source_array,
            source_numbers=source_numbers,
            target_numbers=target_numbers,
            novel_init=0.0,
        )
        assert jnp.allclose(remapped[0], source_array[1])  # C copied
        assert jnp.allclose(remapped[1], jnp.zeros_like(source_array[0]))  # N default

    def test_novel_init_value_is_respected(self) -> None:
        """A non-zero ``novel_init`` fills novel rows with that constant."""
        source_array = jnp.array([[2.0, 3.0]])
        remapped = remap_element_table(
            source_array,
            source_numbers=(1,),
            target_numbers=(1, 2),
            novel_init=-5.0,
        )
        assert jnp.allclose(remapped[1], jnp.full((2,), -5.0))

    def test_output_shape_matches_target(self) -> None:
        """The remapped table has one row per target element."""
        source_array = jnp.ones((3, 4))
        remapped = remap_element_table(
            source_array,
            source_numbers=(1, 6, 8),
            target_numbers=(1, 6, 7, 8, 9),
        )
        assert remapped.shape == (5, 4)

    def test_one_dimensional_table_remaps(self) -> None:
        """A 1-D per-element table (e.g. scalar E0 per element) also remaps."""
        source_array = jnp.array([10.0, 60.0, 80.0])
        remapped = remap_element_table(
            source_array,
            source_numbers=(1, 6, 8),
            target_numbers=(8, 7, 1),
            novel_init=0.0,
        )
        assert jnp.allclose(remapped, jnp.array([80.0, 0.0, 10.0]))


class _FineTuneNet(nnx.Module):
    """A tiny backbone + head model with a LoRA-adapted head for freeze tests."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.backbone = nnx.Linear(3, 3, rngs=rngs)
        head = nnx.Linear(3, 1, rngs=rngs)
        self.head = LoRALinear(head, rank=2, alpha=1.0)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(jax.nn.tanh(self.backbone(x)))


def _loss(model: _FineTuneNet, x: jax.Array) -> jax.Array:
    return jnp.sum(model(x) ** 2)


class TestFreezeBackbone:
    def test_partition_splits_trainable_and_frozen(self) -> None:
        """``freeze_backbone`` returns disjoint trainable / frozen param states."""
        model = _FineTuneNet(rngs=nnx.Rngs(0))
        config = FineTuneConfig()
        _, trainable, frozen = freeze_backbone(model, config=config)
        trainable_paths = {p for p, _ in nnx.to_flat_state(trainable)}
        frozen_paths = {p for p, _ in nnx.to_flat_state(frozen)}
        assert trainable_paths.isdisjoint(frozen_paths)
        # Backbone params are frozen.
        assert any("backbone" in path for path in frozen_paths)
        # LoRA + head-base params are trainable.
        assert any("lora" in str(path) for path in trainable_paths)

    def test_grad_absent_on_backbone_present_on_head(self) -> None:
        """Differentiating wrt the trainable filter zeroes backbone gradients."""
        model = _FineTuneNet(rngs=nnx.Rngs(0))
        # Move B so the head actually has a live gradient.
        model.head.lora_b.value = 0.1 * jnp.ones_like(model.head.lora_b.value)
        config = FineTuneConfig()
        x = jax.random.normal(jax.random.PRNGKey(1), (4, 3))

        grads = nnx.grad(_loss, argnums=nnx.DiffState(0, trainable_filter(config)))(model, x)
        grad_paths = {p for p, _ in nnx.to_flat_state(grads)}
        # No backbone gradient leaf at all.
        assert not any("backbone" in str(path) for path in grad_paths)
        # LoRA gradients are present and nonzero.
        flat = dict(nnx.to_flat_state(grads))
        lora_b_grad = flat[("head", "lora_b")].value
        assert jnp.any(lora_b_grad != 0.0)

    def test_full_grad_does_touch_backbone(self) -> None:
        """Sanity: an unfiltered grad *does* produce a backbone gradient.

        Confirms the frozen behaviour above is caused by the filter, not by a
        structurally dead backbone.
        """
        model = _FineTuneNet(rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(2), (4, 3))
        grads = nnx.grad(_loss, argnums=nnx.DiffState(0, nnx.Param))(model, x)
        grad_paths = {str(p) for p, _ in nnx.to_flat_state(grads)}
        assert any("backbone" in path for path in grad_paths)

    def test_fine_tune_config_is_frozen(self) -> None:
        """``FineTuneConfig`` is an immutable (frozen) dataclass."""
        config = FineTuneConfig()
        try:
            config.lora_rank = 99  # type: ignore[misc]
        except (AttributeError, TypeError):
            return
        raise AssertionError("FineTuneConfig should be immutable")
