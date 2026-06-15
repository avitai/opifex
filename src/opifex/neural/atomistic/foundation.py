r"""Fine-tuning / transfer mechanism for atomistic foundation models.

A foundation interatomic potential is pre-trained on a broad element set; a
downstream user fine-tunes it onto a (usually smaller, sometimes element-shifted)
target system. This module provides the two transfer primitives that make that
possible without retraining from scratch:

#. :func:`remap_element_table` -- the source model's per-element arrays (e.g.
   reference energies ``E0`` or element embeddings) are keyed by the source
   model's atomic numbers; this remaps them onto a *target* element set, copying
   the rows of shared elements verbatim and initialising novel elements to a
   documented default. This is the ``../mace``
   ``mace/tools/finetuning_utils.py`` ``load_foundations_elements`` index remap
   (``source[indices_weights]`` with ``indices_weights = [src_z_to_index(z) for z
   in target_zs]``); the novel-element default mirrors the ``mace/data/utils.py``
   ``0.0`` E0 fallback for elements absent from the source.
#. :func:`freeze_backbone` / :func:`trainable_filter` -- partition a model's
   parameters into the *trainable* group (the property heads and any LoRA
   adapters) and the *frozen* group (the backbone), so a fine-tune optimises only
   the heads (plus LoRA) while the expensive, transferable backbone
   representation is held fixed. The partition is a Flax NNX *filter*: passing it
   to :func:`flax.nnx.grad` (via :class:`flax.nnx.DiffState`) differentiates only
   the matching parameters, so frozen parameters receive **no gradient leaf at
   all** -- the cleanest, ``jit``/``grad``/``vmap``-safe way to freeze in NNX
   (it composes NNX transforms rather than relabelling the model).

The companion low-rank adapter is :class:`opifex.neural.atomistic.lora.LoRALinear`
(Hu et al. 2021, LoRA, arXiv:2106.09685; the ``../mace`` ``mace/modules/lora.py``).

:class:`FineTuneConfig` is a frozen dataclass holding the fine-tune knobs (LoRA
rank / alpha, the parameter-name substrings that mark a parameter trainable, and
the novel-element init).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002


logger = logging.getLogger(__name__)

_DEFAULT_TRAINABLE_NAME_PARTS: tuple[str, ...] = ("head", "lora")
"""Default parameter-path substrings that mark a parameter trainable.

A parameter whose NNX path contains any of these substrings (``"head"`` for the
property heads, ``"lora"`` for the LoRA ``A``/``B`` factors) is fine-tuned; every
other parameter (the backbone) is frozen.
"""

_DEFAULT_NOVEL_INIT = 0.0
"""Default value for per-element rows of elements absent from the source model.

Mirrors the ``../mace`` ``mace/data/utils.py`` ``0.0`` E0 fallback for elements
not present in the source data.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class FineTuneConfig:
    """Configuration for fine-tuning an atomistic foundation model.

    Attributes:
        lora_rank: Low-rank bottleneck width for LoRA adapters.
        lora_alpha: LoRA scaling numerator (correction scaled by
            ``alpha / rank``).
        trainable_name_parts: Parameter-path substrings that mark a parameter
            trainable; everything else is frozen (the backbone).
        novel_element_init: Value used to initialise per-element rows for target
            elements absent from the source model.
    """

    lora_rank: int = 4
    lora_alpha: float = 1.0
    trainable_name_parts: tuple[str, ...] = field(
        default_factory=lambda: _DEFAULT_TRAINABLE_NAME_PARTS
    )
    novel_element_init: float = _DEFAULT_NOVEL_INIT


def remap_element_table(
    source_array: Float[Array, "n_source ..."],
    *,
    source_numbers: tuple[int, ...],
    target_numbers: tuple[int, ...],
    novel_init: float = _DEFAULT_NOVEL_INIT,
) -> Float[Array, "n_target ..."]:
    r"""Remap a source model's per-element array onto a target element set.

    Implements the ``../mace`` ``mace/tools/finetuning_utils.py``
    ``load_foundations_elements`` index remap: for every target atomic number
    present in the source, copy that source row; for every target atomic number
    absent from the source (a *novel* element), fill the row with ``novel_init``
    (the ``mace/data/utils.py`` ``0.0`` E0 fallback by default).

    Args:
        source_array: Per-element array with one leading row per source element,
            i.e. ``source_array[i]`` corresponds to ``source_numbers[i]``. May be
            1-D (scalar per element, e.g. ``E0``) or N-D (e.g. an embedding row
            per element).
        source_numbers: The source model's atomic numbers, aligned with
            ``source_array``'s leading axis.
        target_numbers: The target element set, in the desired output row order.
        novel_init: Constant used to fill rows of target elements absent from the
            source.

    Returns:
        An array with leading dimension ``len(target_numbers)``; row ``j`` is the
        source row for ``target_numbers[j]`` if shared, else a ``novel_init``-filled
        row matching the source row shape.

    Raises:
        ValueError: If ``source_numbers`` length does not match the source array's
            leading dimension.
    """
    if len(source_numbers) != source_array.shape[0]:
        raise ValueError(
            "source_numbers length "
            f"({len(source_numbers)}) must match source_array leading "
            f"dimension ({source_array.shape[0]})."
        )
    source_index = {int(z): i for i, z in enumerate(source_numbers)}
    row_shape = source_array.shape[1:]
    novel_row = jnp.full(row_shape, novel_init, dtype=source_array.dtype)

    rows: list[Array] = []
    novel: list[int] = []
    for z in target_numbers:
        source_pos = source_index.get(int(z))
        if source_pos is None:
            novel.append(int(z))
            rows.append(novel_row)
        else:
            rows.append(source_array[source_pos])
    if novel:
        logger.info(
            "remap_element_table: %d novel element(s) %s initialised to %s.",
            len(novel),
            novel,
            novel_init,
        )
    return jnp.stack(rows, axis=0)


def trainable_filter(config: FineTuneConfig | None = None) -> nnx.filterlib.Filter:
    """Build the NNX filter selecting the trainable (head + LoRA) parameters.

    The returned filter matches an ``nnx.Param`` whose NNX path contains any of
    ``config.trainable_name_parts`` (``"head"`` / ``"lora"`` by default). Passing
    it to :func:`flax.nnx.grad` via :class:`flax.nnx.DiffState` differentiates
    only those parameters, freezing the backbone.

    Args:
        config: Fine-tune configuration supplying the trainable name parts.
            Defaults to :class:`FineTuneConfig`.

    Returns:
        A Flax NNX ``Filter`` matching the trainable parameters.
    """
    config = config if config is not None else FineTuneConfig()
    name_parts = config.trainable_name_parts

    def is_trainable(path: nnx.filterlib.PathParts, value: object) -> bool:
        if not isinstance(value, nnx.Param):
            return False
        path_text = "/".join(str(part) for part in path)
        return any(part in path_text for part in name_parts)

    return is_trainable


def freeze_backbone(
    model: nnx.Module, *, config: FineTuneConfig | None = None
) -> tuple[nnx.GraphDef, nnx.State, nnx.State]:
    """Partition a model into ``(graphdef, trainable_state, frozen_state)``.

    The trainable state holds the head + LoRA parameters (selected by
    :func:`trainable_filter`); the frozen state holds everything else (the
    backbone). The split is the Flax NNX two-filter
    :func:`flax.nnx.split` partition, so the two states are disjoint and
    together reconstruct the model via :func:`flax.nnx.merge`.

    Args:
        model: The assembled (foundation) model to fine-tune.
        config: Fine-tune configuration. Defaults to :class:`FineTuneConfig`.

    Returns:
        ``(graphdef, trainable_state, frozen_state)`` -- the trainable group is
        optimised; the frozen group is held fixed.
    """
    config = config if config is not None else FineTuneConfig()
    is_trainable = trainable_filter(config)
    graphdef, trainable_state, frozen_state = nnx.split(model, is_trainable, nnx.Not(is_trainable))
    return graphdef, trainable_state, frozen_state


__all__ = [
    "FineTuneConfig",
    "freeze_backbone",
    "remap_element_table",
    "trainable_filter",
]
