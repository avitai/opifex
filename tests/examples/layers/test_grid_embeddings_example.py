"""Integration test for the grid-embeddings ablation example.

The example trains an FNO with and without :class:`GridEmbedding2D` on Darcy flow and reports the
relative-L2 gap. This test exercises the example's real building blocks on a tiny synthetic problem
(small grid, few epochs) so it runs fast while still pinning the structural contract: the embedded
model adds the two coordinate channels and both models train end-to-end through the example helpers.
The full-scale accuracy claim is produced by running ``main()`` itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from examples.layers.grid_embeddings_example import (
    _eval_relative_l2,
    _train_one,
    FNO,
    FNOWithGridEmbedding,
)
from flax import nnx


_FNO_KWARGS = {"modes": 4, "hidden_channels": 8, "num_layers": 2}


def _tiny_darcy(resolution: int = 16, n: int = 32, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A small smooth input/output pair, channels-first ``(n, 1, H, W)``, for fast training."""
    key = jax.random.key(seed)
    x = jax.random.normal(key, (n, 1, resolution, resolution))
    # A fixed smooth map gives the model something learnable in a couple of epochs.
    y = jnp.tanh(x)
    return np.asarray(x), np.asarray(y)


def test_grid_embedding_adds_two_coordinate_channels() -> None:
    """``FNOWithGridEmbedding`` feeds the FNO 3 channels (permeability + x + y)."""
    embedded = FNOWithGridEmbedding(**_FNO_KWARGS, rngs=nnx.Rngs(0))
    assert embedded.grid_embedding.out_channels == 3
    # The embedding turns the 1-channel field into a 3-channel (field, x, y) input.
    sample = jnp.asarray(_tiny_darcy(n=1)[0]).transpose(0, 2, 3, 1)  # (1, H, W, 1)
    assert embedded.grid_embedding(sample).shape == (1, 16, 16, 3)


def test_both_models_share_a_single_output_channel() -> None:
    """Plain and embedded models differ only on the input side; both predict one field."""
    plain = FNO(**_FNO_KWARGS, rngs=nnx.Rngs(0))
    embedded = FNOWithGridEmbedding(**_FNO_KWARGS, rngs=nnx.Rngs(0))
    x = jnp.asarray(_tiny_darcy()[0][:2])
    assert plain(x).shape == (2, 1, 16, 16)
    assert embedded(x).shape == (2, 1, 16, 16)


def test_example_helpers_train_and_evaluate_end_to_end() -> None:
    """The example's ``_train_one`` / ``_eval_relative_l2`` run and yield a finite error."""
    x, y = _tiny_darcy()
    trained = _train_one(
        FNOWithGridEmbedding(**_FNO_KWARGS, rngs=nnx.Rngs(0)),
        x,
        y,
        num_epochs=2,
        seed=0,
    )
    error = _eval_relative_l2(trained, x, y, y_mean=0.0, y_std=1.0)
    assert np.isfinite(error)
    assert error >= 0.0
