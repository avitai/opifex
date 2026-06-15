# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Diagonal Laplace posterior for a small MLP classifier

| Property | Value |
|---|---|
| **Level** | Intermediate |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX, Flax NNX, Laplace approximation |

## Overview

Train a small MLP classifier with `optax.adam`, then build a diagonal
Laplace posterior at the MAP point using
``opifex.uncertainty.curvature.diagonal_laplace_posterior``. The
posterior is summarised by the per-parameter precision diagonal
``τ + F_ii``, where ``F`` is the empirical Fisher diagonal.

We then report two calibration summaries:

* **ECE** — expected calibration error against the binary class label
  via ``opifex.uncertainty.calibration.expected_calibration_error``.
* **ANEES** — average normalised estimation error squared on the
  predicted logits via ``opifex.uncertainty.metrics.anees``.

NNX state path with explicit `nnx.Rngs` is exercised throughout.
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.uncertainty.calibration import expected_calibration_error
from opifex.uncertainty.curvature import diagonal_laplace_posterior, DiagonalLaplacePosterior
from opifex.uncertainty.metrics import anees


# %% [markdown]
"""
## Small MLP classifier
"""


# %%
class SmallMLP(nnx.Module):
    """Two-layer MLP for binary classification."""

    def __init__(self, in_features: int, hidden: int, *, rngs: nnx.Rngs) -> None:
        self.layer_one = nnx.Linear(in_features, hidden, rngs=rngs)
        self.layer_two = nnx.Linear(hidden, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the two-layer MLP to a batched input."""
        h = nnx.relu(self.layer_one(x))
        return self.layer_two(h)


# %% [markdown]
"""
## Synthetic binary dataset
"""


# %%
def _make_dataset(
    rng_key: jax.Array, num_samples: int, in_features: int
) -> tuple[jax.Array, jax.Array]:
    weight_key, input_key = jax.random.split(rng_key)
    true_weight = jax.random.normal(weight_key, (in_features,))
    inputs = jax.random.normal(input_key, (num_samples, in_features))
    logits = inputs @ true_weight
    targets = (logits > 0).astype(jnp.int32)
    return inputs, targets


# %% [markdown]
"""
## Train MAP estimate, then build Laplace posterior
"""


# %%
def _train_map(
    *,
    model: SmallMLP,
    inputs: jax.Array,
    targets: jax.Array,
    num_steps: int,
    learning_rate: float,
) -> SmallMLP:
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    def loss_fn(module: SmallMLP) -> jax.Array:
        logits = module(inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        per_sample = -jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
        return jnp.mean(per_sample)

    @nnx.jit
    def step(model: SmallMLP, optimizer: nnx.Optimizer) -> jax.Array:
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    for _ in range(num_steps):
        step(model, optimizer)
    return model


# %% [markdown]
"""
## Run the example
"""


# %%
def main() -> dict[str, jax.Array | float | int]:
    """Train MAP, build diagonal Laplace, report ECE + ANEES."""
    rng_key = jax.random.PRNGKey(0)
    data_key, params_key = jax.random.split(rng_key)
    in_features, hidden, num_samples = 4, 8, 128

    inputs, targets = _make_dataset(data_key, num_samples, in_features)
    model = SmallMLP(in_features, hidden, rngs=nnx.Rngs(params=params_key))
    model = _train_map(
        model=model, inputs=inputs, targets=targets, num_steps=100, learning_rate=1e-2
    )

    # Flatten parameters for Laplace estimation.
    graph_def, params_state = nnx.split(model, nnx.Param)
    flat_params, unflatten = jax.flatten_util.ravel_pytree(params_state)

    def per_sample_loss(
        flat: jax.Array, input_sample: jax.Array, target_sample: jax.Array
    ) -> jax.Array:
        state = unflatten(flat)
        module = nnx.merge(graph_def, state)
        logits = module(input_sample[None, :])[0]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -log_probs[target_sample]

    posterior: DiagonalLaplacePosterior = diagonal_laplace_posterior(
        per_sample_loss=per_sample_loss,
        map_estimate=flat_params,
        inputs=inputs,
        targets=targets,
        prior_precision=1.0,
    )

    # Predict with the MAP module and compute calibration metrics.
    final_logits = model(inputs)
    probabilities = jax.nn.softmax(final_logits, axis=-1)
    ece = expected_calibration_error(
        probabilities=probabilities[:, 1], targets=targets, num_bins=10
    )

    # ANEES on the logit predictions with diagonal predictive covariance
    # derived from the Laplace posterior. We use a coarse per-sample
    # variance proxy: 1 / mean(precision_diagonal) replicated over both
    # logit classes. This is illustrative, not a Bayesian-NN
    # functional-Laplace push-forward.
    posterior_variance = 1.0 / jnp.mean(posterior.precision_diagonal)
    predicted_covariances = jnp.broadcast_to(
        posterior_variance * jnp.eye(2), (final_logits.shape[0], 2, 2)
    )
    true_logits = jnp.stack(
        [jnp.where(targets == 0, 1.0, 0.0), jnp.where(targets == 1, 1.0, 0.0)], axis=-1
    )
    anees_value = anees(
        predicted_means=final_logits,
        predicted_covariances=predicted_covariances,
        references=true_logits,
    )

    return {
        "num_parameters": int(flat_params.shape[0]),
        "posterior_precision_mean": jnp.mean(posterior.precision_diagonal),
        "posterior_precision_min": jnp.min(posterior.precision_diagonal),
        "ece": ece,
        "anees": anees_value,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for label, value in summary.items():
        print(f"{label}: {value}")
