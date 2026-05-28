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
# Matrix-free PINN curvature: Lanczos log-det + XNysTrace Fisher trace

| Property | Value |
|---|---|
| **Level** | Intermediate |
| **Runtime** | < 5 s (CPU) |
| **Prerequisites** | JAX, Flax NNX |

## Overview

A small Bayesian PINN-flavoured classifier is constructed with a single
hidden layer parameterised by a posterior-mean weight tensor. We then
estimate two scalar curvature summaries used in Laplace approximations
and evidence bounds:

* ``slq_logdet`` — stochastic Lanczos quadrature ``log det`` of the
  posterior precision (Lanczos depth + Rademacher probes).
* ``xnys_trace`` — XNysTrace estimate of the Fisher diagonal trace.

Both run matrix-free against a ``jax.jit``-compatible
``matvec(v) -> A @ v`` closure. No explicit Hessian / Fisher matrix is
ever materialised — only ``(dim,) -> (dim,)`` mat-vec products.
"""

# %% [markdown]
"""
## Imports and Setup
"""

# %%
import jax
import jax.numpy as jnp
from flax import nnx

from opifex.uncertainty.linalg.logdet import slq_logdet
from opifex.uncertainty.linalg.trace import xnys_trace


# %% [markdown]
"""
## Tiny Bayesian linear classifier

We use a single linear layer (`nnx.Linear`) — enough to expose the NNX
pytree path while keeping the curvature matrix small enough that the
matrix-free estimators converge in a handful of probes on CPU.
"""


# %%
class TinyBayesianHead(nnx.Module):
    """Single-layer linear head with explicit NNX RNG plumbing."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        self.dense = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the linear head to a batched input."""
        return self.dense(x)


# %% [markdown]
"""
## Curvature operator as a closure

The precision matrix of the Laplace posterior is ``F + τI`` where ``F``
is the empirical Fisher block of the head and ``τ`` is the prior
precision. We build a matvec closure that hits this operator
implicitly via ``jax.vjp``.
"""


# %%
def build_precision_matvec(
    *,
    head: TinyBayesianHead,
    inputs: jax.Array,
    targets: jax.Array,
    prior_precision: float,
):
    """Return ``matvec(v) = (F + τI) v`` for the head's flat weight."""
    graph_def, params_state = nnx.split(head, nnx.Param)
    flat_params, unflatten = jax.flatten_util.ravel_pytree(params_state)

    def loss_for_flat(flat: jax.Array) -> jax.Array:
        state = unflatten(flat)
        module = nnx.merge(graph_def, state)
        logits = module(inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        per_sample = -jnp.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
        return jnp.mean(per_sample)

    hessian_vp = lambda v: jax.grad(lambda p: jnp.vdot(jax.grad(loss_for_flat)(p), v))(flat_params)

    def matvec(v: jax.Array) -> jax.Array:
        return hessian_vp(v) + prior_precision * v

    return matvec, flat_params.shape[0]


# %% [markdown]
"""
## Run the example
"""


# %%
def main() -> dict[str, jax.Array | float | int]:
    """Estimate log-det and Fisher trace of a tiny PINN-style classifier."""
    rng_key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(params=jax.random.PRNGKey(1))
    in_features, out_features, batch_size = 4, 3, 16
    head = TinyBayesianHead(in_features, out_features, rngs=rngs)

    input_key, target_key, logdet_key, trace_key = jax.random.split(rng_key, 4)
    inputs = jax.random.normal(input_key, (batch_size, in_features))
    targets = jax.random.randint(target_key, (batch_size,), 0, out_features)

    matvec, dim = build_precision_matvec(
        head=head, inputs=inputs, targets=targets, prior_precision=1.0
    )
    matvec_jit = jax.jit(matvec)

    log_det_estimate = slq_logdet(
        matvec=matvec_jit,
        dim=dim,
        num_samples=8,
        num_matvecs=min(dim, 12),
        key=logdet_key,
    )
    fisher_trace_estimate = xnys_trace(
        matvec=matvec_jit,
        dim=dim,
        num_samples=min(dim, 8),
        key=trace_key,
    )

    return {
        "dim": int(dim),
        "log_det_estimate": log_det_estimate,
        "fisher_trace_estimate": fisher_trace_estimate,
    }


# %%
if __name__ == "__main__":
    summary = main()
    for label, value in summary.items():
        print(f"{label}: {value}")
