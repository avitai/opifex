# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Meta-Optimization: MAML and Reptile for a Family of PDEs

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~3 min (GPU) / ~10 min (CPU) |
| **Prerequisites** | JAX, Flax NNX, PINNs, gradient-based meta-learning |
| **Format** | Python + Jupyter |
| **Memory** | ~1 GB |

## Overview

Solving a *family* of PDEs that differ only in a parameter (here the Burgers viscosity `nu`) by
training a fresh physics-informed network (PINN) for each instance wastes the structure shared
across the family. **Gradient-based meta-learning** instead learns a single initialisation `theta`
such that a handful of ordinary gradient steps adapt it to any new `nu` — amortising the shared
physics into the starting point.

This example meta-learns that initialisation with opifex's `maml_meta_train` and
`reptile_meta_train` (the real algorithms — MAML, Finn et al. 2017, `arXiv:1703.03400`; Reptile,
Nichol et al. 2018, `arXiv:1803.02999`), then measures few-shot adaptation on **held-out**
viscosities against training a fresh network from a random initialisation.

The Burgers PINN is expressed as a `Task`/`TaskFamily` (the same abstraction the learned-optimiser
stack uses), so the example only supplies the physics — the meta-learning comes from the library.

## What You'll Learn

1. Express a PDE-solver family as an opifex `Task` / `TaskFamily`
2. Meta-learn a PINN initialisation with `maml_meta_train` and `reptile_meta_train`
3. Measure few-shot `adapt`-ation on unseen viscosities vs a from-scratch baseline
"""

# %%
SEED = 42

# Burgers viscosity range that defines the task family.
NU_MIN = 0.005
NU_MAX = 0.05
NUM_TEST_VISCOSITIES = 4

# PINN architecture: (x, t) -> u.
HIDDEN_DIM = 32
LAYER_SIZES = (2, HIDDEN_DIM, HIDDEN_DIM, 1)

# Domain and collocation budget (re-sampled each loss call for support/query separation).
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 0.5
N_DOMAIN, N_INITIAL, N_BOUNDARY = 200, 50, 20

# Meta-learning hyperparameters.
INNER_LR = 0.01  # inner-loop (task adaptation) step size
NUM_TASKS = 8  # tasks sampled per outer step
META_STEPS = 300  # outer meta-training steps
MAML_INNER_STEPS = 5  # inner adaptation steps per task (MAML)
MAML_META_LR = 1e-3  # outer Adam step size (MAML)
REPTILE_INNER_STEPS = 20  # Reptile needs a longer inner trajectory to move theta
REPTILE_META_LR = 0.3  # interpolation rate towards adapted params (Reptile)

# Few-shot evaluation: task loss after this many SGD adaptation steps (0 = zero-shot).
ADAPT_STEP_GRID = (0, 1, 2, 5, 10, 20, 50, 100)

OUTPUT_DIR = "docs/assets/examples/meta_optimization"

# %%
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# %%
from opifex.optimization.l2o.core import Task, TaskFamily
from opifex.optimization.l2o.meta_learning import adapt, maml_meta_train, reptile_meta_train


# %% [markdown]
r"""
## The Burgers PINN as a functional model

The PINN is a small `tanh`-MLP whose parameters are a list of `(weight, bias)` tuples — a plain
pytree, so the library's meta-trainers (which operate on pytrees, not stateful modules) can
differentiate through inner adaptation. The Burgers equation with viscosity `nu` is

$$\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}
   = \nu\,\frac{\partial^2 u}{\partial x^2}.$$
"""

# %%
PinnParams = list[tuple[jax.Array, jax.Array]]


def init_pinn_params(key: jax.Array, layer_sizes: tuple[int, ...] = LAYER_SIZES) -> PinnParams:
    """Sample Glorot-initialised ``(weight, bias)`` pairs for a tanh-MLP PINN."""
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params: PinnParams = []
    for layer_key, fan_in, fan_out in zip(keys, layer_sizes[:-1], layer_sizes[1:], strict=True):
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))  # Glorot/Xavier for tanh
        weight = scale * jax.random.normal(layer_key, (fan_in, fan_out))
        params.append((weight, jnp.zeros((fan_out,))))
    return params


def pinn_forward(params: PinnParams, xt: jax.Array) -> jax.Array:
    """Apply the tanh-MLP PINN to ``(n, 2)`` coordinates, returning ``(n, 1)`` field values."""
    activations = xt
    for index, (weight, bias) in enumerate(params):
        activations = activations @ weight + bias
        if index < len(params) - 1:
            activations = jnp.tanh(activations)
    return activations


def _u_scalar(params: PinnParams, xt_single: jax.Array) -> jax.Array:
    """Scalar field value ``u(x, t)`` at a single coordinate (for autodiff derivatives)."""
    return pinn_forward(params, xt_single.reshape(1, 2))[0, 0]


def burgers_residual(params: PinnParams, xt: jax.Array, nu: jax.Array) -> jax.Array:
    """Burgers PDE residual ``u_t + u u_x - nu u_xx`` at every coordinate in ``xt`` ``(n, 2)``."""

    def residual_single(xt_single: jax.Array) -> jax.Array:
        first = jax.grad(_u_scalar, argnums=1)(params, xt_single)  # (du/dx, du/dt)
        second = jax.grad(lambda s: jax.grad(_u_scalar, argnums=1)(params, s)[0])(xt_single)
        u = _u_scalar(params, xt_single)
        return first[1] + u * first[0] - nu * second[0]

    return jax.vmap(residual_single)(xt)


def sample_collocation(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Draw domain / initial-condition / boundary collocation points and the IC targets."""
    key_x, key_t, key_xi, key_tb = jax.random.split(key, 4)
    x_domain = jax.random.uniform(key_x, (N_DOMAIN,), minval=X_MIN, maxval=X_MAX)
    t_domain = jax.random.uniform(key_t, (N_DOMAIN,), minval=T_MIN, maxval=T_MAX)
    xt_domain = jnp.column_stack([x_domain, t_domain])

    x_initial = jax.random.uniform(key_xi, (N_INITIAL,), minval=X_MIN, maxval=X_MAX)
    xt_initial = jnp.column_stack([x_initial, jnp.zeros(N_INITIAL)])
    u_initial = -jnp.sin(jnp.pi * x_initial)  # u(x, 0) = -sin(pi x)

    t_boundary = jax.random.uniform(key_tb, (N_BOUNDARY,), minval=T_MIN, maxval=T_MAX)
    xt_boundary = jnp.concatenate(
        [
            jnp.column_stack([jnp.full(N_BOUNDARY, X_MIN), t_boundary]),
            jnp.column_stack([jnp.full(N_BOUNDARY, X_MAX), t_boundary]),
        ],
        axis=0,
    )
    return xt_domain, xt_initial, u_initial, xt_boundary


# %% [markdown]
"""
## The task family

A `BurgersTask` is one viscosity: `init` samples PINN parameters, `loss` evaluates the
physics-informed loss (PDE residual + initial + boundary conditions) on a freshly sampled set of
collocation points. `BurgersTaskFamily.sample` draws a viscosity uniformly from `[NU_MIN, NU_MAX]`,
so meta-training must generalise across the whole family rather than memorise one solver.
"""


# %%
@dataclass(frozen=True)
class BurgersTask(Task):
    """A single Burgers PINN objective at a fixed viscosity ``nu``."""

    nu: jax.Array

    def init(self, key: jax.Array) -> PinnParams:
        """Sample a fresh PINN initialisation."""
        return init_pinn_params(key)

    def loss(self, params: PinnParams, key: jax.Array) -> jax.Array:
        """Physics-informed loss on collocation points drawn with ``key``."""
        xt_domain, xt_initial, u_initial, xt_boundary = sample_collocation(key)
        loss_pde = jnp.mean(burgers_residual(params, xt_domain, self.nu) ** 2)
        loss_ic = jnp.mean((pinn_forward(params, xt_initial).squeeze() - u_initial) ** 2)
        loss_bc = jnp.mean(pinn_forward(params, xt_boundary).squeeze() ** 2)
        return loss_pde + loss_ic + loss_bc


@dataclass(frozen=True)
class BurgersTaskFamily(TaskFamily):
    """Burgers PINN tasks with viscosity drawn uniformly from ``[nu_min, nu_max]``."""

    nu_min: float = NU_MIN
    nu_max: float = NU_MAX

    def sample(self, key: jax.Array) -> BurgersTask:
        """Draw a Burgers task at a random viscosity."""
        nu = jax.random.uniform(key, (), minval=self.nu_min, maxval=self.nu_max)
        return BurgersTask(nu=nu)


# %% [markdown]
"""
## Few-shot evaluation

The canonical meta-learning measurement is the **few-shot learning curve**: the task loss after
`k` adaptation steps, as a function of `k`. For each held-out viscosity we `adapt` (plain SGD, the
inner rule meta-training assumes) from each starting point and record the loss at increasing step
counts. A good meta-initialisation already sits in a low-loss basin (`k = 0`, *zero-shot*) and a
random init must climb down with many steps.
"""


# %%
def few_shot_loss(start: PinnParams, nu: jax.Array, key: jax.Array, steps: int) -> float:
    """Task loss after ``steps`` SGD adaptation steps from ``start`` at viscosity ``nu``."""
    task = BurgersTask(nu=nu)
    adapted = adapt(task, start, key, inner_steps=steps, inner_lr=INNER_LR)
    return float(task.loss(adapted, jax.random.fold_in(key, 7)))


def main() -> dict[str, float | int]:
    """Meta-train MAML/Reptile PINN initialisations and benchmark few-shot adaptation."""
    print("=" * 72)
    print("Opifex Example: Meta-Optimization (MAML / Reptile) for a Burgers PDE family")
    print("=" * 72)
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")

    key = jax.random.key(SEED)
    family = BurgersTaskFamily()

    # Shared meta-initialisation: every method starts from the same random PINN.
    key, init_key = jax.random.split(key)
    init_params = init_pinn_params(init_key)
    n_params = sum(int(leaf.size) for leaf in jax.tree.leaves(init_params))
    print(f"\nPINN architecture {LAYER_SIZES}  ({n_params:,} parameters)")
    print(f"Task family: Burgers viscosity ~ U[{NU_MIN}, {NU_MAX}]")

    # --- Meta-training ---
    print(f"\nMeta-training MAML (first-order) for {META_STEPS} steps...")
    key, maml_key = jax.random.split(key)
    maml_params, maml_curve = maml_meta_train(
        family,
        init_params,
        maml_key,
        num_outer_steps=META_STEPS,
        num_tasks=NUM_TASKS,
        inner_steps=MAML_INNER_STEPS,
        inner_lr=INNER_LR,
        meta_lr=MAML_META_LR,
        first_order=True,
    )

    print(f"Meta-training Reptile for {META_STEPS} steps...")
    key, reptile_key = jax.random.split(key)
    reptile_params, reptile_curve = reptile_meta_train(
        family,
        init_params,
        reptile_key,
        num_outer_steps=META_STEPS,
        num_tasks=NUM_TASKS,
        inner_steps=REPTILE_INNER_STEPS,
        inner_lr=INNER_LR,
        meta_lr=REPTILE_META_LR,
    )
    print(f"  MAML    meta-loss: {float(maml_curve[0]):.4f} -> {float(maml_curve[-1]):.4f}")
    print(f"  Reptile meta-loss: {float(reptile_curve[0]):.4f} -> {float(reptile_curve[-1]):.4f}")

    # --- Few-shot learning curves on held-out viscosities ---
    test_viscosities = np.linspace(NU_MIN, NU_MAX, NUM_TEST_VISCOSITIES + 2)[1:-1]
    print(f"\nFew-shot adaptation on {len(test_viscosities)} held-out viscosities")
    print(f"(SGD, inner_lr={INNER_LR}); task loss after k steps, averaged over viscosities:")
    print("-" * 72)

    starts = {"MAML": maml_params, "Reptile": reptile_params, "Random init": init_params}
    # curves[name][k] = mean task loss after k adaptation steps over the held-out viscosities.
    curves: dict[str, list[float]] = {name: [] for name in starts}
    for steps in ADAPT_STEP_GRID:
        for name, start in starts.items():
            per_nu = [
                few_shot_loss(start, jnp.asarray(nu), jax.random.fold_in(key, int(nu * 1e4)), steps)
                for nu in test_viscosities
            ]
            curves[name].append(float(np.mean(per_nu)))

    header = "  steps " + "".join(f"{s:>9}" for s in ADAPT_STEP_GRID)
    print(header)
    for name in starts:
        print(f"  {name:<11}" + "".join(f"{v:>9.4f}" for v in curves[name]))

    zero_shot = {name: curves[name][0] for name in starts}
    random_final = curves["Random init"][-1]
    best_meta_zero_shot = min(zero_shot["MAML"], zero_shot["Reptile"])
    reduction = (1 - best_meta_zero_shot / random_final) * 100

    print()
    print("=" * 72)
    print("RESULTS — task loss on held-out viscosities (lower is better)")
    print("=" * 72)
    print(f"  MAML        zero-shot (0 steps): {zero_shot['MAML']:.4f}")
    print(f"  Reptile     zero-shot (0 steps): {zero_shot['Reptile']:.4f}")
    print(f"  Random init zero-shot (0 steps): {zero_shot['Random init']:.4f}")
    print(f"  Random init after {ADAPT_STEP_GRID[-1]} steps : {random_final:.4f}")
    print(
        f"\n  The best meta-init solves unseen viscosities ZERO-SHOT to a loss "
        f"{reduction:.0f}% below what a random init reaches after {ADAPT_STEP_GRID[-1]} SGD steps."
    )

    # --- Visualisation ---
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    mpl.use("Agg")
    _fig, (ax_meta, ax_few) = plt.subplots(1, 2, figsize=(13, 5))

    ax_meta.semilogy(np.asarray(maml_curve), label="MAML", color="tab:blue", linewidth=2)
    ax_meta.semilogy(np.asarray(reptile_curve), label="Reptile", color="tab:orange", linewidth=2)
    ax_meta.set_xlabel("Meta-step", fontsize=12)
    ax_meta.set_ylabel("Meta-loss (log scale)", fontsize=12)
    ax_meta.set_title("Meta-training convergence", fontsize=13)
    ax_meta.legend()
    ax_meta.grid(True, alpha=0.3)

    styles = {"MAML": "tab:blue", "Reptile": "tab:orange", "Random init": "gray"}
    for name, color in styles.items():
        ax_few.plot(ADAPT_STEP_GRID, curves[name], marker="o", color=color, linewidth=2, label=name)
    ax_few.set_xlabel("Adaptation steps (SGD)", fontsize=12)
    ax_few.set_ylabel("Task loss on held-out viscosities", fontsize=12)
    ax_few.set_title("Few-shot adaptation learning curve", fontsize=13)
    ax_few.legend()
    ax_few.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/meta_optimization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/meta_optimization.png")

    return {
        "num_parameters": n_params,
        "maml_zero_shot_loss": zero_shot["MAML"],
        "reptile_zero_shot_loss": zero_shot["Reptile"],
        "random_zero_shot_loss": zero_shot["Random init"],
        "random_final_loss": random_final,
        "zero_shot_reduction_percent": reduction,
        "num_test_viscosities": len(test_viscosities),
    }


# %%
if __name__ == "__main__":
    summary = main()
    for metric_name, metric_value in summary.items():
        print(f"{metric_name}: {metric_value}")
