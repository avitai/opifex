# Learn-to-Optimize (L2O) Methods

## Overview

Learn-to-Optimize (L2O) replaces a hand-designed optimiser update rule (SGD, Adam) with a small
neural network that is meta-trained to optimise well across a *distribution* of objectives.
Instead of applying the same fixed rule to every problem, a learned optimiser adapts its update
per-coordinate and per-step, having been trained on a family of related problems.

Opifex's L2O subsystem (`opifex.optimization.l2o`) follows the design of Google's
[`learned_optimization`](https://github.com/google/learned_optimization) library and the L2O
literature:

- **Andrychowicz et al. 2016** — *Learning to learn by gradient descent by gradient descent*
  ([arXiv:1606.04474](https://arxiv.org/abs/1606.04474)): the original learned-optimiser
  formulation.
- **Metz et al. 2020** — *Tasks, stability, architecture, and compute*
  ([arXiv:2009.11243](https://arxiv.org/abs/2009.11243)): the per-parameter MLP optimiser
  ("LOLv2") and its input features.
- **Vicol, Metz & Sohl-Dickstein 2021** — *Unbiased gradient estimation in unrolled computation
  graphs with Persistent Evolution Strategies*
  ([arXiv:2112.13835](https://arxiv.org/abs/2112.13835)): the PES meta-training estimator.
- **Metz et al. 2022** — *VeLO* ([arXiv:2211.09760](https://arxiv.org/abs/2211.09760)) is the
  scale frontier, and **Eldan et al. 2023** ([arXiv:2310.18191](https://arxiv.org/abs/2310.18191))
  is the cautionary critique that learned optimisers do not reliably beat tuned baselines
  out-of-distribution — opifex scopes every generalisation claim accordingly.

Every reported number is measured against a properly tuned classical baseline; there are no
fabricated objectives or speedups.

## Core abstractions

### Tasks carry their objective

The defining design choice is that the objective lives *on the task*, not in the solver. A
[`Task`][opifex.optimization.l2o.core.Task] exposes:

- `init(key) -> params` — sample the initial optimisee parameters;
- `loss(params, key) -> scalar` — the objective (the `key` supports stochastic / minibatch tasks);
- `normalizer(loss) -> loss` — map a raw loss onto a comparable scale so a meta-loss aggregated
  across differently-scaled tasks is not dominated by the worst-scaled one.

A [`TaskFamily`][opifex.optimization.l2o.core.TaskFamily] has `sample(key) -> Task`, giving a
meta-training distribution and a held-out meta-test split (a different RNG stream). Opifex ships
two families:

- [`QuadraticTaskFamily`][opifex.optimization.l2o.tasks.QuadraticTaskFamily] — strictly convex
  quadratics `f(x) = ½ (x − x*)ᵀ A (x − x*)` with random SPD `A` of varied conditioning. A
  smoke task with a known optimum.
- [`MLPTaskFamily`][opifex.optimization.l2o.tasks.MLPTaskFamily] — the *showcase* task: a small
  teacher-student MLP regression trained on stochastic minibatches. The objective is non-convex
  in the student weights, which is the regime where a learned optimiser genuinely beats a
  fixed-hyperparameter baseline. It is self-contained (synthetic data, no dataset dependency).

### The optimiser interface

[`Optimizer`][opifex.optimization.l2o.optimizers.Optimizer] is the stateful interface shared by
hand-designed and learned optimisers:

```python
state = optimizer.init(params, num_steps=horizon)   # num_steps -> horizon-aware optimisers
state = optimizer.update(state, grad, loss=loss)     # loss consumed by learned optimisers
params = optimizer.get_params(state)
```

[`OptaxOptimizer`][opifex.optimization.l2o.optimizers.OptaxOptimizer] wraps any
`optax.GradientTransformation` (SGD/Adam/…) as the hand-designed baseline family.

### Learned optimisers

A [`LearnedOptimizer`][opifex.optimization.l2o.learned.LearnedOptimizer] carries meta-parameters
`theta` and an `opt_fn(theta) -> Optimizer` that bakes `theta` into a concrete optimiser:

- [`MLPLearnedOptimizer`][opifex.optimization.l2o.learned.MLPLearnedOptimizer] — the per-parameter
  MLP of Metz et al. 2020. A tiny MLP, shared across all scalar parameters, maps a 19-feature
  per-parameter vector to a `(direction, magnitude)` pair, and the update is

  $$\text{step} = \text{direction} \cdot \exp(\text{magnitude} \cdot \texttt{exp\_mult}) \cdot \texttt{step\_mult}.$$

  The features (`opifex.optimization.l2o.features`) are the gradient, the parameter, and
  multi-timescale momentum EMAs (second-moment-normalised per tensor), concatenated with an
  11-timescale tanh embedding of the iteration — the latter gives the optimiser
  *training-fraction awareness* (an implicit learning-rate schedule).
- [`AdafacMLPLearnedOptimizer`][opifex.optimization.l2o.learned.AdafacMLPLearnedOptimizer] — the
  Adafactor-feature variant (`adafac_mlp_lopt.py`). It extends the per-parameter MLP with
  multi-decay RMS, `m·rsqrt(rms)`, `rsqrt(rms)`, and **factored row/column second-moment features**
  (`features.factored_dims` / `features.update_adafactor_accum`) for rank-≥2 tensors, sharing the
  same `(direction, magnitude)` step head.
- [`LearnableSGD`][opifex.optimization.l2o.learned.LearnableSGD] — SGD with a single learnable log
  learning rate; the simplest learned optimiser, used to validate the meta-gradient estimator.

`theta` is a plain `flax.nnx` state pytree (via `nnx.split`), so it is directly
perturbable/vmappable — the prerequisite for evolution-strategies meta-training — and serialises
with Orbax.

## Meta-training with Persistent Evolution Strategies (PES)

Meta-training searches for `theta` minimising the inner task loss accumulated over an unroll,
across the task distribution. Back-propagating through a long inner unroll is biased for short
truncations and has exploding/chaotic gradients for long ones (Metz et al. 2019). PES instead
estimates the meta-gradient with antithetic Gaussian perturbations of `theta` over short
truncations, while keeping a **persistent accumulator** of the perturbations across truncation
boundaries, so the estimate is unbiased with respect to the full-horizon objective:

$$\widehat{g} = \frac{1}{2\sigma^2}\, \Delta\mathcal{L}\, \cdot\, \xi_{\text{acc}},$$

where `Δℒ` is the antithetic loss difference and `ξ_acc` is the accumulated perturbation. PES does
**not** back-propagate through the unroll — `theta` is updated by the ES estimate fed to an outer
Adam.

[`meta_train`][opifex.optimization.l2o.meta_train.meta_train] runs `num_tasks` inner problems in
parallel (`jax.vmap`) and is fully JIT-compiled (the outer step is `jax.jit` over a `lax.scan`).
Faithful to `learned_optimization`'s `truncated_pes`, it:

- splits each truncation's per-step delta-losses at the per-step horizon reset
  (`has_finished = cumsum(is_done) > 0`): losses before the reset attribute to the full
  accumulator, losses after attribute only to the current perturbation;
- starts each parallel trajectory at a **random clock offset** in `[0, total_horizon)`
  (`random_initial_iteration_offset` in the reference) and resets each trajectory *per inner
  step* when its clock reaches the horizon. Staggering the truncations so the tasks are not
  phase-aligned is load-bearing: it removes the sawtooth a synchronous reset would imprint on the
  meta-loss and lowers the PES variance.

The decisive correctness property — the antithetic-ES meta-gradient equals the autodiff
meta-gradient through the unroll — is enforced by test
(`tests/optimization/l2o/test_l2o_meta_train.py`).

Key hyperparameters: `num_outer_steps`, `num_tasks` (vmap width), `trunc_length` (truncation
length), `total_horizon` (full inner-unroll length), `perturbation_std` (σ), and
`meta_learning_rate` (outer Adam).

## Honest benchmarking

A learned optimiser is only interesting if it beats a *tuned* classical optimiser, so opifex tunes
the baselines with the standard L2O protocol
([`benchmark_on_held_out_tasks`][opifex.optimization.l2o.benchmark.benchmark_on_held_out_tasks]):
a **single** learning rate is selected on a tuning batch from the family (best mean final loss)
and applied unchanged to every held-out task. A per-task learning-rate sweep would be an
undeployable oracle, so it is not used as the comparison baseline.

- Primary metric: the **loss-vs-step learning curve** (`learned_curve_mean`,
  `baseline_curve_mean`).
- Secondary metric: **speedup at a per-task target loss** — `baseline steps to reach target /
  learned steps`, censored when a method never reaches the target (returns the median over tasks,
  robust to censoring).
- [`optimistix_minimise`][opifex.optimization.l2o.baselines.optimistix_minimise] wraps real
  `optimistix` solvers (`BFGS`/`GradientDescent`/`NonlinearCG`) as strong second-order references.

Generalisation claims are scoped to in-distribution held-out tasks. Speedups are reported against
a tuned baseline and accompanied by the VeLO-scaling caveat.

## End-to-end usage

```python
import jax
from opifex.optimization.l2o import L2OEngine, MLPLearnedOptimizer, MLPTaskFamily

# 1. A family of small non-convex training tasks.
family = MLPTaskFamily(input_dim=8, hidden_dim=16, output_dim=4)

# 2. A per-parameter MLP learned optimiser, orchestrated by the engine.
engine = L2OEngine(MLPLearnedOptimizer(step_mult=0.03), family)

# 3. Meta-train with PES (returns the meta-loss curve; stores theta on the engine).
meta_losses = engine.meta_train(
    jax.random.key(0),
    num_outer_steps=3000, num_tasks=32,
    trunc_length=20, total_horizon=100,
    perturbation_std=0.01, meta_learning_rate=3e-3,
)

# 4. Meta-test against a distribution-tuned baseline on held-out tasks.
result = engine.benchmark(jax.random.key(1), num_tasks=48, num_steps=100)
print(result["median_speedup"])          # ~1.8x vs tuned Adam (in-distribution)

# 5. Apply the trained optimiser to a new task, or persist theta.
task = family.sample(jax.random.key(2))
curve = engine.optimize(task, task.init(jax.random.key(3)), num_steps=100, key=jax.random.key(4))
```

On the `MLPTaskFamily` showcase, a learned optimiser meta-trained for ~3000 outer steps (a few
seconds on a GPU) reaches the target loss roughly **1.8× faster than a tuned Adam** baseline and
**2.7× faster than tuned SGD** on held-out tasks, with a lower final loss than both — see the
[Learn-to-Optimize example](../examples/optimization/learn-to-optimize.md).

## Module layout

| Module | Contents |
| --- | --- |
| `core.py` | `Task` / `TaskFamily` abstractions, `single_task_to_family` |
| `optimizers.py` | `Optimizer` ABC, `OptaxOptimizer` |
| `tasks.py` | `QuadraticTaskFamily`, `MLPTaskFamily` |
| `features.py` | per-parameter input features (momentum/RMS EMAs, tanh time embedding) |
| `learned.py` | `LearnedOptimizer` ABC, `MLPLearnedOptimizer`, `LearnableSGD` |
| `meta_train.py` | PES estimator + outer-Adam meta-training loop |
| `baselines.py` | optimistix solvers and tuned-optax baselines |
| `benchmark.py` | learning curves + censored speedup-at-target |
| `engine.py` | high-level `L2OEngine` (meta-train / apply / benchmark / persist) |

## See also

- [Learn-to-Optimize example](../examples/optimization/learn-to-optimize.md) — meta-train, meta-test,
  and benchmark end to end.
- [Meta-Optimization Methods](meta-optimization.md) — the separate `opifex.optimization.meta_optimization`
  framework (`LearnToOptimize`, `MetaOptimizer`, adaptive schedulers, warm-starting).
- [Optimization API reference](../api/optimization.md).
