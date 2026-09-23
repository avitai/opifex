# Framework Architecture

Opifex is built on a **Unified Scientific Machine Learning Architecture** that standardizes how physics-informed and data-driven methods interact. It moves beyond ad-hoc scripts to a rigorous, protocol-based system, designed to scale from research prototypes to production deployment.

## 🏗️ Architecture Layers

The framework is organized into 6 strictly hierarchical layers. Each layer builds upon the precise abstractions of the one below it.

| Layer | Name | Description | Key Components |
|-------|------|-------------|----------------|
| **6** | **Orchestration** | Production & Deployment | MLOps, Model Registry, Serving |
| **5** | **Uncertainty** | UQ & Reliability | `aggregate_solver_solutions`, `summarize_stacked_sample_solution`, `SolutionDistribution`, `opifex.uncertainty.conformal` |
| **4** | **Solvers** | **The Core Abstraction** | `Solver` callable, `Solution` pytree, `PINNSolver`, `NeuralOperatorSolver` |
| **3** | **Primitives** | Neural Architectures | `FourierNeuralOperator`, `DeepONet`, `MultiScalePINN` |
| **2** | **Problem Definition** | Physics & Geometry | `PDEProblem`, `Geometry` Protocol, `Constraint` |
| **1** | **Foundations** | Computation Engine | JAX, Flax NNX, Optax, Diffrax |

---

## Detailed Layer Breakdown

### Layer 1: Foundations (Computation Engine)
At its core, Opifex leverages **JAX** for composable transformations (Just-In-Time compilation, Auto-Differentiation, Vectorization) and **Flax NNX** for robust state management. This layer ensures that all higher-level components are automatically differentiable and accelerators-ready (GPU/TPU).

### Layer 2: Problem & Geometry
We enforce a strict separation between physics and geometry.

- **Geometry Protocol**: `opifex.geometry` objects (`Rectangle`, `SphericalManifold`, `CSGUnion`/`CSGIntersection`/`CSGDifference`) provide exact point sampling and boundary logic.
- **Problem Protocol**: `PDEProblem` defines *what* to solve (equations, boundary conditions), while Geometry defines *where* to solve it. Constraints are defined in `opifex.core.conditions`. This allows the same physics to be tested on different geometries without code changes.

### Layer 3: Neural Primitives
This layer provides optimized implementations of state-of-the-art architectures.

- **Neural Operators**: Discretization-invariant architectures like **FNO** and **DeepONet** that learn mappings between function spaces.
- **Physics-Informed models**: Specialized MLPs (Modified MLP, SIREN) and Fourier Feature embeddings designed to resolve high-frequency physics.
- **Quantum Architectures**: Symmetry-enforcing networks for quantum chemistry and DFT constraints.

### Layer 4: Solvers
A solver is a callable from a problem to a `Solution` (`opifex.core.solver.interface.Solver`).

- **Solution**: a pytree of arrays -- fields, scalar metrics, counters, and an `int32` `status` code -- so it can be built inside `jit` and returned through `jit`, `vmap` and `grad`. Under `vmap` the status carries one outcome per batch element; `is_successful` reads it. Wall-clock timing stays with the caller (`opifex.core.solver.report`).
- **Composition**: `additive_hybrid(classical, neural)` returns a solver that adds the neural fields to the classical ones and reports their discrepancy.
- **Entry points**: `NeuralOperatorSolver.solve` returns a `Solution` but takes a state and configuration beyond the problem, and `PINNSolver.solve` takes a geometry and residual functions and returns a `PINNResult`, so neither is interchangeable with a `Solver`.

### Layer 5: Probabilistic Numerics & Uncertainty
**A Major Pillar of Opifex.** We treat Uncertainty Quantification (UQ) not as an afterthought, but as a first-class citizen via the **Probabilistic Numerics** paradigm.

- **Ensemble / Monte-Carlo aggregation**: `aggregate_solver_solutions(solutions, *, quantiles=..., metadata=...)` packages a list of independent solver runs (deep ensemble or repeated stochastic replays) into a single `Solution` whose `auxiliary_data["uq"]` carries a `SolutionDistribution`-shaped payload (per-field mean, unbiased variance, samples, optional quantile bands).
- **Generative-sample summary**: `summarize_stacked_sample_solution(solution, *, sample_axis=0, quantiles=...)` handles the case where a single base solver returns pre-stacked sample arrays (Diffusion / Flow-Matching outputs from Artifex); statistics are computed along the sample axis without overwriting the original batch.
- **Bayesian Inference**: BlackJAX HMC / NUTS / MALA via `opifex.uncertainty.inference_backends.blackjax.BlackJAXBackend` (thin adapter over Artifex's BlackJAX wrappers).
- **Conformal Prediction**: `opifex.uncertainty.conformal` exposes `SplitConformalRegressor`, `ConformalizedQuantileRegressor`, CV+ / jackknife+, EnbPI / ACI, and field-space conformal calibrators — all with explicit `fit / predict / with_state` cycles and exchangeability diagnostics.

This layer transforms any deterministic solver into a probabilistic one by composing the canonical `PredictiveDistribution` contract with these aggregation utilities.

### Layer 6: Production Ecosystem
The top layer handles the lifecycle of models, including versioning, serving, and monitoring, bridging the gap between research and production value.

---

## Design Principles

The architecture of Opifex is guided by five core philosophies:

### 1. Protocol-First Design
Extension points are structural rather than inherited. A `Geometry` is a Protocol, and a solver is any callable from a problem to a `Solution`, checked against that signature by the type checker. Users inject custom implementations without carrying the weight of a base class hierarchy.

### 2. Composition Over Inheritance
We favor composing behavior over deep inheritance trees. A `PINNSolver` is not a monolith; it is composed of a `PhysicsLoss`, a `Trainer`, and a `RARDRefiner` (adaptive sampler). This makes it easy to swap out the optimizer or the sampling strategy without rewriting the solver.

### 3. Modularity
Each layer is designed to be independently useful. You can use the `Geometry` library for mesh generation without ever touching a neural network. You can use the `NeuralOperator` primitives in your own custom training loops if you prefer.

### 4. Performance & JIT-First
Performance is a feature. All core numerical routines are designed to be JIT-compiled. We explicitly manage state using Flax NNX to ensure pure function compatibility, minimizing the overhead of Python during training.

### 5. Reproducibility
Scientific results must be reproducible. Opifex integrates rigorous state management and random key handling (via `nnx.Rngs`) to ensure that every experiment can be exactly replicated.

## Integration Patterns

### Vertical Integration
Data flows seamlessly up the stack. A `PDEProblem` defined in Layer 2 is consumed by a `Solver` in Layer 4, which uses `Primitives` from Layer 3, all running on the `Foundations` of Layer 1.

### Wrapper-Based Extension
New capabilities (like UQ or Logging) are added via Wrappers. This strictly adheres to the Open-Closed Principle: classes are open for extension (via wrappers) but closed for modification.
