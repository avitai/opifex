# Framework Architecture

Opifex is built on a **Unified Scientific Machine Learning Architecture** that standardizes how physics-informed and data-driven methods interact. It moves beyond ad-hoc scripts to a rigorous, protocol-based system, designed to scale from research prototypes to production deployment.

## 🏗️ Architecture Layers

The framework is organized into 6 strictly hierarchical layers. Each layer builds upon the precise abstractions of the one below it.

| Layer | Name | Description | Key Components |
|-------|------|-------------|----------------|
| **6** | **Orchestration** | Production & Deployment | MLOps, Model Registry, Serving |
| **5** | **Uncertainty** | UQ & Reliability | `EnsembleWrapper`, `ConformalWrapper`, `GenerativeWrapper` |
| **4** | **Unified Solvers** | **The Core Abstraction** | `SciMLSolver` Protocol, `PINNSolver`, `NeuralOperatorSolver` |
| **3** | **Primitives** | Neural Architectures | `FourierNeuralOperator`, `DeepONet`, `MultiScalePINN` |
| **2** | **Problem Definition** | Physics & Geometry | `PDEProblem`, `Geometry` Protocol, `Constraint` |
| **1** | **Foundations** | Computation Engine | JAX, Flax NNX, Optax, Diffrax |

---

## Detailed Layer Breakdown

### Layer 1: Foundations (Computation Engine)
At its core, Opifex leverages **JAX** for composable transformations (Just-In-Time compilation, Auto-Differentiation, Vectorization) and **Flax NNX** for robust state management. This layer ensures that all higher-level components are automatically differentiable and accelerators-ready (GPU/TPU).

### Layer 2: Problem & Geometry
We enforce a strict separation between physics and geometry.

- **Geometry Protocol**: `opifex.geometry` objects (`Rectangle`, `Sphere`, `CSGDomain`) provide exact point sampling and boundary logic.
- **Problem Protocol**: `PDEProblem` defines *what* to solve (equations, boundary conditions), while Geometry defines *where* to solve it. This allows the same physics to be tested on different geometries without code changes.

### Layer 3: Neural Primitives
This layer provides optimized implementations of state-of-the-art architectures.

- **Neural Operators**: Discretization-invariant architectures like **FNO** and **DeepONet** that learn mappings between function spaces.
- **Physics-Informed models**: Specialized MLPs (Modified MLP, SIREN) and Fourier Feature embeddings designed to resolve high-frequency physics.
- **Quantum Architectures**: Symmetry-enforcing networks for quantum chemistry and DFT constraints.

### Layer 4: Unified Solvers (The Heart of Opifex)
This is the pivotal abstraction layer. It defines a standard `SciMLSolver` protocol that all solvers must adhere to, enabling **interchangeability**.

- **Protocol**: `solve(problem) -> Solution`
- **Flexibility**: You can standardize your benchmarking pipeline. Switching from a `PINNSolver` to a `NeuralOperatorSolver` or a `HybridSolver` requires changing just one line of code.
- **Artifex Integration**: The `ArtifexSolverAdapter` brings generative models (Diffusion, Flows) into the solver ecosystem.

### Layer 5: Probabilistic Numerics & Uncertainty
**A Major Pillar of Opifex.** We treat Uncertainty Quantification (UQ) not as an afterthought, but as a first-class citizen via the **Probabilistic Numerics** paradigm.

- **Bayesian Inference**: `EnsembleWrapper` and Hamiltonian Monte Carlo (HMC) integration for rigorous posterior estimation.
- **Conformal Prediction**: `ConformalWrapper` provides frequentist coverage guarantees (e.g., "95% confidence that the true solution is within this band").
- **Generative Modeling**: `GenerativeWrapper` bridges Opifex with **Artifex**, allowing solvers to learn full distributions over solution spaces using Diffusion Models and Flow Matching.

This layer transforms any deterministic `SciMLSolver` into a Probabilistic Solver.

### Layer 6: Production Ecosystem
The top layer handles the lifecycle of models, including versioning, serving, and monitoring, bridging the gap between research and production value.

---

## Design Principles

The architecture of Opifex is guided by five core philosophies:

### 1. Protocol-First Design
Every major component (`SciMLSolver`, `Geometry`, `TrainingComponent`) is defined by a Protocol. This allows users to inject custom implementations without carrying the weight of a base class hierarchy. If it quacks like a Solver, it is a Solver.

### 2. Composition Over Inheritance
We favor composing behavior over deep inheritance trees. A `PINNSolver` is not a monolith; it is composed of a `PhysicsLoss`, a `Trainer`, and an `AdaptiveSampler`. This makes it easy to swap out the optimizer or the sampling strategy without rewriting the solver.

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
