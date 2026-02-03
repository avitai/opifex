# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Learn-to-Optimize (L2O): Neural Optimization for Parametric PDE Families

This example demonstrates Opifex's Learn-to-Optimize (L2O) engine for solving
**families of parametric PDEs**. In scientific computing, we often need to solve
the same type of PDE (e.g., diffusion, Poisson) with varying parameters (coefficients,
boundary conditions, source terms). L2O learns problem-specific strategies that
transfer across the parameter space.

**SciML Context:**
When discretizing elliptic PDEs like `-∇·(κ∇u) = f`, we obtain linear systems
`Au = b` where `A` is symmetric positive definite (SPD). The matrix `A` depends
on the PDE coefficient `κ`, while `b` depends on the source term `f`. L2O learns
to solve these parametric systems faster than generic iterative methods.

**Key Concepts:**
- Parametric PDE families and their discretizations
- Problem encoding for neural optimization
- Automatic algorithm selection based on problem structure
- Meta-learning across related PDE instances
- Amortized optimization for repeated solves
"""

# %%
# Configuration
SEED = 42
NUM_PROBLEMS = 50  # Number of problems to solve
PROBLEM_DIM = 10  # Dimension of optimization variables
PARAM_DIM = 20  # Dimension of problem parameters
NUM_TRAIN_STEPS = 50  # Steps for gradient-based optimization

# Output directory
OUTPUT_DIR = "docs/assets/examples/learn_to_optimize"

# %%
print("=" * 70)
print("Opifex Example: Learn-to-Optimize (L2O) Engine")
print("=" * 70)

# %%
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx


print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %%
from opifex.core.training.config import MetaOptimizerConfig
from opifex.optimization.l2o import (
    L2OEngine,
    L2OEngineConfig,
    OptimizationProblem,
)


# %% [markdown]
"""
## Step 1: Configure L2O Engine

The L2O engine integrates parametric solvers with gradient-based meta-optimization.
Key configuration options:
- `solver_type`: "parametric", "gradient", or "hybrid"
- `use_traditional_fallback`: Enable fallback to traditional methods
- `enable_meta_learning`: Learn from related problems
"""

# %%
print()
print("Configuring L2O Engine...")
print("-" * 50)

# L2O engine configuration
l2o_config = L2OEngineConfig(
    solver_type="hybrid",  # Use both parametric and gradient methods
    problem_encoder_layers=[64, 32, 16],
    use_traditional_fallback=True,
    enable_meta_learning=True,
    integration_mode="unified",
    speedup_threshold=100.0,
    performance_tracking=True,
    adaptive_selection=True,
)

# Meta-optimizer configuration for gradient-based L2O
meta_config = MetaOptimizerConfig(
    meta_algorithm="l2o",
    base_optimizer="adam",
    meta_learning_rate=1e-3,
    adaptation_steps=10,
    performance_tracking=True,
)

print(f"  Solver type: {l2o_config.solver_type}")
print(f"  Integration mode: {l2o_config.integration_mode}")
print(f"  Meta-learning: {l2o_config.enable_meta_learning}")
print(f"  Encoder layers: {l2o_config.problem_encoder_layers}")

# %%
# Initialize L2O engine
print()
print("Initializing L2O Engine...")

rngs = nnx.Rngs(SEED)
l2o_engine = L2OEngine(l2o_config, meta_config, rngs=rngs)

print("  L2O Engine initialized successfully!")

# %% [markdown]
"""
## Step 2: Create Parametric PDE Family

We create a family of **discrete elliptic PDE problems** with varying coefficients.
When discretizing `-∇·(κ∇u) = f` on a grid, we obtain systems `Au = b` where:
- `A` is the discrete Laplacian weighted by diffusion coefficient `κ`
- `b` is the discretized source term `f`

Each problem corresponds to different parameter values (κ, f), representing
different physical scenarios (e.g., varying thermal conductivity, different heat sources).
"""

# %%
print()
print("Creating parametric PDE problem family...")
print("-" * 50)


def create_discrete_elliptic_problem(key, dim):
    """Create a discrete elliptic PDE problem.

    Simulates discretization of: -∇·(κ∇u) = f
    where κ is the diffusion coefficient and f is the source term.
    The resulting system Au = b has A symmetric positive definite (SPD).
    """
    key1, key2 = jax.random.split(key)

    # Random SPD matrix representing discrete diffusion operator
    # A = L^T D L where L is lower triangular, D is diagonal with positive entries
    # This ensures A is SPD (required for elliptic PDE discretizations)
    a_raw = jax.random.normal(key1, (dim, dim))
    a_matrix = jnp.dot(a_raw.T, a_raw) + jnp.eye(dim) * 0.1

    # Random source term (RHS of PDE)
    b_vector = jax.random.normal(key2, (dim,))

    # Problem parameters encode the PDE coefficients
    params = jnp.concatenate(
        [a_matrix.flatten()[:PARAM_DIM], b_vector[: PARAM_DIM - dim * dim]]
    )

    # Pad or truncate to PARAM_DIM
    if params.size < PARAM_DIM:
        params = jnp.pad(params, (0, PARAM_DIM - params.size))
    else:
        params = params[:PARAM_DIM]

    return a_matrix, b_vector, params


# Generate parametric PDE family
key = jax.random.PRNGKey(SEED)
problems = []
problem_params_list = []

for _ in range(NUM_PROBLEMS):
    key, subkey = jax.random.split(key)
    a, b, params = create_discrete_elliptic_problem(subkey, PROBLEM_DIM)

    # Create OptimizationProblem object (quadratic type for SPD systems)
    problem = OptimizationProblem(
        dimension=PROBLEM_DIM,
        problem_type="quadratic",  # SPD systems are quadratic optimization
    )
    problems.append((problem, a, b))
    problem_params_list.append(params)

problem_params = jnp.stack(problem_params_list)

print(f"  Created {NUM_PROBLEMS} parametric elliptic PDE problems")
print(f"  Discretization dimension: {PROBLEM_DIM}")
print(f"  Parameter dimension: {PARAM_DIM}")

# %% [markdown]
"""
## Step 3: Solve with L2O Engine

We solve the optimization problems using the L2O engine and compare
with traditional methods.
"""

# %%
print()
print("Solving optimization problems...")
print("-" * 50)

l2o_solutions = []
l2o_times = []
algorithms_used = []

for i, ((problem, _a, _b), params) in enumerate(
    zip(problems, problem_params_list, strict=False)
):
    start_time = time.time()

    # Get recommendation and solve
    algorithm, solution = l2o_engine.solve_automatically(problem, params)

    l2o_time = time.time() - start_time

    l2o_solutions.append(solution)
    l2o_times.append(l2o_time)
    algorithms_used.append(algorithm)

    if (i + 1) % 10 == 0:
        print(f"  Solved {i + 1}/{NUM_PROBLEMS} problems...")

l2o_solutions = jnp.stack(l2o_solutions)
l2o_times = jnp.array(l2o_times)

print()
print(f"  Total L2O time: {jnp.sum(l2o_times):.4f}s")
print(f"  Mean time per problem: {jnp.mean(l2o_times):.6f}s")
parametric_count = sum(1 for a in algorithms_used if a == "parametric")
gradient_count = sum(1 for a in algorithms_used if a == "gradient")
print(
    f"  Algorithm distribution: parametric={parametric_count}, gradient={gradient_count}"
)

# %% [markdown]
"""
## Step 4: Compare with Iterative PDE Solver

We compare L2O performance against a traditional iterative solver (steepest descent).
In practice, PDE systems are solved using iterative methods like Conjugate Gradient,
GMRES, or multigrid. Here we use steepest descent as a simple baseline.
"""

# %%
print()
print("Comparing with iterative solver (steepest descent)...")
print("-" * 50)


def solve_elliptic_iterative(a_matrix, b_vector, steps=100):
    """Solve discrete elliptic PDE with steepest descent.

    For the system Au = b where A is SPD, steepest descent converges
    with rate depending on the condition number of A.

    Uses adaptive step size based on spectral radius for stability.
    """
    x = jnp.zeros(a_matrix.shape[0])

    # Compute adaptive step size: alpha = 1 / lambda_max(A)
    # For SPD matrices, this ensures convergence
    eigvals = jnp.linalg.eigvalsh(a_matrix)
    spectral_radius = jnp.max(eigvals)
    step_size = 0.9 / spectral_radius  # Safety margin

    for _ in range(steps):
        # Residual r = b - Ax, gradient of ||Au - b||² is A^T(Ax - b) = A(Ax - b)
        # For SPD A, gradient is 2A(x) - b when minimizing x^T A x - b^T x
        grad = jnp.dot(a_matrix, x) - 0.5 * b_vector
        x = x - step_size * grad

    return x


iterative_solutions = []
iterative_times = []

for _problem, a, b in problems:
    start_time = time.time()
    solution = solve_elliptic_iterative(a, b, steps=NUM_TRAIN_STEPS)
    solve_time = time.time() - start_time

    iterative_solutions.append(solution)
    iterative_times.append(solve_time)

iterative_solutions = jnp.stack(iterative_solutions)
iterative_times = jnp.array(iterative_times)

print(f"  Total iterative time: {jnp.sum(iterative_times):.4f}s")
print(f"  Mean time per problem: {jnp.mean(iterative_times):.6f}s")

# %%
# Compute optimal solutions analytically
print()
print("Computing analytical solutions...")

optimal_solutions = []
for _problem, a, b in problems:
    # Optimal solution: x* = -0.5 * A^(-1) * b
    try:
        x_opt = -0.5 * jnp.linalg.solve(a, b)
    except Exception:
        x_opt = jnp.zeros(PROBLEM_DIM)
    optimal_solutions.append(x_opt)

optimal_solutions = jnp.stack(optimal_solutions)

# %%
# Compute errors
l2o_errors = jnp.linalg.norm(l2o_solutions - optimal_solutions, axis=1)
iterative_errors = jnp.linalg.norm(iterative_solutions - optimal_solutions, axis=1)

print()
print("Performance Comparison:")
print("-" * 50)
print(f"  L2O Mean Error:     {jnp.mean(l2o_errors):.6f}")
print(f"  Iterative Mean Error:      {jnp.mean(iterative_errors):.6f}")
print(f"  L2O Mean Time:      {jnp.mean(l2o_times) * 1000:.3f}ms")
print(f"  Iterative Mean Time:       {jnp.mean(iterative_times) * 1000:.3f}ms")
print(f"  Speedup Factor:     {jnp.mean(iterative_times) / jnp.mean(l2o_times):.1f}x")

# %% [markdown]
"""
## Step 5: Meta-Learning Across Problems

Demonstrate how the L2O engine learns from solving multiple problems.
"""

# %%
print()
print("Demonstrating meta-learning...")
print("-" * 50)

# Solve a new batch of problems with meta-learning enabled
meta_solutions = []
meta_metadata = []

for i, ((problem, _a, _b), params) in enumerate(
    zip(problems[:10], problem_params_list[:10], strict=False)
):
    solution, metadata = l2o_engine.solve_with_meta_learning(
        problem, params, problem_id=i
    )
    meta_solutions.append(solution)
    meta_metadata.append(metadata)

print(f"  Problems in memory: {len(l2o_engine.problem_memory)}")
print(f"  Solutions in memory: {len(l2o_engine.solution_memory)}")
print(f"  Meta-learning enabled: {meta_metadata[0]['meta_learning_used']}")

# %% [markdown]
"""
## Step 6: Algorithm Recommendation

Show how the L2O engine recommends algorithms based on problem characteristics.
"""

# %%
print()
print("Algorithm Recommendations:")
print("-" * 50)

# Test different problem types
test_cases = [
    ("Small quadratic", OptimizationProblem(dimension=5, problem_type="quadratic")),
    ("Large quadratic", OptimizationProblem(dimension=150, problem_type="quadratic")),
    ("Linear", OptimizationProblem(dimension=20, problem_type="linear")),
    ("Nonlinear", OptimizationProblem(dimension=30, problem_type="nonlinear")),
]

for name, problem in test_cases:
    recommendation = l2o_engine.recommend_algorithm(problem, jnp.zeros(PARAM_DIM))
    print(f"  {name} (dim={problem.dimension}): {recommendation}")

# %% [markdown]
"""
## Step 7: Visualization
"""

# %%
print()
print("Generating visualizations...")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# %%
# Filter out NaN/Inf values for visualization
l2o_errors_valid = l2o_errors[jnp.isfinite(l2o_errors)]
iterative_errors_valid = iterative_errors[jnp.isfinite(iterative_errors)]

# Figure 1: Error comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Error distribution
ax1 = axes[0]
if len(l2o_errors_valid) > 0:
    ax1.hist(
        l2o_errors_valid,
        bins=20,
        alpha=0.7,
        label="L2O",
        color="blue",
        edgecolor="black",
    )
if len(iterative_errors_valid) > 0:
    ax1.hist(
        iterative_errors_valid,
        bins=20,
        alpha=0.7,
        label="Iterative Solver",
        color="orange",
        edgecolor="black",
    )
ax1.set_xlabel("Solution Error (L2 norm)", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_title("Error Distribution", fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Time comparison
ax2 = axes[1]
ax2.hist(
    l2o_times * 1000, bins=20, alpha=0.7, label="L2O", color="blue", edgecolor="black"
)
ax2.hist(
    iterative_times * 1000,
    bins=20,
    alpha=0.7,
    label="Iterative Solver",
    color="orange",
    edgecolor="black",
)
ax2.set_xlabel("Solve Time (ms)", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("Time Distribution", fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR}/comparison.png")

# %%
# Figure 2: Error vs time trade-off
fig, ax = plt.subplots(figsize=(8, 6))

# Filter valid data for scatter plot
l2o_mask = jnp.isfinite(l2o_errors)
iterative_mask = jnp.isfinite(iterative_errors)

ax.scatter(
    l2o_times[l2o_mask] * 1000,
    l2o_errors[l2o_mask],
    alpha=0.7,
    s=50,
    c="blue",
    label="L2O",
    marker="o",
)
ax.scatter(
    iterative_times[iterative_mask] * 1000,
    iterative_errors[iterative_mask],
    alpha=0.7,
    s=50,
    c="orange",
    label="Iterative Solver",
    marker="s",
)

ax.set_xlabel("Solve Time (ms)", fontsize=12)
ax.set_ylabel("Solution Error", fontsize=12)
ax.set_title("Error vs Time Trade-off", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tradeoff.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Saved: {OUTPUT_DIR}/tradeoff.png")

# %% [markdown]
"""
## Results Summary
"""

# %%
print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print("Configuration:")
print(f"  Number of problems: {NUM_PROBLEMS}")
print(f"  Problem dimension: {PROBLEM_DIM}")
print(f"  Solver type: {l2o_config.solver_type}")
print()
print("Performance:")
l2o_mean_err = float(jnp.mean(l2o_errors[jnp.isfinite(l2o_errors)]))
iterative_mean_err = float(jnp.mean(iterative_errors[jnp.isfinite(iterative_errors)]))
print(f"  L2O Mean Error:  {l2o_mean_err:.6f}")
print(f"  Iterative Mean Error:   {iterative_mean_err:.6f}")
print(f"  L2O Total Time:  {jnp.sum(l2o_times):.4f}s")
print(f"  Iterative Total Time:   {jnp.sum(iterative_times):.4f}s")
print(f"  Speedup Factor:  {jnp.mean(iterative_times) / jnp.mean(l2o_times):.1f}x")
print()
print("Meta-Learning:")
print(f"  Problems in memory: {len(l2o_engine.problem_memory)}")
print("=" * 70)

# %%
print()
print("L2O example completed successfully!")
