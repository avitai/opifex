# Core API Reference

The `opifex.core` package provides the fundamental abstractions and interfaces for the Opifex framework.

## The Solve Protocol

A problem states *what* to solve and a method *how*. A method names the problem family it
solves as its generic argument, and `solve(problem, method)` refuses any other pairing.
Both are frozen pytrees, so one compiled program serves every problem instance and only a
new method configuration compiles again. Time-stepping methods take a static number of
steps through `integrate`, which saves states on a `Snapshots` schedule and
differentiates in reverse mode.

```python
import jax
import jax.numpy as jnp
from opifex.core.solver import Method, Problem, Solution, Status, solve


class DecayProblem(Problem):
    initial: jax.Array
    rate: jax.Array
    time_final: jax.Array


class ExactDecay(Method[DecayProblem]):
    def solve(self, problem: DecayProblem) -> Solution:
        u = problem.initial * jnp.exp(-problem.rate * problem.time_final)
        return Solution(fields={"u": u}, metrics={}, status=jnp.asarray(Status.SUCCESS, jnp.int32))


solution = solve(DecayProblem(jnp.asarray(1.0), jnp.asarray(0.5), jnp.asarray(2.0)), ExactDecay())
```

::: opifex.core.solver.problem
    options:
        show_root_heading: true
        members: [Problem, Method, solve, IncompatibleProblemError]

::: opifex.core.solver.integrate
    options:
        show_root_heading: true
        members: [integrate, Snapshots, FINAL_STATE]

::: opifex.core.solver.solution.Solution
    options:
        show_root_heading: true

## Problems

The `Problem` interface is defined as a Protocol, serving as the unified contract for all scientific machine learning problems.

### Problem Protocol

::: opifex.core.problems.Problem
    options:
        show_root_heading: true
        show_source: true

### PDE Problems

::: opifex.core.problems.PDEProblem
    options:
        show_root_heading: true
        show_source: true

### ODE Problems

::: opifex.core.problems.ODEProblem
    options:
        show_root_heading: true
        show_source: true

### Optimization Problems

::: opifex.core.problems.OptimizationProblem
    options:
        show_root_heading: true
        show_source: true

### Quantum Problems

::: opifex.core.problems.QuantumProblem
    options:
        show_root_heading: true
        show_source: true

::: opifex.core.problems.ElectronicStructureProblem
    options:
        show_root_heading: true
        show_source: true

### Factory Functions

::: opifex.core.problems.create_pde_problem
::: opifex.core.problems.create_ode_problem
::: opifex.core.problems.create_optimization_problem
::: opifex.core.problems.create_neural_dft_problem

## Boundary Conditions

### Base Classes

::: opifex.core.conditions.BoundaryCondition
::: opifex.core.conditions.InitialCondition
::: opifex.core.conditions.Constraint

### Classical Boundary Conditions

::: opifex.core.conditions.DirichletBC
::: opifex.core.conditions.NeumannBC
::: opifex.core.conditions.RobinBC

### Quantum Boundary Conditions & Constraints

::: opifex.core.conditions.WavefunctionBC
::: opifex.core.conditions.DensityConstraint
::: opifex.core.conditions.SymmetryConstraint

### Collections

::: opifex.core.conditions.BoundaryConditionCollection
