# Core API Reference

The `opifex.core` package provides the fundamental abstractions and interfaces for the Opifex framework.

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
