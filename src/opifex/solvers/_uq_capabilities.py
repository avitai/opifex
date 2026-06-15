"""UQ capability declarations for the solver surfaces (Task 7.2).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.solvers.__init__`` which then registers each declaration into the
singleton :class:`UQRegistry`.

Two categories:

* **Model-family entries** — the standard deterministic neural model
  family (``PINNSolver`` / ``HybridSolver`` / ``NeuralOperatorSolver``)
  declared as a single ``deterministic_baseline()`` with the three
  adapter strategies composed via :meth:`UQCapability.with_adapter`.
* **Solver-side aggregation utilities** — capability declarations for
  ``opifex.uncertainty.scientific.solutions.aggregate_solver_solutions``
  and ``summarize_stacked_sample_solution``. Per Phase 6 Task 6.2 the
  four legacy wrapper classes (``BayesianWrapper``, ``ConformalWrapper``,
  ``EnsembleWrapper``, ``GenerativeWrapper``) were deleted in favour of
  these two utility functions; capability metadata therefore points at
  the surviving entry points, not the deleted classes.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


def _adapter_baseline_with_notes(notes: str) -> UQCapability:
    """Return a deterministic baseline + 3 adapter strategies with ``notes``."""
    return (
        UQCapability(
            default_strategy=DefaultStrategy.DETERMINISTIC,
            native_nnx_module=True,
            notes=notes,
        )
        .with_adapter("ensemble")
        .with_adapter("conformal")
        .with_adapter("calibration")
    )


# Standard deterministic neural model family — covers the three
# entry-point solvers in ``opifex.solvers``. All three are deterministic
# trainers; UQ is layered on through the canonical adapter strategies.
_SOLVER_MODEL_CAPABILITIES: dict[str, UQCapability] = {
    "model:deterministic_baseline": _adapter_baseline_with_notes(
        "Standard deterministic neural model family. The three solver "
        "entry points (PINNSolver / HybridSolver / NeuralOperatorSolver) "
        "train a point-prediction NNX module; uncertainty is supplied "
        "by the ensemble / conformal / calibration adapter strategies."
    ),
    "solver:PINNSolver": _adapter_baseline_with_notes(
        "PINNSolver trains a deterministic PINN against a physics residual; UQ is adapter-mediated."
    ),
    "solver:HybridSolver": _adapter_baseline_with_notes(
        "HybridSolver couples a deterministic neural surrogate to a "
        "classical numerical solver; UQ is adapter-mediated on the "
        "neural component."
    ),
    "solver:NeuralOperatorSolver": _adapter_baseline_with_notes(
        "NeuralOperatorSolver wraps a deterministic neural operator; "
        "UQ is adapter-mediated (or supplied by UQNO when used as the "
        "underlying operator)."
    ),
}


# Solver-side UQ aggregation utilities. These are pure-JAX functions
# that take a stack of ``Solution`` objects and emit a moment summary;
# they propagate probabilistic-numerics uncertainty (caller-owned RNG
# draws across the stack) without owning a posterior themselves.
_SOLVER_AGGREGATION_CAPABILITIES: dict[str, UQCapability] = {
    "solver:aggregate_solver_solutions": UQCapability(
        supports_solver_uncertainty=True,
        native_jax_kernel=True,
        default_strategy=DefaultStrategy.PROBABILISTIC_NUMERICS,
        source_package="opifex",
        notes=(
            "Stacks a sequence of opifex.core.solver.interface.Solution "
            "objects and returns a SolutionDistribution carrying "
            "per-trajectory mean / std summaries. Pure JAX over arrays."
        ),
    ),
    "solver:summarize_stacked_sample_solution": UQCapability(
        supports_solver_uncertainty=True,
        native_jax_kernel=True,
        default_strategy=DefaultStrategy.PROBABILISTIC_NUMERICS,
        source_package="opifex",
        notes=(
            "Reduces a stacked-sample Solution into a SolutionDistribution "
            "moment summary. Companion to aggregate_solver_solutions."
        ),
    ),
}


SOLVER_CAPABILITIES: dict[str, UQCapability] = {
    **_SOLVER_MODEL_CAPABILITIES,
    **_SOLVER_AGGREGATION_CAPABILITIES,
}


__all__ = ["SOLVER_CAPABILITIES"]
