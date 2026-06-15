"""UQ capability declarations for the neural quantum chemistry surfaces (Task 7.5).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.neural.quantum.__init__``.

The three public quantum surfaces (``NeuralDFT`` / ``NeuralSCFSolver`` /
``NeuralXCFunctional``) are deterministic NNX modules: they replace
hand-tuned exchange-correlation functionals and SCF mixers with neural
approximants, but do not own a posterior over their parameters. The
honest strategy is:

* ``DETERMINISTIC`` default with the three adapter strategies
  (ensemble / conformal / calibration) layered on top — same pattern as
  the deterministic-baseline neural model family registered by Task 7.2.
* Measurement-sampling uncertainty is conceptually available
  through the wrapped electronic-structure calculation but is not yet
  surfaced as a native UQ flag on these classes; recorded in
  :attr:`notes`.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


def _quantum_adapter_baseline(notes: str) -> UQCapability:
    """Return a deterministic baseline + three adapter strategies."""
    return (
        UQCapability(
            default_strategy=DefaultStrategy.DETERMINISTIC,
            native_nnx_module=True,
            source_package="opifex",
            notes=notes,
        )
        .with_adapter("ensemble")
        .with_adapter("conformal")
        .with_adapter("calibration")
    )


_NEURAL_DFT_CAPABILITY = _quantum_adapter_baseline(
    "NeuralDFT runs the Kohn-Sham SCF cycle with neural exchange-correlation "
    "and density-mixing modules; deterministic by default. UQ is supplied by "
    "ensemble / conformal / calibration adapters wrapping the predicted "
    "energy / density. Measurement-sampling uncertainty (when the wrapped "
    "system is sampled stochastically) is not yet a native flag — see "
    "Phase 8 follow-up."
)


_NEURAL_SCF_SOLVER_CAPABILITY = _quantum_adapter_baseline(
    "NeuralSCFSolver couples a DensityMixingNetwork + ConvergencePredictor "
    "to accelerate the Roothaan SCF loop. Deterministic NNX module; UQ "
    "comes from the standard three adapter strategies."
)


_NEURAL_XC_FUNCTIONAL_CAPABILITY = _quantum_adapter_baseline(
    "NeuralXCFunctional is an attention-based exchange-correlation "
    "functional. Deterministic NNX module; UQ adapter-mediated through "
    "ensemble / conformal / calibration."
)


QUANTUM_CAPABILITIES: dict[str, UQCapability] = {
    "quantum:NeuralDFT": _NEURAL_DFT_CAPABILITY,
    "quantum:NeuralSCFSolver": _NEURAL_SCF_SOLVER_CAPABILITY,
    "quantum:NeuralXCFunctional": _NEURAL_XC_FUNCTIONAL_CAPABILITY,
}


__all__ = ["QUANTUM_CAPABILITIES"]
