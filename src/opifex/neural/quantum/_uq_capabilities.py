"""UQ capability declarations for the neural quantum chemistry surfaces (Task 7.5).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.neural.quantum.__init__``.

The learned exchange-correlation functional ``NeuralXCFunctional`` is a
deterministic NNX module: it replaces a hand-tuned exchange-correlation
functional with a constrained neural approximant inside the real Kohn-Sham
SCF, but does not own a posterior over its parameters. The honest strategy is:

* ``DETERMINISTIC`` default with the three adapter strategies
  (ensemble / conformal / calibration) layered on top — the same pattern as
  the deterministic-baseline neural model family registered by Task 7.2.
* Measurement-sampling uncertainty is conceptually available through the
  wrapped electronic-structure calculation but is not yet surfaced as a
  native UQ flag on the class; recorded in :attr:`notes`.
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


_NEURAL_XC_FUNCTIONAL_CAPABILITY = _quantum_adapter_baseline(
    "NeuralXCFunctional is an attention-based exchange-correlation "
    "functional driving the real Kohn-Sham SCF. Deterministic NNX module; "
    "UQ adapter-mediated through ensemble / conformal / calibration."
)


QUANTUM_CAPABILITIES: dict[str, UQCapability] = {
    "quantum:NeuralXCFunctional": _NEURAL_XC_FUNCTIONAL_CAPABILITY,
}


__all__ = ["QUANTUM_CAPABILITIES"]
