"""UQ capability declarations for the active-learning subpackage (Task 8.3).

Static, module-level constants — no import-time mutable side effects.
The :data:`ACTIVE_CAPABILITIES` table is consumed by the active
subpackage's ``__init__`` to seed the singleton :class:`UQRegistry`.

Plan reference: ``08-phase-pac-bayes-sbi-active-stochastic-fields.md``
lines 755-790 — Task 8.5 flips the per-subsystem capability flag now
that the BALD / EI / EIG / batch-acquisition kernels have landed.

Three registered surfaces:

* ``active:bald`` — single-point BALD / EI / Log-EI / UCB / LCB / PI
  acquisition kernels via :func:`opifex.uncertainty.active.acquire`.
* ``active:batch`` — BatchBALD / batch-MC-EI / q-EHVI multi-point
  acquisitions (``batch_active.py``).
* ``active:bayesian_experimental_design`` —
  :func:`expected_information_gain` + the BED loop driver
  (``experimental_design.py``).

All three advertise :attr:`supports_active_learning=True` and
:attr:`default_strategy=DefaultStrategy.ACTIVE_LEARNING`.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_BALD_ACQUISITION_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    supports_active_learning=True,
    default_strategy=DefaultStrategy.ACTIVE_LEARNING,
    source_package="opifex",
    notes=(
        "Single-point acquisition kernels (BALD / EI / Log-EI / UCB / "
        "LCB / PI) and the :func:`acquire` named-strategy dispatcher. "
        "Ported from trieste's TensorFlow originals into pure JAX; see "
        "per-kernel docstring cites in acquisition.py."
    ),
)


_BATCH_ACTIVE_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    supports_active_learning=True,
    default_strategy=DefaultStrategy.ACTIVE_LEARNING,
    source_package="opifex",
    notes=(
        "Batch acquisitions — BatchBALD greedy joint-MI maximisation, "
        "reparameterised Monte-Carlo batch EI, and q-EHVI multi-objective "
        "acquisition. References: Kirsch et al. arXiv:1906.08158 "
        "(BatchBALD); Wilson et al. arXiv:1712.00424 (qEI); Daulton et al. "
        "arXiv:2006.05078 (qEHVI)."
    ),
)


_BAYESIAN_EXPERIMENTAL_DESIGN_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    supports_active_learning=True,
    default_strategy=DefaultStrategy.ACTIVE_LEARNING,
    source_package="opifex",
    notes=(
        "Bayesian experimental design — :func:`expected_information_gain` "
        "(linear-Gaussian closed form + Monte-Carlo nested-sampling "
        "fallback) and :func:`bayesian_experimental_design_loop` BO loop "
        "driver. Reference: Foster et al. arXiv:1903.05480."
    ),
)


ACTIVE_CAPABILITIES: dict[str, UQCapability] = {
    "active:bald": _BALD_ACQUISITION_CAPABILITY,
    "active:batch": _BATCH_ACTIVE_CAPABILITY,
    "active:bayesian_experimental_design": _BAYESIAN_EXPERIMENTAL_DESIGN_CAPABILITY,
}


__all__ = ["ACTIVE_CAPABILITIES"]
