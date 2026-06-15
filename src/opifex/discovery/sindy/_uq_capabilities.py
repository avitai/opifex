"""UQ capability declarations for the SINDy equation-discovery surfaces (Task 7.5).

Static, module-level constants — no import-time mutable side effects
beyond the constants themselves (Rule 13). Imported by
``opifex.discovery.sindy.__init__``.

The SINDy family advertises three honest UQ strategies:

* :class:`SINDy` and :class:`WeakSINDy` — pure least-squares /
  sparse-regression fits with no native posterior. UQ is adapter-mediated
  via the residual / bootstrap / conformal wrappers documented in
  Brunton et al. (2016) follow-ups; declared as ``DETERMINISTIC`` with
  conformal-adapter support.
* :class:`EnsembleSINDy` — native bootstrap-aggregated ensemble that
  reports per-term coefficient mean/std across resamples. Declared as
  ``ENSEMBLE`` with ``native_jax_kernel=True``.
* :func:`distill_ude_residual` — utility that distills a universal-DE
  residual onto a SINDy library; deterministic in its own right but
  carries solver-residual provenance, so declared as ``DETERMINISTIC``
  with calibration support for downstream wrappers.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_SINDY_CAPABILITY = (
    UQCapability(
        default_strategy=DefaultStrategy.DETERMINISTIC,
        native_jax_kernel=True,
        source_package="opifex",
        notes=(
            "SINDy fits a sparse linear combination over a candidate "
            "library (Brunton et al., 2016). No native posterior; UQ is "
            "adapter-mediated via residual-based bootstrap or conformal "
            "calibration on coefficient estimates."
        ),
    )
    .with_adapter("conformal")
    .with_adapter("calibration")
)


_WEAK_SINDY_CAPABILITY = (
    UQCapability(
        default_strategy=DefaultStrategy.DETERMINISTIC,
        native_jax_kernel=True,
        source_package="opifex",
        notes=(
            "Weak-form SINDy reduces sensitivity to noise via test "
            "functions (Messenger & Bortz, 2021). Deterministic core; "
            "UQ via residual/bootstrap or conformal adapters."
        ),
    )
    .with_adapter("conformal")
    .with_adapter("calibration")
)


_ENSEMBLE_SINDY_CAPABILITY = UQCapability(
    supports_ensemble=True,
    supports_calibration=True,
    default_strategy=DefaultStrategy.ENSEMBLE,
    native_jax_kernel=True,
    source_package="opifex",
    notes=(
        "EnsembleSINDy fits N bootstrap SINDy models on data subsets and "
        "reports per-term coefficient mean and std across the ensemble — "
        "the canonical bootstrap-aggregation strategy for equation-"
        "discovery UQ (Fasel et al., 2022)."
    ),
)


_DISTILL_UDE_RESIDUAL_CAPABILITY = (
    UQCapability(
        default_strategy=DefaultStrategy.DETERMINISTIC,
        native_jax_kernel=True,
        source_package="opifex",
        notes=(
            "Distills a universal-DE neural residual onto a SINDy "
            "library, surfacing the missing-physics term. Deterministic "
            "by construction; downstream conformal/calibration adapters "
            "wrap the resulting coefficient estimates."
        ),
    )
    .with_adapter("conformal")
    .with_adapter("calibration")
)


SINDY_CAPABILITIES: dict[str, UQCapability] = {
    "discovery:SINDy": _SINDY_CAPABILITY,
    "discovery:WeakSINDy": _WEAK_SINDY_CAPABILITY,
    "discovery:EnsembleSINDy": _ENSEMBLE_SINDY_CAPABILITY,
    "discovery:distill_ude_residual": _DISTILL_UDE_RESIDUAL_CAPABILITY,
}


__all__ = ["SINDY_CAPABILITIES"]
