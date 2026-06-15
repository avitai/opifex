"""UQ capability declarations for the equation-discovery surfaces (Task 7.5).

Covers the top-level :mod:`opifex.discovery` exports — at present, the
``SymbolicRegressor`` thin bridge to PySR. The :mod:`opifex.discovery.sindy`
subpackage carries its own declarations (see
:mod:`opifex.discovery.sindy._uq_capabilities`).

Static, module-level constants — no import-time mutable side effects beyond
the constants themselves (Rule 13). Imported by
``opifex.discovery.__init__``.
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


# Symbolic regression is a PySR-mediated brute-force / genetic search over
# expression families. There is no native UQ; uncertainty would have to come
# from bootstrap-style resampling of the candidate-expression population
# (not yet implemented in opifex). Declared honestly as UNSUPPORTED.
_SYMBOLIC_REGRESSOR_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="pysr",
    notes=(
        "SymbolicRegressor is a thin bridge to PySR (Julia-based symbolic "
        "regression) with a brute-force fallback. No native posterior over "
        "expressions; bootstrap/ensemble UQ over the candidate population "
        "is a Phase 8 follow-up. Phase 7 records UNSUPPORTED status."
    ),
)


DISCOVERY_CAPABILITIES: dict[str, UQCapability] = {
    "discovery:SymbolicRegressor": _SYMBOLIC_REGRESSOR_CAPABILITY,
}


__all__ = ["DISCOVERY_CAPABILITIES"]
