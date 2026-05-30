"""Shared formatting helpers for SINDy equation discovery.

Centralises the human-readable equation rendering used by every SINDy
variant (``SINDy``, ``WeakSINDy``, ``EnsembleSINDy``) so the term-building
and ``d{target}/dt = ...`` assembly live in exactly one place (Rule 1, DRY).
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax.numpy as jnp

_COEFFICIENT_EPS: float = 1e-10
_ENSEMBLE_STD_EPS: float = 0.01


def format_sindy_equations(
    coefficients: jnp.ndarray,
    names: Sequence[str],
    target_names: Sequence[str],
    precision: int,
    *,
    std: jnp.ndarray | None = None,
) -> list[str]:
    """Render discovered coefficients as readable ``d{target}/dt = ...`` strings.

    Args:
        coefficients: Coefficient matrix, shape ``(n_targets, n_library_terms)``.
            When ``std`` is given these are interpreted as ensemble means.
        names: Library feature names, one per coefficient column.
        target_names: Names for the state variables, one per coefficient row.
        precision: Decimal places used for every coefficient value.
        std: Optional per-coefficient standard deviations matching the shape of
            ``coefficients``. When provided, terms are rendered with the ensemble
            ``(mean±std) name`` notation; otherwise plain ``coef name`` terms.

    Returns:
        One equation string per target variable.
    """
    equations: list[str] = []
    n_targets = coefficients.shape[0]
    for target_idx in range(n_targets):
        terms: list[str] = []
        for lib_idx, name in enumerate(names):
            coef = float(coefficients[target_idx, lib_idx])
            if std is None:
                if abs(coef) > _COEFFICIENT_EPS:
                    terms.append(f"{coef:.{precision}f} {name}")
            else:
                std_value = float(std[target_idx, lib_idx])
                if abs(coef) > _COEFFICIENT_EPS or std_value > _ENSEMBLE_STD_EPS:
                    terms.append(f"({coef:.{precision}f}±{std_value:.{precision}f}) {name}")

        rhs = " + ".join(terms) if terms else "0"
        equations.append(f"d{target_names[target_idx]}/dt = {rhs}")

    return equations


__all__ = ["format_sindy_equations"]
