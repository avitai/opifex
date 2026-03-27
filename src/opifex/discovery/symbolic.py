"""Symbolic regression wrapper for equation discovery.

Provides a thin bridge to PySR (Julia-based symbolic regression) as
an optional dependency. When PySR is not installed, falls back to a
simplified brute-force search over a small expression set.

Reference:
    Cranmer (2023) "Interpretable Machine Learning for Science with
    PySR and SymbolicRegression.jl"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np


if TYPE_CHECKING:
    import jax

logger = logging.getLogger(__name__)

try:
    import pysr  # type: ignore[import-untyped]

    _PYSR_AVAILABLE = True
except ImportError:
    _PYSR_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]


@dataclass(frozen=True, slots=True, kw_only=True)
class SymbolicRegressionConfig:
    """Configuration for symbolic regression.

    Attributes:
        max_complexity: Maximum expression complexity.
        populations: Number of evolutionary populations.
        niterations: Number of search iterations.
        binary_operators: Allowed binary operators.
        unary_operators: Allowed unary operators.
    """

    max_complexity: int = 20
    populations: int = 30
    niterations: int = 40
    binary_operators: tuple[str, ...] = ("+", "-", "*", "/")
    unary_operators: tuple[str, ...] = ("sin", "cos", "exp", "sqrt")


class SymbolicRegressor:
    """Symbolic regression for discovering closed-form expressions.

    Uses PySR when available, otherwise falls back to a simple
    polynomial fit as a baseline.

    Usage::

        reg = SymbolicRegressor()
        reg.fit(x, y)
        print(reg.best_equation())
        y_pred = reg.predict(x)
    """

    def __init__(self, config: SymbolicRegressionConfig | None = None) -> None:
        """Initialize symbolic regressor.

        Args:
            config: Search configuration. Uses defaults if None.
        """
        self.config = config or SymbolicRegressionConfig()
        self._model: Any = None
        self._fallback_coef: jnp.ndarray | None = None
        self._n_features: int = 0

    def fit(self, x: jax.Array, y: jax.Array) -> None:
        """Fit symbolic regression to data.

        Args:
            x: Input features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
        """
        self._n_features = x.shape[1]

        if _PYSR_AVAILABLE:
            self._fit_pysr(np.asarray(x), np.asarray(y))
        else:
            logger.debug("PySR not available; using polynomial fallback")
            self._fit_fallback(x, y)

    def predict(self, x: jax.Array) -> jnp.ndarray:
        """Predict using the discovered expression.

        Args:
            x: Input features, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).
        """
        if _PYSR_AVAILABLE and self._model is not None:
            return jnp.array(self._model.predict(np.asarray(x)))

        if self._fallback_coef is not None:
            # Polynomial prediction: [1, x0, x1, ..., x0^2, ...]
            features = self._poly_features(x)
            return features @ self._fallback_coef

        raise RuntimeError("Model has not been fit.")

    def best_equation(self) -> str:
        """Get the best discovered equation as a string.

        Returns:
            Human-readable equation string.
        """
        if _PYSR_AVAILABLE and self._model is not None:
            return str(self._model.sympy())

        if self._fallback_coef is not None:
            terms = []
            names = self._poly_names()
            for i, name in enumerate(names):
                c = float(self._fallback_coef[i])
                if abs(c) > 1e-6:
                    terms.append(f"{c:.4f}*{name}")
            return " + ".join(terms) if terms else "0"

        raise RuntimeError("Model has not been fit.")

    def _fit_pysr(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit using PySR."""
        self._model = pysr.PySRRegressor(
            niterations=self.config.niterations,
            binary_operators=list(self.config.binary_operators),
            unary_operators=list(self.config.unary_operators),
            maxsize=self.config.max_complexity,
            populations=self.config.populations,
        )
        self._model.fit(x, y)

    def _fit_fallback(self, x: jax.Array, y: jax.Array) -> None:
        """Fallback: fit polynomial regression."""
        features = self._poly_features(x)
        # Least squares fit
        self._fallback_coef = jnp.linalg.lstsq(features, y)[0]

    def _poly_features(self, x: jax.Array) -> jnp.ndarray:
        """Build degree-2 polynomial features: [1, x0, x1, ..., x0^2, x0*x1, ...]."""
        n = x.shape[1]
        cols = [jnp.ones(x.shape[0])]  # constant
        for i in range(n):
            cols.append(x[:, i])
        for i in range(n):
            for j in range(i, n):
                cols.append(x[:, i] * x[:, j])
        return jnp.column_stack(cols)

    def _poly_names(self) -> list[str]:
        """Get polynomial feature names."""
        n = self._n_features
        names = ["1"]
        for i in range(n):
            names.append(f"x{i}")
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    names.append(f"x{i}^2")
                else:
                    names.append(f"x{i}*x{j}")
        return names


__all__ = ["SymbolicRegressionConfig", "SymbolicRegressor"]
