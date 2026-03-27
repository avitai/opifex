"""Core SINDy model for sparse identification of nonlinear dynamics.

Implements the SINDy algorithm: given time-series data x(t) and
derivatives dx/dt, finds the sparsest set of nonlinear functions
from a candidate library that explains the dynamics.

Reference:
    Brunton et al. (2016) "Discovering governing equations from data
    by sparse identification of nonlinear dynamical systems"
"""

from __future__ import annotations

from typing import Self, TYPE_CHECKING

import jax.numpy as jnp

from opifex.discovery.sindy.config import SINDyConfig
from opifex.discovery.sindy.library import CandidateLibrary
from opifex.discovery.sindy.optimizers import SR3, STLSQ


if TYPE_CHECKING:
    from collections.abc import Sequence


class SINDy:
    """Sparse Identification of Nonlinear Dynamics.

    Discovers governing equations from data by building a library of
    candidate nonlinear functions and using sparse regression to find
    the subset that best explains the observed dynamics.

    Usage::

        config = SINDyConfig(polynomial_degree=2, threshold=0.1)
        model = SINDy(config)
        model.fit(x, x_dot)

        print(model.equations(["x", "y", "z"]))
        x_dot_pred = model.predict(x)

    Attributes:
        config: SINDy configuration.
        library: Candidate function library.
        coefficients: Fitted sparse coefficient matrix (n_targets, n_library_terms).
    """

    def __init__(self, config: SINDyConfig | None = None) -> None:
        """Initialize SINDy model.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or SINDyConfig()
        self.library = CandidateLibrary(
            polynomial_degree=self.config.polynomial_degree,
            include_trig=self.config.include_trig,
            n_frequencies=self.config.n_frequencies,
        )
        self.coefficients: jnp.ndarray | None = None

    def fit(self, x: jnp.ndarray, x_dot: jnp.ndarray) -> Self:
        """Fit the SINDy model to data.

        Args:
            x: State data, shape (n_samples, n_features).
            x_dot: Time derivatives, shape (n_samples, n_features).

        Returns:
            self for method chaining.
        """
        # Build candidate library matrix Theta(x)
        theta = self.library.transform(x)

        # Sparse regression: x_dot ≈ Theta(x) @ Xi^T
        optimizer = self._create_optimizer()
        self.coefficients = optimizer.fit(theta, x_dot)

        return self

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict derivatives using the discovered model.

        Args:
            x: State data, shape (n_samples, n_features).

        Returns:
            Predicted derivatives, shape (n_samples, n_features).

        Raises:
            RuntimeError: If model has not been fit.
        """
        if self.coefficients is None:
            raise RuntimeError("Model has not been fit. Call fit() first.")

        theta = self.library.transform(x)
        return theta @ self.coefficients.T

    def equations(
        self,
        input_names: Sequence[str] | None = None,
        precision: int = 3,
    ) -> list[str]:
        """Get human-readable equation strings.

        Args:
            input_names: Names for state variables (e.g., ['x', 'y', 'z']).
            precision: Decimal places for coefficient values.

        Returns:
            List of equation strings, one per state variable.

        Raises:
            RuntimeError: If model has not been fit.
        """
        if self.coefficients is None:
            raise RuntimeError("Model has not been fit. Call fit() first.")

        names = self.feature_names(input_names)
        n_targets = self.coefficients.shape[0]
        target_names = input_names or [f"x{i}" for i in range(n_targets)]

        eqs = []
        for target_idx in range(n_targets):
            terms = []
            for lib_idx, name in enumerate(names):
                coef = float(self.coefficients[target_idx, lib_idx])
                if abs(coef) > 1e-10:
                    terms.append(f"{coef:.{precision}f} {name}")

            rhs = " + ".join(terms) if terms else "0"
            eqs.append(f"d{target_names[target_idx]}/dt = {rhs}")

        return eqs

    def feature_names(self, input_names: Sequence[str] | None = None) -> list[str]:
        """Get library feature names.

        Args:
            input_names: Names for state variables.

        Returns:
            List of feature name strings.
        """
        return self.library.get_feature_names(input_names)

    def score(self, x: jnp.ndarray, x_dot: jnp.ndarray) -> float:
        """Compute R² score of the model.

        Args:
            x: State data, shape (n_samples, n_features).
            x_dot: True derivatives, shape (n_samples, n_features).

        Returns:
            R² coefficient of determination.
        """
        x_dot_pred = self.predict(x)
        ss_res = jnp.sum((x_dot - x_dot_pred) ** 2)
        ss_tot = jnp.sum((x_dot - jnp.mean(x_dot, axis=0)) ** 2)
        return float(1.0 - ss_res / ss_tot)

    def _create_optimizer(self) -> STLSQ | SR3:
        """Create optimizer based on config."""
        if self.config.optimizer == "sr3":
            return SR3(threshold=self.config.threshold, max_iter=self.config.max_iter)
        return STLSQ(
            threshold=self.config.threshold,
            alpha=self.config.alpha,
            max_iter=self.config.max_iter,
        )


__all__ = ["SINDy"]
