"""Ensemble SINDy for robust equation discovery with uncertainty.

Runs multiple SINDy fits on bootstrapped data subsets and/or library
subsets, providing coefficient statistics (mean, std) for uncertainty
quantification on the discovered equations.

Reference:
    Fasel et al. (2022) "Ensemble-SINDy: Robust sparse model discovery
    in the low-data, high-noise limit"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.discovery.sindy.library import CandidateLibrary
from opifex.discovery.sindy.optimizers import STLSQ


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opifex.discovery.sindy.config import EnsembleSINDyConfig


class EnsembleSINDy:
    """Ensemble SINDy with bootstrap aggregation for uncertainty.

    Fits multiple SINDy models on data subsets (bagging) and reports
    coefficient statistics (mean, std) across the ensemble. This provides
    uncertainty estimates on discovered equation terms.

    Attributes:
        config: Ensemble configuration.
        coef_mean: Mean coefficients across ensemble, shape (n_targets, n_library).
        coef_std: Std of coefficients across ensemble, shape (n_targets, n_library).
        coef_list: List of all individual model coefficients.
    """

    def __init__(self, config: EnsembleSINDyConfig) -> None:
        """Initialize Ensemble SINDy.

        Args:
            config: Ensemble configuration with n_models, bagging_fraction, etc.
        """
        self.config = config
        self.library = CandidateLibrary(
            polynomial_degree=config.polynomial_degree,
            include_trig=config.include_trig,
            n_frequencies=config.n_frequencies,
        )
        self.coef_mean: jnp.ndarray | None = None
        self.coef_std: jnp.ndarray | None = None
        self.coef_list: list[jnp.ndarray] = []

    def fit(
        self,
        x: jnp.ndarray,
        x_dot: jnp.ndarray,
        *,
        key: jax.Array,
    ) -> None:
        """Fit ensemble of SINDy models via bootstrap aggregation.

        Args:
            x: State data, shape (n_samples, n_features).
            x_dot: Time derivatives, shape (n_samples, n_features).
            key: JAX PRNG key for random subsampling.
        """
        theta = self.library.transform(x)
        n_samples = theta.shape[0]
        subset_size = int(n_samples * self.config.bagging_fraction)

        self.coef_list = []
        for i in range(self.config.n_models):
            subkey = jax.random.fold_in(key, i)

            # Bootstrap subsample
            indices = jax.random.choice(subkey, n_samples, shape=(subset_size,), replace=True)
            theta_sub = theta[indices]
            x_dot_sub = x_dot[indices]

            # Fit single model
            optimizer = STLSQ(
                threshold=self.config.threshold,
                alpha=self.config.alpha,
                max_iter=self.config.max_iter,
            )
            coef = optimizer.fit(theta_sub, x_dot_sub)
            self.coef_list.append(coef)

        # Aggregate statistics
        stacked = jnp.stack(self.coef_list, axis=0)  # (n_models, n_targets, n_lib)
        self.coef_mean = jnp.mean(stacked, axis=0)
        self.coef_std = jnp.std(stacked, axis=0)

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict using mean ensemble coefficients.

        Args:
            x: State data, shape (n_samples, n_features).

        Returns:
            Predicted derivatives using mean coefficients.
        """
        if self.coef_mean is None:
            raise RuntimeError("Model has not been fit.")
        theta = self.library.transform(x)
        return theta @ self.coef_mean.T

    def equations(
        self,
        input_names: Sequence[str] | None = None,
        precision: int = 3,
    ) -> list[str]:
        """Get equations using mean coefficients with uncertainty.

        Args:
            input_names: Names for state variables.
            precision: Decimal places for values.

        Returns:
            Equation strings with coefficient ± std notation.
        """
        if self.coef_mean is None or self.coef_std is None:
            raise RuntimeError("Model has not been fit.")

        names = self.library.get_feature_names(input_names)
        n_targets = self.coef_mean.shape[0]
        target_names = input_names or [f"x{i}" for i in range(n_targets)]

        eqs = []
        for tidx in range(n_targets):
            terms = []
            for lidx, name in enumerate(names):
                mean = float(self.coef_mean[tidx, lidx])
                std = float(self.coef_std[tidx, lidx])
                if abs(mean) > 1e-10 or std > 0.01:
                    terms.append(f"({mean:.{precision}f}±{std:.{precision}f}) {name}")
            rhs = " + ".join(terms) if terms else "0"
            eqs.append(f"d{target_names[tidx]}/dt = {rhs}")

        return eqs


__all__ = ["EnsembleSINDy"]
