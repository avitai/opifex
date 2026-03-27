"""Weak-form SINDy for noise-robust equation discovery.

Instead of computing pointwise derivatives (amplifies noise), WeakSINDy
integrates the governing equation against smooth test functions. Integration
by parts transfers derivatives from noisy data onto known test functions.

Given dx/dt = f(x), the weak form is:

    ∫ x · φ_t dt = -∫ f(x) · φ dt

where φ is a smooth compactly-supported test function and φ_t is its
time derivative (known analytically).

Reference:
    Messenger & Bortz (2021) "Weak SINDy: Galerkin-Based Data-Driven
    Model Selection"
"""

from __future__ import annotations

from typing import Self, TYPE_CHECKING

import jax.numpy as jnp

from opifex.discovery.sindy.library import CandidateLibrary
from opifex.discovery.sindy.optimizers import STLSQ


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opifex.discovery.sindy.config import WeakSINDyConfig


def _bump_function(s: jnp.ndarray, order: int = 4) -> jnp.ndarray:
    """Evaluate the bump test function (1 - s²)^p on [-1, 1].

    Args:
        s: Evaluation points in [-1, 1].
        order: Polynomial exponent p.

    Returns:
        Test function values (zero outside [-1, 1]).
    """
    return jnp.where(jnp.abs(s) <= 1.0, (1.0 - s**2) ** order, 0.0)


def _bump_derivative(s: jnp.ndarray, order: int = 4) -> jnp.ndarray:
    """Evaluate the time derivative of the bump test function.

    d/ds (1 - s²)^p = -2p·s·(1 - s²)^(p-1)

    Args:
        s: Evaluation points in [-1, 1].
        order: Polynomial exponent p.

    Returns:
        Test function derivative values.
    """
    return jnp.where(
        jnp.abs(s) <= 1.0,
        -2.0 * order * s * (1.0 - s**2) ** (order - 1),
        0.0,
    )


class WeakSINDy:
    """Weak-form SINDy for noise-robust equation discovery.

    Instead of computing noisy pointwise derivatives, integrates the
    governing equations against smooth bump test functions over
    overlapping time subdomains. This makes the method robust to
    measurement noise up to high SNR levels.

    Usage::

        config = WeakSINDyConfig(polynomial_degree=2, n_subdomains=50)
        model = WeakSINDy(config)
        model.fit(x_data, t)
        print(model.equations(["x", "y"]))

    Attributes:
        config: WeakSINDy configuration.
        coefficients: Fitted sparse coefficient matrix.
    """

    def __init__(self, config: WeakSINDyConfig) -> None:
        """Initialize WeakSINDy.

        Args:
            config: Configuration with polynomial degree, threshold,
                number of subdomains, and test function order.
        """
        self.config = config
        self.library = CandidateLibrary(
            polynomial_degree=config.polynomial_degree,
            include_trig=config.include_trig,
            n_frequencies=config.n_frequencies,
        )
        self.coefficients: jnp.ndarray | None = None
        self._feature_names_cache: list[str] | None = None

    def fit(self, x: jnp.ndarray, t: jnp.ndarray) -> Self:
        """Fit WeakSINDy model to time-series data.

        Args:
            x: State data, shape (n_samples, n_features).
            t: Time array, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        n_samples, n_features = x.shape
        n_subdomains = self.config.n_subdomains
        test_order = self.config.test_function_order

        # Build candidate library on full data
        theta_full = self.library.transform(x)
        n_lib = theta_full.shape[1]

        # Create overlapping subdomains
        subdomain_size = max(n_samples // n_subdomains, 10)
        stride = max((n_samples - subdomain_size) // max(n_subdomains - 1, 1), 1)

        # Weak-form matrices: one row per subdomain
        lhs_rows = []  # ∫ x · φ_t dt (left-hand side)
        rhs_rows = []  # ∫ Θ(x) · φ dt (right-hand side)

        for k in range(n_subdomains):
            start = min(k * stride, n_samples - subdomain_size)
            end = start + subdomain_size

            t_sub = t[start:end]
            x_sub = x[start:end]
            theta_sub = theta_full[start:end]

            # Map subdomain time to [-1, 1]
            t_center = (t_sub[0] + t_sub[-1]) / 2.0
            t_half = (t_sub[-1] - t_sub[0]) / 2.0
            if t_half < 1e-10:
                continue
            s = (t_sub - t_center) / t_half

            dt_sub = t_sub[1] - t_sub[0]

            # Evaluate test function and its derivative
            phi = _bump_function(s, order=test_order)
            phi_t = _bump_derivative(s, order=test_order) / t_half  # chain rule

            # LHS: ∫ x · φ_t dt  (integration by parts: = -∫ dx/dt · φ dt)
            # We compute ∫ x · φ_t dt directly (no derivatives needed!)
            lhs_row = jnp.sum(x_sub * phi_t[:, None], axis=0) * dt_sub

            # RHS: ∫ Θ(x) · φ dt
            rhs_row = jnp.sum(theta_sub * phi[:, None], axis=0) * dt_sub

            lhs_rows.append(lhs_row)
            rhs_rows.append(rhs_row)

        if not lhs_rows:
            self.coefficients = jnp.zeros((n_features, n_lib))
            return self

        # Assemble weak-form system: LHS = RHS @ Xi^T
        # Each row corresponds to one subdomain integral
        lhs = jnp.stack(lhs_rows)  # (n_subdomains, n_features)
        rhs = jnp.stack(rhs_rows)  # (n_subdomains, n_lib)

        # Solve sparse regression: lhs ≈ rhs @ Xi^T
        optimizer = STLSQ(
            threshold=self.config.threshold,
            alpha=self.config.alpha,
            max_iter=self.config.max_iter,
        )
        self.coefficients = optimizer.fit(rhs, lhs)

        return self

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Predict derivatives using discovered model.

        Args:
            x: State data, shape (n_samples, n_features).

        Returns:
            Predicted derivatives, shape (n_samples, n_features).
        """
        if self.coefficients is None:
            raise RuntimeError("Model has not been fit.")
        theta = self.library.transform(x)
        return theta @ self.coefficients.T

    def equations(
        self,
        input_names: Sequence[str] | None = None,
        precision: int = 3,
    ) -> list[str]:
        """Get human-readable equation strings.

        Args:
            input_names: Names for state variables.
            precision: Decimal places for coefficient values.

        Returns:
            List of equation strings.
        """
        if self.coefficients is None:
            raise RuntimeError("Model has not been fit.")

        names = self.library.get_feature_names(input_names)
        n_targets = self.coefficients.shape[0]
        target_names = input_names or [f"x{i}" for i in range(n_targets)]

        eqs = []
        for tidx in range(n_targets):
            terms = []
            for lidx, name in enumerate(names):
                coef = float(self.coefficients[tidx, lidx])
                if abs(coef) > 1e-10:
                    terms.append(f"{coef:.{precision}f} {name}")
            rhs = " + ".join(terms) if terms else "0"
            eqs.append(f"d{target_names[tidx]}/dt = {rhs}")

        return eqs

    def score(self, x: jnp.ndarray, x_dot: jnp.ndarray) -> float:
        """Compute R² score against true derivatives.

        Args:
            x: State data.
            x_dot: True derivatives.

        Returns:
            R² coefficient of determination.
        """
        x_dot_pred = self.predict(x)
        ss_res = jnp.sum((x_dot - x_dot_pred) ** 2)
        ss_tot = jnp.sum((x_dot - jnp.mean(x_dot, axis=0)) ** 2)
        return float(1.0 - ss_res / ss_tot)


__all__ = ["WeakSINDy"]
