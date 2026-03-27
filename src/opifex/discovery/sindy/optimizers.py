"""Sparse regression optimizers for SINDy.

Implements STLSQ (Sequential Thresholded Least Squares) and SR3
(Sparse Relaxed Regularized Regression) in pure JAX.

References:
    - Brunton et al. (2016) "Discovering governing equations from data"
    - Zheng et al. (2019) "A unified framework for sparse relaxed
      regularized regression"
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True, slots=True, kw_only=True)
class STLSQConfig:
    """Configuration for STLSQ optimizer."""

    threshold: float = 0.1
    alpha: float = 0.05
    max_iter: int = 20


class STLSQ:
    """Sequential Thresholded Least Squares optimizer.

    The canonical SINDy optimizer. Alternates between ridge regression
    and hard thresholding until the support (set of nonzero coefficients)
    stabilizes.

    Algorithm:
        1. Initialize via ridge regression
        2. Zero out coefficients below threshold
        3. Re-fit on remaining (nonzero) features
        4. Repeat until convergence or max_iter
    """

    def __init__(
        self,
        threshold: float = 0.1,
        alpha: float = 0.05,
        max_iter: int = 20,
    ) -> None:
        """Initialize STLSQ optimizer.

        Args:
            threshold: Coefficients with absolute value below this are zeroed.
            alpha: L2 regularization parameter for ridge regression.
            max_iter: Maximum number of thresholding iterations.
        """
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Fit sparse coefficients via STLSQ.

        Args:
            x: Library feature matrix, shape (n_samples, n_features).
            y: Target derivatives, shape (n_samples, n_targets).

        Returns:
            Sparse coefficient matrix, shape (n_targets, n_features).
        """
        n_features = x.shape[1]
        n_targets = y.shape[1]

        # Initial ridge regression: (X^T X + alpha I)^{-1} X^T y
        xtx = x.T @ x + self.alpha * jnp.eye(n_features)
        xty = x.T @ y
        coef = jnp.linalg.solve(xtx, xty).T  # (n_targets, n_features)

        # Iterative thresholding
        for _ in range(self.max_iter):
            # Hard threshold
            mask = jnp.abs(coef) >= self.threshold
            coef_new = coef * mask

            # Check convergence (support unchanged)
            old_support = jnp.abs(coef) >= self.threshold
            if jnp.array_equal(mask, old_support) and _ > 0:
                break

            # Re-fit on active features (per target)
            for target_idx in range(n_targets):
                active = mask[target_idx]
                n_active = int(jnp.sum(active))
                if n_active == 0:
                    coef_new = coef_new.at[target_idx].set(jnp.zeros(n_features))
                    continue

                # Extract active columns and solve
                x_active = x[:, active]
                xtx_active = x_active.T @ x_active + self.alpha * jnp.eye(n_active)
                xty_active = x_active.T @ y[:, target_idx]
                coef_active = jnp.linalg.solve(xtx_active, xty_active)

                # Place back into full coefficient vector
                full_coef = jnp.zeros(n_features)
                active_indices = jnp.where(active, size=n_active)[0]
                full_coef = full_coef.at[active_indices].set(coef_active)
                coef_new = coef_new.at[target_idx].set(full_coef)

            coef = coef_new

        # Final threshold
        return coef * (jnp.abs(coef) >= self.threshold)


class SR3:
    """Sparse Relaxed Regularized Regression optimizer.

    Minimizes: 0.5||y - Xw||² + lambda * R(u) + (0.5/nu) * ||w - u||²

    where R is a regularization penalty (L0, L1, or L2) and w, u are
    alternately optimized.

    References:
        Zheng et al. (2019) "A unified framework for sparse relaxed
        regularized regression"
    """

    def __init__(
        self,
        threshold: float = 0.1,
        nu: float = 1.0,
        max_iter: int = 30,
        tol: float = 1e-5,
        regularization: str = "l0",
    ) -> None:
        """Initialize SR3 optimizer.

        Args:
            threshold: Sparsity threshold (lambda in the SR3 formulation).
            nu: Relaxation parameter controlling w-u coupling.
            max_iter: Maximum number of alternating iterations.
            tol: Convergence tolerance.
            regularization: Regularization type ('l0', 'l1', or 'l2').
        """
        self.threshold = threshold
        self.nu = nu
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization

    def fit(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Fit sparse coefficients via SR3.

        Args:
            x: Library feature matrix, shape (n_samples, n_features).
            y: Target derivatives, shape (n_samples, n_targets).

        Returns:
            Sparse coefficient matrix, shape (n_targets, n_features).
        """
        n_features = x.shape[1]

        # Precompute Cholesky factor: (X^T X + (1/nu) I)
        xtx = x.T @ x + (1.0 / self.nu) * jnp.eye(n_features)
        cho = jnp.linalg.cholesky(xtx)
        xty = x.T @ y  # (n_features, n_targets)

        # Initialize
        w = jnp.linalg.solve(xtx, xty)  # (n_features, n_targets)
        u = w.copy()

        for _ in range(self.max_iter):
            u_old = u

            # w-update: solve (X^T X + (1/nu)I) w = X^T y + u/nu
            rhs = xty + u / self.nu
            w = jax_cho_solve(cho, rhs)

            # u-update: proximal operator
            u = self._prox(w, self.threshold * self.nu)

            # Check convergence
            change = jnp.linalg.norm(u - u_old) / (jnp.linalg.norm(u_old) + 1e-10)
            if change < self.tol:
                break

        return u.T  # (n_targets, n_features)

    def _prox(self, w: jnp.ndarray, lam: float) -> jnp.ndarray:
        """Apply proximal operator based on regularization type."""
        if self.regularization == "l0":
            # Hard thresholding
            return w * (jnp.abs(w) > jnp.sqrt(2 * lam))
        if self.regularization == "l1":
            # Soft thresholding
            return jnp.sign(w) * jnp.maximum(jnp.abs(w) - lam, 0.0)
        if self.regularization == "l2":
            # Shrinkage
            return w / (1.0 + 2.0 * lam)
        raise ValueError(f"Unknown regularization: {self.regularization}")


def jax_cho_solve(cho: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Solve Cholesky system L L^T x = b.

    Args:
        cho: Lower Cholesky factor L.
        b: Right-hand side.

    Returns:
        Solution x.
    """
    y = jnp.linalg.solve(cho, b)
    return jnp.linalg.solve(cho.T, y)


__all__ = ["SR3", "STLSQ"]
