"""Candidate function library for SINDy sparse identification.

Generates the Theta(x) matrix of candidate nonlinear functions evaluated
on data. Supports polynomial, trigonometric, and custom basis functions.

All operations are JAX-native and JIT-compatible.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class CandidateLibrary:
    """Generates candidate function library for SINDy.

    Constructs a feature matrix Theta(x) where each column is a candidate
    nonlinear function evaluated on the data. The SINDy algorithm then
    finds the sparse subset of columns that best predicts the derivatives.

    Attributes:
        polynomial_degree: Maximum polynomial degree.
        include_trig: Whether to include sin/cos basis functions.
        n_frequencies: Number of Fourier frequencies.
        custom_functions: Optional list of custom basis functions.
    """

    def __init__(
        self,
        polynomial_degree: int = 2,
        include_trig: bool = False,
        n_frequencies: int = 1,
        custom_functions: Sequence[Callable] | None = None,
    ) -> None:
        """Initialize candidate library.

        Args:
            polynomial_degree: Maximum degree of polynomial terms.
            include_trig: Include sin/cos of each feature.
            n_frequencies: Number of Fourier frequencies (if trig enabled).
            custom_functions: List of callables f(x) -> array, where x is
                (n_samples, n_features) and output is (n_samples, 1) or (n_samples,).
        """
        self.polynomial_degree = polynomial_degree
        self.include_trig = include_trig
        self.n_frequencies = n_frequencies
        self.custom_functions = list(custom_functions) if custom_functions else []
        self._powers: list[tuple[int, ...]] | None = None
        self._n_input_features: int | None = None

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Build the candidate library matrix Theta(x).

        Args:
            x: Data matrix of shape (n_samples, n_features).

        Returns:
            Library matrix Theta of shape (n_samples, n_library_terms).
        """
        n_samples, n_features = x.shape
        self._n_input_features = n_features
        columns = []

        # Polynomial terms (including constant)
        powers = self._compute_powers(n_features, self.polynomial_degree)
        self._powers = powers
        for power in powers:
            term = jnp.ones(n_samples)
            for feat_idx, exp in enumerate(power):
                if exp > 0:
                    term = term * x[:, feat_idx] ** exp
            columns.append(term)

        # Trigonometric terms
        if self.include_trig:
            for freq in range(1, self.n_frequencies + 1):
                for feat_idx in range(n_features):
                    columns.append(jnp.sin(freq * x[:, feat_idx]))
                    columns.append(jnp.cos(freq * x[:, feat_idx]))

        # Custom functions
        for func in self.custom_functions:
            result = func(x)
            if result.ndim == 1:
                columns.append(result)
            else:
                columns.append(result.squeeze(-1))

        return jnp.column_stack(columns)

    def get_feature_names(self, input_names: Sequence[str] | None = None) -> list[str]:
        """Get human-readable names for library terms.

        Args:
            input_names: Names for input features (e.g., ['x', 'y', 'z']).

        Returns:
            List of feature name strings.
        """
        if self._n_input_features is None:
            raise RuntimeError("Call transform() before get_feature_names()")

        n_features = self._n_input_features
        if input_names is None:
            input_names = [f"x{i}" for i in range(n_features)]

        names: list[str] = []
        self._append_polynomial_names(names, input_names)
        self._append_trig_names(names, input_names, n_features)
        self._append_custom_names(names)
        return names

    def _append_polynomial_names(self, names: list[str], input_names: Sequence[str]) -> None:
        """Append polynomial term names to the list."""
        if self._powers is None:
            return
        for power in self._powers:
            if all(p == 0 for p in power):
                names.append("1")
            else:
                parts = []
                for feat_idx, exp in enumerate(power):
                    if exp == 1:
                        parts.append(input_names[feat_idx])
                    elif exp > 1:
                        parts.append(f"{input_names[feat_idx]}^{exp}")
                names.append(" ".join(parts) if parts else "1")

    def _append_trig_names(
        self, names: list[str], input_names: Sequence[str], n_features: int
    ) -> None:
        """Append trigonometric term names to the list."""
        if not self.include_trig:
            return
        for freq in range(1, self.n_frequencies + 1):
            for feat_idx in range(n_features):
                prefix = f"{freq}*" if freq > 1 else ""
                names.append(f"sin({prefix}{input_names[feat_idx]})")
                names.append(f"cos({prefix}{input_names[feat_idx]})")

    def _append_custom_names(self, names: list[str]) -> None:
        """Append custom function names to the list."""
        for i, func in enumerate(self.custom_functions):
            fname = getattr(func, "__name__", f"f{i}")
            names.append(fname)

    @staticmethod
    def _compute_powers(n_features: int, degree: int) -> list[tuple[int, ...]]:
        """Compute all monomial power tuples up to given degree.

        For n_features=2, degree=2:
        [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]

        Returns:
            List of tuples where each tuple is a power vector.
        """
        powers = []
        for d in range(degree + 1):
            for combo in combinations_with_replacement(range(n_features), d):
                power = [0] * n_features
                for idx in combo:
                    power[idx] += 1
                powers.append(tuple(power))
        return powers


__all__ = ["CandidateLibrary"]
