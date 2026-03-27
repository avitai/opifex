"""Configuration dataclasses for SINDy equation discovery."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class SINDyConfig:
    """Configuration for the SINDy sparse identification algorithm.

    Attributes:
        polynomial_degree: Maximum polynomial degree for candidate library.
        threshold: Sparsity threshold for STLSQ optimizer.
        alpha: L2 regularization strength for ridge regression.
        max_iter: Maximum STLSQ iterations.
        include_trig: Include trigonometric basis functions.
        n_frequencies: Number of Fourier frequencies (if trig enabled).
        optimizer: Optimizer name ('stlsq' or 'sr3').
    """

    polynomial_degree: int = 2
    threshold: float = 0.1
    alpha: float = 0.05
    max_iter: int = 20
    include_trig: bool = False
    n_frequencies: int = 1
    optimizer: str = "stlsq"


@dataclass(frozen=True, slots=True, kw_only=True)
class WeakSINDyConfig(SINDyConfig):
    """Configuration for weak-form SINDy (noise-robust variant).

    Attributes:
        n_subdomains: Number of integration subdomains.
        test_function_order: Order of the polynomial test function.
    """

    n_subdomains: int = 100
    test_function_order: int = 4


@dataclass(frozen=True, slots=True, kw_only=True)
class EnsembleSINDyConfig(SINDyConfig):
    """Configuration for ensemble SINDy.

    Attributes:
        n_models: Number of models in the ensemble.
        bagging_fraction: Fraction of data for each bootstrap sample.
        library_dropout: Fraction of library terms to drop per model.
    """

    n_models: int = 20
    bagging_fraction: float = 0.8
    library_dropout: float = 0.0


__all__ = ["EnsembleSINDyConfig", "SINDyConfig", "WeakSINDyConfig"]
