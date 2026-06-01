"""Tests for the Bayesian-SINDy equation-discovery UQ surface (Phase-13 F4).

The Step-10 stubs are replaced by a real regularized-horseshoe Bayesian
SINDy ported from pysindy's ``SBR`` optimizer
(``/mnt/ssd2/Works/pysindy/pysindy/optimizers/sbr.py``) onto a BlackJAX
NUTS sampler. These tests pin the recovery behaviour, the inclusion-probability
contract, log-density jittability, and reproducibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.discovery.sindy.library import CandidateLibrary
from opifex.uncertainty.scientific.equation_discovery import (
    BayesianSINDy,
    PosteriorOverTerms,
)
from opifex.uncertainty.types import PredictionInterval


# Bayesian SINDy fits via blackjax NUTS (full MCMC warmup + sampling); the
# fitting path dominates this file's runtime (~5 min). Mark the module slow so
# fast runs can deselect it with ``-m "not slow"``.
pytestmark = pytest.mark.slow


def _decay_dataset() -> tuple[jax.Array, jax.Array]:
    """Sparse linear system ``x_dot = -2 x`` over a polynomial(deg 2) library.

    With the library ``[1, x0, x0^2]`` only the ``x0`` column is active with
    coefficient ``-2``; ``1`` and ``x0^2`` are inactive.
    """
    x = jnp.linspace(-2.0, 2.0, 60).reshape(-1, 1)
    x_dot = -2.0 * x
    return x, x_dot


def _two_term_dataset() -> tuple[jax.Array, jax.Array]:
    """Two-term system ``x_dot = 1.5 - 0.8 x`` over a polynomial(deg 2) library.

    Active columns: ``1`` (coef ``1.5``) and ``x0`` (coef ``-0.8``); ``x0^2``
    inactive.
    """
    x = jnp.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    x_dot = 1.5 - 0.8 * x
    return x, x_dot


def _fit_decay() -> tuple[BayesianSINDy, PosteriorOverTerms, CandidateLibrary]:
    x, x_dot = _decay_dataset()
    library = CandidateLibrary(polynomial_degree=2)
    model = BayesianSINDy(library, num_warmup=200, num_samples=400)
    posterior = model.fit(x, x_dot, rngs=nnx.Rngs(0))
    return model, posterior, library


# --------------------------------------------------------------------------- #
# Recovery (key correctness check)
# --------------------------------------------------------------------------- #
def test_recovers_sparse_linear_decay_system() -> None:
    """``x_dot = -2 x``: ``x`` highly included, ``1`` / ``x^2`` excluded."""
    model, posterior, _ = _fit_decay()

    inclusion = model.term_inclusion_probabilities()
    assert inclusion["x0"] > 0.8
    assert inclusion["1"] < 0.3
    assert inclusion["x0^2"] < 0.3

    # Posterior-mean coefficient for the active term near the truth (-2).
    coef = model.coefficients()  # (n_targets, n_terms) -> (1, 3)
    names = posterior.feature_names
    x_idx = names.index("x0")
    assert abs(float(coef[0, x_idx]) - (-2.0)) < 0.3


def test_recovery_credible_interval_contains_truth() -> None:
    """95% credible interval for the active term brackets the true ``-2``."""
    model, posterior, _ = _fit_decay()
    intervals = model.coefficient_posterior_intervals(level=0.95)
    x_idx = posterior.feature_names.index("x0")
    lower = float(intervals.lower[0, x_idx])
    upper = float(intervals.upper[0, x_idx])
    assert lower <= -2.0 <= upper
    assert intervals.coverage == pytest.approx(0.95)


def test_recovers_two_term_affine_system() -> None:
    """``x_dot = 1.5 - 0.8 x``: both ``1`` and ``x`` included, ``x^2`` not."""
    x, x_dot = _two_term_dataset()
    library = CandidateLibrary(polynomial_degree=2)
    model = BayesianSINDy(library, num_warmup=200, num_samples=400)
    posterior = model.fit(x, x_dot, rngs=nnx.Rngs(1))

    inclusion = model.term_inclusion_probabilities()
    assert inclusion["1"] > 0.8
    assert inclusion["x0"] > 0.8
    assert inclusion["x0^2"] < 0.3

    coef = model.coefficients()
    names = posterior.feature_names
    assert abs(float(coef[0, names.index("1")]) - 1.5) < 0.4
    assert abs(float(coef[0, names.index("x0")]) - (-0.8)) < 0.4


# --------------------------------------------------------------------------- #
# Inclusion-probability contract
# --------------------------------------------------------------------------- #
def test_inclusion_probabilities_in_unit_interval_and_named() -> None:
    model, posterior, _ = _fit_decay()
    inclusion = model.term_inclusion_probabilities()

    assert set(inclusion) == set(posterior.feature_names)
    for value in inclusion.values():
        assert 0.0 <= value <= 1.0


def test_posterior_shapes_and_feature_names() -> None:
    _model, posterior, _ = _fit_decay()
    assert posterior.beta.shape == (400, 1, 3)
    assert posterior.feature_names == ("1", "x0", "x0^2")
    assert posterior.tau.shape == (400,)
    assert posterior.sigma.shape == (400,)


def test_coefficient_posterior_intervals_shape_and_ordering() -> None:
    model, _, _ = _fit_decay()
    intervals = model.coefficient_posterior_intervals(level=0.9)
    assert intervals.lower.shape == (1, 3)
    assert intervals.upper.shape == (1, 3)
    assert bool(jnp.all(intervals.lower <= intervals.upper))


def test_methods_require_fit_first() -> None:
    library = CandidateLibrary(polynomial_degree=2)
    model = BayesianSINDy(library)
    with pytest.raises(RuntimeError, match="fit"):
        model.coefficients()
    with pytest.raises(RuntimeError, match="fit"):
        model.term_inclusion_probabilities()
    with pytest.raises(RuntimeError, match="fit"):
        model.coefficient_posterior_intervals()


# --------------------------------------------------------------------------- #
# Constructor validation
# --------------------------------------------------------------------------- #
def test_constructor_validates_positive_hyperparameters() -> None:
    library = CandidateLibrary(polynomial_degree=1)
    with pytest.raises(ValueError, match="tau0"):
        BayesianSINDy(library, tau0=0.0)
    with pytest.raises(ValueError, match="slab_nu"):
        BayesianSINDy(library, slab_nu=0.0)
    with pytest.raises(ValueError, match="slab_s"):
        BayesianSINDy(library, slab_s=-1.0)
    with pytest.raises(ValueError, match="noise_lambda"):
        BayesianSINDy(library, noise_lambda=0.0)


def test_constructor_validates_sample_counts() -> None:
    library = CandidateLibrary(polynomial_degree=1)
    with pytest.raises(ValueError, match="num_warmup"):
        BayesianSINDy(library, num_warmup=-1)
    with pytest.raises(ValueError, match="num_samples"):
        BayesianSINDy(library, num_samples=0)


# --------------------------------------------------------------------------- #
# JAX-transform compatibility + reproducibility
# --------------------------------------------------------------------------- #
def test_log_density_is_jittable() -> None:
    """The joint unconstrained log-density compiles under ``jax.jit``."""
    x, x_dot = _decay_dataset()
    library = CandidateLibrary(polynomial_degree=2)
    model = BayesianSINDy(library)
    log_density, init_position = model.build_log_density(x, x_dot)

    value = jax.jit(log_density)(init_position)
    assert value.shape == ()
    assert jnp.isfinite(value)

    # Gradient transform must also succeed (NUTS needs it).
    grads = jax.jit(jax.grad(log_density))(init_position)
    assert jnp.isfinite(grads["beta"]).all()


def test_fit_is_reproducible_with_same_rngs() -> None:
    """Identical seeds reproduce identical posterior draws bit-for-bit."""
    x, x_dot = _decay_dataset()
    library_a = CandidateLibrary(polynomial_degree=2)
    library_b = CandidateLibrary(polynomial_degree=2)
    model_a = BayesianSINDy(library_a, num_warmup=50, num_samples=80)
    model_b = BayesianSINDy(library_b, num_warmup=50, num_samples=80)

    posterior_a = model_a.fit(x, x_dot, rngs=nnx.Rngs(7))
    posterior_b = model_b.fit(x, x_dot, rngs=nnx.Rngs(7))

    assert bool(jnp.array_equal(posterior_a.beta, posterior_b.beta))


def test_module_exposes_public_surface() -> None:
    import opifex.uncertainty.scientific.equation_discovery as mod

    assert mod.__all__ == ["BayesianSINDy", "PosteriorOverTerms"]
    assert issubclass(type(PredictionInterval), type)
