r"""BQ acquisition functions + experimental design loop driver.

Tests for :mod:`opifex.uncertainty.quadrature.acquisitions`. The five
acquisitions and the driver loop are line-by-line ports of the emukit
references at ``../emukit/emukit/quadrature/acquisitions/`` and
``../emukit/emukit/experimental_design/``:

* :func:`uncertainty_sampling` —
  ``emukit/quadrature/acquisitions/uncertainty_sampling.py``.
  ``a(x) = var(f(x)) · p(x)^q``.
* :func:`model_variance` —
  ``emukit/experimental_design/acquisitions/model_variance.py``.
  ``a(x) = var(f(x))`` (the raw GP posterior variance).
* :func:`integral_variance_reduction` —
  ``emukit/quadrature/acquisitions/squared_correlation.py``.
  ``a(x) = predictive_cov² / (integral_var · y_predictive_var)`` ∈ [0, 1].
* :func:`mutual_information` —
  ``emukit/quadrature/acquisitions/mutual_information.py``.
  ``a(x) = -0.5 log(1 - ρ²)`` where ``ρ²`` is the squared correlation.
* :func:`integrated_variance_reduction` —
  ``emukit/experimental_design/acquisitions/integrated_variance.py``.
  Monte-Carlo estimate of expected variance reduction over the
  integration measure.
* :func:`experimental_design_loop` — driver loop adding the
  acquisition-maximising candidate at each step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.quadrature.acquisitions import (
    experimental_design_loop,
    integral_variance_reduction,
    integrated_variance_reduction,
    model_variance,
    mutual_information,
    uncertainty_sampling,
)


def _rbf_kernel(x_left: jax.Array, x_right: jax.Array, lengthscale: float = 1.0) -> jax.Array:
    """RBF kernel matrix with amplitude 1."""
    squared_diff = jnp.sum(
        (x_left[:, None, :] - x_right[None, :, :]) ** 2 / (lengthscale**2), axis=-1
    )
    return jnp.exp(-0.5 * squared_diff)


def _rbf_kernel_mean(points: jax.Array) -> jax.Array:
    r"""``qK(x') = √(1/2) · exp(-x'²/4)`` for unit RBF + standard Gaussian."""
    factor = jnp.sqrt(1.0 / 2.0)
    return factor * jnp.exp(-0.5 * jnp.sum(points**2 / 2.0, axis=-1))


def _standard_gaussian_density(points: jax.Array) -> jax.Array:
    """1-D standard normal density at each row of ``points``."""
    return jnp.exp(-0.5 * jnp.sum(points**2, axis=-1)) / jnp.sqrt(2.0 * jnp.pi)


# ---------------------------------------------------------------------------
# uncertainty_sampling
# ---------------------------------------------------------------------------


def test_uncertainty_sampling_favours_high_variance_points() -> None:
    """``a(x) = var(x) · p(x)^q`` is larger where the posterior variance is high.

    A point far from the training set has higher GP posterior variance
    than a point at the training set (in a noise-free GP). So the
    acquisition value at a far candidate should exceed the value at
    a near-training candidate when the measure densities are comparable.
    """
    train_points = jnp.array([[0.0]])
    candidates = jnp.array([[0.0], [2.5]])
    values = uncertainty_sampling(
        points=candidates,
        train_points=train_points,
        kernel_fn=_rbf_kernel,
        noise_variance=jnp.asarray(1e-6),
        measure_density_fn=_standard_gaussian_density,
        measure_power=jnp.asarray(0.0),
    )
    assert values.shape == (2,)
    assert values[1] > values[0]


def test_uncertainty_sampling_weights_by_measure_density() -> None:
    """With ``measure_power > 0`` the acquisition decays away from the measure mean."""
    train_points = jnp.array([[0.0]])
    candidates = jnp.array([[0.5], [3.5]])
    values = uncertainty_sampling(
        points=candidates,
        train_points=train_points,
        kernel_fn=_rbf_kernel,
        noise_variance=jnp.asarray(1e-6),
        measure_density_fn=_standard_gaussian_density,
        measure_power=jnp.asarray(2.0),
    )
    # The 3.5 candidate is far from the measure mean (density ≈ 0), so its
    # acquisition is suppressed despite the high variance.
    assert values[0] > values[1]


# ---------------------------------------------------------------------------
# model_variance
# ---------------------------------------------------------------------------


def test_model_variance_returns_predictive_variance_at_each_point() -> None:
    """``a(x) = predictive_var(x)`` with no measure weighting."""
    train_points = jnp.array([[0.0]])
    candidates = jnp.array([[0.0], [2.5]])
    values = model_variance(
        points=candidates,
        train_points=train_points,
        kernel_fn=_rbf_kernel,
        noise_variance=jnp.asarray(1e-6),
    )
    # A point at the training location has near-zero variance; a far point
    # has variance near the prior amplitude (1).
    assert values[0] < 0.01
    assert values[1] > 0.5


# ---------------------------------------------------------------------------
# integral_variance_reduction
# ---------------------------------------------------------------------------


def test_integral_variance_reduction_lies_in_unit_interval() -> None:
    """Squared correlation is a probability-like score ∈ [0, 1]."""
    train_points = jnp.array([[-1.0], [1.0]])
    candidates = jnp.array([[0.0], [-2.0], [2.0]])
    qkq = jnp.asarray(1.0 / jnp.sqrt(3.0))

    values = integral_variance_reduction(
        points=candidates,
        train_points=train_points,
        kernel_fn=_rbf_kernel,
        kernel_mean_fn=_rbf_kernel_mean,
        double_kernel_mean=qkq,
        noise_variance=jnp.asarray(1e-6),
    )
    assert values.shape == (3,)
    assert jnp.all(values >= 0.0)
    assert jnp.all(values <= 1.0)


# ---------------------------------------------------------------------------
# mutual_information
# ---------------------------------------------------------------------------


def test_mutual_information_equals_neg_half_log_one_minus_squared_correlation() -> None:
    r"""``MI(x) = -0.5 log(1 - ρ²(x))``."""
    train_points = jnp.array([[-1.0], [1.0]])
    candidates = jnp.array([[0.5], [-0.5]])
    qkq = jnp.asarray(1.0 / jnp.sqrt(3.0))
    common_kwargs = {
        "points": candidates,
        "train_points": train_points,
        "kernel_fn": _rbf_kernel,
        "kernel_mean_fn": _rbf_kernel_mean,
        "double_kernel_mean": qkq,
        "noise_variance": jnp.asarray(1e-6),
    }
    squared_correlation = integral_variance_reduction(**common_kwargs)
    mi = mutual_information(**common_kwargs)
    expected = -0.5 * jnp.log(1.0 - squared_correlation)
    assert jnp.allclose(mi, expected, atol=1e-6)


def test_mutual_information_is_non_negative() -> None:
    """``MI(x) >= 0`` because ``ρ² ∈ [0, 1)``."""
    train_points = jnp.array([[-1.0], [1.0]])
    candidates = jnp.array([[0.0], [-0.5], [0.5]])
    qkq = jnp.asarray(1.0 / jnp.sqrt(3.0))
    mi = mutual_information(
        points=candidates,
        train_points=train_points,
        kernel_fn=_rbf_kernel,
        kernel_mean_fn=_rbf_kernel_mean,
        double_kernel_mean=qkq,
        noise_variance=jnp.asarray(1e-6),
    )
    assert jnp.all(mi >= 0.0)


# ---------------------------------------------------------------------------
# integrated_variance_reduction
# ---------------------------------------------------------------------------


def test_integrated_variance_reduction_is_non_negative() -> None:
    """Adding a point reduces variance — the expected reduction is ``>= 0``."""
    train_points = jnp.array([[-1.0], [1.0]])
    candidates = jnp.array([[0.0], [2.0]])
    key = jax.random.PRNGKey(0)
    monte_carlo_points = jax.random.normal(key, (256, 1))

    values = integrated_variance_reduction(
        points=candidates,
        train_points=train_points,
        kernel_fn=_rbf_kernel,
        noise_variance=jnp.asarray(1e-6),
        monte_carlo_points=monte_carlo_points,
    )
    assert values.shape == (2,)
    assert jnp.all(values >= -1e-6)


def test_integrated_variance_reduction_prefers_central_candidate_under_gaussian_mc() -> None:
    """A candidate near the centre of the MC distribution drives the largest reduction."""
    train_points = jnp.array([[-2.0], [2.0]])
    candidates = jnp.array([[0.0], [4.0]])
    key = jax.random.PRNGKey(1)
    monte_carlo_points = jax.random.normal(key, (512, 1))

    values = integrated_variance_reduction(
        points=candidates,
        train_points=train_points,
        kernel_fn=_rbf_kernel,
        noise_variance=jnp.asarray(1e-6),
        monte_carlo_points=monte_carlo_points,
    )
    assert values[0] > values[1]


# ---------------------------------------------------------------------------
# experimental_design_loop
# ---------------------------------------------------------------------------


def test_experimental_design_loop_grows_training_set_by_num_iterations() -> None:
    """The driver appends one candidate per iteration."""
    initial_points = jnp.array([[-1.0], [1.0]])
    initial_values = jnp.array([0.0, 0.0])

    def target(x: jax.Array) -> jax.Array:
        return jnp.sin(x[0])

    candidate_pool = jnp.linspace(-2.0, 2.0, 16).reshape(-1, 1)

    def acquisition(candidate_points: jax.Array, train_points: jax.Array) -> jax.Array:
        return model_variance(
            points=candidate_points,
            train_points=train_points,
            kernel_fn=_rbf_kernel,
            noise_variance=jnp.asarray(1e-6),
        )

    final_points, final_values = experimental_design_loop(
        initial_points=initial_points,
        initial_values=initial_values,
        target_fn=target,
        acquisition_fn=acquisition,
        candidate_pool=candidate_pool,
        num_iterations=4,
    )
    assert final_points.shape == (6, 1)
    assert final_values.shape == (6,)


def test_experimental_design_loop_first_pick_maximises_acquisition() -> None:
    """At iteration zero the first new point maximises the acquisition over candidates."""
    initial_points = jnp.array([[0.0]])
    initial_values = jnp.array([0.0])

    def target(_: jax.Array) -> jax.Array:
        return jnp.asarray(0.0)

    candidate_pool = jnp.linspace(-3.0, 3.0, 13).reshape(-1, 1)

    def acquisition(candidate_points: jax.Array, train_points: jax.Array) -> jax.Array:
        return model_variance(
            points=candidate_points,
            train_points=train_points,
            kernel_fn=_rbf_kernel,
            noise_variance=jnp.asarray(1e-6),
        )

    initial_acquisition = acquisition(candidate_pool, initial_points)
    expected_first_pick = candidate_pool[jnp.argmax(initial_acquisition)]

    final_points, _ = experimental_design_loop(
        initial_points=initial_points,
        initial_values=initial_values,
        target_fn=target,
        acquisition_fn=acquisition,
        candidate_pool=candidate_pool,
        num_iterations=1,
    )
    assert jnp.allclose(final_points[-1], expected_first_pick)
