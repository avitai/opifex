"""JAX-native ports of specialised probabilistic-numerics algorithms.

Tests for :mod:`opifex.uncertainty.scientific._specialised`. The module
implements three algorithmic primitives referenced by the
probabilistic-numerics catalogue:

* :func:`manifold_update` — IEKF update enforcing ``g(x) = 0`` via
  ``jax.jacrev``. Sibling reference
  ``ProbNumDiffEq.jl/src/callbacks/manifoldupdate.jl``.
* :func:`dense_output_sample` — Single-draw Gaussian sampler used for
  joint posterior sampling at arbitrary density. Sibling reference
  ``ProbNumDiffEq.jl/src/solution_sampling.jl`` (``_rand``).
* :func:`apply_diffusion` — Scalar / vector diffusion scaling applied
  to a PSD process-noise covariance. Sibling reference
  ``ProbNumDiffEq.jl/src/diffusions/apply_diffusion.jl``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from opifex.uncertainty.registry import UQCapability
from opifex.uncertainty.scientific._specialised import (
    apply_diffusion,
    dense_output_sample,
    manifold_update,
)
from opifex.uncertainty.scientific.probabilistic_numerics import (
    ApplyDiffusionSpec,
    DenseOutputSamplingSpec,
    ManifoldUpdateSpec,
)


# ---------------------------------------------------------------------------
# manifold_update
# ---------------------------------------------------------------------------


def test_manifold_update_linear_residual_matches_standard_kalman_update() -> None:
    """For a linear residual ``A u + b``, IEKF reduces to a single EKF step.

    The manifold constraint ``A u + b = 0`` is the linear observation
    ``H x = -b`` with ``H = A @ observation_matrix`` and zero noise.
    """
    state_dim, observation_dim, residual_dim = 4, 2, 1
    key = jax.random.PRNGKey(0)
    k_mean, k_cov, k_proj, k_a, k_b = jax.random.split(key, 5)

    mean = jax.random.normal(k_mean, (state_dim,))
    cov_factor = jax.random.normal(k_cov, (state_dim, state_dim))
    cov = cov_factor @ cov_factor.T + jnp.eye(state_dim)
    observation_matrix = jax.random.normal(k_proj, (observation_dim, state_dim))
    a_matrix = jax.random.normal(k_a, (residual_dim, observation_dim))
    b_vector = jax.random.normal(k_b, (residual_dim,))

    def residual_fn(observation: jax.Array) -> jax.Array:
        return a_matrix @ observation + b_vector

    manifold_mean, manifold_cov = manifold_update(
        mean=mean,
        cov=cov,
        residual_fn=residual_fn,
        observation_matrix=observation_matrix,
        max_iters=1,
    )

    h_matrix = a_matrix @ observation_matrix
    innovation_cov = h_matrix @ cov @ h_matrix.T
    gain = jnp.linalg.solve(innovation_cov, h_matrix @ cov).T
    expected_mean = mean - gain @ (h_matrix @ mean + b_vector)
    expected_cov = cov - gain @ h_matrix @ cov

    assert jnp.allclose(manifold_mean, expected_mean, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(manifold_cov, expected_cov, atol=1e-5, rtol=1e-5)


def test_manifold_update_converges_on_a_satisfying_state() -> None:
    """If the state already satisfies the constraint, the IEKF is a no-op."""
    state_dim = 3
    cov = jnp.eye(state_dim) * 0.5
    mean = jnp.array([1.0, 0.0, 0.0])
    observation_matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def residual_fn(observation: jax.Array) -> jax.Array:
        return observation - observation_matrix @ mean

    manifold_mean, _ = manifold_update(
        mean=mean,
        cov=cov,
        residual_fn=residual_fn,
        observation_matrix=observation_matrix,
        max_iters=5,
    )
    assert jnp.allclose(manifold_mean, mean, atol=1e-6)


def test_manifold_update_compiles_under_jit() -> None:
    """``manifold_update`` must compile under ``jax.jit``."""
    state_dim = 3
    mean = jnp.array([1.0, 2.0, 3.0])
    cov = jnp.eye(state_dim)
    observation_matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def residual_fn(observation: jax.Array) -> jax.Array:
        return observation - jnp.array([0.0, 0.0])

    jitted = jax.jit(manifold_update, static_argnames=("residual_fn", "max_iters"))
    new_mean, _ = jitted(
        mean=mean,
        cov=cov,
        residual_fn=residual_fn,
        observation_matrix=observation_matrix,
        max_iters=3,
    )
    assert jnp.all(jnp.isfinite(new_mean))


# ---------------------------------------------------------------------------
# dense_output_sample
# ---------------------------------------------------------------------------


def test_dense_output_sample_returns_mean_when_covariance_is_zero() -> None:
    """A zero-covariance Gaussian collapses to its mean."""
    mean = jnp.array([1.0, 2.0, 3.0])
    cov = jnp.zeros((3, 3))
    sample = dense_output_sample(mean=mean, cov=cov, key=jax.random.PRNGKey(0))
    assert jnp.allclose(sample, mean)


def test_dense_output_sample_is_deterministic_under_fixed_key() -> None:
    """Identical keys produce identical samples (reproducibility)."""
    mean = jnp.zeros(4)
    cov_factor = jnp.eye(4) + 0.1 * jnp.ones((4, 4))
    cov = cov_factor @ cov_factor.T
    key = jax.random.PRNGKey(42)
    sample_a = dense_output_sample(mean=mean, cov=cov, key=key)
    sample_b = dense_output_sample(mean=mean, cov=cov, key=key)
    assert jnp.array_equal(sample_a, sample_b)


def test_dense_output_sample_empirical_mean_approaches_posterior_mean() -> None:
    """Large-sample empirical mean recovers the posterior mean."""
    mean = jnp.array([0.5, -1.0])
    cov = jnp.array([[1.0, 0.3], [0.3, 0.5]])
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    samples = jax.vmap(lambda k: dense_output_sample(mean=mean, cov=cov, key=k))(keys)
    empirical_mean = samples.mean(axis=0)
    assert jnp.allclose(empirical_mean, mean, atol=0.05)


def test_dense_output_sample_empirical_covariance_approaches_posterior_covariance() -> None:
    """Large-sample empirical covariance recovers the posterior covariance."""
    mean = jnp.zeros(2)
    cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    keys = jax.random.split(jax.random.PRNGKey(1), 8192)
    samples = jax.vmap(lambda k: dense_output_sample(mean=mean, cov=cov, key=k))(keys)
    centred = samples - samples.mean(axis=0)
    empirical_cov = centred.T @ centred / samples.shape[0]
    assert jnp.allclose(empirical_cov, cov, atol=0.1)


def test_dense_output_sample_compiles_under_jit() -> None:
    """``dense_output_sample`` must compile under ``jax.jit``."""
    mean = jnp.zeros(3)
    cov = jnp.eye(3)
    sample = jax.jit(dense_output_sample)(mean=mean, cov=cov, key=jax.random.PRNGKey(0))
    assert sample.shape == (3,)


# ---------------------------------------------------------------------------
# apply_diffusion
# ---------------------------------------------------------------------------


def test_apply_diffusion_scalar_one_is_identity() -> None:
    """Scalar diffusion ``1.0`` leaves Q unchanged."""
    process_noise = jnp.array([[2.0, 0.1], [0.1, 3.0]])
    result = apply_diffusion(
        process_noise_cov=process_noise,
        diffusion=jnp.asarray(1.0),
        num_derivatives=1,
    )
    assert jnp.allclose(result, process_noise)


def test_apply_diffusion_scalar_scales_covariance_proportionally() -> None:
    """Scalar diffusion ``d`` scales Q by exactly ``d``."""
    process_noise = jnp.array([[2.0, 0.1], [0.1, 3.0]])
    result = apply_diffusion(
        process_noise_cov=process_noise,
        diffusion=jnp.asarray(4.0),
        num_derivatives=1,
    )
    assert jnp.allclose(result, 4.0 * process_noise)


def test_apply_diffusion_unit_vector_is_identity() -> None:
    """Vector diffusion of ones leaves Q unchanged."""
    num_derivatives = 1
    state_dim_per_block = 2
    total_dim = (num_derivatives + 1) * state_dim_per_block
    process_noise = jnp.eye(total_dim) + 0.1
    diffusion = jnp.ones(state_dim_per_block)
    result = apply_diffusion(
        process_noise_cov=process_noise,
        diffusion=diffusion,
        num_derivatives=num_derivatives,
    )
    assert jnp.allclose(result, process_noise)


def test_apply_diffusion_vector_scales_each_state_block_independently() -> None:
    """Vector diffusion scales each per-state-dim ``(q+1)`` block by ``d_i``.

    Grounded in opifex's state-major IWP construction: the prior SDE
    has block-diagonal process-noise with one ``(q+1) x (q+1)`` block
    per Wiener-process dimension. Applying vector diffusion ``d``
    multiplies the ``i``-th block by ``d_i``.
    """
    from opifex.uncertainty.scientific._priors_sde import iwp_sde
    from opifex.uncertainty.statespace.lti_sde import discretize_lti_sde

    num_derivatives = 2
    wiener_dim = 3
    drift, dispersion = iwp_sde(
        num_derivatives=num_derivatives, wiener_process_dimension=wiener_dim
    )
    _, process_noise = discretize_lti_sde(
        drift_matrix=drift, dispersion_matrix=dispersion, dt=jnp.asarray(0.25)
    )

    diffusion = jnp.array([0.5, 2.0, 4.0])
    result = apply_diffusion(
        process_noise_cov=process_noise,
        diffusion=diffusion,
        num_derivatives=num_derivatives,
    )

    block_size = num_derivatives + 1
    for state_index in range(wiener_dim):
        start = state_index * block_size
        end = start + block_size
        scaled_block = result[start:end, start:end]
        expected_block = diffusion[state_index] * process_noise[start:end, start:end]
        assert jnp.allclose(scaled_block, expected_block, atol=1e-6)


def test_apply_diffusion_vector_matches_kronecker_scaling() -> None:
    """Symbolic identity: ``S Q S`` with ``S = diag(sqrt(d)) ⊗ I_{q+1}``.

    Confirms the closed-form Kronecker factor matches opifex's
    state-major convention (outer index over state dim, inner index
    over derivative order).
    """
    num_derivatives = 2
    wiener_dim = 3
    total_dim = wiener_dim * (num_derivatives + 1)
    key = jax.random.PRNGKey(0)
    q_factor = jax.random.normal(key, (total_dim, total_dim))
    process_noise = q_factor @ q_factor.T
    diffusion = jnp.array([0.5, 2.0, 4.0])

    result = apply_diffusion(
        process_noise_cov=process_noise,
        diffusion=diffusion,
        num_derivatives=num_derivatives,
    )

    scale_matrix = jnp.kron(jnp.diag(jnp.sqrt(diffusion)), jnp.eye(num_derivatives + 1))
    expected = scale_matrix @ process_noise @ scale_matrix
    assert jnp.allclose(result, expected, atol=1e-6)


def test_apply_diffusion_compiles_under_jit() -> None:
    """``apply_diffusion`` must compile under ``jax.jit``."""
    process_noise = jnp.eye(4)
    diffusion = jnp.array([1.0, 2.0])
    jitted = jax.jit(apply_diffusion, static_argnames=("num_derivatives",))
    result = jitted(process_noise_cov=process_noise, diffusion=diffusion, num_derivatives=1)
    assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# Adapter-spec wrap() concretization
# ---------------------------------------------------------------------------


def test_manifold_update_spec_wrap_returns_manifold_update_callable() -> None:
    """``ManifoldUpdateSpec.wrap`` returns the manifold-update primitive."""
    spec: Any = ManifoldUpdateSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is manifold_update


def test_dense_output_sampling_spec_wrap_returns_sample_callable() -> None:
    """``DenseOutputSamplingSpec.wrap`` returns the dense-output sampler."""
    spec: Any = DenseOutputSamplingSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is dense_output_sample


def test_apply_diffusion_spec_wrap_returns_apply_diffusion_callable() -> None:
    """``ApplyDiffusionSpec.wrap`` returns the diffusion scaler."""
    spec: Any = ApplyDiffusionSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is apply_diffusion
