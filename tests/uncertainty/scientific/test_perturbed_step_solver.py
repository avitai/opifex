r"""Tests for the Conrad+ 2017 perturbed-step probabilistic ODE solver.

Reference
---------
* Conrad, Girolami, Särkkä, Stuart, Zygalakis 2017 — *Statistical
  analysis of differential equations: introducing probability measures
  on numerical solutions*, Statistics and Computing 27, 1065-1082
  (arXiv:1506.04592).

The method takes a deterministic one-step integrator :math:`\Psi_h`
of order :math:`p` and, at every step, adds a calibrated mean-zero
Gaussian state perturbation :math:`\xi_k \sim \mathcal{N}(0,
\sigma^2 h^{2p+1} I)`. The :math:`h^{2p+1}` covariance scaling (std
:math:`\propto h^{p+1/2}`) is the unique choice that preserves the
deterministic method's order-:math:`p` convergence in the mean
(Conrad+ 2017, Assumption 1 / Theorem 2.2).

The verified properties (per task F19):

* (a) Convergence in mean — on ``dy/dt = y, y(0) = 1`` (exact
  :math:`e^t`), the ensemble mean matches the deterministic solver and
  improves at the integrator's order as ``h`` halves.
* (b) Spread scaling — the terminal-state std scales with ``h`` at the
  expected power ``p + 1/2``: halving ``h`` shrinks the std by
  ``2^{-(p+1/2)}``.
* (c) ``sigma = 0`` recovers the deterministic solution exactly.
* (d) Outputs are finite.
* jit / grad / vmap smoke (vmap over the ensemble seed).
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.registry import UQCapability
from opifex.uncertainty.scientific._specialised import perturbed_step_solve
from opifex.uncertainty.scientific.probabilistic_numerics import (
    PerturbedStepSolverSpec,
)


def _exponential_vector_field(time: jax.Array, state: jax.Array) -> jax.Array:
    """Right-hand side of ``dy/dt = y`` (exact solution ``e^t``)."""
    del time
    return state


_INITIAL_STATE = jnp.array([1.0])
_T0 = 0.0
_T1 = 1.0


# ---------------------------------------------------------------------------
# (c) sigma = 0 recovers the deterministic solution exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "order"), [("euler", 1), ("rk4", 4)])
def test_zero_noise_recovers_deterministic_solution(
    method: Literal["euler", "rk4"], order: int
) -> None:
    """With ``noise_scale = 0`` every ensemble member equals the deterministic run."""
    with jax.enable_x64(True):
        ensemble = perturbed_step_solve(
            vector_field=_exponential_vector_field,
            initial_state=_INITIAL_STATE.astype(jnp.float64),
            t0=_T0,
            t1=_T1,
            num_steps=16,
            noise_scale=0.0,
            num_samples=8,
            key=jax.random.PRNGKey(0),
            method=method,
        )
        deterministic = ensemble[0]
        # Every member is bitwise identical to the deterministic run:
        # max - min across the ensemble is exactly zero (``jnp.std`` of
        # identical values carries float roundoff, so use the range).
        max_member_difference = float(jnp.max(jnp.abs(ensemble - deterministic[None, :, :])))
        spread_range = float(jnp.max(jnp.max(ensemble, axis=0) - jnp.min(ensemble, axis=0)))
    assert max_member_difference == 0.0
    assert spread_range == 0.0
    del order


@pytest.mark.parametrize(("method", "order"), [("euler", 1), ("rk4", 4)])
def test_deterministic_path_converges_at_solver_order(
    method: Literal["euler", "rk4"], order: int
) -> None:
    """The ``sigma = 0`` terminal error decays at the integrator's order ``p``.

    The global error of the base integrator is :math:`O(h^p)`, so
    halving ``h`` divides the terminal error by :math:`\\approx 2^p`. We
    measure the empirical rate across a 2x grid refinement and require
    it to bracket :math:`2^p`.
    """

    def terminal_error(num_steps: int) -> float:
        with jax.enable_x64(True):
            ensemble = perturbed_step_solve(
                vector_field=_exponential_vector_field,
                initial_state=_INITIAL_STATE.astype(jnp.float64),
                t0=_T0,
                t1=_T1,
                num_steps=num_steps,
                noise_scale=0.0,
                num_samples=1,
                key=jax.random.PRNGKey(0),
                method=method,
            )
            return float(jnp.abs(ensemble[0, -1, 0] - jnp.exp(1.0)))

    coarse_error = terminal_error(16)
    fine_error = terminal_error(32)
    observed_rate = coarse_error / fine_error
    expected = 2.0**order
    assert 0.6 * expected < observed_rate < 1.4 * expected


# ---------------------------------------------------------------------------
# (a) Convergence in mean — the ensemble mean is the deterministic solution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["euler", "rk4"])
def test_ensemble_mean_matches_deterministic_solution(
    method: Literal["euler", "rk4"],
) -> None:
    """The ensemble mean equals the ``sigma = 0`` trajectory to MC error.

    The Conrad+ 2017 perturbation is mean-zero, so the expected solution
    of the randomised method is exactly the deterministic trajectory
    :math:`\\Psi_h`. Hence the ensemble mean recovers the deterministic
    solver (and thus its order-:math:`p` accuracy versus :math:`e^t`) up
    to Monte-Carlo error that shrinks as :math:`1/\\sqrt{N}`.
    """
    num_steps = 16
    with jax.enable_x64(True):
        deterministic = perturbed_step_solve(
            vector_field=_exponential_vector_field,
            initial_state=_INITIAL_STATE.astype(jnp.float64),
            t0=_T0,
            t1=_T1,
            num_steps=num_steps,
            noise_scale=0.0,
            num_samples=1,
            key=jax.random.PRNGKey(0),
            method=method,
        )[0]
        ensemble = perturbed_step_solve(
            vector_field=_exponential_vector_field,
            initial_state=_INITIAL_STATE.astype(jnp.float64),
            t0=_T0,
            t1=_T1,
            num_steps=num_steps,
            noise_scale=0.05,
            num_samples=8192,
            key=jax.random.PRNGKey(1),
            method=method,
        )
        ensemble_mean = jnp.mean(ensemble, axis=0)
        terminal_std = float(jnp.std(ensemble[:, -1, 0]))
        max_mean_gap = float(jnp.max(jnp.abs(ensemble_mean - deterministic)))
    # Monte-Carlo error of the mean is std / sqrt(N); allow a generous 5x.
    monte_carlo_tolerance = 5.0 * terminal_std / jnp.sqrt(8192.0)
    assert max_mean_gap < monte_carlo_tolerance


# ---------------------------------------------------------------------------
# (b) Spread scales with step size at the expected power p + 1/2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "order"), [("euler", 1), ("rk4", 4)])
def test_terminal_spread_scales_with_step_size(method: Literal["euler", "rk4"], order: int) -> None:
    """Halving ``h`` shrinks the terminal std by ``~2^{-(p+1/2)}``.

    A single perturbation has std ``sigma h^{p+1/2}``; accumulated over
    ``N = (t1 - t0) / h`` steps the terminal std scales like
    ``sqrt(N) * h^{p+1/2} = sqrt(t1 - t0) * h^p``. Halving ``h`` thus
    divides the terminal std by ``~2^p`` to leading order. We assert the
    ratio brackets that power generously to absorb Monte-Carlo noise and
    the dynamics' amplification factor.
    """

    def terminal_std(num_steps: int) -> float:
        with jax.enable_x64(True):
            ensemble = perturbed_step_solve(
                vector_field=_exponential_vector_field,
                initial_state=_INITIAL_STATE.astype(jnp.float64),
                t0=_T0,
                t1=_T1,
                num_steps=num_steps,
                noise_scale=0.1,
                num_samples=8192,
                key=jax.random.PRNGKey(2),
                method=method,
            )
        return float(jnp.std(ensemble[:, -1, 0]))

    coarse_std = terminal_std(8)
    fine_std = terminal_std(16)
    ratio = coarse_std / fine_std
    expected = 2.0**order
    # The accumulated terminal std scales as h^p; bracket the empirical
    # ratio around 2^p with a factor-of-two window on each side.
    assert 0.5 * expected < ratio < 2.0 * expected


def test_larger_noise_scale_gives_larger_spread() -> None:
    """Spread grows monotonically with ``noise_scale`` at fixed grid."""

    def terminal_std(noise_scale: float) -> float:
        with jax.enable_x64(True):
            ensemble = perturbed_step_solve(
                vector_field=_exponential_vector_field,
                initial_state=_INITIAL_STATE.astype(jnp.float64),
                t0=_T0,
                t1=_T1,
                num_steps=16,
                noise_scale=noise_scale,
                num_samples=4096,
                key=jax.random.PRNGKey(3),
                method="rk4",
            )
        return float(jnp.std(ensemble[:, -1, 0]))

    assert terminal_std(0.5) > terminal_std(0.1) > terminal_std(0.01)


# ---------------------------------------------------------------------------
# (d) Outputs are finite and well-shaped
# ---------------------------------------------------------------------------


def test_outputs_are_finite_and_well_shaped() -> None:
    """Returned ensemble is finite with shape ``(num_samples, num_steps + 1, dim)``."""
    num_samples, num_steps, dim = 12, 20, 1
    ensemble = perturbed_step_solve(
        vector_field=_exponential_vector_field,
        initial_state=_INITIAL_STATE,
        t0=_T0,
        t1=_T1,
        num_steps=num_steps,
        noise_scale=0.1,
        num_samples=num_samples,
        key=jax.random.PRNGKey(4),
        method="rk4",
    )
    assert ensemble.shape == (num_samples, num_steps + 1, dim)
    assert bool(jnp.all(jnp.isfinite(ensemble)))
    # The initial condition is shared and unperturbed across the ensemble.
    assert jnp.allclose(ensemble[:, 0, :], _INITIAL_STATE[None, :])


def test_multidimensional_system_is_supported() -> None:
    """A 2-D linear system integrates without shape errors."""

    def rotation(time: jax.Array, state: jax.Array) -> jax.Array:
        del time
        return jnp.array([-state[1], state[0]])

    ensemble = perturbed_step_solve(
        vector_field=rotation,
        initial_state=jnp.array([1.0, 0.0]),
        t0=0.0,
        t1=1.0,
        num_steps=32,
        noise_scale=0.05,
        num_samples=16,
        key=jax.random.PRNGKey(5),
        method="rk4",
    )
    assert ensemble.shape == (16, 33, 2)
    assert bool(jnp.all(jnp.isfinite(ensemble)))


# ---------------------------------------------------------------------------
# jit / grad / vmap transform smoke
# ---------------------------------------------------------------------------


def test_solver_compiles_under_jit() -> None:
    """``perturbed_step_solve`` traces and runs under ``jax.jit``."""
    jitted = jax.jit(
        perturbed_step_solve,
        static_argnames=("vector_field", "num_steps", "num_samples", "method"),
    )
    ensemble = jitted(
        vector_field=_exponential_vector_field,
        initial_state=_INITIAL_STATE,
        t0=_T0,
        t1=_T1,
        num_steps=16,
        noise_scale=0.1,
        num_samples=8,
        key=jax.random.PRNGKey(6),
        method="rk4",
    )
    assert ensemble.shape == (8, 17, 1)
    assert bool(jnp.all(jnp.isfinite(ensemble)))


def test_solver_is_vmappable_over_seed() -> None:
    """``jax.vmap`` over the PRNG key yields a batch of independent ensembles."""

    def run(key: jax.Array) -> jax.Array:
        # Euler with a coarse grid gives a perturbation large enough to
        # make seed-dependence unambiguous.
        return perturbed_step_solve(
            vector_field=_exponential_vector_field,
            initial_state=_INITIAL_STATE,
            t0=_T0,
            t1=_T1,
            num_steps=8,
            noise_scale=0.2,
            num_samples=8,
            key=key,
            method="euler",
        )

    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    batched = jax.vmap(run)(keys)
    assert batched.shape == (4, 8, 9, 1)
    assert bool(jnp.all(jnp.isfinite(batched)))
    # Different seeds give different ensembles past the shared initial state.
    assert not bool(jnp.allclose(batched[0, :, 1:], batched[1, :, 1:]))


def test_solver_is_differentiable_through_initial_state() -> None:
    """Gradient of the mean terminal value w.r.t. the initial state is finite.

    For ``dy/dt = y`` the deterministic-mean terminal value is
    ``~ e * y0``, so the gradient is ``~ e``; we only require a finite,
    positive, order-correct gradient here (transform smoke).
    """

    def mean_terminal(initial_state: jax.Array) -> jax.Array:
        ensemble = perturbed_step_solve(
            vector_field=_exponential_vector_field,
            initial_state=initial_state,
            t0=_T0,
            t1=_T1,
            num_steps=16,
            noise_scale=0.1,
            num_samples=64,
            key=jax.random.PRNGKey(8),
            method="rk4",
        )
        return jnp.mean(ensemble[:, -1, 0])

    gradient = jax.grad(mean_terminal)(_INITIAL_STATE)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(gradient[0]) > 1.0


# ---------------------------------------------------------------------------
# Spec wiring — the deferral / NotImplementedError must be gone
# ---------------------------------------------------------------------------


def test_perturbed_step_solver_spec_wrap_returns_backend() -> None:
    """``PerturbedStepSolverSpec.wrap`` returns the real backend callable."""
    spec = PerturbedStepSolverSpec()
    returned = spec.wrap(model=None, capability=UQCapability())
    assert returned is perturbed_step_solve


def test_perturbed_step_solver_spec_advertises_conrad_tags() -> None:
    """The spec advertises the perturbed-step family tags."""
    spec = PerturbedStepSolverSpec()
    assert "perturbed_step" in spec.family_tags
    assert "stochastic_perturbation" in spec.family_tags
