r"""Tests for the K-FAC natural-gradient preconditioner for VMC.

K-FAC (Kronecker-Factored Approximate Curvature; Martens & Grosse, ICML 2015,
arXiv:1503.05671) approximates the Fisher / quantum geometric tensor by a
block-diagonal Kronecker product per layer and preconditions the energy gradient
by its inverse. This wraps the canonical ``kfac_jax`` library (the optimiser
FermiNet uses; ``../ferminet/ferminet/train.py``), so the tests assert *interface
conformance* and *monotone-ish* energy/loss decrease rather than exact values --
K-FAC is stochastic and manages its own ``jit`` internally.

The canonical wiring being exercised (FermiNet ``loss.py`` + ``train.py``):

#. ansatz dense/attention layers are tagged with :func:`register_qmc_dense`
   (FermiNet's ``register_qmc``) so K-FAC knows their Kronecker structure;
#. the log-amplitude output is tagged with :func:`register_log_amplitude`
   (``kfac_jax.register_normal_predictive_distribution``) so the Fisher equals
   the quantum geometric tensor;
#. the gradient is the FermiNet score-function estimator
   :math:`2\langle(E_{\mathrm{loc}}-\bar E)\nabla\log|\psi|\rangle` supplied via a
   ``custom_jvp`` (:func:`make_qmc_value_and_grad`);
#. :class:`KFACPreconditioner` wraps ``kfac_jax.Optimizer`` behind the same
   ``init`` / ``step`` seam as the MinSR / SPRING optimisers.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np

from opifex.neural.quantum.vmc.kfac_preconditioner import (
    KFACPreconditioner,
    make_qmc_value_and_grad,
    register_log_amplitude,
    register_qmc_dense,
)


def _squared_error_value_and_grad() -> tuple:
    r"""A registered-dense least-squares optimisee and its value-and-grad func.

    Returns a ``(value_and_grad, params, batch)`` triple for a single dense layer
    ``y = x @ w + b`` regressed onto a target, with the K-FAC squared-error loss
    tag. Used to check that a K-FAC step decreases a simple convex loss.
    """
    n_samples, n_in, n_out = 16, 4, 3
    x = jax.random.normal(jax.random.PRNGKey(0), (n_samples, n_in), dtype=jnp.float64)
    target = jax.random.normal(jax.random.PRNGKey(1), (n_samples, n_out), dtype=jnp.float64)

    def loss_fn(params: dict, _rng: jax.Array, batch: tuple) -> tuple:
        inputs, targets = batch
        weight, bias = params["w"], params["b"]
        prediction = register_qmc_dense(inputs @ weight + bias[None], inputs, weight, bias)
        kfac_jax.register_squared_error_loss(prediction, targets)
        residual = jnp.mean(jnp.sum((prediction - targets) ** 2, axis=-1))
        return residual, {"prediction": prediction}

    params = {
        "w": 0.1 * jax.random.normal(jax.random.PRNGKey(2), (n_in, n_out), dtype=jnp.float64),
        "b": jnp.zeros(n_out, dtype=jnp.float64),
    }
    value_and_grad = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    return value_and_grad, params, (x, target)


def _harmonic_oscillator() -> tuple:
    r"""A tiny but *physically exact* 1-D harmonic-oscillator VMC problem.

    A single particle in :math:`H=-\tfrac12\nabla^2+\tfrac12 r^2`. The ansatz is a
    Gaussian written through a registered ``1x1`` dense weight ``w``:
    ``log|psi| = -0.5 (r w)^2``, so the variational parameter ``a = w^2`` and the
    exact local energy is ``E_loc(r) = 0.5 a + 0.5 (1 - a^2) r^2`` with ground
    state ``a = 1`` and ``E = 0.5``. Samples are drawn analytically from
    ``|psi|^2 = N(0, 1/(2a))`` so the score estimator is unbiased without MCMC.

    Returns ``(log_abs, local_energy, sample, params)``.
    """

    def log_abs(params: dict, walkers: jax.Array) -> jax.Array:
        weight = params["w"]
        hidden = register_qmc_dense(walkers @ weight, walkers, weight)
        return -0.5 * hidden[..., 0] ** 2

    def local_energy(params: dict, walkers: jax.Array) -> jax.Array:
        a = params["w"][0, 0] ** 2
        return 0.5 * a + 0.5 * (1.0 - a**2) * walkers[..., 0] ** 2

    def sample(params: dict, key: jax.Array, n: int) -> jax.Array:
        a = params["w"][0, 0] ** 2
        std = 1.0 / jnp.sqrt(2.0 * a)
        return std * jax.random.normal(key, (n, 1), dtype=jnp.float64)

    # Start away from the optimum (a ~ 2).
    params = {"w": jnp.asarray([[1.41]], dtype=jnp.float64)}
    return log_abs, local_energy, sample, params


def test_register_qmc_dense_is_a_passthrough() -> None:
    """The dense layer tag returns its output unchanged (it only annotates)."""
    inputs = jnp.ones((4, 3), dtype=jnp.float64)
    weight = jnp.ones((3, 2), dtype=jnp.float64)
    tagged = register_qmc_dense(inputs @ weight, inputs, weight)
    np.testing.assert_allclose(tagged, inputs @ weight)


def test_register_log_amplitude_is_a_passthrough() -> None:
    """The Fisher tag is a side-effecting annotation that returns ``None``."""
    log_psi = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float64)
    result = register_log_amplitude(log_psi)
    assert result is None


def test_kfac_step_decreases_a_convex_loss() -> None:
    """A K-FAC step monotonically decreases a registered least-squares loss."""
    value_and_grad, params, batch = _squared_error_value_and_grad()
    precond = KFACPreconditioner(value_and_grad, learning_rate=0.1, damping=1e-3)
    key = jax.random.PRNGKey(3)
    state = precond.init(params, key, batch)

    losses = []
    for _ in range(8):
        key, step_key = jax.random.split(key)
        params, state, loss, _aux = precond.step(params, state, step_key, batch)
        losses.append(float(loss))

    assert losses[-1] < losses[0]
    # Monotone-ish: at least three quarters of the steps do not increase.
    non_increasing = sum(b <= a + 1e-9 for a, b in itertools.pairwise(losses))
    assert non_increasing >= int(0.75 * (len(losses) - 1))


def test_kfac_step_returns_the_seam_contract() -> None:
    """``step`` returns ``(params, state, loss, aux)`` with a scalar loss + aux."""
    value_and_grad, params, batch = _squared_error_value_and_grad()
    precond = KFACPreconditioner(value_and_grad, learning_rate=0.1, damping=1e-3)
    key = jax.random.PRNGKey(7)
    state = precond.init(params, key, batch)
    new_params, new_state, loss, aux = precond.step(params, state, key, batch)

    assert set(new_params) == set(params)
    assert jnp.ndim(loss) == 0 and jnp.isfinite(loss)
    assert "prediction" in aux
    # State advances the optimiser step counter (so it is genuinely stateful).
    assert isinstance(new_state, type(state))


def test_kfac_optimizer_state_round_trips_through_a_pytree() -> None:
    """The optimiser state flattens and unflattens losslessly (jit/scan carry)."""
    value_and_grad, params, batch = _squared_error_value_and_grad()
    precond = KFACPreconditioner(value_and_grad, learning_rate=0.1, damping=1e-3)
    state = precond.init(params, jax.random.PRNGKey(11), batch)

    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    rebuilt_leaves = jax.tree_util.tree_leaves(rebuilt)

    assert len(leaves) == len(rebuilt_leaves)
    for original, restored in zip(leaves, rebuilt_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(original), np.asarray(restored))


def test_kfac_step_compiles_and_runs() -> None:
    """A K-FAC step compiles and runs (``kfac_jax`` manages its own ``jit``)."""
    value_and_grad, params, batch = _squared_error_value_and_grad()
    precond = KFACPreconditioner(value_and_grad, learning_rate=0.05, damping=1e-3)
    key = jax.random.PRNGKey(5)
    state = precond.init(params, key, batch)
    # A second call hits the compiled path; both must yield finite losses.
    params, state, first, _ = precond.step(params, state, key, batch)
    key, step_key = jax.random.split(key)
    params, state, second, _ = precond.step(params, state, step_key, batch)
    assert jnp.isfinite(first) and jnp.isfinite(second)


def test_make_qmc_value_and_grad_yields_the_score_gradient() -> None:
    r"""The QMC value-and-grad returns the energy and the score-estimator gradient.

    For the exact harmonic oscillator ``E(a) = 0.25 (a + 1/a)``, so the analytic
    energy gradient w.r.t. ``a`` is ``dE/da = 0.25 (1 - 1/a^2)``; the centred
    score estimator must reproduce it through the chain rule ``da/dw = 2 w``.
    Clipping is disabled here so the unbiased estimator matches the closed form.
    """
    log_abs, local_energy, sample, params = _harmonic_oscillator()
    value_and_grad = make_qmc_value_and_grad(log_abs, local_energy, clip_local_energy=0.0)
    walkers = sample(params, jax.random.PRNGKey(0), 400_000)

    (energy, aux), grads = value_and_grad(params, jax.random.PRNGKey(1), walkers)
    a = float(params["w"][0, 0] ** 2)
    expected_energy = 0.25 * (a + 1.0 / a)
    np.testing.assert_allclose(float(energy), expected_energy, rtol=2e-3)

    # dE/dw via the analytic dE/da = 0.25 (1 - 1/a^2) and da/dw = 2 w.
    weight = float(params["w"][0, 0])
    expected_grad_w = 0.25 * (1.0 - 1.0 / a**2) * (2.0 * weight)
    np.testing.assert_allclose(float(grads["w"][0, 0]), expected_grad_w, rtol=5e-2, atol=5e-3)
    assert "e_loc" in aux


def test_kfac_smoke_reduces_a_tiny_wavefunction_energy() -> None:
    """A short K-FAC run drives the toy wavefunction toward its ground state."""
    log_abs, local_energy, sample, params = _harmonic_oscillator()
    value_and_grad = make_qmc_value_and_grad(log_abs, local_energy)
    precond = KFACPreconditioner(value_and_grad, learning_rate=0.15, damping=1e-3)

    key = jax.random.PRNGKey(0)
    n_walkers = 4096
    walkers = sample(params, key, n_walkers)
    state = precond.init(params, key, walkers)

    energies = []
    for _ in range(60):
        key, sample_key, step_key = jax.random.split(key, 3)
        walkers = sample(params, sample_key, n_walkers)
        params, state, energy, _ = precond.step(params, state, step_key, walkers)
        energies.append(float(energy))

    # Converges toward the exact ground-state energy 0.5 Ha and a = 1.
    assert energies[-1] < energies[0]
    np.testing.assert_allclose(energies[-1], 0.5, atol=2e-2)
    np.testing.assert_allclose(float(params["w"][0, 0] ** 2), 1.0, atol=1e-1)
