"""Tests for Kronecker-factored curvature (KFAC) and KFAC-Laplace.

KFAC factorises each layer's GGN/Fisher block as ``A ⊗ G`` with
``A = E[a a^T]`` (input/activation covariance) and ``G = E[g g^T]``
(pre-activation-gradient covariance), per Martens & Grosse 2015. For a
single-example batch the factorisation is exact: the per-example GGN block
``(a a^T) ⊗ (g g^T)`` equals ``A ⊗ G``. The KFAC-Laplace posterior wraps
the damped factors into a block-diagonal-of-Kronecker precision, per Ritter,
Botev & Barber 2018.

The exact per-layer GGN block used as the reference is built from the same
forward-over-reverse kernel as :func:`ggn_vector_product` (each canonical
basis vector mapped through the GGN gives a column of the dense block).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.curvature.kfac import (
    kfac_factors,
    kfac_laplace_posterior,
)
from opifex.uncertainty.curvature.structured import (
    BlockDiagonal,
    KroneckerProduct,
)


jax.config.update("jax_enable_x64", True)


def _mlp_apply(parameters: tuple[jax.Array, ...], inputs: jax.Array) -> jax.Array:
    """Two-layer ``tanh`` MLP: ``W2 tanh(W1 x)`` (no biases, for clarity)."""
    kernel_one, kernel_two = parameters
    hidden = jnp.tanh(inputs @ kernel_one)
    return hidden @ kernel_two


def _mlp_tapped(
    parameters: tuple[jax.Array, ...],
    perturbations: tuple[jax.Array, ...],
    inputs: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Tapped MLP forward exposing per-layer inputs and pre-activation taps.

    Each layer's pre-activation gains an additive perturbation (all zeros at
    evaluation); the cotangent of that perturbation is the pre-activation
    gradient/Jacobian used to form the right Kronecker factor. The returned
    activation tuple holds each layer's input ``a_l``.
    """
    kernel_one, kernel_two = parameters
    perturbation_one, perturbation_two = perturbations
    activation_one = inputs
    pre_activation_one = activation_one @ kernel_one + perturbation_one
    activation_two = jnp.tanh(pre_activation_one)
    outputs = activation_two @ kernel_two + perturbation_two
    return outputs, (activation_one, activation_two)


def _squared_loss(outputs: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean squared error summed over the output dimension."""
    return 0.5 * jnp.mean(jnp.sum((outputs - targets) ** 2, axis=-1))


def _make_problem(
    *, batch_size: int, in_dim: int, hidden_dim: int, out_dim: int, seed: int
) -> tuple[tuple[jax.Array, jax.Array], jax.Array, jax.Array]:
    """Return ``(parameters, inputs, targets)`` for the test MLP."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    kernel_one = jax.random.normal(keys[0], (in_dim, hidden_dim), dtype=jnp.float64) * 0.3
    kernel_two = jax.random.normal(keys[1], (hidden_dim, out_dim), dtype=jnp.float64) * 0.3
    inputs = jax.random.normal(keys[2], (batch_size, in_dim), dtype=jnp.float64)
    targets = jax.random.normal(keys[3], (batch_size, out_dim), dtype=jnp.float64)
    return (kernel_one, kernel_two), inputs, targets


def _exact_layer_ggn_block(
    parameters: tuple[jax.Array, ...],
    inputs: jax.Array,
    targets: jax.Array,
    layer_index: int,
) -> jax.Array:
    """Dense per-layer GGN block ``J_l^T H_y L J_l`` for one kernel.

    Built column-by-column by mapping each canonical basis vector of the
    layer's flattened kernel through the forward-over-reverse GGN.
    """
    kernel_shape = parameters[layer_index].shape
    flat_size = int(jnp.prod(jnp.asarray(kernel_shape)))

    def model_of_layer(flat_kernel: jax.Array) -> jax.Array:
        updated = list(parameters)
        updated[layer_index] = flat_kernel.reshape(kernel_shape)
        return _mlp_apply(tuple(updated), inputs)

    flat_kernel = parameters[layer_index].reshape(-1)
    outputs, jvp_basis = jax.jvp(
        model_of_layer, (flat_kernel,), (jnp.eye(flat_size, dtype=jnp.float64)[0],)
    )

    def loss_of_outputs(values: jax.Array) -> jax.Array:
        return _squared_loss(values, targets)

    grad_loss = jax.grad(loss_of_outputs)
    _, pullback = jax.vjp(model_of_layer, flat_kernel)

    def ggn_column(basis_vector: jax.Array) -> jax.Array:
        _, jvp_value = jax.jvp(model_of_layer, (flat_kernel,), (basis_vector,))
        _, hvp = jax.jvp(grad_loss, (outputs,), (jvp_value,))
        (column,) = pullback(hvp)
        return column

    del jvp_basis
    return jax.vmap(ggn_column)(jnp.eye(flat_size, dtype=jnp.float64)).T


def test_kfac_factors_returns_per_layer_kronecker_shapes() -> None:
    """``kfac_factors`` yields one ``(A, G)`` pair per linear layer."""
    parameters, inputs, targets = _make_problem(
        batch_size=8, in_dim=3, hidden_dim=4, out_dim=2, seed=0
    )
    factors = kfac_factors(_mlp_tapped, _squared_loss, parameters, inputs, targets)
    assert len(factors) == 2
    activation_one, gradient_one = factors[0]
    activation_two, gradient_two = factors[1]
    assert activation_one.shape == (3, 3)
    assert gradient_one.shape == (4, 4)
    assert activation_two.shape == (4, 4)
    assert gradient_two.shape == (2, 2)


def test_kfac_factors_are_exact_for_single_example_batch() -> None:
    """For a single example, ``A ⊗ G`` equals the exact per-layer GGN block."""
    parameters, inputs, targets = _make_problem(
        batch_size=1, in_dim=3, hidden_dim=4, out_dim=2, seed=1
    )
    factors = kfac_factors(_mlp_tapped, _squared_loss, parameters, inputs, targets)
    for layer_index, (activation, gradient) in enumerate(factors):
        reconstructed = jnp.kron(activation, gradient)
        exact = _exact_layer_ggn_block(parameters, inputs, targets, layer_index)
        assert jnp.allclose(reconstructed, exact, atol=1e-8)


def test_kfac_factors_are_symmetric_positive_semidefinite() -> None:
    """Both factors are symmetric PSD (they are activation/gradient covariances)."""
    parameters, inputs, targets = _make_problem(
        batch_size=16, in_dim=3, hidden_dim=5, out_dim=2, seed=2
    )
    factors = kfac_factors(_mlp_tapped, _squared_loss, parameters, inputs, targets)
    for activation, gradient in factors:
        for matrix in (activation, gradient):
            assert jnp.allclose(matrix, matrix.T, atol=1e-10)
            eigenvalues = jnp.linalg.eigvalsh(matrix)
            assert jnp.all(eigenvalues >= -1e-8)


def test_kfac_laplace_posterior_is_block_diagonal_of_kronecker() -> None:
    """The posterior precision is a block-diagonal of damped Kronecker products."""
    parameters, inputs, targets = _make_problem(
        batch_size=8, in_dim=3, hidden_dim=4, out_dim=2, seed=3
    )
    posterior = kfac_laplace_posterior(
        _mlp_tapped, _squared_loss, parameters, inputs, targets, prior_precision=1.0
    )
    assert isinstance(posterior.precision, BlockDiagonal)
    assert len(posterior.precision.blocks) == 2
    for block in posterior.precision.blocks:
        assert isinstance(block, KroneckerProduct)


def test_kfac_laplace_precision_is_positive_definite() -> None:
    """Damping makes every layer precision block positive definite."""
    parameters, inputs, targets = _make_problem(
        batch_size=8, in_dim=3, hidden_dim=4, out_dim=2, seed=4
    )
    posterior = kfac_laplace_posterior(
        _mlp_tapped, _squared_loss, parameters, inputs, targets, prior_precision=1.0
    )
    for block in posterior.precision.blocks:
        eigenvalues = jnp.linalg.eigvalsh(block.to_dense())
        assert jnp.all(eigenvalues > 0.0)


def test_kfac_laplace_predictive_variance_is_finite_and_shrinks_with_damping() -> None:
    """Predictive variance is finite and monotonically shrinks as damping grows."""
    parameters, inputs, targets = _make_problem(
        batch_size=8, in_dim=3, hidden_dim=4, out_dim=2, seed=5
    )
    test_inputs = jax.random.normal(jax.random.PRNGKey(99), (5, 3), dtype=jnp.float64)

    def total_variance(prior_precision: float) -> jax.Array:
        posterior = kfac_laplace_posterior(
            _mlp_tapped,
            _squared_loss,
            parameters,
            inputs,
            targets,
            prior_precision=prior_precision,
        )
        variance = posterior.predictive_variance(_mlp_apply, parameters, test_inputs)
        return jnp.sum(variance)

    low_damping = total_variance(1.0)
    high_damping = total_variance(100.0)
    assert jnp.all(jnp.isfinite(low_damping))
    assert jnp.all(jnp.isfinite(high_damping))
    assert high_damping < low_damping


def test_kfac_factors_is_jit_compatible() -> None:
    """``kfac_factors`` runs under ``jax.jit``."""
    parameters, inputs, targets = _make_problem(
        batch_size=8, in_dim=3, hidden_dim=4, out_dim=2, seed=6
    )

    @jax.jit
    def factor_traces(
        params: tuple[jax.Array, ...], batch: jax.Array, labels: jax.Array
    ) -> jax.Array:
        factors = kfac_factors(_mlp_tapped, _squared_loss, params, batch, labels)
        return jnp.stack([jnp.trace(activation) for activation, _ in factors])

    traces = factor_traces(parameters, inputs, targets)
    assert jnp.all(jnp.isfinite(traces))


def test_kfac_predictive_variance_is_grad_compatible() -> None:
    """The KFAC predictive variance is differentiable w.r.t. the prior precision."""
    parameters, inputs, targets = _make_problem(
        batch_size=8, in_dim=3, hidden_dim=4, out_dim=2, seed=7
    )
    test_inputs = jax.random.normal(jax.random.PRNGKey(7), (3, 3), dtype=jnp.float64)

    def total_variance(prior_precision: jax.Array) -> jax.Array:
        posterior = kfac_laplace_posterior(
            _mlp_tapped,
            _squared_loss,
            parameters,
            inputs,
            targets,
            prior_precision=prior_precision,
        )
        return jnp.sum(posterior.predictive_variance(_mlp_apply, parameters, test_inputs))

    gradient = jax.grad(total_variance)(jnp.asarray(2.0))
    assert jnp.isfinite(gradient)


def test_kfac_factors_is_vmap_compatible() -> None:
    """A batch of datasets maps through ``kfac_factors``."""
    keys = jax.random.split(jax.random.PRNGKey(8), 3)
    parameters = (
        jax.random.normal(keys[0], (3, 4), dtype=jnp.float64) * 0.3,
        jax.random.normal(keys[1], (4, 2), dtype=jnp.float64) * 0.3,
    )
    dataset_inputs = jax.random.normal(keys[2], (2, 8, 3), dtype=jnp.float64)
    dataset_targets = jnp.zeros((2, 8, 2))

    def first_activation_trace(batch: jax.Array, labels: jax.Array) -> jax.Array:
        factors = kfac_factors(_mlp_tapped, _squared_loss, parameters, batch, labels)
        return jnp.trace(factors[0][0])

    traces = jax.vmap(first_activation_trace)(dataset_inputs, dataset_targets)
    assert traces.shape == (2,)
    assert jnp.all(jnp.isfinite(traces))
