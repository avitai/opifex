r"""Kronecker-factored approximate curvature (KFAC) for Laplace UQ.

KFAC approximates each layer's generalized Gauss-Newton / Fisher block by a
Kronecker product of two small factors. For a linear layer with input
activation ``a`` and pre-activation ``s = a W`` whose output-space
curvature is summarised by the pre-activation Jacobian/gradient ``g``, the
exact per-example GGN block of ``vec(W)`` is

.. math::

    (a a^\top) \otimes (J_s^\top H_y J_s),

so KFAC takes the population factors

.. math::

    A = \mathbb{E}[a a^\top], \qquad
    G = \mathbb{E}[J_s^\top H_y J_s],

and approximates the block by ``A ⊗ G`` (exact when the batch holds a single
example, an approximation otherwise because the expectation of the Kronecker
product is replaced by the Kronecker product of the expectations).

This module computes the factors natively with :func:`jax.vjp` / forward-mode
differentiation through a *tapped* model — a forward pass that returns each
layer's input activation and exposes an additive zero perturbation at every
pre-activation. The cotangent of that perturbation is the pre-activation
Jacobian used to build ``G``. This native route is preferred over extracting
factors from ``kfac_jax`` because the tapped-forward contract yields the
``A`` / ``G`` factors directly and stays differentiable under all JAX
transforms.

The :func:`kfac_laplace_posterior` then assembles a damped, layerwise
block-diagonal-of-Kronecker posterior precision (Ritter, Botev & Barber
2018) built from :mod:`opifex.uncertainty.curvature.structured`, and exposes
the linearised predictive variance ``diag(J Σ J^\top)``.

References
----------
* Martens, J. & Grosse, R. 2015 — *Optimizing Neural Networks with
  Kronecker-factored Approximate Curvature*, arXiv:1503.05671.
* Ritter, H., Botev, A. & Barber, D. 2018 — *A Scalable Laplace
  Approximation for Neural Networks*, ICLR 2018.
* Potapczynski, A. et al. 2023 — *CoLA*, arXiv:2309.03060 (structured
  operators backing the posterior precision).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from opifex.uncertainty.curvature.structured import (
    BlockDiagonal,
    KroneckerProduct,
)


LossFunction = Callable[[jax.Array, jax.Array], jax.Array]
PlainModel = Callable[[tuple[jax.Array, ...], jax.Array], jax.Array]


class TappedModel(Protocol):
    """Forward pass exposing per-layer KFAC taps.

    The model is evaluated as ``(parameters, perturbations, inputs)`` where
    ``perturbations`` is a tuple of zero arrays — one per linear layer — added
    to each layer's pre-activation. The forward returns the network output
    plus the tuple of per-layer **input activations** ``a_l``. Differentiating
    the loss with respect to ``perturbations`` yields the pre-activation
    Jacobians/gradients used to build the right Kronecker factor ``G``.
    """

    def __call__(
        self,
        parameters: tuple[jax.Array, ...],
        perturbations: tuple[jax.Array, ...],
        inputs: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        """Return ``(outputs, per_layer_input_activations)``."""
        ...


def _zero_perturbations(
    parameters: tuple[jax.Array, ...], batch_size: int, output_dim: int
) -> tuple[jax.Array, ...]:
    """Zero pre-activation taps: one ``(batch, out_l)`` array per layer.

    Each linear kernel has shape ``(in_l, out_l)``; the pre-activation of
    layer ``l`` therefore has width ``out_l = kernel.shape[1]``.
    """
    del output_dim
    return tuple(
        jnp.zeros((batch_size, kernel.shape[1]), dtype=kernel.dtype) for kernel in parameters
    )


def kfac_factors(
    tapped_model: TappedModel,
    loss: LossFunction,
    parameters: tuple[jax.Array, ...],
    inputs: jax.Array,
    targets: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    r"""Per-layer Kronecker factors ``(A_l, G_l)`` of the GGN/Fisher.

    ``A_l = (1/N) Σ_n a_{l,n} a_{l,n}^\top`` is the input-activation
    covariance and ``G_l = (1/N) Σ_n J_{s_l,n}^\top H_{y,n} J_{s_l,n}`` is the
    pre-activation GGN factor, where ``J_{s_l}`` is the output Jacobian with
    respect to the layer-``l`` pre-activation tap and ``H_y`` is the
    output-space loss Hessian. For a single-example batch ``A_l ⊗ G_l``
    equals the exact per-layer GGN block.

    Args:
        tapped_model: Forward pass exposing per-layer taps (see
            :class:`TappedModel`).
        loss: Maps ``(outputs, targets) -> scalar`` (convex in ``outputs``).
        parameters: Tuple of per-layer kernels ``W_l`` of shape
            ``(in_l, out_l)``.
        inputs: Batched inputs ``(batch, in_0)``.
        targets: Batched targets.

    Returns:
        One ``(A_l, G_l)`` pair per layer; ``A_l`` is ``(in_l, in_l)`` and
        ``G_l`` is ``(out_l, out_l)``.
    """
    batch_size = inputs.shape[0]
    perturbations = _zero_perturbations(parameters, batch_size, targets.shape[-1])

    outputs, activations = tapped_model(parameters, perturbations, inputs)

    def output_of_perturbations(taps: tuple[jax.Array, ...]) -> jax.Array:
        """Network outputs as a function of the pre-activation taps."""
        output_values, _ = tapped_model(parameters, taps, inputs)
        return output_values

    # Output-space loss Hessian-vector product, supplied to the GGN factor.
    loss_gradient = jax.grad(loss)

    def output_hessian_product(cotangent: jax.Array) -> jax.Array:
        """Apply the output-space loss Hessian ``H_y`` to ``cotangent``."""
        _, hessian_product = jax.jvp(
            lambda values: loss_gradient(values, targets), (outputs,), (cotangent,)
        )
        return hessian_product

    _, perturbation_vjp = jax.vjp(output_of_perturbations, perturbations)
    output_dim = outputs.shape[-1]

    def ggn_factor_column(output_basis: jax.Array) -> tuple[jax.Array, ...]:
        """Push one output-space Hessian column back to every pre-activation tap."""
        broadcast = jnp.broadcast_to(output_basis, outputs.shape)
        hessian_column = output_hessian_product(broadcast)
        (tap_cotangents,) = perturbation_vjp(hessian_column)
        return tap_cotangents

    # Right factors ``G_l = Σ_n J_{s_l,n}^T H_y J_{s_l,n}`` accumulated over the
    # batch via the forward-over-reverse GGN against each output basis vector.
    identity_outputs = jnp.eye(output_dim, dtype=outputs.dtype)
    per_basis_taps = jax.vmap(ggn_factor_column)(identity_outputs)

    factors: list[tuple[jax.Array, jax.Array]] = []
    for layer_index, activation in enumerate(activations):
        input_covariance = activation.T @ activation / batch_size
        tap_jacobian = per_basis_taps[layer_index]  # (output_dim, batch, out_l)
        gradient_covariance = jnp.einsum("cni,cnj->ij", tap_jacobian, tap_jacobian) / batch_size
        factors.append((input_covariance, gradient_covariance))
    return tuple(factors)


@dataclass(frozen=True, slots=True, kw_only=True)
class KroneckerLaplacePosterior:
    """Layerwise Kronecker-factored Laplace posterior.

    Attributes:
        mean: MAP parameters (the tuple of per-layer kernels).
        precision: Block-diagonal-of-Kronecker posterior precision; one
            damped ``KroneckerProduct`` block per layer.
        prior_precision: Scalar prior precision / damping ``τ`` added to each
            factor before the Kronecker product.
    """

    mean: tuple[jax.Array, ...]
    precision: BlockDiagonal
    prior_precision: jax.Array

    def predictive_variance(
        self,
        model: PlainModel,
        parameters: tuple[jax.Array, ...],
        inputs: jax.Array,
    ) -> jax.Array:
        r"""Linearised predictive variance ``diag(J Σ J^\top)`` at ``inputs``.

        The network is locally linearised at the MAP point; the per-output
        marginal variance is the quadratic form of the model Jacobian against
        the Kronecker-factored posterior covariance ``Σ = precision^{-1}``,
        evaluated block-by-block.

        Args:
            model: Plain forward ``(parameters, inputs) -> outputs``.
            parameters: MAP parameters (same structure as :attr:`mean`).
            inputs: Batched evaluation inputs.

        Returns:
            Per-input, per-output predictive variance of shape
            ``(batch, output_dim)``.
        """

        def single_input_variance(single_input: jax.Array) -> jax.Array:
            """Predictive variance for one input row."""

            def model_at(flat_parameters: tuple[jax.Array, ...]) -> jax.Array:
                return model(flat_parameters, single_input[None, :])[0]

            jacobian = jax.jacrev(model_at)(parameters)
            output_dim = model_at(parameters).shape[0]

            variance = jnp.zeros((output_dim,))
            for block, layer_jacobian in zip(self.precision.blocks, jacobian, strict=True):
                flat_jacobian = layer_jacobian.reshape(output_dim, -1)
                solved = jax.vmap(block.solve)(flat_jacobian)
                variance = variance + jnp.sum(flat_jacobian * solved, axis=1)
            return variance

        return jax.vmap(single_input_variance)(inputs)


def kfac_laplace_posterior(
    tapped_model: TappedModel,
    loss: LossFunction,
    parameters: tuple[jax.Array, ...],
    inputs: jax.Array,
    targets: jax.Array,
    *,
    prior_precision: float | jax.Array,
) -> KroneckerLaplacePosterior:
    r"""Assemble a damped KFAC-Laplace posterior at a MAP point.

    Each layer's posterior precision block is the damped Kronecker product

    .. math::

        (A_l + \sqrt{\tau}\,I) \otimes (G_l + \sqrt{\tau}\,I),

    so that the dense block equals ``A_l ⊗ G_l`` plus a positive-definite
    correction (Ritter, Botev & Barber 2018). The damping ``τ`` doubles as
    the isotropic Gaussian prior precision and guarantees a positive-definite,
    invertible posterior.

    Args:
        tapped_model: Forward pass exposing per-layer taps.
        loss: Maps ``(outputs, targets) -> scalar``.
        parameters: MAP parameters (tuple of per-layer kernels).
        inputs: Batched inputs.
        targets: Batched targets.
        prior_precision: Scalar prior precision / damping ``τ`` (``> 0``).

    Returns:
        A :class:`KroneckerLaplacePosterior`.

    Raises:
        ValueError: If ``prior_precision`` is a Python ``int``/``float`` that
            is not strictly positive. The check is skipped when
            ``prior_precision`` is a JAX array/tracer (e.g., under
            :func:`jax.grad`), so the function stays transform-compatible.
    """
    if isinstance(prior_precision, (int, float)) and prior_precision <= 0.0:
        raise ValueError(f"prior_precision must be positive; got {prior_precision!r}")
    prior_precision_array = jnp.asarray(prior_precision)
    factors = kfac_factors(tapped_model, loss, parameters, inputs, targets)
    damping_root = jnp.sqrt(prior_precision_array)

    blocks: list[KroneckerProduct] = []
    for input_covariance, gradient_covariance in factors:
        damped_input = input_covariance + damping_root * jnp.eye(
            input_covariance.shape[0], dtype=input_covariance.dtype
        )
        damped_gradient = gradient_covariance + damping_root * jnp.eye(
            gradient_covariance.shape[0], dtype=gradient_covariance.dtype
        )
        blocks.append(KroneckerProduct(damped_input, damped_gradient))

    return KroneckerLaplacePosterior(
        mean=parameters,
        precision=BlockDiagonal(tuple(blocks)),
        prior_precision=prior_precision_array,
    )


__all__ = [
    "KroneckerLaplacePosterior",
    "TappedModel",
    "kfac_factors",
    "kfac_laplace_posterior",
]
