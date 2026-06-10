r"""Generalized Gauss-Newton (GGN) matrix-vector products.

The GGN matrix at parameters ``θ`` for a model ``y = f(θ, x)`` and a
convex-in-output loss ``L(y, t)`` is

.. math::

    G(θ) = J(θ)^\\top H_y L(f(θ, x), t) J(θ),

where ``J = ∂f/∂θ`` is the model Jacobian and ``H_y L = ∂²L/∂y²`` is the
output-space Hessian of the loss. The GGN is positive-semidefinite by
construction and forms the canonical preconditioner for natural-gradient
descent and the linearised Laplace posterior.

Canonical reference:
* ``../kfac-jax/kfac_jax/_src/loss_functions.py`` ``multiply_ggn``
  (line 206) for the GGN-vp signature; opifex uses the functional
  forward-over-reverse recipe ``vjp ∘ (H_y L) ∘ jvp`` from
  Martens 2014.

References
----------
* Schraudolph, N. N. 2002 — *Fast Curvature Matrix-Vector Products for
  Second-Order Gradient Descent*, Neural Computation 14(7).
* Martens, J. 2014 — *New Insights and Perspectives on the Natural
  Gradient Method*, arXiv:1412.1193.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax


def ggn_vector_product(
    model: Callable[[jax.Array, jax.Array], jax.Array],
    loss: Callable[[jax.Array, jax.Array], jax.Array],
    parameters: jax.Array,
    inputs: jax.Array,
    targets: jax.Array,
    vector: jax.Array,
) -> jax.Array:
    """Compute the GGN-vp ``J^T H_y L J v`` via forward-over-reverse mode.

    Args:
        model: Maps ``(parameters, inputs) -> outputs``.
        loss: Maps ``(outputs, targets) -> scalar``. Should be convex in
            ``outputs`` (e.g., squared loss, cross-entropy with softmax).
        parameters: Model parameters ``θ``.
        inputs: Input batch passed as the second argument of ``model``.
        targets: Targets passed as the second argument of ``loss``.
        vector: Parameter-shaped direction ``v``.

    Returns:
        ``G v`` — same shape as ``parameters``.
    """

    def model_of_parameters(theta: jax.Array) -> jax.Array:
        """Evaluate the model on the fixed input batch as a function of ``theta``."""
        return model(theta, inputs)

    outputs, jacobian_v = jax.jvp(model_of_parameters, (parameters,), (vector,))

    def loss_of_outputs(output_values: jax.Array) -> jax.Array:
        """Evaluate the loss on the fixed targets as a function of model outputs."""
        return loss(output_values, targets)

    loss_gradient = jax.grad(loss_of_outputs)
    _, output_hessian_jv = jax.jvp(loss_gradient, (outputs,), (jacobian_v,))

    _, vjp_function = jax.vjp(model_of_parameters, parameters)
    (ggn_v,) = vjp_function(output_hessian_jv)
    return ggn_v
