r"""Anderson-accelerated fixed-point solver for robust, differentiable SCF.

Plain Roothaan iteration :math:`D \to \mathrm{SCF\_step}(D)` charge-sloshes and
fails to converge for many molecules -- water/LDA, for instance, stalls at a
density residual of :math:`\sim 0.9`. **Anderson acceleration** [Anderson1965]_
[WalkerNi2011]_ -- equivalent to Pulay's DIIS [Pulay1980]_ applied to the density
residual -- mixes a short history of iterates to damp the oscillation and
converge in a handful of steps.

This is a *forward-solver* change only. The solver is handed to
:func:`optimistix.fixed_point` while the differentiated function stays the bare
SCF step :math:`f`, so :class:`optimistix.ImplicitAdjoint` differentiates the
exact fixed-point condition :math:`f(D) - D = 0`. The implicit-function-theorem
gradient :math:`(I - \partial f/\partial D)^{-1}\,\partial f/\partial\theta`
therefore depends only on the converged density and ``f`` -- never on *how* the
forward solve reached it. Accelerating convergence leaves the gradient exact and
unchanged.

The Anderson mixing coefficients solve the constrained least-squares problem

.. math::

    \min_{c}\ \Big\| \sum_i c_i\, r_i \Big\|^2 \quad\text{s.t.}\quad \sum_i c_i = 1,

over the stored residuals :math:`r_i = f(y_i) - y_i`, whose Lagrange solution is
:math:`c \propto M^{-1}\mathbf{1}` with the Gram matrix :math:`M_{ij}=r_i^\top r_j`
-- exactly Pulay's DIIS system. The next iterate extrapolates the stored
:math:`f`-values, optionally relaxed by a mixing factor ``beta``.

References
----------
.. [Anderson1965] D. G. Anderson, "Iterative procedures for nonlinear integral
   equations," *J. ACM* **12**, 547 (1965).
.. [WalkerNi2011] H. F. Walker and P. Ni, "Anderson acceleration for fixed-point
   iterations," *SIAM J. Numer. Anal.* **49**, 1715 (2011).
.. [Pulay1980] P. Pulay, "Convergence acceleration of iterative sequences. The
   case of SCF iteration," *Chem. Phys. Lett.* **73**, 393 (1980).
"""

from __future__ import annotations

import math
from collections.abc import (
    Callable,  # noqa: TC003  (equinox resolves the field annotation at runtime)
)
from typing import Any, cast, NamedTuple

import jax
import jax.numpy as jnp
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Bool, PyTree, Scalar
from optimistix import max_norm, RESULTS


class _AndersonState(NamedTuple):
    """Mutable solver state threaded through :func:`optimistix.fixed_point`.

    Attributes:
        function_history: Stored :math:`f(y_i)` iterates, flattened [Shape: (m, n)].
        residual_history: Stored residuals :math:`f(y_i) - y_i` [Shape: (m, n)].
        slot: Next ring-buffer slot to overwrite (oldest entry).
        filled: Number of valid history slots, capped at the history size.
        relative_error: Max-norm of the scaled residual; ``< 1`` means converged.
    """

    function_history: Array
    residual_history: Array
    slot: Array
    filled: Array
    relative_error: Scalar


class AndersonAcceleration(optx.AbstractFixedPointSolver[Array, Any, _AndersonState]):
    r"""Anderson-accelerated fixed-point solver (Pulay-DIIS on the residual).

    A drop-in replacement for :class:`optimistix.FixedPointIteration` that
    converges the Kohn-Sham density where plain Roothaan iteration oscillates.
    Suitable for :func:`optimistix.fixed_point`; the default
    :class:`~optimistix.ImplicitAdjoint` then yields the exact implicit-function
    -theorem gradient through the converged fixed point.

    Args:
        rtol: Relative tolerance on the residual.
        atol: Absolute tolerance on the residual.
        history_size: Number of past iterates retained for the mixing (the
            Anderson/DIIS depth ``m``).
        mixing: Relaxation factor ``beta`` in ``y_i + beta * r_i``; ``1.0`` is
            pure Anderson extrapolation, smaller values add damping.
        regularization: Tikhonov factor (scaled by the Gram-matrix magnitude)
            stabilising the least-squares solve against linearly dependent
            histories.
        norm: Norm used for the convergence test on the scaled residual.
    """

    rtol: float
    atol: float
    history_size: int = 6
    mixing: float = 1.0
    regularization: float = 1.0e-10
    norm: Callable[[PyTree], Scalar] = max_norm

    def init(
        self,
        fn: Any,
        y: Array,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _AndersonState:
        """Allocate empty history buffers sized from the iterate structure."""
        del fn, args, options, aux_struct, tags
        leaves = jax.tree_util.tree_leaves(f_struct)
        size = sum(math.prod(leaf.shape) for leaf in leaves)
        dtype = leaves[0].dtype if leaves else y.dtype
        zeros = jnp.zeros((self.history_size, size), dtype=dtype)
        return _AndersonState(
            function_history=zeros,
            residual_history=zeros,
            slot=jnp.array(0),
            filled=jnp.array(0),
            relative_error=jnp.array(jnp.inf, dtype=dtype),
        )

    def step(
        self,
        fn: Any,
        y: Array,
        args: PyTree,
        options: dict[str, Any],
        state: _AndersonState,
        tags: frozenset[object],
    ) -> tuple[Array, _AndersonState, Any]:
        """One Anderson update: evaluate ``f``, mix the history, extrapolate."""
        del options, tags
        function_value, aux = fn(y, args)
        y_flat, unravel = ravel_pytree(y)
        g_flat, _ = ravel_pytree(function_value)
        residual = g_flat - y_flat

        functions = state.function_history.at[state.slot].set(g_flat)
        residuals = state.residual_history.at[state.slot].set(residual)
        filled = jnp.minimum(state.filled + 1, self.history_size)

        coefficients = self._mixing_coefficients(residuals, filled)
        relaxed = functions - (1.0 - self.mixing) * residuals
        new_y_flat = coefficients @ relaxed

        scale = self.atol + self.rtol * jnp.abs(g_flat)
        relative_error = self.norm(residual / scale)
        new_state = _AndersonState(
            function_history=functions,
            residual_history=residuals,
            slot=(state.slot + 1) % self.history_size,
            filled=filled,
            relative_error=relative_error,
        )
        return unravel(new_y_flat), new_state, aux

    def _mixing_coefficients(self, residuals: Array, filled: Array) -> Array:
        r"""Pulay-DIIS coefficients :math:`c \propto M^{-1}\mathbf 1`, masked.

        Invalid (unfilled) history slots are masked out of the Gram matrix so
        they receive zero weight; a scaled Tikhonov term keeps the solve robust
        when stored residuals are nearly linearly dependent.
        """
        valid = jnp.arange(self.history_size) < filled
        gram = residuals @ residuals.T
        pair_mask = valid[:, None] & valid[None, :]
        identity = jnp.eye(self.history_size, dtype=gram.dtype)
        # Keep valid-block entries; force invalid rows/cols to the identity so
        # the linear solve stays non-singular and their coefficients vanish.
        gram = jnp.where(pair_mask, gram, identity)
        trace = jnp.trace(gram)
        denom = jnp.maximum(filled, 1).astype(gram.dtype)
        gram = gram + self.regularization * (trace / denom) * identity

        rhs = valid.astype(gram.dtype)
        weights = cast("Array", jnp.linalg.solve(gram, rhs))
        weights = jnp.where(valid, weights, 0.0)
        return weights / jnp.sum(weights)

    def terminate(
        self,
        fn: Any,
        y: Array,
        args: PyTree,
        options: dict[str, Any],
        state: _AndersonState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        """Converged once the scaled residual max-norm drops below one."""
        del fn, y, args, options, tags
        return state.relative_error < 1, RESULTS.successful

    def postprocess(
        self,
        fn: Any,
        y: Array,
        aux: Any,
        args: PyTree,
        options: dict[str, Any],
        state: _AndersonState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Array, Any, dict[str, Any]]:
        """Return the converged iterate unchanged (no post-processing)."""
        del fn, args, options, state, tags, result
        return y, aux, {}
