r"""Natural-gradient optimizers for variational Monte Carlo.

VMC energy minimisation is dramatically accelerated by the *natural* gradient
(stochastic reconfiguration / quantum natural gradient), which preconditions the
energy gradient by the inverse Fisher / quantum geometric tensor. For modern
deep ansaetze the parameter count far exceeds the sample count, so the Fisher is
inverted in the **sample space** instead -- the MinSR / SRT trick (Rende et al.,
2023; Chen & Heyl, *Nat. Phys.* 2024):

.. math::

    \delta\theta = O_L^\top (O_L O_L^\top + \lambda I)^{-1}\, dv,

where ``O_L`` is the centred, ``1/sqrt(N)``-scaled per-sample Jacobian of
``log|psi|`` and ``dv = 2(E_loc - <E_loc>)/sqrt(N)`` is the (centred) energy
gradient signal. The Gram matrix ``O_L O_L^\top`` is only ``N x N``.

:func:`spring_update` adds the SPRING momentum scheme (Goldshlager, Abrahamsen &
Lin, arXiv:2401.10190): a Nesterov-style accumulation of past updates plus a
projection-regulariser ``proj_reg / N`` on the Gram matrix. With zero momentum
and zero ``proj_reg`` it reduces exactly to :func:`minsr_update`.

The math is a pure-JAX port of NetKet ``_src/ngd/srt.py`` (``_compute_srt_update``
and ``_prepare_input`` in ``sr_srt_common.py``); these functions return the raw
parameter update vector, to be applied with an external learning rate (e.g. via
``optax``). Adam is used as the bootstrap optimizer directly through ``optax`` and
needs no wrapper here.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # noqa: TC002


def _prepare_inputs(
    jacobian: Float[Array, "n_samples n_params"],
    local_energies: Float[Array, " n_samples"],
) -> tuple[Array, Array]:
    r"""Centre and ``1/sqrt(N)``-scale the Jacobian and energy signal.

    Returns ``(O_L, dv)`` where ``O_L`` is the centred, scaled Jacobian and
    ``dv = 2 (E - <E>) / sqrt(N)`` (NetKet ``_prepare_input``).
    """
    n = jacobian.shape[0]
    scale = 1.0 / jnp.sqrt(n)
    o_l = (jacobian - jnp.mean(jacobian, axis=0, keepdims=True)) * scale
    dv = 2.0 * (local_energies - jnp.mean(local_energies)) * scale
    return o_l, dv


def _solve_sample_space(
    o_l: Float[Array, "n_samples n_params"],
    dv: Float[Array, " n_samples"],
    *,
    diag_shift: float,
    proj_reg: float = 0.0,
) -> Array:
    r"""Solve the regularised Gram system and lift back to parameter space.

    Computes ``O_L^T (O_L O_L^T + diag_shift I + proj_reg/N) ^{-1} dv``.
    """
    n = o_l.shape[0]
    gram = o_l @ o_l.T + diag_shift * jnp.eye(n, dtype=o_l.dtype)
    if proj_reg:
        gram = gram + proj_reg / n
    return o_l.T @ jnp.linalg.solve(gram, dv)


def minsr_update(
    jacobian: Float[Array, "n_samples n_params"],
    local_energies: Float[Array, " n_samples"],
    *,
    diag_shift: float = 1e-3,
) -> Array:
    r"""Compute the MinSR natural-gradient update via the sample-space Gram solve.

    Args:
        jacobian: Per-sample Jacobian of ``log|psi|`` w.r.t. parameters, shape
            ``(n_samples, n_params)``.
        local_energies: Local energy of each sample, shape ``(n_samples,)``.
        diag_shift: Tikhonov shift added to the Gram diagonal.

    Returns:
        The raw parameter update vector of shape ``(n_params,)``.
    """
    o_l, dv = _prepare_inputs(jacobian, local_energies)
    return _solve_sample_space(o_l, dv, diag_shift=diag_shift)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class SpringState:
    """Carry state for the SPRING optimizer (a JAX pytree).

    Registered as a pytree so it can flow through ``jit``/``scan`` as a traced
    carry rather than a static argument.

    Args:
        old_updates: The previous parameter update vector (the momentum buffer).
    """

    old_updates: Float[Array, " n_params"]


def spring_update(
    jacobian: Float[Array, "n_samples n_params"],
    local_energies: Float[Array, " n_samples"],
    state: SpringState,
    *,
    diag_shift: float = 1e-3,
    momentum: float = 0.99,
    proj_reg: float = 1e-3,
) -> tuple[Array, SpringState]:
    r"""SPRING natural-gradient update (MinSR + Nesterov momentum).

    Args:
        jacobian: Per-sample Jacobian of ``log|psi|`` w.r.t. parameters, shape
            ``(n_samples, n_params)``.
        local_energies: Local energy of each sample, shape ``(n_samples,)``.
        state: The :class:`SpringState` carrying the previous update.
        diag_shift: Tikhonov shift added to the Gram diagonal.
        momentum: Momentum coefficient ``mu`` (``0`` recovers MinSR).
        proj_reg: Weight of the ``1/N`` projection regulariser on the Gram
            matrix (SPRING).

    Returns:
        A ``(update, new_state)`` tuple.
    """
    o_l, dv = _prepare_inputs(jacobian, local_energies)
    # Subtract the momentum component already explained by the old update.
    dv = dv - momentum * (o_l @ state.old_updates)
    update = _solve_sample_space(o_l, dv, diag_shift=diag_shift, proj_reg=proj_reg)
    update = update + momentum * state.old_updates
    return update, SpringState(old_updates=update)


__all__ = ["SpringState", "minsr_update", "spring_update"]
