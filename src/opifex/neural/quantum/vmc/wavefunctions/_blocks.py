r"""Log-domain determinant building blocks for neural wavefunctions.

These are the numerically-stable primitives behind a generalized-Slater
neural-network ansatz, ported from DeepMind's FermiNet
(``ferminet/network_blocks.py`` -- Pfau et al., *Phys. Rev. Research* **2**,
033429 (2020)):

* :func:`slogdet` -- sign and log-magnitude of a (batched) determinant with a
  fast scalar path for ``1x1`` blocks;
* :func:`logdet_matmul` -- a weighted sum of determinants evaluated entirely in
  the log domain via the log-sum-exp trick, returning ``(sign, log|.|)`` so the
  wavefunction never overflows even when individual determinants are huge;
* :func:`construct_input_features` -- the raw electron-nucleus / electron-
  electron displacement and distance features, with the ``r_ee`` diagonal masked
  to keep its gradient well-defined.

The functions are pure JAX and therefore ``jit`` / ``grad`` / ``vmap`` clean.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
from jaxtyping import Array, Float  # noqa: TC002


def slogdet(x: Float[Array, "*batch n n"]) -> tuple[Array, Array]:
    """Return the sign and natural log-magnitude of ``det(x)``.

    A fast scalar path is used for ``1x1`` matrices, which are numerically
    sensitive under :func:`jax.numpy.linalg.slogdet` (FermiNet note).

    Args:
        x: A square matrix or batch of square matrices with shape
            ``(..., n, n)``.

    Returns:
        A ``(sign, logabsdet)`` tuple, each with shape ``(...)``.
    """
    if x.shape[-1] == 1:
        scalar = x[..., 0, 0]
        sign = jnp.sign(scalar)
        logdet = jnp.log(jnp.abs(scalar))
        return sign, logdet
    return jnp.linalg.slogdet(x)


def logdet_matmul(
    determinant_inputs: list[Float[Array, "ndet n n"]],
    weights: Float[Array, "ndet 1"] | None = None,
) -> tuple[Array, Array]:
    r"""Combine determinants with weights in the log domain (log-sum-exp).

    Evaluates :math:`\sum_i w_i \prod_c \det(X_{i,c})` -- a sum over
    determinants (index ``i``) of a product over spin channels (index ``c``) --
    returning the result as a ``(sign, log|.|)`` pair. The reduction uses the
    log-sum-exp trick so the magnitude never overflows.

    Args:
        determinant_inputs: One entry per spin channel. Each entry has shape
            ``(ndet, n, n)`` (a stack of square orbital matrices, one per
            determinant). Length 1 for a full (dense) determinant; length 2 for
            a spin-factorised block-diagonal determinant.
        weights: Optional per-determinant weight of shape ``(ndet, 1)``. If
            ``None`` a uniform unit weight is used.

    Returns:
        A ``(sign, log_magnitude)`` tuple of scalars such that
        ``sign * exp(log_magnitude)`` equals the weighted determinant sum.
    """
    # Product over spin channels of 1x1 determinants (kept out of the log domain
    # because 1x1 determinants are numerically sensitive and can hit zero).
    one_by_one = functools.reduce(
        lambda a, b: a * b,
        [x.reshape(-1) for x in determinant_inputs if x.shape[-1] == 1],
        1.0,
    )
    # Product over spin channels of the larger determinants, in the log domain.
    sign_in, logdet = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]),
        [slogdet(x) for x in determinant_inputs if x.shape[-1] > 1],
        (1.0, 0.0),
    )

    max_logdet = jnp.max(logdet)
    determinants = sign_in * one_by_one * jnp.exp(logdet - max_logdet)
    result = jnp.sum(determinants) if weights is None else jnp.matmul(determinants, weights)[0]

    sign_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + max_logdet
    return sign_out, log_out


def construct_input_features(
    positions: Float[Array, "nelectron ndim"],
    atoms: Float[Array, "natom ndim"],
) -> tuple[Array, Array, Array, Array]:
    """Build electron-nucleus and electron-electron displacement features.

    Args:
        positions: Electron coordinates of shape ``(nelectron, ndim)``.
        atoms: Nuclear coordinates of shape ``(natom, ndim)``.

    Returns:
        A ``(ae, ee, r_ae, r_ee)`` tuple where ``ae`` is the electron-nucleus
        displacement ``(nelectron, natom, ndim)``, ``ee`` the electron-electron
        displacement ``(nelectron, nelectron, ndim)``, ``r_ae`` the electron-
        nucleus distance ``(nelectron, natom, 1)`` and ``r_ee`` the electron-
        electron distance ``(nelectron, nelectron, 1)``. The ``r_ee`` diagonal
        is masked to zero so its gradient stays well-defined.
    """
    ae = positions[:, None, :] - atoms[None, :, :]
    ee = positions[None, :, :] - positions[:, None, :]

    r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
    n = positions.shape[0]
    # Add the identity inside the norm so the (undefined) gradient of ||0|| is
    # never taken, then mask the diagonal back to zero.
    r_ee = jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n))
    return ae, ee, r_ae, r_ee[..., None]


__all__ = ["construct_input_features", "logdet_matmul", "slogdet"]
