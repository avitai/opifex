r"""Matrix exponential and process-noise Gramian by Stillfjord & Tronarp doubling.

For a linear time-invariant SDE :math:`dx = F x\, dt + B\, dW`, with :math:`B B^\top = L Q_c
L^\top`, and a step :math:`\Delta t`, :func:`exponential_and_gramian` returns the transition
:math:`\exp(F \Delta t)` and the process noise

.. math::

    Q(\Delta t) = \int_0^{\Delta t} e^{F s} B B^\top e^{F^\top s}\, ds.

The step is scaled by :math:`2^{-k}` so that :math:`\|F \Delta t\|_1 2^{-k}` is below the bound
:math:`\eta` of the order-9 Pade approximant. There the exponential comes from that approximant and
a Cholesky factor of the Gramian from its Legendre expansion. The pair is then doubled back
:math:`k` times, squaring the exponential and updating the factor with a QR decomposition
(Stillfjord & Tronarp 2023, arXiv:2310.13462, sections 2 and 3.2, algorithms A.1 and A.2). The
factor never forms the growing :math:`e^{-F^\top \Delta t}` of Van Loan's block exponential, and
every component of :math:`Q` stays accurate at steps where :math:`P_\infty - A P_\infty A^\top`
cancels.

The initialisation, the doubling step, the order-9 coefficients, the bounds :math:`\eta` and the
upper-triangular QR with its custom JVP are ported from probdiffeq at commit
c2423e63e5e326080befd83542ccf25fde356b04: ``probdiffeq/util/gram_util.py``
(``_exp_gram_cholesky_init``, ``_exp_gram_cholesky_double``, ``pade_and_legendre_9``) and
``probdiffeq/backend/linalg.py`` (``qr_r``). probdiffeq is distributed under the MIT License::

    MIT License

    Copyright (c) 2022 Nicholas Krämer

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The doubling loop differs from probdiffeq's ``while_loop``, whose data-dependent length reverse-mode
differentiation cannot traverse. A sequence of steps runs 16 doublings, each applied only to the
steps that still need one, and 16 more under a single ``lax.cond`` when any step needs them. Steps
needing more than 32 doublings return NaN. Under an outer ``vmap`` the ``cond`` becomes a
``select`` and both blocks run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# probdiffeq ``pade_and_legendre_9``: Pade coefficients b_0..b_9, Legendre expansion coefficients of
# the Gramian factor, the squared Legendre norms, and the largest scaled 1-norm per precision.
_ORDER = 9
_PADE_COEFFICIENTS = (
    17643225600.0,
    8821612800.0,
    2075673600.0,
    302702400.0,
    30270240.0,
    2162160.0,
    110880.0,
    3960.0,
    90.0,
    1.0,
)
_LEGENDRE_COEFFICIENTS = (
    (17643225600, 0, 605404800, 0, 4324320, 0, 7920, 0, 2, 0),
    (0, 8821612800, 0, 155675520, 0, 617760, 0, 528, 0, 0),
    (0, 0, 1470268800, 0, 15444000, 0, 34320, 0, 10, 0),
    (0, 0, 0, 147026880, 0, 960960, 0, 1092, 0, 0),
    (0, 0, 0, 0, 10501920, 0, 42120, 0, 18, 0),
    (0, 0, 0, 0, 0, 583440, 0, 1320, 0, 0),
    (0, 0, 0, 0, 0, 0, 26520, 0, 26, 0),
    (0, 0, 0, 0, 0, 0, 0, 1020, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 34, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
)
_LEGENDRE_NORMS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
_ETA_FLOAT64 = 0.41098379928173995
_ETA_FLOAT32 = 2.801222
# Doublings applied to every step, and the total available when some step needs more.
_BASE_DOUBLINGS = 16
_MAX_DOUBLINGS = 32


@jax.custom_jvp
def _upper_triangular_factor(matrix: jax.Array) -> jax.Array:
    """Return ``R`` of the reduced QR decomposition ``matrix = Q R``."""
    return jnp.linalg.qr(matrix, mode="r")


@_upper_triangular_factor.defjvp
def _upper_triangular_factor_jvp(  # pyright: ignore[reportUnusedFunction]  # registered as the JVP rule
    primals: tuple[jax.Array], tangents: tuple[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """Differentiate ``R`` with ``Q`` held constant, so ``R_dot = Q^T matrix_dot``.

    This keeps ``R^T R = matrix^T matrix`` differentiable at the zero matrix, where the JVP of the
    full QR decomposition solves with a singular ``R`` (probdiffeq ``qr_r``, issue 668).
    """
    (matrix,) = primals
    (matrix_dot,) = tangents
    orthogonal, upper = jnp.linalg.qr(matrix, mode="reduced")
    return upper, orthogonal.T @ matrix_dot


def _doubling_count(scaled_drift: jax.Array) -> jax.Array:
    """Return the smallest ``k >= 0`` for which ``||F dt||_1 / 2^k`` is below eta."""
    size = scaled_drift.shape[-1]
    eta = _ETA_FLOAT64 if scaled_drift.dtype == jnp.float64 else _ETA_FLOAT32
    norm_count = jnp.log2(jnp.linalg.norm(scaled_drift, ord=1) / eta)
    order_count = jnp.log2(jnp.asarray((size - 1) / _ORDER, dtype=scaled_drift.dtype))
    return jnp.maximum(0.0, jnp.ceil(jnp.maximum(norm_count, order_count)))


def _pade_legendre_initialisation(
    drift: jax.Array, dispersion_factor: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return ``exp(A)`` and a Cholesky factor of ``G(A, B)`` for a scaled pair ``(A, B)``."""
    dtype = drift.dtype
    size = drift.shape[-1]
    pade = jnp.asarray(_PADE_COEFFICIENTS, dtype=dtype)
    identity = jnp.eye(size, dtype=dtype)
    drift_2 = drift @ drift
    drift_4 = drift_2 @ drift_2
    drift_6 = drift_4 @ drift_2
    drift_8 = drift_6 @ drift_2
    odd = drift @ (
        pade[9] * drift_8
        + pade[7] * drift_6
        + pade[5] * drift_4
        + pade[3] * drift_2
        + pade[1] * identity
    )
    even = (
        pade[8] * drift_8
        + pade[6] * drift_6
        + pade[4] * drift_4
        + pade[2] * drift_2
        + pade[0] * identity
    )
    exponential = jnp.linalg.solve(even - odd, even + odd)
    if dispersion_factor.shape[-1] == 0:
        return exponential, jnp.zeros_like(drift)

    legendre = jnp.asarray(_LEGENDRE_COEFFICIENTS, dtype=dtype)
    power = drift_2
    zeros = jnp.zeros_like(dispersion_factor)
    even_terms = [
        power @ dispersion_factor * legendre[0, 2] + dispersion_factor * legendre[0, 0],
        power @ dispersion_factor * legendre[2, 2],
        zeros,
        zeros,
        zeros,
    ]
    odd_terms = [
        power @ dispersion_factor * legendre[1, 3] + dispersion_factor * legendre[1, 1],
        power @ dispersion_factor * legendre[3, 3],
        zeros,
        zeros,
        zeros,
    ]
    for degree in (2, 3, 4):
        power = drift_2 @ power
        even_terms = [
            term + power @ dispersion_factor * legendre[2 * index, 2 * degree]
            for index, term in enumerate(even_terms)
        ]
        odd_terms = [
            term + power @ dispersion_factor * legendre[2 * index + 1, 2 * degree + 1]
            for index, term in enumerate(odd_terms)
        ]
    odd_terms = [drift @ term for term in odd_terms]
    interleaved = [term for pair in zip(even_terms, odd_terms, strict=True) for term in pair]
    norms = jnp.sqrt(jnp.asarray(_LEGENDRE_NORMS, dtype=dtype))
    right_hand_side = jnp.concatenate(
        [term / norm for term, norm in zip(interleaved, norms, strict=True)], axis=-1
    )
    factor = _upper_triangular_factor(jnp.linalg.solve(even - odd, right_hand_side).T).T
    return exponential, jnp.zeros_like(drift).at[: factor.shape[0], : factor.shape[1]].set(factor)


def _masked_doublings(
    exponential: jax.Array, factor: jax.Array, count: jax.Array, start: int, length: int
) -> tuple[jax.Array, jax.Array]:
    """Apply doublings ``start`` to ``start + length - 1`` of one step while below ``count``."""

    def double(
        carry: tuple[jax.Array, jax.Array], index: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Return ``(exp(2t), factor of Q(2t))`` from ``(exp(t), factor of Q(t))`` when active."""
        current_exponential, current_factor = carry
        is_active = index < count
        # An inactive doubling works on zeros, so a product that would overflow cannot put a NaN
        # into the gradient through the discarded branch (JAX FAQ, "gradients contain NaN where
        # using where").
        safe_exponential = jnp.where(is_active, current_exponential, 0.0)
        safe_factor = jnp.where(is_active, current_factor, 0.0)
        stacked = jnp.concatenate((safe_factor, safe_exponential @ safe_factor), axis=-1)
        doubled_factor = _upper_triangular_factor(stacked.T).T
        doubled_exponential = safe_exponential @ safe_exponential
        return (
            jnp.where(is_active, doubled_exponential, current_exponential),
            jnp.where(is_active, doubled_factor, current_factor),
        ), None

    indices = start + jnp.arange(length, dtype=count.dtype)
    (exponential, factor), _ = jax.lax.scan(double, (exponential, factor), indices)
    return exponential, factor


def diffusion_factor(dispersion_matrix: jax.Array, diffusion: jax.Array | None) -> jax.Array:
    """Return ``B = L S`` with ``S S^T = Q_c``, so that ``B B^T = L Q_c L^T``.

    ``jax.numpy.linalg.cholesky`` returns NaN when the decomposition fails (``lax.linalg.cholesky``
    docstring), which selects the symmetric square root for a singular ``Q_c``. Each branch reads a
    safe input when it is not taken, so neither puts a NaN into the gradient of the other.
    """
    if diffusion is None:
        return dispersion_matrix
    size = diffusion.shape[0]
    if size == 0:
        return dispersion_matrix
    dtype = diffusion.dtype
    is_definite = jnp.all(jnp.isfinite(jnp.linalg.cholesky(jax.lax.stop_gradient(diffusion))))
    cholesky = jnp.linalg.cholesky(jnp.where(is_definite, diffusion, jnp.eye(size, dtype=dtype)))
    # Distinct eigenvalues keep the eigendecomposition's derivative finite when it is not taken.
    spectral_input = jnp.where(
        is_definite, jnp.diag(jnp.arange(1, size + 1, dtype=dtype)), diffusion
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(spectral_input)
    is_positive = eigenvalues > 0.0
    roots = jnp.where(is_positive, jnp.sqrt(jnp.where(is_positive, eigenvalues, 1.0)), 0.0)
    square_root = eigenvectors * roots
    return dispersion_matrix @ jnp.where(is_definite, cholesky, square_root)


def exponential_and_gramian(
    drift: jax.Array, dispersion_factor: jax.Array, steps: jax.Array
) -> tuple[jax.Array, jax.Array]:
    r"""Return ``exp(F dt)`` and ``Q(dt)`` for every step of a sequence.

    Args:
        drift: Drift :math:`F` of shape ``(n, n)``.
        dispersion_factor: :math:`B` of shape ``(n, m)`` with :math:`B B^\top = L Q_c L^\top`.
        steps: Non-negative steps :math:`\Delta t_k` of shape ``(N,)``.

    Returns:
        ``(transitions, process_noises)``, each of shape ``(N, n, n)``. Steps needing more than
        32 doublings are NaN. At a zero step the process-noise part of the derivative with respect
        to that step is zero, as in GPJax's ``Matern12SDE.discretise``, because the Gramian factor
        scales with ``sqrt(dt)``.
    """
    dtype = jnp.result_type(drift, dispersion_factor, steps)
    drift = jnp.asarray(drift, dtype=dtype)
    dispersion_factor = jnp.asarray(dispersion_factor, dtype=dtype)
    steps = jnp.asarray(steps, dtype=dtype)

    def initialise(step: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Scale one step below eta; return its initial exponential, factor and count."""
        count = jax.lax.stop_gradient(_doubling_count(drift * step))
        scale = 2.0**count
        # The factor scales with sqrt(dt), whose derivative is infinite at a zero step; take the
        # root of a safe input there (JAX FAQ double-where), so the step gradient stays finite.
        has_length = step > 0.0
        root = jnp.where(has_length, jnp.sqrt(jnp.where(has_length, step / scale, 1.0)), 0.0)
        exponential, factor = _pade_legendre_initialisation(
            drift * (step / scale), dispersion_factor * root
        )
        return exponential, factor, count

    def doublings(
        exponentials: jax.Array, factors: jax.Array, counts: jax.Array, start: int, length: int
    ) -> tuple[jax.Array, jax.Array]:
        """Apply one block of masked doublings to every step."""
        return jax.vmap(
            lambda exponential, factor, count: _masked_doublings(
                exponential, factor, count, start, length
            )
        )(exponentials, factors, counts)

    exponentials, factors, counts = jax.vmap(initialise)(steps)
    exponentials, factors = doublings(exponentials, factors, counts, 0, _BASE_DOUBLINGS)
    exponentials, factors = jax.lax.cond(
        jnp.any(counts > _BASE_DOUBLINGS),
        lambda state: doublings(
            state[0], state[1], counts, _BASE_DOUBLINGS, _MAX_DOUBLINGS - _BASE_DOUBLINGS
        ),
        lambda state: state,
        (exponentials, factors),
    )
    process_noises = factors @ jnp.swapaxes(factors, -1, -2)
    process_noises = 0.5 * (process_noises + jnp.swapaxes(process_noises, -1, -2))
    is_beyond_limit = (counts > _MAX_DOUBLINGS)[:, None, None]
    return (
        jnp.where(is_beyond_limit, jnp.nan, exponentials),
        jnp.where(is_beyond_limit, jnp.nan, process_noises),
    )


__all__ = ["diffusion_factor", "exponential_and_gramian"]
