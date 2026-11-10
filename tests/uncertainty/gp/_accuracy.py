"""Derived tolerances for the GP accuracy tests.

Nothing here is calibrated from a measured disagreement. Two models are used, each taken
from an existing implementation or a textbook condition number, and each applied only where
its premise holds.

Reductions: NVIDIA torch-harmonics' budget
------------------------------------------
``compare_tensors`` in NVIDIA's torch-harmonics (Apache-2.0, PR #339, "Scale
parameter-gradient test tolerances with the reduction") floors the absolute tolerance at
``ulps * eps * max|reference|`` with ``rtol = 0`` and ``ulps = max(64, sqrt(N))``, where
``N`` is the number of terms in the reduction. Its stated reason is the one that matters
here: reassociating a sum --- which is all that separates a compiled evaluation from an eager
one --- produces *an absolute error set by the size of the summed terms, not by the entry it
lands on*. Their worked failure is a gradient entry of ~0.07 missing by 1.7e-3 in a tensor
reaching ~400, which is the same shape as a small component of ``alpha`` here.

That is why :func:`assert_reduction_agreement` takes a ``scale`` and scales the whole tensor
by one number rather than comparing entrywise relative errors. The form is also what devito
issue #3078 argues for after finding its own gradient tolerance was "tuned, not derived":
a norm-like scale, but kept per-entry so a single wrong boundary point still fails.

``sqrt(N)`` is the probabilistic rounding model of Higham & Mary (SIAM J. Sci. Comput. 41,
2019) rather than the worst-case ``N``; it is a modelling choice for tractability, not a
property of floating-point arithmetic, so it is used here only because the reference
implementation uses it. At this suite's sizes the floor of 64 is what binds in every case.

Where the reference implementation's proxy breaks
-------------------------------------------------
``max|reference|`` stands in for "the size of the summed terms" and is only equal to it when
the reduction does not cancel. ``predict_exact_gp`` returns ``k_diag - sum(v * v)``: at a
training point the GP nearly interpolates, so with ``output_scale=1`` and ``noise_std=0.1``
the result is about 7.7e-03 while both operands are about 1. Measured on CPU and CUDA in both
precisions, the compiled-versus-eager gap there is exactly 0.5 or 1.0 ULP **of ``k_diag``** ---
one rounding of the subtraction --- and the unmodified reference model is violated at 1.63 of
its own budget in two of those four runs. So the variance passes ``scale=output_scale**2``,
which is the operand magnitude the reference implementation's sentence actually names.

Kernels: the condition number of the exponential
------------------------------------------------
A Gram matrix is not a reduction --- for one-dimensional inputs each entry is its own short
expression --- and the ``max|reference|`` floor makes the model vacuous there: measured, it
allows every small entry an error twice the entry's own value. The relative condition number
of ``exp(-u)`` with respect to ``u`` is ``u``, so a relative perturbation of the exponent is
amplified by the exponent, and :func:`exponential_kernel_budget` weights the roundings on the
path to ``u`` by ``u`` and the rest not at all.

``u`` must be computed **from the inputs by the kernel's own formula**, never read back as
``-log(k / s^2)``: those differ by ``log(poly)`` for any kernel with a polynomial prefactor,
which is 11.18 against 7.19 for Matern-5/2 on this suite's problem. The convenient version was
measured at 1.228 of its own budget on CUDA float32 --- already violated --- while passing on
CPU and on RBF, which has no prefactor and so hides the mistake completely.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable

    import jax


#: Floor of torch-harmonics' ``param_grad_ulps``. At this suite's reduction lengths
#: (``n = 8``, ``n**2 = 64``) ``sqrt(N)`` never exceeds it, so this is the budget in force.
MINIMUM_REDUCTION_ULPS = 64.0

#: Perturbations for the Taylor remainder test, following pyadjoint's ``taylor_test``
#: (``pyadjoint/verification.py``): a geometric sequence, so successive residuals give a
#: convergence rate directly.
TAYLOR_PERTURBATIONS = tuple(0.01 / 2**index for index in range(4))

#: A correct gradient makes the Taylor remainder second order, so the rate is 2; a wrong one
#: leaves a first-order term and gives 1. **This is a rate, not a tolerance** --- it is scale
#: free, so there is nothing here to calibrate. The floor is pyadjoint's own criterion,
#: slackened only for the finite sample of four steps.
TAYLOR_RATE_FLOOR = 1.9

#: Below this a remainder is floating-point noise, and the ratio of two such residuals
#: measures nothing. pyadjoint discards them the same way.
TAYLOR_RESIDUAL_FLOOR = 1e-15


def machine_epsilon(dtype: np.dtype | type | str) -> float:
    """Return the unit roundoff of ``dtype``.

    Args:
        dtype: Anything ``numpy.finfo`` accepts.

    Returns:
        ``numpy.finfo(dtype).eps``.
    """
    return float(np.finfo(np.dtype(dtype)).eps)


def reduction_ulp_budget(terms: int) -> float:
    """ULPs allowed for reassociating a sum of ``terms`` terms.

    torch-harmonics' ``param_grad_ulps``: ``max(64, sqrt(N))``.

    Args:
        terms: Number of terms in the reduction.

    Returns:
        The ULP budget.
    """
    return max(MINIMUM_REDUCTION_ULPS, math.sqrt(terms))


def assert_reduction_agreement(
    produced: jax.Array | np.ndarray,
    reference: jax.Array | np.ndarray,
    *,
    terms: int,
    what: str,
    scale: float | None = None,
) -> None:
    """Assert two evaluations of one reduction agree, on torch-harmonics' budget.

    ``atol = max(64, sqrt(terms)) * eps * scale`` with ``rtol = 0``, where ``scale`` defaults
    to ``max|reference|``. Pass ``scale`` explicitly when the reduction cancels, so that the
    budget follows the operands rather than the much smaller result --- see the module
    docstring for the measurement that forces this.

    Args:
        produced: The value under test.
        reference: The value to compare against.
        terms: Number of terms in the reduction behind ``produced``.
        what: Name of the quantity, for the failure message.
        scale: Magnitude of the summed terms. Defaults to ``max|reference|``.

    Raises:
        AssertionError: If any entry exceeds the budget.
    """
    # The working precision is the inputs', read before widening them for the arithmetic:
    # taking it afterwards silently budgets every float32 comparison at float64 epsilon.
    working = np.result_type(np.asarray(reference).dtype, np.float32)
    got = np.asarray(produced, dtype=np.float64)
    want = np.asarray(reference, dtype=np.float64)
    magnitude = float(np.max(np.abs(want))) if scale is None else float(scale)
    budget = reduction_ulp_budget(terms) * machine_epsilon(working) * magnitude
    worst = float(np.max(np.abs(got - want)))
    assert worst <= budget, (
        f"{what}: worst gap {worst:.6e} exceeds {budget:.6e} "
        f"({reduction_ulp_budget(terms):.1f} ULP of {magnitude:.6e}), using "
        f"{worst / budget:.4f} of budget. The budget is torch-harmonics' published "
        "reduction tolerance -- raising it is not a local decision."
    )


def assert_reference_agreement(
    produced: jax.Array | np.ndarray,
    reference: jax.Array | np.ndarray,
    *,
    terms: int,
    scale: float,
    what: str,
) -> None:
    """Assert a working-precision value matches an exact reference, on the forward bound.

    **A different question from :func:`assert_reduction_agreement`, and so a different
    budget.** That one compares two evaluations of the same code, where the only difference
    is the order XLA chose, and takes its number from torch-harmonics' published
    *gradient* tolerance. This one compares against an independently computed answer, so the
    budget is the algorithm's own forward error: a sum of ``terms`` terms rounds at the scale
    of the terms, giving ``terms * eps * scale``. There is no 64-ULP floor, because nothing
    is being absorbed here --- the floor would make the test blind to errors larger than the
    computation's own precision.

    Measured on this suite's 8-point problem against a float64 NumPy reference, the float32
    predictive mean and variance each use about a quarter of this budget on both backends,
    and a 2e-6 relative perturbation of the variance's subtrahend exceeds it twofold.

    Args:
        produced: The working-precision value under test.
        reference: An independently computed reference, at higher precision.
        terms: Number of terms in the reduction behind ``produced``.
        scale: Magnitude of the summed terms.
        what: Name of the quantity, for the failure message.

    Raises:
        AssertionError: If any entry exceeds the budget.
    """
    working = np.result_type(np.asarray(produced).dtype, np.float32)
    got = np.asarray(produced, dtype=np.float64)
    want = np.asarray(reference, dtype=np.float64)
    budget = terms * machine_epsilon(working) * float(scale)
    worst = float(np.max(np.abs(got - want)))
    assert worst <= budget, (
        f"{what}: worst gap {worst:.6e} exceeds {budget:.6e} "
        f"({terms} ULP of {float(scale):.6e}), using {worst / budget:.4f} of budget. The "
        "budget is the forward error of a sum of that many terms -- raising it claims the "
        "algorithm is less accurate than it is."
    )


def exponential_kernel_budget(
    reference: jax.Array | np.ndarray,
    *,
    exp_argument: np.ndarray,
    amplified_roundings: int,
    flat_roundings: int,
) -> np.ndarray:
    """Per-entry absolute budget for a kernel of the form ``s^2 * p(u) * exp(-u)``.

    The relative condition number of ``exp(-u)`` with respect to ``u`` is ``u``, so roundings
    on the path *to* the exponent are weighted by the exponent and the rest are not. Both
    counts are read off the kernel's source; ``exp_argument`` must be computed from the inputs
    by the kernel's own formula and never from its output.

    Args:
        reference: The Gram matrix to compare against; sets each entry's scale.
        exp_argument: ``u`` per entry, from the inputs and the length-scale.
        amplified_roundings: Operations on the path to ``u``.
        flat_roundings: Operations after it, including the exponential itself.

    Returns:
        One absolute budget per entry of ``reference``.
    """
    working = np.result_type(np.asarray(reference).dtype, np.float32)
    values = np.asarray(reference, dtype=np.float64)
    weight = amplified_roundings * np.asarray(exp_argument, dtype=np.float64) + flat_roundings
    return weight * machine_epsilon(working) * np.abs(values)


def assert_within_budget(
    produced: jax.Array | np.ndarray,
    reference: jax.Array | np.ndarray,
    *,
    budget: np.ndarray | float,
    what: str,
) -> None:
    """Assert ``|produced - reference| <= budget`` entrywise, reporting the fraction used.

    Args:
        produced: The value under test.
        reference: The value to compare against.
        budget: Absolute budget, broadcastable to ``reference``.
        what: Name of the quantity, for the failure message.

    Raises:
        AssertionError: If any entry exceeds its budget.
    """
    gap = np.abs(np.asarray(produced, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    allowance = np.broadcast_to(np.asarray(budget, dtype=np.float64), gap.shape)
    safe = np.where(allowance > 0.0, allowance, np.finfo(np.float64).tiny)
    used = float(np.max(gap / safe))
    assert used <= 1.0, (
        f"{what}: worst entry used {used:.4f} of its derived budget "
        f"(max gap {float(np.max(gap)):.6e}). The budget is an operation count read off the "
        "source, weighted by the condition number of the exponential -- raising it asserts "
        "the kernel performs more roundings than it does."
    )


def taylor_remainder_rates(
    loss: Callable[[jax.Array], jax.Array],
    point: jax.Array,
    gradient: jax.Array,
    *,
    seed: int = 7,
) -> tuple[list[float], list[float]]:
    """Residuals and convergence rates of the first-order Taylor remainder.

    Follows pyadjoint's ``taylor_test`` (``pyadjoint/verification.py``): for a correct
    gradient ``|J(m + e h) - J(m) - e <grad J, h>|`` is second order in ``e``, so halving the
    step quarters the residual and the rate is 2. A wrong gradient leaves a first-order term
    and the rate falls to 1 --- a verdict that needs no tolerance.

    Args:
        loss: Scalar objective.
        point: Where to expand.
        gradient: The gradient under test, at ``point``.
        seed: Seed for the perturbation direction.

    Returns:
        The residuals, and the rates between successive residuals above the noise floor.
    """
    generator = np.random.default_rng(seed)
    anchor = np.asarray(point)
    direction = np.asarray(generator.normal(size=anchor.shape), dtype=anchor.dtype)
    directional = float(np.sum(np.asarray(gradient, dtype=np.float64) * direction))
    base = float(np.asarray(loss(point), dtype=np.float64))

    residuals: list[float] = []
    for step in TAYLOR_PERTURBATIONS:
        moved = jnp.asarray(anchor + np.asarray(step, dtype=anchor.dtype) * direction)
        shifted = float(np.asarray(loss(moved), dtype=np.float64))
        residuals.append(abs(shifted - base - step * directional))

    rates = [
        float(np.log(finer / coarser) / np.log(0.5))
        for coarser, finer in itertools.pairwise(residuals)
        if coarser > TAYLOR_RESIDUAL_FLOOR and finer > TAYLOR_RESIDUAL_FLOOR
    ]
    return residuals, rates
