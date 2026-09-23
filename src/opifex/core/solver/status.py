"""The outcome of a solve, as a value that survives a transform.

A solver's outcome is ordinary data, not an exception: under ``vmap`` some elements of a
batch succeed while others do not, and a Python ``bool`` cannot represent that. It cannot
even be produced -- ``bool()`` on a traced comparison raises ``ConcretizationTypeError``,
so a result holding one cannot be built inside ``jit``. A Python scalar on a frozen
dataclass also rides in the pytree's static metadata, which is part of the jit cache key,
so each distinct outcome compiles its own program.

So the status that travels is an ``int32`` array. The names are host-side constants; only
the integer crosses a transform. This is what JAX does itself --
``jax.scipy.optimize.OptimizeResults`` declares ``status: int | Array`` -- and what LAPACK
does, where the ``info`` code is shaped like the batch dimensions.

**Success is several codes, not one.** A solve stopped early by request, or stalled at a
point that is genuinely a solution, has succeeded; a solve stalled short of one has not.
The same mechanism -- steps going to zero -- is a failure in the first case and a success
in the second, so the question is semantic and cannot be answered by a flag. Ask
``is_successful``; never ``status == SUCCESS``.

Codes are append-only. They are compared and persisted, so renumbering one silently
changes the meaning of stored results.

References:
    * Rackauckas et al., SciMLBase.jl ``ReturnCode`` -- twenty-two codes of which six are
      successes, reached through ``successful_retcode`` rather than an equality test.
      The ``Stalled`` and ``StalledSuccess`` pair is the distinction modelled here.
"""

from __future__ import annotations

import enum
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np


class Status(enum.IntEnum):
    """Why a solve stopped.

    Host-side constants. The value that crosses a transform is the plain integer, so a
    status may be compared, selected with ``jnp.where`` and batched like any array.
    """

    DEFAULT = 0
    """No outcome was reported. Not a success, so an unset result cannot pass for a good one."""

    SUCCESS = 1
    """Converged to the requested tolerance."""

    TERMINATED = 2
    """Stopped early because the caller asked it to. A success: the solve did as told."""

    STALLED_SUCCESS = 3
    """Steps went to zero at a point that is a solution, such as a valid local minimum."""

    MAX_ITERS = 4
    """The iteration budget ran out before the tolerance was met."""

    STALLED = 5
    """Steps went to zero short of a solution."""

    NONFINITE = 6
    """A NaN or an infinity appeared."""

    CONVERGENCE_FAILURE = 7
    """The iteration ran but did not approach a solution."""

    LINEAR_SOLVE_FAILED = 8
    """An inner linear solve did not succeed, so the outer step could not be taken."""

    STEP_TOO_SMALL = 9
    """The step size fell below what the scheme can represent."""


_SUCCESSFUL: Final[tuple[int, ...]] = (
    Status.SUCCESS,
    Status.TERMINATED,
    Status.STALLED_SUCCESS,
)

_MESSAGES: Final[dict[int, str]] = {
    Status.DEFAULT: "The solve reported no outcome.",
    Status.SUCCESS: "",
    Status.TERMINATED: "The solve stopped early at the caller's request.",
    Status.STALLED_SUCCESS: "Steps reached zero at a point that satisfies the problem.",
    Status.MAX_ITERS: "The iteration budget ran out before the tolerance was met.",
    Status.STALLED: "Steps reached zero before a solution was found.",
    Status.NONFINITE: "A non-finite value appeared during the solve.",
    Status.CONVERGENCE_FAILURE: "The iteration did not approach a solution.",
    Status.LINEAR_SOLVE_FAILED: "An inner linear solve failed, so the step could not be taken.",
    Status.STEP_TOO_SMALL: "The step size fell below what the scheme can represent.",
}


def is_successful(status: jax.Array) -> jax.Array:
    """Whether a status is one of the successful outcomes.

    Traced throughout: the result is a boolean **array**, so under ``vmap`` it carries one
    answer per batch element. It is therefore not usable in a Python ``if`` inside traced
    code, which is the point -- branching on an outcome is what forces a host sync.

    The successful codes are built inside the call rather than at module scope, so
    importing this module does not initialise a JAX backend.

    Args:
        status: An integer status, scalar or batched.

    Returns:
        A boolean array of the same shape as ``status``.
    """
    return jnp.isin(status, jnp.asarray(_SUCCESSFUL, jnp.int32))


def message(status: jax.Array) -> str | list[str]:
    """The human-readable reason for a status, on the host.

    Text is not part of the traced result: it cannot be produced inside a transform, and
    a result object carrying strings would not be a pytree of arrays. Call this at the
    boundary, once, on a concrete status.

    Args:
        status: A concrete integer status, scalar or batched.

    Returns:
        One message for a scalar status, or one per element for a batched one.
    """
    codes = np.asarray(status)
    if codes.ndim == 0:
        return _MESSAGES[int(codes)]
    return [_MESSAGES[int(code)] for code in codes.reshape(-1)]


def combine_statuses(first: jax.Array, second: jax.Array) -> jax.Array:
    """The outcome of two solves taken together: the first unsuccessful one.

    A combined solve succeeds only if both parts did, and when one fails its code is what
    a caller needs -- so a failing code is kept in preference to a successful one, and the
    earlier failure wins. Selection is by ``where`` rather than a branch, so the result
    stays traced and batches under ``vmap``, where a Python ``and`` over flags would sync
    the device and could not be built inside a transform at all.

    Args:
        first: Status of the solve whose failure takes precedence.
        second: Status of the other solve.

    Returns:
        The combined status.
    """
    return jnp.where(is_successful(first), second, first)


__all__ = ["Status", "combine_statuses", "is_successful", "message"]
