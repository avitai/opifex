"""Argument validation that survives JAX transformations.

A GP's scale parameters must be strictly positive, and a Python ``if value <= 0.0``
expresses that directly --- but only for a concrete value. Under :func:`jax.jit` or
:func:`jax.vmap` the comparison yields a tracer and ``bool()`` on it raises
:class:`jax.errors.TracerBoolConversionError`, so the guard forbids exactly the callers who
want to *optimise* the parameter it guards.

That matters here because this package ships
:func:`~opifex.uncertainty.gp.exact_gp_loocv_log_predictive`, the leave-one-out objective of
Rasmussen & Williams 2006 §5.4.2 --- a section of the *Model Selection and Adaptation of
Hyperparameters* chapter, whose purpose is to be differentiated with respect to the
hyperparameters. Differentiating it under ``jit`` requires tracing them.

**The check is kept, not dropped.** Where the value is concrete the error is raised exactly
as before, which is every eager call and every call under :func:`jax.grad` alone, since a
``grad`` tracer still carries a concrete value. Where the value is traced the comparison
cannot be decided at trace time by anyone, so it is left to the caller rather than made into
a refusal to compile.

Alternatives were considered and each costs more than it returns:
:mod:`jax.experimental.checkify` works but changes every return to ``(err, value)`` and would
propagate through the public API; :func:`jax.debug.print` is a host callback that cannot stop
the computation; and a runtime assert inside the traced graph turns a caller error into a
device-side failure with no Python traceback.
"""

from __future__ import annotations

import jax


#: Scale parameters accept a traced scalar as well as a Python float, because a caller
#: optimising them passes whatever the optimiser holds.
PositiveScalar = float | jax.Array


def nonpositive_message(name: str, value: PositiveScalar) -> str | None:
    """The error message for a non-positive scale, where that can be decided.

    Returns the message rather than raising so the ``raise`` stays at the call site. That
    keeps each public function's documented ``Raises:`` section truthful about its own body,
    which a helper that raised on its behalf would not --- and it keeps the trace-aware part
    in one place.

    Args:
        name: Parameter name, used in the message.
        value: The value to check. A traced value is left to the caller.

    Returns:
        The message when ``value`` is concrete and not strictly positive, otherwise
        ``None``. A traced value always gives ``None``: the comparison has no answer at
        trace time, and refusing to compile would make the parameter impossible to optimise
        under ``jit`` --- which is what the LOOCV objective exists for.
    """
    try:
        nonpositive = bool(value <= 0.0)
    except jax.errors.TracerBoolConversionError:
        return None
    return f"{name} must be strictly positive; got {value!r}." if nonpositive else None
