"""What a solve returns.

A ``Solution`` holds arrays and nothing else, so it can be built *inside* ``jit`` and
returned through it. Two kinds of field would prevent that and are deliberately absent.
Wall-clock time cannot be read from traced code: the clock is evaluated once at trace
time and baked in, so a reported duration would be the first compilation's for every
later call. A Python ``bool`` cannot be produced from a traced comparison at all, and on
a frozen dataclass it rides in the static metadata that forms the jit cache key, so each
outcome would compile its own program. Both belong to whoever ran the solve; see
:mod:`opifex.core.solver.report`.

Outcome is a traced status code rather than a flag; :mod:`opifex.core.solver.status` has
the reasoning.
"""

from typing import Any

import jax
from flax import struct

from opifex.core.solver.status import is_successful


class Solution(struct.PyTreeNode, kw_only=True):
    """What a solve produces: arrays, and a code saying how it went.

    Every field is pytree data, so a ``Solution`` survives ``jit``, ``vmap``, ``grad`` and
    ``scan``. Under ``vmap`` the status carries one outcome per batch element, which a
    single flag could not express.

    Keep ``stats`` to scalar reductions. Integer counters cost the backward pass nothing,
    since an integer output has no tangent, whereas a per-step history stacked over a scan
    is residual traffic proportional to the trajectory length.

    Attributes:
        fields: The solution fields, keyed by name.
        metrics: Scalar diagnostics, such as a final loss.
        status: Why the solve stopped; an ``int32`` array, batched under ``vmap``.
        stats: Counted work, such as the number of steps taken.
        auxiliary_data: Further payloads, such as uncertainty summaries. Arrays only: a
            Python object here becomes a pytree leaf and breaks the transform.
    """

    fields: dict[str, Any]
    metrics: dict[str, Any]
    status: jax.Array
    stats: dict[str, Any] = struct.field(default_factory=dict)
    auxiliary_data: dict[str, Any] = struct.field(default_factory=dict)

    @property
    def is_converged(self) -> jax.Array:
        """Whether the status is one of the successful outcomes.

        A boolean **array**, not a Python ``bool``: under ``vmap`` it carries one answer
        per element, and it cannot drive a Python ``if`` inside traced code.
        """
        return is_successful(self.status)


__all__ = ["Solution"]
