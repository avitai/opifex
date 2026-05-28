"""Global-sensitivity utilities (Task 6.4).

Two complementary techniques for model-input sensitivity:

* :func:`sobol_indices` — variance-based first-order and total-order
  indices via the Saltelli (2002) pick-freeze scheme.
* :func:`morris_screening` — elementary-effects screening for cheap
  ranking of influential inputs (Morris 1991 / Campolongo+ 2007).

Both utilities are pure JAX and accept arbitrary scalar-valued
``model: Callable[[jax.Array], jax.Array]`` callables; neither uses
SALib (Task 6.4 forbids it as a hard dependency).
"""

from opifex.uncertainty.sensitivity.morris import morris_screening, MorrisResult
from opifex.uncertainty.sensitivity.sobol import sobol_indices, SobolResult


__all__ = [
    "MorrisResult",
    "SobolResult",
    "morris_screening",
    "sobol_indices",
]
