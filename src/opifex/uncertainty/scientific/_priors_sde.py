r"""SDE matrix builders for probabilistic-ODE state-space priors.

JAX-native ports (NOT runtime imports) of the IWP / IOUP / Matern prior
state-space matrices from probnum and ProbNumDiffEq.jl references. Each
function returns the continuous-time linear SDE pair ``(drift,
dispersion)`` for use with
:func:`opifex.uncertainty.statespace.discretize_lti_sde`.

Canonical references (PORTED, never imported):
* IWP — ``../probnum/src/probnum/randprocs/markov/integrator/_iwp.py``
  ``_drift_matrix_iwp`` (line 142) and ``_dispersion_matrix_iwp`` (line 154).
* IOUP scalar driftspeed — ``../probnum/src/probnum/randprocs/markov/
  integrator/_ioup.py`` ``_drift_matrix_ioup`` (line 146).
* IOUP vector / matrix rate — ``../ProbNumDiffEq.jl/src/priors/ioup.jl``
  ``update_sde_drift!`` (line 103-117).
* Matérn — ``../probnum/src/probnum/randprocs/markov/integrator/
  _matern.py`` ``_drift_matrix_matern`` (line 145).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from scipy.special import binom


def _binomial_coefficients(num_derivatives: int) -> jax.Array:
    """``binom(q+1, 0), binom(q+1, 1), ..., binom(q+1, q+1)`` as a jax array.

    ``scipy.special.binom`` is used at construction time (Python int input,
    deterministic output); the result is materialised as a static jax array.
    """
    dimension = num_derivatives + 1
    return jnp.asarray([binom(dimension, i) for i in range(dimension)])


def iwp_sde(*, num_derivatives: int, wiener_process_dimension: int) -> tuple[jax.Array, jax.Array]:
    r"""Integrated Wiener Process prior SDE.

    For ``q = num_derivatives`` and ``d = wiener_process_dimension``,
    the per-dimension drift is the shift matrix
    ``[[0, 1, 0, ..., 0], [0, 0, 1, ..., 0], ..., [0, 0, ..., 1], [0, 0, ..., 0]]``
    of size ``(q+1) x (q+1)``. Dispersion picks the last derivative.
    The total state has size ``d * (q + 1)`` via Kronecker structure.
    """
    state_per_dim = num_derivatives + 1
    drift_per_dim = jnp.diag(jnp.ones(num_derivatives), k=1)
    dispersion_per_dim = jnp.zeros(state_per_dim).at[-1].set(1.0)[:, None]
    identity = jnp.eye(wiener_process_dimension)
    drift = jnp.kron(identity, drift_per_dim)
    dispersion = jnp.kron(identity, dispersion_per_dim)
    return drift, dispersion


def ioup_sde(
    *,
    num_derivatives: int,
    wiener_process_dimension: int,
    rate_parameter: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Integrated Ornstein-Uhlenbeck process prior SDE.

    Three rate-parameter modes (Julia ``priors/ioup.jl:103-117``):

    * ``scalar``: ``rate_parameter`` is a Python float; the bottom-right
      ``d x d`` block of the drift becomes ``rate_parameter * I_d``.
    * ``vector``: shape ``(d,)``; bottom-right block becomes
      ``diag(rate_parameter)``.
    * ``matrix``: shape ``(d, d)``; bottom-right block becomes
      ``rate_parameter`` verbatim.

    Dispersion is the same as IWP (noise enters the highest derivative).
    """
    state_per_dim = num_derivatives + 1
    state_dim = wiener_process_dimension * state_per_dim
    rate_array = jnp.asarray(rate_parameter)
    if rate_array.ndim == 0:
        rate_block = rate_array * jnp.eye(wiener_process_dimension)
    elif rate_array.ndim == 1:
        rate_block = jnp.diag(rate_array)
    else:
        rate_block = rate_array

    # Start with the IWP drift (shift matrix per dim) and replace the
    # bottom-right d x d block.
    drift, dispersion = iwp_sde(
        num_derivatives=num_derivatives,
        wiener_process_dimension=wiener_process_dimension,
    )
    # The bottom-right d x d block sits at rows / columns
    # [state_per_dim - 1, 2*state_per_dim - 1, ..., d*state_per_dim - 1].
    bottom_indices = jnp.arange(state_per_dim - 1, state_dim, state_per_dim)
    row_grid, col_grid = jnp.meshgrid(bottom_indices, bottom_indices, indexing="ij")
    drift = drift.at[row_grid, col_grid].set(rate_block)
    return drift, dispersion


def matern_sde(
    *,
    num_derivatives: int,
    wiener_process_dimension: int,
    lengthscale: float,
) -> tuple[jax.Array, jax.Array]:
    r"""Matérn-(q+1/2) prior SDE.

    Bottom row of the per-dim drift uses binomial coefficients with
    ``lambda = sqrt(2 * (q + 0.5)) / lengthscale``:

    .. math::

        \mathrm{row} = \bigl[ -\binom{q+1}{i} \lambda^{q+1-i} \bigr]_{i=0}^{q+1-1}.

    Cite ``../probnum/src/probnum/randprocs/markov/integrator/_matern.py:145``.
    """
    dimension = num_derivatives + 1
    nu = num_derivatives + 0.5
    lam = jnp.sqrt(2.0 * nu) / lengthscale
    drift_per_dim = jnp.diag(jnp.ones(num_derivatives), k=1)
    powers = lam ** jnp.arange(dimension, 0, -1)  # [lam^(q+1), lam^q, ..., lam^1]
    coefficients = _binomial_coefficients(num_derivatives)
    bottom_row = -coefficients * powers
    drift_per_dim = drift_per_dim.at[-1, :].set(bottom_row)
    dispersion_per_dim = jnp.zeros(dimension).at[-1].set(1.0)[:, None]
    identity = jnp.eye(wiener_process_dimension)
    drift = jnp.kron(identity, drift_per_dim)
    dispersion = jnp.kron(identity, dispersion_per_dim)
    return drift, dispersion


__all__ = ["ioup_sde", "iwp_sde", "matern_sde"]
