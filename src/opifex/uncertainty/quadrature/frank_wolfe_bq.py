r"""Frank-Wolfe Bayesian Quadrature (Briol et al, NeurIPS 2015).

A JAX-native port of Algorithm 1 (FW-Vanilla) from Briol et al,
*Frank-Wolfe Bayesian Quadrature: Probabilistic Integration with
Theoretical Guarantees* (arXiv:1506.02681). The algorithm constructs
a sparse empirical measure ``π_n = Σ_i w_i δ_{x_i}`` that minimises
the maximum mean discrepancy from a target measure ``π`` in the
kernel RKHS, with the worst-case rate ``MMD(π_n, π) = O(1/n)``
(Briol+ 2015 Theorem 1).

This implementation operates over a finite candidate set: the linear
minimisation oracle at each iteration ``x_{n+1} = argmax_x g_n(x)``
becomes ``argmax`` over the candidate index set. The FW gradient is
``g_n(x) = μ_π(x) - Σ_i w_i k(x, x_i)``, and the iterate update is
the standard FW-Vanilla mass-redistribution
``π_{n+1} = (1 - α_n) π_n + α_n δ_{x_{n+1}}`` with ``α_n = 1/(n + 1)``.

The historical opifex spec ``FFBQAdapterSpec`` (filed as "FFBQ" in
the catalogue) refers to this Frank-Wolfe BQ — the design notes pin
the algorithm to Briol+ 2015 with a file at
``quadrature/frank_wolfe_bq.py`` (design fix #190).

References
----------
* Briol, F.-X. et al. 2015 — *Frank-Wolfe Bayesian Quadrature:
  Probabilistic Integration with Theoretical Guarantees*, NeurIPS.
  arXiv:1506.02681.
* Briol, F.-X. et al. 2019 — *Probabilistic Integration*, Statistical
  Science 34(1). (Survey including FW-BQ analysis.)
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager per opifex convention

import jax
import jax.numpy as jnp


def frank_wolfe_bq(
    *,
    candidate_points: jax.Array,
    kernel_mean_at_candidates: jax.Array,
    kernel_fn: Callable[[jax.Array, jax.Array], jax.Array],
    num_iterations: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Run FW-Vanilla BQ (Briol+ 2015 Algorithm 1) over a candidate set.

    At iteration ``n`` the FW gradient is

    .. math::

        g_n(x) = \mu_\pi(x) - \sum_i w_i^{(n)} k(x, x_i),

    and the linear-minimisation oracle picks the candidate index with
    the maximum gradient. The iterate update is

    .. math::

        w^{(n+1)} = (1 - \alpha_n) w^{(n)} + \alpha_n e_{x_{n+1}},
        \qquad \alpha_n = \frac{1}{n + 1}.

    Initialised with ``x_1 = \mathrm{argmax}\, \mu_\pi`` (the FW
    gradient at ``n = 0`` is just the kernel mean).

    Sibling reference: Briol+ 2015 arXiv:1506.02681 Algorithm 1
    (FW-Vanilla). The original paper has no open-source reference
    implementation; the iteration here follows the published formulae
    verbatim.

    Args:
        candidate_points: ``(N_candidates, d)`` finite candidate set
            from which the FW iterate selects new design points.
        kernel_mean_at_candidates: ``(N_candidates,)`` kernel mean
            embedding ``μ_π(x_i) = ∫ k(x, x_i) p(x) dx`` evaluated at
            each candidate. The caller is responsible for providing
            the closed-form (e.g. via
            :mod:`opifex.uncertainty.quadrature.bayesian_quadrature`'s
            RBF + Gaussian kernel mean).
        kernel_fn: ``(X, Y) -> K`` kernel matrix function. Treated as
            a static argument under :func:`jax.jit`.
        num_iterations: FW iteration budget ``N``. Static under
            :func:`jax.jit`.

    Returns:
        ``(visited_indices, final_weights)``:

        * ``visited_indices`` shape ``(num_iterations,)`` — the order
          in which candidate indices were selected (repeats allowed).
        * ``final_weights`` shape ``(N_candidates,)`` — accumulated
          FW weights, non-negative, summing to one. Only the indices
          ever visited carry non-zero weight.
    """
    num_candidates = candidate_points.shape[0]
    candidate_self_kernel = kernel_fn(candidate_points, candidate_points)

    initial_index = jnp.argmax(kernel_mean_at_candidates)
    initial_weights = jnp.zeros(num_candidates).at[initial_index].set(1.0)

    def step(
        carry: tuple[jax.Array, jax.Array],
        iteration: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        """Perform one Frank-Wolfe iteration, selecting and reweighting a candidate point."""
        weights, _last_index = carry
        gradient = kernel_mean_at_candidates - candidate_self_kernel @ weights
        new_index = jnp.argmax(gradient)
        alpha = 1.0 / (iteration + 2.0)
        new_weights = (1.0 - alpha) * weights
        new_weights = new_weights.at[new_index].add(alpha)
        return (new_weights, new_index), new_index

    (final_weights, _), trailing_indices = jax.lax.scan(
        step,
        (initial_weights, initial_index),
        jnp.arange(num_iterations - 1),
    )
    visited_indices = jnp.concatenate([initial_index[None], trailing_indices])
    return visited_indices, final_weights
